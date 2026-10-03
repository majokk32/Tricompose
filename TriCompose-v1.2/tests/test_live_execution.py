"""Complete fixed-path wiring using invented callbacks; NOT generation results."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import live_execution as execution
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.execution_ledger import restore_ledger
from tricompose_v12.live_workers import registry
from test_live_receipts import data,H
import tricompose_v11.report_contracts as reports


class LiveWiringTests(unittest.TestCase):
    def exercise(self, *, timeout=False, failure=False):
        anchor,image,labels,report,report_labels=data(False)
        policy=json.loads((ROOT/"configs/live_smoke_v1.json").read_text())
        rows=[]; images={}; outputs={}; calls=[]; events=[]; receipts=[]
        for i,(model,seed) in enumerate(policy["image_slots"]):
            inputs={"synthetic_ehr":{"path":"invented_ehr","sha256":H["ehr"]},
                "ehr_facts":{"path":"invented_facts","sha256":H["facts"]},
                "final_prompt":{"path":"invented_prompt","sha256":"b"*64,"clinical_intent_sha256":"c"*64}}
            req={"model_id":model,"seed":seed,"inputs":inputs}
            row={"request":req,"canonical_request_sha256":_digest(req)}; rows.append(row)
            candidate={**copy.deepcopy(image),"candidate_id":"fixture_"+model,"model_id":model,"seed":seed,
                "prompt_sha256":"b"*64,"clinical_intent_sha256":"c"*64,
                "input_request_sha256":row["canonical_request_sha256"],"adapter_added_prefix":False,"cost":{"model_calls":1}}
            candidate["artifact"]["path"]="invented_image_not_on_disk_"+str(i)
            images[i]=candidate
        workers={k:{**v,"status":"preflighted","asset_pins":{}} for k,v in registry().items()}
        workers["xrv"].update(checkpoint_sha256=H["xrv"],thresholds_sha256=H["thresholds"],thresholds_path="invented_thresholds",
            thresholds=labels["thresholds"],scorer_provenance={},calibration_status="invented_frozen_status")
        workers["chexbert"].update(checkpoint_sha256=H["chexbert"])
        plan={"policy":policy,"workers":workers}
        case={"anchor":anchor.record(),"requests":rows}

        class Journal:
            def __init__(self,path): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def append(self,event): events.append(copy.deepcopy(event))

        class Backend:
            frozen=True; execution_mode="approved_slurm_backend"
            def __init__(self,spec,operation_root,timeout,**kwargs):
                self.audit_sha256=_digest(spec); self.spec=spec; self.root=Path(operation_root)
            def invoke(self,request):
                if events[-1]["event"]!="attempt_reserved": raise AssertionError("not reserved before callback")
                calls.append(request.kind)
                if (timeout and request.operation_id=="xrv_0" and events[-1]["attempt_number"]==1):
                    raise TimeoutError("invented fixture")
                if failure and request.operation_id=="cxr_0": raise RuntimeError("invented fixture")
                slot=int(request.operation_id.split("_")[1]); cxr=images[slot]
                text={**copy.deepcopy(report),"parent_cxr_candidate_id":cxr["candidate_id"],
                    "candidate_id":"fixture_report_"+str(slot)}
                text["artifact"]["path"]="invented_report_not_on_disk_"+str(slot)
                outputs[str(self.root)]={"cxr":cxr,"report":text}
                return {"output_root":str(self.root),"worker_audit_sha256":self.audit_sha256}

        def single(payload,spec,*args,**kwargs):
            candidate=outputs[payload["output_root"]]["cxr" if spec["kind"]=="cxr_generator" else "report"]
            return Path(payload["output_root"])/"generated",candidate

        def read(path):
            path=Path(path); obj=outputs[str(path.parent.parent)]
            if path.name=="cxr_finding_labels.json":
                value=copy.deepcopy(labels); value["records"][0]["cxr_candidate_id"]=obj["cxr"]["candidate_id"]
                value["calibration"]["status"]="invented_frozen_status"
            else:
                value=copy.deepcopy(report_labels); value["records"][0]["report_candidate_id"]=obj["report"]["candidate_id"]
            return value

        def write(path,payload):
            if "receipt_id" in payload: receipts.append(copy.deepcopy(payload))
            return Path(path)

        def pack(**kwargs): return {"run_directory":str(Path(kwargs["output_root"])/kwargs["run_id"])}
        import io
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented_fixture"},clear=True), \
                patch.object(execution,"private_directory"),patch.object(execution,"copy_private_bytes"), \
                patch.object(execution,"clone_cxr_request",side_effect=lambda row,path:Path(path)), \
                patch.object(execution,"ProtectedJournal",Journal),patch.object(execution,"LocalFrozenBackend",Backend), \
                patch.object(execution,"single_generated",side_effect=single), \
                patch.object(execution,"read_json",side_effect=read),patch.object(execution,"require_inside",side_effect=lambda path,*a,**k:Path(path)), \
                patch.object(execution,"sha256_file",side_effect=lambda path:_digest([str(path)])), \
                patch.object(execution,"write_private_json",side_effect=write), \
                patch.object(execution,"validate_report_binding"),patch.object(reports,"prepare_report_request_run",side_effect=pack), \
                patch("tricompose_v12.runtime_dispatch._new_private_handle",side_effect=lambda p:io.StringIO()):
            triples,book,successful=execution.one_case(case,plan,Path("invented_root_not_created"))
        restored=restore_ledger(events,case_id=anchor.case_id,ehr_anchor_sha256=anchor.sha256,
            call_budget=10,max_retries=1,execution_mode="approved_slurm_backend",sink=lambda e:None)
        self.assertEqual(restored.snapshot(),book)
        return triples,book,successful,calls,receipts

    def test_two_slots_exact_phase_order_eight_calls_fixed_ehr(self):
        triples,book,successful,calls,receipts=self.exercise()
        self.assertEqual(calls,["cxr_generator","xrv","report_generator","chexbert"]*2)
        self.assertEqual(book["charged_model_attempts"],8); self.assertEqual(len(successful),8)
        self.assertEqual(len(triples),2); self.assertEqual(len(receipts),4)
        self.assertTrue(all(t["ehr_sha256"]==H["ehr"] for t in triples))
        self.assertTrue(all(t["clinical_acceptance"] is False and t["selected_as_best"] is False for t in triples))
        self.assertTrue(all(t["verification_status"]=="unverified_no_direct_ehr_constraints" for t in triples))

    def test_timeout_charged_before_identical_retry_but_no_extra_ehr(self):
        triples,book,successful,calls,_=self.exercise(timeout=True)
        self.assertEqual(len(triples),2); self.assertEqual(book["charged_model_attempts"],9)
        self.assertEqual(book["failed_attempts"],1); self.assertEqual(len(successful),8)
        self.assertEqual(calls.count("xrv"),3)
        reserved=[e for e in book["events"] if e["event"]=="attempt_reserved" and e["request"]["operation_id"]=="xrv_0"]
        self.assertEqual(reserved[0]["request_sha256"],reserved[1]["request_sha256"])

    def test_generator_failure_skips_dependent_models_keeps_other_slot(self):
        triples,book,successful,calls,_=self.exercise(failure=True)
        self.assertEqual(len(triples),1); self.assertEqual(book["charged_model_attempts"],5)
        self.assertEqual(len(successful),4); self.assertEqual(calls[:2],["cxr_generator","cxr_generator"])
        self.assertFalse(book["clinical_acceptance"])


if __name__=="__main__": unittest.main()
