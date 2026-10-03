#!/usr/bin/env python3
"""TWO invented cases only: durable call accounting + verification hook smoke."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.execution_ledger import BoundedCallLedger, CallRequest, CallResult, BudgetExhausted, restore_ledger, SCHEMA
from tricompose_v12.invariant_verification import EHRAnchor, PROVENANCE, _digest, verify_candidate
from tricompose_v12.legacy_replay_adapter import FINDINGS
from tricompose_v12.partial_image_verification import CachedImageEvidence, verify_image_phase, bind_completed_report
from tricompose_v12.runtime_dispatch import ProtectedJournal, dispatch_operation, require_slurm
from contracts import (WORKSPACE, require_inside, sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

AUDIT = _digest(["invented_fixture_audit_not_model_weights"])
STATE_CACHE = _digest(["invented_fixture_vector_not_existing_patient_or_model_cache"])


def validate_config(config):
    if config != {
        "schema_version": "tricompose-bounded-execution-fixture-policy-v1",
        "execution_mode": "invented_fixture_no_models", "call_budget_per_case": 5,
        "max_retries_per_operation": 1,
        "case_ids": ["fixture_execution_000", "fixture_execution_001"],
        "clinical_acceptance_allowed": False, "inference_backend_installed": False,
        "fixture_results_are_generation_outputs": False,
    }:
        raise ValueError("only the predeclared invented fixture is implemented")


class FixtureBackend:
    frozen = True
    audit_sha256 = AUDIT
    execution_mode = "invented_fixture_no_models"

    def __init__(self, anchor, timeout_first_xrv):
        self.anchor = anchor; self.timeout_first_xrv = timeout_first_xrv
        self.partial = None; self.full = None; self.binding = None

    def invoke(self, request):
        # No real model, artifact body, patient input or filesystem read occurs.
        if request.kind == "xrv" and self.timeout_first_xrv:
            self.timeout_first_xrv = False
            raise TimeoutError("invented timeout")
        if request.kind == "cxr_generator":
            return {"artifact_sha256": _digest(["invented_image_not_a_file", request.case_id])}
        if request.kind == "report_generator":
            return {"artifact_sha256": _digest(["invented_report_not_a_file", request.operation_id, request.case_id])}
        if request.kind == "xrv":
            evidence = CachedImageEvidence(request.case_id, "fixture_cxr", "fixture_cxr_model", 0,
                self.anchor.ehr_sha256, self.anchor.ehr_facts_sha256, request.input_image_sha256,
                STATE_CACHE, tuple((n, "positive" if n == "edema" else "unknown") for n in FINDINGS), True)
            self.partial = verify_image_phase(evidence, self.anchor)
            return {"artifact_sha256": _digest(["invented_xrv_labels", request.case_id]),
                "receipt_id": self.partial["receipt_id"]}
        hashes = {"ehr_sha256": self.anchor.ehr_sha256, "ehr_facts_sha256": self.anchor.ehr_facts_sha256,
            "cxr_sha256": request.input_image_sha256, "report_sha256": request.input_report_sha256}
        facts = [{"finding": name, "case_id": request.case_id, "triple_candidate_id": "fixture_triple",
            "cxr_candidate_id": "fixture_cxr", "report_candidate_id": "fixture_report",
            "artifact_hashes": hashes, "evidence_id": _digest([request.case_id, name]),
            "states": {"ehr": state, "xrv": state, "chexbert": state},
            "source_categories": list(categories), "weak_context_promoted": False,
            "clinical_truth_verified": False, "provenance_resolution": PROVENANCE}
            for name, state, categories in self.anchor.findings]
        candidate = {"score_record": {"schema_version":"tricompose-edge-specific-selection-v1.1",
            "case_id": request.case_id, "triple_candidate_id":"fixture_triple",
            "lineage": {**hashes,"cxr_candidate_id":"fixture_cxr","report_candidate_id":"fixture_report"},
            "scoring":{"selection":{"hard_gate_failure_count":0},"modality_quality":{"cxr_basic_validity_pass":True}}},
            "facts":facts}
        self.full = verify_candidate(candidate, self.anchor)
        self.binding = bind_completed_report(self.partial, self.full, self.anchor)
        return {"artifact_sha256": _digest(["invented_chexbert_labels", request.case_id]),
            "receipt_id": self.full["receipt_id"]}


def validate_payload(payload, request):
    expected = {"artifact_sha256", "receipt_id"} if request.kind in {"xrv","chexbert"} else {"artifact_sha256"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("malformed invented backend result")
    return CallResult(payload["artifact_sha256"], payload.get("receipt_id"))


def run(args):
    require_slurm()  # Before any protected read/write or fixture dispatch.
    started = time.monotonic()
    config_path = require_inside(args.policy_config, WORKSPACE, must_exist=True)
    if config_path.stat().st_size > 16*1024: raise ValueError("unbounded fixture config")
    config = json.loads(config_path.read_text()); validate_config(config)
    sources = {"program":Path(__file__),"policy_config":config_path,
        "ledger":ROOT/"src/tricompose_v12/execution_ledger.py",
        "dispatcher":ROOT/"src/tricompose_v12/runtime_dispatch.py",
        "partial_verifier":ROOT/"src/tricompose_v12/partial_image_verification.py",
        "full_verifier":ROOT/"src/tricompose_v12/invariant_verification.py",
        "finding_inventory":ROOT/"src/tricompose_v12/legacy_replay_adapter.py",
        "protocol":WORKSPACE/"docs/bounded_execution_ledger_contract.md"}
    before = {k:sha256_file(p) for k,p in sources.items()}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    snapshots, receipts, files = [], [], []
    try:
        for index, case in enumerate(config["case_ids"]):
            anchor = EHRAnchor(case, _digest(["invented_ehr",case]), _digest(["invented_ehr_facts",case]),
                tuple((n,"positive" if n=="edema" else "unknown",("diagnosis",) if n=="edema" else ()) for n in FINDINGS))
            backend = FixtureBackend(anchor, timeout_first_xrv=index==1)
            journal_path=temporary/f"{case}.journal.jsonl"
            with ProtectedJournal(journal_path) as journal:
                book=BoundedCallLedger(case_id=case,ehr_anchor_sha256=anchor.sha256,
                    call_budget=config["call_budget_per_case"],max_retries=config["max_retries_per_operation"],
                    execution_mode=config["execution_mode"],sink=journal.append)
                def call(req, attempt):
                    log=temporary/f"{case}_{req.operation_id}_{attempt}.log"
                    result=dispatch_operation(book,req,backend,validate_payload,private_log_path=log)
                    files.append(log)
                    return result
                def req(oid, kind, model, **extra):
                    return CallRequest(operation_id=oid,case_id=case,ehr_anchor_sha256=anchor.sha256,
                        kind=kind,model_id=model,frozen_model_audit_sha256=AUDIT,seed=0,**extra)
                image_req=req("image_0","cxr_generator","fixture_cxr_model")
                image=call(image_req,1)
                xrv_req=req("xrv_0","xrv","fixture_xrv",parent_operation_id="image_0",input_image_sha256=image.output_artifact_sha256)
                xrv=call(xrv_req,1)
                if xrv is None: xrv=call(xrv_req,2)
                report_req=req("report_0","report_generator","fixture_report_model",parent_operation_id="xrv_0",input_image_sha256=image.output_artifact_sha256)
                report=call(report_req,1)
                cb_req=req("chexbert_0","chexbert","fixture_chexbert",parent_operation_id="report_0",
                    input_image_sha256=image.output_artifact_sha256,input_report_sha256=report.output_artifact_sha256)
                call(cb_req,1)
                unscored=False
                if index==0:
                    book.reuse(image_req); book.reuse(xrv_req)
                    second_req=req("report_1","report_generator","fixture_second_report_model",
                        parent_operation_id="xrv_0",input_image_sha256=image.output_artifact_sha256)
                    second=call(second_req,1)
                    second_cb=req("chexbert_1","chexbert","fixture_chexbert",parent_operation_id="report_1",
                        input_image_sha256=image.output_artifact_sha256,input_report_sha256=second.output_artifact_sha256)
                    try:
                        book.reserve(second_cb)
                    except BudgetExhausted:
                        unscored=True
                    else:
                        raise AssertionError("fixture budget did not stop the scorer")
                snapshot=book.snapshot()
                snapshot["fixture_generated_unscored_report_at_budget"]=unscored
                snapshots.append(snapshot)
                receipts.append({"case_id":case,"partial":backend.partial,"full":backend.full,"binding":backend.binding})
            stored=[json.loads(line) for line in journal_path.read_text().splitlines() if line]
            restored=restore_ledger(stored,case_id=case,ehr_anchor_sha256=anchor.sha256,call_budget=5,
                max_retries=1,execution_mode=config["execution_mode"],sink=lambda event:None)
            assert restored.snapshot()["events"]==snapshot["events"]
            files.append(journal_path)
        summary={"schema_version":SCHEMA,"status":"completed_invented_fixture_dispatch_smoke",
            "invented_cases":len(snapshots),"actual_model_calls":0,"actual_generation_outputs":0,
            "fixture_charged_attempts":sum(s["charged_model_attempts"] for s in snapshots),
            "fixture_failed_attempts":sum(s["failed_attempts"] for s in snapshots),
            "fixture_completed_full_receipts":len(receipts),"durable_journal_restore_pass":True,
            "clinical_acceptance":False,"clinical_repair_success":False,"inference_backend_installed":False,
            "runtime_seconds":round(time.monotonic()-started,6)}
        if before!={k:sha256_file(p) for k,p in sources.items()}: raise ValueError("frozen fixture source changed")
        files += [write_private_json(temporary/"summary.json",summary),
            write_private_json(temporary/"frozen_policy.json",config),
            write_private_json(temporary/"case_ledger_snapshots.json",{"cases":snapshots}),
            write_private_json(temporary/"invented_verification_receipts.json",{"cases":receipts}),
            write_private_text(temporary/"RESULTS_CN_EN.md", "# Bounded execution fixture / 有预算的执行层模拟测试\n\n"
                "TWO invented cases; zero patient/model/checkpoint inputs and zero actual generation.\n"
                "两个虚构病例，不是原 80 条病例、不是新生成的 CXR/Report，也不是临床或质量评估。\n\n"
                "- Four phases are charged separately. Failed/retried reservations keep their charge.\n"
                "- Case 000 reuses its CXR/XRV, then stops its second report scorer at the five-attempt budget; that report remains unscored.\n"
                "- Case 001 has an invented XRV timeout; retry costs another attempt, not a clinical contradiction or free call.\n"
                "- Partial/full verification hooks preserve the fixed EHR; durable hash-chained journals restore exactly.\n"
                "- No model or GPU ran, no job/API/download/training occurred. No clinical acceptance or repair claim.\n"
                "- Prospective model/provenance adapters are NOT installed yet. This prepares accounting and dispatch, not a completed inference run.\n")]
        write_private_json(temporary/"manifest.json",{"schema_version":SCHEMA,"run_id":args.run_id,
            "source_paths":{k:str(p) for k,p in sources.items()},"source_sha256":before,
            "artifacts":{p.name:{"sha256":sha256_file(p)} for p in files},
            "actual_model_calls":0,"inference_backend_installed":False,"clinical_accuracy_claim_allowed":False})
        commit_atomic_run(temporary,target)
    except Exception:
        # Safe ONLY for this invented/no-model fixture. A prospective run must
        # retain its charged/in-flight journal for worker reconciliation.
        discard_atomic_run(temporary); raise
    return target,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("policy-config","output-root","run-id"): parser.add_argument("--"+name,required=True)
    args=parser.parse_args(); os.umask(0o007)
    try: target,summary=run(args)
    except Exception as exc:
        print(json.dumps({"status":"failed","error_type":type(exc).__name__})); return 1
    print(json.dumps({"status":summary["status"],"runtime_seconds":summary["runtime_seconds"],
        "manifest_sha256":sha256_file(target/"manifest.json")}))
    return 0


if __name__=="__main__": raise SystemExit(main())
