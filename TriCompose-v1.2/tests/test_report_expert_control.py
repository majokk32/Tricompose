"""Invented four-expert states and metadata; no weights, bodies or GPU calls."""
import argparse
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"benchmarks"))
from tricompose_v12 import report_expert_control as expert
from tricompose_v12.invariant_verification import _digest
from test_report_nbest import rows as nbest_rows,seal
from run_report_expert_control import run,method_summary


def rows():
    source=nbest_rows();out=[]
    for index in range(2):
        for j,model in enumerate(expert.MODELS):
            r=copy.deepcopy(source[3*index+(1 if j else 0)])
            r.pop("beam_rank")
            rid="fixture_expert_"+str(index)+"_"+model;sha=_digest(["invented_expert_report",rid])
            r.update(report_candidate_id=rid,report_sha256=sha,report_model_id=model,
                triple_candidate_id="expertpair_"+_digest([r["cxr_candidate_id"],rid])[:32])
            r["receipt"].update(report_candidate_id=rid,report_sha256=sha);seal(r["receipt"])
            r["raw_edge_readouts"]=copy.deepcopy(r["receipt"]["raw_edge_readouts"])
            r["structure"].update(report_candidate_id=rid,report_sha256=sha,image_sha256=r["cxr_sha256"],
                case_id=r["case_id"],report_model_id=model,normalized_report_sha256=sha,
                impression_complete=model=="cxrmate_single",impression_required_by_model_contract=model=="cxrmate_single")
            out.append(r)
    return out


def endpoint(rs):
    return {"used_for_routing":False,"clinical_truth_available":False,"records":[
        {k:r[k] for k in ("case_id","triple_candidate_id","cxr_candidate_id","report_candidate_id","ehr_sha256","ehr_facts_sha256","cxr_sha256","report_sha256")}
        | {"biovil_raw_cosine":.9 if r["report_model_id"]=="cxrmate_single" else .1,
           "status":"computed_secondary_uncalibrated","reason":None,"calibrated":False} for r in rs]}


class ExpertTests(unittest.TestCase):
    def test_required_sections_are_model_specific(self):
        rs=rows()
        for row in rs:self.assertTrue(expert.structure_ok(row))
        for row in rs:
            if row["report_model_id"]!="cxrmate_single":self.assertFalse(row["structure"]["impression_complete"])
        self.assertTrue(expert.compare(rs[0],rs[1])["exploratory_gate_pass"])

    def test_legacy_nbest_gate_remains_unchanged(self):
        from tricompose_v12.report_nbest import compare as original
        rs=rows();self.assertFalse(original(rs[0],rs[1])["exploratory_gate_pass"])
        self.assertTrue(expert.compare(rs[0],rs[1])["exploratory_gate_pass"])
        self.assertNotEqual(expert.POLICY["schema_version"],"tricompose-fixed-image-report-nbest-v1")

    def test_impression_required_for_cxrmate_not_optional(self):
        r=rows()[0];r["structure"]["impression_complete"]=False
        with self.assertRaises(ValueError):expert.structure_ok(r)
        r["structure"]["section_contract_pass"]=False
        self.assertFalse(expert.structure_ok(r))

    def test_official_contract_tamper_rejected(self):
        for model_index in range(4):
            r=rows()[model_index];r["structure"]["impression_required_by_model_contract"]=not r["structure"]["impression_required_by_model_contract"]
            with self.subTest(model_index=model_index),self.assertRaises(ValueError):expert.structure_ok(r)

    def test_structure_lineage_rejected(self):
        for key in ("report_candidate_id","report_sha256","image_sha256","case_id","report_model_id"):
            r=rows()[0];r["structure"][key]="invented_changed"
            with self.subTest(key=key),self.assertRaises(ValueError):expert.structure_ok(r)

    def test_choice_is_predeclared_first_eligible_not_cosine_oracle(self):
        rs=rows();s=expert.freeze(rs)
        self.assertTrue(all(c["selected_model"]=="maira2" for c in s["choices"]))
        result=expert.measure(s,rs,endpoint(rs))
        self.assertAlmostEqual(result["two_case_mean_delta"],-.8)
        self.assertFalse(result["clinical_repair_success"]);self.assertFalse(s["used_biovil"])

    def test_failed_first_expert_can_use_next_predeclared_expert(self):
        rs=rows()
        for row in rs:
            if row["report_model_id"]=="maira2":row["structure"]["unsupported_temporal_comparison_language"]=True
        s=expert.freeze(rs);self.assertTrue(all(c["selected_model"]=="llavarad" for c in s["choices"]))

    def test_no_supported_or_comparable_fact_may_disappear(self):
        for state in ("unknown","uncertain"):
            a,b=rows()[:2]
            b["receipt"]["fact_states"][3]["chexbert"]=state;seal(b["receipt"])
            c=expert.compare(a,b)
            self.assertFalse(c["exploratory_gate_pass"]);self.assertEqual(c["silenced_fact_ids"]["image"],["edema"])

    def test_new_opposition_rejected(self):
        a,b=rows()[:2]
        for r in (a,b):r["receipt"]["fact_states"][0].update(xrv="positive",chexbert="positive")
        b["receipt"]["fact_states"][0]["chexbert"]="negative"
        for r in (a,b):seal(r["receipt"])
        self.assertFalse(expert.compare(a,b)["exploratory_gate_pass"])

    def test_unknown_and_uncertain_reference_are_not_negative(self):
        a,b=rows()[:2]
        for r in (a,b):r["receipt"]["fact_states"][3].update(xrv="unknown",ehr="uncertain");seal(r["receipt"])
        c=expert.compare(a,b)
        self.assertFalse(c["exploratory_gate_pass"]);self.assertFalse(any(c["new_opposition_fact_ids"].values()))

    def test_profile_checkpoint_anchor_and_image_cannot_change(self):
        for key in ("profile","thresholds_sha256","xrv_checkpoint_sha256","chexbert_checkpoint_sha256","ehr_anchor_sha256","cxr_sha256"):
            a,b=rows()[:2];b["receipt"][key]="invented_changed";seal(b["receipt"])
            with self.subTest(key=key),self.assertRaises(ValueError):expert.compare(a,b)

    def test_common_risks_and_repetition_cannot_worsen(self):
        for key,value in (("unsupported_temporal_comparison_language",True),("generic_report",True),
                          ("repeated_sentence_count",1),("repeated_4gram_ratio",.1)):
            a,b=rows()[:2];b["structure"][key]=value
            with self.subTest(key=key):self.assertFalse(expert.compare(a,b)["exploratory_gate_pass"])

    def test_exact_or_normalized_duplicate_not_diversity(self):
        a,b=rows()[:2];b["structure"]["normalized_report_sha256"]=a["structure"]["normalized_report_sha256"]
        c=expert.compare(a,b);self.assertTrue(c["duplicate"]);self.assertFalse(c["exploratory_gate_pass"])

    def test_no_change_retains_baseline_unresolved(self):
        rs=rows()
        for row in rs:
            row["receipt"]["fact_states"][3]["chexbert"]="negative";seal(row["receipt"])
        s=expert.freeze(rs)
        self.assertTrue(all(c["baseline_triple_id"]==c["selected_triple_id"] for c in s["choices"]))
        self.assertTrue(all(c["status"]=="unresolved_baseline_retained" for c in s["choices"]))

    def test_complete_unique_four_model_inventory_required(self):
        for broken in (rows()[:-1],rows()+[rows()[0]],rows()[:-1]+[rows()[0]]):
            with self.assertRaises(ValueError):expert.freeze(broken)
        rs=rows();rs[1]["report_model_id"]="unexpected"
        with self.assertRaises(ValueError):expert.freeze(rs)

    def test_metadata_row_anchor_does_not_drift(self):
        rs=rows();rs[1]["ehr_sha256"]="0"*64
        with self.assertRaises(ValueError):expert.freeze(rs)

    def test_endpoint_na_not_zero(self):
        rs=rows();ep=endpoint(rs);ep["records"][1].update(biovil_raw_cosine=None,status="not_available",reason="full_context_limit")
        result=expert.measure(expert.freeze(rs),rs,ep)
        self.assertIsNone(result["two_case_mean_delta"]);self.assertEqual(result["available_pairs"],1)

    def test_endpoint_lineage_and_values_validated(self):
        rs=rows();s=expert.freeze(rs)
        for key,value in (("ehr_sha256","0"*64),("calibrated",True),("biovil_raw_cosine",float("nan")),("status","made_up")):
            ep=endpoint(rs);ep["records"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):expert.measure(s,rs,ep)
        ep=endpoint(rs);ep["records"]=ep["records"][:-1]
        with self.assertRaises(ValueError):expert.measure(s,rs,ep)

    def test_endpoint_cannot_rewrite_selection(self):
        rs=rows();s=expert.freeze(rs);s["used_biovil"]=True
        with self.assertRaises(ValueError):expert.measure(s,rs,endpoint(rs))

    def test_gpu_guard_precedes_private_input_or_directory(self):
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented_cpu_only","CUDA_VISIBLE_DEVICES":""},clear=True),patch("run_report_expert_control.load") as loader:
            with self.assertRaises(RuntimeError):run(argparse.Namespace())
            loader.assert_not_called()

    def test_method_summary_no_ehr_truth_means_na_not_perfect_score(self):
        rs=rows()
        for r in rs:
            e=r["raw_edge_readouts"]["ehr_report"]
            for k in ("known_reference_facts","comparable_facts","supported_positive","supported_negative","proxy_opposition_facts","missing_comparisons"):e[k]=0
        summary=method_summary(rs,endpoint(rs))
        self.assertEqual(len(summary),4)
        self.assertTrue(all(r["raw_edge_totals"]["ehr_report"]["coverage_over_known"] is None for r in summary))

    def test_no_model_generators_in_verifier_controller(self):
        source=(ROOT/"benchmarks/run_report_expert_control.py").read_text()
        self.assertNotIn("FrozenCXRMateSingleRuntime",source);self.assertNotIn("model.generate(",source)
        self.assertIn('"new_generation_calls":0',source);self.assertIn("os.fsync(journal.fileno())",source)
        self.assertLess(source.index('selection=freeze(rows)'),source.index('child("biovil"'))
        self.assertIn('"automatic_resume":False',source)


if __name__=="__main__":unittest.main()
