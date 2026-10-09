"""Invented labels, texts and fake runtimes only; no model execution or pixels."""
import argparse
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"benchmarks"))
from tricompose_v12 import report_nbest as nbest
from tricompose_v12.invariant_verification import EHRAnchor, _digest
from test_live_receipts import data, full
from run_report_nbest import single_view, native_kwargs, run, generate


def seal(receipt):
    receipt["receipt_id"] = _digest({k: v for k, v in receipt.items() if k != "receipt_id"})


def rows():
    result = []
    for index in range(2):
        for rank in range(3):
            a, i, il, r, tl = copy.deepcopy(data())
            a = EHRAnchor("fixture_nbest_"+str(index), a.ehr_sha256, a.ehr_facts_sha256, a.findings)
            i.update(case_id=a.case_id, candidate_id="fixture_image_"+str(index))
            il["records"][0]["cxr_candidate_id"] = i["candidate_id"]
            r.update(case_id=a.case_id, candidate_id="fixture_report_"+str(index)+"_"+str(rank), parent_cxr_candidate_id=i["candidate_id"])
            r["artifact"]["sha256"] = _digest(["invented_text", index, rank])
            tl["records"][0].update(report_candidate_id=r["candidate_id"], report_sha256=r["artifact"]["sha256"])
            tl["records"][0]["finding_states"]["edema"] = "negative" if rank == 0 else "positive"
            receipt = full(a, i, il, r, tl)
            structure = {k: True for k in nbest.PASS_FLAGS} | {k: False for k in nbest.RISK_FLAGS}
            structure.update(repeated_sentence_count=0, repeated_4gram_ratio=0., normalized_report_sha256=r["artifact"]["sha256"])
            result.append({"case_id": a.case_id, "cxr_candidate_id": i["candidate_id"], "report_candidate_id": r["candidate_id"],
                "cxr_sha256": i["artifact"]["sha256"], "report_sha256": r["artifact"]["sha256"],
                "ehr_sha256": a.ehr_sha256, "ehr_facts_sha256": a.ehr_facts_sha256,
                "triple_candidate_id": "nbestpair_"+_digest([i["candidate_id"], r["candidate_id"]])[:32],
                "beam_rank": rank, "receipt": receipt, "structure": structure})
    return result


def endpoint(rs):
    return {"used_for_routing": False, "clinical_truth_available": False, "records": [
        {k: r[k] for k in ("case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")}
        | {"biovil_raw_cosine": .8 if r["beam_rank"] == 0 else .2, "status": "computed_secondary_uncalibrated", "reason": None, "calibrated": False}
        for r in rs]}


class NBestTests(unittest.TestCase):
    def test_native_settings_not_seed_sampling(self):
        fake = SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0, sep_token_id=3))
        kw = native_kwargs(fake)
        self.assertEqual((kw["num_beams"], kw["num_return_sequences"], kw["max_length"]), (4, 3, 256))
        self.assertFalse(kw["do_sample"]); self.assertNotIn("temperature", kw)
        self.assertNotIn("seed", kw); self.assertNotIn("pixel_values", kw)

    def test_native_section_format_and_missing_section(self):
        self.assertEqual(nbest.canonical_report("invented finding", "invented impression"),
                         "FINDINGS:\ninvented finding\n\nIMPRESSION:\ninvented impression\n")
        self.assertEqual(nbest.canonical_report(" invented ", ""), "FINDINGS:\ninvented\n")
        with self.assertRaises(ValueError): nbest.canonical_report("", " ")

    def test_frozen_policy_changes_rejected(self):
        nbest.validate_policy(copy.deepcopy(nbest.POLICY))
        for key, value in (("do_sample", True), ("num_beams", 8), ("num_return_sequences", 1),
            ("opaque_case_indices", [3,4]), ("uses_biovil", True), ("fixed_image_model", "roentgen_v2")):
            p = copy.deepcopy(nbest.POLICY); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): nbest.validate_policy(p)

    def test_strict_fact_correction_is_admissible_not_clinical(self):
        rs=rows(); c=nbest.compare(rs[0], rs[1])
        self.assertTrue(c["exploratory_gate_pass"]); self.assertFalse(c["clinical_repair_success"])
        self.assertEqual(c["removed_opposition_fact_ids"]["image_opposition"], ["edema"])

    def test_silencing_existing_conflict_is_not_correction(self):
        for state in ("unknown", "uncertain"):
            a,b=rows()[:2]; b["receipt"]["fact_states"][3]["chexbert"]=state; seal(b["receipt"])
            c=nbest.compare(a,b)
            self.assertFalse(c["exploratory_gate_pass"]); self.assertIn("edema", c["silenced_fact_ids"]["image"])

    def test_unknown_report_not_negative(self):
        row=rows()[0]
        for f in row["receipt"]["fact_states"]: f["chexbert"]="unknown"
        seal(row["receipt"]); sets=nbest.evidence_sets(row["receipt"])
        self.assertFalse(any(sets.values()))

    def test_unknown_image_report_positive_is_unverified_not_contradiction(self):
        row=rows()[1]; row["receipt"]["fact_states"][3]["xrv"]="unknown"; seal(row["receipt"])
        sets=nbest.evidence_sets(row["receipt"])
        self.assertEqual(sets["image_opposition"], set()); self.assertEqual(sets["image_positive_support"], set())

    def test_no_lost_image_positive_support(self):
        a,b=rows()[1:3]
        b["receipt"]["fact_states"][3]["chexbert"]="unknown"; seal(b["receipt"])
        c=nbest.compare(a,b)
        self.assertFalse(c["exploratory_gate_pass"]); self.assertEqual(c["lost_fact_ids"]["image_positive_support"], ["edema"])

    def test_new_opposition_rejected(self):
        a,b=rows()[:2]
        for row in (a,b):
            row["receipt"]["fact_states"][0].update(xrv="positive", chexbert="positive")
        b["receipt"]["fact_states"][0]["chexbert"]="negative"
        for row in (a,b): seal(row["receipt"])
        c=nbest.compare(a,b); self.assertFalse(c["exploratory_gate_pass"])
        self.assertEqual(c["new_opposition_fact_ids"]["image_opposition"], ["atelectasis"])

    def test_profile_or_checkpoint_or_reference_change_rejected(self):
        for key in ("profile", "thresholds_sha256", "xrv_checkpoint_sha256", "ehr_anchor_sha256", "cxr_sha256"):
            a,b=rows()[:2]; b["receipt"][key]="invented_changed"; seal(b["receipt"])
            with self.subTest(key=key), self.assertRaises(ValueError): nbest.compare(a,b)

    def test_receipt_digest_tamper_rejected(self):
        r=rows()[0]["receipt"]; r["fact_states"][3]["chexbert"]="positive"
        with self.assertRaises(ValueError): nbest.evidence_sets(r)

    def test_temporal_generic_and_missing_sections_cannot_worsen(self):
        for key,value in (("unsupported_temporal_comparison_language",True),("generic_report",True),
            ("impression_complete",False),("section_contract_pass",False),("empty",True),
            ("repeated_sentence_count",1),("repeated_4gram_ratio",.2)):
            a,b=rows()[:2]; b["structure"][key]=value
            with self.subTest(key=key): self.assertFalse(nbest.compare(a,b)["exploratory_gate_pass"])

    def test_duplicates_get_no_diversity_credit(self):
        for kind in ("exact", "normalized"):
            a,b=rows()[:2]
            if kind=="exact": b["receipt"]["report_sha256"]=a["receipt"]["report_sha256"]; seal(b["receipt"])
            else: b["structure"]["normalized_report_sha256"]=a["structure"]["normalized_report_sha256"]
            self.assertFalse(nbest.compare(a,b)["exploratory_gate_pass"])

    def test_choice_is_first_native_eligible_not_endpoint_oracle(self):
        rs=rows(); selection=nbest.freeze(rs)
        self.assertEqual(selection,nbest.freeze(copy.deepcopy(rs)))
        self.assertTrue(all(c["selected_triple_id"]==rs[1+3*i]["triple_candidate_id"] for i,c in enumerate(selection["choices"])))
        measured=nbest.measure(selection,rs,endpoint(rs))
        self.assertAlmostEqual(measured["two_case_mean_delta"],-.6)
        self.assertFalse(measured["clinical_repair_success"])

    def test_baseline_retained_if_all_alternatives_fail(self):
        rs=rows()
        for r in rs:
            if r["beam_rank"]: r["structure"]["empty"]=True
        selected=nbest.freeze(rs)
        self.assertTrue(all(c["selected_triple_id"]==c["baseline_triple_id"] for c in selected["choices"]))
        self.assertTrue(all(c["status"]=="unresolved_baseline_retained" for c in selected["choices"]))

    def test_incomplete_or_duplicate_inventory_rejected(self):
        for broken in (rows()[:3], rows()[:5], rows()+[rows()[0]], rows()[:5]+[rows()[0]]):
            with self.assertRaises(ValueError): nbest.freeze(broken)
        broken=rows(); broken[2]["beam_rank"]=1
        with self.assertRaises(ValueError): nbest.freeze(broken)

    def test_row_anchor_tamper_rejected(self):
        broken=rows(); broken[1]["ehr_sha256"]="a"*64
        with self.assertRaises(ValueError): nbest.freeze(broken)

    def test_endpoint_na_preserved_not_zero(self):
        rs=rows(); ep=endpoint(rs); ep["records"][1].update(biovil_raw_cosine=None,status="not_available",reason="context_limit")
        measured=nbest.measure(nbest.freeze(rs),rs,ep)
        self.assertIsNone(measured["two_case_mean_delta"]); self.assertEqual(measured["available_pairs"],1)

    def test_endpoint_validation(self):
        rs=rows(); s=nbest.freeze(rs)
        for key,value in (("ehr_sha256","0"*64),("biovil_raw_cosine",float("nan")),("calibrated",True),("reason","invented")):
            ep=endpoint(rs); ep["records"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): nbest.measure(s,rs,ep)
        ep=endpoint(rs); ep["records"]=ep["records"][:-1]
        with self.assertRaises(ValueError): nbest.measure(s,rs,ep)

    def test_endpoint_cannot_rewrite_selection(self):
        rs=rows(); s=nbest.freeze(rs); s["used_biovil"]=True
        with self.assertRaises(ValueError): nbest.measure(s,rs,endpoint(rs))

    def test_single_view_keeps_provenance_no_additional_calls(self):
        bundle={"counts":{"reports":2,"model_calls":2},"records":[{"id":"a"},{"id":"b"}],"producer":{"frozen":True}}
        before=copy.deepcopy(bundle); view=single_view(bundle,"reports","id","a")
        self.assertEqual(bundle,before); self.assertEqual(view["producer"],bundle["producer"])
        self.assertEqual(view["counts"],{"reports":1,"model_calls":1})
        with self.assertRaises(ValueError): single_view(bundle,"reports","id","missing")
        bundle["records"][1]["id"]="a"
        with self.assertRaises(ValueError): single_view(bundle,"reports","id","a")

    def test_gpu_guard_precedes_plan_open_or_directory_creation(self):
        with patch.dict(os.environ,{"SLURM_JOB_ID":"invented_cpu_only","CUDA_VISIBLE_DEVICES":""},clear=True), \
             patch("run_report_nbest.load") as loader:
            for function in (run,generate):
                with self.assertRaises(RuntimeError): function(argparse.Namespace())
            loader.assert_not_called()

    def test_native_call_receipts_written_before_generate_and_not_after_load(self):
        source=(ROOT/"benchmarks/run_report_nbest.py").read_text()
        self.assertLess(source.index('"reserved_before_generate"'),source.index('runtime.model.generate('))
        self.assertIn('os.fsync(handle.fileno())',source)
        self.assertIn('"model_calls": 1 if rank == 0 else 0',source)
        self.assertIn('except BaseException as exc:',source)
        self.assertIn('"automatic_resume": False',source)


if __name__=="__main__": unittest.main()
