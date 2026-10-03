"""Invented metadata/states only. No patient bodies, pixels or model execution."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"benchmarks"))
from tricompose_v12 import live_receipts as receipt
from tricompose_v12.invariant_verification import EHRAnchor, _digest
from tricompose_v12.legacy_replay_adapter import FINDINGS
from tricompose_v12.live_plan import anchor_from_record, validate_policy
from tricompose_v12.live_execution import score_csv, validate_cxr_binding

H = {k: _digest(["invented_not_real", k]) for k in ("ehr", "facts", "image", "report", "xrv", "chexbert", "thresholds", "image_labels", "report_labels")}


def data(known=True):
    anchor = EHRAnchor("fixture_live_0", H["ehr"], H["facts"], tuple((n, "positive" if known and n=="edema" else "unknown",
        ("diagnosis",) if known and n=="edema" else ()) for n in FINDINGS))
    image = {"case_id": anchor.case_id, "candidate_id": "fixture_cxr", "ehr_sha256": anchor.ehr_sha256,
        "ehr_facts_sha256": anchor.ehr_facts_sha256, "artifact": {"sha256": H["image"]}, "frozen_model": True}
    report = {"case_id": anchor.case_id, "candidate_id": "fixture_report", "parent_cxr_candidate_id": "fixture_cxr",
        "input_cxr": {"sha256": H["image"]}, "ehr_sha256_retained_for_lineage": H["ehr"],
        "ehr_facts_sha256_retained_for_lineage": H["facts"], "artifact": {"sha256": H["report"]}, "frozen_model": True,
        "model_input_signature": "single_current_synthetic_cxr", "structured_ehr_content_supplied_to_model": False,
        "source_report_or_real_target_supplied": False}
    thresholds = {n: {"enabled": n=="edema", "negative_max": .3, "positive_min": .7} for n in FINDINGS}
    labels = {"schema_version": "tricompose-cxr-finding-labels-v1.1", "finding_order": list(FINDINGS),
        "producer": {"frozen": True, "checkpoint_sha256": H["xrv"]}, "peak_vram_gib": .5,
        "score_semantics": "xrv_op_norm_0_1", "probability_semantics": False, "primary_metric_eligible": False,
        "calibration": {"source_sha256": H["thresholds"]}, "thresholds": thresholds,
        "counts": {"cxr_candidates": 1, "model_calls": 1}, "records": [{"cxr_candidate_id": "fixture_cxr",
            "image_sha256": H["image"], "finding_states": {n: "positive" if n=="edema" else "unknown" for n in FINDINGS},
            "finding_probabilities": {n: .9 if n=="edema" else None for n in FINDINGS}}]}
    report_labels = {"schema_version": "tricompose-report-finding-labels-v1.1", "finding_order": list(FINDINGS),
        "producer": {"frozen": True, "checkpoint_sha256": H["chexbert"]}, "peak_vram_gib": .2,
        "calibration": {"unknown_is_negative": False}, "counts": {"reports": 1, "model_calls": 1},
        "records": [{"report_candidate_id": "fixture_report", "report_sha256": H["report"],
            "finding_states": {n: "positive" if n=="edema" else "unknown" for n in FINDINGS}}]}
    return anchor, image, labels, report, report_labels


def partial(anchor, image, labels):
    return receipt.image_receipt(anchor, image, labels, label_sha256=H["image_labels"],
        thresholds_sha256=H["thresholds"], checkpoint_sha256=H["xrv"])


def full(anchor, image, labels, report, report_labels, old=None):
    return receipt.completed_receipt(anchor, old or partial(anchor, image, labels), image, labels, report, report_labels,
        image_labels_sha256=H["image_labels"], report_labels_sha256=H["report_labels"], thresholds_sha256=H["thresholds"],
        xrv_checkpoint_sha256=H["xrv"], chexbert_checkpoint_sha256=H["chexbert"])


class FreshReceiptTests(unittest.TestCase):
    def test_profile_separate_and_image_phase_cannot_claim_report(self):
        anchor, image, labels, _, _ = data(); value=partial(anchor,image,labels)
        self.assertEqual(value["profile"],receipt.PROFILE)
        self.assertIsNone(value["raw_edge_readouts"]["ehr_report"])
        self.assertIsNone(value["report_candidate_id"])
        self.assertEqual(value["report_lifecycle_status"],"not_generated")
        self.assertFalse(value["clinical_acceptance"])

    def test_complete_receipt_deterministic_all_three_edges(self):
        d=data(); a=full(*d); b=full(*copy.deepcopy(d))
        self.assertEqual(a,b); self.assertEqual(a["all_three_supported_facts"],1)
        self.assertEqual(set(a["raw_edge_readouts"]),{"ehr_cxr","ehr_report","cxr_report"})
        self.assertFalse(a["clinical_repair_success"])
        self.assertFalse(a["primary_clinical_metric_eligible"])

    def test_no_direct_ehr_keeps_na_not_success_or_dropped(self):
        value=full(*data(False))
        self.assertIsNone(value["all_three_support_over_known"])
        self.assertEqual(value["known_ehr_facts"],0)
        self.assertEqual(value["verification_status"],"unverified_no_direct_ehr_constraints")
        self.assertIsNone(value["raw_edge_readouts"]["ehr_report"]["coverage_over_known"])

    def test_disabled_heads_do_not_become_negative(self):
        anchor,image,labels,_,_=data()
        for name in ("support_devices","no_finding","fracture"):
            broken=copy.deepcopy(labels); broken["records"][0]["finding_states"][name]="negative"
            with self.assertRaises(ValueError): partial(anchor,image,broken)

    def test_score_must_support_recorded_frozen_state(self):
        anchor,image,labels,_,_=data()
        for score in (.1, .5, None, float("nan"), True, -1, 2):
            broken=copy.deepcopy(labels); broken["records"][0]["finding_probabilities"]["edema"]=score
            with self.assertRaises(ValueError): partial(anchor,image,broken)

    def test_uncertain_or_unknown_image_is_missing_not_negative(self):
        for score,state in ((None,"unknown"),(.5,"uncertain")):
            anchor,image,labels,_,_=data()
            labels["records"][0]["finding_probabilities"]["edema"]=score
            labels["records"][0]["finding_states"]["edema"]=state
            edge=partial(anchor,image,labels)["raw_edge_readouts"]["ehr_cxr"]
            self.assertEqual(edge["missing_comparisons"],1)
            self.assertEqual(edge["supported_negative"],0)
            self.assertEqual(edge["proxy_opposition_facts"],0)

    def test_image_checkpoint_threshold_and_score_space_mismatch(self):
        for path,value in ((["producer","checkpoint_sha256"],"f"*64),
            (["calibration","source_sha256"],"f"*64),(["probability_semantics"],True),
            (["primary_metric_eligible"],True),(["score_semantics"],"probability"),(["producer","frozen"],False)):
            anchor,image,labels,_,_=data(); target=labels
            for key in path[:-1]: target=target[key]
            target[path[-1]]=value
            with self.assertRaises(ValueError): partial(anchor,image,labels)

    def test_single_case_single_scorer_call_required(self):
        for counts in ({"cxr_candidates":2,"model_calls":2},{"cxr_candidates":1,"model_calls":0}):
            anchor,image,labels,_,_=data(); labels["counts"]=counts
            with self.assertRaises(ValueError): partial(anchor,image,labels)

    def test_ehr_and_report_source_leakage_rejected(self):
        for field,value in (("structured_ehr_content_supplied_to_model",True),
            ("source_report_or_real_target_supplied",True),("frozen_model",False),
            ("ehr_sha256_retained_for_lineage","a"*64),("parent_cxr_candidate_id","other_cxr")):
            d=data(); d[3][field]=value
            with self.assertRaises(ValueError): full(*d)

    def test_old_partial_tampering_even_when_rehashed_rejected(self):
        d=data(); old=partial(*d[:3]); old["clinical_acceptance"]=True
        old["receipt_id"]=_digest({k:v for k,v in old.items() if k!="receipt_id"})
        with self.assertRaises(ValueError): full(*d,old=old)

    def test_report_unknown_is_not_negative_or_global_no_finding_override(self):
        d=data(); d[4]["records"][0]["finding_states"]["edema"]="unknown"
        d[4]["records"][0]["finding_states"]["no_finding"]="positive"
        value=full(*d)
        self.assertEqual(value["raw_edge_readouts"]["ehr_report"]["proxy_opposition_facts"],0)
        self.assertEqual(value["raw_edge_readouts"]["ehr_report"]["missing_comparisons"],1)

    def test_explicit_opposition_preserved_without_clinical_fault_verdict(self):
        d=data(); d[4]["records"][0]["finding_states"]["edema"]="negative"
        value=full(*d)
        self.assertEqual(value["raw_edge_readouts"]["cxr_report"]["proxy_opposition_facts"],1)
        self.assertEqual(value["verification_status"],"explicit_proxy_opposition_unvalidated")
        self.assertIsNone(value["clinical_accuracy"])

    def test_cached_anchor_roundtrip_and_provenance_rejected(self):
        anchor=data()[0]; self.assertEqual(anchor_from_record(anchor.record()),anchor)
        bad=anchor.record(); bad["independent_clinical_truth_available"]=True
        with self.assertRaises(ValueError): anchor_from_record(bad)

    def test_policy_cannot_enrich_ehr_select_easy_cases_or_adapt(self):
        original=json.loads((ROOT/"configs/live_smoke_v1.json").read_text()); validate_policy(original)
        for field,value in (("opaque_case_indices",[5,6]),("retains_underconditioned_cases",False),
            ("changes_existing_ehr_or_prompts",True),("selection_or_adaptive_repair_enabled",True),
            ("clinical_acceptance_allowed",True),("new_training_allowed",True),("report_models",["maira2"])):
            bad=copy.deepcopy(original); bad[field]=value
            with self.assertRaises(ValueError): validate_policy(bad)

    def test_csv_preserves_na_and_raw_positive_negative_counts(self):
        value=full(*data(False))
        row={"case_id":"fixture_live_0","cxr_model_id":"roentgen_v2","seed":0,"report_model_id":"cxrmate_single",
             "verification_status":value["verification_status"],"raw_edge_readouts":value["raw_edge_readouts"]}
        text=score_csv([row]); self.assertIn("NA",text); self.assertIn("supported_positive",text)
        self.assertIn("coverage_over_known",text)


if __name__=="__main__": unittest.main()
