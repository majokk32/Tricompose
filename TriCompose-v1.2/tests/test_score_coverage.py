"""Invented four-state/hash fixtures only; no medical artifacts or model calls."""
import argparse
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"benchmarks"))
from tricompose_v12 import score_coverage as c
from tricompose_v12.invariant_verification import _digest, _edge
import diagnose_score_coverage as cli


def record(case=0, image=0, model="expert_a", cosine=.2, profile=c.FRESH_PROFILE):
    states = [{"finding": n, "ehr": "unknown", "xrv": "unknown", "chexbert": "unknown"} for n in c.FINDINGS]
    states[0].update(xrv="positive", chexbert="positive")
    return {"profile": profile, "case_id": f"fixture_{case}", "cxr_candidate_id": f"image_{case}_{image}",
        "report_candidate_id": f"report_{case}_{image}_{model}", "report_model_id": model,
        "triple_candidate_id": f"pair_{case}_{image}_{model}",
        "ehr_sha256": _digest(["ehr",case]), "ehr_facts_sha256": _digest(["facts",case]),
        "cxr_sha256": _digest(["image",case,image]), "report_sha256": _digest(["report",case,image,model]),
        "fact_states": states, "source_primary_prefix": [0,-1] if profile == c.FRESH_PROFILE else [0,0,0,-1],
        "biovil_raw_cosine": cosine,
        "endpoint_unavailable_reason": None if cosine is not None else "outside_scored_selected_union"}


def update_prefix(r):
    edge = _edge(r["fact_states"], "xrv", "chexbert")
    missing = sum(f["xrv"] == "positive" and f["chexbert"] in {"unknown","uncertain"} for f in r["fact_states"])
    r["source_primary_prefix"] = [edge["proxy_opposition_facts"] + missing, -edge["supported_positive"]]


class CoverageTests(unittest.TestCase):
    def test_deterministic_and_never_mutates_source(self):
        rows = [record(model="expert_b"),record()]; before = copy.deepcopy(rows)
        result = c.analyze(rows,c.FRESH_PROFILE)
        self.assertEqual(result,c.analyze(list(reversed(rows)),c.FRESH_PROFILE))
        self.assertEqual(rows,before); self.assertFalse(result["summary"]["source_choices_changed"])

    def test_shared_image_and_ehr_states_not_counted_per_report(self):
        result = c.analyze([record(),record(model="expert_b")],c.FRESH_PROFILE)["summary"]
        self.assertEqual(result["image_groups"],1)
        self.assertEqual(sum(result["image_state_counts_one_vector_per_image_id"].values()),14)
        self.assertEqual(sum(result["ehr_state_counts_one_vector_per_case"].values()),14)
        self.assertEqual(sum(result["report_state_counts_one_vector_per_report_id"].values()),28)

    def test_unknown_uncertain_are_not_negative(self):
        r = record(); r["fact_states"][0]["chexbert"] = "uncertain"; update_prefix(r)
        result = c.analyze([r],c.FRESH_PROFILE)
        edge = result["rows"][0]["raw_edge_readouts"]["cxr_report"]
        self.assertEqual(edge["proxy_opposition_facts"],0)
        self.assertEqual(result["summary"]["report_state_counts_one_vector_per_report_id"]["negative"],0)
        self.assertEqual(result["rows"][0]["missing_positive_image_facts"],1)

    def test_no_ehr_constraints_stays_na(self):
        result = c.analyze([record()],c.FRESH_PROFILE)
        self.assertEqual(result["summary"]["no_direct_ehr_cases"],1)
        self.assertIsNone(result["rows"][0]["raw_edge_readouts"]["ehr_report"]["support_over_known"])

    def test_zero_opposition_without_comparison_is_not_success(self):
        r = record(); r["fact_states"][0].update(xrv="negative",chexbert="unknown"); update_prefix(r)
        s = c.analyze([r],c.FRESH_PROFILE)["summary"]
        self.assertEqual(s["zero_opposition_but_no_comparable_candidates"],1)
        self.assertEqual(s["images_with_explicit_labels_all_negative"],1)
        self.assertFalse(s["clinical_acceptance"]); self.assertIsNone(s["clinical_accuracy"])

    def test_no_image_reference_is_not_all_negative(self):
        r = record(); r["fact_states"][0]["xrv"] = "unknown"; update_prefix(r)
        s = c.analyze([r],c.FRESH_PROFILE)["summary"]
        self.assertEqual(s["images_with_no_explicit_reference"],1)
        self.assertEqual(s["images_with_explicit_labels_all_negative"],0)

    def test_negative_support_is_counted_separately(self):
        r = record(); r["fact_states"][0].update(xrv="negative",chexbert="negative"); update_prefix(r)
        edge = c.analyze([r],c.FRESH_PROFILE)["rows"][0]["raw_edge_readouts"]["cxr_report"]
        self.assertEqual(edge["supported_positive"],0); self.assertEqual(edge["supported_negative"],1)

    def test_all_same_image_unordered_pairs_retained(self):
        s = c.analyze([record(model=f"expert_{i}") for i in range(4)],c.FRESH_PROFILE)["summary"]
        self.assertEqual(s["same_image_report_pairs"],6); self.assertEqual(s["primary_prefix_tied_pairs"],6)

    def test_endpoint_changes_diagnosis_not_source_prefix_or_winner(self):
        rows = [record(),record(model="expert_b",cosine=.9)]
        result = c.analyze(rows,c.FRESH_PROFILE)
        self.assertTrue(result["same_image_pairs"][0]["source_primary_prefix_tied"])
        self.assertAlmostEqual(result["summary"]["case_mean_absolute_biovil_gap_among_available_ties"],.7)
        self.assertFalse(result["summary"]["source_choices_changed"])

    def test_average_within_case_not_repeated_pairs_as_patients(self):
        rows = [record(cosine=0),record(model="expert_b",cosine=.2),
                record(image=1,cosine=0),record(image=1,model="expert_b",cosine=.4),
                record(case=1,cosine=0),record(case=1,model="expert_b",cosine=.9)]
        s = c.analyze(rows,c.FRESH_PROFILE)["summary"]
        self.assertAlmostEqual(s["case_mean_absolute_biovil_gap_among_available_ties"],.6)
        self.assertEqual(s["endpoint_available_tied_cases"],2)

    def test_missing_endpoint_not_imputed_or_excluded_from_inventory(self):
        s = c.analyze([record(),record(model="expert_b",cosine=None)],c.FRESH_PROFILE)["summary"]
        self.assertEqual(s["report_candidates"],2); self.assertEqual(s["same_image_report_pairs"],1)
        self.assertEqual(s["both_endpoint_available_pairs"],0)
        self.assertIsNone(s["case_mean_absolute_biovil_gap_among_available_ties"])
        self.assertEqual(s["endpoint_unavailable_reasons"],{"outside_scored_selected_union":1})

    def test_missing_endpoint_reason_required(self):
        r = record(cosine=None); r["endpoint_unavailable_reason"] = None
        with self.assertRaises(ValueError):c.analyze([r],c.FRESH_PROFILE)

    def test_invalid_cosines_rejected(self):
        for value in (True,float("nan"),float("inf"),2):
            r = record(cosine=value)
            with self.assertRaises(ValueError):c.analyze([r],c.FRESH_PROFILE)

    def test_profiles_never_pooled(self):
        with self.assertRaises(ValueError):c.analyze([record(),record(profile=c.LEGACY_PROFILE)],c.FRESH_PROFILE)
        with self.assertRaises(ValueError):c.analyze([record()],"invented_profile")

    def test_unavailable_legacy_prefix_not_equal_successful_scores(self):
        rows = [record(profile=c.LEGACY_PROFILE),record(model="expert_b",profile=c.LEGACY_PROFILE)]
        for r in rows:r["source_primary_prefix"][-1] = None
        s = c.analyze(rows,c.LEGACY_PROFILE)["summary"]
        self.assertEqual(s["primary_prefix_tied_pairs"],0); self.assertEqual(s["primary_prefix_unavailable_pairs"],1)

    def test_duplicate_incomplete_and_invalid_states_rejected(self):
        r = record()
        with self.assertRaises(ValueError):c.analyze([r,r],c.FRESH_PROFILE)
        for states in (r["fact_states"][:-1], [*r["fact_states"][:-1],r["fact_states"][0]]):
            changed = {**r,"fact_states":states}
            with self.assertRaises(ValueError):c.analyze([changed],c.FRESH_PROFILE)
        changed = copy.deepcopy(r); changed["fact_states"][0]["ehr"] = "unavailable_as_negative"
        with self.assertRaises(ValueError):c.analyze([changed],c.FRESH_PROFILE)

    def test_shared_image_or_fixed_ehr_changes_rejected(self):
        for field in ("ehr_sha256","ehr_facts_sha256","cxr_sha256"):
            rows = [record(),record(model="expert_b")];rows[1][field] = _digest("changed")
            with self.assertRaises(ValueError):c.analyze(rows,c.FRESH_PROFILE)
        rows = [record(),record(model="expert_b")];rows[1]["fact_states"][1]["xrv"] = "negative"
        with self.assertRaises(ValueError):c.analyze(rows,c.FRESH_PROFILE)

    def test_false_fresh_prefix_rejected(self):
        r = record(); r["source_primary_prefix"] = [0,0]
        with self.assertRaises(ValueError):c.analyze([r],c.FRESH_PROFILE)

    def test_report_hash_duplicates_explicit_not_extra_diversity(self):
        rows = [record(),record(model="expert_b")];rows[1]["report_sha256"] = rows[0]["report_sha256"]
        result = c.analyze(rows,c.FRESH_PROFILE)
        self.assertEqual(result["summary"]["unique_report_artifact_hashes"],1)
        self.assertTrue(result["same_image_pairs"][0]["same_report_artifact"])

    def test_login_guard_precedes_any_cache_load(self):
        with patch.dict(os.environ,{},clear=True), patch.object(cli,"fresh") as load:
            with self.assertRaises(RuntimeError):cli.run(argparse.Namespace())
            load.assert_not_called()


if __name__ == "__main__":unittest.main()
