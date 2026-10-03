"""Invented endpoint caches only; no image/text/model inspection."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import automatic_secondary as endpoint
from tricompose_v12.automatic_replay import METHODS, replay_case
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from test_legacy_automatic_replay import invented, policy
from score_automatic_replay_biovil import requested_pairs
import merge_automatic_secondary as cli


def fixture():
    scores, details = invented(); p = policy()
    bank = make_legacy_bank(scores, details, p)
    rows = []
    for case, grid in bank.items():
        scope = "explicit_fact_proxy" if case == "invented_case_0" else "no_direct_comparable_ehr_facts"
        for budget in p["model_call_budgets"]:
            for method in METHODS:
                for seed in p["random_seeds"] if method == "random" else [0]:
                    row = replay_case(grid, p, method, budget, random_seed=seed)
                    row["input_ehr_assessment_scope"] = scope
                    rows.append(row)
    request = {"schema_version": endpoint.REQUEST_SCHEMA, "pairs": requested_pairs(rows, scores),
        "modality_source": "fully_synthetic", "selection_used_biovil": False,
        "clinical_truth_available": False, "routing_or_calibration_update_allowed": False,
        "text_policy": "full_report_no_silent_truncation_overlength_is_na"}
    records = [{**pair, "biovil_raw_cosine": int(hashlib.sha256(pair["triple_candidate_id"].encode()).hexdigest()[:4], 16) / 65535,
                "status": "computed_secondary_uncalibrated", "reason": None, "calibrated": False}
               for pair in request["pairs"]]
    secondary = {"schema_version": endpoint.SCORE_SCHEMA, "status": "completed_secondary_biovil",
        "records": records, "used_for_routing": False, "primary_clinical_metric": False,
        "original_selection_changed": False, "clinical_truth_available": False,
        "producer": {"frozen": True, "model_id": "biovil_t", "text_policy": request["text_policy"]}}
    return rows, request, secondary


class AutomaticSecondaryTests(unittest.TestCase):
    def test_overlay_preserves_candidates_actions_proxy_metrics_and_source_objects(self):
        rows, request, scores = fixture(); before = copy.deepcopy((rows, request, scores))
        merged = endpoint.overlay(rows, request, scores)
        self.assertEqual(before, (rows, request, scores))
        for source, target in zip(rows, merged):
            self.assertEqual(source["selected_candidate_id"], target["selected_candidate_id"])
            self.assertEqual(source["action_trace"], target["action_trace"])
            self.assertEqual(source["selected_snapshot"]["optimization_proxy"], target["selected_snapshot"]["optimization_proxy"])
            self.assertIsNone(source["selected_snapshot"]["biovil_cosine_secondary_not_routing"])
            self.assertIsNotNone(target["selected_snapshot"]["biovil_cosine_secondary_not_routing"])

    def test_endpoint_scores_cannot_change_selection_or_routes(self):
        rows, request, scores = fixture(); changed = copy.deepcopy(scores)
        for record in changed["records"]: record["biovil_raw_cosine"] *= -1
        first, second = endpoint.overlay(rows, request, scores), endpoint.overlay(rows, request, changed)
        self.assertEqual([r["selected_candidate_id"] for r in first], [r["selected_candidate_id"] for r in second])
        self.assertEqual([r["action_trace"] for r in first], [r["action_trace"] for r in second])

    def test_missing_endpoint_stays_na_with_reason_not_zero(self):
        rows, request, scores = fixture()
        missing = scores["records"][0]; missing.update(biovil_raw_cosine=None, status="not_available", reason="empty_report")
        merged = endpoint.overlay(rows, request, scores)
        applicable = [r for r in merged if r["selected_candidate_id"] == missing["triple_candidate_id"]]
        self.assertTrue(applicable)
        for row in applicable:
            self.assertIsNone(row["selected_snapshot"]["biovil_cosine_secondary_not_routing"])
            self.assertEqual(row["selected_snapshot"]["secondary_endpoint_unavailable_reason"], "empty_report")

    def test_wrong_hash_case_missing_duplicate_or_extra_record_rejected(self):
        for kind in ("hash", "case", "missing", "duplicate", "extra_field"):
            rows, request, scores = fixture(); first = scores["records"][0]
            if kind == "hash": first["cxr_sha256"] = "a" * 64
            elif kind == "case": first["case_id"] = "another_invented_case"
            elif kind == "missing": scores["records"].pop()
            elif kind == "duplicate": scores["records"].append(copy.deepcopy(first))
            else: first["selection_weight"] = 1
            with self.assertRaises(ValueError): endpoint.overlay(rows, request, scores)

    def test_routing_clinical_truth_or_unfrozen_claim_rejected(self):
        for kind in ("routing", "clinical", "unfrozen", "text_policy"):
            rows, request, scores = fixture()
            if kind == "routing": scores["used_for_routing"] = True
            elif kind == "clinical": scores["clinical_truth_available"] = True
            elif kind == "unfrozen": scores["producer"]["frozen"] = False
            else: request["text_policy"] = "silent_truncation"
            with self.assertRaises(ValueError): endpoint.overlay(rows, request, scores)

    def test_invalid_nan_bool_cosine_calibration_or_na_status_rejected(self):
        for value in (float("nan"), True, 2.0):
            rows, request, scores = fixture(); scores["records"][0]["biovil_raw_cosine"] = value
            with self.assertRaises(ValueError): endpoint.overlay(rows, request, scores)
        rows, request, scores = fixture(); scores["records"][0]["calibrated"] = True
        with self.assertRaises(ValueError): endpoint.overlay(rows, request, scores)
        rows, request, scores = fixture(); scores["records"][0]["biovil_raw_cosine"] = None
        with self.assertRaises(ValueError): endpoint.overlay(rows, request, scores)

    def test_existing_secondary_value_cannot_be_overwritten(self):
        rows, request, scores = fixture(); rows[0]["selected_snapshot"]["biovil_cosine_secondary_not_routing"] = 0
        with self.assertRaises(ValueError): endpoint.overlay(rows, request, scores)

    def test_pair_counts_are_cases_not_random_replicates(self):
        rows, request, scores = fixture(); merged = endpoint.overlay(rows, request, scores)
        comparisons = endpoint.paired_case_comparisons(merged, policy())
        self.assertEqual(len(comparisons), 3 * 5 * 3)
        for row in comparisons:
            self.assertEqual(row["paired_available_ehr_cases"], 3 if row["ehr_evidence_subgroup"] == "all" else 1 if row["ehr_evidence_subgroup"] == "explicit_fact_proxy" else 2)
            self.assertFalse(row["significance_test_performed"])

    def test_one_missing_random_seed_does_not_become_best_of_available_seeds(self):
        rows, request, scores = fixture(); merged = endpoint.overlay(rows, request, scores)
        for row in merged:
            if row["method"] == "random" and row["random_seed"] == 0 and row["case_id"] == "invented_case_0":
                row["selected_snapshot"]["biovil_cosine_secondary_not_routing"] = None
        comparisons = endpoint.paired_case_comparisons(merged, policy())
        for row in comparisons:
            if row["ehr_evidence_subgroup"] == "all" and row["baseline"] == "random":
                self.assertEqual(row["paired_available_ehr_cases"], 2)
                self.assertEqual(row["unavailable_case_pairs"], 1)

    def test_unavailable_case_still_contributes_to_full_cohort_call_cost(self):
        rows, request, scores = fixture(); merged = endpoint.overlay(rows, request, scores)
        for row in merged: row["selected_snapshot"]["biovil_cosine_secondary_not_routing"] = None
        comparisons = endpoint.paired_case_comparisons(merged, policy())
        for row in comparisons:
            self.assertIsNone(row["mean_biovil_delta_targeted_minus_baseline"])
            self.assertEqual(row["paired_available_ehr_cases"], 0)
            self.assertIsNotNone(row["mean_simulated_call_delta_all_fixed_cases"])

    def test_missing_duplicate_seed_or_changed_fixed_ehr_rejected(self):
        rows, request, scores = fixture(); merged = endpoint.overlay(rows, request, scores)
        with self.assertRaises(ValueError): endpoint.paired_case_comparisons(merged[:-1], policy())
        with self.assertRaises(ValueError): endpoint.paired_case_comparisons([*merged, merged[0]], policy())
        changed = copy.deepcopy(merged); changed[0]["selected_ehr_sha256"] = "a" * 64
        with self.assertRaises(ValueError): endpoint.paired_case_comparisons(changed, policy())

    def test_input_order_does_not_change_paired_tables(self):
        rows, request, scores = fixture(); merged = endpoint.overlay(rows, request, scores)
        self.assertEqual(endpoint.paired_case_comparisons(merged, policy()), endpoint.paired_case_comparisons(list(reversed(merged)), policy()))

    def test_cli_refuses_before_any_private_read_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "cached") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
