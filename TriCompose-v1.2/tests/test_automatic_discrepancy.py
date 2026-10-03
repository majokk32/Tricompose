"""Invented state/endpoint caches only, not clinical validity tests."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from tricompose_v12 import automatic_discrepancy as diagnostic
from tricompose_v12.automatic_secondary import overlay
from tricompose_v12.legacy_replay_adapter import FINDINGS, make_legacy_bank
from tricompose_v12.automatic_replay import replay_case
from test_automatic_secondary import fixture
from test_legacy_automatic_replay import invented, policy
import diagnose_automatic_discrepancy as cli


def data():
    rows, request, scores = fixture()
    source, details = invented()
    bank = make_legacy_bank(source, details, policy())
    return overlay(rows, request, scores), bank


def facts():
    return [{"finding": name, "states": {"ehr": "unknown", "xrv": "unknown", "chexbert": "unknown"}}
            for name in FINDINGS]


class AutomaticDiscrepancyTests(unittest.TestCase):
    def test_positive_negative_support_and_opposition_are_separate(self):
        rows = facts()
        rows[0]["states"].update(ehr="positive", chexbert="positive")
        rows[1]["states"].update(ehr="negative", chexbert="negative")
        rows[2]["states"].update(ehr="positive", chexbert="negative")
        result = diagnostic.polarity_profile(rows, "ehr_report")
        self.assertEqual(result["support_positive"], 1)
        self.assertEqual(result["support_negative"], 1)
        self.assertEqual(result["opposition_positive_reference"], 1)

    def test_unknown_uncertain_never_become_negative_or_hallucination(self):
        rows = facts()
        rows[0]["states"].update(ehr="unknown", chexbert="positive")
        rows[1]["states"].update(ehr="uncertain", chexbert="positive")
        rows[2]["states"].update(ehr="positive", chexbert="uncertain")
        result = diagnostic.polarity_profile(rows, "ehr_report")
        self.assertEqual(result["reference_negative"], 0)
        self.assertEqual(result["support_negative"], 0)
        self.assertEqual(result["opposition_positive_reference"], 0)
        self.assertEqual(result["missing_candidate_on_positive_reference"], 1)
        self.assertEqual(result["candidate_positive_on_unknown_reference_not_hallucination"], 1)

    def test_incomplete_duplicate_or_invalid_four_state_inventory_rejected(self):
        with self.assertRaises(ValueError): diagnostic.polarity_profile(facts()[:-1], "ehr_report")
        rows = facts(); rows[0]["finding"] = rows[1]["finding"]
        with self.assertRaises(ValueError): diagnostic.polarity_profile(rows, "ehr_report")
        rows = facts(); rows[0]["states"]["ehr"] = "missing_as_negative"
        with self.assertRaises(ValueError): diagnostic.polarity_profile(rows, "ehr_report")

    def test_artifact_change_uses_hashes_not_model_names(self):
        before = {"selected_snapshot": {"artifact_hashes": {"ehr_sha256": "e", "ehr_facts_sha256": "f", "cxr_sha256": "i", "report_sha256": "r"}}}
        for field, expected in ((None, "same_artifacts"), ("report_sha256", "same_image_report_changed"),
                                ("cxr_sha256", "image_changed_same_report_artifact")):
            after = copy.deepcopy(before)
            if field: after["selected_snapshot"]["artifact_hashes"][field] = "new"
            self.assertEqual(diagnostic.artifact_change(before, after), expected)
        after = copy.deepcopy(before); after["selected_snapshot"]["artifact_hashes"].update(cxr_sha256="new_image", report_sha256="new_report")
        self.assertEqual(diagnostic.artifact_change(before, after), "both_image_and_report_changed")

    def test_changed_ehr_is_rejected_not_a_repair(self):
        rows, bank = data(); after = copy.deepcopy(rows[0])
        after["selected_snapshot"]["artifact_hashes"]["ehr_sha256"] = "a" * 64
        with self.assertRaises(ValueError): diagnostic.artifact_change(rows[0], after)

    def test_legacy_global_normal_adjustment_does_not_change_raw_states(self):
        scores, details = invented(("positive",), no_finding=True)
        grid = make_legacy_bank(scores, details, policy())["invented_case_0"]
        outcome = replay_case(grid, policy(), "fixed", 4)
        result = diagnostic.point(outcome, grid[("chexgenbench_sana", 0, "maira2")])
        self.assertEqual(result["legacy_global_normal_adjustment"], 1)
        self.assertEqual(result["polarity_by_edge"]["ehr_report"]["missing_candidate_on_positive_reference"], 1)

    def test_no_explicit_support_has_na_share_not_zero(self):
        self.assertIsNone(diagnostic._mean([None, None]))
        self.assertEqual(diagnostic._mean([None, 0]), 0)

    def test_analysis_preserves_original_choices_sources_and_rules(self):
        rows, bank = data(); before = copy.deepcopy((rows, bank))
        result = diagnostic.analyze(rows, bank, policy())
        self.assertEqual(before, (rows, bank))
        self.assertEqual(len(result["contrasts"]), 3 * 5 * 2)
        self.assertEqual(result["new_model_calls"], 0)
        self.assertFalse(result["rules_or_winners_changed"])
        self.assertTrue(all(r["delta_support_positive_plus_negative_verified"] for r in result["contrasts"]))

    def test_replay_order_does_not_change_diagnostic_tables(self):
        rows, bank = data()
        self.assertEqual(diagnostic.analyze(rows, bank, policy()), diagnostic.analyze(list(reversed(rows)), bank, policy()))

    def test_model_frequency_counts_trials_not_independent_ehrs(self):
        rows, bank = data(); result = diagnostic.analyze(rows, bank, policy())
        cells = [r for r in result["model_selection_frequencies"] if r["method"] == "random" and r["model_call_budget"] == 30]
        self.assertEqual(sum(r["selection_trials"] for r in cells), 15)
        self.assertTrue(all(r["fixed_ehr_cases"] == 3 for r in cells))
        self.assertTrue(all(not r["is_clinical_model_ranking"] for r in cells))

    def test_common_image_controls_are_correlated_selected_union_not_full_bank(self):
        rows, bank = data(); result = diagnostic.analyze(rows, bank, policy())
        self.assertTrue(result["same_image_pairs"])
        for r in result["same_image_controls"]:
            self.assertEqual(r["sampling_scope"], "scored_selected_union_not_exhaustive_bank")
            self.assertFalse(r["pair_independence_assumed"])
        self.assertTrue(all(not r["clinical_fault_confirmed"] for r in result["same_image_pairs"]))

    def test_missing_biovil_keeps_primary_counts_and_available_case_denominator(self):
        rows, bank = data()
        for row in rows: row["selected_snapshot"]["biovil_cosine_secondary_not_routing"] = None
        result = diagnostic.analyze(rows, bank, policy())
        cells = [r for r in result["action_slices"] if r["artifact_change"] == "all_changes" and r["ehr_evidence_subgroup"] == "all"]
        self.assertTrue(all(r["biovil_paired_available_cases"] == 0 for r in cells))
        self.assertTrue(all(r["mean_biovil_delta_on_available_pairs"] is None for r in cells))
        self.assertTrue(all(r["selected_pair_cases"] == 3 for r in cells))

    def test_hash_proxy_total_or_inconsistent_endpoint_rejected(self):
        rows, bank = data(); changed = copy.deepcopy(rows)
        changed[0]["selected_snapshot"]["artifact_hashes"]["report_sha256"] = "b" * 64
        with self.assertRaises(ValueError): diagnostic.analyze(changed, bank, policy())
        changed = copy.deepcopy(rows); changed[0]["selected_snapshot"]["optimization_proxy"]["total_direct_support_count"] += 1
        with self.assertRaises(ValueError): diagnostic.analyze(changed, bank, policy())

    def test_cli_refuses_before_any_protected_read_without_slurm(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, "cached") as read:
            with self.assertRaisesRegex(RuntimeError, "Slurm"): cli.load(None)
            read.assert_not_called()

    def test_diagnostic_records_have_no_raw_patient_report_or_image_body(self):
        rows, bank = data(); result = str(diagnostic.analyze(rows, bank, policy()))
        for field in ("subject_id", "patient_id", "report_text", "image_path", "source_statement"):
            self.assertNotIn(field, result)


if __name__ == "__main__":
    unittest.main()
