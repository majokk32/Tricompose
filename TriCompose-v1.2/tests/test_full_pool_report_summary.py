"""Invented metadata: pooled summary must preserve missingness and pairing."""
import argparse
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audits"))
from summarize_full_pool_report_control import aggregates, diagnostic, run
from tricompose_v12.full_pool_report_control import freeze, summarize
from test_full_pool_report_control import rows, endpoint


class FullPoolSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows(80)
        cls.comparison = summarize(cls.rows, freeze(cls.rows), endpoint(cls.rows))

    def test_all_five_policies_preserve_complete_case_and_image_counts(self):
        result = aggregates(self.comparison)
        self.assertEqual(len(result), 5)
        self.assertTrue(all((r["fixed_ehr_cases"], r["fixed_images"], r["biovil_available_pairs"]) == (80, 240, 240) for r in result))

    def test_negative_selected_secondary_change_not_hidden(self):
        result = {r["report_policy"]: r for r in aggregates(self.comparison)}
        self.assertAlmostEqual(result["cxrmate_single"]["mean_biovil_raw_cosine"], .9)
        self.assertAlmostEqual(result["first_eligible_expert"]["mean_biovil_raw_cosine"], .1)
        info = diagnostic(self.rows, self.comparison)
        self.assertEqual((info["changed_images"], info["ehr_cases_with_at_least_one_report_switch"], info["secondary_decreased_images"]), (240, 80, 240))

    def test_one_unavailable_path_does_not_become_two_path_mean(self):
        comparison = copy.deepcopy(self.comparison)
        comparison["model_comparison"][0]["mean_biovil_raw_cosine"] = None
        self.assertIsNone(aggregates(comparison)[0]["mean_biovil_raw_cosine"])

    def test_missing_duplicate_and_wrong_case_path_rejected(self):
        comparison = copy.deepcopy(self.comparison)
        comparison["model_comparison"].pop(0)
        with self.assertRaises(ValueError):
            aggregates(comparison)
        comparison = copy.deepcopy(self.comparison)
        comparison["model_comparison"][0]["fixed_ehr_cases"] = 79
        with self.assertRaises(ValueError):
            aggregates(comparison)

    def test_pooled_edge_totals_are_sums_not_rounded_rate_averages(self):
        result = aggregates(self.comparison)[0]
        sources = [r for r in self.comparison["model_comparison"] if r["report_policy"] == result["report_policy"]]
        edge = result["raw_edge_totals"]["cxr_report"]
        self.assertEqual(edge["supported_facts"], sum(r["raw_edge_totals"]["cxr_report"]["supported_facts"] for r in sources))
        self.assertEqual(edge["support_over_known"], edge["supported_facts"] / edge["known_reference_facts"])

    def test_slurm_guard_precedes_input_read_or_directory_creation(self):
        with patch("summarize_full_pool_report_control.require_slurm", side_effect=RuntimeError), \
                patch("summarize_full_pool_report_control.read_json") as reader:
            with self.assertRaises(RuntimeError):
                run(argparse.Namespace())
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
