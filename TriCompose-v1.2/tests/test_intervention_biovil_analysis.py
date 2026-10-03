"""Synthetic-only paired score tests; no model or protected artifact access."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from analyze_intervention_biovil import paired_summary  # noqa: E402
from build_targeted_interventions import build_items  # noqa: E402
from test_targeted_interventions import fixture  # noqa: E402


class InterventionBioViLAnalysisTests(unittest.TestCase):
    def test_paired_drop_is_not_called_localization(self) -> None:
        rows, reports, cxrs = fixture()
        items, _ = build_items(rows, reports, cxrs,
                               cxr_model="chexgenbench_sana", seed=0,
                               report_model="maira2")
        scores = [{"item_id": row["item_id"],
                   "biovil_raw_cosine": {
                       "no_corruption": 0.8, "report_swap": 0.4,
                       "cxr_swap": 0.3}[row["intervention_type"]]}
                  for row in items["intervention_key"]]
        result = paired_summary(items, scores)
        self.assertEqual(result["overall"]["report_swap"]["pairs"], 8)
        self.assertAlmostEqual(result["overall"]["report_swap"]["mean_control_minus_swap"], 0.4)
        self.assertAlmostEqual(result["overall"]["cxr_swap"]["mean_control_minus_swap"], 0.5)
        self.assertEqual(result["overall"]["cxr_swap"]["fraction_score_decreased"], 1.0)

    def test_missing_item_fails_closed(self) -> None:
        rows, reports, cxrs = fixture()
        items, _ = build_items(rows, reports, cxrs,
                               cxr_model="chexgenbench_sana", seed=0,
                               report_model="maira2")
        scores = [{"item_id": row["item_id"], "biovil_raw_cosine": 0.0}
                  for row in items["intervention_key"][:-1]]
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            paired_summary(items, scores)


if __name__ == "__main__":
    unittest.main()
