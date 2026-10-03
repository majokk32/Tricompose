"""Invented labels only; no real patient data are opened."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_validation"))
from hard_negative_biovil import FINDINGS, _id, _state, select_hard_donor, summarize  # noqa: E402


def labels(**changes):
    result = {finding: "unknown" for finding in FINDINGS}
    result.update(changes)
    return result


class HardNegativeTests(unittest.TestCase):
    def test_selects_same_view_shared_positive_and_explicit_conflict(self):
        anchor = {"row_index": 0, "subject_id": "1", "split": "val", "view": "AP",
                  "labels": labels(**{"Cardiomegaly": "positive", "Edema": "positive"})}
        pool = [
            {"row_index": 1, "subject_id": "1", "split": "val", "view": "AP",
             "labels": labels(**{"Cardiomegaly": "positive", "Edema": "negative"})},
            {"row_index": 2, "subject_id": "2", "split": "val", "view": "PA",
             "labels": labels(**{"Cardiomegaly": "positive", "Edema": "negative"})},
            {"row_index": 3, "subject_id": "3", "split": "val", "view": "AP",
             "labels": labels(**{"Cardiomegaly": "positive", "Edema": "negative"})},
        ]
        chosen = select_hard_donor(anchor, pool, seed=7)
        self.assertEqual(chosen["donor"]["row_index"], 3)
        self.assertEqual(chosen["shared_positive_findings"], ["Cardiomegaly"])
        self.assertEqual(chosen["explicit_conflicting_findings"], ["Edema"])
        self.assertEqual(chosen, select_hard_donor(anchor, pool, seed=7))

    def test_unknown_is_not_negative(self):
        anchor = {"row_index": 0, "subject_id": "1", "split": "test", "view": "PA",
                  "labels": labels(**{"Pneumonia": "positive", "Edema": "unknown"})}
        donor = {"row_index": 1, "subject_id": "2", "split": "test", "view": "PA",
                 "labels": labels(**{"Pneumonia": "positive", "Edema": "negative"})}
        self.assertIsNone(select_hard_donor(anchor, [donor], seed=1))
        self.assertEqual(_state(""), "unknown")
        self.assertEqual(_state("-1.0"), "uncertain")
        self.assertEqual(_id("s00123"), "123")

    def test_summary_reports_unavailable_separately(self):
        rows = [
            {"split": "val", "matched_score": 0.8, "random_score": 0.1, "hard_score": 0.7},
            {"split": "test", "matched_score": 0.6, "random_score": 0.0, "hard_score": 0.7},
        ]
        summary = summarize(rows, {"val": 1, "test": 2})
        self.assertEqual(summary["val"]["hard_negative_cases"], 1)
        self.assertEqual(summary["test"]["unavailable"], 2)
        self.assertEqual(summary["test"]["paired_win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
