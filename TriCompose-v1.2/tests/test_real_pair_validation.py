"""Synthetic metadata only; does not open real MIMIC inputs or models."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_validation"))
from biovil_matched_pairs import auc_ap, select_cases, summarize  # noqa: E402


class RealPairValidationTests(unittest.TestCase):
    def test_patient_disjoint_selection_and_derangement(self):
        rows = []
        for split in ("val", "test"):
            for number in range(5):
                rows.append({"split": split, "subject_id": f"{split}_{number}",
                             "study_id": f"study_{number}", "ViewPosition": "AP",
                             "cxr_path": f"image_{split}_{number}",
                             "report_path": f"report_{split}_{number}"})
        rows.append({**rows[0], "study_id": "second_study"})
        selected = select_cases(rows, count_per_split=3, seed=17)
        self.assertEqual(selected, select_cases(rows, count_per_split=3, seed=17))
        self.assertEqual(len(selected), 6)
        for split in ("val", "test"):
            subset = [row for row in selected if row["split"] == split]
            self.assertEqual(len({row["row_index"] for row in subset}), 3)
            self.assertTrue(all(row["row_index"] != row["donor_row_index"] for row in subset))
            self.assertTrue(all("subject_id" not in row for row in subset))

    def test_rejects_patient_overlap_across_splits(self):
        rows = [
            {"split": role, "subject_id": "same", "study_id": role,
             "ViewPosition": "AP", "cxr_path": "image", "report_path": "report"}
            for role in ("val", "test")
        ]
        with self.assertRaises(ValueError):
            select_cases(rows, count_per_split=2, seed=1)
        rows[0]["split"] = "train"
        with self.assertRaises(ValueError):
            select_cases(rows, count_per_split=2, seed=1)

    def test_auc_and_ap(self):
        self.assertEqual(auc_ap([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]),
                         {"auroc": 1.0, "average_precision": 1.0})
        self.assertEqual(auc_ap([1, 0], [0.5, 0.5])["auroc"], 0.5)
        records = [
            {"split": role, "matched_score": 0.9, "mismatched_score": 0.1}
            for role in ("val", "test")
        ]
        summary = summarize(records)
        self.assertEqual(summary["val"]["paired_win_rate"], 1.0)
        self.assertEqual(summary["test"]["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
