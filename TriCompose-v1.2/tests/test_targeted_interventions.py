"""Synthetic-only contract tests; no protected artifact access."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from build_targeted_interventions import build_items, split_cases, validate_items  # noqa: E402


def fixture(count: int = 10):
    rows, reports, cxrs = [], {}, {}
    for index in range(count):
        case = f"case_{index:03d}"
        image_id, report_id = f"cxr_{index}", f"report_{index}"
        image_hash = f"{index + 11:064x}"
        report_hash = f"{index + 101:064x}"
        rows.append({
            "case_id": case, "triple_candidate_id": f"triple_{index}",
            "lineage": {"ehr_sha256": f"{index + 1:064x}",
                        "cxr_sha256": image_hash, "report_sha256": report_hash,
                        "cxr_candidate_id": image_id,
                        "report_candidate_id": report_id,
                        "cxr_model_id": "chexgenbench_sana", "cxr_seed": 0,
                        "report_model_id": "maira2"}})
        state = "positive" if index % 2 else "negative"
        findings = {"edema": state, "pleural_effusion": "unknown"}
        reports[report_id] = {"sha256": report_hash, "states": findings}
        cxrs[image_id] = {"sha256": image_hash, "states": findings}
    return rows, reports, cxrs


class TargetedInterventionTests(unittest.TestCase):
    def test_deterministic_blind_and_one_modality_only(self) -> None:
        rows, reports, cxrs = fixture()
        first, counts = build_items(rows, reports, cxrs,
                                    cxr_model="chexgenbench_sana", seed=0,
                                    report_model="maira2")
        second, _ = build_items(rows, reports, cxrs,
                                cxr_model="chexgenbench_sana", seed=0,
                                report_model="maira2")
        self.assertEqual(first, second)
        self.assertEqual(counts["split_cases"], {"development": 6,
                                                "calibration": 2,
                                                "final_test": 2})
        self.assertEqual(Counter(row["intervention_type"] for row in
                                 first["intervention_key"]),
                         {"no_corruption": 10, "report_swap": 8, "cxr_swap": 8})
        self.assertEqual(validate_items(first)["items"], 26)
        self.assertTrue(all(set(row) == {"item_id"} for row in first["blind_items"]))
        resolver = {row["item_id"]: row for row in first["resolver"]}
        for truth in first["intervention_key"]:
            observed = resolver[truth["item_id"]]
            kind = truth["intervention_type"]
            self.assertEqual(observed["displayed_cxr_sha256"] ==
                             truth["base_cxr_sha256"], kind != "cxr_swap")
            self.assertEqual(observed["displayed_report_sha256"] ==
                             truth["base_report_sha256"], kind != "report_swap")
            self.assertNotIn("intervention_type", observed)
            if kind != "no_corruption":
                self.assertEqual({truth["base_same_modality_state"],
                                  truth["donor_same_modality_state"]},
                                 {"positive", "negative"})
                self.assertFalse(truth["clinical_mismatch_verified"])

    def test_unknown_and_unavailable_are_not_forced(self) -> None:
        rows, reports, cxrs = fixture()
        for row in reports.values():
            row["states"] = {"edema": "unknown", "pleural_effusion": "unknown"}
        output, counts = build_items(rows, reports, cxrs,
                                     cxr_model="chexgenbench_sana", seed=0,
                                     report_model="maira2")
        self.assertEqual(counts["unavailable"]["report_swap"], 10)
        self.assertEqual(Counter(row["intervention_type"] for row in
                                 output["intervention_key"])["report_swap"], 0)

    def test_hash_mismatch_rejected(self) -> None:
        rows, reports, cxrs = fixture()
        reports["report_0"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            build_items(rows, reports, cxrs, cxr_model="chexgenbench_sana",
                        seed=0, report_model="maira2")

    def test_split_case_minimum(self) -> None:
        with self.assertRaises(ValueError):
            split_cases(["a", "b", "c", "d"], seed=1)

    def test_stratification_keeps_a_positive_donor_in_each_split(self) -> None:
        cases = [f"case_{index:03d}" for index in range(80)]
        positives = {"case_001", "case_023", "case_075"}
        splits = split_cases(cases, seed=2701,
                             positive_effusion_cases=positives)
        self.assertEqual(Counter(splits.values()),
                         {"development": 48, "calibration": 16,
                          "final_test": 16})
        self.assertEqual({splits[case] for case in positives},
                         {"development", "calibration", "final_test"})

    def test_rejects_answer_leakage_or_two_modality_swap(self) -> None:
        rows, reports, cxrs = fixture()
        output, _ = build_items(rows, reports, cxrs,
                                cxr_model="chexgenbench_sana", seed=0,
                                report_model="maira2")
        output["resolver"][0]["intervention_type"] = "report_swap"
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_items(output)
        del output["resolver"][0]["intervention_type"]
        answer = next(row for row in output["intervention_key"]
                      if row["intervention_type"] == "report_swap")
        displayed = next(row for row in output["resolver"]
                         if row["item_id"] == answer["item_id"])
        displayed["displayed_cxr_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "wrong number"):
            validate_items(output)


if __name__ == "__main__":
    unittest.main()
