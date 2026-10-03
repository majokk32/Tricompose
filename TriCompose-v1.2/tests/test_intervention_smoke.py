"""Synthetic-only tests: no protected artifacts or patient data are read."""

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from build_intervention_smoke import build_items  # noqa: E402
from diagnose_cached_pair_sensitivity import summarize  # noqa: E402
from contracts import CHEXPERT_FINDINGS  # noqa: E402


def fake_rows(count=4):
    rows = []
    for index in range(count):
        rows.append({
            "case_id": f"case_{index:03d}",
            "triple_candidate_id": f"triple_{index}",
            "lineage": {
                "ehr_sha256": f"{index + 1:064x}",
                "cxr_sha256": f"{index + 11:064x}",
                "report_sha256": f"{index + 21:064x}",
                "cxr_candidate_id": f"cxr_{index}",
                "report_candidate_id": f"report_{index}",
                "cxr_model_id": "chexgenbench_sana",
                "cxr_seed": 0,
                "report_model_id": "maira2",
            },
        })
    return rows


class InterventionSmokeTests(unittest.TestCase):
    def test_three_arms_are_deterministic_and_blind(self):
        rows = fake_rows()
        first = build_items(rows, cxr_model="chexgenbench_sana", seed=0,
                            report_model="maira2")
        second = build_items(rows, cxr_model="chexgenbench_sana", seed=0,
                             report_model="maira2")
        self.assertEqual(first, second)
        self.assertEqual(len(first["blind_items"]), 12)
        self.assertEqual(Counter(row["intervention_type"] for row in
                                 first["intervention_key"]),
                         {"no_corruption": 4, "report_swap": 4, "cxr_swap": 4})
        self.assertTrue(all(set(row) == {"item_id"} for row in first["blind_items"]))
        self.assertTrue(all("intervention_type" not in row for row in first["resolver"]))
        self.assertTrue(all(row["clinical_mismatch_verified"] is False for row in
                            first["intervention_key"]))

    def test_swaps_change_only_one_artifact(self):
        output = build_items(fake_rows(), cxr_model="chexgenbench_sana", seed=0,
                             report_model="maira2")
        resolver = {row["item_id"]: row for row in output["resolver"]}
        for truth in output["intervention_key"]:
            observed = resolver[truth["item_id"]]
            kind = truth["intervention_type"]
            self.assertEqual(observed["displayed_cxr_sha256"] ==
                             truth["base_cxr_sha256"], kind != "cxr_swap")
            self.assertEqual(observed["displayed_report_sha256"] ==
                             truth["base_report_sha256"], kind != "report_swap")

    def test_rejects_incomplete_or_too_small_path(self):
        with self.assertRaises(ValueError):
            build_items(fake_rows(2), cxr_model="chexgenbench_sana", seed=0,
                        report_model="maira2")
        with self.assertRaises(ValueError):
            build_items(fake_rows(4), cxr_model="roentgen_v2", seed=0,
                        report_model="maira2")

    def test_cached_diagnostic_is_only_a_manipulation_check(self):
        def states(edema):
            result = {name: "unknown" for name in CHEXPERT_FINDINGS}
            result["edema"] = edema
            return result

        resolver = [
            {"item_id": "control", "displayed_cxr_candidate_id": "cxr_a",
             "displayed_cxr_sha256": "a", "displayed_report_candidate_id": "report_a",
             "displayed_report_sha256": "c"},
            {"item_id": "report_swap", "displayed_cxr_candidate_id": "cxr_a",
             "displayed_cxr_sha256": "a", "displayed_report_candidate_id": "report_b",
             "displayed_report_sha256": "d"},
            {"item_id": "cxr_swap", "displayed_cxr_candidate_id": "cxr_b",
             "displayed_cxr_sha256": "b", "displayed_report_candidate_id": "report_a",
             "displayed_report_sha256": "c"},
        ]
        truth = [{"item_id": row["item_id"], "base_triple_candidate_id": "base",
                  "intervention_type": row["item_id"] if row["item_id"] != "control"
                  else "no_corruption"} for row in resolver]
        images = {"cxr_a": {"sha256": "a", "states": states("positive")},
                  "cxr_b": {"sha256": "b", "states": states("negative")}}
        reports = {"report_a": {"sha256": "c", "states": states("positive")},
                   "report_b": {"sha256": "d", "states": states("negative")}}
        result = summarize(resolver, truth, images, reports)
        self.assertEqual(result["no_corruption"]["explicit_contradictions"], 0)
        self.assertEqual(result["report_swap"]["more_contradictions_than_control"], 1)
        self.assertEqual(result["cxr_swap"]["more_contradictions_than_control"], 1)


if __name__ == "__main__":
    unittest.main()
