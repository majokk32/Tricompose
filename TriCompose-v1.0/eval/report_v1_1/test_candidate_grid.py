import copy
import unittest
from itertools import product

from candidate_grid import SCHEMA, CXR_MODELS, REPORT_MODELS, path_key, validate_grid


def fixture(cases=2, seeds=(0, 1)):
    contract = {"schema_version": SCHEMA, "case_ids": [f"case_{i:03d}" for i in range(cases)],
                "cxr_seeds": list(seeds), "cxr_models": sorted(CXR_MODELS),
                "report_models": sorted(REPORT_MODELS)}
    rows = []
    for case, cxr, seed, report in product(contract["case_ids"], contract["cxr_models"], seeds, contract["report_models"]):
        cxr_id = f"{case}_{cxr}_{seed}"
        report_id = f"{cxr_id}_{report}"
        rows.append({"case_id": case, "triple_candidate_id": report_id,
                     "lineage": {"cxr_model_id": cxr, "cxr_seed": seed, "cxr_candidate_id": cxr_id,
                                 "report_model_id": report, "report_candidate_id": report_id}})
    return rows, contract


class CandidateGridTests(unittest.TestCase):
    def test_explicit_two_case_two_seed_grid_and_order_independence(self):
        rows, contract = fixture()
        self.assertEqual(len(rows), 48)
        self.assertEqual(validate_grid(rows, contract), contract)
        self.assertEqual(validate_grid(list(reversed(rows)), contract), contract)

    def test_legacy_80_case_single_seed_grid_stays_valid(self):
        rows, contract = fixture(80, (0,))
        self.assertEqual(len(rows), 960)
        self.assertEqual(validate_grid(rows), contract)

    def test_new_grid_cannot_silently_replace_historical_defaults(self):
        rows, _ = fixture()
        with self.assertRaisesRegex(ValueError, "explicit cohort"):
            validate_grid(rows)

    def test_missing_duplicate_wrong_case_and_cross_seed_parent(self):
        rows, contract = fixture()
        bad_case = copy.deepcopy(rows)
        bad_case[0]["case_id"] = "case_999"
        bad_seed_parent = copy.deepcopy(rows)
        bad_seed_parent[4]["lineage"]["cxr_candidate_id"] = rows[0]["lineage"]["cxr_candidate_id"]
        for bad in (rows[:-1], rows + [rows[0]], bad_case, bad_seed_parent):
            with self.subTest(), self.assertRaises(ValueError):
                validate_grid(bad, contract)

    def test_fixed_paths_include_seed_without_cherry_picking(self):
        rows, _ = fixture()
        paths = {path_key(r["lineage"]["cxr_model_id"], r["lineage"]["report_model_id"],
                          r["lineage"]["cxr_seed"], multiple_seeds=True) for r in rows}
        self.assertEqual(len(paths), 24)
        self.assertEqual(path_key("sana", "maira2", 0, multiple_seeds=False), "sana->maira2")


if __name__ == "__main__":
    unittest.main()
