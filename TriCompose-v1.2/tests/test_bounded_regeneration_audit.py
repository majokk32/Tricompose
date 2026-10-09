"""Audit boundaries using invented metadata only; no model/data access."""
import ast
import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audits"))
from audit_bounded_regeneration import audit, verify_csv
from run_bounded_regeneration import tables
from test_bounded_regeneration import examples
from tricompose_v12 import bounded_regeneration as retry


class BoundedAuditTests(unittest.TestCase):
    def test_audit_guard_precedes_source_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch("audit_bounded_regeneration.load") as loader:
            with self.assertRaises(RuntimeError): audit(object())
            loader.assert_not_called()

    def test_audit_never_opens_text_or_pixels_or_instantiates_models(self):
        source = (ROOT / "audits/audit_bounded_regeneration.py").read_text()
        for unsafe in ("read_report_text(", "Image.open(", "model.generate(", "import torch"):
            self.assertNotIn(unsafe, source)
        calls = {node.func.id for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("_record", calls)
        self.assertIn("restore_ledger(", source)
        self.assertIn("completed_receipt(", source)
        self.assertIn("verify_csv(", source)

    def table(self, expected, actual):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "fixture.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(expected[0]))
                writer.writeheader()
                writer.writerows(actual)
            verify_csv(path, expected)

    def test_exact_csv_preserves_na_types_and_raw_counts(self):
        expected = [{"id": "fixture", "score": None, "count": 2, "clinical": False}]
        self.table(expected, [{"id": "fixture", "score": "NA", "count": 2, "clinical": False}])

    def test_na_cannot_be_zero_or_an_extra_missing_case(self):
        expected = [{"id": "fixture", "score": None}]
        for actual in ([{"id": "fixture", "score": 0}], [],
                       [{"id": "fixture", "score": "NA"}] * 2):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                self.table(expected, actual)

    def test_case_comparison_columns_are_scalar_not_nested_json(self):
        base, alt = examples()
        choice = retry.decision(base, base, alt)
        fields = ("case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id",
                  "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")
        endpoint = {"used_for_routing": False, "clinical_truth_available": False,
                    "historical_pool_is_untouched_test": False, "records": []}
        for row in (base, alt):
            endpoint["records"].append({k: row[k] for k in fields} | {"biovil_raw_cosine": .2,
                "status": "computed_secondary_uncalibrated", "reason": None, "calibrated": False})
        table, cases = tables([base, alt], [choice], endpoint)
        self.assertTrue(all(not isinstance(v, (dict, list)) for row in table + cases for v in row.values()))
        self.assertEqual(cases[0]["fixed_ehr_cxr_proxy_opposition_facts"], 1)
        self.assertEqual(cases[0]["bounded_retry_ehr_cxr_proxy_opposition_facts"], 0)


if __name__ == "__main__":
    unittest.main()
