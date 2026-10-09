"""Invented labels/metadata only: table and aggregate audit regressions."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audits"))
from audit_report_expert_control import aggregate, audit, check_csv
from run_report_expert_control import method_summary
from test_report_expert_control import rows, endpoint


class ReportExpertAuditTests(unittest.TestCase):
    def test_independent_aggregate_matches_worker(self):
        rs = rows()
        self.assertEqual(aggregate(rs, endpoint(rs)), method_summary(rs, endpoint(rs)))

    def test_missing_one_secondary_score_is_not_zero_or_available_mean(self):
        rs = rows()
        ep = endpoint(rs)
        ep["records"][0]["biovil_raw_cosine"] = None
        result = aggregate(rs, ep)[0]
        self.assertEqual(result["biovil_available_pairs"], 1)
        self.assertIsNone(result["two_case_mean_biovil_raw_cosine"])

    def test_missing_ehr_reference_keeps_rates_na(self):
        rs = rows()
        for row in rs:
            for name in ("ehr_cxr", "ehr_report"):
                edge = row["raw_edge_readouts"][name]
                for key in ("known_reference_facts", "comparable_facts", "supported_positive",
                            "supported_negative", "proxy_opposition_facts", "missing_comparisons"):
                    edge[key] = 0
        for result in aggregate(rs, endpoint(rs)):
            for name in ("ehr_cxr", "ehr_report"):
                for key in ("coverage_over_known", "support_over_known", "opposition_over_known"):
                    self.assertIsNone(result["raw_edge_totals"][name][key])

    def test_each_declared_expert_needs_two_reports(self):
        rs = rows()
        with self.assertRaises(ValueError):
            aggregate(rs[:-1], endpoint(rs))

    def check_table(self, expected, actual, raises=False):
        # No real/synthetic patient content. Keep temporary test files in workspace.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "fixture.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(actual[0]))
                writer.writeheader()
                writer.writerows(actual)
            if raises:
                with self.assertRaises(ValueError):
                    check_csv(path, expected, "opaque_id")
            else:
                check_csv(path, expected, "opaque_id")

    def test_csv_preserves_na_float_and_bool(self):
        self.check_table([{"opaque_id": "fixture", "value": None, "flag": False, "score": .25}],
                         [{"opaque_id": "fixture", "value": "NA", "flag": False, "score": .25}])

    def test_csv_na_cannot_be_zero(self):
        self.check_table([{"opaque_id": "fixture", "value": None}],
                         [{"opaque_id": "fixture", "value": 0}], raises=True)

    def test_csv_missing_extra_or_duplicate_rows_rejected(self):
        expected = [{"opaque_id": "fixture", "value": 1}]
        for actual in ([], expected + expected, expected + [{"opaque_id": "extra", "value": 1}]):
            if not actual:
                # A header-only file uses the same writer helper with no data rows.
                with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
                    path = Path(directory) / "fixture.csv"
                    path.write_text("opaque_id,value\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        check_csv(path, expected, "opaque_id")
            else:
                self.check_table(expected, actual, raises=True)

    def test_csv_changed_lineage_or_schema_rejected(self):
        expected = [{"opaque_id": "fixture", "value": 1}]
        for actual in ([{"opaque_id": "different", "value": 1}],
                       [{"opaque_id": "fixture", "value": 1, "extra": "x"}]):
            self.check_table(expected, actual, raises=True)

    def test_slurm_guard_precedes_private_input_reads(self):
        with patch("audit_report_expert_control.require_slurm", side_effect=RuntimeError), \
                patch("audit_report_expert_control.read_json") as reader:
            with self.assertRaises(RuntimeError):
                audit(object())
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
