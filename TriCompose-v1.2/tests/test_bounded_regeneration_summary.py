"""Invented aggregate score-table metadata; no clinical text or pixels."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audits"))
from summarize_bounded_regeneration import aggregate
from test_bounded_regeneration import examples


def data():
    rows, choices, records = [], [], []
    fields = ("case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id",
              "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")
    for index in range(2):
        base, alt = examples()
        for row in (base, alt):
            row["case_id"] = "fixture_summary_" + str(index)
            row["triple_candidate_id"] += "_" + str(index)
            rows.append(row)
            records.append({k: row[k] for k in fields} | {"biovil_raw_cosine": .2,
                "status": "computed_secondary_uncalibrated", "reason": None, "calibrated": False})
        choices.append({"case_id": base["case_id"], "baseline_triple_id": base["triple_candidate_id"],
            "static_triple_id": base["triple_candidate_id"], "alternative_triple_id": alt["triple_candidate_id"],
            "selected_triple_id": base["triple_candidate_id"]})
    return rows, {"choices": choices}, {"records": records, "used_for_routing": False,
        "clinical_truth_available": False, "historical_pool_is_untouched_test": False}


class BoundedSummaryTests(unittest.TestCase):
    def test_four_separate_methods_preserve_raw_evidence(self):
        results = aggregate(*data())
        self.assertEqual([r["method"] for r in results], ["fixed", "static", "new_retry_candidate", "bounded_selection"])
        self.assertEqual(results[0]["raw_edges"]["ehr_cxr"]["proxy_opposition_facts"], 2)
        self.assertEqual(results[2]["raw_edges"]["ehr_cxr"]["supported_positive"], 2)
        self.assertEqual(results[3]["raw_edges"], results[1]["raw_edges"])

    def test_unavailable_secondary_is_na_not_zero(self):
        rows, selection, endpoint = data()
        for record in endpoint["records"]:
            record.update(biovil_raw_cosine=None, status="not_available", reason="invented_import_failure")
        results = aggregate(rows, selection, endpoint)
        self.assertTrue(all(r["mean_biovil_raw_cosine"] is None and r["biovil_available_cases"] == 0 for r in results))

    def test_partial_endpoint_cannot_make_observed_subset_mean(self):
        rows, selection, endpoint = data()
        endpoint["records"][0].update(biovil_raw_cosine=None, status="not_available", reason="invented_limit")
        result = aggregate(rows, selection, endpoint)[0]
        self.assertEqual(result["biovil_available_cases"], 1)
        self.assertIsNone(result["mean_biovil_raw_cosine"])

    def test_missing_or_duplicate_case_is_not_two_cases(self):
        rows, selection, endpoint = data()
        for choices in ([selection["choices"][0]], [selection["choices"][0]] * 2):
            with self.assertRaises(ValueError): aggregate(rows, {"choices": choices}, endpoint)

    def test_score_lineage_cannot_change(self):
        rows, selection, endpoint = data()
        endpoint["records"][0]["cxr_sha256"] = "0" * 64
        with self.assertRaises(ValueError): aggregate(rows, selection, endpoint)

    def test_empty_direct_ehr_reference_is_not_perfect_agreement(self):
        rows, selection, endpoint = data()
        for row in rows:
            for edge in ("ehr_cxr", "ehr_report"):
                values = row["raw_edge_readouts"][edge]
                for key in ("known_reference_facts", "comparable_facts", "supported_facts", "supported_positive",
                            "supported_negative", "proxy_opposition_facts", "missing_comparisons"):
                    values[key] = 0
        for result in aggregate(rows, selection, endpoint):
            self.assertIsNone(result["raw_edges"]["ehr_cxr"]["support_over_known"])
            self.assertIsNone(result["raw_edges"]["ehr_report"]["coverage_over_known"])


if __name__ == "__main__":
    unittest.main()
