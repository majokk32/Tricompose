"""Pure table/summary fixtures; no authenticated clinical artifact reads."""
from copy import deepcopy
import csv
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import diagnose_current_image_evidence as d
from test_fixed_image_reachability import invented, analyze


def fixtures():
    plan = {"historical_cost": {"invented_calls": 8},
        "cases": [{"historical_ledger": {"charged_model_attempts": 4}} for _ in range(2)]}
    diagnostics = [{"pair_ordinal": i, "case_id": f"invented_case_{i}", "seed": 1,
        "image_guard": {"status": status, "withhold_image_attribution": True}, **analyze(invented())}
        for i, status in enumerate(("unresolved_missing_image_evidence", "unresolved_scorer_disagreement"))]
    audit = {"completed_report_label_pairs": 2, "charged_new_worker_attempts": 4}
    return plan, diagnostics, audit


class EvidenceDiagnosticTests(unittest.TestCase):
    def test_costs_and_no_unsupported_clinical_or_policy_claims(self):
        plan, cases, audit = fixtures(); before = deepcopy((plan, cases, audit))
        summary = d.build_summary(plan, cases, [{} for _ in range(6)], audit)
        self.assertEqual(summary["earlier_report_phase_charged_worker_attempts"], 4)
        self.assertEqual(summary["old_case_worker_attempts"], [4, 4])
        self.assertEqual(summary["new_model_calls"], 0)
        self.assertEqual(summary["zero_report_only_support_ceiling_cases"], 2)
        self.assertFalse(summary["clinical_repair_success"])
        self.assertFalse(summary["installed_as_policy"])
        self.assertIsNone(summary["clinical_fault_location"])
        self.assertIsNone(summary["measured_saved_model_calls"])
        self.assertEqual((plan, cases, audit), before)

    def test_no_known_constraints_are_not_counted_as_failed_triple_support(self):
        plan, cases, audit = fixtures()
        cases[0].update(analyze(invented(ehr="unknown")))
        summary = d.build_summary(plan, cases, [{} for _ in range(6)], audit)
        self.assertEqual(summary["zero_report_only_support_ceiling_cases"], 1)
        self.assertIsNone(cases[0]["bounds"]["maximum_all_three_support_over_known"])

    def test_earlier_phase_cost_and_completeness_mismatch_rejected(self):
        plan, cases, audit = fixtures()
        for field, value in (("charged_new_worker_attempts", 2), ("completed_report_label_pairs", 1)):
            modified = deepcopy(audit); modified[field] = value
            with self.assertRaises(ValueError): d.build_summary(plan, cases, [], modified)

    def test_table_is_deterministic_and_preserves_denominators_and_evidence_status(self):
        _, cases, _ = fixtures(); before = deepcopy(cases)
        images, reports = d.tables(cases, [])
        parsed = list(csv.DictReader(io.StringIO(d.csv_text(images))))
        self.assertEqual([r["ceiling_denominator"] for r in parsed], ["1", "1"])
        self.assertEqual([r["report_only_triple_support_ceiling"] for r in parsed], ["0", "0"])
        self.assertEqual(len(parsed), 2)
        self.assertEqual(reports, [])
        self.assertEqual(cases, before)
        self.assertEqual(d.csv_text(images), d.csv_text(images))
        self.assertIn("not independent clinical adjudication", d.report_text(cases, images))
        with self.assertRaises(ValueError): d.csv_text([images[0], {"extra": 1}])


if __name__ == "__main__": unittest.main()
