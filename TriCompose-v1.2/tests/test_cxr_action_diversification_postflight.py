"""Invented count/journal metadata only; no models, source bodies or pixels."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import audit_cxr_action_diversification as a
from test_cxr_action_diversification import ANCHOR, book


def fixture(failed=False):
    t1, b1 = book(failed_slot=0 if failed else None, failed_stage=1 if failed else None)
    t2, b2 = book()
    slots = a.worker.slot_outcomes(ANCHOR, t1, b1) + a.worker.slot_outcomes(ANCHOR, t2, b2)
    triples, books = t1 + t2, [b1, b2]
    summary = {"schema_version": a.worker.VERSION,
        "status": "partial_candidate_expansion_unvalidated" if failed else "completed_candidate_expansion_unvalidated",
        "fixed_ehr_cases": 2, "planned_image_slots": 4, "completed_triplets": len(triples),
        "charged_worker_requests": sum(b["charged_model_attempts"] for b in books),
        "validated_worker_calls": sum(b["completed_operations"] for b in books),
        "failed_worker_requests": sum(b["failed_attempts"] for b in books),
        "slot_status_counts": dict(Counter(s["status"] for s in slots)), "retries": 0,
        "new_planner_calls": 0, "external_api_calls": 0, "new_clinically_accepted_repairs": 0,
        "training_performed": False, "clinical_acceptance": False, "selection_changed": False,
        "independent_endpoint_evaluated": False, "clinical_accuracy": None,
        "measured_saved_model_calls": None, "same_call_budget_not_same_gpu_seconds": True,
        "historical_cost_is_shared_sunk_not_free": True, "runtime_seconds_includes_startup_and_io": 2.5}
    return summary, triples, slots, books


class CXRActionPostflightTests(unittest.TestCase):
    def test_full_four_slots_and_sixteen_charges(self):
        payload = fixture(); before = deepcopy(payload)
        counts = a.validate_totals(*payload)
        self.assertEqual(counts["completed_triplets"], 4)
        self.assertEqual(counts["charged_worker_requests"], 16)
        self.assertEqual(payload, before)

    def test_partial_retains_four_slots_and_failed_charge(self):
        payload = fixture(failed=True)
        counts = a.validate_totals(*payload)
        self.assertEqual(counts["completed_triplets"], 3)
        self.assertEqual(counts["charged_worker_requests"], 14)
        self.assertEqual(counts["validated_worker_calls"], 13)
        self.assertEqual(counts["failed_worker_requests"], 1)
        self.assertEqual(len(payload[2]), 4)

    def test_dropped_unavailable_slot_rejected(self):
        summary, triples, slots, books = fixture(failed=True)
        with self.assertRaises(ValueError): a.validate_totals(summary, triples, slots[1:], books)

    def test_hidden_or_refunded_charge_rejected(self):
        for key in ("charged_worker_requests", "validated_worker_calls", "failed_worker_requests", "completed_triplets"):
            payload = fixture(); payload[0][key] += 1
            with self.assertRaises(ValueError): a.validate_totals(*payload)
        payload = fixture(); payload[2][0]["charged_worker_requests"] -= 1
        with self.assertRaises(ValueError): a.validate_totals(*payload)

    def test_boolean_counts_rejected(self):
        payload = fixture(); payload[0]["new_planner_calls"] = False
        with self.assertRaises(ValueError): a.validate_totals(*payload)

    def test_clinical_or_cost_claim_rejected(self):
        for key, value in (("clinical_accuracy", 1), ("measured_saved_model_calls", 4),
            ("clinical_acceptance", True), ("selection_changed", True), ("new_clinically_accepted_repairs", 1),
            ("training_performed", True), ("same_call_budget_not_same_gpu_seconds", False)):
            payload = fixture(); payload[0][key] = value
            with self.assertRaises(ValueError): a.validate_totals(*payload)

    def test_partial_cannot_claim_complete(self):
        payload = fixture(failed=True); payload[0]["status"] = "completed_candidate_expansion_unvalidated"
        with self.assertRaises(ValueError): a.validate_totals(*payload)

    def test_nonfinite_or_negative_or_boolean_runtime_rejected(self):
        for value in (float("nan"), float("inf"), -1, True):
            payload = fixture(); payload[0]["runtime_seconds_includes_startup_and_io"] = value
            with self.assertRaises(ValueError): a.validate_totals(*payload)

    def test_slot_status_inventory_drift_rejected(self):
        payload = fixture(failed=True); payload[0]["slot_status_counts"] = {"completed_unvalidated": 4}
        with self.assertRaises(ValueError): a.validate_totals(*payload)

    def test_cpu_guard_before_all_source_reads(self):
        with patch.object(a.worker.existing.gate, "cpu_guard", side_effect=RuntimeError("blocked")), \
                patch.object(a.worker, "load_plan") as load:
            with self.assertRaises(RuntimeError): a.audit(SimpleNamespace())
            load.assert_not_called()


if __name__ == "__main__": unittest.main()
