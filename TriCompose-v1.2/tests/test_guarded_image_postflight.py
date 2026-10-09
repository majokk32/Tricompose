"""Invented metadata only: replay and reject fabricated guard/cost claims."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import audit_guarded_image_probe as audit
from test_guarded_image_probe import invented


def package(mode="disagreement"):
    generation, cases, records = invented()
    images = audit.worker.two_reference_images(generation)
    index = {r["cxr_candidate_id"]: r for r in records}
    records = [index[i["cxr_candidate_id"]] for i in images]
    if mode == "deferred":
        for record in records: record["states"]["edema"] = "negative"
    elif mode == "unavailable":
        records[0].update(contract_status="failed_unavailable", states=None)
    plan = {"schema_version": audit.worker.VERSION, "config": deepcopy(audit.worker.CONFIG),
        "image_inputs": images, "selection_change_allowed": False}
    predictions = {"schema_version": audit.worker.VERSION, "image_only": True,
        "frozen": True, "model_received_ehr_reports_ids_or_scores": False, "records": records}
    observer = [{"event": "observer_load_attempt_reserved", "charged_load_attempts": 1}]
    for ordinal, record in enumerate(records):
        observer.extend(({"event": "call_reserved", "ordinal": ordinal, "cxr_sha256": record["cxr_sha256"]},
            {"event": "call_finished", "ordinal": ordinal, "contract_status": record["contract_status"]}))
    digest = "8" * 64
    journal = [{"event": "fresh_observer_predictions_sealed", "sha256": digest}]
    results = audit.worker.dispatch_cases(cases, records, journal.append)
    dispatch = {"schema_version": audit.worker.VERSION, "records": results}
    pending = {"schema_version": audit.worker.VERSION,
        "records": [r for r in results if r["dispatch"]["status"] == "deferred_separately_approved_backend_required"],
        "automatic_submission_allowed": False, "requires_complete_script_and_explicit_approval": True}
    summary = {"schema_version": audit.worker.VERSION, "status": audit.STATUS, "fixed_development_cases": 2,
        "fresh_observer_calls": 2, "charged_load_attempts": 1,
        "complete_responses": sum(r["contract_status"] == "complete" for r in records),
        "guard_status_counts": dict(Counter(r["guard"]["status"] for r in results)),
        "dispatch_status_counts": dict(Counter(r["dispatch"]["status"] for r in results)),
        "new_numeric_planner_calls": 0, "new_generator_or_primary_scorer_calls": 0,
        "new_backend_dispatch_attempts": 0, "numeric_intents_are_cached": True,
        "historical_cost_not_erased": True, "new_model_retries": 0,
        "clinical_acceptance": False, "clinical_accuracy": None, "measured_saved_model_calls": None,
        "historical_selection_changed": False, "llm_superiority_demonstrated": False,
        "training_performed": False, "runtime_seconds": 1.0, "peak_torch_allocated_vram_gib": 1.0}
    return plan, cases, predictions, observer, dispatch, pending, journal, summary, digest


class GuardedImagePostflightTests(unittest.TestCase):
    def test_replay_preserves_inputs_and_does_not_claim_clinical_accuracy(self):
        data = package(); before = deepcopy(data)
        result = audit.verify_payload(*data)
        self.assertEqual(data, before)
        self.assertTrue(result["execution_order_replay_pass"])
        self.assertEqual(result["fresh_observer_calls"], 2)
        self.assertEqual(result["new_backend_dispatch_attempts"], 0)
        self.assertIsNone(result["clinical_accuracy"])

    def test_unblocked_intents_are_pending_not_generator_calls(self):
        result = audit.verify_payload(*package("deferred"))
        self.assertEqual(result["pending_separately_approved_requests"], 2)
        self.assertEqual(result["new_generator_or_primary_scorer_calls"], 0)
        self.assertFalse(result["clinical_acceptance"])

    def test_failed_observer_remains_charged_and_unavailable(self):
        result = audit.verify_payload(*package("unavailable"))
        self.assertEqual(result["complete_responses"], 1)
        self.assertEqual(result["fresh_observer_calls"], 2)
        self.assertEqual(result["pending_separately_approved_requests"], 0)

    def test_missing_reservation_duplicate_finish_or_reordering_rejected(self):
        for kind in ("missing", "duplicate", "order"):
            data = list(package())
            if kind == "missing": data[3].pop(1)
            elif kind == "duplicate": data[3].append(deepcopy(data[3][-1]))
            else: data[3][1], data[3][2] = data[3][2], data[3][1]
            with self.subTest(kind=kind), self.assertRaises(ValueError): audit.verify_payload(*data)

    def test_prediction_sealing_must_precede_dispatch(self):
        data = list(package()); data[6][0], data[6][1] = data[6][1], data[6][0]
        with self.assertRaises(ValueError): audit.verify_payload(*data)

    def test_hash_or_image_identity_changes_rejected(self):
        for kind in ("image", "seal"):
            data = list(package())
            if kind == "image": data[2]["records"][0]["cxr_sha256"] = "4" * 64
            else: data[-1] = "4" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError): audit.verify_payload(*data)

    def test_fake_calls_savings_acceptance_and_changed_winners_rejected(self):
        for field, value in (("fresh_observer_calls", 0), ("new_numeric_planner_calls", 2),
                ("new_generator_or_primary_scorer_calls", 1), ("measured_saved_model_calls", 2),
                ("clinical_acceptance", True), ("historical_selection_changed", True),
                ("llm_superiority_demonstrated", True), ("numeric_intents_are_cached", False),
                ("runtime_seconds", float("nan")), ("peak_torch_allocated_vram_gib", -1)):
            data = list(package()); data[7][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): audit.verify_payload(*data)

    def test_deferred_request_authorization_or_result_fabrication_rejected(self):
        for kind in ("submit", "pending", "dispatch"):
            data = list(package())
            if kind == "submit": data[5]["automatic_submission_allowed"] = True
            elif kind == "pending": data[5]["records"] = data[4]["records"]
            else: data[4]["records"][0]["retained_reference_changed"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): audit.verify_payload(*data)

    def test_cpu_guard_precedes_metadata_reads(self):
        with patch.object(audit.worker.cached.observer.source.previous.gate, "cpu_guard",
                side_effect=RuntimeError("invented_login")), \
                patch.object(audit.worker.cached.observer.source.previous.postflight, "MetadataReader") as reader:
            with self.assertRaises(RuntimeError): audit.audit(None)
            reader.assert_not_called()


if __name__ == "__main__": unittest.main()
