"""Invented numeric records and call journals; no model or artifact body reads."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import audit_current_image_reports as auditor
from test_current_image_requested_reports import bound_case
from test_guarded_requested_reports import ingredients
from test_fresh_output_acceptance import fixture


def data(*, fail_at=None):
    w = auditor.worker
    case = bound_case()
    case["partial_receipt"] = {"receipt_id": case["baseline_row"]["receipt"]["partial_receipt_id"]}
    workers = deepcopy(ingredients()[3]["workers"])
    workers["llavarad"] = {"status": "invented_frozen_fixture"}
    row, _ = fixture(image_state="negative", report_state="unknown", seed=1,
        image_id="invented_trial_image", report_id="invented_shared_report", model="llavarad")
    pair = w.previous.assess_pair(case, row, workers)
    anchor = w.previous.gate.anchor_from_record(case["anchor"])
    journal = [{"event": "shared_dependencies_authenticated", "source_policy_manifest_sha256": w.POLICY_RUN_SHA,
        "source_ledger_sha256": case["historical_ledger_sha256"],
        "cached_partial_receipt_id": case["partial_receipt"]["receipt_id"], "new_budget": 2,
        "policies": ["qwen", "rule"], "physical_report_attempts_reserved_by_plan": 1,
        "distinct_policy_outcomes_claimed": False, "clinical_acceptance": False}]
    charged = completed = failed = 0
    report_hash = None
    for ordinal in range(2):
        kind, model = ("report_generator", "llavarad") if ordinal == 0 else ("chexbert", "chexbert")
        request = w.CallRequest("requested_report" if ordinal == 0 else "requested_chexbert",
            anchor.case_id, anchor.sha256, kind, model, w._digest(workers[model]), 1,
            case["cached_xrv_operation_id"] if ordinal == 0 else "requested_report",
            case["baseline_row"]["cxr_sha256"], report_hash if ordinal else None)
        journal.append({"event": "new_worker_reserved", "ordinal": ordinal, "request": request.record(),
            "charged_worker_attempts": 1, "dependency_origin": "authenticated_cached_xrv" if ordinal == 0 else "new_validated_report",
            "historical_ledger_changed": False})
        charged += 1
        if fail_at == ordinal:
            journal.append({"event": "new_worker_failed_charged", "ordinal": ordinal, "error_code": "timeout",
                "elapsed_seconds": .1, "automatic_retry": False, "failure_is_clinical_contradiction": False})
            failed += 1
            row = None
            pair = {"case_id": case["case_id"], "status": "failed_charged_no_retry",
                "branch_selected_candidate_id": case["baseline_row"]["triple_candidate_id"],
                "original_winner_changed": False, "clinical_acceptance": False}
            break
        result = w.CallResult(row["report_sha256"] if ordinal == 0 else row["receipt"]["chexbert_labels_sha256"],
            None if ordinal == 0 else row["receipt"]["receipt_id"])
        journal.append({"event": "new_worker_completed", "ordinal": ordinal, "result": result.record(),
            "elapsed_seconds": .1, "clinical_acceptance": False})
        if ordinal == 0: report_hash = result.output_artifact_sha256
        completed += 1
    snapshot = {"charged_new_worker_attempts": charged, "completed_new_worker_operations": completed,
        "failed_new_worker_attempts": failed, "pending_new_worker_attempts": 0,
        "new_worker_budget": 2, "automatic_retry": False, "historical_ledger_changed": False}
    journal.append({"event": "new_phase_sealed", **snapshot, "clinical_acceptance": False})
    return case, workers, journal, snapshot, row, pair


class SharedReportAuditTests(unittest.TestCase):
    def test_successful_exact_replay_without_winner_promotion(self):
        values = data(); before = deepcopy(values)
        result = auditor.replay_case(*values)
        self.assertTrue(result["exact_phase_replay_pass"])
        self.assertEqual(result["charged_new_worker_attempts"], 2)
        self.assertEqual(result["journal_events"], 6)
        self.assertFalse(result["distinct_policy_outcomes_claimed"])
        self.assertEqual(values, before)

    def test_failed_report_or_verifier_is_charged_not_clinical_contradiction(self):
        for phase in (0, 1):
            result = auditor.replay_case(*data(fail_at=phase))
            self.assertEqual(result["charged_new_worker_attempts"], phase + 1)
            self.assertEqual(result["failed_new_worker_attempts"], 1)
            self.assertIsNone(result["requested_scores"])

    def test_tampered_source_policy_worker_image_parent_and_seed_rejected(self):
        for kind in ("policy", "model", "image", "parent", "seed", "audit"):
            values = list(data()); event = values[2][1]
            if kind == "policy": values[2][0]["source_policy_manifest_sha256"] = "0" * 64
            elif kind == "model": event["request"]["model_id"] = "maira2"
            elif kind == "image": event["request"]["input_image_sha256"] = "0" * 64
            elif kind == "parent": event["request"]["parent_operation_id"] = "invented_unreserved"
            elif kind == "seed": event["request"]["seed"] = 0
            else: event["request"]["frozen_model_audit_sha256"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError): auditor.replay_case(*values)

    def test_report_and_verifier_receipt_hash_binding_rejected_when_tampered(self):
        for kind in ("report", "labels", "receipt"):
            values = list(data())
            if kind == "report": values[2][2]["result"]["output_artifact_sha256"] = "0" * 64
            elif kind == "labels": values[2][4]["result"]["output_artifact_sha256"] = "0" * 64
            else: values[2][4]["result"]["verification_receipt_id"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError): auditor.replay_case(*values)

    def test_refunded_failed_attempt_or_fake_retry_is_rejected(self):
        for kind in ("refund", "retry", "clinical"):
            values = list(data(fail_at=0))
            if kind == "refund": values[3]["charged_new_worker_attempts"] = 0
            elif kind == "retry": values[2][2]["automatic_retry"] = True
            else: values[2][2]["failure_is_clinical_contradiction"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): auditor.replay_case(*values)

    def test_gate_and_cost_replay_cannot_be_overridden(self):
        for kind in ("gate", "budget", "old", "cost", "nan", "journal"):
            values = list(data())
            if kind == "gate": values[5]["comparison"]["exploratory_gate_pass"] = True
            elif kind == "budget": values[3]["new_worker_budget"] = 4
            elif kind == "old": values[0]["historical_ledger"]["call_budget"] = 8
            elif kind == "cost": values[3]["charged_new_worker_attempts"] = 4
            elif kind == "nan": values[2][2]["elapsed_seconds"] = float("nan")
            else: values[2].append(deepcopy(values[2][-1]))
            with self.subTest(kind=kind), self.assertRaises(ValueError): auditor.replay_case(*values)


if __name__ == "__main__": unittest.main()
