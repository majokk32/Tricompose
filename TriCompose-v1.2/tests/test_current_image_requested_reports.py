"""Invented numeric requests and fake frozen backends; no model/body reads."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_current_image_requested_reports as worker
from test_current_image_policy_audit import ingredients as policy_ingredients
from test_guarded_requested_reports import case_fixture_bound, request
from tricompose_v12.execution_ledger import CallResult
from tricompose_v12.invariant_verification import _digest


def requests():
    plan, results, pending, *_ = policy_ingredients()
    return plan, results, pending


def bound_case():
    plan, results, pending = requests()
    shared = worker.shared_requests(plan, results, pending)
    return worker.bind_case(case_fixture_bound(), plan["cases"][0], shared[0])


def backend(req, callback=lambda op: {}):
    return SimpleNamespace(frozen=True, audit_sha256=req.frozen_model_audit_sha256,
        execution_mode="approved_slurm_backend", invoke=callback)


class CurrentImageRequestedReportTests(unittest.TestCase):
    def test_four_provenance_rows_are_exactly_two_physical_requests(self):
        values = requests(); before = deepcopy(values)
        shared = worker.shared_requests(*values)
        self.assertEqual(values, before)
        self.assertEqual(len(shared), 2)
        self.assertEqual(sum(r["new_physical_report_attempts"] for r in shared), 2)
        self.assertEqual(sum(r["new_physical_chexbert_attempts"] for r in shared), 2)
        self.assertTrue(all(r["policies"] == ["qwen", "rule"] for r in shared))
        self.assertEqual(sum(r["qwen_policy_calls_already_incurred"] for r in shared), 2)
        self.assertEqual(sum(r["rule_model_calls_already_incurred"] for r in shared), 0)

    def test_disagreement_or_failed_planner_requires_a_different_reviewed_plan(self):
        for kind in ("disagreement", "failure", "replay", "case", "worker"):
            plan, results, pending = requests()
            if kind == "disagreement": results[0]["same_effective_action_and_target_as_rule"] = False
            elif kind == "failure": results[0]["decision"] = None
            elif kind == "replay": results[0]["dispatch"]["actual_worker_model_calls"] = 1
            elif kind == "case": results[0]["case_id"] = "case_987"
            else: results[0]["decision"]["target_id"] = "t0103"
            with self.subTest(kind=kind), self.assertRaises(ValueError): worker.shared_requests(plan, results, pending)

    def test_pending_rows_cannot_be_swapped_duplicated_or_autoauthorized(self):
        for kind in ("model", "hash", "seed", "duplicate", "auto", "count"):
            plan, results, pending = requests()
            if kind == "model": pending["records"][0]["request"] = {**pending["records"][0]["request"], "model_id": "maira2"}
            elif kind == "hash": pending["records"][0]["request"] = {**pending["records"][0]["request"], "input_image_sha256": _digest("wrong")}
            elif kind == "seed": pending["records"][0]["request"] = {**pending["records"][0]["request"], "seed": 0}
            elif kind == "duplicate": pending["records"][0]["policy"] = "rule"
            elif kind == "auto": pending["automatic_submission_allowed"] = True
            else: pending["records"].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError): worker.shared_requests(plan, results, pending)

    def test_new_binding_preserves_old_exhausted_ledger_and_baseline(self):
        old = case_fixture_bound(); original = deepcopy(old)
        plan, results, pending = requests()
        shared = worker.shared_requests(plan, results, pending)[0]
        case = worker.bind_case(old, plan["cases"][0], shared)
        self.assertEqual(old, original)
        self.assertEqual(case["baseline_row"], old["baseline_row"])
        self.assertEqual(case["historical_ledger"], old["historical_ledger"])
        self.assertEqual(case["historical_ledger"]["charged_model_attempts"], 4)
        self.assertEqual(case["requested_model"], "llavarad")
        self.assertFalse(case["original_selection_change_allowed"])

    def test_wrong_image_ehr_seed_or_retained_branch_rejected(self):
        for kind in ("image", "ehr", "seed", "branch"):
            plan, results, pending = requests()
            shared = worker.shared_requests(plan, results, pending)[0]
            planned = plan["cases"][0]
            if kind == "image": planned["trial_image_sha256"] = _digest("wrong")
            elif kind == "ehr": planned["ehr_sha256"] = _digest("wrong")
            elif kind == "seed": shared["request"]["seed"] = 0
            else: planned["retained_branch_candidate_id"] = "invented_wrong_retained"
            with self.subTest(kind=kind), self.assertRaises(ValueError): worker.bind_case(case_fixture_bound(), planned, shared)

    def test_shared_worker_still_reserves_once_before_lazy_backend_construction(self):
        case = bound_case(); events = []
        calls = worker.previous.RequestedReportCalls(case, sink=events.append)
        r = request(case)
        def factory():
            self.assertEqual(events[-1]["event"], "new_worker_reserved")
            return backend(r)
        first = calls.invoke(r, factory=factory, validator=lambda *args: CallResult(_digest("new_report")))
        c = request(case, "chexbert", report_hash=first.output_artifact_sha256)
        calls.invoke(c, factory=lambda: backend(c),
            validator=lambda *args: CallResult(_digest("new_labels"), _digest("new_receipt")))
        self.assertEqual(calls.snapshot()["charged_new_worker_attempts"], 2)
        self.assertEqual(case["historical_ledger"]["charged_model_attempts"], 4)
        self.assertEqual(len([e for e in events if e["event"] == "new_worker_reserved"]), 2)

    def test_failed_new_report_is_charged_without_retry_or_second_policy_execution(self):
        case = bound_case(); calls = worker.previous.RequestedReportCalls(case, sink=lambda event: None)
        r = request(case)
        def fail(): raise TimeoutError("invented_failure")
        self.assertIsNone(calls.invoke(r, factory=fail, validator=lambda *args: None))
        self.assertEqual(calls.snapshot()["charged_new_worker_attempts"], 1)
        self.assertEqual(calls.snapshot()["failed_new_worker_attempts"], 1)
        again = Mock()
        with self.assertRaises(ValueError): calls.invoke(r, factory=again, validator=lambda *args: None)
        again.assert_not_called()

    def test_new_phase_journal_failure_never_starts_model(self):
        case = bound_case()
        def fail(event): raise OSError("invented_durable_sink_failure")
        calls = worker.previous.RequestedReportCalls(case, sink=fail)
        factory = Mock()
        with self.assertRaises(OSError): calls.invoke(request(case), factory=factory, validator=lambda *args: None)
        factory.assert_not_called()


if __name__ == "__main__": unittest.main()
