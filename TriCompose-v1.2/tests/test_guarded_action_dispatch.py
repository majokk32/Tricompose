"""Authored callbacks only; never instantiate a GPU model or touch artifacts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from tricompose_llm.contracts import ContractError
from tricompose_llm.guarded_action_dispatch import GuardedActionDispatcher
from test_image_evidence_guard import fixture, proposal


class GuardedActionDispatchTests(unittest.TestCase):
    def test_disagreement_blocks_factory_and_backend_at_execution_boundary(self):
        state, guard = fixture(); events = []
        factory = Mock(side_effect=AssertionError("invented_must_not_construct"))
        dispatcher = GuardedActionDispatcher(sink=events.append)
        result, payload = dispatcher.dispatch(proposal(state), state, guard, backend_factory=factory)
        factory.assert_not_called()
        self.assertEqual(result["status"], "guard_abstained_unverified")
        self.assertEqual(result["backend_dispatch_attempts"], 0)
        self.assertEqual(result["actual_worker_model_calls"], 0)
        self.assertIsNone(payload)
        self.assertEqual([e["event"] for e in events], ["execution_guard_applied", "guarded_dispatch_sealed"])

    def test_missing_unavailable_and_no_ehr_constraints_never_launch_image_backend(self):
        for arguments in ({"observer": "unknown"}, {"observer": "uncertain"},
                          {"observer": None}, {"ehr": "unknown"}):
            with self.subTest(arguments=arguments):
                state, guard = fixture(**arguments); factory = Mock()
                result, _ = GuardedActionDispatcher(sink=lambda e: None).dispatch(
                    proposal(state), state, guard, backend_factory=factory)
                factory.assert_not_called()
                self.assertTrue(result["decision_withheld"])

    def test_agreement_without_approved_backend_is_deferred_not_inference(self):
        state, guard = fixture(observer="negative")
        result, _ = GuardedActionDispatcher(sink=lambda e: None).dispatch(proposal(state), state, guard)
        self.assertEqual(result["status"], "deferred_separately_approved_backend_required")
        self.assertEqual(result["actual_worker_model_calls"], 0)
        self.assertFalse(result["clinical_acceptance"])

    def test_registered_report_action_is_not_vetoed_and_reservation_precedes_factory(self):
        state, guard = fixture(); events = []
        def factory():
            self.assertEqual(events[-1]["event"], "backend_dispatch_reserved")
            return lambda decision: ("invented_private_payload", decision["action"])
        result, payload = GuardedActionDispatcher(sink=events.append).dispatch(
            proposal(state, "regenerate_report"), state, guard, backend_factory=factory)
        self.assertFalse(result["decision_withheld"])
        self.assertEqual(result["backend_dispatch_attempts"], 1)
        self.assertIsNone(result["actual_worker_model_calls"])
        self.assertEqual(payload, ("invented_private_payload", "regenerate_report"))
        self.assertNotIn("invented_private_payload", str(events))
        self.assertFalse(result["clinical_acceptance"])
        self.assertTrue(result["original_acceptance_gate_must_still_run"])

    def test_factory_or_backend_failure_stays_reserved_no_retry(self):
        state, guard = fixture(observer="negative")
        factories = [Mock(side_effect=RuntimeError("invented_failure")),
                     Mock(return_value=Mock(side_effect=RuntimeError("invented_backend_failure"))),
                     Mock(return_value=None)]
        for factory in factories:
            events = []; dispatcher = GuardedActionDispatcher(sink=events.append)
            result, _ = dispatcher.dispatch(proposal(state), state, guard, backend_factory=factory)
            factory.assert_called_once()
            self.assertEqual(result["status"], "backend_failed_attempt_retained_no_retry")
            self.assertEqual(result["backend_dispatch_attempts"], 1)
            self.assertFalse(result["automatic_retry_allowed"])
            self.assertEqual(sum(e["event"] == "backend_dispatch_reserved" for e in events), 1)
            with self.assertRaises(ContractError):
                dispatcher.dispatch(proposal(state), state, guard, backend_factory=factory)

    def test_sink_failure_prevents_any_backend_construction(self):
        state, guard = fixture(observer="negative"); factory = Mock()
        events = []
        def sink(event):
            if event["event"] == "backend_dispatch_reserved":
                raise OSError("invented_journal_failure")
            events.append(event)
        dispatcher = GuardedActionDispatcher(sink=sink)
        with self.assertRaises(OSError):
            dispatcher.dispatch(proposal(state), state, guard, backend_factory=factory)
        factory.assert_not_called()
        with self.assertRaises(ContractError):
            dispatcher.dispatch(proposal(state), state, guard, backend_factory=factory)

    def test_invalid_decision_does_not_launch_or_turn_into_successful_abstention(self):
        state, guard = fixture(); factory = Mock(); events = []
        invalid = proposal(state); invalid["extra_text"] = "invented_invalid"
        with self.assertRaises(ContractError):
            GuardedActionDispatcher(sink=events.append).dispatch(invalid, state, guard, backend_factory=factory)
        factory.assert_not_called(); self.assertEqual(events, [])

    def test_inputs_are_immutable_and_terminal_step_is_not_replayed(self):
        state, guard = fixture(); decision = proposal(state)
        before = deepcopy((state, guard, decision))
        dispatcher = GuardedActionDispatcher(sink=lambda e: None)
        dispatcher.dispatch(decision, state, guard)
        self.assertEqual(before, (state, guard, decision))
        with self.assertRaises(ContractError): dispatcher.dispatch(decision, state, guard)


if __name__ == "__main__":
    unittest.main()
