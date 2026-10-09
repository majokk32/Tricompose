"""Authored numeric/journal fixtures, without patient data or model inference."""
import copy
import csv
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import audit_fresh_cxr_agent as audit
from test_fresh_output_acceptance import fixture
from test_fresh_cxr_agent_bridge import append_chain, decision, new_ledger
from tricompose_llm.live_cxr_bridge import FreshCXRSession
from tricompose_v12.invariant_verification import _digest


def package(*, action="regenerate_cxr", image_state="positive", known=True, failure=None):
    base, ctx = fixture(image_state="negative", known=known)
    case = {"case_id": base["case_id"], "cached_rows": [base], "reference": base,
        "anchor": ctx["anchor"]}
    ledger = new_ledger(ctx)
    journal = []
    session = FreshCXRSession(case["cached_rows"], base["triple_candidate_id"], ctx,
        ledger.snapshot(), source_manifest_sha256=audit.producer.source.SOURCE_SHA, sink=journal.append)
    state = session.begin_planning()
    step = None
    if state is not None:
        step = {"state": state, "outcome": None, "proposal": None}
        if action == "policy_failure":
            session.policy_failed()
        else:
            proposed_decision = decision(state, action)
            outcome = {"status": "completed", "decision": proposed_decision,
                "audit": {"frozen": True, "scope": "invented_fixture_no_models"}}
            step["outcome"] = outcome
            if session.receive_decision(proposed_decision):
                proposal, _ = fixture(image_state=image_state, image_id="fixture_new_image",
                    report_id="fixture_new_report", seed=1)
                book = append_chain(ledger, proposal, failure=failure)
                step["proposal"] = None if failure else proposal
                session.finish_image(step["proposal"], book)
    selection = session.result()
    digest = _digest(["invented_selection", selection])
    journal.append({"event": "selection_sealed", "sha256": digest})
    return {"case": case, "context": ctx, "final": session.ledger,
        "journal": journal, "step": step, "selection": selection,
        "selection_sha256": digest}


class FreshCXRPostflightTests(unittest.TestCase):
    def test_completed_gain_replays_exactly_without_mutation_or_clinical_claim(self):
        data = package()
        before = copy.deepcopy(data)
        result = audit.replay_case(**data)
        self.assertEqual(data, before)
        self.assertEqual(result["selection"]["accepted_proxy_transitions"], 1)
        self.assertTrue(result["new_image_own_report_chain_pass"])
        self.assertFalse(result["clinical_repair_success"])
        self.assertIsNone(result["clinical_accuracy"])

    def test_veto_replays_and_keeps_reference(self):
        data = package(image_state="negative")
        result = audit.replay_case(**data)
        self.assertEqual(result["selection"]["selected_candidate_id"], data["case"]["reference"]["triple_candidate_id"])
        self.assertEqual(result["selection"]["charged_new_worker_attempts"], 4)
        self.assertEqual(result["selection"]["accepted_proxy_transitions"], 0)

    def test_unknown_does_not_count_as_resolved_opposition(self):
        result = audit.replay_case(**package(image_state="unknown"))
        self.assertEqual(result["selection"]["accepted_proxy_transitions"], 0)
        self.assertEqual(result["selection"]["history"][0]["lost_comparison_count"], 1)

    def test_every_failed_phase_preserves_reserved_cost_and_reference(self):
        for count, kind in enumerate(("cxr_generator", "xrv", "report_generator", "chexbert"), 1):
            data = package(failure=kind)
            result = audit.replay_case(**data)
            self.assertEqual(result["selection"]["charged_new_worker_attempts"], count)
            self.assertEqual(result["selection"]["accepted_proxy_transitions"], 0)
            self.assertFalse(result["new_image_own_report_chain_pass"])

    def test_stop_abstain_and_policy_failure_do_not_invent_new_worker(self):
        for action in ("stop", "abstain", "policy_failure"):
            result = audit.replay_case(**package(action=action))
            self.assertEqual(result["selection"]["charged_new_worker_attempts"], 0)
            self.assertEqual(result["selection"]["charged_policy_requests"], 1)
            self.assertFalse(result["new_image_own_report_chain_pass"])

    def test_underconditioned_case_kept_without_policy_or_worker(self):
        data = package(known=False)
        self.assertIsNone(data["step"])
        result = audit.replay_case(**data)
        self.assertEqual(result["selection"]["charged_policy_requests"], 0)
        self.assertIsNone(result["selected_raw_edges"]["ehr_cxr"]["support_over_known"])
        records = list(csv.DictReader(io.StringIO(audit.comparison_csv([result]))))
        self.assertEqual(records[0]["selected_ehr_cxr_support_over_known"], "NA")

    def test_future_or_changed_numeric_state_is_rejected(self):
        for corruption in ("future", "count"):
            data = package()
            state = data["step"]["state"]
            if corruption == "future":
                state["evidence"].append(copy.deepcopy(state["evidence"][0]))
            else:
                state["evidence"][0]["edges"]["ehr_cxr"]["opposed"] = 0
            with self.assertRaises(ValueError): audit.replay_case(**data)

    def test_changed_policy_decision_seal_or_feedback_is_rejected(self):
        for target in ("outcome", "selection", "journal", "seal"):
            data = package()
            if target == "outcome": data["step"]["outcome"]["decision"]["target_id"] = "t9999"
            elif target == "selection": data["selection"]["accepted_proxy_transitions"] = 0
            elif target == "journal": data["journal"][-2]["history"]["accepted_proxy_transition"] = False
            else: data["selection_sha256"] = _digest("invented_wrong_seal")
            with self.assertRaises(ValueError): audit.replay_case(**data)

    def test_uncharged_or_modified_ledger_is_rejected(self):
        data = package()
        data["final"]["charged_model_attempts"] = 0
        with self.assertRaises(ValueError): audit.replay_case(**data)
        data = package()
        data["final"] = new_ledger(data["context"]).snapshot()
        with self.assertRaises(ValueError): audit.replay_case(**data)

    def test_terminal_policy_cannot_receive_new_output(self):
        data = package(action="stop")
        proposal, _ = fixture(image_id="fixture_new_image", report_id="fixture_new_report", seed=1)
        data["step"]["proposal"] = proposal
        with self.assertRaises(ValueError): audit.replay_case(**data)

    def test_event_after_seal_is_rejected(self):
        data = package()
        data["journal"].append({"event": "invented_after_seal"})
        with self.assertRaises(ValueError): audit.replay_case(**data)

    def test_changed_ehr_cannot_be_hidden_in_fresh_receipt(self):
        data = package()
        data["step"]["proposal"]["ehr_sha256"] = _digest("invented_changed_ehr")
        with self.assertRaises(ValueError): audit.replay_case(**data)

    def summary_fixture(self):
        data = package()
        selections, books = [data["selection"]], [data["final"]]
        fresh = [data["step"]["proposal"]]
        policies = [data["step"]["outcome"]["audit"]]
        plan = {"cases": [data["case"]], "historical_worker_attempts": 20, "historical_policy_requests": 6}
        summary = {"schema_version": audit.producer.VERSION, "status": "completed_one_image_probe_unvalidated",
            "fixed_ehr_cases": 1, "completed_new_triplets": 1, "accepted_proxy_transitions": 1,
            "charged_new_worker_attempts": 4, "charged_policy_requests": 1,
            "historical_worker_attempts": 20, "historical_policy_requests": 6,
            "historical_cost_scope": "shared_sunk_not_measured_not_zero", "external_api_calls": 0,
            "clinical_accuracy": None, "measured_gpu_seconds": None,
            "clinical_repair_success": False, "llm_superiority_demonstrated": False,
            "scorers_or_thresholds_changed": False, "training_performed": False,
            "independent_endpoint_used_for_selection": False, "runtime_seconds_including_load_io": 1.25}
        return plan, summary, selections, books, fresh, policies

    def test_summary_arithmetic_and_phase_inventory(self):
        values = self.summary_fixture()
        counts, phases = audit.validate_summary(*values)
        self.assertEqual(counts["charged_new_worker_attempts"], 4)
        self.assertEqual(phases, {"cxr_generator": 1, "xrv": 1, "report_generator": 1, "chexbert": 1})

    def test_forged_counts_or_clinical_claims_are_rejected(self):
        for field, value in (("charged_new_worker_attempts", 0), ("external_api_calls", False),
            ("historical_worker_attempts", 0), ("clinical_accuracy", 1.0),
            ("llm_superiority_demonstrated", True), ("scorers_or_thresholds_changed", True),
            ("independent_endpoint_used_for_selection", True)):
            values = self.summary_fixture()
            values[1][field] = value
            with self.assertRaises(ValueError): audit.validate_summary(*values)

    def test_invalid_runtime_measurement_is_rejected(self):
        for value in (float("inf"), float("nan"), -1.0, True):
            values = self.summary_fixture(); values[1]["runtime_seconds_including_load_io"] = value
            with self.assertRaises(ValueError): audit.validate_summary(*values)

    def test_cpu_guard_precedes_any_metadata_reader_or_model_use(self):
        with patch.object(audit.producer.gate, "cpu_guard", side_effect=RuntimeError("no_cpu_allocation")), \
             patch.object(audit.postflight, "MetadataReader") as read:
            with self.assertRaises(RuntimeError): audit.audit_run(None)
            read.assert_not_called()
