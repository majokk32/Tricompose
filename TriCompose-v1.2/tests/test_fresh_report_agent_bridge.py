"""Invented fresh receipts and ledger calls only; no body/model/GPU loading."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from test_fresh_output_acceptance import fixture, reseal
from tricompose_llm.contracts import ContractError, DECISION_SCHEMA, validate_public_state
from tricompose_llm.live_report_bridge import FreshReportSession
from tricompose_v12.execution_ledger import BoundedCallLedger, CallRequest, CallResult
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_plan import anchor_from_record


def base_ledger(row, ctx):
    anchor = anchor_from_record(ctx["anchor"])
    ledger = BoundedCallLedger(case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
        call_budget=10, max_retries=0, execution_mode="invented_fixture_no_models", sink=lambda e: None)
    parent = None
    for kind in ("cxr_generator", "xrv", "report_generator", "chexbert"):
        model = row["cxr_model_id"] if kind == "cxr_generator" else row["report_model_id"] if kind == "report_generator" else kind
        request = CallRequest("baseline_" + kind, anchor.case_id, anchor.sha256, kind, model,
            _digest(["fixture_audit", model]), row["seed"], parent,
            None if kind == "cxr_generator" else row["cxr_sha256"],
            row["report_sha256"] if kind == "chexbert" else None)
        token = ledger.reserve(request)
        artifact = {"cxr_generator": row["cxr_sha256"], "xrv": row["receipt"]["xrv_labels_sha256"],
            "report_generator": row["report_sha256"], "chexbert": row["receipt"]["chexbert_labels_sha256"]}[kind]
        receipt = row["receipt"]["partial_receipt_id"] if kind == "xrv" else row["receipt"]["receipt_id"] if kind == "chexbert" else None
        ledger.complete(token, CallResult(artifact, receipt), elapsed_seconds=.1)
        parent = request.operation_id
    return ledger


def append_report(ledger, row, *, suffix="1", fail=None):
    request = CallRequest("fixture_new_report_" + suffix, ledger.case_id, ledger.anchor,
        "report_generator", row["report_model_id"], _digest(["fixture_audit", row["report_model_id"]]),
        row["seed"], "baseline_xrv", row["cxr_sha256"])
    token = ledger.reserve(request)
    if fail == "report":
        ledger.fail(token, error_code="runtime_exception", retryable=False, elapsed_seconds=.1)
        return ledger.snapshot()
    ledger.complete(token, CallResult(row["report_sha256"]), elapsed_seconds=.1)
    label = CallRequest("fixture_new_labels_" + suffix, ledger.case_id, ledger.anchor,
        "chexbert", "chexbert", _digest("fixture_label_audit"), row["seed"], request.operation_id,
        row["cxr_sha256"], row["report_sha256"])
    token = ledger.reserve(label)
    if fail == "labels":
        ledger.fail(token, error_code="timeout", retryable=True, elapsed_seconds=.1)
    else:
        ledger.complete(token, CallResult(row["receipt"]["chexbert_labels_sha256"], row["receipt"]["receipt_id"]), elapsed_seconds=.1)
    return ledger.snapshot()


def decision(state, *, action="regenerate_report", index=0):
    return {"schema_version": DECISION_SCHEMA, "step_id": state["step_id"], "action": action,
        "target_id": state["tools"][index]["tool_id"] if action == "regenerate_report" else None,
        "evidence_ids": [state["evidence"][0]["evidence_id"]], "reason_code": "explore_alternative"}


class FreshReportBridgeTests(unittest.TestCase):
    def setup_session(self, known=True):
        base, ctx = fixture(known=known)
        ledger = base_ledger(base, ctx)
        events = []
        session = FreshReportSession(base, ctx, ledger.snapshot(), sink=events.append)
        return base, ctx, ledger, session, events

    def acquire(self, session, ledger, row, *, suffix="1", fail=None):
        state = session.begin_planning()
        model = session.receive_decision(decision(state))
        self.assertEqual(model, row["report_model_id"])
        book = append_report(ledger, row, suffix=suffix, fail=fail)
        return session.finish_report(None if fail else row, book)

    def test_numeric_only_state_and_report_only_tool_menu(self):
        base, ctx, ledger, session, events = self.setup_session()
        state = validate_public_state(session.begin_planning())
        self.assertEqual(state["budget"]["spent_units"], 5)
        self.assertEqual(state["budget"]["limit_units"], 13)
        self.assertEqual(len(state["tools"]), 3)
        self.assertEqual({t["action"] for t in state["tools"]}, {"regenerate_report"})
        self.assertEqual(events[0]["event"], "policy_reserved")
        rendered = str(state)
        for forbidden in (base["case_id"], "maira2", "edema", "path", base["ehr_sha256"]):
            self.assertNotIn(forbidden, rendered)

    def test_unknown_ehr_stays_unknown_and_known_count_zero(self):
        _, _, _, session, _ = self.setup_session(known=False)
        row = session.begin_planning()["evidence"][0]
        self.assertEqual(row["edges"]["ehr_cxr"]["known"], 0)
        self.assertEqual(row["edges"]["ehr_report"]["known"], 0)
        self.assertEqual(row["uncertainty"]["ehr_unknown"], 14)

    def test_fresh_strict_gain_selected_without_clinical_claim(self):
        _, _, ledger, session, _ = self.setup_session()
        alt, _ = fixture(report_state="positive", report_id="fixture_alt", model="maira2")
        self.assertTrue(self.acquire(session, ledger, alt))
        self.assertEqual(session.current["triple_candidate_id"], alt["triple_candidate_id"])
        state = session.begin_planning()
        session.receive_decision(decision(state, action="stop"))
        result = session.result()
        self.assertEqual(result["charged_worker_attempts"], 6)
        self.assertEqual(result["charged_policy_requests"], 2)
        self.assertEqual(result["accepted_proxy_transitions"], 1)
        self.assertFalse(result["clinical_repair_success"])
        self.assertFalse(result["image_regeneration_installed"])

    def test_cosmetic_change_is_charged_but_rejected(self):
        base, _, ledger, session, _ = self.setup_session()
        alt, _ = fixture(report_id="fixture_alt", model="maira2")
        self.assertFalse(self.acquire(session, ledger, alt))
        self.assertEqual(session.current, base)
        self.assertEqual(session.ledger["charged_model_attempts"], 6)

    def test_generic_or_temporal_risk_cannot_improve_by_labels_alone(self):
        for flag in ("generic_report", "unsupported_temporal_comparison_language"):
            _, _, ledger, session, _ = self.setup_session()
            alt, _ = fixture(report_state="positive", report_id="fixture_alt", model="maira2")
            alt["structure"][flag] = True
            self.assertFalse(self.acquire(session, ledger, alt))

    def test_conflict_silencing_is_not_repair(self):
        base, ctx = fixture(report_state="negative")
        ledger = base_ledger(base, ctx)
        session = FreshReportSession(base, ctx, ledger.snapshot(), sink=lambda e: None)
        alt, _ = fixture(report_state="unknown", report_id="fixture_alt", model="maira2")
        self.assertFalse(self.acquire(session, ledger, alt))

    def test_current_report_protected_in_addition_to_initial_baseline(self):
        base, ctx = fixture()
        base["receipt"]["fact_states"][0]["xrv"] = "positive"
        reseal(base)
        ledger = base_ledger(base, ctx)
        session = FreshReportSession(base, ctx, ledger.snapshot(), sink=lambda e: None)
        first, _ = fixture(report_state="positive", report_id="fixture_first", model="maira2")
        first["receipt"]["fact_states"][0]["xrv"] = "positive"
        reseal(first)
        self.assertTrue(self.acquire(session, ledger, first))
        second, _ = fixture(report_state="unknown", report_id="fixture_second", model="llavarad")
        second["receipt"]["fact_states"][0].update(xrv="positive", chexbert="positive")
        reseal(second)
        self.assertFalse(self.acquire(session, ledger, second, suffix="2"))
        self.assertEqual(session.current["report_model_id"], "maira2")

    def test_report_failure_costs_one_without_retry(self):
        _, _, ledger, session, _ = self.setup_session()
        alt, _ = fixture(report_id="fixture_alt", model="maira2")
        self.assertFalse(self.acquire(session, ledger, alt, fail="report"))
        self.assertEqual(session.ledger["charged_model_attempts"], 5)
        self.assertEqual(session.ledger["failed_attempts"], 1)
        state = session.begin_planning()
        self.assertNotIn("m0001", [t["model_id"] for t in state["tools"]])

    def test_label_failure_costs_two_without_retry(self):
        _, _, ledger, session, _ = self.setup_session()
        alt, _ = fixture(report_id="fixture_alt", model="maira2")
        self.assertFalse(self.acquire(session, ledger, alt, fail="labels"))
        self.assertEqual(session.ledger["charged_model_attempts"], 6)
        self.assertTrue(session.history[0]["failed"])

    def test_all_experts_budgeted_once_and_no_empty_menu_llm_call(self):
        _, _, ledger, session, events = self.setup_session()
        for i, model in enumerate(("maira2", "llavarad", "chexagent2")):
            alt, _ = fixture(report_id=f"fixture_alt_{i}", model=model)
            self.acquire(session, ledger, alt, suffix=str(i))
        self.assertIsNone(session.begin_planning())
        result = session.result()
        self.assertEqual(result["charged_worker_attempts"], 10)
        self.assertEqual(result["charged_policy_requests"], 3)
        self.assertEqual(sum(e["event"] == "policy_reserved" for e in events), 3)

    def test_stop_has_no_worker_side_effect(self):
        _, _, _, session, _ = self.setup_session()
        state = session.begin_planning()
        self.assertIsNone(session.receive_decision(decision(state, action="abstain")))
        self.assertEqual(session.result()["charged_worker_attempts"], 4)

    def test_policy_failure_charged_and_terminal(self):
        _, _, _, session, _ = self.setup_session()
        session.begin_planning(); session.policy_failed()
        self.assertEqual(session.result()["charged_policy_requests"], 1)
        self.assertEqual(session.result()["charged_worker_attempts"], 4)

    def test_reservation_sink_failure_prevents_policy_start(self):
        _, _, _, session, _ = self.setup_session()
        def failed_sink(event):
            raise OSError("invented_fs_failure")
        session.sink = failed_sink
        with self.assertRaises(OSError): session.begin_planning()
        self.assertEqual(session.policy_requests, 0)

    def test_invalid_decision_does_not_start_report_worker(self):
        _, _, _, session, _ = self.setup_session()
        state = session.begin_planning()
        for field, value in (("action", "regenerate_cxr"), ("target_id", "t9999"), ("step_id", "s9999")):
            bad = decision(state); bad[field] = value
            with self.assertRaises(ContractError): session.receive_decision(bad)
        self.assertIsNone(session.active_tool)

    def test_cached_or_wrong_ledger_proposal_cannot_be_free(self):
        _, _, ledger, session, _ = self.setup_session()
        state = session.begin_planning(); session.receive_decision(decision(state))
        alt, _ = fixture(report_state="positive", report_id="fixture_alt", model="maira2")
        with self.assertRaises(ContractError): session.finish_report(alt, ledger.snapshot())

    def test_different_ehr_or_thresholds_rejected(self):
        for field in ("ehr_sha256", "thresholds_sha256"):
            _, _, ledger, session, _ = self.setup_session()
            state = session.begin_planning(); session.receive_decision(decision(state))
            alt, _ = fixture(report_state="positive", report_id="fixture_alt", model="maira2")
            book = append_report(ledger, alt)
            if field == "ehr_sha256": alt[field] = _digest("invented_changed_ehr")
            else: alt["receipt"][field] = _digest("invented_changed_thresholds")
            with self.assertRaises(ValueError): session.finish_report(alt, book)

    def test_invalid_baseline_structure_abstains_without_llm(self):
        base, ctx = fixture(); base["structure"].update(findings_complete=False, section_contract_pass=False)
        ledger = base_ledger(base, ctx)
        session = FreshReportSession(base, ctx, ledger.snapshot(), sink=lambda e: None)
        self.assertIsNone(session.begin_planning())
        self.assertIsNone(session.result()["selected_candidate_id"])
        self.assertEqual(session.policy_requests, 0)

    def test_inflight_prefix_and_undeclared_retry_profile_rejected(self):
        base, ctx, ledger, session, _ = self.setup_session()
        changed = copy.deepcopy(ledger.snapshot()); changed["max_retries"] = 1
        with self.assertRaises(ValueError): FreshReportSession(base, ctx, changed, sink=lambda e: None)

    def test_worker_gpu_guard_precedes_input_read(self):
        from tricompose_llm import fresh_report_worker
        with patch.object(fresh_report_worker, "require_gpu_slurm", side_effect=RuntimeError("no_gpu")), \
             patch.object(fresh_report_worker, "load_cxr_candidates") as read:
            with self.assertRaises(RuntimeError):
                fresh_report_worker.additional_report({}, {}, Path("."), {}, {}, model="maira2", step_index=0)
            read.assert_not_called()

    def test_entrypoints_do_not_load_models_at_import(self):
        for name in ("run_fresh_report_agent", "run_local_qwen_decision"):
            spec = importlib.util.spec_from_file_location("fixture_" + name, ROOT / "agent" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            with patch("tricompose_llm.local_qwen.LocalQwenPlanner.__init__", side_effect=AssertionError("no_model_factory")):
                spec.loader.exec_module(module)
            self.assertTrue(callable(module.run))


if __name__ == "__main__":
    unittest.main()
