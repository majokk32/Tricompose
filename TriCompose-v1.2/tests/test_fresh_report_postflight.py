"""Invented numeric states/receipts only; no model or artifact body reads."""
import copy
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import audit_fresh_report_agent as audit
from test_fresh_report_agent_bridge import base_ledger, append_report, decision
from test_fresh_output_acceptance import fixture
from tricompose_llm.live_report_bridge import FreshReportSession
from tricompose_v12.invariant_verification import _digest


def recorded(*, alternative=True, worker_failure=None, policy_failure=False,
             eligible=True, known=True, strict_gain=True):
    base, ctx = fixture(known=known)
    if not eligible:
        base["structure"].update(findings_complete=False, section_contract_pass=False)
    ledger = base_ledger(base, ctx)
    initial = ledger.snapshot()
    events, steps, rows = [], {}, [base]
    session = FreshReportSession(base, ctx, initial, sink=events.append)
    state = session.begin_planning()
    if state is not None:
        if policy_failure:
            steps[0] = {"state": state, "outcome": None, "continuation": None}
            session.policy_failed()
        elif not alternative:
            d = decision(state, action="stop")
            steps[0] = {"state": state, "outcome": {"status": "completed", "decision": d,
                "audit": {"invented_policy": True}}, "continuation": None}
            session.receive_decision(d)
        else:
            d = decision(state)
            session.receive_decision(d)
            alt, _ = fixture(known=known, report_state="positive" if strict_gain else "unknown",
                report_id="fixture_fresh_alt", model="maira2")
            book = append_report(ledger, alt, fail=worker_failure)
            session.finish_report(None if worker_failure else alt, book)
            if not worker_failure:
                rows.append(alt)
            steps[0] = {"state": state, "outcome": {"status": "completed", "decision": d,
                "audit": {"invented_policy": True}}, "continuation": book}
            state = session.begin_planning()
            d = decision(state, action="stop")
            steps[1] = {"state": state, "outcome": {"status": "completed", "decision": d,
                "audit": {"invented_policy": True}}, "continuation": None}
            session.receive_decision(d)
    sealed = session.result()
    seal = _digest("invented_sealed_file_bytes")
    events.append({"event": "selection_sealed", "sha256": seal})
    return {"rows": rows, "context": ctx, "initial": initial, "final": session.ledger,
        "journal": events, "steps": steps, "sealed": sealed, "selection_sha256": seal}


class FreshReportPostflightTests(unittest.TestCase):
    def test_completed_fresh_gain_replayed_exactly(self):
        result = audit.replay_session(**recorded())
        self.assertTrue(result["ledger_replay_pass"])
        self.assertTrue(result["policy_feedback_replay_pass"])
        self.assertEqual(result["selected_report_model"], "maira2")
        self.assertEqual(result["selection"]["charged_worker_attempts"], 6)
        self.assertEqual(result["selection"]["charged_policy_requests"], 2)
        self.assertEqual(result["selection"]["accepted_proxy_transitions"], 1)
        self.assertFalse(result["clinical_repair_success"])
        self.assertIsNone(result["clinical_accuracy"])

    def test_cosmetic_report_keeps_original_and_both_costs(self):
        result = audit.replay_session(**recorded(strict_gain=False))
        self.assertEqual(result["selected_report_model"], "cxrmate_single")
        self.assertEqual(result["selection"]["accepted_proxy_transitions"], 0)
        self.assertEqual(result["selection"]["charged_worker_attempts"], 6)

    def test_immediate_stop_has_no_generator_cost(self):
        result = audit.replay_session(**recorded(alternative=False))
        self.assertEqual(result["selection"]["charged_worker_attempts"], 4)
        self.assertEqual(result["selection"]["charged_policy_requests"], 1)

    def test_policy_failure_keeps_reservation_without_claiming_generation(self):
        result = audit.replay_session(**recorded(policy_failure=True))
        self.assertEqual(result["selection"]["charged_policy_requests"], 1)
        self.assertEqual(result["successful_policy_audits"], [])
        self.assertEqual(result["selection"]["charged_worker_attempts"], 4)

    def test_worker_and_label_failures_are_separately_charged(self):
        for phase, expected in (("report", 5), ("labels", 6)):
            with self.subTest(phase=phase):
                recorded_run = recorded(worker_failure=phase)
                result = audit.replay_session(**recorded_run)
                self.assertEqual(result["selection"]["charged_worker_attempts"], expected)
                self.assertEqual(result["selected_report_model"], "cxrmate_single")
                self.assertTrue(result["selection"]["history"][0]["failed"])
                counts = audit.phase_counts([recorded_run["final"]])
                self.assertEqual(counts["chexbert"], 1)
                self.assertEqual(counts["report_generator"], 1 if phase == "report" else 2)

    def test_invalid_structure_no_policy_request(self):
        result = audit.replay_session(**recorded(eligible=False))
        self.assertEqual(result["selection"]["charged_policy_requests"], 0)
        self.assertIsNone(result["selection"]["selected_candidate_id"])

    def test_unknown_ehr_ratios_stay_unavailable(self):
        result = audit.replay_session(**recorded(known=False))
        self.assertEqual(result["known_ehr_facts"], 0)
        self.assertIsNone(result["selected_raw_edges"]["ehr_report"]["support_over_known"])

    def test_comparison_table_keeps_ehr_na_and_valid_image_readouts(self):
        result = audit.replay_session(**recorded(known=False))
        rows = list(csv.DictReader(io.StringIO(audit.comparison_csv([result]))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["selected_ehr_report_support_over_known"], "NA")
        self.assertEqual(rows[0]["baseline_ehr_report_support_over_known"], "NA")
        self.assertEqual(rows[0]["selected_cxr_report_supported_positive"], "1")
        self.assertEqual(rows[0]["charged_worker_attempts"], "6")

    def test_no_selected_candidate_is_not_all_negative_or_zero_quality(self):
        result = audit.replay_session(**recorded(eligible=False))
        row = next(csv.DictReader(io.StringIO(audit.comparison_csv([result]))))
        self.assertEqual(row["selected_cxr_report_supported_positive"], "NA")
        self.assertEqual(row["selected_report_model"], "NA")
        self.assertEqual(row["charged_policy_requests"], "0")

    def test_future_evidence_cannot_appear_in_first_state(self):
        data = recorded()
        data["steps"][0]["state"]["evidence"].append(copy.deepcopy(data["steps"][1]["state"]["evidence"][1]))
        with self.assertRaisesRegex(ValueError, "future_or_altered"):
            audit.replay_session(**data)

    def test_numeric_state_count_tampering_rejected(self):
        data = recorded()
        data["steps"][0]["state"]["evidence"][0]["edges"]["cxr_report"]["supported"] += 1
        with self.assertRaises(ValueError): audit.replay_session(**data)

    def test_bool_cannot_masquerade_as_integer_count(self):
        for change in ("state", "selection", "ledger"):
            data = recorded(alternative=False)
            if change == "state": data["steps"][0]["state"]["budget"]["planner_units_per_call"] = True
            elif change == "selection": data["sealed"]["charged_policy_requests"] = True
            else: data["final"]["pending_attempts"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.replay_session(**data)

    def test_cost_refund_tampering_rejected(self):
        data = recorded(worker_failure="labels")
        data["sealed"]["charged_worker_attempts"] -= 1
        with self.assertRaisesRegex(ValueError, "selection_replay"):
            audit.replay_session(**data)

    def test_unreserved_model_call_not_hidden(self):
        data = recorded()
        data["final"]["charged_model_attempts"] += 1
        with self.assertRaisesRegex(ValueError, "snapshot_drift"):
            audit.replay_session(**data)

    def test_acceptance_feedback_tampering_rejected(self):
        data = recorded()
        event = next(e for e in data["journal"] if e["event"] == "fresh_report_finished")
        event["history"]["accepted_proxy_transition"] = False
        with self.assertRaisesRegex(ValueError, "journal_replay"):
            audit.replay_session(**data)

    def test_changed_policy_decision_rejected(self):
        data = recorded()
        data["steps"][0]["outcome"]["decision"]["target_id"] = "t0002"
        with self.assertRaises(ValueError): audit.replay_session(**data)

    def test_stopped_policy_cannot_dispatch_worker(self):
        data = recorded(alternative=False)
        data["steps"][0]["continuation"] = data["final"]
        with self.assertRaisesRegex(ValueError, "terminal_decision"):
            audit.replay_session(**data)

    def test_failed_policy_cannot_dispatch_worker(self):
        data = recorded(policy_failure=True)
        data["steps"][0]["continuation"] = data["final"]
        with self.assertRaisesRegex(ValueError, "failed_policy_started"):
            audit.replay_session(**data)

    def test_missing_worker_snapshot_rejected(self):
        data = recorded()
        data["steps"][0]["continuation"] = None
        with self.assertRaisesRegex(ValueError, "cost_missing"):
            audit.replay_session(**data)

    def test_candidate_order_and_duplicate_rows_rejected(self):
        for change in ("reverse", "duplicate"):
            data = recorded()
            data["rows"] = list(reversed(data["rows"])) if change == "reverse" else data["rows"] + [data["rows"][0]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.replay_session(**data)

    def test_sealed_selection_hash_bound_and_terminal_no_extra_events(self):
        for change in ("seal", "suffix"):
            data = recorded()
            if change == "seal": data["journal"][-1]["sha256"] = _digest("invented_wrong_seal")
            else: data["journal"].append({"event": "invented_after_stop"})
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.replay_session(**data)

    def test_unreserved_policy_file_rejected(self):
        data = recorded(alternative=False)
        data["steps"][2] = copy.deepcopy(data["steps"][0])
        with self.assertRaisesRegex(ValueError, "unreserved_policy"):
            audit.replay_session(**data)

    def test_no_completed_receipt_from_different_image_or_ehr(self):
        for field in ("cxr_sha256", "ehr_sha256"):
            data = recorded()
            data["rows"][1][field] = _digest("invented_different_artifact")
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.replay_session(**data)

    def test_replay_deterministic_and_does_not_mutate_inputs(self):
        data = recorded()
        before = copy.deepcopy(data)
        self.assertEqual(audit.replay_session(**data), audit.replay_session(**data))
        self.assertEqual(data, before)

    def test_all_three_fresh_experts_and_budget_terminal_replayed(self):
        base, ctx = fixture()
        ledger = base_ledger(base, ctx)
        initial = ledger.snapshot()
        events, steps, rows = [], {}, [base]
        session = FreshReportSession(base, ctx, initial, sink=events.append)
        for i, model in enumerate(("maira2", "llavarad", "chexagent2")):
            state = session.begin_planning()
            d = decision(state)
            session.receive_decision(d)
            alt, _ = fixture(report_state="positive" if i == 0 else "unknown",
                report_id=f"fixture_expert_{i}", model=model)
            book = append_report(ledger, alt, suffix=str(i))
            session.finish_report(alt, book)
            rows.append(alt)
            steps[i] = {"state": state, "outcome": {"status": "completed", "decision": d,
                "audit": {"invented_policy": True}}, "continuation": book}
        self.assertIsNone(session.begin_planning())
        sealed = session.result()
        seal = _digest("invented_sealed_file_bytes")
        events.append({"event": "selection_sealed", "sha256": seal})
        result = audit.replay_session(rows, ctx, initial, session.ledger, events, steps, sealed, seal)
        self.assertEqual(result["selection"]["charged_worker_attempts"], 10)
        self.assertEqual(result["selection"]["charged_policy_requests"], 3)
        self.assertEqual(result["selection"]["status"], "abstain_budget_limit")
        self.assertEqual(result["selected_report_model"], "maira2")
        self.assertEqual(len(result["selection"]["observed_candidate_ids"]), 4)

    def test_successful_policy_requires_frozen_pins_and_one_generation(self):
        pins, state_sha = {"model.safetensors": _digest("invented_weight")}, _digest("invented_state")
        outcome = {"status": "completed", "state_sha256": state_sha,
            "audit": {"frozen": True, "asset_pins": pins, "local_attempts": 1,
                "generate_attempts": 1, "failure_counts": {"invented_error": 0}}}
        audit.successful_policy(outcome, state_sha, pins)
        for field, value in (("frozen", False), ("generate_attempts", 2),
                             ("asset_pins", {}), ("failure_counts", {"invented_error": 1})):
            bad = copy.deepcopy(outcome); bad["audit"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.successful_policy(bad, state_sha, pins)
        with self.assertRaises(ValueError): audit.successful_policy(outcome, _digest("other_state"), pins)

    def test_cli_guard_runs_before_any_input_read(self):
        with patch.object(audit.gate, "cpu_guard", side_effect=RuntimeError("invented_no_cpu")), \
             patch.object(audit, "MetadataReader") as reader:
            with self.assertRaises(RuntimeError): audit.audit_run(object())
            reader.assert_not_called()

    def test_reader_never_leaves_protected_root_or_reads_unbounded_metadata(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); os.chmod(root, 0o2770)
            with patch.object(audit, "PROTECTED_ROOT", root):
                reader = audit.MetadataReader(root)
                with self.assertRaises(ValueError): reader.json(Path(__file__).resolve())
                with patch.object(reader, "path") as p:
                    p.return_value.stat.return_value.st_size = 16 * 1024 * 1024 + 1
                    with self.assertRaisesRegex(ValueError, "size_limit"): reader.json(root / "oversized.json")
                    p.return_value.read_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
