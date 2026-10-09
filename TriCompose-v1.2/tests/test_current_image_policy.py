"""Authored receipts, numeric states and fake processors; no patient/model reads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_current_image_policy as worker
from tricompose_llm.contracts import DECISION_SCHEMA
from tricompose_llm.current_image_qwen import (CurrentImageQwenPlanner, validate_packet,
    request_messages, rule_decision)
from test_fresh_output_acceptance import fixture
from test_guarded_requested_reports import case_fixture_bound, ingredients
from test_local_qwen_typed_content import planner


def case_fixture():
    old = case_fixture_bound()
    proposed, _ = fixture(image_state="negative", report_state="unknown", seed=1,
        image_id="invented_trial_image", report_id="invented_rejected_report", model="maira2")
    workers = ingredients()[3]["workers"]
    pair = worker.source.assess_pair(old, proposed, workers)
    base = old["baseline_row"]
    names = worker.cached.observer.existing.image_interface.FINDINGS
    facts = {f["finding"]: f for f in base["receipt"]["fact_states"]}
    observer = {"cxr_candidate_id": base["cxr_candidate_id"], "cxr_sha256": base["cxr_sha256"],
        "states": {n: facts[n]["xrv"] for n in names}, "contract_status": "complete"}
    observer["states"]["edema"] = "positive"  # Authored scorer disagreement, not truth.
    return worker.make_case(old, proposed, pair, observer, workers)


def decision(packet):
    state = packet["observation"]
    return {"schema_version": DECISION_SCHEMA, "step_id": state["step_id"],
        "action": "regenerate_report", "target_id": "t0102", "evidence_ids": ["e0000", "e0001"],
        "reason_code": "explore_alternative"}


class CurrentImagePolicyTests(unittest.TestCase):
    def test_exact_pair_not_other_image_evidence_and_deterministic_rule(self):
        case = case_fixture(); before = deepcopy(case)
        worker.validate_case(case)
        packet = case["packet"]
        self.assertEqual(len(packet["observation"]["evidence"]), 2)
        self.assertEqual(rule_decision(packet), decision(packet))
        self.assertEqual(case["rule_control"]["actual_model_calls"], 0)
        self.assertFalse(case["rule_control"]["reserved_planning_allowance_is_incurred_cost"])
        self.assertEqual(case, before)

    def test_typed_text_and_no_body_path_fact_name_or_endpoint_channel(self):
        packet = case_fixture()["packet"]
        messages = request_messages(packet)
        self.assertEqual(json.loads(messages[1]["content"][0]["text"]), packet)
        for m in messages: self.assertEqual(m["content"][0]["type"], "text")
        public = messages[1]["content"][0]["text"]
        for token in ("sha256", "/project2/", "edema", "maira2", "report_text", "biovil", "expert_history"):
            self.assertNotIn(token, public)

    def test_closed_extra_channels_rejected_before_tokenization(self):
        for name in ("expert_history", "raw_ehr", "biovil_score", "image_group", "report_text"):
            packet = case_fixture()["packet"]; packet[name] = "authored_forbidden"
            value = planner(CurrentImageQwenPlanner)
            with self.subTest(name=name), self.assertRaises(ValueError): value.propose_current(packet)
            self.assertIsNone(value.processor.seen); self.assertEqual(value.generate_attempts, 0)

    def test_feedback_cannot_change_quality_rejection_or_set_count_algebra(self):
        for kind in ("accept", "quality", "lost", "silenced", "hidden"):
            packet = case_fixture()["packet"]; feedback = packet["rejected_transition"]
            if kind == "accept": feedback["accepted_proxy_transition"] = True
            elif kind == "quality": feedback["quality_no_worse"] = False
            elif kind == "lost": feedback["lost"]["ehr_comparable"] = 1
            elif kind == "silenced": feedback["silenced"]["ehr"] = 1
            else: feedback["reason_text"] = "forbidden"
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate_packet(packet)

    def test_tools_are_exact_untried_experts_not_predictions_or_image_actions(self):
        for kind in ("model", "seed", "cost", "image", "score", "budget"):
            packet = case_fixture()["packet"]; t = packet["observation"]["tools"][0]
            if kind == "model": t["model_id"] = "m0000"
            elif kind == "seed": t["seed"] = 0
            elif kind == "cost": t["cost_units"] = 1
            elif kind == "image": t["action"] = "regenerate_cxr"
            elif kind == "score": t["predicted_quality"] = .99
            else: packet["observation"]["budget"]["spent_units"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate_packet(packet)

    def test_new_generate_path_not_programmatic_rule_or_cached_response(self):
        packet = case_fixture()["packet"]
        value = planner(CurrentImageQwenPlanner, text=json.dumps(decision(packet)))
        self.assertEqual(value.propose_current(packet), decision(packet))
        self.assertEqual(value.model.calls, 1); self.assertEqual(value.generate_attempts, 1)
        self.assertTrue(value.last_response_metadata["response_sha256"])

    def test_invalid_outputs_fail_charged_without_fallback(self):
        packet = case_fixture()["packet"]
        for kind in ("json", "wrong_tool", "context", "limit", "model"):
            d = decision(packet)
            if kind == "wrong_tool": d["target_id"] = "t0100"
            kwargs = {"text": json.dumps(d)}
            if kind == "json": kwargs["text"] = "authored_private_invalid_response"
            elif kind == "context": kwargs["input_length"] = 8193
            elif kind == "limit": kwargs["output_length"] = 384
            elif kind == "model": kwargs["fail"] = True
            value = planner(CurrentImageQwenPlanner, **kwargs)
            with self.subTest(kind=kind), self.assertRaises(ValueError): value.propose_current(packet)
            self.assertEqual(value.local_attempts, 1)
            self.assertEqual(value.model.calls, 0 if kind == "context" else 1)
            self.assertNotIn("authored_private_invalid_response", str(value.last_response_metadata))

    def test_dispatch_remains_deferred_and_stop_not_clinical_acceptance(self):
        packet = case_fixture()["packet"]; events = []
        dispatch = worker.dispatch_record(decision(packet), packet, sink=events.append)
        self.assertEqual(dispatch["status"], "deferred_separately_approved_backend_required")
        self.assertEqual(dispatch["actual_worker_model_calls"], 0)
        self.assertFalse(dispatch["clinical_acceptance"])
        d = decision(packet); d.update(action="stop", target_id=None, reason_code="no_further_action")
        blocked = worker.dispatch_record(d, packet, sink=lambda event: None)
        self.assertEqual(blocked["effective_decision"]["action"], "abstain")
        self.assertFalse(blocked["clinical_acceptance"])

    def test_risk_numeric_types_and_current_image_counts_bound(self):
        for kind in ("bool", "nan", "model", "image"):
            packet = case_fixture()["packet"]
            if kind == "bool": packet["observed_reports"][0]["risk_flags"]["empty"] = 0
            elif kind == "nan": packet["observed_reports"][0]["repeated_4gram_ratio"] = float("nan")
            elif kind == "model": packet["observed_reports"][1]["model_id"] = "m0000"
            else: packet["observation"]["evidence"][1]["edges"]["ehr_cxr"]["supported"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate_packet(packet)

    def test_private_catalog_and_rule_cannot_be_swapped_after_packet_sealing(self):
        for kind in ("catalog", "rule", "hash"):
            case = case_fixture()
            if kind == "catalog": case["tool_catalog"]["t0102"]["model_id"] = "maira2"
            elif kind == "rule": case["rule_control"]["actual_model_calls"] = 1
            else: case["policy_input_sha256"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError): worker.validate_case(case)


if __name__ == "__main__": unittest.main()
