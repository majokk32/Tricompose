"""Authored numeric-envelope and fake-processor tests; no torch/model loads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"agent"))
from tricompose_llm.guard_aware_qwen import GuardAwareQwenPlanner, validate_packet, request_messages
from test_guarded_fresh_policy import case_fixture, decision
from test_local_qwen_typed_content import planner


class GuardAwareQwenTests(unittest.TestCase):
    def test_typed_text_only_packet_is_closed_and_unchanged(self):
        packet = case_fixture()["packet"]; before = deepcopy(packet)
        messages = request_messages(packet)
        self.assertEqual(packet, before)
        self.assertTrue(all(m["content"][0]["type"] == "text" for m in messages))
        public = messages[1]["content"][0]["text"]
        self.assertEqual(json.loads(public), packet)
        for token in ("ehr_sha256", "cxr_sha256", "/project2/", "edema", "maira2", "report_text", "invented_trial"):
            self.assertNotIn(token, public)

    def test_real_fresh_propose_path_not_cached_or_programmatic(self):
        packet = case_fixture()["packet"]
        value = planner(GuardAwareQwenPlanner, text=json.dumps(decision(packet)))
        result = value.propose_guarded(packet)
        self.assertEqual(result, decision(packet))
        self.assertEqual(value.generate_attempts, 1)
        self.assertEqual(value.local_attempts, 1)
        self.assertEqual(value.model.calls, 1)
        self.assertTrue(value.last_response_metadata["response_sha256"])

    def test_private_extra_fields_rejected_before_tokenizer(self):
        packet = case_fixture()["packet"]; packet["raw_ehr"] = "forbidden_authored_fixture"
        value = planner(GuardAwareQwenPlanner)
        with self.assertRaises(ValueError): value.propose_guarded(packet)
        self.assertIsNone(value.processor.seen)
        self.assertEqual(value.generate_attempts, 0)

    def test_history_cannot_be_relabelled_as_current_image_evidence(self):
        for kind in ("same_image", "future_evidence", "duplicate_model", "current_model_tool"):
            packet = case_fixture()["packet"]
            if kind == "same_image": packet["candidate_image_groups"][0]["image_group_id"] = "i0001"
            elif kind == "future_evidence": packet["expert_history"][0]["evidence_id"] = "e9999"
            elif kind == "duplicate_model": packet["expert_history"][0]["model_id"] = packet["expert_history"][1]["model_id"]
            else: packet["observation"]["tools"][0]["model_id"] = packet["current_report_model_id"]
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate_packet(packet)

    def test_tool_cost_budget_seed_and_generation_authority_cannot_be_hidden(self):
        for kind in ("cost", "seed", "budget", "backend"):
            packet = case_fixture()["packet"]
            if kind == "cost": packet["observation"]["tools"][0]["cost_units"] = 1
            elif kind == "seed": packet["observation"]["tools"][0]["seed"] = 0
            elif kind == "budget": packet["observation"]["budget"]["spent_units"] = 0
            else: packet["generation_backend_installed"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate_packet(packet)

    def test_wrong_tool_or_cxr_request_is_rejected_after_charged_generate(self):
        packet = case_fixture()["packet"]
        for kind in ("tool", "image"):
            result = decision(packet)
            if kind == "tool": result["target_id"] = "t9999"
            else: result["action"] = "regenerate_cxr"
            value = planner(GuardAwareQwenPlanner, text=json.dumps(result))
            with self.subTest(kind=kind), self.assertRaises(ValueError): value.propose_guarded(packet)
            self.assertEqual(value.generate_attempts, 1)
            self.assertEqual(value.failure_counts["decision_contract"], 1)

    def test_invalid_json_and_token_limit_stay_fail_closed_not_repaired(self):
        packet = case_fixture()["packet"]
        for kind in ("json", "limit", "context", "model"):
            kwargs = {"text": json.dumps(decision(packet))}
            if kind == "json": kwargs["text"] = "private_authored_response"
            elif kind == "limit": kwargs["output_length"] = 384
            elif kind == "context": kwargs["input_length"] = 8193
            else: kwargs["fail"] = True
            value = planner(GuardAwareQwenPlanner, **kwargs)
            with self.subTest(kind=kind), self.assertRaises(ValueError): value.propose_guarded(packet)
            self.assertEqual(value.local_attempts, 1)
            self.assertEqual(value.model.calls, 0 if kind == "context" else 1)
            self.assertNotIn("private_authored_response", str(value.last_response_metadata))


if __name__ == "__main__": unittest.main()
