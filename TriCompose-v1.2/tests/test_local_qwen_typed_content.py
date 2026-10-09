"""Reproduce the actual processor boundary in pure Python, no GPU or weights."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"agent"))
from tricompose_llm.contracts import ContractError, DECISION_SCHEMA
from tricompose_llm.controller import LLMRepairController, tools_from_orders
from tricompose_llm.demo import observation
from tricompose_llm.local_qwen import LocalQwenPlanner as LegacyPlanner
from tricompose_llm.local_qwen_v2 import LocalQwenPlanner, STAGES, request_messages


def state():
    tools = tools_from_orders([("chexgenbench_sana", 0), ("roentgen_v2", 0)], ["maira2", "cxrmate_single"])
    return LLMRepairController(observation(), tools).public_state(0)


def decision(value):
    return {"schema_version": DECISION_SCHEMA, "step_id": value["step_id"],
        "action": "regenerate_report", "target_id": "t0001", "evidence_ids": ["e0000"],
        "reason_code": "report_mismatch"}


class Batch(dict):
    def to(self, device): return self


class Generated:
    def __init__(self, length): self.shape = (1, length)
    def __getitem__(self, key): return "authored_token_fixture"


class Processor:
    def __init__(self, text, length=3):
        self.text, self.length, self.seen = text, length, None
    def apply_chat_template(self, messages, **kwargs):
        # The same tokenize=True content traversal as installed processing_utils.py:1640.
        for message in messages:
            visuals = [content for content in message["content"] if content["type"] in ("image", "video")]
            if visuals: raise AssertionError("numeric text planner must never send images")
        self.seen = deepcopy(messages)
        return Batch(input_ids=SimpleNamespace(shape=(1, self.length)))
    def batch_decode(self, *args, **kwargs): return [self.text]


class Model:
    def __init__(self, length=13, fail=False): self.length, self.fail, self.calls = length, fail, 0
    def generate(self, **kwargs):
        self.calls += 1
        if self.fail: raise RuntimeError("private_exception_never_logged")
        return Generated(self.length)


def planner(kind=LocalQwenPlanner, text=None, input_length=3, output_length=10, fail=False):
    value = object.__new__(kind)  # No constructor, torch import, checkpoint or inference.
    value.processor = Processor(text or json.dumps(decision(state())), length=input_length)
    value.model = Model(length=input_length+output_length, fail=fail)
    value.torch = SimpleNamespace(inference_mode=nullcontext)
    value.usage = {"input_tokens": 0, "output_tokens": 0}
    value.local_attempts = 0
    value.generate_attempts = 0
    value.failure_counts = {key: 0 for key in STAGES}
    value.last_response_metadata = None
    return value


class TypedContentTests(unittest.TestCase):
    def test_legacy_bug_reproduces_before_generation(self):
        value = planner(LegacyPlanner)
        with self.assertRaises(ContractError): value.propose(state())
        self.assertEqual(value.model.calls, 0)
        self.assertEqual(value.usage, {"input_tokens": 0, "output_tokens": 0})

    def test_v2_passes_same_processor_traversal_and_decision_contract(self):
        value = planner()
        self.assertEqual(value.propose(state()), decision(state()))
        self.assertEqual(value.model.calls, 1)
        self.assertEqual(value.generate_attempts, 1)
        self.assertEqual(value.usage, {"input_tokens": 3, "output_tokens": 10})
        self.assertFalse(any(value.failure_counts.values()))

    def test_system_and_user_both_typed_text_only(self):
        messages = request_messages(state())
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        for message in messages:
            self.assertIsInstance(message["content"], list)
            self.assertEqual(len(message["content"]), 1)
            self.assertEqual(set(message["content"][0]), {"type", "text"})
            self.assertEqual(message["content"][0]["type"], "text")

    def test_numeric_state_is_byte_equivalent_after_message_wrapping(self):
        value = state(); original = deepcopy(value)
        content = request_messages(value)[1]["content"][0]["text"]
        self.assertEqual(json.loads(content), value)
        self.assertEqual(value, original)
        for key in ("sha256", "edema", "diagnosis", "authored_case", "/project2/"):
            self.assertNotIn(key, content)

    def test_extra_private_content_never_reaches_processor(self):
        value = planner(); source = state(); source["report_text"] = "private_fixture"
        with self.assertRaises(ContractError): value.propose(source)
        self.assertIsNone(value.processor.seen)
        self.assertEqual(value.failure_counts["state_contract"], 1)

    def test_stage_diagnostics_preserve_masked_failure(self):
        value = planner(fail=True)
        with self.assertRaisesRegex(ContractError, "^local_planner_failed_or_invalid$"):
            value.propose(state())
        self.assertEqual(value.failure_counts["generate"], 1)
        self.assertEqual(value.generate_attempts, 1)
        self.assertNotIn("private_exception", str(value.failure_counts))

    def test_context_limit_before_generation(self):
        value = planner(input_length=8193)
        with self.assertRaises(ContractError): value.propose(state())
        self.assertEqual(value.failure_counts["context_bound"], 1)
        self.assertEqual(value.model.calls, 0)

    def test_truncated_output_still_rejected_and_charged(self):
        value = planner(output_length=384)
        with self.assertRaises(ContractError): value.propose(state())
        self.assertEqual(value.failure_counts["decode"], 1)
        self.assertTrue(value.last_response_metadata["token_limit_reached"])
        self.assertEqual(value.generate_attempts, 1)

    def test_invalid_json_does_not_log_raw_response(self):
        value = planner(text="authored_bad_response")
        with self.assertRaises(ContractError): value.propose(state())
        self.assertEqual(value.failure_counts["json_contract"], 1)
        self.assertEqual(len(value.last_response_metadata["response_sha256"]), 64)
        self.assertNotIn("authored_bad_response", str(value.last_response_metadata))

    def test_unknown_tool_still_rejected_after_valid_json(self):
        obj = decision(state()); obj["target_id"] = "t9999"
        value = planner(text=json.dumps(obj))
        with self.assertRaises(ContractError): value.propose(state())
        self.assertEqual(value.failure_counts["decision_contract"], 1)

    def test_markdown_not_relaxed_to_make_smoke_pass(self):
        value = planner(text="```json\n"+json.dumps(decision(state()))+"\n```")
        with self.assertRaises(ContractError): value.propose(state())
        self.assertEqual(value.failure_counts["json_contract"], 1)


if __name__ == "__main__":
    unittest.main()
