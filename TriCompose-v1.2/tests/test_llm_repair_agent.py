"""Authored numeric fixtures only; no patients, model loading or external API."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

AGENT = Path(__file__).resolve().parents[1]/"agent"
sys.path.insert(0, str(AGENT))

from tricompose_llm.contracts import (ContractError, DECISION_SCHEMA, decode_json,
    decision_schema, validate_decision, validate_public_state)
from tricompose_llm.controller import LLMRepairController, Tool, tools_from_orders
from tricompose_llm.demo import DemoExecutor, observation
from tricompose_llm.planner import DemoPlanner, OpenAICompatiblePlanner, endpoint_url


def controller(initial=None, **kwargs):
    tools = tools_from_orders([("chexgenbench_sana", 0), ("roentgen_v2", 0)],
                              ["maira2", "cxrmate_single", "llavarad"])
    return LLMRepairController(initial or observation(), tools, **kwargs)


def proposal(state, action="regenerate_report", tool_id=None):
    row = next(r for r in state["evidence"] if r["candidate_id"] == state["current_candidate_id"])
    target = next(t["tool_id"] for t in state["tools"] if t["action"] == action) if action.startswith("regenerate") else None
    return {"schema_version": DECISION_SCHEMA, "step_id": state["step_id"], "action": action,
            "target_id": tool_id or target, "evidence_ids": [row["evidence_id"]],
            "reason_code": "explore_alternative"}


class FixedPlanner:
    def __init__(self, func): self.func = func
    def propose(self, state): return self.func(state)


class FixtureExecutor:
    mode = "authored_fixture"
    def __init__(self, func): self.func, self.calls = func, 0
    def execute(self, request, current):
        self.calls += 1
        return self.func(request, current)


class PublicContractTests(unittest.TestCase):
    def test_projection_contains_no_clinical_content_paths_or_hashes(self):
        value = observation()
        value["case_id"], value["candidate_id"] = "private_anchor_do_not_export", "private_report_do_not_export"
        state = controller(value).public_state(0)
        text = json.dumps(state)
        for forbidden in ("private_anchor", "private_report", "sha256", "diagnosis", "edema", "maira2", "sana", "1"*64):
            self.assertNotIn(forbidden, text)
        self.assertEqual(state["current_candidate_id"], "c0000")

    def test_numeric_state_is_roundtrippable_and_closed(self):
        state = controller().public_state(0)
        self.assertEqual(validate_public_state(decode_json(json.dumps(state))), state)
        for key in ("report_text", "patient_id", "ehr", "image", "path", "hash"):
            value = deepcopy(state); value[key] = "forbidden_fixture"
            with self.assertRaises(ContractError): validate_public_state(value)

    def test_nested_injection_channels_rejected(self):
        state = controller().public_state(0)
        for where in (state["budget"], state["evidence"][0], state["evidence"][0]["quality"],
                      state["evidence"][0]["edges"]["ehr_cxr"], state["tools"][0]):
            where["raw_report"] = "authored_injection"
            with self.assertRaises(ContractError): validate_public_state(state)
            del where["raw_report"]

    def test_arbitrary_string_ids_rejected_at_boundary(self):
        state = controller().public_state(0)
        for bad in ("case123", "/private/file", "patient_id", "e0000\nsecret", "a"*64):
            value = deepcopy(state); value["evidence"][0]["evidence_id"] = bad
            with self.assertRaises(ContractError): validate_public_state(value)

    def test_unknown_remains_missing_not_negative(self):
        state = controller(observation(ehr="unknown")).public_state(0)
        self.assertEqual(state["evidence"][0]["edges"]["ehr_cxr"]["known"], 0)
        self.assertEqual(state["evidence"][0]["edges"]["ehr_report"]["opposed"], 0)
        self.assertEqual(state["evidence"][0]["uncertainty"]["ehr_unknown"], 14)

    def test_finite_quality_and_integer_counts_required(self):
        for bad in (float("nan"), float("inf"), True, "0.8", 1.1):
            state = controller().public_state(0); state["evidence"][0]["quality"]["report_structure"] = bad
            with self.assertRaises(ContractError): validate_public_state(state)
        for bad in (True, -1, 14.0, 15):
            state = controller().public_state(0); state["evidence"][0]["edges"]["ehr_cxr"]["known"] = bad
            with self.assertRaises(ContractError): validate_public_state(state)

    def test_null_quality_is_not_filled(self):
        value = observation(); value["quality"]["report_structure_quality_score_0_1"] = None
        self.assertIsNone(controller(value).public_state(0)["evidence"][0]["quality"]["report_structure"])

    def test_count_algebra_cannot_silently_change(self):
        state = controller().public_state(0)
        state["evidence"][0]["edges"]["ehr_cxr"]["supported"] = 0
        with self.assertRaises(ContractError): validate_public_state(state)

    def test_strict_decision_requires_current_step_and_observed_evidence(self):
        state = controller().public_state(0)
        self.assertEqual(validate_decision(proposal(state), state)["action"], "regenerate_report")
        for field, value in (("step_id", "s0009"), ("evidence_ids", ["e9999"]),
                             ("evidence_ids", []), ("evidence_ids", ["e0000", "e0000"]),
                             ("action", "replace_ehr"), ("reason_code", "invented_free_text"),
                             ("target_id", "t9999")):
            decision = proposal(state); decision[field] = value
            with self.assertRaises(ContractError): validate_decision(decision, state)

    def test_target_action_must_match(self):
        state = controller().public_state(0)
        decision = proposal(state); decision["action"] = "regenerate_cxr"
        with self.assertRaises(ContractError): validate_decision(decision, state)

    def test_terminal_has_no_tool_target(self):
        state = controller().public_state(0)
        for action in ("stop", "abstain"):
            value = proposal(state, action)
            validate_decision(value, state)
            value["target_id"] = "t0001"
            with self.assertRaises(ContractError): validate_decision(value, state)

    def test_extra_rationale_or_prompt_is_rejected(self):
        state = controller().public_state(0)
        for field in ("prompt", "reasoning", "confidence", "patient_id"):
            decision = proposal(state); decision[field] = "invented_content"
            with self.assertRaises(ContractError): validate_decision(decision, state)

    def test_schema_is_closed_and_only_observed_ids_allowed(self):
        state = controller().public_state(0); schema = decision_schema(state)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(schema["properties"]["evidence_ids"]["items"]["enum"], ["e0000"])

    def test_json_duplicate_nonfinite_markdown_and_prefix_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '```json\n{}\n```', 'Here is {}'):
            with self.assertRaises(ContractError): decode_json(text)
        with self.assertRaises(ContractError): decode_json(" "*32769)


class LoopTests(unittest.TestCase):
    def test_mock_demo_commits_strict_proxy_gain_and_stops(self):
        result = controller().run(DemoPlanner(), DemoExecutor())
        self.assertEqual(result["accepted_proxy_transitions"], 1)
        self.assertEqual(result["terminal_reason"], "stop")
        self.assertEqual(result["spent_units"], 8)  # initial4 + two planners + report2
        self.assertEqual(result["actual_generator_calls"], 0)
        self.assertIsNone(result["clinical_repair_success"])

    def test_llm_can_choose_alternative_not_forced_deterministic_argmax(self):
        c = controller(max_steps=1)
        result = c.run(FixedPlanner(lambda state: proposal(state, tool_id="t0002")), DemoExecutor())
        self.assertEqual(result["selected_observation"]["lineage"]["report_model_id"], "llavarad")
        self.assertEqual(result["history"][0]["tool_id"], "t0002")

    def test_llm_selects_image_branch_and_report_expert_stays_fixed(self):
        result = controller(observation(image="negative", report="positive"), max_steps=1).run(
            FixedPlanner(lambda s: proposal(s, "regenerate_cxr")), DemoExecutor())
        self.assertEqual(result["accepted_proxy_transitions"], 1)
        self.assertEqual(result["selected_observation"]["lineage"]["report_model_id"], "maira2")
        self.assertEqual(result["spent_units"], 9)

    def test_invalid_planner_charged_but_no_worker(self):
        executor = FixtureExecutor(lambda *_: self.fail("worker must not run"))
        result = controller().run(FixedPlanner(lambda _: {"action": "erase_ehr"}), executor)
        self.assertEqual(result["terminal_reason"], "planner_failed_or_invalid")
        self.assertEqual(result["spent_units"], 5)
        self.assertEqual(executor.calls, 0)

    def test_failed_worker_charged_once_no_retry(self):
        def fail(*_): raise RuntimeError("sensitive_error_not_logged")
        result = controller(max_steps=1).run(FixedPlanner(proposal), FixtureExecutor(fail))
        self.assertEqual(result["spent_units"], 7)
        self.assertTrue(result["history"][0]["failed"])
        self.assertNotIn("sensitive_error", json.dumps(result))

    def test_unknown_cannot_silence_conflicts(self):
        for state in ("unknown", "uncertain"):
            def output(req, current, state=state):
                value = DemoExecutor().execute(req, current); value["states"]["chexbert"]["edema"] = state
                return value
            result = controller(max_steps=1).run(FixedPlanner(proposal), FixtureExecutor(output))
            self.assertEqual(result["accepted_proxy_transitions"], 0)
            self.assertEqual(result["selected_observation"]["candidate_id"], "authored_triple_0")
            self.assertEqual(result["history"][0]["lost_comparison_count"], 2)

    def test_changed_ehr_never_commits(self):
        for field in ("ehr_sha256", "ehr_facts_sha256"):
            def output(req, current, field=field):
                value = DemoExecutor().execute(req, current); value["lineage"][field] = "9"*64
                return value
            result = controller(max_steps=1).run(FixedPlanner(proposal), FixtureExecutor(output))
            self.assertEqual(result["accepted_proxy_transitions"], 0)
            self.assertTrue(result["history"][0]["failed"])
            self.assertEqual(result["selected_observation"]["lineage"][field], controller().initial["lineage"][field])

    def test_same_report_expert_required_on_image_branch(self):
        def output(req, current):
            value = DemoExecutor().execute(req, current); value["lineage"]["report_model_id"] = "llavarad"
            return value
        result = controller(observation(image="negative", report="positive"), max_steps=1).run(
            FixedPlanner(lambda s: proposal(s, "regenerate_cxr")), FixtureExecutor(output))
        self.assertTrue(result["history"][0]["failed"])
        self.assertEqual(result["accepted_proxy_transitions"], 0)

    def test_new_contradiction_rolls_back(self):
        def output(req, current):
            value = DemoExecutor().execute(req, current)
            value["states"]["xrv"]["pneumothorax"] = "negative"
            value["states"]["chexbert"]["pneumothorax"] = "positive"
            return value
        # Fixed-image state must stay fixed too; here changing XRV also fails.
        result = controller(max_steps=1).run(FixedPlanner(proposal), FixtureExecutor(output))
        self.assertEqual(result["accepted_proxy_transitions"], 0)

    def test_duplicate_bytes_not_improvement(self):
        def output(req, current):
            value = DemoExecutor().execute(req, current)
            value["lineage"]["report_sha256"] = current["lineage"]["report_sha256"]
            return value
        result = controller(max_steps=1).run(FixedPlanner(proposal), FixtureExecutor(output))
        self.assertFalse(result["history"][0]["accepted_proxy_transition"])

    def test_budget_menu_reserves_planner_cost_first(self):
        seen = []
        def choose(state):
            seen.append(state)
            self.assertEqual(state["budget"]["spent_units"], 5)
            self.assertEqual(state["tools"], [])
            return proposal(state, "abstain")
        result = controller(budget_units=6).run(FixedPlanner(choose), DemoExecutor())
        self.assertEqual(result["spent_units"], 5)
        self.assertEqual(result["tool_attempts"], 0)

    def test_no_planner_if_no_budget_remains(self):
        result = controller(budget_units=4).run(FixedPlanner(lambda _: self.fail()), DemoExecutor())
        self.assertEqual(result["terminal_reason"], "budget_exhausted")
        self.assertEqual(result["planner_calls"], 0)

    def test_bounded_steps_and_no_repeat_run(self):
        c = controller(max_steps=1)
        result = c.run(DemoPlanner(), DemoExecutor())
        self.assertEqual(result["terminal_reason"], "step_limit")
        with self.assertRaises(ContractError): c.run(DemoPlanner(), DemoExecutor())

    def test_live_executor_not_authorized_by_api_flag_or_slurm_name(self):
        executor = FixtureExecutor(lambda *_: self.fail()); executor.mode = "approved_slurm_backend"
        with self.assertRaises(ContractError): controller().run(DemoPlanner(), executor)
        self.assertEqual(executor.calls, 0)

    def test_only_completed_requested_observations_exposed(self):
        seen = []
        def choose(state):
            seen.append(deepcopy(state))
            return proposal(state) if len(seen) == 1 else proposal(state, "stop")
        controller().run(FixedPlanner(choose), DemoExecutor())
        self.assertEqual(len(seen[0]["evidence"]), 1)
        self.assertEqual(len(seen[1]["evidence"]), 2)
        self.assertTrue(seen[1]["history"][0]["accepted_proxy_transition"])

    def test_planner_cannot_mutate_local_state(self):
        def choose(state):
            value = proposal(state); state["evidence"][0]["quality"]["report_structure"] = 0
            return value
        c = controller(max_steps=1); c.run(FixedPlanner(choose), DemoExecutor())
        self.assertEqual(c.initial["quality"]["report_structure_quality_score_0_1"], .8)

    def test_durable_reservation_events_precede_effects(self):
        events = []
        def choose(state):
            self.assertEqual(events[-1]["event"], "planner_reserved")
            return proposal(state)
        def execute(req, current):
            self.assertEqual(events[-1]["event"], "tool_reserved")
            return DemoExecutor().execute(req, current)
        controller(max_steps=1, event_sink=events.append).run(FixedPlanner(choose), FixtureExecutor(execute))
        self.assertEqual(events[-1]["event"], "tool_completed")

    def test_journal_failure_never_retries_or_invokes_worker(self):
        def sink(_): raise OSError("journal failed")
        executor = FixtureExecutor(lambda *_: self.fail())
        with self.assertRaises(OSError): controller(event_sink=sink).run(DemoPlanner(), executor)
        self.assertEqual(executor.calls, 0)

    def test_bad_tool_ids_seeds_and_duplicate_bindings_rejected(self):
        for tools in ([Tool("bad", "regenerate_report", "maira2")],
                      [Tool("t0000", "erase", "maira2")],
                      [Tool("t0000", "regenerate_report", "maira2", True)],
                      [Tool("t0000", "regenerate_report", "maira2"), Tool("t0001", "regenerate_report", "maira2")]):
            with self.assertRaises(ContractError): LLMRepairController(observation(), tools)


class TransportTests(unittest.TestCase):
    def test_url_requires_tls_except_loopback_and_disallows_credentials(self):
        self.assertEqual(endpoint_url("https://example.invalid/v1"), "https://example.invalid/v1/chat/completions")
        self.assertEqual(endpoint_url("http://127.0.0.1:8000/v1/chat/completions"), "http://127.0.0.1:8000/v1/chat/completions")
        for url in ("http://example.invalid/v1", "https://user:secret@example.invalid/v1", "https://example.invalid/v1?secret=yes",
                    "https://example.invalid/v1#x", "file:///secret", "https://example.invalid/\nsecret"):
            with self.assertRaises(ContractError): endpoint_url(url)

    def test_request_strict_schema_and_no_private_fields(self):
        p = OpenAICompatiblePlanner(base_url="https://example.invalid/v1", model="authored-model")
        payload = p.request_payload(controller().public_state(0))
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertFalse(payload["stream"])
        self.assertNotIn("private", payload["messages"][1]["content"])
        self.assertNotIn("sha256", payload["messages"][1]["content"])

    def test_compatibility_json_object_is_explicit_and_has_local_schema(self):
        p = OpenAICompatiblePlanner(base_url="https://example.invalid", model="authored-model", response_format="json_object")
        payload = p.request_payload(controller().public_state(0))
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("Required JSON schema", payload["messages"][0]["content"])

    def call(self, response, state=None):
        state = state or controller().public_state(0)
        p = OpenAICompatiblePlanner(base_url="https://example.invalid", model="authored-model", api_key="FAKE_UNIT_TEST_KEY")
        raw = json.dumps(response).encode()
        class Response(io.BytesIO):
            pass
        class Opener:
            def open(self, request, timeout):
                self.request = request
                return Response(raw)
        opener = Opener()
        with patch("urllib.request.build_opener", return_value=opener):
            result = p.propose(state)
        return p, opener, result

    def test_injected_transport_roundtrip_without_network(self):
        state = controller().public_state(0)
        p, opener, result = self.call({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(proposal(state))}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}}, state)
        self.assertEqual(result["target_id"], "t0001")
        self.assertEqual(p.api_attempts, 1)
        self.assertEqual(p.usage["total_tokens"], 50)
        self.assertEqual(p.usage_available_calls, 1)
        data = json.loads(opener.request.data)
        self.assertNotIn("FAKE_UNIT_TEST_KEY", json.dumps(data))

    def test_refusal_truncation_tools_extras_and_malformed_decisions_fail_closed(self):
        state = controller().public_state(0)
        for choice in ({"finish_reason": "length", "message": {"content": json.dumps(proposal(state))}},
                       {"finish_reason": "stop", "message": {"refusal": "no", "content": "{}"}},
                       {"finish_reason": "stop", "message": {"tool_calls": [{"function": "unsafe"}], "content": "{}"}},
                       {"finish_reason": "stop", "message": {"content": "```json\n{}\n```"}}):
            with self.assertRaisesRegex(ContractError, "planner_call_failed_or_invalid"):
                self.call({"choices": [choice]}, state)

    def test_api_failure_has_no_body_credentials_or_retries(self):
        p = OpenAICompatiblePlanner(base_url="https://example.invalid", model="authored-model", api_key="FAKE_SECRET")
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = RuntimeError("sensitive_private_body")
            with self.assertRaisesRegex(ContractError, "^planner_call_failed_or_invalid$"):
                p.propose(controller().public_state(0))
            self.assertEqual(opener.return_value.open.call_count, 1)
            self.assertEqual(p.api_attempts, 1)

    def test_boolean_usage_is_not_accepted_as_tokens(self):
        state = controller().public_state(0)
        p, _, _ = self.call({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(proposal(state))}}],
            "usage": {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3}}, state)
        self.assertEqual(p.usage_available_calls, 0)

    def test_no_redirect_adapter(self):
        from tricompose_llm.planner import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid"))

    def test_local_qwen_fails_before_torch_import_outside_gpu(self):
        from tricompose_llm.local_qwen import LocalQwenPlanner
        with patch("tricompose_llm.local_qwen.gpu_guard", side_effect=ContractError("gpu_required")):
            with self.assertRaisesRegex(ContractError, "gpu_required"):
                LocalQwenPlanner("/not/read")

    def test_cli_defaults_are_offline_mock(self):
        spec = importlib.util.spec_from_file_location("_llm_repair_cli_test", AGENT/"run.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        args = module.build_parser().parse_args(["--run-id", "authored_cli_test"])
        self.assertEqual((args.mode, args.planner, args.allow_network), ("demo", "mock", False))


if __name__ == "__main__":
    unittest.main()
