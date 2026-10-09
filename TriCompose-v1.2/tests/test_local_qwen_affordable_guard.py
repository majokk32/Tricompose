"""Wholly authored contract regression, no model, dataset, API or GPU."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"agent"))
from tricompose_llm.contracts import ContractError, DECISION_SCHEMA, validate_decision
from tricompose_llm.controller import LLMRepairController, tools_from_orders
from tricompose_llm.demo import observation
from tricompose_llm.local_qwen_bounded import LocalQwenPlanner


def tools():
    return tools_from_orders([("chexgenbench_sana", 0), ("roentgen_v2", 0)],
        ["maira2", "cxrmate_single", "llavarad", "chexagent2"])


def state(): return LLMRepairController(observation(), tools()).public_state(0)


def planner():
    value = object.__new__(LocalQwenPlanner)  # Bypass constructor/torch/weights.
    value.programmatic_terminal_stops, value.proposal_sources = 0, []
    value.local_attempts = 0
    return value


def valid_decision(source):
    tool = next(t for t in source["tools"] if t["action"] == "regenerate_report")
    return {"schema_version": DECISION_SCHEMA, "step_id": source["step_id"],
        "action": tool["action"], "target_id": tool["tool_id"], "evidence_ids": ["e0000"],
        "reason_code": "explore_alternative"}


class AffordableGuardTests(unittest.TestCase):
    def test_empty_menu_abstains_before_any_model_attempt(self):
        source = state(); source["tools"] = []
        value = planner()
        with patch("tricompose_llm.local_qwen_bounded.TypedQwenPlanner.propose") as model:
            decision = value.propose(source)
            model.assert_not_called()
        validate_decision(decision, source)
        self.assertEqual(decision["action"], "abstain")
        self.assertIsNone(decision["target_id"])
        self.assertEqual(value.local_attempts, 0)
        self.assertEqual(value.programmatic_terminal_stops, 1)
        self.assertEqual(value.proposal_sources[0]["source"], "guard_empty_menu")

    def test_feasible_menu_uses_real_planner_not_argmax_override(self):
        source = state(); value = planner(); result = valid_decision(source)
        with patch("tricompose_llm.local_qwen_bounded.TypedQwenPlanner.propose", return_value=result) as model:
            self.assertEqual(value.propose(source), result)
            model.assert_called_once_with(source)
        self.assertEqual(value.programmatic_terminal_stops, 0)
        self.assertEqual(value.proposal_sources[0]["source"], "llm")

    def test_invalid_model_decision_not_converted_to_successful_stop(self):
        with patch("tricompose_llm.local_qwen_bounded.TypedQwenPlanner.propose",
                   side_effect=ContractError("local_planner_failed_or_invalid")):
            value = planner()
            with self.assertRaises(ContractError): value.propose(state())
        self.assertEqual(value.programmatic_terminal_stops, 0)
        self.assertEqual(value.proposal_sources[0]["source"], "llm")

    def test_closed_numeric_state_enforced_before_guard(self):
        source = state(); source["tools"] = []; source["report_text"] = "private_fixture"
        value = planner()
        with self.assertRaises(ContractError): value.propose(source)
        self.assertEqual(value.proposal_sources, [])

    def test_programmatic_stop_does_not_mutate_source_state(self):
        source = state(); source["tools"] = []; before = deepcopy(source)
        planner().propose(source)
        self.assertEqual(source, before)

    def test_guard_uses_current_observed_evidence_not_unseen_reference(self):
        source = state(); source["tools"] = []
        decision = planner().propose(source)
        self.assertEqual(decision["evidence_ids"], [source["evidence"][0]["evidence_id"]])

    def test_fourth_request_stops_after_three_failed_report_probes(self):
        value = planner()
        def fake_model(source):
            value.local_attempts += 1
            return valid_decision(source)
        class FailedGainExecutor:
            mode = "authored_fixture"
            def __init__(self): self.count = 0
            def execute(self, request, current):
                self.count += 1
                result = deepcopy(current)
                result["candidate_id"] = f"authored_alternative_{self.count}"
                result["lineage"]["report_model_id"] = request.slot[2]
                result["lineage"]["report_candidate_id"] = f"authored_report_{self.count}"
                result["lineage"]["report_sha256"] = f"{self.count+10:064x}"
                result["states"]["chexbert"]["edema"] = "unknown"
                return result
        executor = FailedGainExecutor()
        controller = LLMRepairController(observation(), tools(), budget_units=16, max_steps=4)
        with patch("tricompose_llm.local_qwen_bounded.TypedQwenPlanner.propose", side_effect=fake_model):
            result = controller.run(value, executor)
        self.assertEqual(result["terminal_reason"], "abstain")
        self.assertEqual(executor.count, 3)
        self.assertEqual(result["accepted_proxy_transitions"], 0)
        self.assertEqual(value.local_attempts, 3)
        self.assertEqual(value.programmatic_terminal_stops, 1)
        self.assertEqual(result["planner_calls"], 4)  # Conservative policy requests, NOT model calls.
        self.assertEqual(result["spent_units"], 14)
        self.assertIsNone(result["clinical_repair_success"])

    def test_guard_does_not_claim_clinical_acceptance(self):
        source = state(); source["tools"] = []
        decision = planner().propose(source)
        self.assertNotIn("clinical_acceptance", decision)
        self.assertEqual(decision["reason_code"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
