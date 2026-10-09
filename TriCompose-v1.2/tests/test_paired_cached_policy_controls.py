"""Authored numeric snapshots only; no patient/model/API/GPU access."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from tricompose_llm.contracts import ContractError, validate_decision
from tricompose_llm.controller import LLMRepairController, ProbeRequest, tools_from_orders
from tricompose_llm.demo import observation
from tricompose_llm.paired_cache_comparison import (CountRulePolicy, NumericCacheExecutor,
    RANDOM_SEEDS, RandomAffordablePolicy, RecordedDecisionPolicy, aggregate, compare_case,
    readout, verify_recorded_result)
from tricompose_v12.probe_repair_v1 import digest


REPORTS = ["maira2", "cxrmate_single", "llavarad", "chexagent2"]
IMAGES = [("chexgenbench_sana", 0), ("roentgen_v2", 0)]


def fixture(*, ehr="positive"):
    grid = {}
    for model, seed in IMAGES:
        for report in REPORTS:
            grid[(model, seed, report)] = observation(len(grid), ehr=ehr,
                image_model=model, report_model=report, image="positive",
                report="negative" if report == "maira2" else "positive")
    tools = tools_from_orders(IMAGES, REPORTS)
    initial = grid[(*IMAGES[0], REPORTS[0])]
    return initial, grid, tools


def receipt(*, ehr="positive"):
    initial, grid, tools = fixture(ehr=ehr)
    result = LLMRepairController(initial, tools, budget_units=16, max_steps=4).run(
        CountRulePolicy(), NumericCacheExecutor(grid))
    sources = [{"request_index": i, "step_id": f"s{i:04d}", "source": "llm"}
               for i in range(result["planner_calls"])]
    return initial, grid, tools, result, sources


class CountRuleTests(unittest.TestCase):
    def test_contradiction_chooses_registered_report_tool(self):
        initial, _, tools = fixture()
        state = LLMRepairController(initial, tools).public_state(0)
        result = CountRulePolicy().propose(state)
        validate_decision(result, state)
        self.assertEqual(result["action"], "regenerate_report")
        self.assertEqual(result["reason_code"], "report_mismatch")

    def test_missing_coverage_explores_without_declaring_contradiction(self):
        initial, _, tools = fixture(ehr="unknown")
        initial = deepcopy(initial); initial["states"]["chexbert"]["edema"] = "unknown"
        state = LLMRepairController(initial, tools).public_state(0)
        self.assertEqual(state["evidence"][0]["edges"]["cxr_report"]["opposed"], 0)
        result = CountRulePolicy().propose(state)
        self.assertEqual(result["action"], "regenerate_report")
        self.assertEqual(result["reason_code"], "explore_alternative")

    def test_failed_report_feedback_changes_eligible_branch_only(self):
        _, _, tools = fixture()
        initial = observation(image="negative", report="positive")
        state = LLMRepairController(initial, tools).public_state(1)
        state["history"] = [{"step_id": "s0000", "action": "regenerate_report", "tool_id": "t0001",
            "accepted_proxy_transition": False, "failed": False, "resolved_count": 0,
            "new_opposition_count": 0, "lost_comparison_count": 0}]
        self.assertEqual(CountRulePolicy().propose(state)["action"], "regenerate_cxr")
        self.assertEqual(CountRulePolicy(feedback=False).propose(state)["action"], "regenerate_report")

    def test_empty_menu_abstains_and_does_not_mutate_state(self):
        initial, _, tools = fixture()
        state = LLMRepairController(initial, tools).public_state(0)
        state["tools"] = []; before = deepcopy(state)
        self.assertEqual(CountRulePolicy().propose(state)["action"], "abstain")
        self.assertEqual(state, before)

    def test_no_comparisons_is_not_positive_success(self):
        _, _, tools = fixture()
        initial = observation(image="unknown", report="unknown", ehr="unknown")
        state = LLMRepairController(initial, tools).public_state(0)
        self.assertEqual(CountRulePolicy().propose(state)["action"], "abstain")

    def test_extra_private_channel_rejected_before_policy(self):
        initial, _, tools = fixture()
        state = LLMRepairController(initial, tools).public_state(0)
        state["report_text"] = "authored_fixture_do_not_export"
        for policy in (CountRulePolicy(), RandomAffordablePolicy(0)):
            with self.assertRaises(ContractError): policy.propose(state)

    def test_seeded_random_is_repeatable_and_registered(self):
        initial, _, tools = fixture()
        state = LLMRepairController(initial, tools).public_state(0)
        left, right = RandomAffordablePolicy(2), RandomAffordablePolicy(2)
        for _ in range(8):
            a, b = left.propose(state), right.propose(state)
            self.assertEqual(a, b); validate_decision(a, state)

    def test_random_seeds_are_fixed_and_bounded(self):
        for seed in (-1, 5, True, "1"):
            with self.assertRaises(ContractError): RandomAffordablePolicy(seed)


class PairedReplayTests(unittest.TestCase):
    def test_recorded_replay_matches_full_receipt_without_llm(self):
        initial, grid, tools, result, _ = receipt()
        replayed = verify_recorded_result(initial, grid, tools, result, budget=16, max_steps=4)
        self.assertEqual(digest(replayed), digest(result))

    def test_receipt_event_credit_tampering_rejected(self):
        initial, grid, tools, result, _ = receipt()
        completed = next(event for event in result["trace"] if event["event"] == "tool_completed")
        completed["credit"]["quality_nonregression"] = False
        with self.assertRaises(ContractError):
            verify_recorded_result(initial, grid, tools, result, budget=16, max_steps=4)

    def test_recorded_numeric_state_tampering_rejected(self):
        initial, grid, tools, result, _ = receipt()
        validated = next(event for event in result["trace"] if event["event"] == "proposal_validated")
        validated["state"]["budget"]["spent_units"] += 1
        with self.assertRaises(ContractError):
            verify_recorded_result(initial, grid, tools, result, budget=16, max_steps=4)

    def test_budget_or_step_setting_mismatch_rejected(self):
        initial, grid, tools, result, _ = receipt()
        with self.assertRaises(ContractError):
            verify_recorded_result(initial, grid, tools, result, budget=8, max_steps=1)

    def test_replay_has_no_unrecorded_proposals(self):
        _, _, _, result, _ = receipt()
        policy = RecordedDecisionPolicy(result)
        state = policy.proposals[0]["state"]
        policy.position = len(policy.proposals)
        with self.assertRaises(ContractError): policy.propose(state)

    def test_executor_returns_only_requested_copy_with_parent_pin(self):
        initial, grid, _ = fixture()
        slot = (*IMAGES[0], REPORTS[1]); executor = NumericCacheExecutor(grid)
        request = ProbeRequest("s0000", "t0001", "regenerate_report", slot, digest(initial), 2)
        returned = executor.execute(request, initial)
        returned["states"]["ehr"]["edema"] = "unknown"
        self.assertEqual(grid[slot]["states"]["ehr"]["edema"], "positive")
        bad = ProbeRequest("s0000", "t0001", "regenerate_report", slot, "0"*64, 2)
        with self.assertRaises(ContractError): executor.execute(bad, initial)

    def test_all_controls_share_budget_and_immutable_ehr(self):
        initial, grid, tools, recorded, sources = receipt()
        before = deepcopy(grid)
        rows, outcomes = compare_case(0, initial, grid, tools, recorded, sources, budget=16, max_steps=4)
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(outcomes), 9)
        self.assertEqual(grid, before)
        for row in rows:
            self.assertLessEqual(row["simulated_units"], 16)
            self.assertEqual(row["actual_generator_calls"], 0)
            self.assertIsNone(row["clinical_repair_success"])
        for record in outcomes:
            self.assertEqual(record["result"]["selected_observation"]["lineage"]["ehr_sha256"],
                             initial["lineage"]["ehr_sha256"])
        self.assertEqual(rows[0]["simulated_units"], 4)
        self.assertEqual(rows[0]["cache_attempts"], 0)

    def test_guard_source_cannot_impersonate_llm_decision(self):
        initial, grid, tools, recorded, sources = receipt()
        sources[0]["source"] = "guard_empty_menu"
        with self.assertRaises(ContractError):
            compare_case(0, initial, grid, tools, recorded, sources, budget=16, max_steps=4)


class AggregationTests(unittest.TestCase):
    def trials(self):
        initial, grid, tools, recorded, sources = receipt(ehr="unknown")
        return compare_case(0, initial, grid, tools, recorded, sources, budget=16, max_steps=4)[0]

    def test_unknown_ehr_ratios_stay_na_with_zero_available_cases(self):
        _, table = aggregate(self.trials())
        for row in table:
            self.assertEqual(row["ehr_cxr_known_mean"], 0)
            self.assertIsNone(row["ehr_cxr_coverage_mean"])
            self.assertEqual(row["ehr_cxr_coverage_available_cases"], 0)
            self.assertIsNone(row["ehr_report_support_over_known_mean"])

    def test_random_replicates_are_not_counted_as_extra_patients(self):
        _, table = aggregate(self.trials())
        random_row = next(row for row in table if row["method"] == "random_affordable")
        self.assertEqual(random_row["paired_ehr_cases"], 1)
        self.assertEqual(random_row["cxr_report_known_available_cases"], 1)

    def test_incomplete_or_duplicate_random_seeds_rejected(self):
        trials = self.trials()
        incomplete = [row for row in trials if not (row["method"] == "random_affordable" and row["random_seed"] == 4)]
        with self.assertRaises(ContractError): aggregate(incomplete)
        duplicate = trials + [deepcopy(next(row for row in trials if row["random_seed"] == 0))]
        with self.assertRaises(ContractError): aggregate(duplicate)

    def test_all_methods_require_identical_case_membership(self):
        trials = self.trials(); extra = deepcopy(trials[0]); extra["case_index"] = 1
        with self.assertRaises(ContractError): aggregate(trials+[extra])

    def test_na_random_replica_does_not_become_a_success_score(self):
        trials = self.trials()
        next(row for row in trials if row["random_seed"] == 0)["metrics"]["report_structure"] = None
        per_case, table = aggregate(trials)
        random_row = next(row for row in table if row["method"] == "random_affordable")
        self.assertIsNone(random_row["report_structure_mean"])
        self.assertEqual(random_row["report_structure_available_cases"], 0)

    def test_no_global_scalar_score_or_clinical_certification(self):
        _, table = aggregate(self.trials())
        for row in table:
            self.assertIsNone(row["clinical_repair_success"])
            self.assertNotIn("overall_score", row)


if __name__ == "__main__":
    unittest.main()
