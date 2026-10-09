"""Paired numeric-cache controls; no models, images, reports or clinical gold.

Keep the consumed pilot/controller/scorers unchanged. Controls see exactly
the same closed numeric state as Qwen, never unrequested candidate scores.
Both rule variants and random share the pilot's simulated dispatch charge;
this is a budget convention, not equal measured CPU/GPU cost.
"""
from copy import deepcopy
import random

from tricompose_v12.probe_repair_v1 import digest, edges, validate_snapshot
from .contracts import DECISION_SCHEMA, require, validate_decision, validate_public_state
from .controller import LLMRepairController

VERSION = "tricompose-paired-cached-policy-controls-v1"
METHODS = ("fixed_path", "rule_count_feedback", "rule_count_report_first",
           "random_affordable", "local_qwen_recorded")
RANDOM_SEEDS = (0, 1, 2, 3, 4)


def current_evidence(state):
    validate_public_state(state)
    return next(row for row in state["evidence"]
                if row["candidate_id"] == state["current_candidate_id"])


def decision(state, tool=None, *, reason="explore_alternative", terminal="abstain"):
    return validate_decision({"schema_version": DECISION_SCHEMA,
        "step_id": state["step_id"], "action": tool["action"] if tool else terminal,
        "target_id": tool["tool_id"] if tool else None,
        "evidence_ids": [current_evidence(state)["evidence_id"]],
        "reason_code": reason}, state)


class CountRulePolicy:
    """Predeclared count-only rule; coverage gaps motivate exploration, not guilt.

    No all-label clinical requirement: a missing comparison is not a negative
    or a contradiction. The immutable local acceptance gate still decides
    whether a completed alternative can replace the current candidate.
    """
    def __init__(self, *, feedback=True):
        require(type(feedback) is bool, "explicit_rule_variant")
        self.feedback = feedback

    def propose(self, state):
        row = current_evidence(state)
        if not state["tools"]:
            return decision(state, reason="insufficient_evidence")
        edge, quality = row["edges"], row["quality"]
        image_need = not quality["image_basic_valid"] or bool(edge["ehr_cxr"]["opposed"])
        report_opposed = any(edge[key]["opposed"] for key in ("ehr_report", "cxr_report"))
        report_gap = any(edge[key]["known"] > edge[key]["comparable"]
                         for key in ("ehr_report", "cxr_report"))
        report_need = report_opposed or report_gap or quality["artifact_failures"] > 0
        reports = [tool for tool in state["tools"] if tool["action"] == "regenerate_report"]
        images = [tool for tool in state["tools"] if tool["action"] == "regenerate_cxr"]
        failed_report = bool(state["history"]
            and state["history"][-1]["action"] == "regenerate_report"
            and not state["history"][-1]["accepted_proxy_transition"])
        use_image = image_need and images and (not report_need or not reports
            or self.feedback and failed_report)
        if use_image:
            return decision(state, images[0], reason="image_mismatch")
        if report_need and reports:
            return decision(state, reports[0],
                reason="report_mismatch" if report_opposed else "explore_alternative")
        available = any(values["comparable"] for values in edge.values())
        terminal = "stop" if available and not image_need and not report_need else "abstain"
        return decision(state, reason="no_further_action" if terminal == "stop"
                        else "insufficient_evidence", terminal=terminal)


class RandomAffordablePolicy:
    """Seeded action selection with the SAME strict completion/rollback guard."""
    def __init__(self, seed):
        require(type(seed) is int and seed in RANDOM_SEEDS, "predeclared_control_seed")
        self.random = random.Random(seed)

    def propose(self, state):
        validate_public_state(state)
        tool = self.random.choice(state["tools"]) if state["tools"] else None
        return decision(state, tool, reason="explore_alternative" if tool else "insufficient_evidence")


class NumericCacheExecutor:
    """Requested snapshots only. Never interpret an acquisition as generation."""
    mode = "cache_replay"

    def __init__(self, grid):
        self._grid = grid

    def execute(self, request, current):
        require(request.parent_sha256 == digest(current), "cache_parent_changed")
        require(request.slot in self._grid, "requested_cache_slot_unavailable")
        return deepcopy(self._grid[request.slot])


class RecordedDecisionPolicy:
    """Verify an existing LLM receipt by replay, never synthesize an LLM call."""
    def __init__(self, result):
        self.proposals = [event for event in result["trace"]
                          if event["event"] == "proposal_validated"]
        require(len(self.proposals) == result["planner_calls"], "complete_valid_pilot_decisions_required")
        self.position = 0

    def propose(self, state):
        require(self.position < len(self.proposals), "no_unrecorded_llm_decision")
        record = self.proposals[self.position]
        require(state == record["state"], "exact_recorded_numeric_state_required")
        self.position += 1
        return deepcopy(record["decision"])


def verify_recorded_result(initial, grid, tools, recorded, *, budget, max_steps):
    policy = RecordedDecisionPolicy(recorded)
    replay = LLMRepairController(initial, tools, budget_units=budget, max_steps=max_steps).run(
        policy, NumericCacheExecutor(grid))
    require(policy.position == len(policy.proposals) and digest(replay) == digest(recorded),
            "pilot_replay_must_match_every_event_credit_and_selection")
    return replay


def readout(value):
    validate_snapshot(value)
    result = {"image_basic_valid": int(value["quality"]["cxr_basic_validity_pass"]),
              "report_structure": value["quality"]["report_structure_quality_score_0_1"],
              "artifact_gate_failures": value["artifact_gate_failures"]}
    for name, edge in edges(value).items():
        known, comparable = len(edge["known"]), len(edge["comparable"])
        for key in ("known", "comparable", "support", "positive_support", "opposition"):
            result[name+"_"+key] = len(edge[key])
        result[name+"_coverage"] = comparable / known if known else None
        result[name+"_support_over_known"] = len(edge["support"]) / known if known else None
        result[name+"_opposition_over_comparable"] = len(edge["opposition"]) / comparable if comparable else None
    return result


def compare_case(index, initial, grid, tools, recorded, proposal_sources, *, budget, max_steps):
    """One matched case, five methods, random replicas averaged within case."""
    require(type(index) is int and index >= 0, "opaque_integer_case_index")
    verified = verify_recorded_result(initial, grid, tools, recorded, budget=budget, max_steps=max_steps)
    require(len(proposal_sources) == recorded["planner_calls"], "one_source_per_policy_request")
    for position, source in enumerate(proposal_sources):
        require(source["source"] in ("llm", "guard_empty_menu")
                and source["step_id"] == f"s{position:04d}", "recorded_policy_sources_required")
        if source["source"] == "guard_empty_menu":
            proposal = RecordedDecisionPolicy(recorded).proposals[position]
            require(not proposal["state"]["tools"] and proposal["decision"]["action"] == "abstain",
                    "guard_stop_must_precede_model_with_empty_menu")
    trials = []
    outcomes = []

    def add(method, seed, result, local_attempts=0):
        selected = result["selected_observation"]
        require(selected["lineage"]["ehr_sha256"] == initial["lineage"]["ehr_sha256"]
                and selected["states"]["ehr"] == initial["states"]["ehr"]
                and selected["ehr_sources"] == initial["ehr_sources"], "fixed_ehr_control_required")
        require(result["spent_units"] <= budget, "paired_budget_bound")
        trials.append({"case_index": index, "method": method, "random_seed": seed,
            "terminal_reason": result["terminal_reason"], "simulated_units": result["spent_units"],
            "policy_requests": result["planner_calls"], "cache_attempts": result["tool_attempts"],
            "accepted_proxy_transitions": result["accepted_proxy_transitions"],
            "failed_cache_attempts": sum(row["failed"] for row in result.get("history", [])),
            "recorded_local_llm_attempts": local_attempts,
            "selected_observation_sha256": digest(selected), "metrics": readout(selected),
            "clinical_repair_success": None, "actual_generator_calls": 0})
        outcomes.append({"case_index": index, "method": method, "random_seed": seed,
                         "result": result})

    add("fixed_path", None, {"selected_observation": deepcopy(initial), "terminal_reason": "fixed_path",
        "spent_units": 4, "planner_calls": 0, "tool_attempts": 0, "accepted_proxy_transitions": 0})
    for method, planner in (("rule_count_feedback", CountRulePolicy()),
                            ("rule_count_report_first", CountRulePolicy(feedback=False))):
        result = LLMRepairController(initial, tools, budget_units=budget, max_steps=max_steps).run(
            planner, NumericCacheExecutor(grid))
        add(method, None, result)
    for seed in RANDOM_SEEDS:
        result = LLMRepairController(initial, tools, budget_units=budget, max_steps=max_steps).run(
            RandomAffordablePolicy(seed), NumericCacheExecutor(grid))
        add("random_affordable", seed, result)
    add("local_qwen_recorded", None, verified,
        sum(source["source"] == "llm" for source in proposal_sources))
    return trials, outcomes


def aggregate(trials):
    """Cases are the statistical unit, not labels, actions or random replicas."""
    require(bool(trials), "nonempty_paired_controls")
    metric_keys = tuple(trials[0]["metrics"])
    count_keys = ("simulated_units", "policy_requests", "cache_attempts",
                  "accepted_proxy_transitions", "failed_cache_attempts", "recorded_local_llm_attempts")
    grouped = {}
    for trial in trials:
        require(trial["method"] in METHODS and tuple(trial["metrics"]) == metric_keys,
                "same_frozen_readout_for_all_methods")
        grouped.setdefault((trial["case_index"], trial["method"]), []).append(trial)
    per_case = []
    for (index, method), rows in sorted(grouped.items()):
        require(sorted(row["random_seed"] for row in rows) == list(RANDOM_SEEDS)
                if method == "random_affordable" else len(rows) == 1 and rows[0]["random_seed"] is None,
                "complete_predeclared_control_replicas")
        value = {"case_index": index, "method": method, "replicates": len(rows)}
        for key in count_keys:
            value[key] = sum(row[key] for row in rows) / len(rows)
        for key in metric_keys:
            numbers = [row["metrics"][key] for row in rows]
            value[key] = sum(numbers) / len(numbers) if all(x is not None for x in numbers) else None
        per_case.append(value)
    all_indices = {trial["case_index"] for trial in trials}
    table = []
    for method in METHODS:
        rows = [row for row in per_case if row["method"] == method]
        require({row["case_index"] for row in rows} == all_indices, "same_cases_for_every_method")
        value = {"method": method, "paired_ehr_cases": len(rows), "clinical_repair_success": None}
        for key in (*count_keys, *metric_keys):
            numbers = [row[key] for row in rows if row[key] is not None]
            value[key+"_mean"] = sum(numbers) / len(numbers) if numbers else None
            value[key+"_available_cases"] = len(numbers)
        table.append(value)
    return per_case, table
