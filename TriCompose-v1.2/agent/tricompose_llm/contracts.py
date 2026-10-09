"""Closed public decision contract. Clinical snapshots stay local."""
from __future__ import annotations

import json
import math
import re

from . import VERSION

PUBLIC_SCHEMA = VERSION + "-numeric-state"
DECISION_SCHEMA = VERSION + "-decision"
ACTIONS = ("regenerate_report", "regenerate_cxr", "stop", "abstain")
REASONS = ("report_mismatch", "image_mismatch", "explore_alternative",
           "insufficient_evidence", "budget_limit", "no_further_action")
EDGES = ("ehr_cxr", "ehr_report", "cxr_report")
DECISION_FIELDS = {"schema_version", "step_id", "action", "target_id",
                   "evidence_ids", "reason_code"}


class ContractError(ValueError):
    """Fixed-code failures only; never echo model output or private inputs."""


def require(condition, code):
    if not condition:
        raise ContractError(code)


def exact(value, keys, code):
    require(isinstance(value, dict) and set(value) == set(keys), code)


def count(value, maximum=10000):
    require(type(value) is int and 0 <= value <= maximum, "invalid_count")


def opaque(value, prefix):
    require(isinstance(value, str) and re.fullmatch(prefix + r"[0-9]{4,6}", value),
            "invalid_rebased_id")


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate_json_key")
        obj[key] = value
    return obj


def decode_json(text, limit=32768):
    require(isinstance(text, str) and len(text.encode("utf-8")) <= limit,
            "bounded_json_required")
    try:
        return json.loads(text, object_pairs_hook=unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite_json")))
    except (ValueError, TypeError):
        raise ContractError("invalid_strict_json") from None


def validate_public_state(state):
    """Fail closed, including nested metadata and free-form string channels."""
    exact(state, ("schema_version", "step_id", "current_candidate_id", "evidence",
                  "tools", "budget", "history"), "closed_numeric_state_required")
    require(state["schema_version"] == PUBLIC_SCHEMA, "numeric_state_version")
    opaque(state["step_id"], "s")
    opaque(state["current_candidate_id"], "c")
    exact(state["budget"], ("limit_units", "spent_units", "planner_units_per_call"), "closed_budget")
    for value in state["budget"].values():
        count(value)
    require(state["budget"]["spent_units"] <= state["budget"]["limit_units"]
            and state["budget"]["planner_units_per_call"] > 0, "budget_bounds")
    require(isinstance(state["evidence"], list) and 1 <= len(state["evidence"]) <= 256,
            "bounded_observed_evidence_required")
    ids, candidates = set(), set()
    for row in state["evidence"]:
        exact(row, ("evidence_id", "candidate_id", "edges", "quality", "uncertainty"), "closed_evidence")
        opaque(row["evidence_id"], "e"); opaque(row["candidate_id"], "c")
        require(row["evidence_id"] not in ids and row["candidate_id"] not in candidates,
                "unique_observed_evidence")
        ids.add(row["evidence_id"]); candidates.add(row["candidate_id"])
        exact(row["edges"], EDGES, "closed_edges")
        for metrics in row["edges"].values():
            exact(metrics, ("known", "comparable", "supported", "opposed", "positive_supported"), "closed_edge_counts")
            for value in metrics.values(): count(value, 14)
            require(metrics["supported"] + metrics["opposed"] == metrics["comparable"]
                    <= metrics["known"] and metrics["positive_supported"] <= metrics["supported"],
                    "edge_count_algebra")
        exact(row["quality"], ("image_basic_valid", "report_structure", "artifact_failures"), "closed_quality")
        require(type(row["quality"]["image_basic_valid"]) is bool, "typed_validity")
        score = row["quality"]["report_structure"]
        require(score is None or type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                "finite_quality_or_null")
        count(row["quality"]["artifact_failures"])
        exact(row["uncertainty"], ("ehr_unknown", "image_unknown", "report_unknown",
                                   "ehr_uncertain", "image_uncertain", "report_uncertain"), "closed_uncertainty")
        for value in row["uncertainty"].values(): count(value, 14)
    require(state["current_candidate_id"] in candidates, "observed_current_candidate_required")
    require(isinstance(state["tools"], list) and len(state["tools"]) <= 256, "bounded_tool_menu")
    tool_ids = set()
    for tool in state["tools"]:
        exact(tool, ("tool_id", "action", "model_id", "seed", "cost_units"), "closed_tool")
        opaque(tool["tool_id"], "t"); opaque(tool["model_id"], "m")
        require(tool["tool_id"] not in tool_ids and tool["action"] in ACTIONS[:2], "unique_tool_action")
        tool_ids.add(tool["tool_id"])
        require(type(tool["seed"]) is int and 0 <= tool["seed"] < 2**63, "typed_seed")
        count(tool["cost_units"])
        require(tool["cost_units"] > 0 and tool["cost_units"] <= state["budget"]["limit_units"] - state["budget"]["spent_units"],
                "affordable_tool_required")
    require(isinstance(state["history"], list) and len(state["history"]) <= 128, "bounded_history")
    for item in state["history"]:
        exact(item, ("step_id", "action", "tool_id", "accepted_proxy_transition", "failed",
                     "resolved_count", "new_opposition_count", "lost_comparison_count"), "closed_history")
        opaque(item["step_id"], "s"); opaque(item["tool_id"], "t")
        require(item["action"] in ACTIONS[:2] and type(item["accepted_proxy_transition"]) is bool
                and type(item["failed"]) is bool, "typed_history")
        for key in ("resolved_count", "new_opposition_count", "lost_comparison_count"): count(item[key], 42)
    return state


def validate_decision(value, state):
    validate_public_state(state)
    exact(value, DECISION_FIELDS, "closed_decision_required")
    require(value["schema_version"] == DECISION_SCHEMA and value["step_id"] == state["step_id"],
            "current_step_decision_required")
    require(value["action"] in ACTIONS and value["reason_code"] in REASONS, "registered_action_reason")
    references = value["evidence_ids"]
    allowed = {row["evidence_id"] for row in state["evidence"]}
    require(isinstance(references, list) and 1 <= len(references) <= 8
            and all(isinstance(x, str) for x in references) and len(set(references)) == len(references)
            and set(references) <= allowed, "observed_evidence_references_required")
    if value["action"] in ACTIONS[:2]:
        matches = [t for t in state["tools"] if t["tool_id"] == value["target_id"]]
        require(len(matches) == 1 and matches[0]["action"] == value["action"], "feasible_registered_tool_required")
    else:
        require(value["target_id"] is None, "terminal_target_must_be_null")
    return value


def decision_schema(state):
    validate_public_state(state)
    properties = {
        "schema_version": {"type": "string", "enum": [DECISION_SCHEMA]},
        "step_id": {"type": "string", "enum": [state["step_id"]]},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "target_id": {"type": ["string", "null"], "enum": [None] + [t["tool_id"] for t in state["tools"]]},
        "evidence_ids": {"type": "array", "items": {"type": "string", "enum": [r["evidence_id"] for r in state["evidence"]]}},
        "reason_code": {"type": "string", "enum": list(REASONS)},
    }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
