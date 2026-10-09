"""Versioned numeric-only planner input for an observed same-image report pair.

No historical other-image expert scores, clinical prose, endpoint similarities,
new labels, or predictions of unseen reports. The existing frozen loader and
acceptance/dispatch guards remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math

from .contracts import (ContractError, DECISION_SCHEMA, count, decision_schema,
    decode_json, exact, opaque, require, validate_decision, validate_public_state)
from .image_evidence_guard import guarded_state, validate_guard
from .local_qwen_v2 import LocalQwenPlanner as TypedFrozenPlanner
from .planner import SYSTEM_PROMPT

VERSION = "tricompose-current-image-numeric-policy-v1"
RISKS = ("empty", "generic_report", "unsupported_temporal_comparison_language")
EFFECTS = ("image_positive_support", "ehr_direct_support", "image_comparable",
    "ehr_comparable", "image_opposition", "ehr_opposition")
PROMPT = """Both observations describe reports actually generated on ONE current
synthetic image with ONE fixed EHR. No other-image reports are supplied. c0000
is the retained report; c0001 is an observed rejected replacement, not a future
prediction. observed_reports adds measured common risk flags and repetition.
rejected_transition gives count-only feedback from the unchanged acceptance
gate. Higher image/report support does not justify losing EHR support, removing
comparisons, silencing conflict, or adding unsupported temporal language. These
counts are uncalibrated proxies, not clinical truth. Scorer disagreement means
unresolved image attribution; do not infer image fault from report voting. The
remaining tools are untried report experts on this exact image. Their future
scores are UNKNOWN. You may propose one affordable request or abstain/stop;
no generator is installed. A request requires separate approval, and stop is
not clinical acceptance. Budgets are abstract attempt units, not GPU seconds;
historical failed/rejected calls remain sunk costs. BioViL and other endpoint
scores are withheld. Return only the closed decision JSON, no free-form text.
"""


def validate_packet(packet):
    exact(packet, ("schema_version", "observation", "image_guard", "observed_reports",
        "rejected_transition", "generation_backend_installed"), "closed_current_image_packet_required")
    require(packet["schema_version"] == VERSION and packet["generation_backend_installed"] is False,
        "current_image_planning_only_required")
    state, guard = packet["observation"], packet["image_guard"]
    validate_public_state(state); validate_guard(guard)
    require(state["step_id"] == "s0000" and state["current_candidate_id"] == "c0000"
        and state["budget"] == {"limit_units": 3, "spent_units": 1, "planner_units_per_call": 1}
        and state["history"] == [] and len(state["evidence"]) == 2
        and [(r["candidate_id"], r["evidence_id"]) for r in state["evidence"]]
            == [("c0000", "e0000"), ("c0001", "e0001")]
        and guarded_state(state, guard) == state, "two_current_observations_and_new_phase_budget_required")
    a, b = state["evidence"]
    require(a["edges"]["ehr_cxr"] == b["edges"]["ehr_cxr"]
        and all(a["uncertainty"][k] == b["uncertainty"][k] for k in
            ("ehr_unknown", "ehr_uncertain", "image_unknown", "image_uncertain"))
        and a["edges"]["ehr_report"]["known"] == b["edges"]["ehr_report"]["known"]
        and a["edges"]["cxr_report"]["known"] == b["edges"]["cxr_report"]["known"],
        "same_fixed_ehr_and_image_numeric_scope_required")
    reports = packet["observed_reports"]
    require(isinstance(reports, list) and len(reports) == 2, "two_observed_report_risk_rows_required")
    models = set()
    for row, evidence in zip(reports, state["evidence"], strict=True):
        exact(row, ("candidate_id", "evidence_id", "model_id", "risk_flags",
            "repeated_sentence_count", "repeated_4gram_ratio"), "closed_report_risk_row_required")
        opaque(row["model_id"], "m")
        require(row["model_id"] in ("m0000", "m0001") and row["model_id"] not in models
            and row["candidate_id"] == evidence["candidate_id"] and row["evidence_id"] == evidence["evidence_id"],
            "bound_observed_report_expert_required")
        models.add(row["model_id"])
        exact(row["risk_flags"], RISKS, "closed_report_risk_flags_required")
        require(all(type(v) is bool for v in row["risk_flags"].values()), "typed_report_risk_flags_required")
        count(row["repeated_sentence_count"])
        value = row["repeated_4gram_ratio"]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "finite_repetition_ratio_required")
        require(not row["risk_flags"]["empty"] or evidence["quality"]["report_structure"] == 0,
            "empty_report_cannot_pass_structure")
    require(len(state["tools"]) == 2 and [(t["tool_id"], t["model_id"]) for t in state["tools"]]
        == [("t0102", "m0002"), ("t0103", "m0003")]
        and all(t["action"] == "regenerate_report" and t["seed"] == 1 and t["cost_units"] == 2
            for t in state["tools"]), "exact_two_untried_same_image_experts_required")
    feedback = packet["rejected_transition"]
    exact(feedback, ("baseline_candidate_id", "proposed_candidate_id", "accepted_proxy_transition",
        "quality_no_worse", "duplicate", "lost", "gained", "new_opposition", "removed_opposition",
        "silenced"), "closed_rejected_transition_required")
    require(feedback["baseline_candidate_id"] == "c0000" and feedback["proposed_candidate_id"] == "c0001"
        and feedback["accepted_proxy_transition"] is False
        and type(feedback["quality_no_worse"]) is bool and type(feedback["duplicate"]) is bool,
        "actual_rejected_baseline_proposal_required")
    groups = {"lost": EFFECTS[:4], "gained": EFFECTS[:4], "new_opposition": EFFECTS[4:],
        "removed_opposition": EFFECTS[4:], "silenced": ("image", "ehr")}
    for name, fields in groups.items():
        exact(feedback[name], fields, "closed_transition_count_group_required")
        for value in feedback[name].values(): count(value, 14)
    for key, edge, metric in (("image_positive_support", "cxr_report", "positive_supported"),
            ("ehr_direct_support", "ehr_report", "supported"),
            ("image_comparable", "cxr_report", "comparable"), ("ehr_comparable", "ehr_report", "comparable")):
        before, after = a["edges"][edge][metric], b["edges"][edge][metric]
        require(feedback["lost"][key] <= before and feedback["gained"][key] <= after
            and feedback["gained"][key] - feedback["lost"][key] == after - before,
            "transition_support_coverage_count_algebra")
    for key, edge in (("image_opposition", "cxr_report"), ("ehr_opposition", "ehr_report")):
        before, after = a["edges"][edge]["opposed"], b["edges"][edge]["opposed"]
        require(feedback["removed_opposition"][key] <= before and feedback["new_opposition"][key] <= after
            and feedback["new_opposition"][key] - feedback["removed_opposition"][key] == after - before,
            "transition_opposition_count_algebra")
        prefix = key.split("_")[0]
        require(feedback["silenced"][prefix] <= min(feedback["removed_opposition"][key],
            feedback["lost"][prefix + "_comparable"]), "silenced_count_bound")
    ra, rb = reports
    quality = (all(r["quality"]["report_structure"] == 1 for r in (a, b))
        and all(not rb["risk_flags"][k] or ra["risk_flags"][k] for k in RISKS)
        and rb["repeated_sentence_count"] <= ra["repeated_sentence_count"]
        and rb["repeated_4gram_ratio"] <= ra["repeated_4gram_ratio"])
    require(feedback["quality_no_worse"] is quality, "transition_quality_projection_changed")
    strict = (not any(feedback["lost"].values()) and not any(feedback["new_opposition"].values())
        and (any(feedback["gained"].values()) or any(feedback["removed_opposition"].values())))
    require(not (strict and quality and not feedback["duplicate"]), "rejected_transition_cannot_be_gate_pass")
    return packet


def request_messages(packet):
    validate_packet(packet)
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT + "\n" + PROMPT
        + "\nRequired JSON schema: " + json.dumps(decision_schema(packet["observation"]))}]},
        {"role": "user", "content": [{"type": "text", "text": json.dumps(packet, sort_keys=True, allow_nan=False)}]}]


def rule_decision(packet):
    """Fixed priority, same evidence/action allowance; zero actual LLM calls.

    Uses no unseen-expert quality prior. Numeric budget includes a common reserved
    planning allowance; that allowance is NOT an incurred rule model call.
    """
    validate_packet(packet)
    state = packet["observation"]; current = state["evidence"][0]
    if (not packet["image_guard"]["counts"]["known_ehr"]
            or not current["quality"]["image_basic_valid"] or current["quality"]["artifact_failures"]):
        action, target, reason = "abstain", None, "insufficient_evidence"
    elif (any(v["opposed"] or v["comparable"] < v["known"] for v in current["edges"].values())
            or current["quality"]["report_structure"] != 1
            or any(packet["observed_reports"][0]["risk_flags"].values())):
        action, target, reason = "regenerate_report", state["tools"][0]["tool_id"], "explore_alternative"
    else:
        action, target, reason = "stop", None, "no_further_action"
    return validate_decision({"schema_version": DECISION_SCHEMA, "step_id": state["step_id"],
        "action": action, "target_id": target, "evidence_ids": ["e0000", "e0001"], "reason_code": reason}, state)


class CurrentImageQwenPlanner(TypedFrozenPlanner):
    def propose_current(self, packet):
        stage = "state_contract"; self.local_attempts += 1
        try:
            messages = request_messages(packet)
            stage = "tokenize"
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            length = int(inputs["input_ids"].shape[-1])
            stage = "context_bound"; require(length <= 8192, "bounded_local_planner_context")
            stage = "generate"; self.generate_attempts += 1
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=384, do_sample=False)
            output_length = int(generated.shape[-1]) - length
            self.usage["input_tokens"] += length; self.usage["output_tokens"] += output_length
            self.last_response_metadata = {"input_tokens": length, "output_tokens": output_length,
                "token_limit_reached": output_length >= 384}
            stage = "decode"; require(output_length < 384, "complete_local_planner_output_required")
            response = self.processor.batch_decode(generated[:, length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            self.last_response_metadata["response_sha256"] = hashlib.sha256(response.encode()).hexdigest()
            stage = "json_contract"; decision = decode_json(response)
            stage = "decision_contract"
            return validate_decision(decision, packet["observation"])
        except Exception:
            self.failure_counts[stage] += 1
            raise ContractError("current_image_planner_failed_or_invalid") from None

    def audit(self):
        return {**super().audit(), "interface_version": VERSION,
            "input_scope": "two_current_image_numeric_reports_guard_risks_and_rejected_gate_counts",
            "historical_other_image_experts_supplied": False, "endpoint_scores_supplied": False}
