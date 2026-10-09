"""New frozen numeric planner with an authenticated, closed image guard.

Legacy planners stay immutable. Previous-image expert readouts are explicitly
not observations of future reports. This entry proposes requests only; it never
installs a generator, replaces an EHR, changes a scorer or certifies clinical truth.
"""
from __future__ import annotations

import hashlib
import json

from .contracts import (ContractError, decision_schema, decode_json, exact, opaque,
    require, validate_decision, validate_public_state)
from .image_evidence_guard import guarded_state, validate_guard
from .local_qwen_v2 import LocalQwenPlanner as TypedFrozenPlanner
from .planner import SYSTEM_PROMPT

VERSION = "tricompose-guard-aware-numeric-policy-v1"
GUARD_PROMPT = """The input envelope includes numeric observations and an image-evidence guard.
Scorer disagreement means unresolved attribution, NOT that the image is wrong.
The supplied tool menu has already applied this guard and budget feasibility.
expert_history refers to reports on a DIFFERENT cached image of the SAME EHR;
it is weak scheduling context, NOT the unseen report's score or an independent
vote about the current image. candidate_image_groups separates these images.
Only the current candidate describes the current trial branch. Do not attach an
old report to it. No generation backend is installed in this planning-only run:
an action proposes a deferred request requiring separate approval. Stop/abstain
is valid. Do not invent a verification tool or erase already-incurred costs.
"""


def validate_packet(packet):
    exact(packet, ("schema_version", "observation", "image_guard", "expert_history",
        "candidate_image_groups", "current_report_model_id", "generation_backend_installed"),
        "closed_guard_aware_packet_required")
    require(packet["schema_version"] == VERSION and packet["generation_backend_installed"] is False,
        "planning_only_packet_version_required")
    state, guard = packet["observation"], packet["image_guard"]
    validate_public_state(state); validate_guard(guard)
    require(len(state["evidence"]) == 5 and state["budget"] == {
        "limit_units": 3, "spent_units": 1, "planner_units_per_call": 1},
        "five_observations_and_declared_new_phase_budget_required")
    require(guarded_state(state, guard) == state, "image_veto_must_precede_fresh_planning")
    rows = {r["candidate_id"]: r for r in state["evidence"]}
    groups = packet["candidate_image_groups"]
    require(isinstance(groups, list) and len(groups) == len(rows), "complete_observed_image_groups_required")
    group_index = {}
    for group in groups:
        exact(group, ("candidate_id", "image_group_id"), "closed_image_group_required")
        opaque(group["candidate_id"], "c"); opaque(group["image_group_id"], "i")
        require(group["candidate_id"] in rows and group["candidate_id"] not in group_index,
            "unique_observed_image_group_required")
        group_index[group["candidate_id"]] = group["image_group_id"]
    history = packet["expert_history"]
    require(isinstance(history, list) and len(history) == 4, "four_cached_expert_observations_required")
    models, observed = set(), set()
    current_group = group_index[state["current_candidate_id"]]
    for item in history:
        exact(item, ("model_id", "candidate_id", "evidence_id"), "closed_expert_history_required")
        opaque(item["model_id"], "m")
        require(item["model_id"] not in models and item["candidate_id"] in rows
            and item["candidate_id"] not in observed
            and rows[item["candidate_id"]]["evidence_id"] == item["evidence_id"]
            and group_index[item["candidate_id"]] != current_group,
            "historical_expert_must_be_other_image_observation")
        models.add(item["model_id"]); observed.add(item["candidate_id"])
    opaque(packet["current_report_model_id"], "m")
    require(packet["current_report_model_id"] in models
        and observed == set(rows) - {state["current_candidate_id"]}
        and all(t["action"] == "regenerate_report" and t["model_id"] in models
            and t["model_id"] != packet["current_report_model_id"]
            and t["seed"] == 1 and t["cost_units"] == 2 for t in state["tools"]),
        "only_registered_untried_same_image_report_requests_allowed")
    return packet


def request_messages(packet):
    validate_packet(packet)
    schema = decision_schema(packet["observation"])
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT
        + "\n" + GUARD_PROMPT + "\nRequired JSON schema: " + json.dumps(schema)}]},
        {"role": "user", "content": [{"type": "text", "text":
            json.dumps(packet, sort_keys=True, allow_nan=False)}]}]


class GuardAwareQwenPlanner(TypedFrozenPlanner):
    """Same frozen checkpoint/loader; new typed numeric-envelope input."""

    def propose_guarded(self, packet):
        stage = "state_contract"
        self.local_attempts += 1
        try:
            messages = request_messages(packet)
            stage = "tokenize"
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            length = int(inputs["input_ids"].shape[-1])
            stage = "context_bound"
            require(length <= 8192, "bounded_local_planner_context")
            stage = "generate"
            self.generate_attempts += 1
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=384, do_sample=False)
            output_length = int(generated.shape[-1]) - length
            self.usage["input_tokens"] += length
            self.usage["output_tokens"] += output_length
            self.last_response_metadata = {"input_tokens": length, "output_tokens": output_length,
                "token_limit_reached": output_length >= 384}
            stage = "decode"
            require(output_length < 384, "complete_local_planner_output_required")
            response = self.processor.batch_decode(generated[:, length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            self.last_response_metadata["response_sha256"] = hashlib.sha256(response.encode()).hexdigest()
            stage = "json_contract"
            decision = decode_json(response)
            stage = "decision_contract"
            return validate_decision(decision, packet["observation"])
        except Exception:
            self.failure_counts[stage] += 1
            raise ContractError("guard_aware_planner_failed_or_invalid") from None

    def audit(self):
        return {**super().audit(), "interface_version": VERSION,
            "input_scope": "closed_numeric_observations_guard_and_other_image_expert_history",
            "programmatic_empty_menu_decisions_counted_as_llm": False}
