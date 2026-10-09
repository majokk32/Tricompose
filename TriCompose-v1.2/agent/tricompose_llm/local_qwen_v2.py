"""Typed-content correction; v1 source and failed run remain immutable.

Installed ProcessorMixin.apply_chat_template(tokenize=True) iterates content
blocks and indexes block['type']. Chat-completions string messages are not
valid at this local processor boundary. No model, scorer or policy changes.
"""
from __future__ import annotations

import hashlib
import json

from .contracts import ContractError, decode_json, decision_schema, require, validate_decision
from .local_qwen import LocalQwenPlanner as FrozenQwenLoader
from .planner import SYSTEM_PROMPT

INTERFACE_VERSION = "tricompose-local-qwen-typed-content-v2"
STAGES = ("state_contract", "tokenize", "context_bound", "generate", "decode", "json_contract", "decision_contract")


def request_messages(state):
    schema = decision_schema(state)  # Same closed, numeric-only contract.
    return [
        {"role": "system", "content": [{"type": "text", "text":
            SYSTEM_PROMPT + "\nRequired JSON schema: " + json.dumps(schema)}]},
        {"role": "user", "content": [{"type": "text", "text":
            json.dumps(state, sort_keys=True, allow_nan=False)}]},
    ]


class LocalQwenPlanner(FrozenQwenLoader):
    """Same audited frozen loader, same prompt, same strict decision validator."""

    def __init__(self, model_path):
        super().__init__(model_path)
        self.failure_counts = {key: 0 for key in STAGES}
        self.generate_attempts = 0
        self.last_response_metadata = None

    def propose(self, state):
        stage = "state_contract"
        self.local_attempts += 1
        try:
            messages = request_messages(state)
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
            text = self.processor.batch_decode(generated[:, length:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
            self.last_response_metadata["response_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            stage = "json_contract"
            value = decode_json(text)
            stage = "decision_contract"
            return validate_decision(value, state)
        except Exception:
            self.failure_counts[stage] += 1
            raise ContractError("local_planner_failed_or_invalid") from None

    def audit(self):
        return {**super().audit(), "interface_version": INTERFACE_VERSION,
                "failure_counts": dict(self.failure_counts), "generate_attempts": self.generate_attempts,
                "last_response_metadata": self.last_response_metadata}
