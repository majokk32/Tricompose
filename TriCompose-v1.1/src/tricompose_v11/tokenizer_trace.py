"""Observe tokenizer calls without changing the official inference arguments.

This module has no model dependencies. Payloads contain synthetic prompt text
and MUST only be persisted under the protected run directory, never logged.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager

from .cxr_contracts import canonical_json_sha256, read_json, require_inside, sha256_file


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_tokenizer_trace(candidate, protected_root):
    """Validate new observations without relabeling unobserved legacy outputs."""
    observation = candidate.get("tokenizer_input")
    if observation is None:
        return
    if observation.get("runtime_observed") is not True:
        raise ValueError("candidate tokenizer observation status is invalid")
    text_path = require_inside(observation["path"], protected_root, must_exist=True)
    trace_path = require_inside(observation["trace_path"], protected_root, must_exist=True)
    if sha256_file(text_path) != observation["sha256"] or sha256_file(trace_path) != observation["trace_sha256"]:
        raise ValueError("candidate tokenizer artifact hash mismatch")
    trace = read_json(trace_path)
    if trace.get("runtime_observed") is not True or trace.get("official_generation_arguments_changed") is not False:
        raise ValueError("candidate tokenizer trace status is invalid")
    if trace.get("supplied_prompt_sha256") != candidate["prompt_sha256"]:
        raise ValueError("tokenizer trace is bound to another supplied prompt")
    if trace.get("positive_tokenizer_text_sha256") != observation["sha256"] or text_sha256(trace["positive_tokenizer_text"]) != observation["sha256"]:
        raise ValueError("tokenizer trace text hash mismatch")
    if trace.get("positive_input_ids_sha256") != observation["input_ids_sha256"]:
        raise ValueError("tokenizer input ID hash mismatch")


def _rows(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("unsupported tokenizer output shape")
    if not value or isinstance(value[0], int):
        value = [value]
    rows = [list(row) for row in value]
    if any(any(not isinstance(token, int) for token in row) for row in rows):
        raise ValueError("tokenizer IDs must be integers")
    return rows


class ObservedTokenizer:
    """Forward calls/attributes unchanged; record tensor-tokenization boundaries."""

    def __init__(self, tokenizer):
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "events", [])

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __setattr__(self, name, value):
        # Official Sana sets padding_side; this must reach the real tokenizer.
        setattr(self._tokenizer, name, value)

    def __call__(self, *args, **kwargs):
        result = self._tokenizer(*args, **kwargs)
        if kwargs.get("return_tensors") != "pt":
            return result  # Adapter's unpadded length check, not encoder input.
        text = args[0] if args else kwargs.get("text")
        texts = [text] if isinstance(text, str) else list(text or [])
        if not texts or not all(isinstance(item, str) for item in texts):
            raise ValueError("expected text at the tokenizer boundary")
        ids = _rows(result["input_ids"])
        mask = _rows(result.get("attention_mask"))
        if len(ids) != len(texts) or (mask is not None and len(mask) != len(ids)):
            raise ValueError("tokenizer batch shape mismatch")
        if mask is not None and any(len(m) != len(i) for m, i in zip(mask, ids, strict=True)):
            raise ValueError("tokenizer attention-mask shape mismatch")
        self.events.append({
            "call_index": len(self.events), "texts": texts,
            "text_sha256": [text_sha256(item) for item in texts],
            "input_ids_sha256": [canonical_json_sha256({"ids": row}) for row in ids],
            "padded_token_count": [len(row) for row in ids],
            "attention_token_count": [sum(row) for row in mask] if mask is not None else [None] * len(ids),
            "padding": kwargs.get("padding"), "truncation": kwargs.get("truncation"),
            "max_length": kwargs.get("max_length"),
        })
        return result

    def positive_batch(self, supplied_prompts):
        # The three supported official pipelines tokenize positives first with
        # max-length padding. Record all calls so this identification is auditable.
        eligible = [e for e in self.events if e["padding"] == "max_length"]
        if not eligible or len(eligible[0]["texts"]) != len(supplied_prompts):
            raise ValueError("positive tokenizer call was not observed")
        event = eligible[0]
        for supplied, observed in zip(supplied_prompts, event["texts"], strict=True):
            normalized = supplied.lower().strip()
            if not normalized or not (observed == supplied or observed == normalized or observed.endswith(normalized)):
                raise ValueError("positive tokenizer text is not aligned with supplied prompts")
        return event

    def candidate_payload(self, index, supplied_prompts):
        positive = self.positive_batch(supplied_prompts)
        return {
            "schema_version": "tricompose-tokenizer-observation-v1",
            "runtime_observed": True,
            "observation_boundary": "tokenizer_call_not_text_encoder_hook",
            "positive_call_identification": "first_max_length_pt_call_with_aligned_prompt_suffix",
            "batch_index": index, "positive_call_index": positive["call_index"],
            "supplied_prompt_sha256": text_sha256(supplied_prompts[index]),
            "positive_tokenizer_text": positive["texts"][index],
            "positive_tokenizer_text_sha256": positive["text_sha256"][index],
            "positive_input_ids_sha256": positive["input_ids_sha256"][index],
            "padded_token_count": positive["padded_token_count"][index],
            "attention_token_count": positive["attention_token_count"][index],
            "pipeline_changed_text": positive["texts"][index] != supplied_prompts[index],
            "tensor_tokenizer_calls": self.events,
            "official_generation_arguments_changed": False,
        }


@contextmanager
def observe_pipeline_tokenizer(runtime):
    pipe = getattr(runtime, "pipe", None)
    tokenizer = getattr(pipe, "tokenizer", None)
    if pipe is None or tokenizer is None:
        raise ValueError("frozen runtime lacks a traceable pipeline tokenizer")
    if isinstance(tokenizer, ObservedTokenizer):
        raise ValueError("nested tokenizer observation is not supported")
    observer = ObservedTokenizer(tokenizer)
    # Avoid DiffusionPipeline.__setattr__: its module-registration side effects
    # would otherwise rewrite pipeline configuration for the temporary proxy.
    object.__setattr__(pipe, "tokenizer", observer)
    try:
        yield observer
    finally:
        object.__setattr__(pipe, "tokenizer", tokenizer)
