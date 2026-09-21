"""Call an OpenAI-compatible LLM with a sanitized TriCompose agent state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from tricompose.agent.report_selector import choose_action
from tricompose.privacy import (
    create_private_stage_dir,
    require_private_file,
    sha256_file,
    write_private_json,
)


PRODUCER_VERSION = "1.0.0"
PROMPT_VERSION = "routing_json_v1"
OUTPUT_SCHEMA_VERSION = "tricompose.llm_agent_decision.v1"
SOURCE_SCHEMA_VERSION = "tricompose.agent_decision.v1"
ALLOWED_ACTIONS = {
    "select",
    "verify_more",
    "regenerate_report",
    "stop",
}
SYSTEM_PROMPT = """You are the routing policy for a synthetic medical-data
composition system. You receive only sanitized candidate IDs, numeric scores,
thresholds, cost metadata, and allowed actions. Never infer missing clinical
facts. Choose exactly one allowed next action.

Return one JSON object with:
- action: select, verify_more, regenerate_report, or stop
- candidate_id: required only for select or regenerate_report
- reason_code: a short lowercase snake_case code

Do not return Markdown or any text outside the JSON object."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--demo-state", action="store_true")
    state_group.add_argument("--agent-decision")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TRICOMPOSE_LLM_BASE_URL"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TRICOMPOSE_LLM_MODEL"),
    )
    parser.add_argument(
        "--api-key-env",
        default="TRICOMPOSE_LLM_API_KEY",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", required=True)
    return parser


def demo_state() -> dict[str, object]:
    """Return a patient-free state for testing an API connection."""
    return {
        "state_id": "synthetic_demo",
        "candidate_scores": {
            "candidate_a": 0.72,
            "candidate_b": 0.51,
        },
        "peer_metrics": {
            "report_pair_bleu_1": 0.18,
            "report_pair_rouge_l": 0.22,
        },
        "policy": {
            "minimum_score": 0.55,
            "minimum_margin": 0.10,
        },
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "privacy": "synthetic numeric demo; no patient data",
    }


def sanitize_agent_state(payload: dict[str, Any]) -> dict[str, object]:
    """Extract only the allowlisted numeric state sent to an LLM."""
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported source agent decision schema")
    evidence = payload.get("evidence")
    policy = payload.get("policy")
    if not isinstance(evidence, dict) or not isinstance(policy, dict):
        raise TypeError("agent evidence or policy is missing")
    candidate_scores = evidence.get("candidate_scores")
    peer_metrics = evidence.get("peer_metrics")
    if not isinstance(candidate_scores, dict) or not isinstance(
        peer_metrics,
        dict,
    ):
        raise TypeError("agent numeric evidence is missing")

    clean_candidate_scores: dict[str, float] = {}
    for candidate_id, value in candidate_scores.items():
        if (
            not isinstance(candidate_id, str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
                candidate_id,
            )
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("invalid candidate score state")
        clean_candidate_scores[candidate_id] = float(value)

    clean_peer_metrics: dict[str, float] = {}
    for metric, value in peer_metrics.items():
        if (
            not isinstance(metric, str)
            or not re.fullmatch(r"[a-z0-9_]{1,80}", metric)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("invalid peer metric state")
        clean_peer_metrics[metric] = float(value)

    minimum_score = policy.get("minimum_score")
    minimum_margin = policy.get("minimum_margin")
    if (
        not isinstance(minimum_score, (int, float))
        or isinstance(minimum_score, bool)
        or not 0.0 <= float(minimum_score) <= 1.0
        or not isinstance(minimum_margin, (int, float))
        or isinstance(minimum_margin, bool)
        or not 0.0 <= float(minimum_margin) <= 1.0
    ):
        raise ValueError("invalid selector thresholds")

    return {
        "state_id": str(payload.get("run_id", "protected_run")),
        "candidate_scores": dict(sorted(clean_candidate_scores.items())),
        "peer_metrics": dict(sorted(clean_peer_metrics.items())),
        "policy": {
            "minimum_score": float(minimum_score),
            "minimum_margin": float(minimum_margin),
        },
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "privacy": (
            "allowlisted numeric state only; no EHR, image, report text, "
            "patient identifier, artifact path, or hash"
        ),
    }


def _chat_completions_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid LLM base URL")
    normalized = base_url.rstrip("/")
    return (
        normalized
        if normalized.endswith("/chat/completions")
        else normalized + "/chat/completions"
    )


def _parse_llm_decision(content: str) -> dict[str, str]:
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace < 0 or last_brace <= first_brace:
        raise ValueError("LLM response did not contain a JSON object")
    payload = json.loads(content[first_brace : last_brace + 1])
    if not isinstance(payload, dict):
        raise TypeError("LLM decision must be a JSON object")
    action = payload.get("action")
    candidate_id = payload.get("candidate_id")
    reason_code = payload.get("reason_code")
    if action not in ALLOWED_ACTIONS:
        raise ValueError("LLM returned an unsupported action")
    if not isinstance(reason_code, str) or not re.fullmatch(
        r"[a-z0-9_]{1,80}",
        reason_code,
    ):
        raise ValueError("LLM returned an invalid reason code")
    decision = {
        "action": action,
        "reason_code": reason_code,
    }
    if candidate_id is not None:
        if not isinstance(candidate_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
            candidate_id,
        ):
            raise ValueError("LLM returned an invalid candidate ID")
        decision["candidate_id"] = candidate_id
    return decision


def guard_llm_decision(
    llm_decision: dict[str, str],
    state: dict[str, object],
) -> dict[str, object]:
    """Enforce candidate membership and the deterministic selection gate."""
    candidate_scores = state.get("candidate_scores")
    policy = state.get("policy")
    if not isinstance(candidate_scores, dict) or not isinstance(policy, dict):
        raise TypeError("invalid sanitized state")
    deterministic = choose_action(
        candidate_scores,
        minimum_score=float(policy["minimum_score"]),
        minimum_margin=float(policy["minimum_margin"]),
        suggested_next_verifier="independent_cxr_report_fact_scorer",
    )
    action = llm_decision["action"]
    candidate_id = llm_decision.get("candidate_id")
    if action in {"select", "regenerate_report"} and candidate_id not in candidate_scores:
        return {
            "action": "verify_more",
            "status": "guard_override",
            "reason_code": "invalid_candidate_id",
        }
    if action == "select" and (
        deterministic["action"] != "select"
        or candidate_id != deterministic.get("selected_candidate_id")
    ):
        return {
            "action": "verify_more",
            "status": "guard_override",
            "reason_code": "selection_gate_not_satisfied",
        }
    return {
        **llm_decision,
        "status": "accepted",
    }


def _call_api(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    state: dict[str, object],
    timeout_seconds: float,
) -> tuple[dict[str, str], dict[str, int]]:
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(state, sort_keys=True),
            },
        ],
        "temperature": 0,
        "max_tokens": 120,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(request_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout_seconds,
    ) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    choices = response_payload.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("message"), dict)
        or not isinstance(choices[0]["message"].get("content"), str)
    ):
        raise TypeError("invalid chat-completions response")
    usage_payload = response_payload.get("usage")
    usage = {
        key: int(value)
        for key, value in (
            usage_payload.items()
            if isinstance(usage_payload, dict)
            else []
        )
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, int)
    }
    return _parse_llm_decision(choices[0]["message"]["content"]), usage


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)
    stage_dir = create_private_stage_dir(args.output_dir)
    started = time.monotonic()
    try:
        if not isinstance(args.base_url, str) or not args.base_url:
            raise ValueError("LLM base URL is not configured")
        if not isinstance(args.model, str) or not args.model:
            raise ValueError("LLM model is not configured")
        if not re.fullmatch(r"TRICOMPOSE_[A-Z0-9_]{1,64}", args.api_key_env):
            raise ValueError("invalid API-key environment variable name")
        if args.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        state = (
            demo_state()
            if args.demo_state
            else sanitize_agent_state(
                json.loads(
                    require_private_file(args.agent_decision).read_text(
                        encoding="utf-8"
                    )
                )
            )
        )
        api_key = os.environ.get(args.api_key_env)
        llm_decision, usage = _call_api(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            state=state,
            timeout_seconds=args.timeout_seconds,
        )
        guarded_decision = guard_llm_decision(llm_decision, state)
        state_hash = hashlib.sha256(
            json.dumps(state, sort_keys=True).encode("utf-8")
        ).hexdigest()
        output_path = write_private_json(
            stage_dir / "llm_agent_decision.json",
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "producer": {
                    "name": "tricompose_openai_compatible_policy",
                    "version": PRODUCER_VERSION,
                    "prompt_version": PROMPT_VERSION,
                },
                "backend": {
                    "model": args.model,
                    "protocol": "openai_compatible_chat_completions",
                },
                "input": {
                    "state_id": state["state_id"],
                    "sanitized_state_sha256": state_hash,
                    "demo": bool(args.demo_state),
                },
                "llm_decision": llm_decision,
                "guarded_decision": guarded_decision,
                "usage": usage,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )
        print(
            json.dumps(
                {
                    "stage": "openai_compatible_policy",
                    "status": "ok",
                    "artifact": output_path.name,
                    "sha256": sha256_file(output_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        failure_path = write_private_json(
            stage_dir / "failure.json",
            {
                "stage": "openai_compatible_policy",
                "status": "failed",
                "error_type": type(exc).__name__,
            },
        )
        print(
            json.dumps(
                {
                    "stage": "openai_compatible_policy",
                    "status": "failed",
                    "artifact": failure_path.name,
                    "sha256": sha256_file(failure_path),
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
