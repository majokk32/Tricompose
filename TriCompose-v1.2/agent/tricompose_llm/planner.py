"""Old chat/completions protocol, new closed decisions and numeric-only state.

No credential files, model imports, network calls or environment reads at
import. No retries or redirects. Use an injected planner for offline tests.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from .contracts import (DECISION_SCHEMA, ContractError, decision_schema, decode_json,
                        require, validate_decision, validate_public_state)

SYSTEM_PROMPT = """You schedule frozen tools for a synthetic multimodal workflow.
Input contains ONLY anonymous IDs, numeric proxy readouts, uncertainty counts,
costs and completed action history. No clinical truth, images or report text
is provided. A mismatch does not establish which modality is clinically wrong.
Unknown and uncertain are not negative. A report and its CXR are dependent;
agreement is not independent confirmation. Missing comparison is not success.
Choose ONE affordable registered tool or stop/abstain. Tools changing a CXR
include regenerating its report with the same expert and re-verifying both.
Costs include generation and verification; planner calls are also charged.
Use observed action effects to decide whether to try another report or image,
not just the highest scalar score. Do not replace the EHR or invent facts,
prompts, tools, IDs or unobserved candidates. Cite observed evidence IDs.
Return ONLY the required JSON object, no markdown or free-text rationale.
Local checks, NOT your decision, decide whether a proposed replacement commits.
Stopping does not certify clinical correctness. Never claim clinical success.
"""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def endpoint_url(base):
    require(isinstance(base, str) and len(base) <= 512 and not any(ord(c) <= 32 for c in base), "invalid_endpoint")
    parsed = urllib.parse.urlsplit(base)
    require(parsed.scheme in ("http", "https") and parsed.hostname is not None
            and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment, "invalid_endpoint")
    require(parsed.scheme == "https" or parsed.hostname in ("localhost", "127.0.0.1", "::1"),
            "plaintext_only_on_loopback")
    require(re.fullmatch(r"[A-Za-z0-9/_-]*", parsed.path) is not None, "invalid_endpoint_path")
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class OpenAICompatiblePlanner:
    """Model/provider chosen explicitly; no default model, key or silent fallback."""
    def __init__(self, *, base_url, model, api_key=None, timeout_seconds=30,
                 response_format="json_schema"):
        self.url = endpoint_url(base_url)
        require(isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/_.:-]{0,191}", model), "invalid_model_name")
        require(type(timeout_seconds) is int and 1 <= timeout_seconds <= 60, "bounded_api_timeout")
        require(response_format in ("json_schema", "json_object"), "explicit_response_format")
        require(api_key is None or isinstance(api_key, str) and api_key and not any(ord(c) <= 32 for c in api_key), "invalid_api_key")
        self.model, self._api_key = model, api_key
        self.timeout, self.response_format = timeout_seconds, response_format
        self.api_attempts = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.usage_available_calls = 0

    def request_payload(self, state):
        validate_public_state(state)
        fmt = {"type": self.response_format}
        if self.response_format == "json_schema":
            fmt["json_schema"] = {"name": "tricompose_repair_decision", "strict": True,
                                  "schema": decision_schema(state)}
        prompt = SYSTEM_PROMPT
        if self.response_format == "json_object":
            prompt += "\nRequired JSON schema: " + json.dumps(decision_schema(state), sort_keys=True)
        return {"model": self.model, "messages": [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(state, sort_keys=True, allow_nan=False)}],
                "max_completion_tokens": 384, "n": 1, "stream": False, "response_format": fmt}

    def propose(self, state):
        payload = self.request_payload(state)
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = "Bearer " + self._api_key
        request = urllib.request.Request(self.url, data=json.dumps(payload, allow_nan=False).encode(),
                                         headers=headers, method="POST")
        self.api_attempts += 1  # Failed/invalid calls are still attempts; never auto-retry.
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(524289)
            require(len(raw) <= 524288, "bounded_api_response")
            value = decode_json(raw.decode("utf-8"), limit=524288)
            require(isinstance(value, dict), "invalid_api_response")
            usage = value.get("usage")
            if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in self.usage):
                for key in self.usage: self.usage[key] += usage[key]
                self.usage_available_calls += 1
            choices = value.get("choices")
            require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
                    and choices[0].get("finish_reason") == "stop", "complete_single_response_required")
            message = choices[0].get("message")
            require(isinstance(message, dict) and not message.get("refusal") and not message.get("tool_calls"),
                    "plain_decision_response_required")
            decision = decode_json(message.get("content"))
            return validate_decision(decision, state)
        except Exception:
            # Do not expose provider error bodies, credentials, URL or responses.
            raise ContractError("planner_call_failed_or_invalid") from None


class DemoPlanner:
    """Explicit rule MOCK, never described as an LLM or measured method."""
    def propose(self, state):
        validate_public_state(state)
        current = next(r for r in state["evidence"] if r["candidate_id"] == state["current_candidate_id"])
        image = current["edges"]["ehr_cxr"]["opposed"] or not current["quality"]["image_basic_valid"]
        report = (current["edges"]["ehr_report"]["opposed"]
                  or current["edges"]["cxr_report"]["opposed"])
        failed_report = bool(state["history"] and state["history"][-1]["action"] == "regenerate_report"
                             and not state["history"][-1]["accepted_proxy_transition"])
        preferred = "regenerate_cxr" if image and (failed_report or not report) else "regenerate_report"
        menu = [t for t in state["tools"] if t["action"] == preferred]
        target = menu[0] if (image or report) and menu else None
        return {"schema_version": DECISION_SCHEMA, "step_id": state["step_id"],
                "action": target["action"] if target else "abstain" if image or report else "stop",
                "target_id": target["tool_id"] if target else None,
                "evidence_ids": [current["evidence_id"]],
                "reason_code": ("image_mismatch" if preferred == "regenerate_cxr" else "report_mismatch")
                    if target else "insufficient_evidence" if image or report else "no_further_action"}
