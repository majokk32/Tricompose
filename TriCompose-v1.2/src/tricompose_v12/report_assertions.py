"""Source-bound report assertions with explicit scope coverage and abstention.

This gate does not correct a model label, prove that a report describes its
image, or authorize regeneration. It filters proposals using an unchanged,
limited literal-scope checker. Unknown is never an explicit negative.
"""
from __future__ import annotations

import hashlib

FINDINGS = ("cardiomegaly", "consolidation", "pleural_effusion", "pneumothorax")
STATES = frozenset({"positive", "negative", "uncertain", "unknown"})
GATE_VERSION = "report-literal-scope-abstention-v1"
CHECKED_REASONS = frozenset({
    "explicit_pre_negation", "explicit_post_negation", "explicit_presence_copula",
    "explicit_uncertainty", "qualified_absence_not_global_negative",
})
CONTEXT_FLAGS = ("is_negated", "is_uncertain", "is_historical", "is_hypothetical", "is_family")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checked_span(report, start, end):
    if not isinstance(report, str) or not 1 <= len(report) <= 8192:
        raise ValueError("invalid bounded report")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(report):
        raise ValueError("invalid Unicode source offsets")
    return {"quote": report[start:end], "char_start": start, "char_end": end,
        "quote_sha256": digest(report[start:end]), "offset_unit": "unicode_codepoint"}


def validate_span(report, span):
    expected = checked_span(report, span["char_start"], span["char_end"])
    if any(span.get(key) != value for key, value in expected.items()):
        raise ValueError("source quote/hash/offset mismatch")


def gate_assertion(report, finding, proposed_state, scope_checker):
    """No key, expected state, threshold, image or other patient's text."""
    if finding not in FINDINGS or proposed_state not in STATES:
        raise ValueError("invalid finding/state")
    if not isinstance(report, str) or not 1 <= len(report) <= 8192:
        raise ValueError("invalid bounded report")
    report_sha = digest(report)
    base = {"finding": finding, "proposed_state": proposed_state, "state": "unknown",
        "report_sha256": report_sha, "gate_version": GATE_VERSION,
        "independent_clinical_validation": False, "regeneration_authorized": False,
        "evidence_dependency": "same_report_text", "evidence": [], "scope_verified": False}
    if proposed_state == "unknown":
        return {**base, "decision": "no_model_assertion", "reason": "missing_is_not_negative",
            "review_action": "no_comparable_report_fact"}
    # Deterministically use the entire source, not a gold-selected sentence.
    # Every literal mention must be covered; full-source conflicts can veto.
    span = checked_span(report, 0, len(report))
    decision = scope_checker(report, span, finding, proposed_state)
    mentions = decision.get("literal_mentions_checked")
    reasons = decision.get("scope_reasons")
    suggested = decision.get("rule_suggested_states")
    if type(mentions) is not int or mentions < 0 or not isinstance(reasons, list) or not reasons:
        raise ValueError("invalid scope-check contract")
    if not isinstance(suggested, list) or any(value not in STATES-{"unknown"} for value in suggested):
        raise ValueError("invalid scope suggested states")
    covered = (mentions > 0 and set(reasons) <= CHECKED_REASONS and suggested == [proposed_state])
    commit = decision.get("veto") is False and covered
    evidence_id = digest(f"{report_sha}|{finding}|0|{len(report)}")
    return {**base, "state": proposed_state if commit else "unknown",
        "decision": "scope_commit" if commit else "abstain",
        "reason": "proposal_has_checked_literal_scope" if commit else "veto_or_incomplete_scope",
        "scope_verified": commit, "scope_check": decision,
        "evidence": [{**span, "evidence_id": evidence_id}],
        "review_action": "retain_report_assertion" if commit else "verify_more"}


def context_state(flags):
    """Official ConText flags mapped to current assertions; unmodified != verified."""
    if set(flags) != set(CONTEXT_FLAGS) or any(type(value) is not bool for value in flags.values()):
        raise ValueError("invalid ConText flags")
    if flags["is_historical"] or flags["is_hypothetical"] or flags["is_family"]:
        return "unknown"
    if flags["is_uncertain"]:
        return "uncertain"
    if flags["is_negated"]:
        return "negative"
    # Conventional ConText baseline only. This default is not clinical proof.
    return "positive"


def aggregate_context_mentions(mentions):
    states = {context_state(row["flags"]) for row in mentions}-{"unknown"}
    if "uncertain" in states or {"positive", "negative"} <= states:
        return "uncertain"
    return next(iter(states)) if states else "unknown"


def crosscheck_context(gated, context_record, report):
    """A second parser may only veto a commit; it cannot promote/flip a label."""
    if gated["report_sha256"] != digest(report):
        raise ValueError("gate/source binding differs")
    if context_record.get("state") not in STATES:
        raise ValueError("invalid context state")
    for mention in context_record["mentions"]:
        validate_span(report, mention["span"])
        context_state(mention["flags"])
        for modifier in mention["modifiers"]:
            validate_span(report, modifier["cue"])
            validate_span(report, modifier["scope"])
    if aggregate_context_mentions(context_record["mentions"]) != context_record["state"]:
        raise ValueError("context aggregation differs")
    agree = context_record["state"] == gated["proposed_state"]
    result = {**gated, "second_parser": "medspacy_context", "context_state": context_record["state"],
        "parser_agreement": agree, "parsers_are_independent_clinical_votes": False}
    if gated["decision"] == "scope_commit" and not agree:
        return {**result, "state": "unknown", "decision": "abstain", "scope_verified": False,
            "reason": "parser_disagreement", "review_action": "verify_more"}
    return result
