"""Lossless, veto-only eligibility metadata for cached assertion proposals.

Agreement between two prompts of one frozen model is correlated soft evidence,
not a clinical qualification. This policy never changes a proposed state,
assigns an image finding, ranks candidates, or authorizes an action.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

SCHEMA = "tricompose-assertion-abstention-v1"
STATES = ("positive", "negative", "uncertain", "unknown")
EXPLICIT = frozenset(("positive", "negative"))
POLICY = {
    "version": SCHEMA,
    "finding": "lung_opacity",
    "require_complete_source_contract": True,
    "require_all_selected_assertions_same_explicit_state": True,
    "require_complete_same_state_cross_format_proposal": True,
    "cross_format_model_dependency": "same_model_same_source_correlated",
    "scope_qualification": "no_frozen_opacity_scope_head",
    "independent_clinical_qualification": False,
    "hard_action_eligible": False,
    "unknown_is_negative": False,
    "abstention_is_unknown_match": False,
    "threshold_fitting": False,
    "state_correction": False,
    "selection_enabled": False,
    "regeneration_authorized": False,
    "declared_after_known_v3_development_results": True,
    "heldout_test": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _validate_outcome(outcome):
    if not isinstance(outcome, dict) or outcome.get("status") not in (
            "complete", "failed_unavailable"):
        raise ValueError("explicit_extractor_status_required")
    state = outcome.get("state")
    if outcome["status"] == "complete":
        if not isinstance(state, str) or state not in STATES:
            raise ValueError("complete_four_state_proposal_required")
        if outcome.get("failure_reason") is not None:
            raise ValueError("complete_outcome_cannot_have_failure")
    elif state is not None or not isinstance(outcome.get("failure_reason"), str) or not outcome["failure_reason"]:
        raise ValueError("unavailable_requires_null_state_and_reason")


def gate(primary, crosscheck, *, source_contract_checked):
    """Return a quote-free sidecar; no reference label or clinical text needed.

    `source_contract_checked` attests mechanical hash/span/decoder consistency,
    not that the selected sentences or their interpreted polarity are correct.
    Unknown and uncertain proposals remain visible, but are not signed evidence.
    """
    if type(source_contract_checked) is not bool:
        raise ValueError("explicit_source_contract_boolean_required")
    for outcome in (primary, crosscheck):
        _validate_outcome(outcome)
    state = primary["state"]
    common = {
        "schema_version": SCHEMA,
        "policy_sha256": digest(POLICY),
        "raw_status": primary["status"],
        "raw_state": state,
        "raw_failure_reason": primary.get("failure_reason"),
        "crosscheck_status": crosscheck["status"],
        "crosscheck_state": crosscheck["state"],
        "crosscheck_failure_reason": crosscheck.get("failure_reason"),
        "source_contract_checked": source_contract_checked,
        "soft_retained_state": None,
        "soft_comparable": False,
        "scope_qualification": POLICY["scope_qualification"],
        "semantic_scope_verified": False,
        "independent_clinical_qualification": False,
        "cross_format_evidence_independent": False,
        "clinical_score": None,
        "hard_action_eligible": False,
        "clinical_fault_localization": False,
        "selection_changed": False,
        "regeneration_authorized": False,
        "state_corrected": False,
        "candidate_dropped": False,
    }
    if not source_contract_checked:
        decision = "source_contract_unavailable"
    elif primary["status"] != "complete":
        decision = "extractor_unavailable"
    elif state == "unknown":
        decision = "not_comparable_unknown"
    elif state == "uncertain":
        decision = "not_comparable_uncertain"
    else:
        assertions, selected = primary.get("assertions"), primary.get("selected_segment_ids")
        if (not isinstance(selected, list) or not selected or
                any(type(i) is not int or i < 0 for i in selected) or selected != sorted(set(selected)) or
                not isinstance(assertions, list) or
                any(not isinstance(a, dict) or set(a) != {"segment_id", "state"} or
                    type(a["segment_id"]) is not int or not isinstance(a["state"], str) or
                    a["state"] not in STATES for a in assertions) or
                [a["segment_id"] for a in assertions] != selected):
            raise ValueError("exhaustive_bound_assertion_inventory_required")
        if any(a["state"] != state for a in assertions):
            decision = "abstain_mixed_segment_states"
        elif crosscheck["status"] != "complete":
            decision = "abstain_cross_format_unavailable"
        elif crosscheck["state"] != state:
            decision = "abstain_cross_format_disagreement"
        else:
            decision = "correlated_agreement_soft_only"
            common.update(soft_retained_state=state, soft_comparable=True)
    return {**common, "decision": decision}


def summarize(rows):
    """Availability counts; never denominator-free accuracy or clinical score."""
    return {
        "rows_retained": len(rows),
        "decision_counts": dict(sorted(Counter(r["decision"] for r in rows).items())),
        "raw_state_counts": dict(sorted(Counter(r["raw_state"] or "unavailable" for r in rows).items())),
        "soft_comparable": sum(r["soft_comparable"] for r in rows),
        "soft_coverage_all_rows": sum(r["soft_comparable"] for r in rows) / len(rows) if rows else None,
        "hard_action_eligible": sum(r["hard_action_eligible"] for r in rows),
        "state_corrections": sum(r["state_corrected"] for r in rows),
        "candidates_dropped": sum(r["candidate_dropped"] for r in rows),
        "new_model_calls": 0,
        "selection_changed": False,
        "regeneration_authorized": False,
    }


def authored_readout(checks):
    """Post-freeze development diagnostics, not calibration or clinical gold.

    A withheld/unknown/uncertain proposal is NOT credited as a correct unknown.
    Report retained errors and lost correct assertions alongside coverage.
    """
    if any(r.get("expected_state") not in STATES for r in checks):
        raise ValueError("four_state_authored_reference_required")
    determinate = [r for r in checks if r["raw_status"] == "complete" and r["raw_state"] in EXPLICIT]
    retained = [r for r in checks if r["soft_comparable"]]
    if any(r["soft_retained_state"] != r["raw_state"] or r["raw_state"] not in EXPLICIT for r in retained):
        raise ValueError("gate_cannot_correct_or_promote")
    correct = lambda r: r["raw_state"] == r["expected_state"]
    supported = sum(correct(r) for r in retained)
    return {
        "authored_rows": len(checks),
        "raw_four_state_matches": sum(r["raw_status"] == "complete" and correct(r) for r in checks),
        "raw_determinate_proposals": len(determinate),
        "raw_correct_determinate": sum(correct(r) for r in determinate),
        "raw_incorrect_determinate": sum(not correct(r) for r in determinate),
        "soft_retained": len(retained),
        "soft_correct": supported,
        "soft_incorrect": len(retained) - supported,
        "conditional_authored_match": supported / len(retained) if retained else None,
        "soft_coverage_all_rows": len(retained) / len(checks) if checks else None,
        "incorrect_determinate_withheld": sum(not correct(r) and not r["soft_comparable"] for r in determinate),
        "correct_determinate_withheld": sum(correct(r) and not r["soft_comparable"] for r in determinate),
        "hard_action_eligible": sum(r["hard_action_eligible"] for r in checks),
        "abstention_credited_as_match": False,
        "overall_gated_accuracy": None,
        "clinical_accuracy": None,
        "clinical_qualification_passed": False,
    }
