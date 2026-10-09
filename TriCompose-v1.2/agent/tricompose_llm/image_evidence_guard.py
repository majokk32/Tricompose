"""Prospective veto for unresolved image evidence; no clinical majority vote.

This pure, optional adapter does not modify the consumed V1 planner/session.
Upstream callers must authenticate and bind each observer to the current image
and fixed EHR. Rebased IDs and self-consistent counts are not artifact trust.
Only direct known EHR constraints affect this guard, not unmentioned findings,
reports, similarity scores or fabricated negatives. Agreement never promotes
an output to clinical truth or bypasses the original acceptance gate.
"""
from __future__ import annotations

from copy import deepcopy

from tricompose_v12.legacy_replay_adapter import FINDINGS, STATES
from .contracts import (DECISION_SCHEMA, count, exact, opaque, require,
                        validate_decision, validate_public_state)

SCHEMA = "tricompose-image-evidence-disagreement-guard-v1"
EXPLICIT = frozenset(("positive", "negative"))
COUNT_FIELDS = (
    "finding_slots", "known_ehr", "joint_known_comparable",
    "known_scorer_agreement", "known_scorer_disagreement",
    "known_missing_comparison", "joint_ehr_support", "joint_ehr_opposition",
    "xrv_ehr_support", "xrv_ehr_opposition", "xrv_ehr_missing",
    "observer_ehr_support", "observer_ehr_opposition", "observer_ehr_missing",
    "outside_known_ehr_scorer_disagreement", "observer_unavailable_slots",
)
GUARD_FIELDS = (
    "schema_version", "current_candidate_id", "current_evidence_id", "counts",
    "status", "withhold_image_attribution", "clinical_fault_location",
    "clinical_acceptance", "primary_metric_eligible", "majority_vote_used",
)


def _vector(value, names):
    require(isinstance(value, dict) and set(value) == set(names)
            and all(isinstance(v, str) and v in STATES for v in value.values()),
            "complete_named_four_state_vector_required")


def _status(c):
    if not c["known_ehr"]:
        return "unresolved_no_known_ehr_constraints"
    if c["known_scorer_disagreement"]:
        return "unresolved_scorer_disagreement"
    if c["known_missing_comparison"]:
        return "unresolved_missing_image_evidence"
    return "scorers_agree_unvalidated"


def build_guard(ehr, xrv, observer, *, candidate_id, evidence_id,
                observer_status="complete"):
    """Count one finding once per image, not once per generated report.

    A failed observer must have None, not an invented all-unknown/all-negative
    vector. A complete observer may explicitly return uncertain or unknown.
    Finding scope is the explicitly supplied shared scorer inventory.
    """
    require(isinstance(ehr, dict) and 1 <= len(ehr) <= 14
            and set(ehr) <= set(FINDINGS), "declared_shared_finding_scope_required")
    names = sorted(ehr)
    _vector(ehr, names); _vector(xrv, names)
    require(observer_status in ("complete", "failed_unavailable"),
            "explicit_observer_availability_required")
    if observer_status == "complete":
        _vector(observer, names)
    else:
        require(observer is None, "failed_observer_must_not_have_labels")
    c = dict.fromkeys(COUNT_FIELDS, 0)
    c["finding_slots"] = len(names)
    c["observer_unavailable_slots"] = len(names) if observer is None else 0
    for name in names:
        ref, a = ehr[name], xrv[name]
        b = observer[name] if observer is not None else None
        comparable = a in EXPLICIT and b in EXPLICIT
        if ref not in EXPLICIT:
            c["outside_known_ehr_scorer_disagreement"] += int(comparable and a != b)
            continue
        c["known_ehr"] += 1
        for prefix, state in (("xrv", a), ("observer", b)):
            kind = "missing" if state not in EXPLICIT else "support" if state == ref else "opposition"
            c[prefix + "_ehr_" + kind] += 1
        if not comparable:
            c["known_missing_comparison"] += 1
        else:
            c["joint_known_comparable"] += 1
            c["known_scorer_agreement" if a == b else "known_scorer_disagreement"] += 1
            if a == b:
                c["joint_ehr_support" if a == ref else "joint_ehr_opposition"] += 1
    status = _status(c)
    result = {"schema_version": SCHEMA, "current_candidate_id": candidate_id,
        "current_evidence_id": evidence_id, "counts": c, "status": status,
        "withhold_image_attribution": status != "scorers_agree_unvalidated",
        "clinical_fault_location": None, "clinical_acceptance": False,
        "primary_metric_eligible": False, "majority_vote_used": False}
    return validate_guard(result)


def validate_guard(guard):
    """Closed numeric projection; no patient names, paths, hashes or prose."""
    exact(guard, GUARD_FIELDS, "closed_numeric_image_guard_required")
    require(guard["schema_version"] == SCHEMA, "image_guard_version")
    opaque(guard["current_candidate_id"], "c")
    opaque(guard["current_evidence_id"], "e")
    c = guard["counts"]
    exact(c, COUNT_FIELDS, "closed_image_guard_counts_required")
    for value in c.values():
        count(value, 14)
    n, known = c["finding_slots"], c["known_ehr"]
    require(1 <= n <= 14 and known <= n
        and c["joint_known_comparable"] == c["known_scorer_agreement"] + c["known_scorer_disagreement"]
        and c["joint_known_comparable"] + c["known_missing_comparison"] == known
        and c["joint_ehr_support"] + c["joint_ehr_opposition"] == c["known_scorer_agreement"]
        and c["outside_known_ehr_scorer_disagreement"] <= n - known
        and c["observer_unavailable_slots"] in (0, n), "image_guard_count_algebra")
    for prefix in ("xrv", "observer"):
        require(sum(c[prefix + "_ehr_" + k] for k in ("support", "opposition", "missing")) == known
            and c["joint_ehr_support"] <= c[prefix + "_ehr_support"]
            and c["joint_ehr_opposition"] <= c[prefix + "_ehr_opposition"]
            and c[prefix + "_ehr_missing"] <= c["known_missing_comparison"], "image_guard_edge_algebra")
    if c["observer_unavailable_slots"]:
        require(c["observer_ehr_missing"] == c["known_missing_comparison"] == known
            and c["joint_known_comparable"] == c["outside_known_ehr_scorer_disagreement"] == 0,
            "unavailable_observer_cannot_contribute_comparisons")
    status = _status(c)
    require(guard["status"] == status
        and type(guard["withhold_image_attribution"]) is bool
        and guard["withhold_image_attribution"] is (status != "scorers_agree_unvalidated")
        and guard["clinical_fault_location"] is None
        and all(guard[k] is False for k in ("clinical_acceptance", "primary_metric_eligible", "majority_vote_used")),
        "image_guard_status_or_claim_changed")
    return guard


def _bound(guard, state):
    validate_guard(guard); validate_public_state(state)
    current = next(r for r in state["evidence"] if r["candidate_id"] == state["current_candidate_id"])
    require(guard["current_candidate_id"] == current["candidate_id"]
            and guard["current_evidence_id"] == current["evidence_id"], "guard_current_observation_binding_required")
    edge, c = current["edges"]["ehr_cxr"], guard["counts"]
    require(c["known_ehr"] == edge["known"] and c["xrv_ehr_support"] == edge["supported"]
            and c["xrv_ehr_opposition"] == edge["opposed"]
            and c["xrv_ehr_missing"] == edge["known"] - edge["comparable"],
            "guard_scope_or_primary_image_counts_changed")


def guarded_state(state, guard):
    """Remove image-regeneration tools only; keep report tools and old scores.

    This is an optional NEW prospective policy view, not a mutation of a saved
    state. Budgets, charged history and all measured evidence remain unchanged.
    """
    _bound(guard, state)
    result = deepcopy(state)
    if guard["withhold_image_attribution"]:
        result["tools"] = [t for t in result["tools"] if t["action"] != "regenerate_cxr"]
    return validate_public_state(result)


def guarded_decision(decision, original_state, guard):
    """Enforce the veto even if a planner ignores it; never forgive bad JSON.

    Only a valid original decision can be conservatively withheld. A terminal
    stop with unresolved constraints becomes abstain, not clinical acceptance.
    The raw decision and charged call must be retained upstream.
    """
    _bound(guard, original_state)
    validate_decision(decision, original_state)
    blocked = guard["withhold_image_attribution"] and decision["action"] in ("regenerate_cxr", "stop")
    effective = deepcopy(decision)
    if blocked:
        effective = {"schema_version": DECISION_SCHEMA, "step_id": original_state["step_id"],
            "action": "abstain", "target_id": None,
            "evidence_ids": [guard["current_evidence_id"]], "reason_code": "insufficient_evidence"}
    validate_decision(effective, guarded_state(original_state, guard))
    return {"raw_decision": deepcopy(decision), "effective_decision": effective,
        "decision_withheld": bool(blocked), "guard_status": guard["status"],
        "clinical_acceptance": False, "clinical_fault_location": None,
        "original_acceptance_gate_must_still_run": True}
