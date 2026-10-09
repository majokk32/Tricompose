"""Symbolic report-state ceiling with EHR and frozen image labels held fixed.

This is NOT a scorer, router, clinical adjudicator or generation method. The
four symbolic report states are possibilities, not model outputs or attainable
decoded reports. No threshold, observation, gate or winner is changed.
"""
from __future__ import annotations

import math

from tricompose_v12.legacy_replay_adapter import FINDINGS, STATES
from .contracts import require

VERSION = "tricompose-fixed-image-report-reachability-v1"
REPORT_STATES = ("positive", "negative", "uncertain", "unknown")
EXPLICIT = frozenset(("positive", "negative"))


def relation(left, right):
    require(left in STATES and right in STATES, "four_state_relation_required")
    return ("not_comparable" if left not in EXPLICIT or right not in EXPLICIT
        else "support" if left == right else "proxy_opposition")


def truth_table(ehr, xrv):
    """Four single-finding possibilities, not an exhaustive report search."""
    fixed = relation(ehr, xrv)
    return [{"symbolic_report_state": state, "immutable_ehr_cxr_relation": fixed,
        "ehr_report_relation": relation(ehr, state), "cxr_report_relation": relation(xrv, state),
        "all_three_explicit_support": ehr in EXPLICIT and xrv == ehr == state,
        "known_ehr_report_comparable": ehr in EXPLICIT and state in EXPLICIT,
        "symbolic_only_not_generated_or_selected": True} for state in REPORT_STATES]


def threshold_readout(value, entry, state):
    """Audit unchanged operating-point scores; legacy field is not probability."""
    require(state in STATES and type(entry.get("enabled")) is bool,
        "typed_four_state_and_frozen_mask_required")
    low, high = entry["negative_max"], entry["positive_min"]
    require(all(type(x) in (int, float) and math.isfinite(x) for x in (low, high))
        and 0 <= low <= high <= 1, "ordered_finite_frozen_boundaries_required")
    require(value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1),
        "finite_operating_point_normalized_score_required")
    expected = ("unknown" if value is None or not entry["enabled"] else
        "positive" if value >= high else "negative" if value <= low else "uncertain")
    require(state == expected, "state_must_match_unchanged_frozen_boundaries")
    active = value is not None and entry["enabled"]
    return {"enabled": entry["enabled"], "score": value, "negative_max": low, "positive_min": high,
        "state": state, "score_semantics": "xrv_op_norm_0_1", "probability_semantics": False,
        "score_minus_negative_boundary": value - low if active else None,
        "score_minus_positive_boundary": value - high if active else None,
        "boundary_distances_are_clinical_confidence": False,
        "threshold_changed": False, "score_missing": value is None}


def analyze(facts, scores, thresholds, observer, *, observer_scope, observer_status):
    """Linear per-finding symbolic ceiling, retaining missingness denominators."""
    require(isinstance(facts, list) and len(facts) == len(FINDINGS)
        and [f["finding"] for f in facts] == list(FINDINGS), "exact_named_receipt_inventory_required")
    require(isinstance(scores, dict) and isinstance(thresholds, dict)
        and set(scores) == set(thresholds) == set(FINDINGS), "complete_frozen_score_inventory_required")
    require(isinstance(observer_scope, (tuple, list)) and len(set(observer_scope)) == len(observer_scope)
        and 1 <= len(observer_scope) <= 14 and set(observer_scope) <= set(FINDINGS),
        "explicit_unique_observer_scope_required")
    require(observer_status in ("complete", "failed_unavailable"), "explicit_observer_status_required")
    require((observer_status == "complete" and isinstance(observer, dict)
        and set(observer) == set(observer_scope) and all(s in STATES for s in observer.values()))
        or (observer_status == "failed_unavailable" and observer is None),
        "failed_observer_is_not_invented_negative_or_unknown_vector")
    records = []
    for f in facts:
        name, ehr, xrv = f["finding"], f["ehr"], f["xrv"]
        fixed = relation(ehr, xrv)
        available = observer_status == "complete" and name in observer_scope
        other = observer[name] if available else None
        records.append({"finding": name, "ehr_state": ehr, "xrv_state": xrv,
            "known_ehr": ehr in EXPLICIT, "fixed_ehr_cxr_relation": fixed,
            "observer_in_scope": name in observer_scope, "observer_available": available,
            "observer_state": other, "observer_ehr_relation": relation(ehr, other) if available else None,
            "observer_xrv_relation": relation(xrv, other) if available else None,
            "unchanged_threshold_readout": threshold_readout(scores[name], thresholds[name], xrv),
            "symbolic_report_truth_table": truth_table(ehr, xrv)})
    known = [r for r in records if r["known_ehr"]]
    support = sum(r["fixed_ehr_cxr_relation"] == "support" for r in known)
    opposed = sum(r["fixed_ehr_cxr_relation"] == "proxy_opposition" for r in known)
    missing = len(known) - support - opposed
    return {"schema_version": VERSION, "records": records,
        "bounds": {"inventory_findings": len(FINDINGS), "known_ehr_facts": len(known),
            "fixed_ehr_cxr_supported_facts": support, "fixed_ehr_cxr_opposition_facts": opposed,
            "fixed_ehr_cxr_missing_facts": missing,
            "maximum_all_three_support_if_only_report_changes": support,
            "maximum_all_three_support_over_known": support / len(known) if known else None,
            "minimum_remaining_fixed_ehr_cxr_oppositions": opposed,
            "minimum_report_edge_oppositions_at_full_known_ehr_report_comparison": opposed,
            "every_known_ehr_fact_can_have_triple_support_under_fixed_labels": support == len(known) if known else None,
            "bound_scope": "four_state_label_domain_only_not_language_decoder_or_acceptance_gate",
            "symbolic_state_slots": len(FINDINGS) * len(REPORT_STATES),
            "bound_is_clinical_accuracy_or_validated_fault_location": False},
        "report_states_generated": 0, "new_model_calls": 0, "installed_as_policy": False,
        "clinical_acceptance": False, "clinical_fault_location": None, "majority_vote_used": False}


def report_readout(row, facts):
    """Existing actual report receipt only; no consensus or hypothetical output."""
    require([(f["finding"], f["ehr"], f["xrv"]) for f in row["receipt"]["fact_states"]]
        == [(f["finding"], f["ehr"], f["xrv"]) for f in facts], "same_fixed_ehr_image_vectors_required")
    known = [f for f in facts if f["ehr"] in EXPLICIT]
    states = {f["finding"]: f["chexbert"] for f in row["receipt"]["fact_states"]}
    require(set(states) == set(FINDINGS) and all(s in STATES for s in states.values()),
        "actual_named_four_state_report_labels_required")
    supported = sum(f["ehr"] == f["xrv"] == states[f["finding"]] for f in known)
    require(supported == row["receipt"]["all_three_supported_facts"], "actual_joint_receipt_replay_required")
    return {"triple_candidate_id": row["triple_candidate_id"], "report_model": row["report_model_id"],
        "report_sha256": row["report_sha256"], "receipt_id": row["receipt"]["receipt_id"],
        "actual_report_not_symbolic": True, "raw_edge_readouts": row["raw_edge_readouts"],
        "all_three_supported_facts": supported, "known_ehr_facts": len(known),
        "all_three_support_over_known": supported / len(known) if known else None,
        "section_contract_pass": row["structure"]["section_contract_pass"],
        "unsupported_temporal_comparison_language": row["structure"]["unsupported_temporal_comparison_language"],
        "clinical_accuracy": None, "clinical_acceptance": False}
