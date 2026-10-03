"""Immutable cached-EHR anchor and per-generation verification receipts.

Evidence accounting only: no model execution, policy update or clinical repair
claim. Unknown/uncertain never negative; scalar scores cannot change receipts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re

from .automatic_discrepancy import point
from .automatic_secondary import paired_case_comparisons
from .legacy_replay_adapter import FINDINGS, STATES, HASH_FIELDS, PROFILE
from .ranking_switch_controls import observed_candidates

SCHEMA = "tricompose-invariant-verification-interface-v1"
POLICY_SCHEMA = "tricompose-invariant-verification-interface-policy-v1"
PROVENANCE = "cached_direct_state_and_source_category_not_new_raw_ehr_review"
EXPLICIT = {"positive", "negative"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if config != {"schema_version": POLICY_SCHEMA, "profile": PROFILE,
        "reference": "fixed_cached_ehr_states_hashes_and_source_categories",
        "candidate_observation_boundary": "original_requested_slots_only",
        "unknown_uncertain_are_negative": False, "weak_context_is_hard_constraint": False,
        "secondary_scores_used_for_verification": False,
        "source_global_no_finding_adjustment_changes_raw_states": False,
        "same_image_reports_are_independent_votes": False,
        "changes_original_routing_selection_or_costs": False, "model_execution_allowed": False,
        "clinical_repair_success_allowed": False, "clinical_acceptance_allowed": False}:
        raise ValueError("unsupported immutable verification interface")


@dataclass(frozen=True)
class EHRAnchor:
    case_id: str
    ehr_sha256: str
    ehr_facts_sha256: str
    # Immutable tuples: finding, four-state value, cached source-category tuple.
    findings: tuple

    def __post_init__(self):
        if (not isinstance(self.case_id, str) or not _ID.fullmatch(self.case_id)
                or any(not isinstance(h, str) or not _HASH.fullmatch(h)
                       for h in (self.ehr_sha256, self.ehr_facts_sha256))
                or type(self.findings) is not tuple or len(self.findings) != len(FINDINGS)):
            raise ValueError("invalid immutable EHR anchor")
        for expected, item in zip(FINDINGS, self.findings):
            if (type(item) is not tuple or len(item) != 3 or item[0] != expected
                    or item[1] not in STATES or type(item[2]) is not tuple
                    or any(not isinstance(c, str) or not _ID.fullmatch(c) for c in item[2])
                    or tuple(sorted(set(item[2]))) != item[2]
                    or item[1] != "unknown" and not item[2]):
                raise ValueError("invalid cached EHR state/category provenance")

    def record(self):
        return {"schema_version": "tricompose-fixed-cached-ehr-anchor-v1", "profile": PROFILE,
            "case_id": self.case_id, "ehr_sha256": self.ehr_sha256, "ehr_facts_sha256": self.ehr_facts_sha256,
            "provenance_resolution": PROVENANCE, "independent_clinical_truth_available": False,
            "findings": [{"finding": name, "state": state, "source_categories": list(categories),
                "reference_evidence_id": _digest([self.ehr_sha256, self.ehr_facts_sha256, name, state, categories])}
                for name, state, categories in self.findings]}

    @property
    def sha256(self):
        return _digest(self.record())


def _candidate_facts(candidate):
    row, facts = candidate["score_record"], candidate["facts"]
    if (row.get("schema_version") != "tricompose-edge-specific-selection-v1.1"
            or len(facts) != len(FINDINGS) or {f["finding"] for f in facts} != set(FINDINGS)):
        raise ValueError("complete validated legacy candidate required")
    lineage = row["lineage"]
    if any(not isinstance(k, str) or not _ID.fullmatch(k) for k in
           (row["case_id"], row["triple_candidate_id"], lineage["cxr_candidate_id"], lineage["report_candidate_id"])):
        raise ValueError("invalid opaque candidate identity")
    hashes = {k: lineage[k] for k in HASH_FIELDS}
    if any(not isinstance(h, str) or not _HASH.fullmatch(h) for h in hashes.values()):
        raise ValueError("invalid artifact hash")
    seen = set()
    for f in facts:
        if (f["case_id"] != row["case_id"] or f["triple_candidate_id"] != row["triple_candidate_id"]
                or f["artifact_hashes"] != hashes or f["cxr_candidate_id"] != lineage["cxr_candidate_id"]
                or f["report_candidate_id"] != lineage["report_candidate_id"]
                or set(f["states"]) != {"ehr", "xrv", "chexbert"}
                or any(s not in STATES for s in f["states"].values())
                or not isinstance(f["evidence_id"], str) or not _HASH.fullmatch(f["evidence_id"])
                or f["evidence_id"] in seen or f.get("weak_context_promoted") is not False
                or f.get("clinical_truth_verified") is not False
                or f.get("provenance_resolution") != PROVENANCE):
            raise ValueError("cached candidate evidence/hash/provenance differs")
        seen.add(f["evidence_id"])
    return {f["finding"]: f for f in facts}


def anchor_from_cached_candidate(candidate):
    facts = _candidate_facts(candidate)
    lineage = candidate["score_record"]["lineage"]
    return EHRAnchor(candidate["score_record"]["case_id"], lineage["ehr_sha256"], lineage["ehr_facts_sha256"],
        tuple((name, facts[name]["states"]["ehr"], tuple(sorted(facts[name]["source_categories"]))) for name in FINDINGS))


def _edge(facts, left, right):
    known = [f for f in facts if f[left] in EXPLICIT]
    comparable = [f for f in known if f[right] in EXPLICIT]
    supported = [f for f in comparable if f[left] == f[right]]
    return {"inventory_findings": len(FINDINGS), "known_reference_facts": len(known),
        "comparable_facts": len(comparable), "supported_facts": len(supported),
        "supported_positive": sum(f[left] == "positive" for f in supported),
        "supported_negative": sum(f[left] == "negative" for f in supported),
        "proxy_opposition_facts": len(comparable)-len(supported), "missing_comparisons": len(known)-len(comparable),
        "coverage_over_known": len(comparable)/len(known) if known else None,
        "support_over_known": len(supported)/len(known) if known else None,
        "clinical_accuracy": None}


def _status(edges, known, observed, gate, image_valid):
    return ("artifact_metadata_invalid" if gate or not image_valid else
        "unverified_no_direct_ehr_constraints" if not known else
        "explicit_proxy_opposition_unvalidated" if any(e["proxy_opposition_facts"] for e in edges.values()) else
        "unverified_missing_direct_ehr_comparison" if observed < known else
        "direct_ehr_label_agreement_unvalidated")


def _receipt_payload(candidate, anchor):
    if not isinstance(anchor, EHRAnchor): raise TypeError("immutable EHRAnchor required")
    facts = _candidate_facts(candidate)
    if anchor_from_cached_candidate(candidate) != anchor:
        raise ValueError("fixed EHR hash/state/provenance anchor changed")
    row = candidate["score_record"]
    states = [{"finding": name, **facts[name]["states"], "cached_candidate_evidence_id": facts[name]["evidence_id"]}
              for name in FINDINGS]
    edges = {"ehr_cxr": _edge(states, "ehr", "xrv"), "ehr_report": _edge(states, "ehr", "chexbert"),
             "cxr_report": _edge(states, "xrv", "chexbert")}
    known = edges["ehr_cxr"]["known_reference_facts"]
    known_states = [f for f in states if f["ehr"] in EXPLICIT]
    fully_observed = [f for f in known_states if f["xrv"] in EXPLICIT and f["chexbert"] in EXPLICIT]
    all_three = sum(f["ehr"] == f["xrv"] == f["chexbert"] for f in fully_observed)
    gate = row["scoring"]["selection"]["hard_gate_failure_count"]
    image_valid = row["scoring"]["modality_quality"]["cxr_basic_validity_pass"]
    if type(gate) is not int or gate < 0 or type(image_valid) is not bool:
        raise ValueError("invalid source artifact gate")
    status = _status(edges, known, len(fully_observed), gate, image_valid)
    return {"schema_version": "tricompose-invariant-candidate-receipt-v1", "profile": PROFILE,
        "case_id": row["case_id"], "triple_candidate_id": row["triple_candidate_id"],
        "cxr_candidate_id": row["lineage"]["cxr_candidate_id"], "report_candidate_id": row["lineage"]["report_candidate_id"],
        "artifact_hashes": {k: row["lineage"][k] for k in HASH_FIELDS}, "ehr_anchor_sha256": anchor.sha256,
        "fact_states": states, "raw_edge_readouts": edges, "known_ehr_facts": known,
        "fully_observed_direct_ehr_facts": len(fully_observed), "all_three_supported_facts": all_three,
        "all_three_support_over_known": all_three/known if known else None,
        "artifact_gate_failure_count": gate, "cxr_basic_validity_pass": image_valid,
        "verification_status": status, "report_scope_status": "not_computed_for_legacy_profile",
        "raw_report_states_unverified": True, "same_image_reports_are_independent_votes": False,
        "clinical_repair_success": False, "clinical_acceptance": False,
        "confirmed_faulty_modality": None, "model_execution_allowed": False,
        "independent_clinical_truth_available": False}


def verify_candidate(candidate, anchor):
    payload = _receipt_payload(candidate, anchor)
    return {**payload, "receipt_id": _digest(payload)}


def _validate_receipt(receipt, anchor):
    expected_fields = {"schema_version", "profile", "case_id", "triple_candidate_id", "cxr_candidate_id",
        "report_candidate_id", "artifact_hashes", "ehr_anchor_sha256", "fact_states", "raw_edge_readouts",
        "known_ehr_facts", "fully_observed_direct_ehr_facts", "all_three_supported_facts",
        "all_three_support_over_known", "artifact_gate_failure_count", "cxr_basic_validity_pass", "verification_status",
        "report_scope_status", "raw_report_states_unverified", "same_image_reports_are_independent_votes",
        "clinical_repair_success", "clinical_acceptance", "confirmed_faulty_modality", "model_execution_allowed",
        "independent_clinical_truth_available", "receipt_id"}
    if not isinstance(anchor, EHRAnchor) or set(receipt) != expected_fields:
        raise ValueError("strict anchor/receipt schema required")
    payload = {k: v for k, v in receipt.items() if k != "receipt_id"}
    if (receipt.get("receipt_id") != _digest(payload) or receipt.get("ehr_anchor_sha256") != anchor.sha256
            or receipt.get("case_id") != anchor.case_id or receipt.get("profile") != PROFILE
            or receipt.get("schema_version") != "tricompose-invariant-candidate-receipt-v1"
            or receipt.get("clinical_repair_success") is not False or receipt.get("clinical_acceptance") is not False
            or receipt.get("model_execution_allowed") is not False
            or receipt.get("confirmed_faulty_modality") is not None
            or receipt.get("independent_clinical_truth_available") is not False):
        raise ValueError("receipt hash/anchor/claim changed")
    hashes = receipt["artifact_hashes"]
    if (set(hashes) != set(HASH_FIELDS) or hashes["ehr_sha256"] != anchor.ehr_sha256
            or hashes["ehr_facts_sha256"] != anchor.ehr_facts_sha256
            or any(not isinstance(h, str) or not _HASH.fullmatch(h) for h in hashes.values())):
        raise ValueError("receipt EHR hashes differ")
    states = receipt["fact_states"]
    if len(states) != len(FINDINGS) or [f["finding"] for f in states] != list(FINDINGS):
        raise ValueError("receipt finding inventory differs")
    for f, (_, state, _) in zip(states, anchor.findings):
        if (set(f) != {"finding", "ehr", "xrv", "chexbert", "cached_candidate_evidence_id"}
                or f["ehr"] != state or any(f[k] not in STATES for k in ("ehr", "xrv", "chexbert"))
                or not isinstance(f["cached_candidate_evidence_id"], str)
                or not _HASH.fullmatch(f["cached_candidate_evidence_id"])):
            raise ValueError("receipt reference/state differs")
    edges = {"ehr_cxr": _edge(states, "ehr", "xrv"), "ehr_report": _edge(states, "ehr", "chexbert"),
             "cxr_report": _edge(states, "xrv", "chexbert")}
    known = edges["ehr_cxr"]["known_reference_facts"]
    observed = [f for f in states if all(f[k] in EXPLICIT for k in ("ehr", "xrv", "chexbert"))]
    supported = sum(f["ehr"] == f["xrv"] == f["chexbert"] for f in observed)
    gate, valid = receipt["artifact_gate_failure_count"], receipt["cxr_basic_validity_pass"]
    if (type(gate) is not int or gate < 0 or type(valid) is not bool or receipt["raw_edge_readouts"] != edges
            or receipt["known_ehr_facts"] != known or receipt["fully_observed_direct_ehr_facts"] != len(observed)
            or receipt["all_three_supported_facts"] != supported
            or receipt["all_three_support_over_known"] != (supported/known if known else None)
            or receipt["verification_status"] != _status(edges, known, len(observed), gate, valid)
            or receipt["report_scope_status"] != "not_computed_for_legacy_profile"
            or receipt["raw_report_states_unverified"] is not True
            or receipt["same_image_reports_are_independent_votes"] is not False):
        raise ValueError("receipt evidence arithmetic/scope differs")
    return {f["finding"]: f for f in states}


def verify_transition(before, after, anchor):
    a, b = _validate_receipt(before, anchor), _validate_receipt(after, anchor)
    changed_image = before["artifact_hashes"]["cxr_sha256"] != after["artifact_hashes"]["cxr_sha256"]
    changed_report = before["artifact_hashes"]["report_sha256"] != after["artifact_hashes"]["report_sha256"]
    if not changed_image and any(a[k]["xrv"] != b[k]["xrv"] for k in FINDINGS):
        raise ValueError("unchanged image has different classifier states")
    if not changed_report and any(a[k]["chexbert"] != b[k]["chexbert"] for k in FINDINGS):
        raise ValueError("unchanged report has different report states")
    known = [name for name, state, _ in anchor.findings if state in EXPLICIT]
    changes = {k: [] for k in ("image_support_added", "image_support_lost", "image_opposition_added",
                              "image_opposition_removed", "image_comparison_missing_before", "image_comparison_missing_after")}
    for name in known:
        ref = a[name]["ehr"]; old, new = a[name]["xrv"], b[name]["xrv"]
        if new == ref and old != ref: changes["image_support_added"].append(name)
        if old == ref and new != ref: changes["image_support_lost"].append(name)
        if new in EXPLICIT and new != ref and not (old in EXPLICIT and old != ref): changes["image_opposition_added"].append(name)
        if old in EXPLICIT and old != ref and not (new in EXPLICIT and new != ref): changes["image_opposition_removed"].append(name)
        if old not in EXPLICIT: changes["image_comparison_missing_before"].append(name)
        if new not in EXPLICIT: changes["image_comparison_missing_after"].append(name)
    gain = bool(changes["image_support_added"])
    loss = bool(changes["image_support_lost"] or changes["image_opposition_added"])
    status = ("no_image_artifact_change" if not changed_image else
        "artifact_metadata_invalid" if after["artifact_gate_failure_count"] or not after["cxr_basic_validity_pass"] else
        "unverified_no_direct_ehr_constraints" if not known else
        "mixed_fixed_ehr_proxy_change_unvalidated" if gain and loss else
        "fixed_ehr_proxy_support_withdrawn_or_opposition_added" if loss else
        "unverified_missing_image_comparison" if changes["image_comparison_missing_after"] else
        "fixed_ehr_proxy_support_increased_unvalidated" if gain else
        "no_fixed_ehr_proxy_support_increase")
    return {"schema_version": "tricompose-invariant-generation-transition-v1", "case_id": anchor.case_id,
        "ehr_anchor_sha256": anchor.sha256, "before_receipt_id": before["receipt_id"], "after_receipt_id": after["receipt_id"],
        "image_changed": changed_image, "report_changed": changed_report, "known_ehr_facts": len(known),
        "verification_status": status, "finding_changes": changes,
        "all_three_support_delta": after["all_three_supported_facts"]-before["all_three_supported_facts"] if known else None,
        "clinical_repair_success": False, "clinical_acceptance": False, "confirmed_faulty_modality": None,
        "model_execution_allowed": False, "independent_clinical_truth_available": False}


def stream_cached_verification(outcomes, bank, policy, config):
    """Pipeline-hook integration on original requested events; does not route."""
    validate_config(config); paired_case_comparisons(outcomes, policy)
    if set(bank) != {r["case_id"] for r in outcomes}: raise ValueError("entire original cohort required")
    candidate_ids = [c["score_record"]["triple_candidate_id"] for grid in bank.values() for c in grid.values()]
    if len(set(candidate_ids)) != len(candidate_ids): raise ValueError("globally unique candidate IDs required")
    anchors = {case: anchor_from_cached_candidate(grid[(*policy["image_order"][0], policy["report_order"][0])])
               for case, grid in bank.items()}
    receipts, events, trials = {}, [], []
    receipt_by_candidate = {}
    for row in sorted(outcomes, key=lambda r: (r["case_id"], r["method"], r["model_call_budget"],
                                               -1 if r["random_seed"] is None else r["random_seed"])):
        case = row["case_id"]; anchor = anchors[case]
        if row["selected_ehr_sha256"] != anchor.ehr_sha256:
            raise ValueError("original trial EHR anchor changed")
        observed = observed_candidates(row, bank[case]); previous = None; observed_ids = set()
        for step, (candidate, action) in enumerate(zip(observed, row["action_trace"])):
            cid = candidate["score_record"]["triple_candidate_id"]
            if cid not in receipt_by_candidate:
                receipt = verify_candidate(candidate, anchor)
                receipt_by_candidate[cid] = receipt["receipt_id"]; receipts[receipt["receipt_id"]] = receipt
            receipt = receipts[receipt_by_candidate[cid]]
            observed_ids.add(cid)
            transition = None if previous is None else verify_transition(previous, receipt, anchor)
            events.append({"case_id": case, "method": row["method"], "model_call_budget": row["model_call_budget"],
                "random_seed": row["random_seed"], "step": step, "original_action": action["action"],
                "original_observed_candidate_id": cid, "receipt_id": receipt["receipt_id"],
                "ehr_anchor_sha256": anchor.sha256, "charged_model_calls": action["charged_model_calls"],
                "cumulative_model_calls": action["cumulative_model_calls"], "transition": transition,
                "original_action_changed": False, "new_model_calls": 0, "clinical_repair_success": False})
            previous = receipt
        cid = row["selected_candidate_id"]
        if cid is not None:
            if cid not in observed_ids: raise ValueError("final selected candidate was not requested")
            candidate = next(c for c in observed if c["score_record"]["triple_candidate_id"] == cid)
            point(row, candidate)
        selected_receipt = receipts[receipt_by_candidate[cid]] if cid is not None else None
        trials.append({"case_id": case, "method": row["method"], "model_call_budget": row["model_call_budget"],
            "random_seed": row["random_seed"], "selected_candidate_id": cid,
            "selected_receipt_id": selected_receipt["receipt_id"] if selected_receipt else None,
            "selected_verification_status": selected_receipt["verification_status"] if selected_receipt else "no_eligible_candidate",
            "ehr_anchor_sha256": anchor.sha256, "original_terminal_reason": row["terminal_reason"],
            "simulated_calls": dict(row["simulated_calls"]), "simulated_model_calls": row["simulated_model_calls"],
            "original_selection_changed": False, "clinical_acceptance": False, "new_model_calls": 0})
    by_method = defaultdict(list)
    for e in events:
        if e["transition"] is not None: by_method[(e["method"], e["model_call_budget"])].append(e)
    statuses = []
    for (method, budget), group in sorted(by_method.items()):
        categories = Counter(e["transition"]["verification_status"] for e in group)
        for status, count in sorted(categories.items()):
            statuses.append({"method": method, "model_call_budget": budget, "transition_status": status,
                "observed_transition_events": count,
                "distinct_fixed_ehr_cases_in_status": len({e["case_id"] for e in group if e["transition"]["verification_status"] == status}),
                "transition_events_are_not_independent_patients": True, "clinical_repair_success": False})
    return {"schema_version": SCHEMA, "anchors": [dict(a.record(), ehr_anchor_sha256=a.sha256) for _, a in sorted(anchors.items())],
        "receipts": [receipts[k] for k in sorted(receipts)], "events": events, "trials": trials,
        "transition_status_counts": statuses, "new_model_calls": 0, "original_selection_changed": False,
        "clinical_repair_success": False, "independent_clinical_truth_available": False}
