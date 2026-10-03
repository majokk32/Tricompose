"""Report-blind CXR-phase receipts over the explicitly legacy cached profile.

No inference, routing or clinical acceptance. A not-generated report is NOT a
report containing unknown labels. Completed receipts are linked, not overwritten.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .invariant_verification import (
    EHRAnchor, EXPLICIT, _digest, _edge, _HASH, _ID, _validate_receipt,
    anchor_from_cached_candidate, verify_candidate,
)
from .legacy_replay_adapter import FINDINGS, PROFILE, STATES

SCHEMA = "tricompose-partial-image-verification-v1"
POLICY_SCHEMA = "tricompose-partial-image-verification-policy-v1"
ORIGIN = "cached_legacy_state_vector_not_new_xrv_execution"
RECEIPT_SCHEMA = "tricompose-cxr-before-report-receipt-v1"


def validate_config(config):
    if config != {
        "schema_version": POLICY_SCHEMA, "profile": PROFILE,
        "reference": "unchanged_invariant_ehr_anchor",
        "phase": "cxr_and_cached_xrv_available_report_not_generated",
        "report_edges_when_not_generated": None,
        "unknown_uncertain_are_negative": False,
        "uses_report_labels_scores_or_votes_for_image_receipt": False,
        "weak_context_is_hard_constraint": False,
        "overwrites_partial_receipts": False,
        "changes_original_routing_selection_or_costs": False,
        "model_execution_allowed": False, "clinical_acceptance_allowed": False,
        "clinical_repair_success_allowed": False,
    }:
        raise ValueError("unsupported partial-image verification contract")


@dataclass(frozen=True)
class CachedImageEvidence:
    case_id: str
    cxr_candidate_id: str
    cxr_model_id: str
    seed: int
    ehr_sha256: str
    ehr_facts_sha256: str
    cxr_sha256: str
    state_cache_sha256: str
    finding_states: tuple
    cxr_basic_validity_pass: bool | None

    def __post_init__(self):
        if (any(not isinstance(x, str) or not _ID.fullmatch(x) for x in
                (self.case_id, self.cxr_candidate_id, self.cxr_model_id))
                or any(not isinstance(h, str) or not _HASH.fullmatch(h) for h in
                       (self.ehr_sha256, self.ehr_facts_sha256, self.cxr_sha256, self.state_cache_sha256))
                or type(self.seed) is not int or self.seed < 0
                or self.cxr_basic_validity_pass is not None and type(self.cxr_basic_validity_pass) is not bool
                or type(self.finding_states) is not tuple or len(self.finding_states) != len(FINDINGS)):
            raise ValueError("invalid immutable cached image evidence")
        for name, item in zip(FINDINGS, self.finding_states):
            if type(item) is not tuple or len(item) != 2 or item[0] != name or item[1] not in STATES:
                raise ValueError("invalid cached image finding inventory")

    def record(self):
        return {
            "schema_version": "tricompose-cached-image-evidence-v1", "profile": PROFILE,
            "case_id": self.case_id, "cxr_candidate_id": self.cxr_candidate_id,
            "cxr_model_id": self.cxr_model_id, "seed": self.seed,
            "artifact_hashes": {"ehr_sha256": self.ehr_sha256,
                "ehr_facts_sha256": self.ehr_facts_sha256, "cxr_sha256": self.cxr_sha256},
            "state_cache_sha256": self.state_cache_sha256, "evidence_origin": ORIGIN,
            "finding_states": [{"finding": name, "xrv": state} for name, state in self.finding_states],
            "cxr_basic_validity_pass": self.cxr_basic_validity_pass,
            "calibrated_clinical_labels": False, "new_classifier_execution": False,
        }


def image_evidence_from_legacy_candidate(candidate, state_cache_sha256):
    """Project image fields only. Do NOT inspect a report, triple gate or score.

    The cohort loader authenticates the whole cache separately. This projection
    can also consume an image-only dictionary lacking all report fields.
    """
    row = candidate["score_record"]; lineage = row["lineage"]
    if row.get("schema_version") != "tricompose-edge-specific-selection-v1.1":
        raise ValueError("legacy image projection required")
    states = {}
    for fact in candidate["facts"]:
        name = fact["finding"]
        if (name not in FINDINGS or name in states or fact["case_id"] != row["case_id"]
                or fact["cxr_candidate_id"] != lineage["cxr_candidate_id"]
                or any(fact["artifact_hashes"][k] != lineage[k]
                       for k in ("ehr_sha256", "ehr_facts_sha256", "cxr_sha256"))):
            raise ValueError("cached image lineage/inventory differs")
        states[name] = fact["states"]["xrv"]
    if set(states) != set(FINDINGS):
        raise ValueError("complete legacy image inventory required")
    return CachedImageEvidence(
        row["case_id"], lineage["cxr_candidate_id"], lineage["cxr_model_id"], lineage["cxr_seed"],
        lineage["ehr_sha256"], lineage["ehr_facts_sha256"], lineage["cxr_sha256"], state_cache_sha256,
        tuple((name, states[name]) for name in FINDINGS),
        row["scoring"]["modality_quality"]["cxr_basic_validity_pass"],
    )


def verify_image_phase(image, anchor):
    if not isinstance(image, CachedImageEvidence) or not isinstance(anchor, EHRAnchor):
        raise TypeError("immutable image evidence and EHRAnchor required")
    if (image.case_id != anchor.case_id or image.ehr_sha256 != anchor.ehr_sha256
            or image.ehr_facts_sha256 != anchor.ehr_facts_sha256):
        raise ValueError("fixed EHR anchor changed before report generation")
    states = [{"finding": name, "ehr": ehr, "xrv": xrv}
              for (name, ehr, _), (_, xrv) in zip(anchor.findings, image.finding_states)]
    edge = _edge(states, "ehr", "xrv")
    known = edge["known_reference_facts"]
    status = (
        "artifact_metadata_invalid" if image.cxr_basic_validity_pass is False else
        "unverified_image_validity_unavailable" if image.cxr_basic_validity_pass is None else
        "unverified_no_direct_ehr_constraints" if not known else
        "explicit_image_proxy_opposition_unvalidated" if edge["proxy_opposition_facts"] else
        "unverified_missing_image_comparison" if edge["missing_comparisons"] else
        "direct_ehr_image_label_agreement_unvalidated"
    )
    payload = {
        "schema_version": RECEIPT_SCHEMA, "profile": PROFILE,
        "phase": "cxr_and_cached_xrv_available_report_not_generated",
        "case_id": anchor.case_id, "ehr_anchor_sha256": anchor.sha256,
        "image_evidence": image.record(), "fact_states": states,
        "report_candidate_id": None, "report_sha256": None,
        "report_lifecycle_status": "not_generated", "report_label_states": None,
        "raw_edge_readouts": {"ehr_cxr": edge, "ehr_report": None, "cxr_report": None},
        "known_ehr_facts": known, "all_three_supported_facts": None,
        "all_three_support_over_known": None, "verification_status": status,
        "clinical_acceptance": False, "clinical_repair_success": False,
        "confirmed_faulty_modality": None, "model_execution_allowed": False,
        "independent_clinical_truth_available": False,
    }
    return {**payload, "receipt_id": _digest(payload)}


def validate_partial_receipt(receipt, anchor):
    record = receipt["image_evidence"]; hashes = record["artifact_hashes"]
    image = CachedImageEvidence(
        record["case_id"], record["cxr_candidate_id"], record["cxr_model_id"], record["seed"],
        hashes["ehr_sha256"], hashes["ehr_facts_sha256"], hashes["cxr_sha256"],
        record["state_cache_sha256"], tuple((f["finding"], f["xrv"]) for f in record["finding_states"]),
        record["cxr_basic_validity_pass"],
    )
    # Python equality conflates False/0 and integer/float values. Compare the
    # canonical serialized content, not only dictionaries' semantic equality.
    if _digest(receipt) != _digest(verify_image_phase(image, anchor)):
        raise ValueError("partial receipt hash/evidence/arithmetic/phase differs")
    return image


def bind_completed_report(partial, completed, anchor):
    """Retrospectively link two immutable receipts; never imply execution order."""
    image = validate_partial_receipt(partial, anchor)
    facts = _validate_receipt(completed, anchor)
    if (completed["cxr_candidate_id"] != image.cxr_candidate_id
            or completed["artifact_hashes"]["cxr_sha256"] != image.cxr_sha256
            or any(facts[name]["xrv"] != state for name, state in image.finding_states)
            or completed["raw_edge_readouts"]["ehr_cxr"] != partial["raw_edge_readouts"]["ehr_cxr"]
            or image.cxr_basic_validity_pass is not None
            and completed["cxr_basic_validity_pass"] != image.cxr_basic_validity_pass):
        raise ValueError("completed report changed the previously verified image")
    payload = {
        "schema_version": "tricompose-image-to-completed-report-binding-v1",
        "case_id": anchor.case_id, "ehr_anchor_sha256": anchor.sha256,
        "partial_receipt_id": partial["receipt_id"], "completed_receipt_id": completed["receipt_id"],
        "cxr_candidate_id": image.cxr_candidate_id, "cxr_sha256": image.cxr_sha256,
        "triple_candidate_id": completed["triple_candidate_id"],
        "report_candidate_id": completed["report_candidate_id"],
        "report_sha256": completed["artifact_hashes"]["report_sha256"],
        "binding_mode": "retrospective_phase_reconstruction_not_actual_execution_order",
        "report_lifecycle_status": "completed_cached_report_linked",
        "report_states_clinically_verified": False, "partial_receipt_overwritten": False,
        "clinical_acceptance": False, "clinical_repair_success": False, "model_execution_allowed": False,
    }
    return {**payload, "binding_id": _digest(payload)}


def bind_cached_image_phases(bank, completed_receipts, state_cache_sha256, config):
    validate_config(config)
    ids = [c["score_record"]["triple_candidate_id"] for grid in bank.values() for c in grid.values()]
    receipts = {r["triple_candidate_id"]: r for r in completed_receipts}
    if (not bank or len(set(ids)) != len(ids) or len(receipts) != len(completed_receipts)
            or set(receipts) != set(ids)):
        raise ValueError("entire original cached cohort/completed inventory required")
    anchors, partials, bindings = [], {}, []
    for case, grid in sorted(bank.items()):
        anchor = anchor_from_cached_candidate(grid[sorted(grid)[0]])
        anchors.append(anchor.record())
        for slot, candidate in sorted(grid.items()):
            full = receipts[candidate["score_record"]["triple_candidate_id"]]
            if _digest(full) != _digest(verify_candidate(candidate, anchor)):
                raise ValueError("original completed candidate receipt changed")
            image = image_evidence_from_legacy_candidate(candidate, state_cache_sha256)
            if slot[:2] != (image.cxr_model_id, image.seed):
                raise ValueError("original image slot changed")
            partial = verify_image_phase(image, anchor)
            key = image.cxr_candidate_id
            if key in partials and partials[key] != partial:
                raise ValueError("same cached image differs between report paths")
            partials[key] = partial
            bindings.append(bind_completed_report(partial, full, anchor))
    ordered = [partials[k] for k in sorted(partials)]
    return {
        "anchors": anchors, "partial_receipts": ordered, "report_bindings": bindings,
        "image_status_counts": dict(sorted(Counter(p["verification_status"] for p in ordered).items())),
    }
