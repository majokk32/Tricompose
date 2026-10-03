"""Fresh frozen-scorer receipts, separate from historical fourteen-head cache.

Pure metadata/state validation. File hashes/authenticity are checked by the
worker adapter. No clinical truth, acceptance, router or new scorer is trained.
"""
from __future__ import annotations

import math

from .invariant_verification import EHRAnchor, _digest, _edge, _HASH, _ID, EXPLICIT
from .legacy_replay_adapter import FINDINGS, STATES

PROFILE = "fresh_frozen_xrv_chexbert_four_state_diagnostic_v1"


def _hash(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("valid SHA256 required")
    return value


def _vector(states):
    if not isinstance(states, dict) or set(states) != set(FINDINGS) or any(s not in STATES for s in states.values()):
        raise ValueError("complete named four-state vector required")
    return states


def _single(bundle, schema, count_name):
    if (bundle.get("schema_version") != schema or bundle.get("producer", {}).get("frozen") is not True
            or bundle.get("finding_order") != list(FINDINGS)
            or bundle.get("counts") != {count_name: 1, "model_calls": 1}
            or not isinstance(bundle.get("records"), list) or len(bundle["records"]) != 1):
        raise ValueError("exactly one frozen single-case scorer result required")
    _hash(bundle["producer"]["checkpoint_sha256"])
    peak = bundle.get("peak_vram_gib")
    if type(peak) not in (int, float) or not math.isfinite(peak) or peak < 0:
        raise ValueError("invalid scorer memory metadata")
    return bundle["records"][0]


def image_receipt(anchor, cxr, labels, *, label_sha256, thresholds_sha256, checkpoint_sha256):
    if not isinstance(anchor, EHRAnchor) or cxr["case_id"] != anchor.case_id:
        raise ValueError("immutable case/EHR anchor required")
    if cxr["ehr_sha256"] != anchor.ehr_sha256 or cxr["ehr_facts_sha256"] != anchor.ehr_facts_sha256:
        raise ValueError("fresh generation changed fixed EHR")
    _hash(label_sha256); _hash(thresholds_sha256); _hash(checkpoint_sha256); _hash(cxr["artifact"]["sha256"])
    if cxr.get("frozen_model") is not True: raise ValueError("frozen image generator required")
    if not _ID.fullmatch(cxr["candidate_id"]): raise ValueError("opaque image ID required")
    row = _single(labels, "tricompose-cxr-finding-labels-v1.1", "cxr_candidates")
    if (row["cxr_candidate_id"] != cxr["candidate_id"] or row["image_sha256"] != cxr["artifact"]["sha256"]
            or labels["producer"]["checkpoint_sha256"] != checkpoint_sha256
            or labels.get("score_semantics") != "xrv_op_norm_0_1" or labels.get("probability_semantics") is not False
            or labels.get("primary_metric_eligible") is not False
            or labels["calibration"]["source_sha256"] != thresholds_sha256):
        raise ValueError("fresh classifier lineage/score-space/threshold provenance differs")
    xrv = _vector(row["finding_states"])
    thresholds = labels["thresholds"]
    if set(thresholds) != set(FINDINGS): raise ValueError("threshold inventory differs")
    scores = row["finding_probabilities"]  # Legacy field name, not probabilities.
    if set(scores) != set(FINDINGS): raise ValueError("score inventory differs")
    disabled = []
    for name in FINDINGS:
        entry = thresholds[name]
        if type(entry.get("enabled")) is not bool: raise ValueError("explicit calibrated head mask required")
        value = scores[name]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("invalid operating-point-normalized score")
        low, high = entry["negative_max"], entry["positive_min"]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (low, high)) or not 0 <= low <= high <= 1:
            raise ValueError("invalid frozen finding thresholds")
        expected = ("unknown" if value is None or not entry["enabled"] else
            "positive" if value >= high else "negative" if value <= low else "uncertain")
        if xrv[name] != expected: raise ValueError("finding state differs from frozen score/head mask")
        if not entry["enabled"]:
            disabled.append(name)
            if xrv[name] != "unknown": raise ValueError("disabled head invented a comparison")
    states = [{"finding": n, "ehr": s, "xrv": xrv[n]} for n, s, _ in anchor.findings]
    edge = _edge(states, "ehr", "xrv")
    status = ("unverified_no_direct_ehr_constraints" if not edge["known_reference_facts"] else
        "explicit_image_proxy_opposition_unvalidated" if edge["proxy_opposition_facts"] else
        "unverified_missing_image_comparison" if edge["missing_comparisons"] else
        "direct_ehr_image_label_agreement_unvalidated")
    payload = {"schema_version": "tricompose-fresh-image-receipt-v1", "profile": PROFILE,
        "case_id": anchor.case_id, "ehr_anchor_sha256": anchor.sha256,
        "cxr_candidate_id": cxr["candidate_id"], "cxr_sha256": cxr["artifact"]["sha256"],
        "xrv_labels_sha256": label_sha256, "thresholds_sha256": thresholds_sha256,
        "classifier_checkpoint_sha256": checkpoint_sha256,
        "disabled_xrv_findings": disabled, "fact_states": states,
        "raw_edge_readouts": {"ehr_cxr": edge, "ehr_report": None, "cxr_report": None},
        "report_lifecycle_status": "not_generated", "report_candidate_id": None, "report_sha256": None,
        "all_three_supported_facts": None, "all_three_support_over_known": None,
        "verification_status": status, "image_validity_scope": "hash_and_png_dimensions_metadata_not_anatomy",
        "clinical_acceptance": False, "clinical_repair_success": False, "clinical_accuracy": None,
        "primary_clinical_metric_eligible": False, "model_execution_allowed_by_receipt": False}
    return {**payload, "receipt_id": _digest(payload)}


def completed_receipt(anchor, partial, cxr, image_labels, report, report_labels, *,
                      image_labels_sha256, report_labels_sha256, thresholds_sha256,
                      xrv_checkpoint_sha256, chexbert_checkpoint_sha256):
    expected = image_receipt(anchor, cxr, image_labels, label_sha256=image_labels_sha256,
        thresholds_sha256=thresholds_sha256, checkpoint_sha256=xrv_checkpoint_sha256)
    if _digest(expected) != _digest(partial): raise ValueError("previous image receipt changed")
    if (report["case_id"] != anchor.case_id or report["parent_cxr_candidate_id"] != cxr["candidate_id"]
            or report["input_cxr"]["sha256"] != cxr["artifact"]["sha256"]
            or report["ehr_sha256_retained_for_lineage"] != anchor.ehr_sha256
            or report["ehr_facts_sha256_retained_for_lineage"] != anchor.ehr_facts_sha256
            or report.get("model_input_signature") != "single_current_synthetic_cxr"
            or report.get("structured_ehr_content_supplied_to_model") is not False
            or report.get("source_report_or_real_target_supplied") is not False
            or report.get("frozen_model") is not True):
        raise ValueError("report changed fixed EHR/image or supplied a real target")
    _hash(report_labels_sha256); _hash(chexbert_checkpoint_sha256); _hash(report["artifact"]["sha256"])
    if not _ID.fullmatch(report["candidate_id"]): raise ValueError("opaque report ID required")
    row = _single(report_labels, "tricompose-report-finding-labels-v1.1", "reports")
    if (row["report_candidate_id"] != report["candidate_id"] or row["report_sha256"] != report["artifact"]["sha256"]
            or report_labels["producer"]["checkpoint_sha256"] != chexbert_checkpoint_sha256
            or report_labels["calibration"].get("unknown_is_negative") is not False):
        raise ValueError("report label checkpoint/lineage/missingness differs")
    text = _vector(row["finding_states"])
    states = [{**f, "chexbert": text[f["finding"]]} for f in partial["fact_states"]]
    edges = {"ehr_cxr": _edge(states, "ehr", "xrv"), "ehr_report": _edge(states, "ehr", "chexbert"),
        "cxr_report": _edge(states, "xrv", "chexbert")}
    known = edges["ehr_cxr"]["known_reference_facts"]
    observed = [f for f in states if all(f[k] in EXPLICIT for k in ("ehr", "xrv", "chexbert"))]
    supported = sum(f["ehr"] == f["xrv"] == f["chexbert"] for f in observed)
    status = ("unverified_no_direct_ehr_constraints" if not known else
        "explicit_proxy_opposition_unvalidated" if any(e["proxy_opposition_facts"] for e in edges.values()) else
        "unverified_missing_direct_ehr_comparison" if len(observed) < known else
        "direct_ehr_label_agreement_unvalidated")
    payload = {"schema_version": "tricompose-fresh-completed-receipt-v1", "profile": PROFILE,
        "case_id": anchor.case_id, "ehr_anchor_sha256": anchor.sha256,
        "partial_receipt_id": partial["receipt_id"], "cxr_candidate_id": cxr["candidate_id"],
        "report_candidate_id": report["candidate_id"], "cxr_sha256": cxr["artifact"]["sha256"],
        "report_sha256": report["artifact"]["sha256"], "xrv_labels_sha256": image_labels_sha256,
        "chexbert_labels_sha256": report_labels_sha256, "thresholds_sha256": thresholds_sha256,
        "xrv_checkpoint_sha256": xrv_checkpoint_sha256, "chexbert_checkpoint_sha256": chexbert_checkpoint_sha256,
        "fact_states": states, "raw_edge_readouts": edges, "known_ehr_facts": known,
        "all_three_supported_facts": supported, "all_three_support_over_known": supported/known if known else None,
        "report_lifecycle_status": "generated_and_labelled", "verification_status": status,
        "report_assertion_scope": "raw_chexbert_unverified_not_guarded_assertions",
        "clinical_acceptance": False, "clinical_repair_success": False, "clinical_accuracy": None,
        "primary_clinical_metric_eligible": False, "model_execution_allowed_by_receipt": False}
    return {**payload, "receipt_id": _digest(payload)}
