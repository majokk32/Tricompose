"""Frozen-score threshold selection, not model/probability calibration.

Pure functions accept already extracted scores and explicit reference labels.
No model, dataset, network, or filesystem access occurs here. Real validation
inputs must be prepared/consumed within an approved protected Slurm job.
Threshold fitting alone never makes a metric paper-primary.
"""
from __future__ import annotations

import math
import re

from contracts import CHEXPERT_FINDINGS

BUNDLE_SCHEMA = "tricompose-xrv-thresholds-v2"
SCORE_SPACE = "xrv_op_norm_0_1"
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value):
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise ValueError("missing or invalid provenance SHA256")
    return value


def choose_threshold(scores, states, *, min_per_class=20, uncertainty_margin=0.0):
    """Maximize balanced accuracy, then minimize distance to 0.5, then threshold.

    Unknown/uncertain reference states and absent scores are excluded, not
    negatives. Insufficient support disables this finding rather than falling
    back to an uncalibrated decision. The uncertainty margin is prespecified.
    """
    if len(scores) != len(states) or isinstance(min_per_class, bool) or not isinstance(min_per_class, int) or min_per_class < 1:
        raise ValueError("invalid threshold-selection inputs")
    if not math.isfinite(uncertainty_margin) or not 0 <= uncertainty_margin < 0.5:
        raise ValueError("invalid prespecified uncertainty margin")
    pairs = []
    for score, state in zip(scores, states, strict=True):
        if state not in {"positive", "negative", "uncertain", "unknown"}:
            raise ValueError("invalid reference state")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid frozen classifier score")
        if score is not None and state in {"positive", "negative"}:
            pairs.append((float(score), int(state == "positive")))
    positive = sum(y for _, y in pairs)
    negative = len(pairs) - positive
    common = {"positive_count": positive, "negative_count": negative,
              "excluded_count": len(scores) - len(pairs)}
    if min(positive, negative) < min_per_class:
        return {**common, "status": "insufficient_binary_support", "enabled": False,
                "negative_max": 0.5, "positive_min": 0.5, "balanced_accuracy": None}
    unique = sorted({score for score, _ in pairs})
    thresholds = sorted({0.0, 0.5, 1.0, *unique,
                         *((a + b) / 2 for a, b in zip(unique, unique[1:]))})
    def accuracy(threshold):
        tp = sum(y == 1 and score >= threshold for score, y in pairs)
        tn = sum(y == 0 and score < threshold for score, y in pairs)
        return (tp / positive + tn / negative) / 2
    chosen = min(thresholds, key=lambda t: (-accuracy(t), abs(t - 0.5), t))
    return {**common, "status": "threshold_fitted", "enabled": True,
            "negative_max": max(0.0, chosen - uncertainty_margin),
            "positive_min": min(1.0, chosen + uncertainty_margin),
            "balanced_accuracy": accuracy(chosen),
            "fit_statistic_is_heldout_performance": False}


def validate_provenance(provenance):
    for key in ("checkpoint_sha256", "preprocessing_sha256", "finding_mapping_sha256",
                "calibration_input_sha256", "split_manifest_sha256", "reference_sha256"):
        _hash(provenance.get(key))
    if provenance.get("score_space") != SCORE_SPACE:
        raise ValueError("thresholds must use the exact frozen scorer output space")
    if provenance.get("split_role") != "calibration" or provenance.get("source_kind") != "matched_real_validation":
        raise ValueError("requires a declared matched real calibration split")
    if provenance.get("reference_kind") not in {"adjudicated_image_labels", "dataset_image_labels", "report_extracted_weak_labels"}:
        raise ValueError("reference label provenance is required")
    return dict(provenance)


def fit_bundle(records, provenance, *, heldout_group_hashes, min_per_class=20, uncertainty_margin=0.0):
    """Validate patient disjointness and fit per-finding operating points.

    Records use opaque per-study hashes and protected patient-group hashes;
    never put raw patient identifiers in this contract. No records are returned
    in the threshold bundle. Reference quality remains a reviewed assumption.
    """
    provenance = validate_provenance(provenance)
    if not records or not heldout_group_hashes:
        raise ValueError("calibration and reserved held-out groups are required")
    heldout = {_hash(value) for value in heldout_group_hashes}
    groups, seen = set(), set()
    for row in records:
        sample = _hash(row.get("sample_sha256"))
        if sample in seen:
            raise ValueError("duplicate calibration study")
        seen.add(sample)
        groups.add(_hash(row.get("patient_group_sha256")))
        if set(row.get("scores", {})) != set(CHEXPERT_FINDINGS) or set(row.get("reference_states", {})) != set(CHEXPERT_FINDINGS):
            raise ValueError("incomplete calibration finding vectors")
    if groups & heldout:
        raise ValueError("patient overlap between calibration and reserved held-out splits")
    findings = {
        finding: choose_threshold([r["scores"][finding] for r in records],
                                  [r["reference_states"][finding] for r in records],
                                  min_per_class=min_per_class, uncertainty_margin=uncertainty_margin)
        for finding in CHEXPERT_FINDINGS
    }
    return {"schema_version": BUNDLE_SCHEMA,
            "calibration_status": "thresholds_fitted_pending_independent_review",
            "primary_metric_eligible": False,
            "probability_calibration_performed": False,
            "provenance": provenance,
            "counts": {"studies": len(records), "calibration_patient_groups": len(groups),
                       "reserved_heldout_patient_groups": len(heldout)},
            "selection_rule": "balanced_accuracy_then_nearest_0_5_then_lowest_v1",
            "min_per_class": min_per_class, "uncertainty_margin": uncertainty_margin,
            "findings": findings}


def validate_bundle(payload, *, checkpoint_sha256=None, expected_provenance=None):
    if payload.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unsupported threshold bundle")
    provenance = validate_provenance(payload.get("provenance", {}))
    if checkpoint_sha256 is not None and provenance["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("threshold/checkpoint mismatch")
    for key, value in (expected_provenance or {}).items():
        if provenance.get(key) != value:
            raise ValueError("threshold/scorer-protocol mismatch")
    if payload.get("primary_metric_eligible") is not False:
        raise ValueError("threshold fitting alone cannot confer paper eligibility")
    findings = payload.get("findings", {})
    if set(findings) != set(CHEXPERT_FINDINGS):
        raise ValueError("incomplete threshold inventory")
    for row in findings.values():
        low, high = row.get("negative_max"), row.get("positive_min")
        if not isinstance(row.get("enabled"), bool):
            raise ValueError("missing finding enable mask")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (low, high)) or not 0 <= low <= high <= 1:
            raise ValueError("invalid finding thresholds")
        if row["enabled"] and row.get("status") != "threshold_fitted":
            raise ValueError("unfitted finding cannot be enabled")
    return findings
