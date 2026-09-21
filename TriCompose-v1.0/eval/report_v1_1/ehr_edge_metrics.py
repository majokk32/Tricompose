"""Edge-specific EHR-CXR and EHR-report metrics for TriCompose V1.1.

The module deliberately separates direct radiographic evidence from weak
clinical priors. A weak prior may receive support or an incompatibility flag,
but it never becomes a hard contradiction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from contracts import CHEXPERT_FINDINGS, validate_finding_states
from crossmodal_metrics import ehr_direct_finding_states, score_state_pair


PRIMARY_EHR_CXR_FINDINGS = (
    "edema",
    "pleural_effusion",
    "cardiomegaly",
    "pneumonia",
    "pneumothorax",
    "support_devices",
)
SOURCE_CATEGORIES = ("diagnosis", "medication", "lab", "vital", "other")
DIURETIC_PATTERNS = (
    "furosemide",
    "lasix",
    "bumetanide",
    "bumex",
    "torsemide",
    "loop diuretic",
)


def _source_category(field: str) -> str:
    normalized = field.strip().lower()
    if normalized.startswith("diagnoses["):
        return "diagnosis"
    if normalized.startswith("medications["):
        return "medication"
    if normalized.startswith("labs["):
        return "lab"
    if normalized.startswith("vitals["):
        return "vital"
    return "other"


def strict_states_by_source(facts: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Mask direct radiographic states by preserved structured-EHR source."""

    overall = ehr_direct_finding_states(facts)
    direct = facts["direct_facts"]
    fact_to_finding = {
        "atelectasis": "atelectasis",
        "cardiomegaly": "cardiomegaly",
        "consolidation": "consolidation",
        "lung_opacity": "lung_opacity",
        "pleural_effusion": "pleural_effusion",
        "pneumonia": "pneumonia",
        "pneumothorax": "pneumothorax",
        "pulmonary_edema": "edema",
        "endotracheal_tube": "support_devices",
        "central_venous_catheter": "support_devices",
        "enteric_tube": "support_devices",
        "cardiac_pacemaker": "support_devices",
    }
    output = {
        source: {finding: "unknown" for finding in CHEXPERT_FINDINGS}
        for source in SOURCE_CATEGORIES
    }
    sources_by_finding: dict[str, set[str]] = defaultdict(set)
    for fact_id, finding in fact_to_finding.items():
        fact = direct.get(fact_id, {})
        for field in fact.get("source_fields", []):
            sources_by_finding[finding].add(_source_category(str(field)))
    for finding, state in overall.items():
        if state == "unknown":
            continue
        for source in sources_by_finding.get(finding, {"other"}):
            output[source][finding] = state
    return {source: validate_finding_states(states) for source, states in output.items()}


def _latest_medication_surfaces(canonical_ehr: Mapping[str, Any]) -> list[str]:
    latest = len(canonical_ehr["timeline"]) - 1
    return [
        " ".join(
            str(event.get(key, "")) for key in ("code", "description", "token")
        ).lower()
        for event in canonical_ehr["medications"]
        if event.get("visit_index") == latest
    ]


def weak_clinical_priors(
    facts: Mapping[str, Any], canonical_ehr: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return support-only rules from the supplied project metric document."""

    rules: list[dict[str, Any]] = []
    chf = facts["direct_facts"].get("congestive_heart_failure", {})
    if chf.get("state") == "positive":
        rules.append(
            {
                "rule_id": "heart_failure_to_cardiomegaly_or_edema_weak_v1",
                "source_category": "diagnosis",
                "target_findings": ["cardiomegaly", "edema"],
                "semantics": "support_only_not_a_required_radiographic_finding",
            }
        )
    medications = _latest_medication_surfaces(canonical_ehr)
    if any(pattern in surface for pattern in DIURETIC_PATTERNS for surface in medications):
        rules.append(
            {
                "rule_id": "loop_diuretic_to_edema_weak_v1",
                "source_category": "medication",
                "target_findings": ["edema"],
                "semantics": "support_only_not_a_required_radiographic_finding",
            }
        )
    return rules


def score_weak_priors(
    rules: Iterable[Mapping[str, Any]],
    candidate_states: Mapping[str, Any],
    probabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score weak evidence without promoting it to hard ground truth."""

    states = validate_finding_states(candidate_states)
    rows: list[dict[str, Any]] = []
    for rule in rules:
        targets = [str(value) for value in rule["target_findings"]]
        target_states = {finding: states[finding] for finding in targets}
        if any(state == "positive" for state in target_states.values()):
            relation = "weak_support"
        elif target_states and all(state == "negative" for state in target_states.values()):
            relation = "weak_incompatibility_not_hard_contradiction"
        else:
            relation = "unknown"
        available_probabilities = []
        if probabilities is not None:
            for finding in targets:
                value = probabilities.get(finding)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    available_probabilities.append(float(value))
        rows.append(
            {
                "rule_id": rule["rule_id"],
                "source_category": rule["source_category"],
                "target_findings": targets,
                "relation": relation,
                "maximum_target_probability": (
                    None if not available_probabilities else round(max(available_probabilities), 8)
                ),
            }
        )
    return {
        "rule_count": len(rows),
        "weak_support_count": sum(row["relation"] == "weak_support" for row in rows),
        "weak_incompatibility_count": sum(
            row["relation"] == "weak_incompatibility_not_hard_contradiction"
            for row in rows
        ),
        "unknown_count": sum(row["relation"] == "unknown" for row in rows),
        "rules": rows,
    }


def score_direct_ehr_report(
    ehr_states: Mapping[str, Any], report_states: Mapping[str, Any]
) -> dict[str, Any]:
    """Score hard EHR-report contradictions, including explicit No Finding."""

    reference = validate_finding_states(ehr_states)
    report = validate_finding_states(report_states)
    adjusted = dict(report)
    no_finding_conflicts: list[str] = []
    if report["no_finding"] == "positive":
        for finding in CHEXPERT_FINDINGS:
            if finding in {"no_finding", "support_devices"}:
                continue
            if reference[finding] == "positive" and report[finding] not in {
                "positive",
                "negative",
            }:
                adjusted[finding] = "negative"
                no_finding_conflicts.append(finding)
    result = score_state_pair(reference, adjusted)
    result["no_finding_cross_contradiction_findings"] = no_finding_conflicts
    result["hard_contradiction_semantics"] = (
        "explicit_opposite_same_finding_or_positive_no_finding_statement"
    )
    return result


def _binary_metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not labels:
        return {
            "status": "not_applicable_no_explicit_binary_reference",
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "ece_10bin": None,
        }
    brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities, strict=True)) / len(labels)
    bins: list[list[tuple[int, float]]] = [[] for _ in range(10)]
    for label, probability in zip(labels, probabilities, strict=True):
        index = min(9, int(probability * 10))
        bins[index].append((label, probability))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        accuracy = sum(label for label, _ in bucket) / len(bucket)
        confidence = sum(probability for _, probability in bucket) / len(bucket)
        ece += len(bucket) / len(labels) * abs(accuracy - confidence)
    if positives == 0 or negatives == 0:
        return {
            "status": "diagnostic_calibration_only_single_reference_class",
            "count": len(labels),
            "positive_count": positives,
            "negative_count": negatives,
            "auroc": None,
            "auprc": None,
            "brier": round(brier, 8),
            "ece_10bin": round(ece, 8),
        }
    positive_probabilities = [p for y, p in zip(labels, probabilities, strict=True) if y == 1]
    negative_probabilities = [p for y, p in zip(labels, probabilities, strict=True) if y == 0]
    wins = sum(
        1.0 if positive > negative else 0.5 if math.isclose(positive, negative) else 0.0
        for positive in positive_probabilities
        for negative in negative_probabilities
    )
    auroc = wins / (positives * negatives)
    ordered = sorted(zip(probabilities, labels, strict=True), reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return {
        "status": "computed_diagnostic_uncalibrated_classifier",
        "count": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "auroc": round(auroc, 8),
        "auprc": round(precision_sum / positives, 8),
        "brier": round(brier, 8),
        "ece_10bin": round(ece, 8),
    }


def probability_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    finding_order: Iterable[str] = PRIMARY_EHR_CXR_FINDINGS,
) -> dict[str, Any]:
    """Compute AUROC/AUPRC/Brier/ECE only where explicit binary EHR labels exist."""

    rows = list(records)
    output: dict[str, Any] = {}
    pooled_labels: list[int] = []
    pooled_probabilities: list[float] = []
    for finding in finding_order:
        labels: list[int] = []
        probabilities: list[float] = []
        for row in rows:
            state = row["ehr_finding_states"][finding]
            probability = row["cxr_finding_probabilities"].get(finding)
            if state not in {"positive", "negative"}:
                continue
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                continue
            labels.append(int(state == "positive"))
            probabilities.append(float(probability))
        output[finding] = _binary_metrics(labels, probabilities)
        pooled_labels.extend(labels)
        pooled_probabilities.extend(probabilities)
    return {
        "interpretation": (
            "Diagnostic only until evaluated on a held-out explicit-label cohort; "
            "weak priors are excluded from AUROC/AUPRC/Brier/ECE."
        ),
        "overall": _binary_metrics(pooled_labels, pooled_probabilities),
        "per_disease": output,
    }


__all__ = [
    "PRIMARY_EHR_CXR_FINDINGS",
    "SOURCE_CATEGORIES",
    "probability_metrics",
    "score_direct_ehr_report",
    "score_weak_priors",
    "strict_states_by_source",
    "weak_clinical_priors",
]
