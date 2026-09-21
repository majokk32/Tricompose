"""Deterministic metrics shared by EHR-report and report-CXR evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from contracts import CHEXPERT_FINDINGS, validate_finding_states


EXPLICIT_STATES = frozenset({"positive", "negative"})

# Only direct radiographic facts are used for the primary EHR edge.  CHF and
# clinical-context diagnoses are not converted into asserted image findings.
DIRECT_EHR_TO_CHEXPERT = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "lung_opacity": "lung_opacity",
    "pleural_effusion": "pleural_effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "pulmonary_edema": "edema",
}
DEVICE_FACTS = (
    "endotracheal_tube",
    "central_venous_catheter",
    "enteric_tube",
    "cardiac_pacemaker",
)


def ehr_direct_finding_states(facts_payload: Mapping[str, Any]) -> dict[str, str]:
    if facts_payload.get("schema_version") != "tricompose-facts-v1.1":
        raise ValueError("EHR facts are not V1.1")
    if facts_payload.get("missing_is_unknown") is not True:
        raise ValueError("EHR facts do not preserve unknown semantics")
    direct = facts_payload.get("direct_facts")
    if not isinstance(direct, dict):
        raise TypeError("EHR direct facts are missing")
    states = {finding: "unknown" for finding in CHEXPERT_FINDINGS}
    for fact_id, finding in DIRECT_EHR_TO_CHEXPERT.items():
        fact = direct.get(fact_id)
        if not isinstance(fact, dict):
            continue
        state = str(fact.get("state"))
        if state not in {"positive", "negative", "uncertain", "unknown"}:
            raise ValueError(f"invalid EHR fact state: {fact_id}")
        states[finding] = state

    device_states = [
        str(direct.get(fact_id, {}).get("state", "unknown")) for fact_id in DEVICE_FACTS
    ]
    if "positive" in device_states:
        states["support_devices"] = "positive"
    elif device_states and all(state == "negative" for state in device_states):
        states["support_devices"] = "negative"
    elif "uncertain" in device_states:
        states["support_devices"] = "uncertain"
    # Otherwise one absent device never means that all support devices are absent.
    return validate_finding_states(states)


def _source_category(field: str) -> str:
    normalized = field.strip().lower()
    if normalized.startswith("diagnoses["):
        return "diagnosis"
    if normalized.startswith(("medications[", "medication[", "meds[")):
        return "medication"
    if normalized.startswith(("labs[", "laboratory[")):
        return "lab"
    if normalized.startswith(("vitals[", "vital_signs[", "charts[")):
        return "vital"
    return "other"


def ehr_direct_states_by_source(
    facts_payload: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Mask direct EHR finding states by their preserved provenance type."""

    overall = ehr_direct_finding_states(facts_payload)
    direct = facts_payload["direct_facts"]
    categories = ("diagnosis", "medication", "lab", "vital", "other")
    output = {
        category: {finding: "unknown" for finding in CHEXPERT_FINDINGS}
        for category in categories
    }
    finding_sources: dict[str, set[str]] = defaultdict(set)
    for fact_id, finding in DIRECT_EHR_TO_CHEXPERT.items():
        fact = direct.get(fact_id, {})
        for field in fact.get("source_fields", []):
            finding_sources[finding].add(_source_category(str(field)))
    for fact_id in DEVICE_FACTS:
        fact = direct.get(fact_id, {})
        for field in fact.get("source_fields", []):
            finding_sources["support_devices"].add(_source_category(str(field)))
    for finding, state in overall.items():
        if state == "unknown":
            continue
        sources = finding_sources.get(finding) or {"other"}
        for category in sources:
            output[category][finding] = state
    return {
        category: validate_finding_states(states)
        for category, states in output.items()
    }


def relation(reference_state: str, candidate_state: str) -> str:
    if reference_state not in EXPLICIT_STATES or candidate_state not in EXPLICIT_STATES:
        return "unknown"
    return "support" if reference_state == candidate_state else "contradiction"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 8)


def _binary_f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else round(2 * tp / denominator, 8)


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[state] / len(pairs) * right_counts[state] / len(pairs)
        for state in EXPLICIT_STATES
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return round((observed - expected) / (1.0 - expected), 8)


def summarize_state_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    reference_key: str,
    candidate_key: str,
) -> dict[str, Any]:
    """Summarize finding pairs while preserving unknown/uncertain coverage."""

    rows = list(records)
    known_reference = 0
    candidate_explicit_on_known = 0
    support = 0
    contradiction = 0
    unknown_or_uncertain_candidate = 0
    comparable_pairs: list[tuple[str, str]] = []
    per_finding_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    confusion = Counter()
    report_positive_reference_unknown = 0
    report_positive_total = 0

    for row in rows:
        reference = validate_finding_states(row[reference_key])
        candidate = validate_finding_states(row[candidate_key])
        for finding in CHEXPERT_FINDINGS:
            left, right = reference[finding], candidate[finding]
            if right == "positive":
                report_positive_total += 1
                if left == "unknown":
                    report_positive_reference_unknown += 1
            if left not in EXPLICIT_STATES:
                continue
            known_reference += 1
            if right not in EXPLICIT_STATES:
                unknown_or_uncertain_candidate += 1
                continue
            candidate_explicit_on_known += 1
            comparable_pairs.append((left, right))
            per_finding_rows[finding].append((left, right))
            current_relation = relation(left, right)
            support += int(current_relation == "support")
            contradiction += int(current_relation == "contradiction")
            if left == "positive" and right == "positive":
                confusion["tp"] += 1
            elif left == "negative" and right == "positive":
                confusion["fp"] += 1
            elif left == "positive" and right == "negative":
                confusion["fn"] += 1
            else:
                confusion["tn"] += 1

    per_finding: dict[str, dict[str, Any]] = {}
    macro_f1_values: list[float] = []
    for finding in CHEXPERT_FINDINGS:
        pairs = per_finding_rows.get(finding, [])
        local = Counter()
        for left, right in pairs:
            if left == "positive" and right == "positive":
                local["tp"] += 1
            elif left == "negative" and right == "positive":
                local["fp"] += 1
            elif left == "positive" and right == "negative":
                local["fn"] += 1
            else:
                local["tn"] += 1
        f1 = _binary_f1(local["tp"], local["fp"], local["fn"])
        if f1 is not None:
            macro_f1_values.append(f1)
        per_finding[finding] = {
            "comparable_count": len(pairs),
            "support_count": sum(left == right for left, right in pairs),
            "contradiction_count": sum(left != right for left, right in pairs),
            "precision_positive": _safe_ratio(
                local["tp"], local["tp"] + local["fp"]
            ),
            "recall_positive": _safe_ratio(
                local["tp"], local["tp"] + local["fn"]
            ),
            "f1_positive": f1,
        }

    return {
        "record_count": len(rows),
        "known_reference_fact_count": known_reference,
        "comparable_explicit_fact_count": candidate_explicit_on_known,
        "known_fact_coverage": _safe_ratio(candidate_explicit_on_known, known_reference),
        "support_count": support,
        "support_recall_over_all_known_facts": _safe_ratio(support, known_reference),
        "explicit_contradiction_count": contradiction,
        "explicit_contradiction_rate_over_known_facts": _safe_ratio(
            contradiction, known_reference
        ),
        "explicit_contradiction_rate_over_comparable_facts": _safe_ratio(
            contradiction, candidate_explicit_on_known
        ),
        "unknown_or_uncertain_candidate_count": unknown_or_uncertain_candidate,
        "exact_agreement_on_comparable_facts": _safe_ratio(
            support, candidate_explicit_on_known
        ),
        "micro_precision_positive": _safe_ratio(
            confusion["tp"], confusion["tp"] + confusion["fp"]
        ),
        "micro_recall_positive": _safe_ratio(
            confusion["tp"], confusion["tp"] + confusion["fn"]
        ),
        "micro_f1_positive": _binary_f1(
            confusion["tp"], confusion["fp"], confusion["fn"]
        ),
        "macro_f1_positive": (
            None
            if not macro_f1_values
            else round(sum(macro_f1_values) / len(macro_f1_values), 8)
        ),
        "cohen_kappa_on_comparable_states": _cohen_kappa(comparable_pairs),
        "candidate_positive_reference_unknown_count": report_positive_reference_unknown,
        "candidate_positive_reference_unknown_rate": _safe_ratio(
            report_positive_reference_unknown, report_positive_total
        ),
        "per_finding": per_finding,
    }


def score_state_pair(
    reference_states: Mapping[str, Any],
    candidate_states: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an auditable score for one exact cross-modal edge.

    The support score uses every known reference fact as its denominator, so
    an omitted/unknown candidate statement does not receive positive credit.
    Explicit contradiction is kept separate from missing coverage.
    """

    reference = validate_finding_states(reference_states)
    candidate = validate_finding_states(candidate_states)
    known = 0
    comparable = 0
    support = 0
    contradiction = 0
    for finding in CHEXPERT_FINDINGS:
        left, right = reference[finding], candidate[finding]
        if left not in EXPLICIT_STATES:
            continue
        known += 1
        if right not in EXPLICIT_STATES:
            continue
        comparable += 1
        if left == right:
            support += 1
        else:
            contradiction += 1
    support_recall = _safe_ratio(support, known)
    contradiction_rate = _safe_ratio(contradiction, known)
    return {
        "known_reference_fact_count": known,
        "comparable_explicit_fact_count": comparable,
        "coverage": _safe_ratio(comparable, known),
        "support_count": support,
        "support_recall": support_recall,
        "contradiction_count": contradiction,
        "contradiction_rate": contradiction_rate,
        "agreement_on_comparable": _safe_ratio(support, comparable),
        "edge_score": (
            None
            if support_recall is None or contradiction_rate is None
            else round(support_recall - contradiction_rate, 8)
        ),
    }


def pairwise_state_disagreement(
    state_vectors: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure candidate disagreement without treating unknown as negative."""

    vectors = [validate_finding_states(states) for states in state_vectors]
    comparable = 0
    disagreements = 0
    pair_count = 0
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            pair_count += 1
            for finding in CHEXPERT_FINDINGS:
                if left[finding] not in EXPLICIT_STATES:
                    continue
                if right[finding] not in EXPLICIT_STATES:
                    continue
                comparable += 1
                disagreements += int(left[finding] != right[finding])
    return {
        "candidate_count": len(vectors),
        "candidate_pair_count": pair_count,
        "comparable_fact_pair_count": comparable,
        "disagreement_count": disagreements,
        "disagreement_rate": _safe_ratio(disagreements, comparable),
    }


def report_cxr_error_counts(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    rows = list(records)
    cxr_positive = 0
    report_omitted = 0
    explicit_omission = 0
    cxr_negative = 0
    hallucinated = 0
    for row in rows:
        cxr = validate_finding_states(row["cxr_finding_states"])
        report = validate_finding_states(row["report_finding_states"])
        for finding in CHEXPERT_FINDINGS:
            if cxr[finding] == "positive":
                cxr_positive += 1
                if report[finding] != "positive":
                    report_omitted += 1
                if report[finding] == "negative":
                    explicit_omission += 1
            elif cxr[finding] == "negative":
                cxr_negative += 1
                if report[finding] == "positive":
                    hallucinated += 1
    return {
        "cxr_positive_fact_count": cxr_positive,
        "omitted_finding_count_including_unmentioned": report_omitted,
        "omitted_finding_rate_including_unmentioned": _safe_ratio(
            report_omitted, cxr_positive
        ),
        "explicit_opposite_omission_count": explicit_omission,
        "explicit_opposite_omission_rate": _safe_ratio(explicit_omission, cxr_positive),
        "cxr_negative_fact_count": cxr_negative,
        "hallucinated_positive_finding_count": hallucinated,
        "hallucinated_positive_finding_rate": _safe_ratio(hallucinated, cxr_negative),
    }


__all__ = [
    "DIRECT_EHR_TO_CHEXPERT",
    "DEVICE_FACTS",
    "ehr_direct_finding_states",
    "ehr_direct_states_by_source",
    "pairwise_state_disagreement",
    "relation",
    "report_cxr_error_counts",
    "score_state_pair",
    "summarize_state_pairs",
]
