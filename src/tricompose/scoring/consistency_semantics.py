"""Reference-free support/contradiction/unknown aggregation semantics."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


RELATIONS = frozenset({"support", "contradiction", "unknown"})
SEVERITIES = frozenset({"none", "weak", "strong"})


def summarize_fact_judgments(
    judgments: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize protected fact judgments without treating unknown as negative."""

    relation_counts: Counter[str] = Counter()
    strong_contradictions = 0
    total = 0
    for judgment in judgments:
        relation = judgment.get("relation")
        severity = judgment.get("severity", "none")
        if relation not in RELATIONS:
            raise ValueError("invalid consistency relation")
        if severity not in SEVERITIES:
            raise ValueError("invalid contradiction severity")
        if relation != "contradiction" and severity == "strong":
            raise ValueError("strong severity is reserved for contradictions")
        relation_counts[str(relation)] += 1
        strong_contradictions += int(
            relation == "contradiction" and severity == "strong"
        )
        total += 1

    comparable = relation_counts["support"] + relation_counts["contradiction"]
    support_rate = (
        None if comparable == 0 else relation_counts["support"] / comparable
    )
    contradiction_rate = (
        None if comparable == 0 else relation_counts["contradiction"] / comparable
    )
    return {
        "fact_count": total,
        "support_count": relation_counts["support"],
        "contradiction_count": relation_counts["contradiction"],
        "unknown_count": relation_counts["unknown"],
        "comparable_count": comparable,
        "support_rate_among_comparable": (
            None if support_rate is None else round(support_rate, 8)
        ),
        "contradiction_rate_among_comparable": (
            None if contradiction_rate is None else round(contradiction_rate, 8)
        ),
        "strong_contradiction_count": strong_contradictions,
        "hard_gate_pass": strong_contradictions == 0,
    }

