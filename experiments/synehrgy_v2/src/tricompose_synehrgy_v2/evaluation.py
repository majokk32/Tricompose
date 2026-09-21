"""Reference-free smoke metrics for fully synthetic SynEHRgy outputs."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any, Iterable


SECTION_NAMES = ("covariates", "problems", "labs", "charts")


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    for index in range(len(tokens) - n + 1):
        yield tuple(tokens[index : index + n])


def evaluate_synthetic_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute small-cohort validity and diversity metrics.

    These metrics require no real reference cohort and therefore do not claim
    distributional fidelity or downstream utility.
    """

    if not cases:
        raise ValueError("at least one synthetic case is required")
    sequence_keys = [tuple(case["tokens"]) for case in cases]
    unique_sequence_count = len(set(sequence_keys))
    visit_counts = [len(case["structure"]["visits"]) for case in cases]
    token_lengths = [len(case["tokens"]) for case in cases]
    adjacent_repeat_fractions: list[float] = []
    for tokens in (case["tokens"] for case in cases):
        denominator = max(1, len(tokens) - 1)
        repeats = sum(left == right for left, right in zip(tokens, tokens[1:]))
        adjacent_repeat_fractions.append(repeats / denominator)

    total_visits = sum(visit_counts)
    section_metrics: dict[str, Any] = {}
    for section in SECTION_NAMES:
        per_case = [
            any(visit[section] for visit in case["structure"]["visits"])
            for case in cases
        ]
        visit_sequences = [
            visit[section]
            for case in cases
            for visit in case["structure"]["visits"]
        ]
        nonempty_visits = sum(bool(sequence) for sequence in visit_sequences)
        unique_tokens = {
            token for sequence in visit_sequences for token in sequence
        }
        section_metrics[section] = {
            "case_coverage": sum(per_case) / len(cases),
            "nonempty_visit_rate": nonempty_visits / max(1, total_visits),
            "unique_token_count": len(unique_tokens),
        }

    problem_sequences = [
        visit["problems"]
        for case in cases
        for visit in case["structure"]["visits"]
        if visit["problems"]
    ]
    problem_ngram_metrics: dict[str, Any] = {}
    for n in (1, 2, 3):
        counts = Counter(
            ngram for sequence in problem_sequences for ngram in _ngrams(sequence, n)
        )
        total = sum(counts.values())
        problem_ngram_metrics[str(n)] = {
            "total_count": total,
            "unique_count": len(counts),
            "top_ngram_fraction": max(counts.values(), default=0) / max(1, total),
        }

    return {
        "scope": "reference_free_small_cohort_smoke",
        "fidelity_claimed": False,
        "case_count": len(cases),
        "strict_valid_rate": sum(
            bool(case["validation"]["strict_valid"]) for case in cases
        )
        / len(cases),
        "eos_completion_rate": sum(
            bool(case["validation"]["ended_with_eos"]) for case in cases
        )
        / len(cases),
        "unique_sequence_count": unique_sequence_count,
        "unique_sequence_rate": unique_sequence_count / len(cases),
        "exact_duplicate_count": len(cases) - unique_sequence_count,
        "adjacent_repeat_fraction_mean": mean(adjacent_repeat_fractions),
        "token_length": {
            "min": min(token_lengths),
            "median": median(token_lengths),
            "max": max(token_lengths),
        },
        "visit_count": {
            "min": min(visit_counts),
            "median": median(visit_counts),
            "max": max(visit_counts),
        },
        "sections": section_metrics,
        "problem_ngram_diversity": problem_ngram_metrics,
    }

