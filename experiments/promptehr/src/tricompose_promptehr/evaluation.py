"""Reference-free smoke evaluation for protected PromptEHR outputs."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from typing import Any


MODALITIES = ("diag", "prod", "med")


def _sequence_digest(case: dict[str, Any]) -> str:
    canonical = json.dumps(case["event_indices"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize validity, coverage, diversity, and prompt overlap.

    This is intentionally reference-free. It does not claim fidelity to MIMIC-III
    or MIMIC-IV and it does not run downstream prediction tasks.
    """

    if not cases:
        raise ValueError("at least one case is required")

    valid_count = sum(bool(case["validation"]["strict_valid"]) for case in cases)
    sequence_digests = [_sequence_digest(case) for case in cases]
    digest_counts = Counter(sequence_digests)
    visit_counts = [len(case["event_indices"]) for case in cases]
    modality_summary: dict[str, dict[str, float | int]] = {}

    total_output = 0
    total_overlap = 0
    total_novel = 0
    for modality in MODALITIES:
        event_values: list[int] = []
        nonempty_visits = 0
        total_visits = 0
        covered_cases = 0
        modality_overlap = 0
        modality_novel = 0
        modality_output = 0
        for case in cases:
            case_events: list[int] = []
            for visit in case["event_indices"]:
                values = [int(value) for value in visit[modality]]
                case_events.extend(values)
                event_values.extend(values)
                total_visits += 1
                nonempty_visits += int(bool(values))
            covered_cases += int(bool(case_events))
            counts = case["prompt_comparison"][modality]
            modality_overlap += int(counts["overlap_count"])
            modality_novel += int(counts["novel_output_count"])
            modality_output += int(counts["output_count"])

        total_output += modality_output
        total_overlap += modality_overlap
        total_novel += modality_novel
        modality_summary[modality] = {
            "case_coverage": covered_cases / len(cases),
            "nonempty_visit_rate": nonempty_visits / total_visits if total_visits else 0.0,
            "total_event_count": len(event_values),
            "unique_event_count": len(set(event_values)),
            "prompt_overlap_fraction": (
                modality_overlap / modality_output if modality_output else 0.0
            ),
            "novel_output_fraction": (
                modality_novel / modality_output if modality_output else 0.0
            ),
        }

    return {
        "scope": "reference_free_official_synthetic_seed_smoke",
        "case_count": len(cases),
        "strict_valid_rate": valid_count / len(cases),
        "unique_sequence_count": len(digest_counts),
        "unique_sequence_rate": len(digest_counts) / len(cases),
        "exact_duplicate_count": sum(count - 1 for count in digest_counts.values()),
        "visit_count": {
            "min": min(visit_counts),
            "median": statistics.median(visit_counts),
            "max": max(visit_counts),
        },
        "modalities": modality_summary,
        "prompt_overlap_fraction": total_overlap / total_output if total_output else 0.0,
        "novel_output_fraction": total_novel / total_output if total_output else 0.0,
        "fidelity_claimed": False,
        "downstream_evaluation_run": False,
    }

