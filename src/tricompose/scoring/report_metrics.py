"""Compute protected report-to-report BLEU, ROUGE-L, and METEOR scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

from tricompose.privacy import (
    create_private_stage_dir,
    require_private_file,
    sha256_file,
    write_private_json,
)


SCHEMA_VERSION = "tricompose.score_bundle.v1"
PRODUCER_VERSION = "1.0.0"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Use the stable Phase-0 report tokenization contract."""
    return TOKEN_PATTERN.findall(text.lower())


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(max(0, len(tokens) - order + 1))
    )


def _modified_precision(
    candidate: Sequence[str],
    reference: Sequence[str],
    order: int,
) -> float:
    candidate_counts = _ngram_counts(candidate, order)
    if not candidate_counts:
        return 0.0
    reference_counts = _ngram_counts(reference, order)
    clipped = sum(
        min(count, reference_counts[ngram])
        for ngram, count in candidate_counts.items()
    )
    return clipped / sum(candidate_counts.values())


def bleu_n(
    candidate: Sequence[str],
    reference: Sequence[str],
    max_order: int,
    *,
    smoothing: float = 1e-9,
) -> float:
    """Sentence BLEU-N with uniform weights and fixed epsilon smoothing."""
    if not candidate or not reference:
        return 0.0
    precisions = [
        max(_modified_precision(candidate, reference, order), smoothing)
        for order in range(1, max_order + 1)
    ]
    brevity_penalty = (
        1.0
        if len(candidate) > len(reference)
        else math.exp(1.0 - len(reference) / len(candidate))
    )
    return brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l(candidate: Sequence[str], reference: Sequence[str]) -> float:
    """Token-level ROUGE-L F1."""
    if not candidate or not reference:
        return 0.0
    lcs = _lcs_length(candidate, reference)
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )


def _exact_alignment(
    candidate: Sequence[str],
    reference: Sequence[str],
) -> list[tuple[int, int]]:
    """Greedily align exact tokens one-to-one for dependency-free METEOR."""
    unused_reference_indices: dict[str, list[int]] = {}
    for reference_index, token in enumerate(reference):
        unused_reference_indices.setdefault(token, []).append(reference_index)

    alignment: list[tuple[int, int]] = []
    for candidate_index, token in enumerate(candidate):
        indices = unused_reference_indices.get(token)
        if indices:
            alignment.append((candidate_index, indices.pop(0)))
    return alignment


def meteor(candidate: Sequence[str], reference: Sequence[str]) -> float:
    """Exact-token METEOR with the standard fragmentation penalty.

    This dependency-free Phase-0 variant does not add stem or WordNet synonym
    matches. The output bundle records that limitation explicitly.
    """
    if not candidate or not reference:
        return 0.0
    alignment = _exact_alignment(candidate, reference)
    matches = len(alignment)
    if matches == 0:
        return 0.0

    precision = matches / len(candidate)
    recall = matches / len(reference)
    harmonic = (10.0 * precision * recall) / (recall + 9.0 * precision)

    chunks = 1
    for previous, current in zip(alignment, alignment[1:]):
        if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return harmonic * (1.0 - penalty)


METRICS: dict[str, Callable[[Sequence[str], Sequence[str]], float]] = {
    "bleu_1": lambda candidate, reference: bleu_n(candidate, reference, 1),
    "bleu_2": lambda candidate, reference: bleu_n(candidate, reference, 2),
    "bleu_3": lambda candidate, reference: bleu_n(candidate, reference, 3),
    "rouge_l": rouge_l,
    "meteor": meteor,
}


def score_direction(
    candidate: Sequence[str],
    reference: Sequence[str],
) -> dict[str, float]:
    return {
        metric_name: round(metric(candidate, reference), 8)
        for metric_name, metric in METRICS.items()
    }


def _validate_opaque_name(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise ValueError(f"invalid {label}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-a", required=True)
    parser.add_argument("--candidate-a-id", required=True)
    parser.add_argument("--candidate-b", required=True)
    parser.add_argument("--candidate-b-id", required=True)
    parser.add_argument(
        "--comparison-role",
        choices=("peer", "reference"),
        default="peer",
        help="peer computes symmetric agreement; reference treats B as ground truth.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _build_bundle(args: argparse.Namespace) -> dict[str, object]:
    run_id = _validate_opaque_name(args.run_id, label="run ID")
    candidate_a_id = _validate_opaque_name(
        args.candidate_a_id,
        label="candidate A ID",
    )
    candidate_b_id = _validate_opaque_name(
        args.candidate_b_id,
        label="candidate B ID",
    )
    if candidate_a_id == candidate_b_id:
        raise ValueError("candidate IDs must be distinct")

    candidate_a_path = require_private_file(args.candidate_a)
    candidate_b_path = require_private_file(args.candidate_b)
    candidate_a_tokens = tokenize(
        candidate_a_path.read_text(encoding="utf-8", errors="replace")
    )
    candidate_b_tokens = tokenize(
        candidate_b_path.read_text(encoding="utf-8", errors="replace")
    )
    if not candidate_a_tokens or not candidate_b_tokens:
        raise ValueError("report candidate is empty after tokenization")

    a_to_b = score_direction(candidate_a_tokens, candidate_b_tokens)
    records: list[dict[str, object]] = []
    bundle: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "producer": {
            "name": "tricompose_report_metrics",
            "version": PRODUCER_VERSION,
        },
        "comparison_role": args.comparison_role,
        "metric_contract": {
            "tokenization": "lowercase ASCII alphanumeric and underscore tokens",
            "bleu": "sentence BLEU-N, uniform weights, epsilon=1e-9",
            "rouge_l": "token LCS F1",
            "meteor": (
                "exact-token METEOR with fragmentation penalty; "
                "no stemming or WordNet synonyms"
            ),
        },
        "candidates": {
            candidate_a_id: {
                "sha256": sha256_file(candidate_a_path),
                "tokens": len(candidate_a_tokens),
            },
            candidate_b_id: {
                "sha256": sha256_file(candidate_b_path),
                "tokens": len(candidate_b_tokens),
            },
        },
        "directional": {
            f"{candidate_a_id}_to_{candidate_b_id}": a_to_b,
        },
        "records": records,
    }

    if args.comparison_role == "reference":
        for metric_name, value in a_to_b.items():
            records.append(
                {
                    "metric": f"report_reference_{metric_name}",
                    "scope": "report_reference",
                    "candidate_ids": [candidate_a_id],
                    "reference_id": candidate_b_id,
                    "value": value,
                    "higher_is_better": True,
                }
            )
    else:
        b_to_a = score_direction(candidate_b_tokens, candidate_a_tokens)
        symmetric = {
            metric_name: round(
                (a_to_b[metric_name] + b_to_a[metric_name]) / 2.0,
                8,
            )
            for metric_name in METRICS
        }
        directional = bundle["directional"]
        if not isinstance(directional, dict):
            raise TypeError("internal directional score contract failed")
        directional[f"{candidate_b_id}_to_{candidate_a_id}"] = b_to_a
        bundle["symmetric_mean"] = symmetric
        bundle["interpretation"] = (
            "Pairwise agreement only. These values cannot determine which "
            "candidate is clinically correct."
        )
        for metric_name, value in symmetric.items():
            records.append(
                {
                    "metric": f"report_pair_{metric_name}",
                    "scope": "report_pair",
                    "candidate_ids": [candidate_a_id, candidate_b_id],
                    "value": value,
                    "higher_is_better": True,
                }
            )
    return bundle


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)
    stage_dir: Path | None = None
    try:
        stage_dir = create_private_stage_dir(args.output_dir)
        bundle = _build_bundle(args)
        output_path = write_private_json(stage_dir / "report_metrics.json", bundle)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "artifact": output_path.name,
                    "sha256": sha256_file(output_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "producer": "tricompose_report_metrics",
            "error_type": type(exc).__name__,
        }
        if stage_dir is not None:
            try:
                write_private_json(stage_dir / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
