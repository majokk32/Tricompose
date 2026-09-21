#!/usr/bin/env python3
"""Run fixed-path, random, and exhaustive static-reranking baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from contracts import (
    PROTECTED_ROOT,
    commit_atomic_run,
    discard_atomic_run,
    new_atomic_run,
    read_json,
    require_inside,
    sha256_file,
    write_private_json,
    write_private_text,
)


TABLE_SCHEMA = "tricompose-unified-score-table-v1.1"
SCHEMA_VERSION = "tricompose-selection-baselines-v1.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table-run", required=True)
    parser.add_argument("--random-repetitions", type=int, default=100)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("score-table row is not an object")
                rows.append(value)
    return rows


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return None if not valid else round(statistics.fmean(valid), 8)


def _selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected_count": len(rows),
        "hard_gate_failure_count": sum(bool(row["selection"]["hard_gate_failures"]) for row in rows),
        "mean_static_total_score": _mean([row["static_scoring"]["total_score"] for row in rows]),
        "mean_clinical_consistency_score": _mean(
            [row["static_scoring"]["clinical_consistency_score"] for row in rows]
        ),
        "mean_report_quality_score": _mean(
            [row["static_scoring"]["report_quality_score"] for row in rows]
        ),
        "mean_edge_score": {
            edge: _mean([row["cross_modal"][edge]["edge_score"] for row in rows])
            for edge in ("ehr_cxr", "ehr_report", "report_cxr")
        },
        "mean_edge_coverage": {
            edge: _mean([row["cross_modal"][edge]["coverage"] for row in rows])
            for edge in ("ehr_cxr", "ehr_report", "report_cxr")
        },
        "mean_edge_contradiction_rate": {
            edge: _mean([row["cross_modal"][edge]["contradiction_rate"] for row in rows])
            for edge in ("ehr_cxr", "ehr_report", "report_cxr")
        },
    }


def _selection_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "triple_candidate_id": row["triple_candidate_id"],
            "cxr_candidate_id": row["lineage"]["cxr_candidate_id"],
            "cxr_model_id": row["lineage"]["cxr_model_id"],
            "report_model_id": row["lineage"]["report_model_id"],
            "static_total_score": row["static_scoring"]["total_score"],
            "hard_gate_failures": row["selection"]["hard_gate_failures"],
        }
        for row in sorted(rows, key=lambda item: str(item["case_id"]))
    ]


def _hash_choice(rows: list[dict[str, Any]], repetition: int, case_id: str) -> dict[str, Any]:
    digest = hashlib.sha256(f"tricompose-random-v1.1:{repetition}:{case_id}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(rows)
    return sorted(rows, key=lambda row: str(row["triple_candidate_id"]))[index]


def _static_key(row: dict[str, Any]) -> tuple[float, float, str]:
    score = row["static_scoring"]["total_score"]
    runtime = row["cost"]["known_runtime_seconds"]
    return (
        -math.inf if score is None else float(score),
        -math.inf if runtime is None else -float(runtime),
        str(row["triple_candidate_id"]),
    )


def _markdown(payload: dict[str, Any]) -> str:
    static = payload["static_reranking"]["summary"]
    random_summary = payload["random"]["aggregate"]
    fixed = payload["fixed_paths"]["same_cohort_descriptive_best"]
    return "\n".join(
        [
            "# TriCompose V1.1 Non-agent Baselines",
            "",
            "| Baseline | CXR calls/case | Report calls/case | Mean static score |",
            "|---|---:|---:|---:|",
            f"| Same-cohort best fixed path (descriptive only) | 1 | 1 | {fixed['mean_static_total_score']} |",
            f"| Random candidate ({payload['random']['repetitions']} repetitions) | 1 | 1 | {random_summary['mean_of_mean_static_total_score']} |",
            f"| Exhaustive static reranking | 3 | 12 | {static['mean_static_total_score']} |",
            "",
            "The best fixed path is labelled descriptive because it is ranked on this same cohort; it is not a held-out model-selection result.",
            "Static reranking generates all 3 CXR and all 12 report paths before choosing, so it is a high-cost non-agent baseline.",
            "Qwen/BioViL and cost are not hidden inside the static total. No dynamic policy or regeneration is implemented here.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.random_repetitions < 1:
        raise ValueError("random repetitions must be positive")
    table_run = require_inside(args.score_table_run, PROTECTED_ROOT, must_exist=True)
    summary = read_json(table_run / "score_table_summary.json")
    if summary.get("schema_version") != TABLE_SCHEMA:
        raise ValueError("unsupported unified score table")
    if summary.get("table_status") != "complete_frozen_evidence":
        raise ValueError("baselines require the completed score table")
    rows = _read_jsonl(table_run / "score_table.jsonl")
    if len(rows) != 960:
        raise ValueError("baseline input must contain 960 candidates")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row["case_id"])
        by_case[case_id].append(row)
        path = f"{row['lineage']['cxr_model_id']}->{row['lineage']['report_model_id']}"
        by_path[path].append(row)
    if len(by_case) != 80 or {len(values) for values in by_case.values()} != {12}:
        raise ValueError("each case must provide exactly 12 candidates")
    if len(by_path) != 12 or {len(values) for values in by_path.values()} != {80}:
        raise ValueError("the fixed-path grid must contain 12 complete paths")

    fixed_results = {
        path: {
            "path": path,
            "prospective_model_calls_per_case": {"cxr": 1, "report": 1, "total": 2},
            **_selection_summary(path_rows),
        }
        for path, path_rows in sorted(by_path.items())
    }
    fixed_best_path, fixed_best = max(
        fixed_results.items(),
        key=lambda item: (
            -math.inf if item[1]["mean_static_total_score"] is None else item[1]["mean_static_total_score"],
            item[0],
        ),
    )
    fixed_best = {**fixed_best, "path": fixed_best_path, "selection_scope": "same_cohort_descriptive_not_held_out"}

    random_runs: list[dict[str, Any]] = []
    random_seed0: list[dict[str, Any]] = []
    for repetition in range(args.random_repetitions):
        chosen = [
            _hash_choice(candidates, repetition, case_id)
            for case_id, candidates in sorted(by_case.items())
        ]
        if repetition == 0:
            random_seed0 = chosen
        random_runs.append({"repetition": repetition, **_selection_summary(chosen)})
    random_means = [
        float(row["mean_static_total_score"])
        for row in random_runs
        if row["mean_static_total_score"] is not None
    ]
    random_aggregate = {
        "valid_repetitions": len(random_means),
        "mean_of_mean_static_total_score": None if not random_means else round(statistics.fmean(random_means), 8),
        "std_of_mean_static_total_score": None if len(random_means) < 2 else round(statistics.stdev(random_means), 8),
        "minimum_mean_static_total_score": None if not random_means else round(min(random_means), 8),
        "maximum_mean_static_total_score": None if not random_means else round(max(random_means), 8),
    }

    static_selected: list[dict[str, Any]] = []
    rejected_cases: list[str] = []
    for case_id, candidates in sorted(by_case.items()):
        eligible = [row for row in candidates if row["selection"]["diagnostic_eligible"]]
        if not eligible:
            rejected_cases.append(case_id)
            continue
        static_selected.append(max(eligible, key=_static_key))

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": (
            "paper_primary" if summary.get("paper_primary_selection_ready") else "diagnostic_uncalibrated_cxr_labels"
        ),
        "source": {
            "score_table_run": str(table_run),
            "score_table_manifest_sha256": sha256_file(table_run / "manifest.json"),
        },
        "counts": {"cases": len(by_case), "candidate_rows": len(rows), "fixed_paths": len(by_path)},
        "fixed_paths": {
            "all_path_results": fixed_results,
            "same_cohort_descriptive_best": fixed_best,
        },
        "random": {
            "repetitions": args.random_repetitions,
            "prospective_model_calls_per_case": {"cxr": 1, "report": 1, "total": 2},
            "aggregate": random_aggregate,
            "per_repetition": random_runs,
            "repetition_0_selections": _selection_records(random_seed0),
        },
        "static_reranking": {
            "policy_id": summary["policy"]["policy_id"],
            "prospective_model_calls_per_case": {"cxr": 3, "report": 12, "total": 15},
            "summary": _selection_summary(static_selected),
            "rejected_case_count": len(rejected_cases),
            "rejected_case_ids": rejected_cases,
            "selections": _selection_records(static_selected),
        },
        "agent_implemented": False,
    }


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o007)
    started = time.monotonic()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        payload = run(args)
        payload["run_id"] = args.run_id
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        details = write_private_json(temporary / "baseline_results.json", payload)
        markdown = write_private_text(temporary / "baseline_summary.md", _markdown(payload))
        manifest = write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "evaluation_status": payload["evaluation_status"],
                "artifacts": {
                    path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                    for path in (details, markdown)
                },
            },
        )
        manifest_hash = sha256_file(manifest)
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(
        json.dumps(
            {
                "status": "completed_non_agent_baselines",
                "run_id": args.run_id,
                "evaluation_status": payload["evaluation_status"],
                "manifest_sha256": manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
