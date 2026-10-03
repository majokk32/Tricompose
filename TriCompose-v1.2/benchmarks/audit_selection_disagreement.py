#!/usr/bin/env python3
"""Descriptive cross-scorer audit of a protected synthetic selection run.

This never changes the winner. BioViL-T cosine is secondary and uncalibrated;
the review flag is an exploratory triage rule, not a clinical veto.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from contracts import (
    PROTECTED_ROOT,
    commit_atomic_run,
    discard_atomic_run,
    new_atomic_run,
    require_inside,
    sha256_file,
    write_private_json,
    write_private_text,
)


SCHEMA = "tricompose-selection-disagreement-audit-v1"


def _number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def audit_rows(
    rows: list[dict[str, str]],
    *,
    fixed_cxr_model: str,
    fixed_seed: int,
    fixed_report_model: str,
) -> list[dict[str, object]]:
    if not rows or len({r["triple_candidate_id"] for r in rows}) != len(rows):
        raise ValueError("score table is empty or contains duplicate candidates")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    result: list[dict[str, object]] = []
    for case_id, candidates in sorted(grouped.items()):
        selected = [r for r in candidates if r["selected"].lower() == "true"]
        fixed = [
            r for r in candidates
            if r["cxr_model_id"] == fixed_cxr_model
            and int(r["cxr_seed"]) == fixed_seed
            and r["report_model_id"] == fixed_report_model
        ]
        if len(selected) != 1 or len(fixed) != 1:
            raise ValueError(f"case {case_id} does not have one selected and one fixed row")
        chosen, control = selected[0], fixed[0]
        selected_cosine = _number(chosen, "biovil_raw_cosine_secondary")
        fixed_cosine = _number(control, "biovil_raw_cosine_secondary")
        ordered = sorted(
            candidates,
            key=lambda row: _number(row, "biovil_raw_cosine_secondary"),
            reverse=True,
        )
        rank = 1 + next(
            i for i, row in enumerate(ordered)
            if row["triple_candidate_id"] == chosen["triple_candidate_id"]
        )
        same_image = [
            row for row in candidates
            if row["cxr_candidate_id"] == chosen["cxr_candidate_id"]
        ]
        if len(same_image) < 2:
            raise ValueError(f"case {case_id} lacks a same-image report alternative")
        same_image_best = max(
            _number(row, "biovil_raw_cosine_secondary") for row in same_image
        )
        # Post-hoc triage only: bottom quartile among candidate reports AND
        # >=0.30 lower than another report on the identical synthetic image.
        bottom_quartile = rank > math.ceil(0.75 * len(candidates))
        review_flag = bottom_quartile and same_image_best - selected_cosine >= 0.30
        result.append(
            {
                "case_id": case_id,
                "selected_triple_candidate_id": chosen["triple_candidate_id"],
                "fixed_triple_candidate_id": control["triple_candidate_id"],
                "selected_cxr_model": chosen["cxr_model_id"],
                "selected_cxr_seed": int(chosen["cxr_seed"]),
                "selected_report_model": chosen["report_model_id"],
                "selected_biovil_raw_cosine": selected_cosine,
                "fixed_biovil_raw_cosine": fixed_cosine,
                "selected_biovil_rank_within_case": rank,
                "candidate_count_within_case": len(candidates),
                "best_same_image_biovil_raw_cosine": same_image_best,
                "same_image_cosine_gap": round(same_image_best - selected_cosine, 8),
                "selected_report_cxr_support_recall": _number(
                    chosen, "report_cxr_support_recall"
                ),
                "fixed_report_cxr_support_recall": _number(
                    control, "report_cxr_support_recall"
                ),
                "selected_hard_contradictions": int(
                    chosen["total_hard_contradiction_count"]
                ),
                "fixed_hard_contradictions": int(
                    control["total_hard_contradiction_count"]
                ),
                "secondary_disagreement_review_flag": review_flag,
            }
        )
    return result


def _markdown(records: list[dict[str, object]]) -> str:
    lines = [
        "# Exploratory selection-disagreement audit",
        "",
        "This is a synthetic-only, non-clinical triage check. It does not change",
        "the static selection or treat BioViL-T cosine as a calibrated probability.",
        "The review flag is post-hoc and must not be used for paper-primary claims.",
        "",
        "| Case | Selected CXR / report | BioViL rank | Selected / fixed cosine | Best same-image cosine | Review? |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in records:
        lines.append(
            "| {case_id} | {selected_cxr_model} seed {selected_cxr_seed} / "
            "{selected_report_model} | {selected_biovil_rank_within_case}/"
            "{candidate_count_within_case} | {selected_biovil_raw_cosine:.3f} / "
            "{fixed_biovil_raw_cosine:.3f} | "
            "{best_same_image_biovil_raw_cosine:.3f} | "
            "{secondary_disagreement_review_flag} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Flag rule: selected cosine ranks in the bottom quartile within its",
            "case and is at least 0.30 below a different report on the same image.",
            "This rule is for inspection priority only; its threshold was not",
            "calibrated on independent clinical data.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fixed-cxr-model", default="chexgenbench_sana")
    parser.add_argument("--fixed-seed", type=int, default=0)
    parser.add_argument("--fixed-report-model", default="maira2")
    args = parser.parse_args()
    os.umask(0o007)
    source = require_inside(args.score_table, PROTECTED_ROOT, must_exist=True)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    records = audit_rows(
        rows,
        fixed_cxr_model=args.fixed_cxr_model,
        fixed_seed=args.fixed_seed,
        fixed_report_model=args.fixed_report_model,
    )
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        details = write_private_json(
            temporary / "disagreement_audit.json",
            {
                "schema_version": SCHEMA,
                "selection_changed": False,
                "primary_metric_eligible": False,
                "source_score_table_sha256": sha256_file(source),
                "records": records,
            },
        )
        summary = write_private_text(
            temporary / "disagreement_summary.md", _markdown(records)
        )
        write_private_json(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA,
                "run_id": args.run_id,
                "cases": len(records),
                "review_flags": sum(
                    bool(row["secondary_disagreement_review_flag"])
                    for row in records
                ),
                "artifacts": {
                    details.name: sha256_file(details),
                    summary.name: sha256_file(summary),
                },
            },
        )
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "cases": len(records),
                "review_flags": sum(
                    bool(row["secondary_disagreement_review_flag"])
                    for row in records
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
