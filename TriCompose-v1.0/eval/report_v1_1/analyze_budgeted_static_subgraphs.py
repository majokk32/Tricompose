#!/usr/bin/env python3
"""Replay fixed, nested candidate grids using an existing protected selection table.

This is a diagnostic analysis of already generated candidates. It neither runs
models nor measures savings in the actual candidate-bank generation run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
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


IMAGE_ORDER = (
    ("chexgenbench_sana", 0),
    ("chexgenbench_pixart", 0),
    ("roentgen_v2", 0),
    ("chexgenbench_sana", 1),
    ("chexgenbench_pixart", 1),
    ("roentgen_v2", 1),
)
REPORT_ORDER = ("maira2", "cxrmate_single", "llavarad", "chexagent2")
STAGES = (
    ("fixed", 1, 1),
    ("two_reports", 1, 2),
    ("four_reports", 1, 4),
    ("two_cxrs", 2, 4),
    ("three_cxrs", 3, 4),
    ("all_six_cxrs", 6, 4),
)


def analyze(table_path: Path) -> dict:
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("candidate table is empty")
    cases: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    expected_paths = {
        (model, seed, report)
        for model, seed in IMAGE_ORDER
        for report in REPORT_ORDER
    }
    for case_rows in cases.values():
        paths = {
            (row["cxr_model_id"], int(row["cxr_seed"]), row["report_model_id"])
            for row in case_rows
        }
        ranks = {int(row["selection_rank_within_case"]) for row in case_rows}
        if len(case_rows) != 24 or paths != expected_paths or ranks != set(range(1, 25)):
            raise ValueError("candidate grid or within-case ranking is incomplete")

    stages = []
    for stage_id, image_count, report_count in STAGES:
        images = set(IMAGE_ORDER[:image_count])
        reports = set(REPORT_ORDER[:report_count])
        selected = []
        for case_id, case_rows in sorted(cases.items()):
            available = [
                row for row in case_rows
                if (row["cxr_model_id"], int(row["cxr_seed"])) in images
                and row["report_model_id"] in reports
            ]
            chosen = min(available, key=lambda row: int(row["selection_rank_within_case"]))
            selected.append(chosen)
        support = sum(int(row["total_direct_support_count"]) for row in selected)
        contradictions = sum(int(row["total_hard_contradiction_count"]) for row in selected)
        known = sum(int(row["total_known_reference_fact_count"]) for row in selected)
        if known == 0:
            raise ValueError("no comparable reference facts")
        stages.append({
            "stage_id": stage_id,
            "candidate_cxrs_per_case": image_count,
            "candidate_reports_per_case": image_count * report_count,
            "replay_generation_calls_per_case": image_count * (report_count + 1),
            "selection_scorer_calls_excluded": True,
            "selected_cases": len(selected),
            "known_reference_facts": known,
            "supported_facts": support,
            "hard_contradictions": contradictions,
            "support_recall": round(support / known, 8),
            "clinical_balance_score_0_100": round(
                50 * (1 + (support - contradictions) / known), 8
            ),
            "selected_cxr_models": dict(sorted(Counter(row["cxr_model_id"] for row in selected).items())),
            "selected_report_models": dict(sorted(Counter(row["report_model_id"] for row in selected).items())),
            "selected_candidate_ids": {
                row["case_id"]: row["triple_candidate_id"] for row in selected
            },
        })
    return {
        "schema_version": "tricompose-budgeted-static-subgraph-replay-v1",
        "status": "retrospective_diagnostic_only",
        "cases": len(cases),
        "candidate_rows": len(rows),
        "image_order": [{"model_id": model, "seed": seed} for model, seed in IMAGE_ORDER],
        "report_order": list(REPORT_ORDER),
        "stages": stages,
        "interpretation": [
            "Every candidate was generated before this replay; actual bank generation cost was 30 generator calls per case.",
            "Each stage selects on the same uncalibrated evidence used to rank the full bank.",
            "Generation calls omit XRV, CheXbert, BioViL, failed calls, and verifier runtime.",
            "This is a nested static-subgraph comparison, not a dynamic policy or independent clinical evaluation.",
        ],
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Direct-fact cohort: retrospective candidate-budget replay",
        "",
        f"Cases: {payload['cases']}; existing candidate rows: {payload['candidate_rows']}.",
        "",
        "| Candidate grid | Generator calls/case | Support | Hard contradictions | Balance score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["stages"]:
        lines.append(
            f"| {row['stage_id']} | {row['replay_generation_calls_per_case']} | "
            f"{row['supported_facts']}/{row['known_reference_facts']} | "
            f"{row['hard_contradictions']} | {row['clinical_balance_score_0_100']:.2f} |"
        )
    lines.extend(["", *[f"- {item}" for item in payload["interpretation"]], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    selection = require_inside(args.selection_run, PROTECTED_ROOT, must_exist=True)
    manifest_path = selection / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tricompose-edge-specific-selection-v1.1":
        raise ValueError("requires a completed edge-specific selection run")
    table = selection / "candidate_score_table.csv"
    payload = analyze(table)
    payload["source"] = {
        "selection_manifest_sha256": sha256_file(manifest_path),
        "candidate_table_sha256": sha256_file(table),
    }
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        output = write_private_json(temporary / "budget_replay.json", payload)
        summary = write_private_text(temporary / "budget_replay.md", render_markdown(payload))
        write_private_json(temporary / "manifest.json", {
            "schema_version": payload["schema_version"],
            "run_id": args.run_id,
            "budget_replay_sha256": sha256_file(output),
            "summary_sha256": sha256_file(summary),
            "status": payload["status"],
        })
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(json.dumps({"status": payload["status"], "cases": payload["cases"],
                      "stages": len(payload["stages"]), "output_run": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
