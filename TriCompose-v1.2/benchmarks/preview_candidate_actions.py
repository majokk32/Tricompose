#!/usr/bin/env python3
"""Preview verification actions from protected SYNTHETIC cached scope facts.

No text/image/model load, generation, clinical acceptance, or selection update.
Runs only inside an existing Slurm allocation; creates a new atomic private run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.decision_preview import Budget, SCHEMA, preview_actions
from tricompose_v12.report_scope_table import SCHEMA as SOURCE_SCHEMA
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)


def load_cached_facts(scope_run):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required before protected cache access")
    root = require_inside(scope_run, PROTECTED_ROOT, must_exist=True)
    manifest_path = require_inside(root / "manifest.json", root, must_exist=True)
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != SOURCE_SCHEMA
            or manifest.get("model_calls") != 0
            or manifest.get("selection_changed") is not False
            or manifest.get("regeneration_authorized") is not False
            or manifest.get("primary_metric_eligible") is not False):
        raise ValueError("immutable diagnostic synthetic scope run required")
    path = require_inside(root / "fact_scope_table.jsonl", root, must_exist=True)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("bounded cached pilot required")
    fact_hash = sha256_file(path)
    if fact_hash != manifest.get("artifacts", {}).get(path.name, {}).get("sha256"):
        raise ValueError("cached fact-table hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # No case may be dropped for lacking EHR evidence or for disagreement.
    if len(rows) != 384 or len({row["triple_candidate_id"] for row in rows}) != 48 or len({row["case_id"] for row in rows}) != 2:
        raise ValueError("complete fixed two-case/48-candidate scope inventory required")
    return rows, {"source_manifest": manifest_path, "source_fact_table": path}


def render_table(records):
    fields = ["case_id", "triple_candidate_id", "next_action", "suggested_verification_target",
              "reason_codes", "trigger_evidence_ids", "direct_ehr_coverage_gap_evidence_ids",
              "scoped_edges", "verification_affordability", "clinical_acceptance",
              "model_execution_allowed", "selection_changed"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        flat = {key: record.get(key) for key in fields}
        flat["verification_affordability"] = record["budget"]["verification_affordability"]
        for key in fields:
            if isinstance(flat[key], (dict, list)):
                flat[key] = json.dumps(flat[key], sort_keys=True)
        writer.writerow(flat)
    return output.getvalue()


def markdown(summary):
    return "\n".join([
        "# V1.2 decision preview / 决策接口预览", "",
        "This is an offline engineering preview, not a clinical repair experiment.",
        "每条候选保留 EHR、图像、报告 hash 和触发证据 ID；没有重新生成，也没有更新旧最优结果。", "",
        f"Cached candidates: {summary['candidate_triples']}; fixed EHR cases: {summary['ehr_cases']}.",
        "", "| Proposed next action | Candidates |", "|---|---:|",
        *[f"| {name} | {count} |" for name, count in sorted(summary["action_counts"].items())], "",
        "See `action_preview.csv` for candidate-level reasons and `decisions.jsonl` for complete lineage/budget details.",
        "`suggested_verification_target` is a cached agreement-pattern hint, NOT confirmed error localization.",
        "Four reports on the same CXR are correlated. Opposite labels need independent verification, not majority voting.",
        "Known EHR facts without usable report/image evidence remain coverage gaps; unknown is never negative.",
        "Even fully agreeing cached labels cannot authorize clinical acceptance or regeneration.",
        "Budget limits apply to additional calls after the bank, not the actual cost already spent generating it.",
        "No verification cost estimate is invented. Proposed actions are NOT executed and spend zero new model calls.",
        "Clinical localization accuracy, repair success, and actual compute savings remain unavailable.", ""])


def run(args):
    started = time.monotonic()
    rows, sources = load_cached_facts(args.scope_run)
    before = {name: sha256_file(path) for name, path in sources.items()}
    budget = Budget(args.max_additional_calls, args.max_additional_gpu_seconds)
    records = preview_actions(rows, budget)
    if before != {name: sha256_file(path) for name, path in sources.items()}:
        raise ValueError("cached source changed during preview")
    summary = {"schema_version": SCHEMA, "status": "completed_offline_decision_preview",
        "candidate_triples": len(records), "fact_rows": len(rows),
        "ehr_cases": len({row["case_id"] for row in records}),
        "action_counts": dict(Counter(row["next_action"] for row in records)),
        "provisional_verification_target_counts": dict(Counter(row["suggested_verification_target"] or "unresolved" for row in records)),
        "reason_counts": dict(Counter(reason for row in records for reason in row["reason_codes"])),
        "additional_budget_limits": {"model_calls": budget.max_model_calls, "gpu_seconds": budget.max_gpu_seconds},
        "new_model_calls": 0, "actual_additional_gpu_seconds": 0,
        "selection_changed": False, "regeneration_authorized": False,
        "clinical_acceptance_authorized": False, "primary_metric_eligible": False,
        "clinical_localization_accuracy": None, "clinical_repair_success": None,
        "actual_compute_savings": None, "runtime_seconds": round(time.monotonic() - started, 6)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_text(temporary / "decisions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)),
            write_private_text(temporary / "action_preview.csv", render_table(records)),
            write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "README_CN_EN.md", markdown(summary))]
        source_files = {**sources, "program": Path(__file__), "decision_interface": ROOT / "src/tricompose_v12/decision_preview.py",
                        "relation_contract": ROOT / "src/tricompose_v12/report_scope_table.py"}
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
            "source_paths": {name: str(path) for name, path in source_files.items()},
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files},
            "model_calls": 0, "selection_changed": False, "regeneration_authorized": False,
            "clinical_acceptance_authorized": False, "primary_metric_eligible": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scope-run", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-additional-calls", type=int, required=True)
    parser.add_argument("--max-additional-gpu-seconds", type=float, required=True)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds"],
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
