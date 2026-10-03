#!/usr/bin/env python3
"""Bind immutable EHR verification receipts to existing generation events."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.invariant_verification import SCHEMA, stream_cached_verification, validate_config
from diagnose_automatic_discrepancy import load as load_source
from run_automatic_proxy_replay import render_csv
from contracts import (WORKSPACE, require_inside, sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    config_path = require_inside(args.interface_config, WORKSPACE, must_exist=True)
    if config_path.stat().st_size > 16 * 1024: raise ValueError("unbounded interface config")
    config = json.loads(config_path.read_text()); validate_config(config)
    rows, bank, policy, sources = load_source(args)
    sources["interface_config"] = config_path
    return rows, bank, policy, config, sources


def trial_statuses(trials):
    groups = defaultdict(list)
    for row in trials: groups[(row["method"], row["model_call_budget"], row["selected_verification_status"])].append(row)
    return [{"method": method, "model_call_budget": budget, "selected_verification_status": status,
        "original_replay_trials": len(rows), "distinct_fixed_ehr_cases": len({r["case_id"] for r in rows}),
        "random_seeds_are_not_independent_cases": True, "clinical_acceptance": False}
        for (method, budget, status), rows in sorted(groups.items())]


def markdown(summary):
    lines = ["# Invariant verification hook / 固定 EHR 验证接口接入", "",
        "Status: cache-bound engineering integration, NOT a new generation, independent clinical validation or improved repair algorithm.",
        "固定 EHR 证据后，为原请求过的每个候选绑定验证记录；不改变模型、病例、评分规则、动作、调用账本或最终赢家。", "",
        f"Fixed EHR anchors: {summary['fixed_ehr_cases']}; unique candidate receipts: {summary['candidate_receipts']}; original replay trials: {summary['original_replay_trials']}; bound observation events: {summary['bound_observation_events']}.", "",
        "## What the hook records / 接口记录什么", "",
        "1. `ehr_anchors.jsonl`: one immutable reference per EHR, including states, cached categories and EHR/facts hashes. Reference IDs never depend on the chosen image/report.",
        "2. `candidate_receipts.jsonl`: candidate/artifact lineage, all three raw edges, explicit positive/negative support, opposition, missingness and direct-EHR coverage.",
        "3. `verification_events.jsonl`: bind each original requested step to its receipt and before/after transition. Image replacement tracks added/lost EHR support and new/removed opposition.",
        "4. `trial_bindings.jsonl`: bind the original final choice and unchanged call ledger/stop reason. A budget stop or proxy stop is NOT clinical acceptance.", "",
        "## Unique candidate receipt statuses / 候选验证状态", "",
        "| Status | Unique candidate receipts |", "|---|---:|"]
    for status, count in sorted(summary["unique_receipt_counts_by_status"].items()):
        lines.append(f"| {status} | {count} |")
    lines += ["", "## Interface semantics / 接口语义", "",
        "- `unverified_no_direct_ehr_constraints`: the cached EHR has no comparable explicit finding. Keep the case; its direct-EHR support/coverage rates are NA, not zero, normal, failure or repair success.",
        "- `fixed_ehr_proxy_support_increased_unvalidated`: only a cached image-label support increase. It does not prove original image error, clinical repair or three-modal truth.",
        "- `fixed_ehr_proxy_support_withdrawn_or_opposition_added` / mixed / missing statuses preserve evidence loss and ambiguity, without naming a confirmed faulty modality.",
        "- Unknown/uncertain never become negative. Legacy global No-Finding adjustments are NOT applied to raw receipt states. Negative agreement can be appropriate; no disease/device is inserted.",
        "- Report-only changes cannot acquire an image-repair claim. Same-image classifier labels and same-report label vectors must stay unchanged; altered receipts fail hash/arithmetic checks.",
        "- No report body/image pixel, patient source input, external API, model or GPU was opened/called. Raw report labels are unverified and the newer eight-head scope interface is unchanged.",
        "- Scalars such as BioViL, runtime or source proxy balance are not inputs to this receipt. They cannot manufacture evidence or change the reference.",
        "- A receipt does not choose an action, rerank a triple or authorize inference. Every clinical acceptance/repair-success/model-execution flag is false.",
        "- Event/seed counts are dependent replay observations, not independent patients. Original model calls stay simulated; there is no new generation or measured GPU saving.",
        "- The cohort is previously inspected development data. Prospective execution still needs separately approved scripts/resources and a frozen policy/evaluation design.", "",
        "See `transition_status_counts.csv`, `selected_status_counts.csv`, `summary.json` and `manifest.json` for coverage, status and source/result hash checks. Detailed facts remain project-private.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    rows, bank, policy, config, sources = load(args)
    sources.update(program=Path(__file__), interface_contract=ROOT / "src/tricompose_v12/invariant_verification.py",
        source_event_contract=ROOT / "src/tricompose_v12/ranking_switch_controls.py",
        source_rank_contract=ROOT / "src/tricompose_v12/automatic_replay.py",
        source_evidence_contract=ROOT / "src/tricompose_v12/legacy_replay_adapter.py",
        protocol=WORKSPACE / "docs/invariant_verification_interface.md")
    before = {k: sha256_file(p) for k, p in sources.items()}
    result = stream_cached_verification(rows, bank, policy, config)
    selected_counts = trial_statuses(result["trials"])
    summary = {"schema_version": SCHEMA, "status": "completed_cache_bound_verification_integration",
        "fixed_ehr_cases": len(result["anchors"]), "candidate_receipts": len(result["receipts"]),
        "original_replay_trials": len(result["trials"]), "bound_observation_events": len(result["events"]),
        "bound_transition_events": sum(e["transition"] is not None for e in result["events"]),
        "unique_receipt_counts_by_status": dict(sorted(Counter(r["verification_status"] for r in result["receipts"]).items())),
        "transition_status_counts": result["transition_status_counts"], "selected_status_counts": selected_counts,
        "new_model_calls": 0, "original_selection_changed": False, "original_actions_or_costs_changed": False,
        "actual_regeneration_executed": False, "actual_gpu_savings": None,
        "clinical_accuracy": None, "clinical_repair_success": False,
        "independent_clinical_truth_available": False, "runtime_seconds": round(time.monotonic()-started, 6)}
    if before != {k: sha256_file(p) for k, p in sources.items()}:
        raise ValueError("immutable verification source changed")
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "summary.json", summary),
            write_private_json(temporary / "frozen_interface.json", config),
            write_private_text(temporary / "transition_status_counts.csv", render_csv(result["transition_status_counts"])),
            write_private_text(temporary / "selected_status_counts.csv", render_csv(selected_counts)),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary))]
        for name, records in (("ehr_anchors.jsonl", result["anchors"]), ("candidate_receipts.jsonl", result["receipts"]),
                              ("verification_events.jsonl", result["events"]), ("trial_bindings.jsonl", result["trials"])):
            files.append(write_private_text(temporary / name, "".join(json.dumps(r, sort_keys=True)+"\n" for r in records)))
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {k: str(v) for k, v in sources.items()}, "source_sha256": before,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files}, "new_model_calls": 0,
            "original_selection_changed": False, "original_actions_or_costs_changed": False,
            "clinical_repair_success": False, "clinical_accuracy_claim_allowed": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-run", "interface-config", "output-root", "run-id"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try: target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds"],
        "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
