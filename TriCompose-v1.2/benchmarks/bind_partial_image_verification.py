#!/usr/bin/env python3
"""Reconstruct report-blind image receipts without any new model execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "TriCompose-v1.0/eval/report_v1_1"))
from tricompose_v12.partial_image_verification import SCHEMA, bind_cached_image_phases, validate_config
from tricompose_v12.invariant_verification import SCHEMA as FULL_SCHEMA
from diagnose_automatic_discrepancy import load as load_source
from merge_automatic_secondary import cached
from contracts import (WORKSPACE, require_inside, sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("existing Slurm allocation required")
    config_path = require_inside(args.interface_config, WORKSPACE, must_exist=True)
    if config_path.stat().st_size > 16 * 1024:
        raise ValueError("unbounded partial-image config")
    config = json.loads(config_path.read_text()); validate_config(config)
    rows, bank, policy, sources = load_source(args)
    manifest, manifest_path, receipt_path = cached(
        args.completed_run, FULL_SCHEMA, "candidate_receipts.jsonl", 32 * 1024 * 1024)
    if (manifest.get("original_selection_changed") is not False
            or manifest.get("original_actions_or_costs_changed") is not False
            or manifest.get("new_model_calls") != 0
            or manifest.get("clinical_repair_success") is not False):
        raise ValueError("unchanged completed verification run required")
    # Authenticate its frozen dependencies as well as the receipt file itself.
    for name, path in manifest["source_paths"].items():
        source = require_inside(path, WORKSPACE, must_exist=True)
        if sha256_file(source) != manifest["source_sha256"][name]:
            raise ValueError("completed verification dependency changed")
        sources["completed_source_" + name] = source
    for name in ("source_scores", "source_edges", "endpoint_outcomes"):
        if sha256_file(sources[name]) != manifest["source_sha256"][name]:
            raise ValueError("completed verification uses a different cohort")
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines() if line]
    sources.update(interface_config=config_path, completed_manifest=manifest_path, completed_receipts=receipt_path)
    return bank, receipts, config, sources


def markdown(summary):
    lines = ["# CXR-before-report receipts / 报告生成前的图像验证记录", "",
        "Status: retrospective engineering cache test, not newly executed generation, an improved controller or clinical validation.",
        "只复用现有缓存构造阶段记录；不新增模型调用、不改变 EHR、现有赢家、动作或成本。", "",
        f"Fixed EHRs: {summary['fixed_ehr_cases']}; unique CXR partial receipts: {summary['unique_image_partial_receipts']}; completed triple bindings: {summary['completed_report_bindings']}.", "",
        "## Image-phase readout / 图像阶段读数", "",
        "| Cached verification status | Unique images |", "|---|---:|"]
    for name, count in summary["image_status_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines += ["", "## What is available / 什么已算、什么没算", "",
        "- Before report generation: EHR–CXR cached state comparison is available where explicit fixed EHR evidence and explicit XRV states overlap.",
        "- EHR–Report and CXR–Report are `null / not_generated`, NOT zero, perfect consistency, a missing label vector or a negative finding. All-three support is also unavailable.",
        "- Unknown/uncertain image/EHR labels never become negative. No-direct-EHR cases remain with NA support/coverage; no diagnosis/device is added.",
        "- A completed report is linked in a NEW binding to the unchanged partial receipt. Image hash, XRV states, EHR anchor and EHR–CXR counts must match.",
        "- The image projection accepts records with all report fields absent. Report labels, report structure, triple gate, scores, winner flags and costs cannot change an image receipt.",
        "- Raw fourteen-head legacy states remain uncalibrated/unverified; they are not the eight-head scoped preview or clinical ground truth. Multiple reports on one image are not independent votes.",
        "- Cache reconstruction is explicitly marked retrospective. It does not prove actual generation order, skipped report calls, saved GPU time or better clinical quality.",
        "- No source patient input, report body or image pixels were opened; no inference, training, GPU job, download, API or Slurm submission occurred.",
        "- No clinical acceptance/repair-success/fault label is granted. This is previously inspected development evidence, not new independent confirmation.", "",
        "## Files / 文件", "",
        "`image_partial_receipts.jsonl`: immutable image-phase records. `completed_report_bindings.jsonl`: links to archived full receipts. `ehr_anchors.jsonl`: fixed references. `summary.json`, `frozen_interface.json`, `manifest.json`: coverage and source/result hashes.", "",
        "Next: prospective inference hooks and a frozen bounded execution/cost policy. Every new GPU job still needs full-script/resource approval; no automatic job was submitted.", ""]
    return "\n".join(lines)


def run(args):
    started = time.monotonic()
    bank, receipts, config, sources = load(args)
    sources.update(program=Path(__file__), partial_contract=ROOT / "src/tricompose_v12/partial_image_verification.py",
        protocol=WORKSPACE / "docs/partial_image_verification_contract.md")
    before = {k: sha256_file(p) for k, p in sources.items()}
    result = bind_cached_image_phases(bank, receipts, before["source_edges"], config)
    summary = {
        "schema_version": SCHEMA, "status": "completed_cached_partial_image_integration",
        "fixed_ehr_cases": len(result["anchors"]),
        "unique_image_partial_receipts": len(result["partial_receipts"]),
        "completed_report_bindings": len(result["report_bindings"]),
        "image_status_counts": result["image_status_counts"],
        "report_edges_at_image_phase": None, "all_three_support_at_image_phase": None,
        "partial_receipts_overwritten": False, "original_selection_changed": False,
        "original_actions_or_costs_changed": False, "new_model_calls": 0,
        "actual_regeneration_executed": False, "actual_gpu_savings": None,
        "actual_execution_order_validated": False, "clinical_acceptance": False,
        "clinical_repair_success": False, "clinical_accuracy": None,
        "runtime_seconds": round(time.monotonic() - started, 6),
    }
    if before != {k: sha256_file(p) for k, p in sources.items()}:
        raise ValueError("partial-image source changed")
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "summary.json", summary),
            write_private_json(temporary / "frozen_interface.json", config),
            write_private_text(temporary / "RESULTS_CN_EN.md", markdown(summary))]
        for name, records in (("ehr_anchors.jsonl", result["anchors"]),
                ("image_partial_receipts.jsonl", result["partial_receipts"]),
                ("completed_report_bindings.jsonl", result["report_bindings"])):
            files.append(write_private_text(temporary / name,
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)))
        write_private_json(temporary / "manifest.json", {
            "schema_version": SCHEMA, "run_id": args.run_id,
            "source_paths": {k: str(v) for k, v in sources.items()}, "source_sha256": before,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "new_model_calls": 0, "original_selection_changed": False,
            "original_actions_or_costs_changed": False, "actual_execution_order_validated": False,
            "clinical_accuracy_claim_allowed": False,
        })
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-run", "completed-run", "interface-config", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds"],
        "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
