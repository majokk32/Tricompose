#!/usr/bin/env python3
"""Verify a completed numeric-only pilot and run matched CPU-only controls.

Actual Slurm required; no inference, downloads, API or raw patient artifact IO.
The previous GPU run is reused, never called again or overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

from run import WORKSPACE, PROTECTED_ROOT, new_private_run, private_json, require, sha256_file
from benchmark_probe_repair_v1 import load_primary
from tricompose_v12.probe_repair_v1 import snapshot
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_llm.controller import tools_from_orders
from tricompose_llm.paired_cache_comparison import VERSION, aggregate, compare_case


def checked_pilot(path):
    root = path.absolute()
    require(root.is_relative_to(PROTECTED_ROOT) and root.resolve(strict=True) == root,
            "literal_protected_pilot_root")
    manifest_path = root / "manifest.json"
    require(manifest_path.stat().st_size <= 1024**2, "bounded_pilot_manifest")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema_version"] == "tricompose-local-qwen-affordable-menu-v3"
            and manifest["status"] == "complete" and manifest["mode"] == "cache-replay"
            and manifest["actual_api_attempts"] == manifest["actual_generator_calls"] == 0
            and manifest["actual_regeneration_executed"] is False
            and manifest["clinical_repair_success"] is None
            and manifest["local_planner_audit"]["frozen"] is True,
            "completed_frozen_numeric_only_pilot_required")
    for relative, pin in manifest["code_pins"].items():
        source = WORKSPACE / relative
        require(not source.is_symlink() and source.resolve().is_relative_to(WORKSPACE)
                and source.suffix == ".py" and sha256_file(source) == pin,
                "consumed_pilot_code_changed")
    for name, metadata in manifest["artifacts"].items():
        require(re.fullmatch(r"(?:start_manifest|summary|case_[0-9]{3})\.json|events\.jsonl|native\.log", name)
                is not None, "only_known_pilot_receipt_artifacts")
        file = root / name
        require(not file.is_symlink() and file.stat().st_size == metadata["size_bytes"]
                and sha256_file(file) == metadata["sha256"], "pilot_artifact_integrity")
    for file in (root, manifest_path, *(root / name for name in manifest["artifacts"])):
        require(file.stat().st_gid in (96293, 65534)
                and file.stat().st_mode & 0o7777 == (0o2770 if file.is_dir() else 0o660),
                "protected_project_group_receipts_required")
    return manifest


def run(args):
    require_slurm()
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and f"/job_{job}/" in Path("/proc/self/cgroup").read_text(),
            "actual_slurm_cpu_comparison_required")
    manifest = checked_pilot(args.pilot_root)
    start = json.loads((args.pilot_root / "start_manifest.json").read_text())
    count = manifest["cases_requested"]
    require(type(count) is int and 1 <= count <= 80
            and count == manifest["cases_completed"] == manifest["cases_processed"] == start["case_count"],
            "complete_fixed_source_order_pilot_cases_required")
    sources = {}
    bank, policy = load_primary(sources)
    require(sources == manifest["source_pins"] == start["source_pins"], "same_immutable_numeric_bank")
    budget, steps = start["budget_units_per_case"], start["max_steps"]
    tools = tools_from_orders(policy["image_order"], policy["report_order"])
    first = (*policy["image_order"][0], policy["report_order"][0])
    root = new_private_run(args.output_root, args.run_id)
    code_pins = {str(path.relative_to(WORKSPACE)): sha256_file(path)
        for path in sorted((WORKSPACE / "TriCompose-v1.2/agent").rglob("*.py"))}
    pilot_hash = sha256_file(args.pilot_root / "manifest.json")
    private_json(root / "start_manifest.json", {"schema_version": VERSION, "status": "started",
        "source_pilot_manifest_sha256": pilot_hash, "source_pins": sources, "code_pins": code_pins,
        "paired_cases": count, "budget_units_per_case": budget, "max_policy_requests": steps,
        "actual_local_llm_calls_executed": 0, "actual_generator_calls": 0})
    started = time.monotonic()
    trials, outcomes = [], []
    audit = manifest["local_planner_audit"]
    attribution, offset = audit["proposal_sources"], 0
    require(len(attribution) == manifest["policy_requests"]
            and all(row["request_index"] == index for index, row in enumerate(attribution)),
            "complete_ordered_policy_request_attribution")
    for index, key in enumerate(list(bank)[:count]):
        grid = {slot: snapshot(value) for slot, value in bank[key].items()}
        result = json.loads((args.pilot_root / f"case_{index:03d}.json").read_text())
        size = result["planner_calls"]
        rows, records = compare_case(index, grid[first], grid, tools, result,
            attribution[offset:offset+size], budget=budget, max_steps=steps)
        trials.extend(rows); outcomes.extend(records); offset += size
    require(offset == len(attribution) and sum(row["source"] == "llm" for row in attribution)
            == manifest["actual_local_planner_attempts"], "recorded_qwen_calls_match_attribution")
    per_case, table = aggregate(trials)
    private_json(root / "trials.json", trials)
    private_json(root / "control_receipts.json", outcomes)
    private_json(root / "per_case.json", per_case)
    private_json(root / "score_table.json", table)
    require(all(sha256_file(WORKSPACE / path) == expected for path, expected in code_pins.items())
            and sha256_file(args.pilot_root / "manifest.json") == pilot_hash,
            "comparison_sources_changed_during_run")
    result = {"schema_version": VERSION, "status": "complete", "paired_ehr_cases": count,
        "trial_rows": len(trials), "methods": len(table), "random_replicates_per_case": 5,
        "budget_units_per_case": budget, "max_policy_requests": steps,
        "frozen_case_order": "source_order_first_n", "independent_clinical_validation": False,
        "clinical_repair_success": None, "actual_api_calls": 0,
        "actual_local_llm_calls_executed": 0, "actual_generator_calls": 0,
        "recorded_local_llm_attempts_reused": manifest["actual_local_planner_attempts"],
        "recorded_llm_runtime_seconds": manifest["elapsed_seconds"],
        "cost_units_are_not_measured_equal_cpu_gpu_time": True,
        "metric_aggregation": "random_replicates_within_ehr_then_ehr_equal_weight",
        "no_bootstrap_or_significance_claim": True,
        "elapsed_seconds": round(time.monotonic()-started, 3),
        "original_bank_and_pilot_unchanged": True}
    private_json(root / "summary.json", result)
    artifacts = {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in root.iterdir() if path.is_file()}
    private_json(root / "manifest.json", {**result, "source_pilot_manifest_sha256": pilot_hash,
        "source_pilot_manifest": str(args.pilot_root / "manifest.json"), "source_pins": sources,
        "pilot_code_pins": manifest["code_pins"], "code_pins": code_pins, "artifacts": artifacts})
    print(json.dumps({"stage": "paired_cached_policy_comparison", "status": "complete",
        "elapsed_seconds": result["elapsed_seconds"], "manifest_sha256": sha256_file(root / "manifest.json")}, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path,
                        default=PROTECTED_ROOT / "tricompose_v1_2/llm_agent_comparisons")
    args = parser.parse_args()
    os.umask(0o007)
    try:
        return run(args)
    except Exception:
        print(json.dumps({"stage": "paired_cached_policy_comparison", "status": "failed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
