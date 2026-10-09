#!/usr/bin/env python3
"""One entry point: authored demo or immutable numeric-cache routing pilot.

Default is MOCK + authored demo, no network or GPU. API and local-Qwen
planners are explicit; cache replay is not regeneration. No live generator
adapter is enabled in this initial release.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
for relative in ("src", "TriCompose-v1.2/agent", "TriCompose-v1.2/src", "TriCompose-v1.2/tools",
                 "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(WORKSPACE/relative))

from contracts import PROTECTED_ROOT, sha256_file
from tricompose_v12.runtime_dispatch import _new_private_handle
from tricompose_v12.probe_repair_v1 import snapshot
from tricompose_llm import VERSION
from tricompose_llm.contracts import ContractError, require
from tricompose_llm.controller import CachedExecutor, LLMRepairController, tools_from_orders
from tricompose_llm.demo import DemoExecutor, observation
from tricompose_llm.planner import DemoPlanner, OpenAICompatiblePlanner


def new_private_run(output_root, run_id):
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id), "opaque_run_id_required")
    base = PROTECTED_ROOT.resolve(strict=True)
    require(stat.S_IMODE(base.stat().st_mode) == 0o2770 and base.stat().st_gid in (96293, 65534),
            "private_protected_root_required")
    target = Path(output_root).absolute()
    require(target.is_relative_to(base), "protected_output_required")
    current = base
    for name in target.relative_to(base).parts:
        require(name not in (".", ".."), "literal_private_output_path_required")
        current = current/name
        require(not current.is_symlink(), "no_output_symlinks")
        if not current.exists():
            current.mkdir(mode=0o2770)
            os.chmod(current, 0o2770)
        require(current.is_dir() and stat.S_IMODE(current.stat().st_mode) == 0o2770
                and current.stat().st_gid in (96293, 65534), "private_project_group_parent_required")
    require(not target.is_symlink(), "no_output_symlinks")
    root = target/run_id
    root.mkdir(mode=0o2770)  # Exclusive. Existing/incomplete runs are NEVER overwritten.
    os.chmod(root, 0o2770)
    require(root.stat().st_gid in (96293, 65534), "project_group_run_required")
    return root


def private_json(path, value):
    with _new_private_handle(path) as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
        handle.flush(); os.fsync(handle.fileno())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "cache-replay"), default="demo")
    parser.add_argument("--planner", choices=("mock", "api", "local-qwen"), default="mock")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=PROTECTED_ROOT/"tricompose_v1_2/llm_agent_runs")
    parser.add_argument("--case-count", type=int, default=10)
    parser.add_argument("--budget-units", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--api-key-env", default="TRICOMPOSE_LLM_API_KEY")
    parser.add_argument("--response-format", choices=("json_schema", "json_object"), default="json_schema")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--allow-network", action="store_true")
    return parser


def run(args):
    require(1 <= args.case_count <= 80 and 4 <= args.budget_units <= 10000
            and 1 <= args.max_steps <= 64, "bounded_pilot_options")
    require(args.planner != "api" or args.allow_network, "explicit_network_opt_in_required")
    require(not args.allow_network or args.planner == "api", "network_only_for_api_mode")
    require(args.planner == "local-qwen" or args.model_path is None, "local_model_path_only_for_local_planner")
    sources = {}
    if args.mode == "cache-replay":
        from tricompose_v12.runtime_dispatch import require_slurm
        require_slurm()
        job = os.environ.get("SLURM_JOB_ID", "")
        require(job.isdigit() and f"/job_{job}/" in Path("/proc/self/cgroup").read_text(), "actual_cache_replay_slurm_required")
        # Reuse the old hash-pinned, synthetic-only metadata loader unchanged.
        from benchmark_probe_repair_v1 import load_primary
        bank, policy = load_primary(sources)
        cases = list(bank)[:args.case_count]  # Fixed source order; no easy-case screening.
        tools = tools_from_orders(policy["image_order"], policy["report_order"])
        first_slot = (*policy["image_order"][0], policy["report_order"][0])
        inputs = [(snapshot(bank[case][first_slot]), CachedExecutor(bank[case])) for case in cases]
    else:
        tools = tools_from_orders([("chexgenbench_sana", 0), ("roentgen_v2", 0)], ["maira2", "cxrmate_single"])
        inputs = [(observation(), DemoExecutor())]
    if args.planner == "local-qwen":
        require(args.model_path is not None, "explicit_audited_qwen_path_required")
        from tricompose_llm.local_qwen import gpu_guard
        gpu_guard()
    if args.planner == "api":
        require(re.fullmatch(r"TRICOMPOSE_[A-Z0-9_]{1,64}", args.api_key_env) is not None, "explicit_api_key_env_name")
        planner = OpenAICompatiblePlanner(base_url=args.base_url or os.environ.get("TRICOMPOSE_LLM_BASE_URL"),
            model=args.model or os.environ.get("TRICOMPOSE_LLM_MODEL"), api_key=os.environ.get(args.api_key_env),
            timeout_seconds=args.timeout_seconds, response_format=args.response_format)
    elif args.planner == "mock":
        planner = DemoPlanner()
    else:
        planner = None  # Construct in protected native log only, never on login/CPU Slurm.
    root = new_private_run(args.output_root, args.run_id)
    code_pins = {str(p.relative_to(WORKSPACE)): sha256_file(p) for p in sorted((WORKSPACE/"TriCompose-v1.2/agent").rglob("*.py"))}
    for relative in ("TriCompose-v1.2/src/tricompose_v12/probe_repair_v1.py",
            "TriCompose-v1.2/src/tricompose_v12/legacy_replay_adapter.py",
            "TriCompose-v1.2/src/tricompose_v12/invariant_verification.py",
            "TriCompose-v1.2/src/tricompose_v12/runtime_dispatch.py",
            "TriCompose-v1.2/src/tricompose_v12/live_workers.py",
            "TriCompose-v1.2/src/tricompose_v12/automatic_replay.py",
            "TriCompose-v1.2/tools/benchmark_probe_repair_v1.py",
            "TriCompose-v1.0/eval/report_v1_1/contracts.py",
            "src/tricompose/verifiers/qwenvl_cxr_report.py"):
        code_pins[relative] = sha256_file(WORKSPACE/relative)
    private_json(root/"start_manifest.json", {"schema_version": VERSION, "status": "started",
        "mode": args.mode, "planner": args.planner, "case_count": len(inputs),
        "budget_units_per_case": args.budget_units, "max_steps": args.max_steps,
        "source_pins": sources, "code_pins": code_pins, "actual_generator_calls": 0,
        "clinical_repair_success": None, "original_selection_changed": False})
    results = []
    started = time.monotonic()
    with _new_private_handle(root/"native.log") as log, _new_private_handle(root/"events.jsonl") as journal:
        def record(event):
            journal.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
            journal.flush(); os.fsync(journal.fileno())
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if planner is None:
                from tricompose_llm.local_qwen import LocalQwenPlanner
                planner = LocalQwenPlanner(args.model_path)
            for index, (initial, executor) in enumerate(inputs):
                def sink(event, index=index):
                    record({"case_index": index, **event})
                controller = LLMRepairController(initial, tools, budget_units=args.budget_units,
                    max_steps=args.max_steps, event_sink=sink)
                result = controller.run(planner, executor)
                private_json(root/f"case_{index:03d}.json", result)
                results.append(result)
                if result["terminal_reason"] == "planner_failed_or_invalid": break
    audit = planner.audit() if args.planner == "local-qwen" else None
    require(all(sha256_file(WORKSPACE/path) == expected for path, expected in code_pins.items()), "agent_code_changed_during_run")
    summary = {"schema_version": VERSION, "status": "complete" if len(results) == len(inputs)
        and all(r["terminal_reason"] != "planner_failed_or_invalid" for r in results) else "planner_failed",
        "mode": args.mode, "planner": args.planner, "cases_requested": len(inputs), "cases_completed": len(results),
        "planner_calls": sum(r["planner_calls"] for r in results), "tool_attempts": sum(r["tool_attempts"] for r in results),
        "accepted_proxy_transitions": sum(r["accepted_proxy_transitions"] for r in results),
        "spent_simulated_units": sum(r["spent_units"] for r in results),
        "actual_api_attempts": getattr(planner, "api_attempts", 0),
        "actual_local_planner_attempts": getattr(planner, "local_attempts", 0),
        "api_token_usage": getattr(planner, "usage", None) if args.planner == "api" else None,
        "api_usage_available_calls": getattr(planner, "usage_available_calls", 0),
        "actual_generator_calls": 0, "actual_regeneration_executed": False,
        "clinical_fault_location": None, "clinical_repair_success": None,
        "local_planner_audit": audit, "elapsed_seconds": round(time.monotonic()-started, 3),
        "original_selection_changed": False}
    private_json(root/"summary.json", summary)
    artifacts = {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                 for p in root.iterdir() if p.is_file()}
    private_json(root/"manifest.json", {**summary, "artifacts": artifacts, "code_pins": code_pins, "source_pins": sources})
    print(json.dumps({"stage": "llm_repair_agent", "status": summary["status"],
                      "elapsed_seconds": summary["elapsed_seconds"], "manifest_sha256": sha256_file(root/"manifest.json")}, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


def main():
    args = build_parser().parse_args()
    os.umask(0o007)
    try:
        return run(args)
    except Exception:
        # Interrupted started runs keep their charged durable journal for audit.
        print(json.dumps({"stage": "llm_repair_agent", "status": "failed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
