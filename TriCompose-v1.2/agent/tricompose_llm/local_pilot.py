"""Shared versioned local-pilot runner with an explicit trusted planner factory.

Only immutable numeric-cache replay is enabled, never fresh generation. The
original v1/v2 entry points remain unchanged and reproducible.
"""
import contextlib
import json
import os
import time

from run import (WORKSPACE, _new_private_handle, new_private_run, private_json, require, sha256_file)
from benchmark_probe_repair_v1 import load_primary
from tricompose_v12.probe_repair_v1 import snapshot
from .controller import CachedExecutor, LLMRepairController, tools_from_orders
from .local_qwen import gpu_guard


def run_local_pilot(args, *, planner_factory, interface_version):
    require(args.mode == "cache-replay" and args.planner == "local-qwen"
        and args.model_path is not None and not args.allow_network, "local_cached_pilot_only")
    require(1 <= args.case_count <= 80 and 4 <= args.budget_units <= 10000
        and 1 <= args.max_steps <= 64, "bounded_pilot_options")
    gpu_guard()
    sources = {}
    bank, policy = load_primary(sources)
    cases = list(bank)[:args.case_count]
    tools = tools_from_orders(policy["image_order"], policy["report_order"])
    first_slot = (*policy["image_order"][0], policy["report_order"][0])
    root = new_private_run(args.output_root, args.run_id)
    pins = {str(p.relative_to(WORKSPACE)): sha256_file(p)
        for p in sorted((WORKSPACE/"TriCompose-v1.2/agent").rglob("*.py"))}
    for relative in ("TriCompose-v1.2/src/tricompose_v12/probe_repair_v1.py",
        "TriCompose-v1.2/src/tricompose_v12/legacy_replay_adapter.py",
        "TriCompose-v1.2/src/tricompose_v12/invariant_verification.py",
        "TriCompose-v1.2/src/tricompose_v12/runtime_dispatch.py",
        "TriCompose-v1.2/src/tricompose_v12/live_workers.py",
        "TriCompose-v1.2/src/tricompose_v12/automatic_replay.py",
        "TriCompose-v1.2/tools/benchmark_probe_repair_v1.py",
        "TriCompose-v1.0/eval/report_v1_1/contracts.py",
        "src/tricompose/verifiers/qwenvl_cxr_report.py"):
        pins[relative] = sha256_file(WORKSPACE/relative)
    private_json(root/"start_manifest.json", {"schema_version": interface_version, "status": "started",
        "mode": "cache-replay", "planner": "local-qwen", "case_count": len(cases),
        "budget_units_per_case": args.budget_units, "max_steps": args.max_steps,
        "source_pins": sources, "code_pins": pins, "actual_generator_calls": 0,
        "clinical_repair_success": None, "original_selection_changed": False})
    results = []
    started = time.monotonic()
    with _new_private_handle(root/"native.log") as log, _new_private_handle(root/"events.jsonl") as journal:
        def record(event):
            journal.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
            journal.flush(); os.fsync(journal.fileno())
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            planner = planner_factory(args.model_path)
            for index, case in enumerate(cases):
                def sink(event, index=index): record({"case_index": index, **event})
                controller = LLMRepairController(snapshot(bank[case][first_slot]), tools,
                    budget_units=args.budget_units, max_steps=args.max_steps, event_sink=sink)
                result = controller.run(planner, CachedExecutor(bank[case]))
                private_json(root/f"case_{index:03d}.json", result)
                results.append(result)
                if result["terminal_reason"] == "planner_failed_or_invalid": break
    audit = planner.audit()
    require(audit.get("frozen") is True and audit.get("interface_version") == interface_version,
            "declared_frozen_planner_interface_required")
    require(all(sha256_file(WORKSPACE/path) == expected for path, expected in pins.items()),
            "agent_code_changed_during_run")
    completed = sum(r["terminal_reason"] != "planner_failed_or_invalid" for r in results)
    summary = {"schema_version": interface_version,
        "status": "complete" if completed == len(cases) else "planner_failed",
        "mode": "cache-replay", "planner": "local-qwen", "cases_requested": len(cases),
        "cases_processed": len(results), "cases_completed": completed,
        "policy_requests": sum(r["planner_calls"] for r in results),
        "tool_attempts": sum(r["tool_attempts"] for r in results),
        "accepted_proxy_transitions": sum(r["accepted_proxy_transitions"] for r in results),
        "spent_simulated_units": sum(r["spent_units"] for r in results),
        "actual_api_attempts": 0, "actual_local_planner_attempts": planner.local_attempts,
        "programmatic_terminal_stops": audit.get("programmatic_terminal_stops", 0),
        "actual_generator_calls": 0, "actual_regeneration_executed": False,
        "clinical_fault_location": None, "clinical_repair_success": None,
        "local_planner_audit": audit, "elapsed_seconds": round(time.monotonic()-started, 3),
        "original_selection_changed": False}
    private_json(root/"summary.json", summary)
    artifacts = {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        for p in root.iterdir() if p.is_file()}
    private_json(root/"manifest.json", {**summary, "artifacts": artifacts, "code_pins": pins, "source_pins": sources})
    print(json.dumps({"stage": "llm_bounded_routing_pilot", "status": summary["status"],
        "elapsed_seconds": summary["elapsed_seconds"], "manifest_sha256": sha256_file(root/"manifest.json")}, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1
