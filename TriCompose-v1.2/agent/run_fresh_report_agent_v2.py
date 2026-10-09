#!/usr/bin/env python3
"""Supplemental-weight preflight for the unchanged fresh report bridge.

Reuses the sealed CPU-only v1 preflight; no generator/scorer/policy changes.
Includes Vicuna, BiomedBERT, CheXagent vision, CheXbert BERT and LLaVA source.
The consumed v1 file/plan remain immutable and byte-authenticated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path

import run_fresh_report_agent as bridge
from contracts import (PROTECTED_ROOT, new_atomic_run, commit_atomic_run,
    discard_atomic_run, read_json, require_inside, sha256_file, write_private_json)
from tricompose_v12.live_workers import CHEXBERT_BERT, check_pins

VERSION = "tricompose-fresh-report-supplemental-assets-v2"
SOURCE = PROTECTED_ROOT / "tricompose_v1_2/llm_fresh_report_plans/fresh_report2_12799642_001"
SOURCE_SHA = "f23a7d5f63c5ac520868455364a4a5e831a7d70d89a2ad74868008b021b1fb37"


def supplemental_inventory(workers):
    """Enumerate metadata only from explicit model/source dirs, not datasets."""
    result = {}
    for model, spec in workers.items():
        extras = spec.get("extra_args", [])
        roots = [(Path(extras[i+1]), extras[i] == "--external-root")
            for i in range(0, len(extras), 2)
            if extras[i] in ("--model-base", "--biomedbert-dir", "--vision-dir", "--external-root")]
        if model == "chexbert":
            roots.append((CHEXBERT_BERT, False))
        files = []
        for root, source_only in roots:
            if not root.is_dir():
                raise ValueError("declared_secondary_model_assets_missing")
            suffixes = {".py"} if source_only else {".json", ".txt", ".model", ".py", ".safetensors", ".bin"}
            selected = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in suffixes]
            if not selected:
                raise ValueError("declared_secondary_model_assets_empty")
            files.extend(str(p) for p in selected)
        result[model] = sorted(set(files))
    return result


def validate(plan):
    bridge.validate_plan(plan)
    if (plan.get("supplemental_asset_version") != VERSION
            or plan.get("preflight_source_manifest_sha256") != SOURCE_SHA
            or plan.get("supplemental_asset_inventory") != supplemental_inventory(plan["workers"])):
        raise ValueError("complete_declared_supplemental_assets_required")
    for model, paths in plan["supplemental_asset_inventory"].items():
        if not set(paths) <= set(plan["workers"][model]["asset_pins"]):
            raise ValueError("secondary_weight_or_source_pin_missing")


def prepare(args):
    bridge.gate.cpu_guard()
    old = bridge.load_plan(argparse.Namespace(plan_run=SOURCE, plan_manifest_sha256=SOURCE_SHA))
    plan = copy.deepcopy(old)
    inventory = supplemental_inventory(plan["workers"])
    for model, paths in inventory.items():
        for path in paths:
            p = Path(path)
            plan["workers"][model]["asset_pins"][path] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
    plan.update(supplemental_asset_version=VERSION, supplemental_asset_inventory=inventory,
                preflight_source_manifest_sha256=SOURCE_SHA)
    for p in (Path(__file__).resolve(), bridge.ROOT / "TriCompose-v1.2/tests/test_fresh_report_supplemental_assets.py"):
        plan["source_pins"][str(p)] = sha256_file(p)
    plan["artifact_pins"][str(SOURCE / "manifest.json")] = SOURCE_SHA
    plan["artifact_pins"][str(SOURCE / "plan.json")] = sha256_file(SOURCE / "plan.json")
    validate(plan)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temporary / "plan.json", plan)
        write_private_json(temporary / "manifest.json", {"schema_version": bridge.VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "supplemental_asset_version": VERSION,
            "plan_sha256": sha256_file(p), "planned_maximum": plan["planned_maximum"],
            "source_pins_sha256": bridge._digest(plan["source_pins"]), "new_model_calls": 0,
            "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def run(args):
    bridge.gpu_guard()
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / "manifest.json") != args.plan_manifest_sha256:
        raise ValueError("reviewed_supplemental_plan_changed")
    manifest = read_json(root / "manifest.json")
    if manifest.get("supplemental_asset_version") != VERSION:
        raise ValueError("supplemental_preflight_plan_required")
    # bridge.run authenticates the entire plan/source/all main+extra weights
    # before model calls, and checks them again afterwards. No redundant third
    # whole-weight pass here. The original v1 implementation stays unchanged.
    target = bridge.run(args)
    summary = read_json(target / "summary.json")
    books = read_json(target / "execution_summary.json")["case_ledgers"]
    kinds = Counter()
    for book in books:
        requests = {e["request"]["operation_id"]: e["request"] for e in book["events"]
                    if e["event"] == "attempt_reserved"}
        kinds.update(requests[e["operation_id"]]["kind"] for e in book["events"]
                     if e["event"] == "attempt_completed")
    verification = {"schema_version": VERSION,
        "base_manifest_sha256": sha256_file(target / "manifest.json"),
        "plan_manifest_sha256": args.plan_manifest_sha256,
        "confirmed_completed_worker_phases": dict(kinds),
        "charged_worker_attempts": summary["charged_generator_verifier_attempts"],
        "confirmed_new_cxr_artifacts": kinds["cxr_generator"],
        "confirmed_new_report_artifacts": kinds["report_generator"],
        "completed_scored_report_candidates": summary["completed_report_candidates"],
        "fresh_generation_confirmed_by_completed_receipts": bool(kinds["cxr_generator"] or kinds["report_generator"]),
        "legacy_fresh_generation_executed_field_scope": "worker_dispatch_attempted_not_proof_of_completed_artifacts",
        "clinical_repair_success": False, "training_performed": False}
    p = write_private_json(target / "execution_verification_v2.json", verification)
    write_private_json(target / "manifest_v2.json", {"schema_version": VERSION,
        "status": "completed_fresh_report_bridge_contract_unvalidated",
        "base_manifest_sha256": verification["base_manifest_sha256"],
        "execution_verification_sha256": sha256_file(p), "clinical_acceptance": False})
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--run-id", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan-run", type=Path, required=True)
    r.add_argument("--plan-manifest-sha256", required=True)
    r.add_argument("--output-root", type=Path, required=True); r.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        if args.command == "prepare":
            target = prepare(args)
        else:
            target = run(args)
        manifest_name = "manifest.json" if args.command == "prepare" else "manifest_v2.json"
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
                          "manifest_sha256": sha256_file(target / manifest_name)}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
