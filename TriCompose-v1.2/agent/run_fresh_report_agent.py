#!/usr/bin/env python3
"""Versioned local-Qwen -> real frozen report -> unchanged verifier bridge.

prepare: CPU Slurm hashing/metadata only; does not instantiate models.
run: separately approved GPU Slurm only, no training or external service.
All old controller, worker, scorer and historical runs stay unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for relative in ("src", "TriCompose-v1.0/src", "TriCompose-v1.1/src", "TriCompose-v1.2/src",
                 "TriCompose-v1.2/benchmarks", "TriCompose-v1.2/tools", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT / relative))

from contracts import (PROTECTED_ROOT, RUN_ID_PATTERN, commit_atomic_run, discard_atomic_run,
    new_atomic_run, private_directory, read_json, require_inside, sha256_file,
    write_private_json, write_private_text)
import fresh_output_acceptance as gate
from run_bounded_regeneration import candidate_row, load as load_parent
from tricompose_v11.cxr_contracts import canonical_json_sha256, validate_cxr_request
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import one_case, score_csv
from tricompose_v12.live_receipts import PROFILE
from tricompose_v12.live_workers import (check_pins, preflight_generator, registry,
    run_private_process, source_pins)
from tricompose_v12.report_expert_control import MODELS
from tricompose_v12.runtime_dispatch import ProtectedJournal
from tricompose_llm.contracts import validate_decision
from tricompose_llm.fresh_report_worker import additional_report
from tricompose_llm.live_report_bridge import FreshReportSession, VERSION
from tricompose_llm.local_qwen import EXPECTED_WEIGHT, gpu_guard

QWEN_MODEL = Path("/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct")
QWEN_PYTHON = Path("/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python")
POLICY = {"schema_version": VERSION, "report_models": ["cxrmate_single"],
    "call_budget_per_case": 10, "max_retries_per_operation": 0,
    "timeout_seconds_per_worker_process": 180, "policy_timeout_seconds": 150,
    "policy_max_steps": 3, "alternative_report_models": list(MODELS[1:]),
    "image_regeneration_enabled": False, "external_api_enabled": False,
    "case_scope": "two_existing_predeclared_ehr_only_development_anchors",
    "training_allowed": False, "clinical_acceptance": False}


def own_source_pins():
    files = list((ROOT / "TriCompose-v1.2/agent").rglob("*.py"))
    files += [ROOT / "TriCompose-v1.2/tools/fresh_output_acceptance.py",
              ROOT / "TriCompose-v1.2/tests/test_fresh_report_agent_bridge.py"]
    return {str(p): sha256_file(p) for p in files}


def validate_plan(plan):
    if (plan["schema_version"] != VERSION or plan["policy"] != POLICY
            or plan["profile"] != PROFILE or len(plan["cases"]) != 2
            or len({c["case_id"] for c in plan["cases"]}) != 2
            or set(plan["workers"]) != {"roentgen_v2", "xrv", "chexbert", *MODELS}
            or plan["data_origin"] != "original_fully_synthetic_pool80"
            or plan["new_model_calls"] != 0 or plan["source_bodies_parsed"] is not False
            or plan["minimum_gpu_vram_gib"] != 40
            or plan["planned_maximum"] != {"new_cxrs": 2, "new_reports": 8,
                "charged_generator_verifier_attempts": 20, "charged_policy_requests": 6}):
        raise ValueError("exact_two_case_report_only_frozen_plan_required")
    for case in plan["cases"]:
        anchor = gate.anchor_from_record(case["anchor"])
        if case["ehr_anchor_sha256"] != anchor.sha256 or case["case_id"] != anchor.case_id or len(case["requests"]) != 1:
            raise ValueError("fixed_ehr_case_required")
        row = case["requests"][0]; request = row["request"]
        validate_cxr_request(request)
        if (request["model_id"] != "roentgen_v2" or request["seed"] != 0
                or request["case_id"] != anchor.case_id
                or canonical_json_sha256(request) != row["canonical_request_sha256"]
                or sha256_file(row["source_path"]) != row["source_sha256"]
                or read_json(row["source_path"]) != request
                or request["inputs"]["synthetic_ehr"]["sha256"] != anchor.ehr_sha256
                or request["inputs"]["ehr_facts"]["sha256"] != anchor.ehr_facts_sha256):
            raise ValueError("original_prompt_and_ehr_must_be_unchanged")
        gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
    if (plan["qwen"]["model_path"] != str(QWEN_MODEL) or plan["qwen"]["python"] != str(QWEN_PYTHON)
            or plan["qwen"]["asset_pins"][str(QWEN_MODEL / "model.safetensors")]["sha256"] != EXPECTED_WEIGHT):
        raise ValueError("existing_audited_local_qwen_required")
    check_pins(plan["source_pins"])
    check_pins(plan["artifact_pins"])
    check_pins(plan["qwen"]["asset_pins"])
    for spec in plan["workers"].values():
        if spec["status"] != "preflighted" or spec["factory_instantiated"] is not False:
            raise ValueError("frozen_preflighted_workers_required")
        check_pins(spec["asset_pins"])


def prepare(args):
    gate.cpu_guard()
    parent_path, parent_sha = gate.SOURCES["plan"]
    parent = load_parent(argparse.Namespace(plan_run=parent_path, plan_manifest_sha256=parent_sha))
    pins = {**source_pins(), **own_source_pins()}
    workers = {name: copy.deepcopy(parent["workers"][name]) for name in
               ("roentgen_v2", "cxrmate_single", "xrv", "chexbert")}
    for model in MODELS[1:]:
        workers[model] = preflight_generator(registry()[model])
    if not os.access(QWEN_PYTHON, os.X_OK):
        raise ValueError("existing_qwen_environment_required")
    qfiles = [QWEN_MODEL / name for name in ("config.json", "generation_config.json", "processor_config.json",
              "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors")]
    qpins = {str(p): {"sha256": sha256_file(p), "size_bytes": p.stat().st_size} for p in qfiles}
    cases = []
    for old in parent["cases"]:
        row = copy.deepcopy(old["requests"][0])
        row["request"] = copy.deepcopy(old["original_request"])
        row["canonical_request_sha256"] = canonical_json_sha256(row["request"])
        cases.append({k: copy.deepcopy(old[k]) for k in
            ("case_id", "anchor", "ehr_anchor_sha256", "opaque_source_index")})
        cases[-1]["requests"] = [row]
    plan = {"schema_version": VERSION, "profile": PROFILE, "policy": POLICY, "cases": cases,
        "workers": workers, "source_pins": pins,
        "artifact_pins": {**parent["artifact_pins"], str(parent_path / "manifest.json"): parent_sha,
                          str(parent_path / "plan.json"): sha256_file(parent_path / "plan.json")},
        "qwen": {"python": str(QWEN_PYTHON), "model_path": str(QWEN_MODEL), "asset_pins": qpins},
        "data_origin": "original_fully_synthetic_pool80", "source_bodies_parsed": False,
        "new_model_calls": 0, "minimum_gpu_vram_gib": 40,
        "planned_maximum": {"new_cxrs": 2, "new_reports": 8,
            "charged_generator_verifier_attempts": 20, "charged_policy_requests": 6}}
    validate_plan(plan)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temporary / "plan.json", plan)
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(p),
            "source_pins_sha256": _digest(pins), "planned_maximum": plan["planned_maximum"],
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def load_plan(args):
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / "manifest.json") != args.plan_manifest_sha256:
        raise ValueError("reviewed_plan_manifest_changed")
    manifest = read_json(root / "manifest.json")
    if (manifest["schema_version"] != VERSION or manifest["status"] != "prepared_cpu_only_gpu_not_submitted"
            or sha256_file(root / "plan.json") != manifest["plan_sha256"]):
        raise ValueError("sealed_cpu_preflight_required")
    plan = read_json(root / "plan.json")
    validate_plan(plan)
    return plan


def propose_in_subprocess(state, step_root, plan):
    private_directory(step_root)
    runtime = step_root / "runtime"
    private_directory(runtime)
    state_path = write_private_json(step_root / "numeric_state.json", state)
    digest = sha256_file(state_path)
    result_path = step_root / "policy_result.json"
    run_private_process([plan["qwen"]["python"], str(ROOT / "TriCompose-v1.2/agent/run_local_qwen_decision.py"),
        "--state", str(state_path), "--state-sha256", digest, "--output", str(result_path),
        "--model-path", plan["qwen"]["model_path"]], runtime, POLICY["policy_timeout_seconds"])
    # run_private_process waits for normal process exit before returning:
    # no resident Qwen model can overlap the subsequent large report expert.
    result = read_json(result_path)
    audit = result["audit"]
    expected = {Path(p).name: v["sha256"] for p, v in plan["qwen"]["asset_pins"].items()}
    if (result["status"] != "completed" or result["state_sha256"] != digest
            or audit["frozen"] is not True or audit["asset_pins"] != expected
            or audit["local_attempts"] != 1 or audit["generate_attempts"] != 1
            or any(audit["failure_counts"].values())):
        raise ValueError("one_successful_pinned_local_policy_call_required")
    validate_decision(result["decision"], state)
    return result["decision"], audit


def run(args):
    gpu_guard()
    plan = load_plan(args)
    # Capacity query only; this coordinator never loads model weights in CUDA.
    import torch
    if torch.cuda.get_device_properties(0).total_memory < plan["minimum_gpu_vram_gib"] * 1024 ** 3:
        raise ValueError("allocated_gpu_below_frozen_worker_planning_memory")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("opaque_run_id_required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = output / args.run_id
    private_directory(root)
    private_directory(root / "cases")
    started = time.monotonic()
    selections, rows, triples, books, policy_audits = [], [], [], [], []
    write_private_json(root / "start_manifest.json", {"schema_version": VERSION, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256, "automatic_resume": False})
    try:
        for case in plan["cases"]:
            initial, book, _ = one_case(case, plan, root)
            cr = root / "cases" / case["case_id"]
            if len(initial) != 1:
                selections.append({"case_id": case["case_id"], "status": "abstain_baseline_chain_incomplete",
                    "selected_candidate_id": None, "charged_worker_attempts": book["charged_model_attempts"],
                    "charged_policy_requests": 0, "clinical_repair_success": False})
                books.append(book)
                continue
            base = candidate_row(initial[0])
            rows.append(base); triples.append(initial[0])
            context = gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
            with ProtectedJournal(cr / "live_policy.journal.jsonl") as journal:
                session = FreshReportSession(base, context, book, max_steps=POLICY["policy_max_steps"], sink=journal.append)
                while True:
                    state = session.begin_planning()
                    if state is None:
                        break
                    step = session.policy_requests - 1
                    try:
                        decision, audit = propose_in_subprocess(state, cr / f"policy_step_{step}", plan)
                    except Exception:
                        session.policy_failed()
                        break
                    policy_audits.append(audit)
                    model = session.receive_decision(decision)
                    if model is None:
                        break
                    fresh, book = additional_report(case, plan, root, initial[0], session.ledger,
                                                    model=model, step_index=step)
                    row = candidate_row(fresh) if fresh is not None else None
                    session.finish_report(row, book)
                    if row is not None:
                        rows.append(row); triples.append(fresh)
                result = session.result()
                selection_path = write_private_json(cr / "sealed_selection.json", result)
                journal.append({"event": "selection_sealed", "sha256": sha256_file(selection_path)})
                selections.append(result); books.append(session.ledger)
        write_private_json(root / "selection.json", {"records": selections})
        write_private_json(root / "score_rows.json", {"records": rows})
        write_private_text(root / "score_table.csv", score_csv([
            {**r, "verification_status": r["receipt"]["verification_status"]} for r in rows]))
        write_private_json(root / "completed_triplets.json", {"records": triples})
        write_private_json(root / "execution_summary.json", {"case_ledgers": books})
        write_private_json(root / "policy_audits.json", {"records": policy_audits})
        validate_plan(plan)
        summary = {"schema_version": VERSION, "status": "completed_fresh_report_bridge_unvalidated",
            "fixed_ehr_cases": len(selections), "completed_report_candidates": len(rows),
            "accepted_proxy_transitions": sum(s.get("accepted_proxy_transitions", 0) for s in selections),
            "charged_generator_verifier_attempts": sum(b["charged_model_attempts"] for b in books),
            "charged_policy_requests": sum(s["charged_policy_requests"] for s in selections),
            "completed_qwen_decisions": len(policy_audits), "measured_gpu_seconds": None,
            "runtime_seconds_including_load_io": round(time.monotonic() - started, 3),
            "fresh_generation_executed": True, "image_regeneration_installed": False,
            "external_api_calls": 0, "clinical_repair_success": False, "clinical_accuracy": None,
            "training_performed": False, "scorers_or_thresholds_changed": False,
            "comparison_to_numeric_cache_pilot_is_not_like_for_like": True}
        write_private_json(root / "summary.json", summary)
        files = ["selection.json", "score_rows.json", "score_table.csv", "completed_triplets.json",
                 "execution_summary.json", "policy_audits.json", "summary.json"]
        write_private_json(root / "manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "plan_manifest_sha256": args.plan_manifest_sha256,
            "artifacts": {name: {"sha256": sha256_file(root / name)} for name in files},
            "clinical_acceptance": False})
    except BaseException:
        write_private_json(root / "failed_manifest.json", {"schema_version": VERSION,
            "status": "failed_retained_for_cost_reconciliation", "automatic_resume": False})
        raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan-run", type=Path, required=True)
    r.add_argument("--plan-manifest-sha256", required=True)
    r.add_argument("--output-root", type=Path, required=True)
    r.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        target = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
                          "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
