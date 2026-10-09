#!/usr/bin/env python3
"""Two fixed synthetic anchors, two predeclared new CXR actions; no planner.

CPU prepare reads requests/receipts/file stats, not clinical bodies or weights.
Approved GPU run reuses the existing frozen generation/verification chain.
This expands candidates, not clinical repair or an LLM-versus-rule experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

import run_fresh_report_agent as existing
import run_fresh_report_secondary as history
from contracts import (new_atomic_run, commit_atomic_run, private_directory,
    require_inside, sha256_file, write_private_json, write_private_text)
from tricompose_v11.cxr_contracts import canonical_json_sha256, validate_cxr_request
from tricompose_v12.live_execution import one_case, score_csv
from tricompose_v12.live_workers import check_pins, registry, require_gpu_slurm
from tricompose_v12.bounded_regeneration import compare
from tricompose_v12.execution_ledger import restore_ledger

BASE = history.PROTECTED_ROOT / "tricompose_v1_2"
Reader = history.postflight.MetadataReader
require = history.require
VERSION = "tricompose-fixed-ehr-cxr-action-diversification-v1"
NATIVE = BASE / "llm_fresh_report_plans/fresh_report2_fullassets_12799642_001"
NATIVE_SHA = "01516d6965c3b0e287acff715da1735086580ddc8aa8d9ef39525ea8a9533e1f"
SANA = BASE / "live_worker_plans/live_workers2_12605930_001"
SANA_SHA = "1e18eb7664b406551100ffad296dbbf4c75f8876fe349b062f1d1db8089cc55e"
SLOTS = (("roentgen_v2", 2), ("chexgenbench_sana", 1))
WORKERS = {"roentgen_v2", "chexgenbench_sana", "cxrmate_single", "xrv", "chexbert"}
POLICY = {"report_models": ["cxrmate_single"], "call_budget_per_case": 8,
    "max_retries_per_operation": 0, "timeout_seconds_per_worker_process": 120}
CONFIG = {"cases": 2, "image_actions_per_case": 2, "image_slots": [list(s) for s in SLOTS],
    "report_expert": "cxrmate_single", "maximum_worker_requests": 16,
    "maximum_new_cxrs": 4, "maximum_new_reports": 4, "new_planner_calls": 0,
    "external_api_calls": 0, "model_retries": 0, "minimum_gpu_vram_gib": 16,
    "training_allowed": False, "clinical_acceptance": False, "selection_changed": False}


def sealed(root, pin, readers, *, plan=False):
    reader = Reader(root); readers.append(reader)
    reader.hash(reader.root / "manifest.json", pin)
    manifest = reader.json(reader.root / "manifest.json")
    if plan:
        reader.hash(reader.root / "plan.json", manifest["plan_sha256"])
        return reader, reader.json(reader.root / "plan.json")
    return reader, manifest


def reseed(original, model, seed):
    """Only the seed and its request ID may change; no prompt editing."""
    require(type(seed) is int and original["model_id"] == model and type(original["seed"]) is int
        and original["seed"] == 0 and (model, seed) in SLOTS, "fixed_original_and_declared_action_required")
    result = deepcopy(original)
    result["seed"] = seed
    result["request_id"] = f"cxrreq_{result['case_id']}_{model}_s{seed:06d}"
    return result


def shared_intent(requests, anchor):
    require(len(requests) == 2 and tuple((r["model_id"], r["seed"]) for r in requests) == SLOTS,
        "exact_fixed_two_action_order_required")
    for r in requests:
        require(r["case_id"] == anchor["case_id"]
            and r["inputs"]["synthetic_ehr"]["sha256"] == anchor["ehr_sha256"]
            and r["inputs"]["ehr_facts"]["sha256"] == anchor["ehr_facts_sha256"]
            and r["frozen_model_required"] is True and r["adapter_must_not_add_prefix"] is True,
            "same_fixed_ehr_facts_and_frozen_contract_required")
    a, b = (r["inputs"]["final_prompt"] for r in requests)
    require(a["clinical_intent_sha256"] == b["clinical_intent_sha256"]
        and a["included_direct_fact_ids"] == b["included_direct_fact_ids"],
        "same_grounded_clinical_intent_not_same_surface_text_required")
    require(requests[0]["inputs"]["staging_run_manifest"] == requests[1]["inputs"]["staging_run_manifest"],
        "same_original_staging_contract_required")


def validate_plan(plan, *, weights=False):
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG and plan["policy"] == POLICY
        and plan["data_origin"] == "original_fully_synthetic_pool80"
        and plan["new_model_calls"] == 0 and plan["source_bodies_parsed"] is False
        and len(plan["cases"]) == 2 and len({c["case_id"] for c in plan["cases"]}) == 2
        and set(plan["workers"]) == WORKERS, "reviewed_fixed_synthetic_probe_plan_required")
    check_pins(plan["source_pins"]); check_pins(plan["artifact_pins"])
    for name, spec in plan["workers"].items():
        require(spec["status"] == "preflighted" and spec["factory_instantiated"] is False
            and spec["model_id"] == name and os.access(spec["python"], os.X_OK), "unchanged_available_frozen_workers_required")
        require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == v
            for p, v in spec["asset_stats"].items()), "unchanged_model_asset_stats_required")
        if weights: check_pins(spec["asset_pins"])
    for case in plan["cases"]:
        anchor = existing.gate.anchor_from_record(case["anchor"])
        require(anchor.case_id == case["case_id"] and anchor.sha256 == case["ehr_anchor_sha256"], "fixed_anchor_required")
        shared_intent([r["request"] for r in case["requests"]], anchor.record())
        for row in case["requests"]:
            original = Reader(Path(row["source_path"]).parent)
            original.hash(row["source_path"], row["source_sha256"])
            require(reseed(original.json(row["source_path"]), row["request"]["model_id"], row["request"]["seed"])
                == row["request"] and canonical_json_sha256(row["request"]) == row["canonical_request_sha256"],
                "only_seed_and_request_id_changed")
            validate_cxr_request(row["request"]); original.recheck()
        ctx = existing.gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
        existing.gate.validate_observation(case["baseline_row"], ctx)
        require(case["baseline_row"]["cxr_model_id"] == "roentgen_v2"
            and case["baseline_row"]["seed"] == 0 and case["baseline_row"]["report_model_id"] == "cxrmate_single",
            "original_common_expert_baseline_required")


def prepare(args):
    existing.gate.cpu_guard()  # Before any receipt/source inspection or output.
    readers = []
    nr, native = sealed(NATIVE, NATIVE_SHA, readers, plan=True)
    sr, sana = sealed(SANA, SANA_SHA, readers, plan=True)
    hr = Reader(history.SOURCE); readers.append(hr)
    hr.hash(hr.root / "manifest_v2.json", history.SOURCE_SHA)
    supplemental = hr.json(hr.root / "manifest_v2.json")
    hr.hash(hr.root / "manifest.json", supplemental["base_manifest_sha256"])
    hm = hr.json(hr.root / "manifest.json")
    hr.hash(hr.root / "score_rows.json", hm["artifacts"]["score_rows.json"]["sha256"])
    historical = hr.json(hr.root / "score_rows.json")["records"]
    workers = {name: deepcopy(native["workers"][name]) for name in WORKERS if name != "chexgenbench_sana"}
    workers["chexgenbench_sana"] = deepcopy(sana["workers"]["chexgenbench_sana"])
    registered = registry()["chexgenbench_sana"]
    require(all(workers["chexgenbench_sana"][k] == registered[k] for k in registered),
        "same_existing_sana_adapter_checkpoint_and_precision_required")
    for spec in workers.values():
        spec["asset_stats"] = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in spec["asset_pins"]}
        require(all(v[0] == spec["asset_pins"][p]["size_bytes"] for p, v in spec["asset_stats"].items()),
            "authenticated_asset_sizes_required")
    cases = []
    for old in native["cases"]:
        # This manifest is a request inventory, not a clinical body.
        source_root = Path(old["requests"][0]["source_path"]).parents[1]
        r = Reader(source_root); readers.append(r)
        manifest_pin = native["artifact_pins"].get(str(source_root / "manifest.json"))
        require(manifest_pin is not None, "original_request_manifest_pin_required")
        r.hash(source_root / "manifest.json", manifest_pin)
        source_manifest = r.json(source_root / "manifest.json")
        requests = []
        for model, seed in SLOTS:
            found = [row for row in source_manifest["requests"]
                if (row["case_id"], row["model_id"], row["seed"]) == (old["case_id"], model, 0)]
            require(len(found) == 1, "one_original_request_per_case_model_required")
            metadata = found[0]; path = r.path(source_root / metadata["path"])
            r.hash(path, metadata["sha256"]); original = r.json(path)
            request = reseed(original, model, seed)
            validate_cxr_request(request)
            requests.append({"request": request, "source_path": str(path), "source_sha256": metadata["sha256"],
                "canonical_request_sha256": canonical_json_sha256(request)})
        baseline = [row for row in historical if row["case_id"] == old["case_id"]
            and row["cxr_model_id"] == "roentgen_v2" and row["seed"] == 0 and row["report_model_id"] == "cxrmate_single"]
        require(len(baseline) == 1, "one_completed_common_expert_baseline_required")
        cases.append({**{k: deepcopy(old[k]) for k in ("case_id", "anchor", "ehr_anchor_sha256", "opaque_source_index")},
            "requests": requests, "baseline_row": baseline[0]})
    sources = {**native["source_pins"], **sana["source_pins"]}
    for path in (Path(__file__).resolve(), existing.ROOT / "TriCompose-v1.2/tests/test_cxr_action_diversification.py",
                 existing.ROOT / "docs/cxr_action_diversification_protocol.md"):
        sources[str(path)] = sha256_file(path)
    artifacts = {**native["artifact_pins"], **sana["source_artifact_pins"]}
    for reader in readers: reader.recheck(); artifacts.update(reader.pins)
    plan = {"schema_version": VERSION, "config": CONFIG, "policy": POLICY, "workers": workers, "cases": cases,
        "source_pins": sources, "artifact_pins": artifacts, "data_origin": "original_fully_synthetic_pool80",
        "source_bodies_parsed": False, "new_model_calls": 0, "historical_cohort_is_untouched_test": False,
        "historical_work_is_shared_sunk_not_free": True, "clinical_acceptance": False,
        "native_manifest_sha256": NATIVE_SHA, "sana_manifest_sha256": SANA_SHA,
        "historical_baseline_manifest_sha256": history.SOURCE_SHA}
    validate_plan(plan)
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    write_private_json(tmp / "plan.json", plan)
    write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
        "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(tmp / "plan.json"),
        "config": CONFIG, "new_model_calls": 0, "clinical_acceptance": False})
    commit_atomic_run(tmp, target)
    return target


def load_plan(args, *, weights=False):
    readers = []
    _, plan = sealed(args.plan_run, args.plan_manifest_sha256, readers, plan=True)
    validate_plan(plan, weights=weights)
    for reader in readers: reader.recheck()
    return plan


def slot_outcomes(case, triples, ledger):
    """Keep each predeclared slot, including incomplete/failed branches."""
    require(ledger["case_id"] == case["case_id"] and ledger["ehr_anchor_sha256"] == case["ehr_anchor_sha256"]
        and ledger["call_budget"] == 8
        and ledger["max_retries"] == 0 and ledger["pending_attempts"] == 0,
        "complete_case_level_bounded_ledger_required")
    replayed = restore_ledger(ledger["events"], case_id=ledger["case_id"],
        ehr_anchor_sha256=ledger["ehr_anchor_sha256"], call_budget=8, max_retries=0,
        execution_mode=ledger["execution_mode"], sink=lambda event: None).snapshot()
    require(replayed == ledger, "exact_durable_ledger_replay_required")
    result = []; events = ledger["events"]
    for slot, (model, seed) in enumerate(SLOTS):
        names = {f"cxr_{slot}", f"xrv_{slot}", f"report_{slot}_0", f"chexbert_{slot}_0"}
        reserved = [e for e in events if e["event"] == "attempt_reserved" and e["request"]["operation_id"] in names]
        ids = {e["reservation_id"] for e in reserved}
        terminals = [e for e in events if e["event"] in ("attempt_completed", "attempt_failed") and e["reservation_id"] in ids]
        rows = [t for t in triples if t["case_id"] == case["case_id"] and (t["cxr_model_id"], t["seed"]) == (model, seed)]
        require(len(rows) <= 1 and len(reserved) <= 4
            and all(row["report_model_id"] == "cxrmate_single" for row in rows), "one_chain_per_declared_slot_required")
        expected = [("cxr_generator", model), ("xrv", "xrv"), ("report_generator", "cxrmate_single"), ("chexbert", "chexbert")]
        require([(e["request"]["kind"], e["request"]["model_id"]) for e in reserved] == expected[:len(reserved)]
            and all(e["request"]["seed"] == seed for e in reserved), "exact_action_chain_and_seed_required")
        require(not rows or len(reserved) == 4 and all(e["event"] == "attempt_completed" for e in terminals),
            "completed_triple_requires_own_four_validated_calls")
        wall = sum(e["backend_elapsed_seconds"] for e in terminals)
        require(math.isfinite(wall) and wall >= 0, "finite_charged_worker_wall_time_required")
        result.append({"case_id": case["case_id"], "cxr_model_id": model, "seed": seed,
            "report_model_id": "cxrmate_single", "status": "completed_unvalidated" if rows else
                "failed_branch_unavailable" if any(e["event"] == "attempt_failed" for e in terminals) else "incomplete_unavailable",
            "charged_worker_requests": len(reserved), "validated_worker_calls": sum(e["event"] == "attempt_completed" for e in terminals),
            "worker_wall_seconds_including_startup_io": round(wall, 6), "clinical_acceptance": False,
            "triple": rows[0] if rows else None})
    require(sum(row["charged_worker_requests"] for row in result) == ledger["charged_model_attempts"],
        "no_unlisted_hidden_requests_or_dropped_charges")
    require(len(triples) == sum(row["triple"] is not None for row in result), "no_unlisted_duplicate_or_other_case_triple")
    return result


def run(args):
    require_gpu_slurm()  # Before reading clinical files or spawning workers.
    plan = load_plan(args, weights=True)
    import torch
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16*1024**3,
        "registered_generator_planning_memory_class_required")
    output = require_inside(args.output_root, BASE, must_exist=False)
    private_directory(output, exist_ok=True)
    # Stable paths: existing adapters embed absolute candidate paths.
    require(existing.RUN_ID_PATTERN.fullmatch(args.run_id) is not None, "opaque_run_id_required")
    root = output / args.run_id; private_directory(root); private_directory(root / "cases")
    write_private_json(root / "start_manifest.json", {"schema_version": VERSION, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "automatic_resume": False, "clinical_acceptance": False})
    started = time.monotonic(); triples = []; books = []; slots = []; comparisons = []; rows = []
    try:
        for case in plan["cases"]:
            produced, book, _ = one_case(case, plan, root)
            triples.extend(produced); books.append(book); slots.extend(slot_outcomes(case, produced, book))
            for triple in produced:
                row = existing.candidate_row(triple); rows.append(row)
                comparisons.append({"case_id": case["case_id"], "cxr_model_id": row["cxr_model_id"], "seed": row["seed"],
                    "baseline_row": case["baseline_row"], "new_row": row,
                    "frozen_proxy_preservation_comparison": compare(case["baseline_row"], row),
                    "diagnostic_only_no_selection": True, "clinical_acceptance": False})
        load_plan(args)
        summary = {"schema_version": VERSION, "status": "completed_candidate_expansion_unvalidated" if len(triples) == 4
            else "partial_candidate_expansion_unvalidated", "fixed_ehr_cases": 2, "planned_image_slots": 4,
            "completed_triplets": len(triples), "slot_status_counts": dict(Counter(s["status"] for s in slots)),
            "charged_worker_requests": sum(b["charged_model_attempts"] for b in books),
            "validated_worker_calls": sum(b["completed_operations"] for b in books),
            "failed_worker_requests": sum(b["failed_attempts"] for b in books), "retries": 0,
            "runtime_seconds_includes_startup_and_io": round(time.monotonic()-started, 3),
            "new_planner_calls": 0, "external_api_calls": 0, "training_performed": False,
            "clinical_acceptance": False, "clinical_accuracy": None, "selection_changed": False,
            "new_clinically_accepted_repairs": 0, "measured_saved_model_calls": None,
            "same_call_budget_not_same_gpu_seconds": True, "independent_endpoint_evaluated": False,
            "historical_cost_is_shared_sunk_not_free": True}
        files = {"summary.json": summary, "completed_triplets.json": {"records": triples},
            "slot_outcomes.json": {"records": slots}, "execution_summary.json": {"case_ledgers": books},
            "score_rows.json": {"records": rows}, "paired_proxy_comparison.json": {"records": comparisons}}
        for name, payload in files.items(): write_private_json(root / name, payload)
        write_private_text(root / "score_table.csv", score_csv(triples))
        artifacts = {name: {"sha256": sha256_file(root / name)} for name in (*files, "score_table.csv", "start_manifest.json")}
        write_private_json(root / "manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "plan_manifest_sha256": args.plan_manifest_sha256, "artifacts": artifacts,
            "clinical_acceptance": False, "selection_changed": False})
    except BaseException:
        write_private_json(root / "failure.json", {"schema_version": VERSION,
            "status": "failed_closed_charged_journals_retained", "automatic_resume": False,
            "clinical_acceptance": False, "selection_changed": False})
        raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-root", type=Path, required=True)
        sub.add_argument("--run-id", required=True)
        if name == "run":
            sub.add_argument("--plan-run", type=Path, required=True)
            sub.add_argument("--plan-manifest-sha256", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            root = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(root / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
