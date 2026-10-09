#!/usr/bin/env python3
"""Versioned one-image action after the audited frozen report-only bridge.

CPU prepare reads numeric metadata and hashes bytes; no model factories.
GPU run requires separately reviewed Slurm and fixes EHR, text, checkpoints,
scorers, thresholds and the retained report expert. Only CXR seed changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import run_fresh_report_agent as report_bridge
import run_fresh_report_secondary as source
from contracts import (PROTECTED_ROOT, RUN_ID_PATTERN, new_atomic_run,
    commit_atomic_run, discard_atomic_run, private_directory, read_json,
    require_inside, sha256_file, write_private_json, write_private_text)
import fresh_output_acceptance as gate
from run_bounded_regeneration import candidate_row
from tricompose_v11.cxr_contracts import canonical_json_sha256, validate_cxr_request
from tricompose_v12.bounded_regeneration import alternate_request
from tricompose_v12.execution_ledger import BoundedCallLedger
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import one_case, score_csv
from tricompose_v12.live_receipts import PROFILE
from tricompose_v12.live_workers import check_pins
from tricompose_v12.runtime_dispatch import ProtectedJournal
from tricompose_llm.live_cxr_bridge import FreshCXRSession, VERSION
from tricompose_llm.local_qwen import gpu_guard

ROOT = report_bridge.ROOT
POLICY = {"schema_version": VERSION, "call_budget_per_case": 4,
    "max_retries_per_operation": 0, "timeout_seconds_per_worker_process": 150,
    "policy_timeout_seconds": 150, "maximum_policy_requests_per_case": 1,
    "cxr_model": "roentgen_v2", "seed": 1,
    "report_expert_rule": "same_as_previously_sealed_retained_report",
    "changes_ehr_or_prompt": False, "scorers_or_thresholds_changed": False,
    "selection_uses_independent_endpoint": False, "external_api_enabled": False,
    "clinical_acceptance": False, "training_allowed": False}
MAXIMUM = {"new_cxrs": 2, "new_reports": 2,
    "charged_generator_verifier_attempts": 8, "charged_policy_requests": 2}


def worker_plan(plan, case):
    """Derive only the presealed report expert; no change of worker/budget."""
    result = {"workers": plan["workers"], "policy": {
        **plan["policy"], "report_models": [case["reference"]["report_model_id"]]}}
    return result


def selected_triplets(rows, triples, selections, *, fresh_ids):
    """Export exact selected artifact pointers, without ranking or body reads."""
    index = {row["triple_candidate_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("unique_candidate_inventory_required")
    result = []
    fields = ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256",
              "report_sha256", "cxr_model_id", "report_model_id", "seed")
    for selection in selections:
        cid = selection["selected_candidate_id"]
        if cid is None:
            result.append({"case_id": selection["case_id"], "selected_candidate_id": None,
                "status": selection["status"], "artifact": None, "clinical_acceptance": False})
            continue
        row = index.get(cid)
        if row is None or row["case_id"] != selection["case_id"]:
            raise ValueError("selected_case_and_observed_candidate_required")
        matched = [t for t in triples if all(t[key] == row[key] for key in fields)]
        if len(matched) != 1:
            raise ValueError("unique_selected_triplet_lineage_required")
        result.append({"case_id": selection["case_id"], "selected_candidate_id": cid,
            "status": selection["status"], "origin": "fresh_prospective_output" if cid in fresh_ids
                else "authenticated_historical_reference", "artifact": copy.deepcopy(matched[0]),
            "clinical_acceptance": False})
    return result


def validate_case(case, workers):
    anchor = gate.anchor_from_record(case["anchor"])
    if case["case_id"] != anchor.case_id or case["ehr_anchor_sha256"] != anchor.sha256:
        raise ValueError("immutable_case_anchor_required")
    row = case["requests"][0]
    original = read_json(row["source_path"])
    validate_cxr_request(row["request"])
    if (len(case["requests"]) != 1 or sha256_file(row["source_path"]) != row["source_sha256"]
            or alternate_request(original) != row["request"]
            or canonical_json_sha256(row["request"]) != row["canonical_request_sha256"]
            or row["request"]["case_id"] != anchor.case_id
            or row["request"]["inputs"]["synthetic_ehr"]["sha256"] != anchor.ehr_sha256
            or row["request"]["inputs"]["ehr_facts"]["sha256"] != anchor.ehr_facts_sha256):
        raise ValueError("only_seed_may_change_in_original_request")
    ctx = gate.context(case["anchor"], workers["xrv"], workers["chexbert"])
    book = BoundedCallLedger(case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
        call_budget=4, max_retries=0, execution_mode="approved_slurm_backend", sink=lambda event: None).snapshot()
    session = FreshCXRSession(case["cached_rows"], case["reference"]["triple_candidate_id"],
        ctx, book, source_manifest_sha256=source.SOURCE_SHA, sink=lambda event: None)
    if (session.reference != gate.controller_view(case["reference"])
            or {r["report_model_id"] for r in case["cached_rows"]} != set(source.MODELS)
            or len(case["cached_rows"]) != 4):
        raise ValueError("all_observed_historical_experts_and_sealed_reference_required")


def validate_plan(plan):
    if (plan["schema_version"] != VERSION or plan["policy"] != POLICY
            or plan["profile"] != PROFILE or plan["planned_maximum"] != MAXIMUM
            or plan["source_manifest_sha256"] != source.SOURCE_SHA
            or plan["source_audit_manifest_sha256"] != source.AUDIT_SHA
            or plan["data_origin"] != "original_fully_synthetic_pool80"
            or plan["source_bodies_parsed"] is not False or plan["new_model_calls"] != 0
            or plan["minimum_gpu_vram_gib"] != 40 or len(plan["cases"]) != 2
            or len({c["case_id"] for c in plan["cases"]}) != 2):
        raise ValueError("exact_two_development_anchor_image_probe_plan_required")
    expected_workers = {"roentgen_v2", "xrv", "chexbert"} | {
        c["reference"]["report_model_id"] for c in plan["cases"]}
    if set(plan["workers"]) != expected_workers or plan["qwen"]["model_path"] != str(report_bridge.QWEN_MODEL):
        raise ValueError("only_presealed_frozen_workers_allowed")
    for name in ("source_pins", "artifact_pins"):
        check_pins(plan[name])
    check_pins(plan["qwen"]["asset_pins"])
    if (plan["qwen"]["python"] != str(report_bridge.QWEN_PYTHON)
            or plan["qwen"]["asset_pins"][str(report_bridge.QWEN_MODEL / "model.safetensors")]["sha256"]
            != report_bridge.EXPECTED_WEIGHT):
        raise ValueError("unchanged_pinned_local_qwen_required")
    for spec in plan["workers"].values():
        if spec["status"] != "preflighted" or spec["factory_instantiated"] is not False:
            raise ValueError("existing_frozen_preflighted_worker_required")
        check_pins(spec["asset_pins"])
    for case in plan["cases"]:
        validate_case(case, plan["workers"])


def prepare(args):
    gate.cpu_guard()
    ar = source.postflight.MetadataReader(source.AUDIT)
    sr = source.postflight.MetadataReader(source.SOURCE)
    ar.hash(source.AUDIT / "manifest.json", source.AUDIT_SHA)
    manifest = ar.json(source.AUDIT / "manifest.json")
    ar.hash(source.AUDIT / "audit.json", manifest["audit_sha256"])
    audit = ar.json(source.AUDIT / "audit.json")
    if (audit["schema_version"] != source.postflight.VERSION
            or audit["status"] != "completed_metadata_contract_audit"
            or audit["source_manifest_v2_sha256"] != source.SOURCE_SHA
            or audit["metadata_and_byte_hash_checks_pass"] is not True):
        raise ValueError("completed_source_postflight_required")
    check_pins(audit["source_pins"]); check_pins(audit["artifact_pins"])
    sr.hash(source.SOURCE / "manifest_v2.json", source.SOURCE_SHA)
    rows = sr.json(source.SOURCE / "score_rows.json")["records"]
    selections = sr.json(source.SOURCE / "selection.json")["records"]
    books = {b["case_id"]: b for b in sr.json(source.SOURCE / "execution_summary.json")["case_ledgers"]}
    paths = [Path(p) for p in audit["artifact_pins"]
             if Path(p).name == "manifest.json" and "/llm_fresh_report_plans/" in p]
    if len(paths) != 1:
        raise ValueError("unique_previously_reviewed_source_plan_required")
    pr = source.postflight.MetadataReader(paths[0].parent)
    pr.hash(paths[0], audit["artifact_pins"][str(paths[0])])
    old = pr.json(paths[0].parent / "plan.json")
    contexts = {c["case_id"]: gate.context(c["anchor"], old["workers"]["xrv"], old["workers"]["chexbert"])
                for c in old["cases"]}
    controls = source.freeze_controls(rows, selections, contexts, books)
    triples = sr.json(source.SOURCE / "completed_triplets.json")["records"]
    if len(triples) != 8 or len(old["cases"]) != 2:
        raise ValueError("unchanged_complete_historical_source_required")
    cases = []
    for parent in old["cases"]:
        cid = parent["case_id"]
        selection = next(s for s in selections if s["case_id"] == cid)
        group = sorted([r for r in rows if r["case_id"] == cid],
                       key=lambda r: gate.completed_operation(r, books[cid]))
        reference = next(r for r in group if r["triple_candidate_id"] == selection["selected_candidate_id"])
        row = copy.deepcopy(parent["requests"][0])
        row["request"] = alternate_request(row["request"])
        row["canonical_request_sha256"] = canonical_json_sha256(row["request"])
        cases.append({**{key: copy.deepcopy(parent[key]) for key in
            ("case_id", "anchor", "ehr_anchor_sha256", "opaque_source_index")},
            "requests": [row], "cached_rows": [gate.controller_view(r) for r in group],
            "reference": gate.controller_view(reference)})
    needed = {"roentgen_v2", "xrv", "chexbert"} | {c["reference"]["report_model_id"] for c in cases}
    sources = {**audit["source_pins"]}
    for p in (Path(__file__).resolve(), Path(source.__file__).resolve(),
              Path(source.postflight.__file__).resolve(),
              ROOT / "TriCompose-v1.2/agent/tricompose_llm/live_cxr_bridge.py",
              ROOT / "TriCompose-v1.2/tests/test_fresh_cxr_agent_bridge.py"):
        sources[str(p)] = sha256_file(p)
    plan = {"schema_version": VERSION, "profile": PROFILE, "policy": POLICY,
        "cases": cases, "workers": {m: copy.deepcopy(old["workers"][m]) for m in needed},
        "qwen": copy.deepcopy(old["qwen"]), "source_pins": sources,
        "artifact_pins": {**audit["artifact_pins"], **ar.pins, **sr.pins, **pr.pins},
        "source_manifest_sha256": source.SOURCE_SHA, "source_audit_manifest_sha256": source.AUDIT_SHA,
        "cached_controls": controls, "cached_triplets": triples,
        "historical_worker_attempts": sum(b["charged_model_attempts"] for b in books.values()),
        "historical_policy_requests": sum(s["charged_policy_requests"] for s in selections),
        "historical_cost_scope": "shared_sunk_not_measured_not_zero",
        "data_origin": "original_fully_synthetic_pool80", "source_bodies_parsed": False,
        "new_model_calls": 0, "minimum_gpu_vram_gib": 40, "planned_maximum": MAXIMUM}
    validate_plan(plan); ar.recheck(); sr.recheck(); pr.recheck()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temporary / "plan.json", plan)
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(p),
            "planned_maximum": MAXIMUM, "new_model_calls": 0, "clinical_acceptance": False})
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
    if (manifest["schema_version"] != VERSION
            or manifest["status"] != "prepared_cpu_only_gpu_not_submitted"
            or sha256_file(root / "plan.json") != manifest["plan_sha256"]):
        raise ValueError("sealed_cpu_plan_required")
    plan = read_json(root / "plan.json")
    validate_plan(plan)
    return plan


def run(args):
    gpu_guard()
    plan = load_plan(args)
    import torch
    if torch.cuda.get_device_properties(0).total_memory < plan["minimum_gpu_vram_gib"] * 1024 ** 3:
        raise ValueError("allocated_gpu_below_pinned_worker_planning_memory")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("opaque_run_id_required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = output / args.run_id
    private_directory(root); private_directory(root / "cases")
    started = time.monotonic()
    write_private_json(root / "start_manifest.json", {"schema_version": VERSION,
        "status": "in_progress", "plan_manifest_sha256": args.plan_manifest_sha256,
        "automatic_resume": False})
    selections, fresh_rows, fresh_triples, books, audits = [], [], [], [], []
    try:
        for case in plan["cases"]:
            cid = case["case_id"]
            ctx = gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
            empty = BoundedCallLedger(case_id=cid, ehr_anchor_sha256=case["ehr_anchor_sha256"],
                call_budget=4, max_retries=0, execution_mode="approved_slurm_backend", sink=lambda event: None).snapshot()
            # one_case owns root/cases/<cid>; policy artifacts live separately.
            policy_root = root / "policy" / cid
            private_directory(root / "policy", exist_ok=True); private_directory(policy_root)
            with ProtectedJournal(policy_root / "live_policy.journal.jsonl") as journal:
                session = FreshCXRSession(case["cached_rows"], case["reference"]["triple_candidate_id"],
                    ctx, empty, source_manifest_sha256=source.SOURCE_SHA, sink=journal.append)
                state = session.begin_planning()
                if state is not None:
                    try:
                        decision, audit = report_bridge.propose_in_subprocess(state, policy_root / "step_0", plan)
                    except Exception:
                        session.policy_failed()
                    else:
                        audits.append(audit)
                        if session.receive_decision(decision):
                            generated, book, _ = one_case(case, worker_plan(plan, case), root)
                            if len(generated) > 1:
                                raise ValueError("one_requested_image_chain_only")
                            row = candidate_row(generated[0]) if generated else None
                            session.finish_image(row, book)
                            if row is not None:
                                fresh_rows.append(row); fresh_triples.extend(generated)
                selection = session.result()
                p = write_private_json(policy_root / "sealed_selection.json", selection)
                journal.append({"event": "selection_sealed", "sha256": sha256_file(p)})
                selections.append(selection); books.append(session.ledger)
        rows = [r for case in plan["cases"] for r in case["cached_rows"]] + fresh_rows
        write_private_json(root / "selection.json", {"records": selections})
        write_private_json(root / "score_rows.json", {"records": rows})
        write_private_text(root / "score_table.csv", score_csv([
            {**r, "verification_status": r["receipt"]["verification_status"]} for r in rows]))
        write_private_json(root / "completed_triplets.json", {
            "cached_references": plan["cached_triplets"], "new_records": fresh_triples})
        write_private_json(root / "selected_triplets.json", {"records": selected_triplets(
            rows, plan["cached_triplets"] + fresh_triples, selections,
            fresh_ids={r["triple_candidate_id"] for r in fresh_rows})})
        write_private_json(root / "execution_summary.json", {"case_ledgers": books})
        write_private_json(root / "policy_audits.json", {"records": audits})
        summary = {"schema_version": VERSION, "status": "completed_one_image_probe_unvalidated",
            "fixed_ehr_cases": len(selections), "completed_new_triplets": len(fresh_triples),
            "accepted_proxy_transitions": sum(s["accepted_proxy_transitions"] for s in selections),
            "charged_new_worker_attempts": sum(b["charged_model_attempts"] for b in books),
            "charged_policy_requests": sum(s["charged_policy_requests"] for s in selections),
            "historical_worker_attempts": plan["historical_worker_attempts"],
            "historical_policy_requests": plan["historical_policy_requests"],
            "historical_cost_scope": plan["historical_cost_scope"], "clinical_accuracy": None,
            "clinical_repair_success": False, "llm_superiority_demonstrated": False,
            "runtime_seconds_including_load_io": round(time.monotonic() - started, 3),
            "measured_gpu_seconds": None, "external_api_calls": 0,
            "scorers_or_thresholds_changed": False, "training_performed": False,
            "independent_endpoint_used_for_selection": False}
        write_private_json(root / "summary.json", summary)
        validate_plan(plan)
        files = ("selection.json", "score_rows.json", "score_table.csv", "completed_triplets.json", "selected_triplets.json",
                 "execution_summary.json", "policy_audits.json", "summary.json")
        write_private_json(root / "manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "plan_manifest_sha256": args.plan_manifest_sha256,
            "source_manifest_sha256": source.SOURCE_SHA,
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
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--run-id", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan-run", type=Path, required=True)
    r.add_argument("--plan-manifest-sha256", required=True)
    r.add_argument("--output-root", type=Path, required=True); r.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
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
