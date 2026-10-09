#!/usr/bin/env python3
"""Separate frozen BioViL endpoint for the audited fresh report bridge.

CPU prepare seals fixed/static/LLM choices before scores exist. GPU run calls
the unchanged existing BioViL scorer; never regenerates or changes a selection.
Shared-pool static replay is not a cost-matched online baseline or clinical truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

import audit_fresh_report_agent as postflight
from contracts import (PROTECTED_ROOT, commit_atomic_run, new_atomic_run,
    discard_atomic_run, private_directory, require_inside, read_json,
    sha256_file, write_private_json, write_private_text)
import fresh_output_acceptance as gate
from prepare_fixed_image_reports import BIOVIL_MODEL, BIOVIL_PYTHON
from score_automatic_replay_biovil import REQUEST_SCHEMA, MODEL_HASHES, PAIR_FIELDS, score
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.full_pool_report_control import validate_endpoint
from tricompose_v12.live_workers import check_pins
from tricompose_v12.report_expert_control import MODELS, compare
from tricompose_llm.local_qwen import gpu_guard

VERSION = "tricompose-fresh-report-independent-endpoint-v1"
ROOT = postflight.ROOT
SOURCE = PROTECTED_ROOT / "tricompose_v1_2/llm_fresh_report_runs/llm_fresh2_12817356"
SOURCE_SHA = "39f74f4abd00e15c5fece64c7545ddb97aa9632d3058af9c6d8b80539b354096"
AUDIT = PROTECTED_ROOT / "tricompose_v1_2/llm_fresh_report_audits/fresh_report2_12817356_12817261_001"
AUDIT_SHA = "0c08af97545c869f57b56a38140d3a8ff7181e1badb925fba532540c16c33151"
VENDOR = ROOT / "runtime/vendor/hi_ml_multimodal_0_2_2"
require = postflight.require


def freeze_controls(rows, selections, contexts, books, *, expected_cases=2):
    """Use the previously defined first-preserving expert order, not BioViL."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    require(len(groups) == expected_cases and len(rows) == expected_cases * 4
        and len({r["triple_candidate_id"] for r in rows}) == len(rows)
        and set(groups) == set(contexts) == set(books)
        and len(selections) == expected_cases
        and {s["case_id"] for s in selections} == set(groups), "complete_audited_four_expert_grid_required")
    choices, comparisons = [], []
    for selected in selections:
        cid = selected["case_id"]
        group, ctx, book = groups[cid], contexts[cid], books[cid]
        require(len(group) == 4 and {r["report_model_id"] for r in group} == set(MODELS),
            "one_output_per_predeclared_report_expert_required")
        for row in group:
            gate.validate_observation(gate.controller_view(row), ctx)
            gate.completed_operation(row, book)
        indexed = {r["report_model_id"]: r for r in group}
        base = indexed[MODELS[0]]
        require(selected["baseline_candidate_id"] == base["triple_candidate_id"]
            and selected["selected_candidate_id"] in {r["triple_candidate_id"] for r in group}
            and selected["clinical_repair_success"] is False,
            "sealed_baseline_and_selected_candidate_required")
        inventory = [{"row": gate.controller_view(r), "origin": {"kind": "completed_ledger_receipt"}}
            for r in sorted(group, key=lambda r: gate.completed_operation(r, book))]
        eligible = []
        for model in MODELS[1:]:
            row = indexed[model]
            receipt = gate.assess_fresh_output(base["triple_candidate_id"], row["triple_candidate_id"],
                inventory, ctx, ledger_snapshot=book)
            transition = compare(base, row)
            passed = receipt["proposal_passes_proxy_preservation"] and transition["exploratory_gate_pass"]
            comparisons.append({"case_id": cid, "report_model": model, "baseline_veto": receipt,
                "current_baseline_comparison": transition, "eligible": bool(passed)})
            if passed:
                eligible.append(row["triple_candidate_id"])
        choices.append({"case_id": cid, "fixed_candidate_id": base["triple_candidate_id"],
            "static_candidate_id": eligible[0] if eligible else base["triple_candidate_id"],
            "llm_candidate_id": selected["selected_candidate_id"],
            "eligible_static_candidate_ids": eligible,
            "source_worker_attempts": selected["charged_worker_attempts"],
            "source_policy_requests": selected["charged_policy_requests"]})
    return {"schema_version": VERSION, "choices": choices, "comparisons": comparisons,
        "static_policy": "existing_model_order_first_strict_proxy_preserving_expert",
        "static_expert_order": list(MODELS), "candidate_rows_sha256": _digest(rows),
        "endpoint_used_for_selection": False, "clinical_accuracy": None,
        "static_scope": "offline_shared_observed_pool_not_cost_matched_online_policy",
        "original_selection_changed": False, "new_generator_calls": 0}


def csv_text(records):
    fields = list(records[0]) if records else ["case_id"]
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
    for row in records:
        writer.writerow({k: "NA" if v is None else v for k, v in row.items()})
    return out.getvalue()


def endpoint_tables(rows, controls, endpoint):
    scores = validate_endpoint(rows, endpoint)
    require(controls["candidate_rows_sha256"] == _digest(rows)
        and controls["endpoint_used_for_selection"] is False, "presealed_choices_required")
    index = {r["triple_candidate_id"]: r for r in rows}
    pairs = []
    for choice in controls["choices"]:
        record = {"case_id": choice["case_id"]}
        for arm in ("fixed", "static", "llm"):
            candidate = choice[arm + "_candidate_id"]
            require(index[candidate]["case_id"] == choice["case_id"], "paired_case_scope_changed")
            record[arm + "_report_model"] = index[candidate]["report_model_id"]
            record[arm + "_raw_cosine"] = scores[candidate]["biovil_raw_cosine"]
            record[arm + "_availability_reason"] = scores[candidate]["reason"]
        for arm in ("static", "llm"):
            a, b = record["fixed_raw_cosine"], record[arm + "_raw_cosine"]
            record[arm + "_minus_fixed"] = b - a if a is not None and b is not None else None
        record["llm_equals_static_candidate"] = choice["llm_candidate_id"] == choice["static_candidate_id"]
        record["source_worker_attempts"] = choice["source_worker_attempts"]
        record["source_policy_requests"] = choice["source_policy_requests"]
        pairs.append(record)
    table = [{"case_id": row["case_id"], "triple_candidate_id": row["triple_candidate_id"],
        "report_model": row["report_model_id"], "biovil_raw_cosine": scores[row["triple_candidate_id"]]["biovil_raw_cosine"],
        "status": scores[row["triple_candidate_id"]]["status"], "reason": scores[row["triple_candidate_id"]]["reason"]}
        for row in rows]
    summary = {"case_count": len(pairs), "requested_report_pairs": len(rows),
        "available_pairs": sum(s["biovil_raw_cosine"] is not None for s in scores.values()),
        "llm_static_same_candidate_count": sum(p["llm_equals_static_candidate"] for p in pairs),
        "endpoint_used_for_selection": False, "clinical_accuracy": None,
        "clinical_repair_success": False, "no_clinical_or_cost_superiority_claim": True}
    return table, pairs, summary


def prepare(args):
    gate.cpu_guard()
    ar, sr = postflight.MetadataReader(AUDIT), postflight.MetadataReader(SOURCE)
    ar.hash(AUDIT / "manifest.json", AUDIT_SHA)
    am = ar.json(AUDIT / "manifest.json"); ar.hash(AUDIT / "audit.json", am["audit_sha256"])
    audit = ar.json(AUDIT / "audit.json")
    require(audit["schema_version"] == postflight.VERSION
        and audit["status"] == "completed_metadata_contract_audit"
        and audit["source_manifest_v2_sha256"] == SOURCE_SHA
        and audit["metadata_and_byte_hash_checks_pass"] is True, "exact_completed_audit_required")
    check_pins(audit["source_pins"]); check_pins(audit["artifact_pins"])
    sr.hash(SOURCE / "manifest_v2.json", SOURCE_SHA)
    rows = sr.json(SOURCE / "score_rows.json")["records"]
    selections = sr.json(SOURCE / "selection.json")["records"]
    books = {b["case_id"]: b for b in sr.json(SOURCE / "execution_summary.json")["case_ledgers"]}
    contexts = {}
    for r in audit["case_results"]:
        cid = r["selection"]["case_id"]
        # Reuse the exact reviewed plan metadata from the authenticated audit.
        require(r["ledger_replay_pass"] and r["policy_feedback_replay_pass"], "complete_live_replay_required")
        contexts[cid] = None
    manifest_path = next(Path(p) for p in audit["artifact_pins"]
        if Path(p).name == "manifest.json" and "/llm_fresh_report_plans/" in p)
    pr = postflight.MetadataReader(manifest_path.parent)
    plan = pr.json(manifest_path.parent / "plan.json")
    require(set(contexts) == {c["case_id"] for c in plan["cases"]}, "fixed_plan_case_scope_changed")
    contexts = {c["case_id"]: gate.context(c["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
        for c in plan["cases"]}
    controls = freeze_controls(rows, selections, contexts, books)
    triples = sr.json(SOURCE / "completed_triplets.json")["records"]
    require(len(triples) == 8 and len(rows) == 8 and len({r["cxr_sha256"] for r in rows}) == 2,
        "exact_completed_two_image_eight_report_source_required")
    model_pins = {}
    for name, expected in MODEL_HASHES.items():
        p = BIOVIL_MODEL / name
        require(sha256_file(p) == expected, "existing_frozen_biovil_weights_changed")
        model_pins[str(p)] = {"sha256": expected, "size_bytes": p.stat().st_size}
    for p in sorted(BIOVIL_MODEL.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".txt", ".py"):
            model_pins[str(p)] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
    vendor_files = sorted(VENDOR.rglob("*.py"))
    require(vendor_files and os.access(BIOVIL_PYTHON, os.X_OK), "deployed_biovil_environment_required")
    sources = {**audit["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
        str(ROOT / "TriCompose-v1.2/tests/test_fresh_report_secondary.py"):
            sha256_file(ROOT / "TriCompose-v1.2/tests/test_fresh_report_secondary.py")}
    for p in vendor_files:
        sources[str(p)] = sha256_file(p)
    check_pins(sources); ar.recheck(); sr.recheck(); pr.recheck()
    request = {"schema_version": REQUEST_SCHEMA, "pairs": [{k: r[k] for k in PAIR_FIELDS} for r in rows],
        "modality_source": "fully_synthetic", "selection_used_biovil": False,
        "clinical_truth_available": False, "routing_or_calibration_update_allowed": False,
        "text_policy": "full_report_no_silent_truncation_overlength_is_na",
        "selection_sha256_sealed_before_endpoint": sr.hash(SOURCE / "selection.json")}
    payload = {"schema_version": VERSION, "source_run": str(SOURCE), "source_manifest_sha256": SOURCE_SHA,
        "source_audit_manifest_sha256": AUDIT_SHA, "controls": controls, "rows": rows,
        "source_pins": sources, "artifact_pins": {**audit["artifact_pins"], **ar.pins, **sr.pins, **pr.pins},
        "model_pins": model_pins, "model_path": str(BIOVIL_MODEL), "python": str(BIOVIL_PYTHON),
        "vendor_path": str(VENDOR),
        "cxr_runs": sorted({str(require_inside(t["cxr_run"], SOURCE, must_exist=True)) for t in triples}),
        "report_runs": sorted({str(require_inside(t["report_run"], SOURCE, must_exist=True)) for t in triples}),
        "maximum_image_encodings": 2, "maximum_text_encodings": 8,
        "new_generator_calls": 0, "new_primary_scoring_calls": 0, "source_bodies_parsed": False,
        "endpoint_used_for_selection": False, "factory_instantiated": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        private_directory(temporary / "endpoint_request")
        rp = write_private_json(temporary / "endpoint_request/request.json", request)
        rm = write_private_json(temporary / "endpoint_request/manifest.json", {"schema_version": REQUEST_SCHEMA,
            "artifacts": {"request.json": {"sha256": sha256_file(rp)}}})
        payload["artifact_pins"].update({str(target / "endpoint_request/request.json"): sha256_file(rp),
            str(target / "endpoint_request/manifest.json"): sha256_file(rm)})
        p = write_private_json(temporary / "plan.json", payload)
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(p),
            "new_model_calls": 0, "planned_report_pairs": 8, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def run(args):
    gpu_guard()  # Actual GPU cgroup, before model imports, inputs or directories.
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    require(sha256_file(root / "manifest.json") == args.plan_manifest_sha256, "reviewed_endpoint_plan_changed")
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == VERSION and sha256_file(root / "plan.json") == manifest["plan_sha256"],
        "sealed_endpoint_plan_required")
    plan = read_json(root / "plan.json")
    require(plan["source_manifest_sha256"] == SOURCE_SHA and plan["source_run"] == str(SOURCE)
        and plan["source_audit_manifest_sha256"] == AUDIT_SHA and plan["vendor_path"] == str(VENDOR)
        and plan["maximum_image_encodings"] == 2 and plan["maximum_text_encodings"] == 8
        and plan["new_generator_calls"] == plan["new_primary_scoring_calls"] == 0
        and plan["endpoint_used_for_selection"] is False and plan["source_bodies_parsed"] is False,
        "frozen_secondary_only_scope_required")
    for field in ("source_pins", "artifact_pins", "model_pins"):
        check_pins(plan[field])
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic()
    try:
        write_private_json(temporary / "attempt_reserved.json", {"schema_version": VERSION,
            "charged_endpoint_worker_attempts": 1, "maximum_image_encodings": 2,
            "maximum_text_encodings": 8, "status": "reserved_before_load_no_free_failed_work"})
        sys.path.insert(0, plan["vendor_path"])
        endpoint = score(argparse.Namespace(request_run=root / "endpoint_request", model_path=plan["model_path"],
            cxr_run=plan["cxr_runs"], report_run=plan["report_runs"]))
        endpoint["historical_pool_is_untouched_test"] = False
        require(endpoint["producer"]["frozen"] is True
            and endpoint["producer"]["checkpoint_sha256"] == MODEL_HASHES
            and endpoint["counts"]["requested_pairs"] == 8
            and endpoint["counts"]["image_encoder_calls"] <= 2
            and endpoint["counts"]["text_encoder_calls"] <= 8, "endpoint_scope_or_encoding_budget_changed")
        table, pairs, summary = endpoint_tables(plan["rows"], plan["controls"], endpoint)
        files = [write_private_json(temporary / "scores.json", endpoint),
            write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "score_table.csv", csv_text(table)),
            write_private_text(temporary / "case_comparison.csv", csv_text(pairs))]
        for field in ("source_pins", "artifact_pins", "model_pins"):
            check_pins(plan[field])
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "completed_independent_endpoint_unvalidated", "plan_manifest_sha256": args.plan_manifest_sha256,
            "source_manifest_sha256": SOURCE_SHA, "charged_endpoint_worker_attempts": 1,
            "encoding_counts": endpoint["counts"], "peak_vram_gib": endpoint["peak_vram_gib"],
            "runtime_seconds_including_load_io": round(time.monotonic()-started, 3),
            "new_generator_calls": 0, "endpoint_used_for_selection": False, "clinical_acceptance": False,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files}})
        commit_atomic_run(temporary, target)
    except BaseException:
        write_private_json(temporary / "failure.json", {"schema_version": VERSION,
            "status": "failed_endpoint_attempt_retained", "charged_endpoint_worker_attempts": 1,
            "completed_encoding_counts": None, "automatic_resume": False})
        commit_atomic_run(temporary, target)
        raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); execution = sub.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for command in (prep, execution):
        command.add_argument("--output-root", type=Path, required=True); command.add_argument("--run-id", required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        target = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
