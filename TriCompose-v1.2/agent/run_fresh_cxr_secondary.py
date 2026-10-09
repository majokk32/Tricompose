#!/usr/bin/env python3
"""Small, separate BioViL-T endpoint for the completed one-image probe.

Prepare reads authenticated metadata and byte hashes only. Run uses the
unchanged frozen scorer on existing synthetic artifacts, without regeneration,
primary scoring, selection changes, calibration or training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import audit_fresh_cxr_agent as postflight
import run_fresh_report_secondary as previous
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, private_directory, read_json, require_inside, sha256_file,
    write_private_json, write_private_text)
from tricompose_v12.full_pool_report_control import validate_endpoint
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_workers import check_pins
from tricompose_llm.local_qwen import gpu_guard

VERSION = "tricompose-fresh-cxr-independent-endpoint-v1"
ROOT = previous.ROOT
BASE = PROTECTED_ROOT / "tricompose_v1_2"
SOURCE = BASE / "llm_fresh_cxr_runs/llm_cxr2_12834413"
SOURCE_SHA = "0140a1c8644ec1d1c4f92529f88117ed7267c1cc85211652544ba4bee6ec8a8b"
AUDIT = BASE / "llm_fresh_cxr_audits/fresh_cxr2_12834413_12817261_001"
AUDIT_SHA = "c77ef7951a3eb2979f89df27200a77816b947a6e08d90ee53ab0c579c7d12fe4"
GENERATION_PLAN = BASE / "llm_fresh_cxr_plans/fresh_cxr2_12817261_001"
GENERATION_PLAN_SHA = "807f136342fd0c837962b3b860462df8e08df6a197a0899b6de3b7e5682f1163"
require = previous.require
ARMS = ("fixed", "static", "before", "selected", "probe")


def freeze_controls(rows, generation_plan, selections, books):
    """Preserve choices made before this endpoint, including rejected probes."""
    cases = generation_plan["cases"]
    cached = [r for c in cases for r in c["cached_rows"]]
    ids = {c["case_id"] for c in cases}
    index = {r["triple_candidate_id"]: r for r in rows}
    old = {c["case_id"]: c for c in generation_plan["cached_controls"]["choices"]}
    require(len(ids) == 2 and len(cached) == 8 and rows[:8] == cached
        and 8 <= len(rows) <= 10 and len(index) == len(rows)
        and {r["case_id"] for r in rows} == ids == set(old)
        and len(selections) == len(books) == 2
        and {s["case_id"] for s in selections} == {b["case_id"] for b in books} == ids,
        "complete_audited_two_case_inventory_required")
    fresh = rows[8:]
    choices = []
    for case in cases:
        cid = case["case_id"]
        selected = next(s for s in selections if s["case_id"] == cid)
        book = next(b for b in books if b["case_id"] == cid)
        probes = [r for r in fresh if r["case_id"] == cid]
        before = case["reference"]["triple_candidate_id"]
        require(len(probes) <= 1 and before == old[cid]["llm_candidate_id"]
            and selected["clinical_repair_success"] is False
            and selected["charged_new_worker_attempts"] == book["charged_model_attempts"]
            and book["pending_attempts"] == 0, "unchanged_reference_and_charged_probe_required")
        choice = {"case_id": cid,
            "fixed_candidate_id": old[cid]["fixed_candidate_id"],
            "static_candidate_id": old[cid]["static_candidate_id"],
            "before_candidate_id": before,
            "selected_candidate_id": selected["selected_candidate_id"],
            "probe_candidate_id": probes[0]["triple_candidate_id"] if probes else None,
            "accepted_proxy_transitions": selected["accepted_proxy_transitions"],
            "charged_new_worker_attempts": selected["charged_new_worker_attempts"],
            "charged_policy_requests": selected["charged_policy_requests"]}
        for arm in ARMS:
            candidate = choice[arm + "_candidate_id"]
            require(candidate is None or candidate in index and index[candidate]["case_id"] == cid,
                "arm_must_be_observed_on_same_case")
        reference = index[before]
        for row in [r for r in rows if r["case_id"] == cid]:
            require(all(row[k] == reference[k] for k in ("ehr_sha256", "ehr_facts_sha256")),
                "fixed_ehr_anchor_required")
        for row in probes:
            require(row["report_model_id"] == reference["report_model_id"]
                and row["cxr_model_id"] == "roentgen_v2" and row["seed"] == 1,
                "seed_only_image_probe_and_retained_expert_required")
        choices.append(choice)
    return {"schema_version": VERSION, "choices": choices,
        "candidate_rows_sha256": _digest(rows),
        "new_candidate_ids": [r["triple_candidate_id"] for r in fresh],
        "endpoint_used_for_selection": False, "original_selection_changed": False,
        "static_scope": "historical_shared_pool_not_cost_matched_online_control"}


def endpoint_tables(rows, controls, endpoint):
    scores = validate_endpoint(rows, endpoint)
    require(controls["schema_version"] == VERSION
        and controls["candidate_rows_sha256"] == _digest(rows)
        and controls["endpoint_used_for_selection"] is False
        and controls["original_selection_changed"] is False, "presealed_choices_required")
    index = {r["triple_candidate_id"]: r for r in rows}
    pairs = []
    for choice in controls["choices"]:
        record = {"case_id": choice["case_id"]}
        for arm in ARMS:
            candidate = choice[arm + "_candidate_id"]
            require(candidate is None or candidate in index and index[candidate]["case_id"] == choice["case_id"],
                "paired_case_scope_changed")
            row = index[candidate] if candidate is not None else None
            result = scores[candidate] if candidate is not None else None
            record[arm + "_report_model"] = row["report_model_id"] if row else None
            record[arm + "_raw_cosine"] = result["biovil_raw_cosine"] if result else None
            record[arm + "_availability_reason"] = result["reason"] if result else "no_completed_candidate"
        for arm in ("selected", "probe"):
            before, after = record["before_raw_cosine"], record[arm + "_raw_cosine"]
            record[arm + "_minus_before"] = after - before if before is not None and after is not None else None
        record["selected_equals_before"] = choice["selected_candidate_id"] == choice["before_candidate_id"]
        for key in ("accepted_proxy_transitions", "charged_new_worker_attempts", "charged_policy_requests"):
            record[key] = choice[key]
        pairs.append(record)
    fresh = set(controls["new_candidate_ids"])
    table = [{"case_id": r["case_id"], "triple_candidate_id": r["triple_candidate_id"],
        "origin": "new_image_probe" if r["triple_candidate_id"] in fresh else "historical_cached",
        "cxr_model": r["cxr_model_id"], "report_model": r["report_model_id"], "seed": r["seed"],
        **{k: scores[r["triple_candidate_id"]][k] for k in ("biovil_raw_cosine", "status", "reason")}}
        for r in rows]
    summary = {"case_count": len(pairs), "requested_report_pairs": len(rows),
        "new_probe_pairs": len(fresh), "available_pairs": sum(s["biovil_raw_cosine"] is not None for s in scores.values()),
        "accepted_proxy_transitions": sum(c["accepted_proxy_transitions"] for c in controls["choices"]),
        "new_generator_calls": 0, "new_primary_scoring_calls": 0,
        "endpoint_used_for_selection": False, "original_selection_changed": False,
        "clinical_accuracy": None, "clinical_repair_success": False,
        "llm_superiority_demonstrated": False, "training_performed": False}
    return table, pairs, summary


def prepare(args):
    previous.gate.cpu_guard()
    readers = [previous.postflight.MetadataReader(p) for p in (AUDIT, SOURCE, GENERATION_PLAN)]
    ar, sr, pr = readers
    ar.hash(AUDIT / "manifest.json", AUDIT_SHA)
    am = ar.json(AUDIT / "manifest.json")
    ar.hash(AUDIT / "audit.json", am["artifacts"]["audit.json"]["sha256"])
    audit = ar.json(AUDIT / "audit.json")
    require(audit["schema_version"] == postflight.VERSION
        and audit["status"] == "metadata_and_ledger_verified_not_clinical"
        and audit["source_manifest_sha256"] == SOURCE_SHA
        and audit["plan_manifest_sha256"] == GENERATION_PLAN_SHA
        and audit["new_model_calls"] == 0
        and all(r["ledger_replay_pass"] and r["policy_feedback_replay_pass"] for r in audit["case_results"]),
        "exact_completed_metadata_audit_required")
    check_pins(audit["source_pins"]); check_pins(audit["artifact_pins"])
    sr.hash(SOURCE / "manifest.json", SOURCE_SHA)
    sm = sr.json(SOURCE / "manifest.json")
    require(sm["status"] == "completed_one_image_probe_unvalidated"
        and sm["plan_manifest_sha256"] == GENERATION_PLAN_SHA, "completed_source_required")
    for name, pin in sm["artifacts"].items():
        sr.hash(SOURCE / name, pin["sha256"])
    pr.hash(GENERATION_PLAN / "manifest.json", GENERATION_PLAN_SHA)
    pm = pr.json(GENERATION_PLAN / "manifest.json")
    pr.hash(GENERATION_PLAN / "plan.json", pm["plan_sha256"])
    generation = pr.json(GENERATION_PLAN / "plan.json")
    rows = sr.json(SOURCE / "score_rows.json")["records"]
    selections = sr.json(SOURCE / "selection.json")["records"]
    books = sr.json(SOURCE / "execution_summary.json")["case_ledgers"]
    controls = freeze_controls(rows, generation, selections, books)
    triples = sr.json(SOURCE / "completed_triplets.json")
    triples = triples["cached_references"] + triples["new_records"]
    require(len(triples) == len(rows), "complete_existing_artifact_inventory_required")
    model_pins = {}
    for name, expected in previous.MODEL_HASHES.items():
        p = previous.BIOVIL_MODEL / name
        require(sha256_file(p) == expected, "existing_frozen_biovil_weights_changed")
        model_pins[str(p)] = {"sha256": expected, "size_bytes": p.stat().st_size}
    for p in sorted(previous.BIOVIL_MODEL.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".txt", ".py"):
            model_pins[str(p)] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
    vendor_files = sorted(previous.VENDOR.rglob("*.py"))
    require(vendor_files and os.access(previous.BIOVIL_PYTHON, os.X_OK), "deployed_biovil_environment_required")
    sources = {**audit["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
        str(ROOT / "TriCompose-v1.2/tests/test_fresh_cxr_secondary.py"):
            sha256_file(ROOT / "TriCompose-v1.2/tests/test_fresh_cxr_secondary.py"),
        **{str(p): sha256_file(p) for p in vendor_files}}
    allowed = (SOURCE, previous.SOURCE)
    def existing_run(path):
        p = require_inside(path, PROTECTED_ROOT, must_exist=True)
        require(any(p.is_relative_to(root) for root in allowed), "only_audited_existing_synthetic_runs_allowed")
        return str(p)
    request = {"schema_version": previous.REQUEST_SCHEMA,
        "pairs": [{k: r[k] for k in sorted(previous.PAIR_FIELDS)} for r in rows],
        "modality_source": "fully_synthetic", "selection_used_biovil": False,
        "clinical_truth_available": False, "routing_or_calibration_update_allowed": False,
        "text_policy": "full_report_no_silent_truncation_overlength_is_na",
        "selection_sha256_sealed_before_endpoint": sr.hash(SOURCE / "selection.json")}
    payload = {"schema_version": VERSION, "source_run": str(SOURCE), "source_manifest_sha256": SOURCE_SHA,
        "source_audit_manifest_sha256": AUDIT_SHA, "generation_plan_manifest_sha256": GENERATION_PLAN_SHA,
        "controls": controls, "rows": rows, "source_pins": sources,
        "artifact_pins": {**audit["artifact_pins"], **ar.pins, **sr.pins, **pr.pins},
        "model_pins": model_pins, "model_path": str(previous.BIOVIL_MODEL),
        "python": str(previous.BIOVIL_PYTHON), "vendor_path": str(previous.VENDOR),
        "cxr_runs": sorted({existing_run(t["cxr_run"]) for t in triples}),
        "report_runs": sorted({existing_run(t["report_run"]) for t in triples}),
        "maximum_image_encodings": len({r["cxr_candidate_id"] for r in rows}),
        "maximum_text_encodings": len(rows), "new_generator_calls": 0, "new_primary_scoring_calls": 0,
        "source_bodies_parsed": False, "endpoint_used_for_selection": False, "factory_instantiated": False}
    require(payload["maximum_image_encodings"] <= 4, "bounded_four_image_endpoint_required")
    check_pins(sources)
    for reader in readers: reader.recheck()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        private_directory(temporary / "endpoint_request")
        rp = write_private_json(temporary / "endpoint_request/request.json", request)
        rm = write_private_json(temporary / "endpoint_request/manifest.json", {"schema_version": previous.REQUEST_SCHEMA,
            "artifacts": {"request.json": {"sha256": sha256_file(rp)}}})
        payload["artifact_pins"].update({str(target / "endpoint_request/request.json"): sha256_file(rp),
            str(target / "endpoint_request/manifest.json"): sha256_file(rm)})
        p = write_private_json(temporary / "plan.json", payload)
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(p),
            "new_model_calls": 0, "planned_report_pairs": len(rows), "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def run(args):
    gpu_guard()
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    require(sha256_file(root / "manifest.json") == args.plan_manifest_sha256, "reviewed_endpoint_plan_changed")
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only_gpu_not_submitted"
        and sha256_file(root / "plan.json") == manifest["plan_sha256"], "sealed_endpoint_plan_required")
    plan = read_json(root / "plan.json")
    require(plan["schema_version"] == VERSION and plan["source_run"] == str(SOURCE)
        and plan["source_manifest_sha256"] == SOURCE_SHA and plan["source_audit_manifest_sha256"] == AUDIT_SHA
        and plan["generation_plan_manifest_sha256"] == GENERATION_PLAN_SHA
        and plan["vendor_path"] == str(previous.VENDOR) and plan["model_path"] == str(previous.BIOVIL_MODEL)
        and plan["python"] == str(previous.BIOVIL_PYTHON)
        and 2 <= plan["maximum_image_encodings"] <= 4 and 8 <= plan["maximum_text_encodings"] <= 10
        and plan["maximum_text_encodings"] == len(plan["rows"])
        and plan["new_generator_calls"] == plan["new_primary_scoring_calls"] == 0
        and plan["endpoint_used_for_selection"] is False and plan["source_bodies_parsed"] is False,
        "frozen_secondary_only_scope_required")
    for field in ("source_pins", "artifact_pins", "model_pins"): check_pins(plan[field])
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic()
    try:
        write_private_json(temporary / "attempt_reserved.json", {"schema_version": VERSION,
            "charged_endpoint_worker_attempts": 1, "maximum_image_encodings": plan["maximum_image_encodings"],
            "maximum_text_encodings": plan["maximum_text_encodings"],
            "status": "reserved_before_load_no_free_failed_work"})
        sys.path.insert(0, plan["vendor_path"])
        endpoint = previous.score(argparse.Namespace(request_run=root / "endpoint_request", model_path=plan["model_path"],
            cxr_run=plan["cxr_runs"], report_run=plan["report_runs"]))
        endpoint["historical_pool_is_untouched_test"] = False
        require(endpoint["producer"]["frozen"] is True
            and endpoint["producer"]["checkpoint_sha256"] == previous.MODEL_HASHES
            and endpoint["counts"]["requested_pairs"] == len(plan["rows"])
            and endpoint["counts"]["image_encoder_calls"] <= plan["maximum_image_encodings"]
            and endpoint["counts"]["text_encoder_calls"] <= plan["maximum_text_encodings"],
            "endpoint_scope_or_encoding_budget_changed")
        table, pairs, summary = endpoint_tables(plan["rows"], plan["controls"], endpoint)
        files = [write_private_json(temporary / "scores.json", endpoint),
            write_private_json(temporary / "summary.json", summary),
            write_private_text(temporary / "score_table.csv", previous.csv_text(table)),
            write_private_text(temporary / "case_comparison.csv", previous.csv_text(pairs))]
        for field in ("source_pins", "artifact_pins", "model_pins"): check_pins(plan[field])
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "completed_independent_endpoint_unvalidated", "plan_manifest_sha256": args.plan_manifest_sha256,
            "source_manifest_sha256": SOURCE_SHA, "charged_endpoint_worker_attempts": 1,
            "encoding_counts": endpoint["counts"], "peak_vram_gib": endpoint["peak_vram_gib"],
            "runtime_seconds_including_load_io": round(time.monotonic() - started, 3),
            "new_generator_calls": 0, "endpoint_used_for_selection": False, "clinical_acceptance": False,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files}})
        commit_atomic_run(temporary, target)
    except BaseException:
        write_private_json(temporary / "failure.json", {"schema_version": VERSION,
            "status": "failed_endpoint_attempt_retained", "charged_endpoint_worker_attempts": 1,
            "completed_encoding_counts": None, "automatic_resume": False})
        commit_atomic_run(temporary, target); raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep, execution = sub.add_parser("prepare"), sub.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for command in (prep, execution):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
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
