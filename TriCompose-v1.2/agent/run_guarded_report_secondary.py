#!/usr/bin/env python3
"""Frozen secondary-only BioViL-T for the two executed report requests.

CPU prepare authenticates metadata/byte hashes only. GPU run reuses the
existing scorer for four reports on two unchanged synthetic images. No new
planner/generator/primary scorer, training, threshold update or winner change.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import run_guarded_requested_reports as source
import run_fresh_report_secondary as previous
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, private_directory, read_json, require_inside, sha256_file,
    write_private_json, write_private_text)
from tricompose_v12.full_pool_report_control import validate_endpoint
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_workers import check_pins
from tricompose_llm.local_qwen import gpu_guard

VERSION = "tricompose-guarded-requested-report-secondary-v1"
BASE = source.BASE
SOURCE = BASE / "guarded_report_execution_runs/guarded_report2_12850244"
SOURCE_SHA = "677482fd514cd4d788b5e6904ead00cc309c9388b98c1d39e2dd8bef079ab6e8"
AUDIT = BASE / "guarded_report_execution_audits/guarded_report2_12850244_12827441_001"
AUDIT_SHA = "ddf8f0f69092f38a4e0eb6b42a27fff11199f07ca1ce664844532bd7185be8bd"
SOURCE_PLAN = BASE / "guarded_report_execution_plans/guarded_report2_12827441_001"
SOURCE_PLAN_SHA = "9eee28509cd594cc5449d3e643b2194ed23da734a0109317a741682b20274f17"
require = source.require


def freeze_controls(rows, pairs, *, expected_cases=2):
    """Seal actual same-image choices, including alternatives vetoed by the gate."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    index = {r["triple_candidate_id"]: r for r in rows}
    require(type(expected_cases) is int and expected_cases > 0
        and len(rows) == 2 * expected_cases and len(index) == len(rows)
        and len(groups) == expected_cases and len(pairs) == expected_cases
        and {p["case_id"] for p in pairs} == set(groups), "complete_two_report_pairs_required")
    choices = []
    for pair in pairs:
        group = groups[pair["case_id"]]
        require(len(group) == 2, "exact_two_reports_per_existing_image_required")
        base, alt = (index[pair[k]] for k in ("baseline_candidate_id", "proposed_candidate_id"))
        require(base != alt and base in group and alt in group
            and all(base[k] == alt[k] for k in
                ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_candidate_id", "cxr_sha256", "seed"))
            and base["report_model_id"] == pair["baseline_model"]
            and alt["report_model_id"] == pair["requested_model"], "same_image_ehr_seed_and_expert_required")
        comparison = source.compare(base, alt)
        selected = alt if comparison["exploratory_gate_pass"] else base
        require(pair["comparison"] == comparison
            and pair["baseline_scores"] == base["raw_edge_readouts"]
            and pair["proposed_scores"] == alt["raw_edge_readouts"]
            and pair["branch_selected_candidate_id"] == selected["triple_candidate_id"]
            and pair["original_winner_changed"] is False and pair["clinical_acceptance"] is False
            and pair["clinical_repair_success"] is False, "unchanged_pre_endpoint_gate_selection_required")
        choices.append({"case_id": pair["case_id"], "baseline_candidate_id": base["triple_candidate_id"],
            "requested_candidate_id": alt["triple_candidate_id"],
            "selected_candidate_id": selected["triple_candidate_id"],
            "branch_accepted": comparison["exploratory_gate_pass"],
            "rejection_reasons": comparison["reasons"],
            "current_image_evidence_cited": pair["current_image_evidence_cited"]})
    return {"schema_version": VERSION, "candidate_rows_sha256": _digest(rows),
        "paired_comparisons_sha256": _digest(pairs), "choices": choices,
        "endpoint_used_for_selection": False, "original_selection_changed": False}


def endpoint_tables(rows, controls, endpoint):
    scores = validate_endpoint(rows, endpoint)
    require(controls["schema_version"] == VERSION and controls["candidate_rows_sha256"] == _digest(rows)
        and controls["endpoint_used_for_selection"] is False
        and controls["original_selection_changed"] is False, "presealed_choices_required")
    index = {r["triple_candidate_id"]: r for r in rows}
    pairs = []
    for choice in controls["choices"]:
        record = {"case_id": choice["case_id"], "branch_accepted": choice["branch_accepted"]}
        for role in ("baseline", "requested", "selected"):
            cid = choice[role + "_candidate_id"]
            require(cid in index and index[cid]["case_id"] == choice["case_id"], "paired_case_scope_changed")
            record[role + "_model"] = index[cid]["report_model_id"]
            record[role + "_raw_cosine"] = scores[cid]["biovil_raw_cosine"]
            record[role + "_availability_reason"] = scores[cid]["reason"]
        for role in ("requested", "selected"):
            a, b = record["baseline_raw_cosine"], record[role + "_raw_cosine"]
            record[role + "_minus_baseline"] = b - a if a is not None and b is not None else None
        pairs.append(record)
    table = [{"case_id": r["case_id"], "triple_candidate_id": r["triple_candidate_id"],
        "report_model": r["report_model_id"], **{k: scores[r["triple_candidate_id"]][k]
            for k in ("biovil_raw_cosine", "status", "reason")}} for r in rows]
    summary = {"case_count": len(pairs), "requested_report_pairs": len(rows),
        "available_pairs": sum(s["biovil_raw_cosine"] is not None for s in scores.values()),
        "proxy_accepted_branch_changes": sum(c["branch_accepted"] for c in controls["choices"]),
        "new_generator_calls": 0, "new_primary_scoring_calls": 0, "new_planner_calls": 0,
        "endpoint_used_for_selection": False, "original_selection_changed": False,
        "clinical_accuracy": None, "clinical_repair_success": False, "training_performed": False,
        "llm_superiority_demonstrated": False, "measured_saved_model_calls": None}
    return table, pairs, summary


def prepare(args):
    source.gate.cpu_guard()
    readers = [source.Reader(p) for p in (AUDIT, SOURCE, SOURCE_PLAN)]
    ar, sr, pr = readers
    for reader, pin in zip(readers, (AUDIT_SHA, SOURCE_SHA, SOURCE_PLAN_SHA), strict=True):
        reader.hash(reader.root / "manifest.json", pin)
    am = ar.json(AUDIT / "manifest.json"); ar.hash(AUDIT / "audit.json", am["audit_sha256"])
    audit = ar.json(AUDIT / "audit.json")
    require(audit["source_manifest_sha256"] == SOURCE_SHA
        and audit["plan_manifest_sha256"] == SOURCE_PLAN_SHA
        and audit["exact_new_phase_replay_pass"] is True and audit["csv_bytes_exact_replay"] is True
        and audit["protected_permissions_pass"] is True and audit["new_model_calls"] == 0,
        "completed_metadata_postflight_required")
    sm = sr.json(SOURCE / "manifest.json")
    for name, pin in sm["artifacts"].items(): sr.hash(SOURCE / name, pin["sha256"])
    pm = pr.json(SOURCE_PLAN / "manifest.json"); pr.hash(SOURCE_PLAN / "plan.json", pm["plan_sha256"])
    original = pr.json(SOURCE_PLAN / "plan.json")
    for field in ("source_pins", "artifact_pins"): check_pins(original[field])
    check_pins(audit["reader_pins"])
    rows = sr.json(SOURCE / "score_rows.json")["records"]
    pairs = sr.json(SOURCE / "paired_report_comparison.json")["records"]
    controls = freeze_controls(rows, pairs)
    completed = sr.json(SOURCE / "completed_triplets.json")["new_records"]
    triples = [c["baseline_triple"] for c in original["cases"]] + completed
    require(len(completed) == len(original["cases"]) == 2
        and {t["report_sha256"] for t in triples} == {r["report_sha256"] for r in rows},
        "complete_existing_report_artifact_inventory_required")
    for case, pair in zip(original["cases"], pairs, strict=True):
        new = next(r for r in rows if r["triple_candidate_id"] == pair["proposed_candidate_id"])
        require(source.assess_pair(case, new, original["workers"]) == pair, "source_gate_replay_changed")
    model_pins = {}
    for name, pin in previous.MODEL_HASHES.items():
        path = previous.BIOVIL_MODEL / name
        require(sha256_file(path) == pin, "existing_frozen_biovil_weights_changed")
        model_pins[str(path)] = {"sha256": pin, "size_bytes": path.stat().st_size}
    for path in sorted(previous.BIOVIL_MODEL.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".txt", ".py"):
            model_pins[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    vendors = sorted(previous.VENDOR.rglob("*.py"))
    require(vendors and os.access(previous.BIOVIL_PYTHON, os.X_OK), "existing_biovil_environment_required")
    pins = {**original["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
        str(previous.ROOT / "TriCompose-v1.2/tests/test_guarded_report_secondary.py"):
            sha256_file(previous.ROOT / "TriCompose-v1.2/tests/test_guarded_report_secondary.py"),
        **{str(path): sha256_file(path) for path in vendors}}
    # The inherited source plan already pins the old generic scorer and adapters.
    for path in (Path(previous.__file__), Path(sys.modules[previous.score.__module__].__file__)):
        resolved = str(path.resolve()); current = sha256_file(path)
        require(resolved not in pins or pins[resolved] == current, "consumed_scorer_source_changed")
        pins[resolved] = current
    def existing_run(path):
        resolved = require_inside(path, PROTECTED_ROOT, must_exist=True)
        require(any(resolved.is_relative_to(r) for r in (SOURCE, source.source.SOURCE)),
            "only_completed_synthetic_source_runs_allowed")
        return str(resolved)
    request = {"schema_version": previous.REQUEST_SCHEMA,
        "pairs": [{k: r[k] for k in sorted(previous.PAIR_FIELDS)} for r in rows],
        "modality_source": "fully_synthetic", "selection_used_biovil": False,
        "clinical_truth_available": False, "routing_or_calibration_update_allowed": False,
        "text_policy": "full_report_no_silent_truncation_overlength_is_na",
        "selection_sha256_sealed_before_endpoint": sr.hash(SOURCE / "paired_report_comparison.json")}
    payload = {"schema_version": VERSION, "source_run": str(SOURCE), "source_manifest_sha256": SOURCE_SHA,
        "source_audit_manifest_sha256": AUDIT_SHA, "source_plan_manifest_sha256": SOURCE_PLAN_SHA,
        "rows": rows, "paired_comparisons": pairs, "controls": controls, "source_pins": pins,
        "artifact_pins": {**original["artifact_pins"], **audit["reader_pins"], **ar.pins, **sr.pins, **pr.pins},
        "model_pins": model_pins, "model_path": str(previous.BIOVIL_MODEL),
        "python": str(previous.BIOVIL_PYTHON), "vendor_path": str(previous.VENDOR),
        "cxr_runs": sorted({existing_run(t["cxr_run"]) for t in triples}),
        "report_runs": sorted({existing_run(t["report_run"]) for t in triples}),
        "maximum_image_encodings": 2, "maximum_text_encodings": 4,
        "source_new_worker_attempts": 4, "historical_cost": original["historical_cost"],
        "new_generator_calls": 0, "new_primary_scoring_calls": 0, "new_planner_calls": 0,
        "endpoint_used_for_selection": False, "source_bodies_parsed": False, "factory_instantiated": False}
    require(len(payload["cxr_runs"]) == 2 and len(payload["report_runs"]) == 4, "two_image_four_report_scope_required")
    check_pins(pins)
    for reader in readers: reader.recheck()
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        private_directory(tmp / "endpoint_request")
        rp = write_private_json(tmp / "endpoint_request/request.json", request)
        mp = write_private_json(tmp / "endpoint_request/manifest.json", {"schema_version": previous.REQUEST_SCHEMA,
            "artifacts": {"request.json": {"sha256": sha256_file(rp)}}})
        payload["artifact_pins"].update({str(target / "endpoint_request/request.json"): sha256_file(rp),
            str(target / "endpoint_request/manifest.json"): sha256_file(mp)})
        path = write_private_json(tmp / "plan.json", payload)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(path),
            "planned_report_pairs": 4, "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(tmp, target)
    except BaseException:
        discard_atomic_run(tmp); raise
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
        and plan["source_plan_manifest_sha256"] == SOURCE_PLAN_SHA
        and plan["model_path"] == str(previous.BIOVIL_MODEL) and plan["python"] == str(previous.BIOVIL_PYTHON)
        and plan["vendor_path"] == str(previous.VENDOR) and plan["maximum_image_encodings"] == 2
        and plan["maximum_text_encodings"] == 4 and len(plan["rows"]) == 4
        and plan["new_generator_calls"] == plan["new_primary_scoring_calls"] == plan["new_planner_calls"] == 0
        and plan["endpoint_used_for_selection"] is False and plan["source_bodies_parsed"] is False
        and freeze_controls(plan["rows"], plan["paired_comparisons"]) == plan["controls"], "secondary_only_scope_required")
    for field in ("source_pins", "artifact_pins", "model_pins"): check_pins(plan[field])
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic()
    try:
        write_private_json(tmp / "attempt_reserved.json", {"schema_version": VERSION,
            "charged_endpoint_worker_attempts": 1, "maximum_image_encodings": 2,
            "maximum_text_encodings": 4, "status": "reserved_before_load_no_free_failed_work"})
        sys.path.insert(0, plan["vendor_path"])
        endpoint = previous.score(argparse.Namespace(request_run=root / "endpoint_request", model_path=plan["model_path"],
            cxr_run=plan["cxr_runs"], report_run=plan["report_runs"]))
        endpoint["historical_pool_is_untouched_test"] = False
        require(endpoint["producer"]["frozen"] is True
            and endpoint["producer"]["checkpoint_sha256"] == previous.MODEL_HASHES
            and endpoint["counts"]["requested_pairs"] == 4 and endpoint["counts"]["image_encoder_calls"] <= 2
            and endpoint["counts"]["text_encoder_calls"] <= 4, "frozen_endpoint_or_encoding_budget_changed")
        table, pairs, summary = endpoint_tables(plan["rows"], plan["controls"], endpoint)
        files = [write_private_json(tmp / "scores.json", endpoint), write_private_json(tmp / "summary.json", summary),
            write_private_text(tmp / "score_table.csv", previous.csv_text(table)),
            write_private_text(tmp / "case_comparison.csv", previous.csv_text(pairs))]
        for field in ("source_pins", "artifact_pins", "model_pins"): check_pins(plan[field])
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "completed_secondary_only_clinical_unvalidated", "plan_manifest_sha256": args.plan_manifest_sha256,
            "source_manifest_sha256": SOURCE_SHA, "charged_endpoint_worker_attempts": 1,
            "encoding_counts": endpoint["counts"], "peak_vram_gib": endpoint["peak_vram_gib"],
            "runtime_seconds_including_load_io": round(time.monotonic() - started, 3),
            "new_generator_calls": 0, "new_planner_calls": 0, "endpoint_used_for_selection": False,
            "original_winner_changed": False, "clinical_acceptance": False,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files}})
        commit_atomic_run(tmp, target)
    except BaseException:
        write_private_json(tmp / "failure.json", {"schema_version": VERSION,
            "status": "failed_endpoint_attempt_retained", "charged_endpoint_worker_attempts": 1,
            "completed_encoding_counts": None, "automatic_resume": False})
        commit_atomic_run(tmp, target); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep, execution = sub.add_parser("prepare"), sub.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for command in (prep, execution):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
