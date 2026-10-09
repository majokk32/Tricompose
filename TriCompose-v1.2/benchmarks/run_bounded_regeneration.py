#!/usr/bin/env python3
"""Approved GPU Slurm only: one conditional fresh CXR/report chain per EHR.

Reuses unchanged official frozen workers and durable per-attempt ledgers.
Seals decisions before evaluating BioViL. Never replaces historical winners.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (WORKSPACE, PROTECTED_ROOT, RUN_ID_PATTERN, require_inside, read_json, sha256_file,
    load_cxr_candidates, load_report_candidates, private_directory, write_private_json, write_private_text)
from tricompose_v12.bounded_regeneration import SCHEMA, POLICY, validate_policy, alternate_request, route, decision, validate_row
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_execution import one_case
from tricompose_v12.live_workers import require_gpu_slurm, check_pins, run_private_process
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.full_pool_report_control import validate_endpoint
from tricompose_v12.runtime_dispatch import ProtectedJournal
from tricompose_v11.cxr_contracts import validate_cxr_request, canonical_json_sha256
from run_fixed_image_reports import normalize_owned_modes
from run_automatic_proxy_replay import render_csv
from compare_frozen_report_paths import flatten
from score_automatic_replay_biovil import PAIR_FIELDS, REQUEST_SCHEMA


def load(args):
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / "manifest.json") != args.plan_manifest_sha256:
        raise ValueError("approved retry plan changed")
    manifest = read_json(root / "manifest.json")
    if (manifest["schema_version"] != SCHEMA or manifest["status"] != "cpu_preflight_complete_gpu_retry_not_run"
            or sha256_file(root / "plan.json") != manifest["plan_sha256"]):
        raise ValueError("sealed CPU plan required")
    plan = read_json(root / "plan.json")
    validate_policy(plan["policy"])
    if (plan["schema_version"] != SCHEMA or len(plan["cases"]) != 2
            or any(plan[k] is not False for k in ("factory_instantiated", "source_bodies_parsed", "secondary_values_read",
                                                "clinical_acceptance", "historical_pool_is_untouched_test"))
            or plan["new_model_calls"] != 0 or plan["endpoint_timeout_seconds"] != 150
            or plan["cohort"]["original_ehr_cases"] != 80 or plan["cohort"]["smoke_cases"] != 2
            or len({c["case_id"] for c in plan["cases"]}) != 2):
        raise ValueError("exact nonclinical two-case scope required")
    check_pins(plan["source_pins"])
    check_pins(plan["artifact_pins"])
    for spec in plan["workers"].values():
        check_pins(spec["asset_pins"])
    check_pins(plan["biovil_asset_pins"])
    for case in plan["cases"]:
        anchor = anchor_from_record(case["anchor"])
        baseline, static = case["baseline"], case["static"]
        validate_row(baseline)
        validate_row(static)
        if (anchor.sha256 != case["ehr_anchor_sha256"] or anchor.case_id != case["case_id"]
                or baseline["receipt"]["ehr_anchor_sha256"] != anchor.sha256
                or static["receipt"]["ehr_anchor_sha256"] != anchor.sha256
                or baseline["cxr_candidate_id"] != static["cxr_candidate_id"]
                or baseline["cxr_sha256"] != static["cxr_sha256"]
                or baseline["cxr_model_id"] != POLICY["baseline_cxr_model"]
                or baseline["report_model_id"] != POLICY["report_model"] or baseline["seed"] != 0
                or route(baseline) != case["frozen_action"] or len(case["requests"]) != 1):
            raise ValueError("fixed baseline, static reference or action differs")
        row = case["requests"][0]
        if (sha256_file(row["source_path"]) != row["source_sha256"]
                or read_json(row["source_path"]) != case["original_request"]
                or alternate_request(case["original_request"]) != row["request"]
                or canonical_json_sha256(row["request"]) != row["canonical_request_sha256"]):
            raise ValueError("seed-only retry source differs")
        validate_cxr_request(row["request"])
    return plan


def candidate_row(triple):
    """Internal GPU controller only: existing structure extractor reads synthetic text."""
    from evaluate_report_structure import _record
    images = load_cxr_candidates([triple["cxr_run"]])
    reports = load_report_candidates([triple["report_run"]], cxr_candidates=images)
    if len(images) != 1 or len(reports) != 1:
        raise ValueError("one image and one report per chain required")
    image, report = next(iter(images.values())), next(iter(reports.values()))
    receipt_path = require_inside(triple["receipt_path"], PROTECTED_ROOT, must_exist=True)
    if sha256_file(receipt_path) != triple["receipt_sha256"]:
        raise ValueError("new completed receipt changed")
    receipt = read_json(receipt_path)
    row = {"case_id": triple["case_id"], "ehr_sha256": triple["ehr_sha256"],
        "ehr_facts_sha256": triple["ehr_facts_sha256"], "cxr_candidate_id": image["candidate_id"],
        "report_candidate_id": report["candidate_id"], "cxr_sha256": image["artifact"]["sha256"],
        "report_sha256": report["artifact"]["sha256"], "cxr_model_id": image["model_id"],
        "report_model_id": report["model_id"], "seed": image["seed"], "receipt": receipt,
        "raw_edge_readouts": receipt["raw_edge_readouts"], "structure": _record(report, image, normalized_frequency={})}
    row["triple_candidate_id"] = "retrypair_" + _digest([image["candidate_id"], report["candidate_id"]])[:32]
    validate_row(row)
    return row


def secondary(root, plan, rows, triples, selection_sha, journal):
    pack = root / "endpoint_request"
    private_directory(pack)
    request = {"schema_version": REQUEST_SCHEMA, "pairs": [{k: row[k] for k in PAIR_FIELDS} for row in rows],
        "modality_source": "fully_synthetic", "selection_used_biovil": False,
        "clinical_truth_available": False, "routing_or_calibration_update_allowed": False,
        "text_policy": "full_report_no_silent_truncation_overlength_is_na",
        "selection_sha256_sealed_before_endpoint": selection_sha}
    path = write_private_json(pack / "request.json", request)
    write_private_json(pack / "manifest.json", {"schema_version": REQUEST_SCHEMA,
        "artifacts": {"request.json": {"sha256": sha256_file(path)}}})
    argv = [plan["biovil_python"], str(WORKSPACE / "TriCompose-v1.2/benchmarks/score_automatic_replay_biovil.py"),
        "score", "--request-run", str(pack), "--model-path", plan["biovil_model"],
        "--output-root", str(root), "--run-id", "secondary"]
    for path in sorted(set(plan["source_cxr_runs"] + [t["cxr_run"] for t in triples])):
        argv += ["--cxr-run", path]
    for path in sorted(set(plan["source_report_runs"] + [t["report_run"] for t in triples])):
        argv += ["--report-run", path]
    runtime = root / "endpoint_runtime"
    private_directory(runtime)
    journal.append({"stage": "secondary_endpoint", "status": "reserved_before_spawn",
        "counts_not_in_generation_verification_budget": True,
        "maximum_pairs": 6, "maximum_image_encodings": 4, "maximum_text_encodings": 6,
        "timeout_seconds": plan["endpoint_timeout_seconds"]})
    started = time.monotonic()
    try:
        run_private_process(argv, runtime, plan["endpoint_timeout_seconds"])
        manifest = read_json(root / "secondary/manifest.json")
        score_path = root / "secondary/scores.json"
        if sha256_file(score_path) != manifest["artifacts"]["scores.json"]["sha256"]:
            raise ValueError("secondary endpoint output changed")
        endpoint = read_json(score_path)
        endpoint["historical_pool_is_untouched_test"] = False
        indexed = {r["triple_candidate_id"]: r for r in endpoint["records"]}
        if (set(indexed) != {r["triple_candidate_id"] for r in rows}
                or len(indexed) != len(endpoint["records"]) or endpoint["used_for_routing"] is not False
                or endpoint["clinical_truth_available"] is not False
                or endpoint["counts"]["image_encoder_calls"] > 4 or endpoint["counts"]["text_encoder_calls"] > 6):
            raise ValueError("bounded complete secondary-only endpoint required")
        for row in rows:
            if any(row[k] != indexed[row["triple_candidate_id"]][k] for k in PAIR_FIELDS):
                raise ValueError("endpoint artifact lineage differs")
        validate_endpoint(rows, endpoint)
        journal.append({"stage": "secondary_endpoint", "status": "completed_validated",
            "wall_seconds_including_load_io": time.monotonic() - started, "encodings": endpoint["counts"]})
        return endpoint
    except Exception as exc:
        journal.append({"stage": "secondary_endpoint", "status": "failed_charged_not_free",
            "error_type": type(exc).__name__, "wall_seconds_including_load_io": time.monotonic() - started})
        return {"records": [{k: row[k] for k in PAIR_FIELDS} | {
            "biovil_raw_cosine": None, "status": "not_available", "reason": "bounded_secondary_worker_failed",
            "calibrated": False} for row in rows], "used_for_routing": False,
            "clinical_truth_available": False, "historical_pool_is_untouched_test": False, "counts": None}


def tables(rows, choices, endpoint):
    scores = validate_endpoint(rows, endpoint)
    indexed = {r["triple_candidate_id"]: r for r in rows}
    table, cases = [], []
    for row in rows:
        record = flatten({k: row[k] for k in ("case_id", "triple_candidate_id", "cxr_model_id",
            "report_model_id", "seed", "raw_edge_readouts")})
        record.update({k: scores[row["triple_candidate_id"]][k] for k in ("biovil_raw_cosine", "status", "reason")})
        table.append({k: "NA" if v is None else v for k, v in record.items()})
    for choice in choices:
        record = {"case_id": choice["case_id"], "status": choice["status"], "action": choice["action"]["action"]}
        for name, key in (("fixed", "baseline_triple_id"), ("static", "static_triple_id"), ("bounded_retry", "selected_triple_id")):
            row = indexed[choice[key]]
            record[name + "_biovil_raw_cosine"] = scores[choice[key]]["biovil_raw_cosine"]
            record.update({name + "_" + k: v for k, v in
                           flatten({"raw_edge_readouts": row["raw_edge_readouts"]}).items()})
        cases.append({k: "NA" if v is None else v for k, v in record.items()})
    return table, cases


def run(args):
    require_gpu_slurm()  # BEFORE input reads, directories or model instantiation.
    plan = load(args)
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("opaque immutable run ID required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = require_inside(output / args.run_id, PROTECTED_ROOT, must_exist=False)
    private_directory(root)
    private_directory(root / "cases")
    started = time.monotonic()
    write_private_json(root / "start_manifest.json", {"schema_version": SCHEMA,
        "status": "in_progress", "plan_manifest_sha256": args.plan_manifest_sha256,
        "clinical_acceptance": False, "automatic_resume": False})
    triples, books, choices, rows = [], [], [], {}
    try:
        with ProtectedJournal(root / "controller.journal.jsonl") as journal:
            for case in plan["cases"]:
                base, static = case["baseline"], case["static"]
                rows[base["triple_candidate_id"]] = base
                rows[static["triple_candidate_id"]] = static
                action = route(base)
                journal.append({"stage": "route", "case_id": case["case_id"], "action": action})
                alt, failure = None, None
                if action["action"] == "regenerate_cxr":
                    completed, ledger, _ = one_case(case, plan, root)
                    books.append(ledger)
                    if len(completed) > 1:
                        raise ValueError("retry exceeded one-chain budget")
                    triples += completed
                    if completed:
                        alt = candidate_row(completed[0])
                        rows[alt["triple_candidate_id"]] = alt
                    else:
                        failure = "bounded_generation_or_verification_incomplete"
                choices.append(decision(base, static, alt, failure))
            selection = {"schema_version": SCHEMA, "policy": POLICY, "cohort": plan["cohort"],
                "choices": choices, "used_biovil": False, "clinical_acceptance": False,
                "original_winners_changed": False}
            selection_path = write_private_json(root / "selection.json", selection)
            selection_sha = sha256_file(selection_path)
            journal.append({"stage": "choice", "status": "sealed_before_secondary_values",
                "selection_sha256": selection_sha})
            row_list = [rows[k] for k in sorted(rows)]
            write_private_json(root / "score_rows.json", {"records": row_list})
            write_private_json(root / "completed_chains.json", {"records": triples})
            write_private_json(root / "execution_summary.json", {"case_ledgers": books})
            endpoint = secondary(root, plan, row_list, triples, selection_sha, journal)
            if sha256_file(selection_path) != selection_sha:
                raise ValueError("selection changed after endpoint")
            write_private_json(root / "endpoint.json", endpoint)
            table, case_comparison = tables(row_list, choices, endpoint)
            write_private_text(root / "score_table.csv", render_csv(table))
            write_private_text(root / "case_comparison.csv", render_csv(case_comparison))
            if sum(b["charged_model_attempts"] for b in books) > 8:
                raise ValueError("fixed generation/verification budget exceeded")
            check_pins(plan["source_pins"])
            check_pins(plan["artifact_pins"])
            for spec in plan["workers"].values():
                check_pins(spec["asset_pins"])
            check_pins(plan["biovil_asset_pins"])
            journal.append({"stage": "finish", "status": "completed_unvalidated_no_original_replacement"})
        files = ("selection.json", "score_rows.json", "completed_chains.json", "execution_summary.json",
                 "endpoint.json", "score_table.csv", "case_comparison.csv", "controller.journal.jsonl")
        result = {"schema_version": SCHEMA, "status": "completed_bounded_retry_engineering_unvalidated",
            "plan_manifest_sha256": args.plan_manifest_sha256, "cohort": plan["cohort"],
            "routed_retry_cases": sum(c["action"]["action"] == "regenerate_cxr" for c in choices),
            "completed_new_chains": len(triples),
            "exploratory_gate_pass_cases": sum(c["status"] == "exploratory_retry_gate_pass" for c in choices),
            "generation_verification_attempts_including_failures": sum(b["charged_model_attempts"] for b in books),
            "failed_generation_verification_attempts": sum(b["failed_attempts"] for b in books),
            "endpoint_encodings_separate_from_retry_budget": endpoint["counts"],
            "historical_generation_scoring_cost_is_not_zero": True,
            "wall_seconds_including_load_and_io": round(time.monotonic() - started, 3), "gpu_seconds": None,
            "selection_sha256_before_endpoint": selection_sha,
            "artifacts": {name: {"sha256": sha256_file(root / name)} for name in files},
            "clinical_acceptance": False, "clinical_repair_success": False,
            "clinical_fault_location_accuracy": None, "historical_pool_is_untouched_test": False,
            "same_budget_static_reranking_control_available": False,
            "original_winners_changed": False, "automatic_resume": False}
        write_private_json(root / "manifest.json", result)
    except BaseException as exc:
        write_private_json(root / "failure_manifest.json", {"schema_version": SCHEMA,
            "status": "failed_or_interrupted_charged_artifacts_retained", "error_type": type(exc).__name__,
            "automatic_resume": False, "clinical_acceptance": False})
        raise
    finally:
        normalize_owned_modes(root)
    return root, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-run", "plan-manifest-sha256", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    os.umask(0o007)
    try:
        root, result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "bounded_retry_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": result["status"], "runtime_seconds": result["wall_seconds_including_load_and_io"],
        "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
