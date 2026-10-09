#!/usr/bin/env python3
"""Recover only the missing secondary endpoint; preserve the completed run.

CPU prepare authenticates the sealed retry and metadata audit. GPU mode uses
the already deployed, pinned BioViL vendor path, just as the successful prior
workers do. No EHR/image/report generation or choice/threshold modification.
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
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file,
    private_directory, new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v12.live_workers import require_gpu_slurm, check_pins
from tricompose_v12.full_pool_report_control import validate_endpoint
from run_bounded_regeneration import load, tables
from run_automatic_proxy_replay import render_csv
from score_automatic_replay_biovil import PAIR_FIELDS, REQUEST_SCHEMA, score

SCHEMA = "tricompose-bounded-regeneration-secondary-recovery-v1"
SOURCE = PROTECTED_ROOT / "tricompose_v1_2/bounded_regeneration_runs/retry2_12632531"
SOURCE_SHA = "eb8099458d895da46281f1955c8e73f19082b61b0759bb8064bcb79ff922c32d"
AUDIT = PROTECTED_ROOT / "tricompose_v1_2/bounded_regeneration_audits/retry2_12632531"
AUDIT_SHA = "58def50df7b72bc1d2cd6fc99bd4dd411f10241cc3633625d2c1e98e846a53f5"
OLD_PLAN = PROTECTED_ROOT / "tricompose_v1_2/bounded_regeneration_plans/retry2_12625457_001"
OLD_PLAN_SHA = "5a4e3bfbfd7e87879e14ea545932e3fef6331f3440bb4ef2973e52355fec8542"
VENDOR = WORKSPACE / "runtime/vendor/hi_ml_multimodal_0_2_2"


def prepare(args):
    require_slurm()
    if sha256_file(SOURCE / "manifest.json") != SOURCE_SHA or sha256_file(AUDIT / "manifest.json") != AUDIT_SHA:
        raise ValueError("exact completed retry and audit required")
    source, audit_manifest = read_json(SOURCE / "manifest.json"), read_json(AUDIT / "manifest.json")
    audit_path = AUDIT / "audit.json"
    if sha256_file(audit_path) != audit_manifest["audit_sha256"]:
        raise ValueError("audit receipt changed")
    audit = read_json(audit_path)
    if (audit["source_manifest_sha256"] != SOURCE_SHA or audit["status"] != "passed_metadata_not_clinical"
            or audit["plan_manifest_sha256"] != OLD_PLAN_SHA or source["endpoint_encodings_separate_from_retry_budget"] is not None
            or source["clinical_acceptance"] is not False):
        raise ValueError("only the missing secondary endpoint may be recovered")
    old = load(argparse.Namespace(plan_run=OLD_PLAN, plan_manifest_sha256=OLD_PLAN_SHA))
    pins = {str(SOURCE / "manifest.json"): SOURCE_SHA, str(AUDIT / "manifest.json"): AUDIT_SHA,
            str(audit_path): audit_manifest["audit_sha256"], **old["artifact_pins"]}
    for name, entry in source["artifacts"].items():
        pins[str(require_inside(SOURCE / name, SOURCE, must_exist=True))] = entry["sha256"]
    check_pins(pins)
    rows = read_json(SOURCE / "score_rows.json")["records"]
    selection = read_json(SOURCE / "selection.json")
    if sha256_file(SOURCE / "selection.json") != source["selection_sha256_before_endpoint"] or selection["used_biovil"] is not False:
        raise ValueError("original choice must remain sealed")
    original_pack = SOURCE / "endpoint_request"
    request = read_json(original_pack / "request.json")
    original_manifest = read_json(original_pack / "manifest.json")
    if (sha256_file(original_pack / "request.json") != original_manifest["artifacts"]["request.json"]["sha256"]
            or request["pairs"] != [{k: r[k] for k in PAIR_FIELDS} for r in rows]
            or request["selection_sha256_sealed_before_endpoint"] != source["selection_sha256_before_endpoint"]
            or not 2 <= len(rows) <= 6 or len({r["case_id"] for r in rows}) != 2):
        raise ValueError("authenticated original bounded endpoint request required")
    pins[str(original_pack / "request.json")] = sha256_file(original_pack / "request.json")
    pins[str(original_pack / "manifest.json")] = sha256_file(original_pack / "manifest.json")
    triples = read_json(SOURCE / "completed_chains.json")["records"]
    if len(triples) != 2:
        raise ValueError("completed two-chain source required; no regeneration here")
    if not (VENDOR / "health_multimodal/__init__.py").is_file():
        raise ValueError("existing local BioViL runtime package missing")
    sources = {**old["source_pins"], str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    check_pins(sources)
    check_pins(old["biovil_asset_pins"])
    payload = {"schema_version": SCHEMA, "source_run": str(SOURCE), "source_manifest_sha256": SOURCE_SHA,
        "source_audit_manifest_sha256": AUDIT_SHA, "selection_sha256": source["selection_sha256_before_endpoint"],
        "source_pins": sources, "artifact_pins": pins, "biovil_asset_pins": old["biovil_asset_pins"],
        "model_path": old["biovil_model"], "python": old["biovil_python"], "vendor_path": str(VENDOR),
        "cxr_runs": sorted(set(old["source_cxr_runs"] + [r["cxr_run"] for r in triples])),
        "report_runs": sorted(set(old["source_report_runs"] + [r["report_run"] for r in triples])),
        "maximum_pairs": 6, "maximum_image_encodings": 4, "maximum_text_encodings": 6,
        "planned_pairs": len(rows), "new_generation_calls": 0, "new_primary_scoring_calls": 0,
        "selection_or_threshold_update_allowed": False, "factory_instantiated": False, "source_bodies_parsed": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        pack = temporary / "endpoint_request"
        private_directory(pack)
        rp = write_private_json(pack / "request.json", request)
        rm = write_private_json(pack / "manifest.json", {"schema_version": REQUEST_SCHEMA,
            "artifacts": {"request.json": {"sha256": sha256_file(rp)}}})
        payload["artifact_pins"].update({str(target / "endpoint_request/request.json"): sha256_file(rp),
            str(target / "endpoint_request/manifest.json"): sha256_file(rm)})
        path = write_private_json(temporary / "plan.json", payload)
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA,
            "status": "prepared_secondary_only_no_new_generation", "plan_sha256": sha256_file(path),
            "planned_pairs": len(rows), "new_model_calls": 0, "source_run_unchanged": True})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target


def run(args):
    require_gpu_slurm()  # BEFORE inputs, directories, or heavyweight imports.
    plan_root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(plan_root / "manifest.json") != args.plan_manifest_sha256:
        raise ValueError("approved recovery plan changed")
    manifest = read_json(plan_root / "manifest.json")
    if manifest["schema_version"] != SCHEMA or sha256_file(plan_root / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("sealed secondary recovery plan required")
    plan = read_json(plan_root / "plan.json")
    if (plan["vendor_path"] != str(VENDOR) or plan["source_run"] != str(SOURCE) or plan["source_manifest_sha256"] != SOURCE_SHA
            or plan["new_generation_calls"] != 0 or plan["new_primary_scoring_calls"] != 0
            or plan["selection_or_threshold_update_allowed"] is not False or plan["factory_instantiated"] is not False
            or plan["source_bodies_parsed"] is not False or plan["maximum_pairs"] != 6
            or plan["maximum_image_encodings"] != 4 or plan["maximum_text_encodings"] != 6):
        raise ValueError("fixed secondary-only scope required")
    check_pins(plan["source_pins"])
    check_pins(plan["artifact_pins"])
    check_pins(plan["biovil_asset_pins"])
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic()
    try:
        write_private_json(temporary / "attempt_reserved.json", {"schema_version": SCHEMA,
            "status": "reserved_before_model_load_and_encodings", "maximum_pairs": 6,
            "maximum_image_encodings": 4, "maximum_text_encodings": 6,
            "new_generation_calls": 0, "selection_sha256": plan["selection_sha256"]})
        # Required bootstrap, copied from the already successful local workers.
        sys.path.insert(0, plan["vendor_path"])
        scores = score(argparse.Namespace(request_run=plan_root / "endpoint_request", model_path=plan["model_path"],
            cxr_run=plan["cxr_runs"], report_run=plan["report_runs"]))
        scores["historical_pool_is_untouched_test"] = False
        rows = read_json(SOURCE / "score_rows.json")["records"]
        selection = read_json(SOURCE / "selection.json")
        validate_endpoint(rows, scores)
        if (scores["counts"]["requested_pairs"] != plan["planned_pairs"] or scores["counts"]["image_encoder_calls"] > 4
                or scores["counts"]["text_encoder_calls"] > 6 or sha256_file(SOURCE / "selection.json") != plan["selection_sha256"]):
            raise ValueError("endpoint exceeded budget or altered sealed selection")
        score_path = write_private_json(temporary / "scores.json", scores)
        table, cases = tables(rows, selection["choices"], scores)
        table_path = write_private_text(temporary / "score_table.csv", render_csv(table))
        cases_path = write_private_text(temporary / "case_comparison.csv", render_csv(cases))
        check_pins(plan["source_pins"])
        check_pins(plan["artifact_pins"])
        check_pins(plan["biovil_asset_pins"])
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA,
            "status": "completed_secondary_recovery_not_clinical", "source_manifest_sha256": SOURCE_SHA,
            "source_selection_sha256": plan["selection_sha256"], "plan_manifest_sha256": args.plan_manifest_sha256,
            "source_run_unchanged": True, "original_failed_endpoint_attempt_is_not_free": True,
            "new_generation_calls": 0, "new_primary_scoring_calls": 0, "counts": scores["counts"],
            "wall_seconds_including_load_io": round(time.monotonic() - started, 3), "peak_vram_gib": scores["peak_vram_gib"],
            "endpoint_used_for_selection": False, "clinical_acceptance": False,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in (score_path, table_path, cases_path)}})
        commit_atomic_run(temporary, target)
    except BaseException as exc:
        # Preserve failed charged work; never erase it or auto-rerun generation.
        write_private_json(temporary / "failure.json", {"schema_version": SCHEMA, "error_type": type(exc).__name__,
            "status": "failed_endpoint_attempt_retained", "automatic_resume": False,
            "elapsed_seconds_including_load_io": time.monotonic() - started})
        commit_atomic_run(temporary, target)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    execution = sub.add_parser("run")
    execution.add_argument("--plan-run", required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for command in (prep, execution):
        command.add_argument("--output-root", required=True)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    started = time.monotonic()
    try:
        root = prepare(args) if args.mode == "prepare" else run(args)
    except Exception as exc:
        print(json.dumps({"status": "secondary_recovery_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "prepared_secondary_only_no_new_generation" if args.mode == "prepare" else "completed_secondary_recovery_not_clinical",
        "runtime_seconds": round(time.monotonic() - started, 3), "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
