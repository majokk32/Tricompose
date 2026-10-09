#!/usr/bin/env python3
"""CPU-only full-pool receipt/selection/table audit in a Slurm allocation.

Authenticate cached structure metadata, but do not reopen report bodies or
image pixels. This audit cannot establish independent clinical correctness.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file, load_cxr_candidates,
    load_report_candidates, new_atomic_run, write_private_json, commit_atomic_run, discard_atomic_run)
from run_full_pool_report_control import load, build_rows, table_rows
from compare_frozen_report_paths import flatten
from tricompose_v12.full_pool_report_control import SCHEMA, freeze, summarize
from tricompose_v12.live_workers import check_pins
from tricompose_v12.runtime_dispatch import require_slurm


def verify_csv(path, expected):
    with path.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    rendered = [{k: "NA" if v is None else str(v) for k, v in row.items()} for row in expected]
    if actual != rendered:
        raise ValueError("exact CSV row inventory, fields, values or NA differs")


def audit(args):
    require_slurm()
    started = time.monotonic()
    root = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    mp = root / "manifest.json"
    manifest_sha = sha256_file(mp)
    m = read_json(mp)
    if (m.get("schema_version") != SCHEMA or m.get("status") != "completed_full_pool_fresh_report_control_unvalidated"
            or m.get("new_generation_calls") != 0 or m.get("new_xrv_scored_images") != 240
            or m.get("new_chexbert_scored_reports") != 960 or m.get("new_biovil_encodings") != 0
            or m.get("reused_secondary_pairs") != 960 or m.get("requested_chexbert_batch_size") != 16
            or m.get("legacy_model_calls_are_sample_counts") is not True
            or m.get("historical_generation_and_secondary_costs_are_not_zero") is not True
            or any(m.get(k) is not False for k in ("clinical_acceptance", "clinical_repair_success",
                "adaptive_repair_executed", "original_winners_changed", "biovil_used_for_selection", "historical_pool_is_untouched_test"))):
        raise ValueError("completed nonclinical fresh full-pool report control required")
    args.plan_manifest_sha256 = m["plan_manifest_sha256"]
    plan = load(args)
    for name, entry in m["artifacts"].items():
        if sha256_file(require_inside(root / name, root, must_exist=True)) != entry["sha256"]:
            raise ValueError("published artifact changed")
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or info.st_gid not in (96293, 65534)
                or stat.S_IMODE(info.st_mode) != (0o2770 if path.is_dir() else 0o660)):
            raise ValueError("project-private mode/group differs")
    for spec in (plan["xrv"], plan["chexbert"]):
        check_pins(spec["asset_pins"])
    cxrs = load_cxr_candidates(plan["cxr_runs"])
    reports = load_report_candidates(plan["report_runs"], cxr_candidates=cxrs)
    ip, tp = root / "image_labels/cxr_finding_labels.json", root / "report_labels/report_finding_labels.json"
    if sha256_file(ip) != m["xrv_labels_sha256"] or sha256_file(tp) != m["chexbert_labels_sha256"]:
        raise ValueError("fresh scorer bundle changed")
    il, tl = read_json(ip), read_json(tp)
    structures = read_json(root / "structure_records.json")["records"]
    indexed = {r["report_candidate_id"]: r for r in structures}
    if len(structures) != 960 or set(indexed) != set(reports):
        raise ValueError("full structure metadata inventory differs")
    frequency = Counter(s["normalized_report_sha256"] for s in structures)
    if any(s["normalized_template_frequency"] != frequency[s["normalized_report_sha256"]] for s in structures):
        raise ValueError("normalized template counts differ")
    rows, partials = build_rows(plan, cxrs, reports, il, tl, indexed,
        image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp))
    if (read_json(root / "score_rows.json") != {"records": rows}
            or read_json(root / "image_receipts.json") != {"records": partials}):
        raise ValueError("fresh source-bound receipts or raw score rows differ")
    selection = read_json(root / "selection.json")
    endpoint = read_json(root / "secondary.json")
    cached = read_json(plan["cached_secondary_path"])
    expected_endpoint = {"records": cached["records"], "used_for_routing": False, "clinical_truth_available": False,
        "historical_pool_is_untouched_test": False, "new_biovil_encodings": 0,
        "source_sha256": plan["cached_secondary_sha256"], "selection_sha256": m["selection_sha256_before_secondary"]}
    if (sha256_file(root / "selection.json") != m["selection_sha256_before_secondary"]
            or freeze(rows) != selection or endpoint != expected_endpoint):
        raise ValueError("sealed decision or historical endpoint values/lineage differ")
    comparison = summarize(rows, selection, endpoint)
    if (comparison != read_json(root / "comparison.json")
            or comparison["summary"] != read_json(root / "summary.json")):
        raise ValueError("raw totals or case-level paired bootstrap differs")
    verify_csv(root / "score_table.csv", table_rows(rows, endpoint))
    verify_csv(root / "model_comparison.csv", [flatten(r) for r in comparison["model_comparison"]])
    verify_csv(root / "case_outcomes.csv", comparison["case_outcomes"])
    if (m["peak_vram_gib"] != {"xrv": il["peak_vram_gib"], "chexbert": tl["peak_vram_gib"]}
            or [r["stage"] for r in m["worker_costs"]] != ["xrv", "chexbert"]):
        raise ValueError("scorer sample/memory/stage accounting differs")
    events = [json.loads(s) for s in (root / "cost_journal.jsonl").read_text().splitlines()]
    if [(e.get("stage"), e["status"]) for e in events] != [
            ("xrv", "reserved_before_spawn"), ("xrv", "process_completed_unvalidated"),
            ("chexbert", "reserved_before_spawn"), ("chexbert", "process_completed_unvalidated"),
            ("choice", "sealed_before_secondary_values"), ("secondary", "cached_measurement_validated")]:
        raise ValueError("bounded worker reservations or seal-before-endpoint journal differs")
    if (events[0]["maximum_sample_units"] != {"xrv_scored_images": 240}
            or events[2]["maximum_sample_units"] != {"chexbert_scored_reports": 960}
            or events[0]["timeout_seconds"] != 420 or events[2]["timeout_seconds"] != 420
            or events[4]["selection_sha256"] != m["selection_sha256_before_secondary"]
            or events[5]["new_biovil_encodings"] != 0 or events[5]["cached_pairs"] != 960):
        raise ValueError("worker bounds or historical reuse accounting differs")
    check_pins(plan["source_pins"])
    check_pins(plan["artifact_pins"])
    if sha256_file(mp) != manifest_sha:
        raise ValueError("source manifest changed during audit")
    result = {"schema_version": "tricompose-full-pool-report-audit-v1", "status": "metadata_hash_receipt_audit_passed",
        "source_manifest_sha256": manifest_sha, "source_plan_manifest_sha256": m["plan_manifest_sha256"],
        "fixed_ehr_cases": 80, "fixed_images": 240, "report_candidates": 960,
        "raw_edge_csv_verified": True, "model_comparison_verified": True,
        "receipts_and_sealed_choices_recomputed": True, "case_level_bootstraps_recomputed": True,
        "original_secondary_records_retained_exactly": True,
        "direct_ehr_cases": comparison["summary"]["direct_ehr_cases"],
        "no_direct_ehr_cases": comparison["summary"]["no_direct_ehr_cases"],
        "gate_passing_alternatives": comparison["summary"]["gate_passing_alternatives"],
        "changed_images": comparison["summary"]["changed_images"],
        "source_bodies_or_image_pixels_opened": False,
        "structure_scope": "authenticated_worker_metadata_not_body_reexecution",
        "new_model_calls": 0, "clinical_acceptance": False, "clinical_repair_success": False,
        "runtime_seconds": round(time.monotonic() - started, 3)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(temporary / "audit.json", result)
        write_private_json(temporary / "manifest.json", {"schema_version": result["schema_version"],
            "audit_sha256": sha256_file(path), "source_manifest_sha256": manifest_sha,
            "auditor_sha256": sha256_file(Path(__file__))})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "output-root", "run-id"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    os.umask(0o007)
    try:
        root, result = audit(args)
    except Exception as exc:
        print(json.dumps({"status": "audit_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"],
        "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
