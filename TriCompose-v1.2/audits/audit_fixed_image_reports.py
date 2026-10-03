#!/usr/bin/env python3
"""CPU Slurm post-run audit, metadata/hashes only; no bodies/pixels/models.

Recompute evidence arithmetic from the same frozen proxies. This authenticates
execution/selection, not independent clinical correctness or GPU-kernel time.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.2/audits",
                 "TriCompose-v1.1/src", "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file, load_cxr_candidates,
    load_report_candidates, new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json)
from tricompose_v12.fixed_image_reports import report_row, freeze_selection, secondary_comparison
from tricompose_v12.invariant_verification import _digest, _HASH
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt
from tricompose_v12.live_workers import check_pins
from tricompose_v12.runtime_dispatch import require_slurm
from run_fixed_image_reports import load, csv_table
from audit_live_smoke import private_modes
from score_automatic_replay_biovil import MODEL_HASHES, PAIR_FIELDS, validate_pairs

STAGES = (("maira", {"report_generation_samples": 4}, 480),
          ("chexbert", {"chexbert_scored_samples": 4}, 180),
          ("biovil", {"image_encoder_samples": 4, "text_encoder_samples": 8}, 180))


def validate_journal(events, summary, endpoint_counts):
    """No unlogged, missing, reordered, failed or retried bulk stages."""
    if len(events) != 9 or len(summary["worker_costs"]) != 3:
        raise ValueError("complete three-stage durable journal required")
    wall = []
    for i, (name, units, timeout) in enumerate(STAGES):
        reserve, process, validated = events[3*i:3*i+3]
        if (reserve.get("stage") != name or reserve.get("status") != "reserved_before_spawn"
                or reserve.get("maximum_sample_units") != units or reserve.get("timeout_seconds") != timeout
                or not isinstance(reserve.get("argv_sha256"), str) or not _HASH.fullmatch(reserve["argv_sha256"])
                or process != {"stage": name, "status": "process_completed_unvalidated"}):
            raise ValueError("bounded pre-spawn reservation/process order differs")
        expected = {"stage": name, "status": "validated"}
        expected.update(units if name != "biovil" else {"counts": endpoint_counts})
        if validated != expected: raise ValueError("validated stage counts differ")
        cost = summary["worker_costs"][i]
        seconds = cost.get("wall_seconds_including_startup_io")
        if (cost.get("stage") != name or cost.get("maximum_sample_units") != units
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("worker cost units or wall time differs")
        wall.append(seconds)
    total = summary.get("wall_seconds_including_startup_io")
    if type(total) not in (int, float) or not math.isfinite(total) or total + .01 < sum(wall):
        raise ValueError("total wall time cannot be less than stage times")


def validate_endpoint(endpoint, rows):
    if (endpoint.get("schema_version") != "tricompose-automatic-replay-secondary-biovil-v1"
            or endpoint.get("status") != "completed_secondary_biovil"
            or endpoint.get("used_for_routing") is not False
            or endpoint.get("clinical_truth_available") is not False
            or endpoint.get("primary_clinical_metric") is not False
            or endpoint.get("original_selection_changed") is not False
            or endpoint.get("producer", {}).get("frozen") is not True
            or endpoint["producer"].get("checkpoint_sha256") != MODEL_HASHES
            or endpoint["producer"].get("text_policy") != "full_report_no_silent_truncation_overlength_is_na"):
        raise ValueError("frozen secondary-only full-text endpoint required")
    records = endpoint["records"]
    if len(records) != 8 or len(rows) != 8: raise ValueError("all eight pairs retained required")
    source = {r["triple_candidate_id"]: r for r in rows}
    returned = {r["triple_candidate_id"]: r for r in records}
    if len(source) != 8 or len(returned) != 8 or set(source) != set(returned):
        raise ValueError("unique complete endpoint pairs required")
    good = []
    for r in records:
        if any(r[name] != source[r["triple_candidate_id"]][name] for name in PAIR_FIELDS):
            raise ValueError("secondary endpoint case/parent/artifact lineage differs")
        if r.get("calibrated") is not False: raise ValueError("raw cosine is not calibrated")
        value = r.get("biovil_raw_cosine")
        if value is None:
            if r.get("status") != "not_available" or not r.get("reason"):
                raise ValueError("missing secondary readout requires status/reason")
        else:
            if (type(value) not in (int, float) or not math.isfinite(value) or not -1.01 <= value <= 1.01
                    or r.get("status") != "computed_secondary_uncalibrated" or r.get("reason") is not None):
                raise ValueError("finite uncalibrated raw cosine required")
            good.append(r)
    expected = {"requested_pairs": 8, "image_encoder_calls": len({r["cxr_candidate_id"] for r in good}),
                "text_encoder_calls": len({r["report_candidate_id"] for r in good}),
                "unavailable_reports": 8-len(good)}
    if endpoint.get("counts") != expected: raise ValueError("encoder calls/missingness differ")


def run(args):
    require_slurm()  # Rehashing large checkpoint files must not run on login.
    plan = load(args)
    root = require_inside(args.run, PROTECTED_ROOT, must_exist=True)
    summary = read_json(root/"manifest.json")
    if (summary.get("schema_version") != "tricompose-fixed-image-report-control-v1"
            or summary.get("status") != "completed_same_image_report_control_unvalidated"
            or summary.get("plan_manifest_sha256") != args.plan_manifest_sha256
            or summary.get("fixed_ehr_cases") != 2 or summary.get("fixed_cxr_candidates") != 4
            or summary.get("report_candidates") != 8 or summary.get("new_report_samples") != 4
            or summary.get("new_chexbert_scored_samples") != 4 or summary.get("new_cxr_or_ehr_samples") != 0
            or summary.get("source_charged_single_sample_calls") != 16
            or any(summary.get(k) is not False for k in ("clinical_acceptance", "adaptive_repair_executed",
                "same_image_reports_are_independent_votes", "biovil_used_for_selection", "original_ehr_cxr_or_winners_changed"))):
        raise ValueError("completed, non-adaptive fixed image control required")
    private_modes(root)
    for name, entry in summary["artifacts"].items():
        path = require_inside(root/name, root, must_exist=True)
        if sha256_file(path) != entry["sha256"]: raise ValueError("published artifact changed")
    for name in ("maira", "chexbert"): check_pins(plan[name]["asset_pins"])
    check_pins(plan["biovil_asset_pins"])
    cxrs = load_cxr_candidates(plan["cxr_runs"])
    baseline = load_report_candidates(plan["baseline_report_runs"], cxr_candidates=cxrs)
    new = load_report_candidates([root/"generated"], cxr_candidates=cxrs)
    gm = read_json(root/"generated/manifest.json")
    if (len(cxrs) != 4 or len(baseline) != 4 or len(new) != 4 or gm.get("frozen_model") is not True
            or gm.get("model_id") != "maira2" or gm.get("candidate_count") != 4
            or gm.get("model_audit") != plan["maira"]["model_audit"]
            or gm.get("model_revision") != plan["maira"]["model_revision"]
            or gm.get("source_request_run_manifest_sha256") != sha256_file(Path(plan["request_run"])/"manifest.json")):
        raise ValueError("fixed candidate inventory or MAIRA audit differs")
    rows = []; checkpoint = plan["chexbert"]["checkpoint_sha256"]
    for original in plan["fixed_triplets"]:
        image = next(c for c in cxrs.values() if c["artifact"]["sha256"] == original["cxr_sha256"]
                     and c["case_id"] == original["case_id"] and c["model_id"] == original["cxr_model_id"])
        partial = read_json(original["partial_receipt_path"])
        image_file = Path(original["partial_receipt_path"]).parent/"scored/cxr_finding_labels.json"
        anchor = anchor_from_record(read_json(Path(plan["source_run"])/"cases"/image["case_id"]/"ehr_anchor.json"))
        expected = image_receipt(anchor, image, read_json(image_file), label_sha256=sha256_file(image_file),
            thresholds_sha256=partial["thresholds_sha256"], checkpoint_sha256=partial["classifier_checkpoint_sha256"])
        if partial != expected: raise ValueError("original image receipt arithmetic changed")
        old = next(r for r in baseline.values() if r["parent_cxr_candidate_id"] == image["candidate_id"])
        other = next(r for r in new.values() if r["parent_cxr_candidate_id"] == image["candidate_id"])
        for report, file in ((old, Path(original["baseline_label_path"])),
                             (other, root/"labels/report_finding_labels.json")):
            if report["cost"]["model_calls"] != 1: raise ValueError("hidden report generation calls")
            rows.append(report_row(image, report, partial, read_json(file),
                labels_sha256=sha256_file(file), checkpoint_sha256=checkpoint))
    rows.sort(key=lambda r: r["triple_candidate_id"])
    if read_json(root/"score_rows.json")["records"] != rows: raise ValueError("saved proxy rows differ")
    selection = freeze_selection(rows)
    if (read_json(root/"selection.json") != selection
            or sha256_file(root/"selection.json") != summary["selection_sha256_before_endpoint"]):
        raise ValueError("pre-endpoint choices changed")
    request = read_json(root/"secondary_request/request.json")
    if (request["pairs"] != [{k: r[k] for k in PAIR_FIELDS} for r in rows]
            or request["selection_sha256"] != summary["selection_sha256_before_endpoint"]):
        raise ValueError("endpoint request changed pre-endpoint selection")
    validate_pairs(request["pairs"], cxrs, {**baseline, **new})
    endpoint = read_json(root/"secondary.json"); validate_endpoint(endpoint, rows)
    if endpoint["request_sha256"] != sha256_file(root/"secondary_request/request.json"):
        raise ValueError("endpoint request lineage differs")
    if (read_json(root/"comparison.json") != secondary_comparison(selection, rows, endpoint)
            or (root/"score_table.csv").read_bytes() != csv_table(rows, endpoint).encode()):
        raise ValueError("paired comparison or CSV differs")
    events = [json.loads(line) for line in (root/"cost_journal.jsonl").read_text().splitlines() if line]
    validate_journal(events, summary, endpoint["counts"])
    if Counter(r["case_id"] for r in rows) != Counter({c: 4 for c in {r["case_id"] for r in rows}}):
        raise ValueError("complete per-EHR grid required")
    report = {"schema_version": "tricompose-fixed-image-report-postrun-audit-v1",
        "status": "metadata_hash_selection_audit_passed", "source_run_manifest_sha256": sha256_file(root/"manifest.json"),
        "fixed_ehr_cases": 2, "fixed_cxr_candidates": 4, "report_candidates": 8,
        "source_charged_single_sample_calls": 16, "new_report_generation_samples": 4,
        "new_chexbert_scored_samples": 4, "secondary_counts": endpoint["counts"],
        "private_modes_and_groups_passed": True, "pre_endpoint_selection_recomputed": True,
        "proxy_rows_csv_and_endpoint_pairing_recomputed": True, "durable_cost_journal_passed": True,
        "new_audit_model_calls": 0, "body_or_pixel_review_performed": False,
        "independent_clinical_truth_available": False, "clinical_acceptance": False,
        "clinical_repair_success": False, "gpu_kernel_time_measured": False,
        "full_report_context_policy_retokenized_by_audit": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        file = write_private_json(temporary/"audit_summary.json", report)
        write_private_json(temporary/"manifest.json", {"schema_version": report["schema_version"],
            "source_run": str(root), "source_run_manifest_sha256": report["source_run_manifest_sha256"],
            "audit_program_sha256": sha256_file(__file__), "audit_summary_sha256": sha256_file(file),
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-run", "plan-manifest-sha256", "run", "output-root", "run-id"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007)
    try: root = run(args)
    except Exception as exc:
        print(json.dumps({"status": "audit_failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": "metadata_hash_selection_audit_passed", "new_model_calls": 0,
        "manifest_sha256": sha256_file(root/"manifest.json")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
