#!/usr/bin/env python3
"""Approved GPU Slurm: four MAIRA reports, frozen labels, independent endpoint.

This is an offline fixed-image control, not the single-case adaptive controller.
Batch cost reservations are persisted before each child; interrupted work stays
private and cannot automatically resume or overwrite a previous run.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (WORKSPACE, PROTECTED_ROOT, RUN_ID_PATTERN, require_inside, read_json, sha256_file,
    load_cxr_candidates, load_report_candidates, private_directory, write_private_json, write_private_text)
from tricompose_v12.fixed_image_reports import POLICY, report_row, freeze_selection, secondary_comparison
from tricompose_v12.live_workers import require_gpu_slurm, check_pins, run_private_process, argv_for, CHEXBERT_BERT
from tricompose_v12.invariant_verification import _digest
from tricompose_v11.cxr_contracts import canonical_json_sha256
from prepare_fixed_image_reports import SCHEMA
from score_automatic_replay_biovil import REQUEST_SCHEMA, PAIR_FIELDS


def load(args):
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root/"manifest.json") != args.plan_manifest_sha256:
        raise ValueError("approved plan manifest changed")
    manifest = read_json(root/"manifest.json")
    if manifest["schema_version"] != SCHEMA or sha256_file(root/"plan.json") != manifest["plan_sha256"]:
        raise ValueError("approved plan changed")
    plan = read_json(root/"plan.json")
    if (plan["schema_version"] != SCHEMA or plan["policy"] != POLICY
            or plan["new_model_calls"] != 0 or plan["factory_instantiated"] is not False
            or plan["new_cxr_or_ehr_samples"] != 0 or len(plan["fixed_triplets"]) != 4):
        raise ValueError("unsupported fixed image/report plan")
    check_pins(plan["source_pins"]); check_pins(plan["artifact_pins"])
    return plan


def csv_table(rows, endpoint):
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    columns = ["case_id", "cxr_candidate_id", "report_model_id", "report_candidate_id",
               "proxy_penalty", "missing_positive_image_facts", "biovil_raw_cosine", "biovil_status"]
    names = ("known_reference_facts", "comparable_facts", "supported_positive", "supported_negative",
             "proxy_opposition_facts", "missing_comparisons", "coverage_over_known", "support_over_known")
    columns += [e+"_"+n for e in ("ehr_cxr", "ehr_report", "cxr_report") for n in names]
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=columns); writer.writeheader()
    for row in rows:
        score = scores[row["triple_candidate_id"]]
        record = {k: row[k] for k in columns[:6]}
        record.update(biovil_raw_cosine=score["biovil_raw_cosine"], biovil_status=score["status"])
        for edge, values in row["raw_edge_readouts"].items():
            for name in names: record[edge+"_"+name] = values[name]
        writer.writerow({k: "NA" if v is None else v for k,v in record.items()})
    return out.getvalue()


def normalize_owned_modes(root):
    # Only this newly created run; never checkpoint/source/old-run permissions.
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_gid not in (96293, 65534):
            raise ValueError("private run contains a symlink or non-project group")
        os.chmod(path, 0o2770 if path.is_dir() else 0o660)


def run(args):
    require_gpu_slurm()  # Must precede plan reads, directory creation and children.
    plan = load(args)
    if not RUN_ID_PATTERN.fullmatch(args.run_id): raise ValueError("opaque run ID required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = require_inside(output/args.run_id, PROTECTED_ROOT, must_exist=False)
    # Stable paths: existing adapters embed absolute artifact paths.
    private_directory(root); started = time.monotonic()
    write_private_json(root/"start_manifest.json", {"schema_version": SCHEMA, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256, "clinical_acceptance": False})
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root/"cost_journal.jsonl", flags, 0o660)
    os.fchmod(descriptor, 0o660)
    costs = []
    with os.fdopen(descriptor, "w", encoding="utf-8") as journal:
        def event(record):
            journal.write(json.dumps(record, sort_keys=True)+"\n"); journal.flush(); os.fsync(journal.fileno())

        def child(stage, argv, reserved_units, timeout):
            runtime = root/(stage+"_runtime"); private_directory(runtime)
            event({"stage": stage, "status": "reserved_before_spawn", "maximum_sample_units": reserved_units,
                   "argv_sha256": _digest(argv), "timeout_seconds": timeout})
            beginning = time.monotonic()
            run_private_process(argv, runtime, timeout)
            costs.append({"stage": stage, "wall_seconds_including_startup_io": round(time.monotonic()-beginning, 3),
                          "maximum_sample_units": reserved_units})
            # Process completion alone does not validate artifacts; separate event below.
            event({"stage": stage, "status": "process_completed_unvalidated"})

        try:
            check_pins(plan["maira"]["asset_pins"]); check_pins(plan["chexbert"]["asset_pins"])
            check_pins(plan["biovil_asset_pins"])
            cxrs = load_cxr_candidates(plan["cxr_runs"])
            baseline = load_report_candidates(plan["baseline_report_runs"], cxr_candidates=cxrs)
            if len(cxrs) != 4 or len(baseline) != 4: raise ValueError("fixed baseline inventory differs")
            maira = plan["maira"]
            argv = argv_for(maira, request_run=plan["request_run"], output_root=root)
            child("maira", argv, {"report_generation_samples": 4}, 480)
            new_run = root/"generated"; gm = read_json(new_run/"manifest.json")
            if (gm["frozen_model"] is not True or gm["model_id"] != "maira2" or gm["candidate_count"] != 4
                    or gm["model_revision"] != maira["model_revision"] or gm["model_audit"] != maira["model_audit"]
                    or gm["source_request_run_manifest_sha256"] != sha256_file(Path(plan["request_run"])/"manifest.json")):
                raise ValueError("four-report frozen MAIRA audit differs")
            new = load_report_candidates([new_run], cxr_candidates=cxrs)
            rm = read_json(Path(plan["request_run"])/"manifest.json")
            request_hashes = {}
            for r in rm["requests"]:
                path = Path(plan["request_run"])/r["path"]
                if sha256_file(path) != r["sha256"]: raise ValueError("fixed image request changed")
                request_hashes[r["parent_cxr_candidate_id"]] = canonical_json_sha256(read_json(path))
            if len(new) != 4 or {r["parent_cxr_candidate_id"] for r in new.values()} != set(cxrs):
                raise ValueError("exactly one MAIRA report per fixed image required")
            for report in new.values():
                if (report["input_report_request_sha256"] != request_hashes[report["parent_cxr_candidate_id"]]
                        or report["cost"]["model_calls"] != 1):
                    raise ValueError("MAIRA request/cost lineage differs")
            event({"stage": "maira", "status": "validated", "report_generation_samples": 4})
            chex = plan["chexbert"]
            argv = [chex["python"], chex["script"], "--report-run", str(new_run),
                "--checkpoint", chex["checkpoint"], "--bert-path", str(CHEXBERT_BERT), "--batch-size", "4",
                "--output-root", str(root), "--run-id", "labels"]
            for path in plan["cxr_runs"]: argv += ["--cxr-run", path]
            child("chexbert", argv, {"chexbert_scored_samples": 4}, 180)
            label_path = root/"labels/report_finding_labels.json"; labels = read_json(label_path)
            if (labels["counts"] != {"reports": 4, "model_calls": 4}
                    or {r["report_candidate_id"] for r in labels["records"]} != set(new)):
                raise ValueError("new report label inventory differs")
            rows = []
            for original in plan["fixed_triplets"]:
                image = next(c for c in cxrs.values() if c["artifact"]["sha256"] == original["cxr_sha256"]
                             and c["case_id"] == original["case_id"] and c["model_id"] == original["cxr_model_id"])
                partial = read_json(original["partial_receipt_path"])
                old = next(r for r in baseline.values() if r["parent_cxr_candidate_id"] == image["candidate_id"])
                other = next(r for r in new.values() if r["parent_cxr_candidate_id"] == image["candidate_id"])
                for report, lp in ((old, Path(original["baseline_label_path"])), (other, label_path)):
                    rows.append(report_row(image, report, partial, read_json(lp),
                        labels_sha256=sha256_file(lp), checkpoint_sha256=chex["checkpoint_sha256"]))
            rows.sort(key=lambda r: r["triple_candidate_id"])
            selection = freeze_selection(rows)
            write_private_json(root/"score_rows.json", {"records": rows})
            selection_path = write_private_json(root/"selection.json", selection)
            selection_sha = sha256_file(selection_path)
            event({"stage": "chexbert", "status": "validated", "chexbert_scored_samples": 4})
            # Freeze selection before any independent endpoint inference.
            request_dir = root/"secondary_request"; private_directory(request_dir)
            request = {"schema_version": REQUEST_SCHEMA, "pairs": [{k: r[k] for k in PAIR_FIELDS} for r in rows],
                "experiment": "fixed_image_two_report_control_not_automatic_replay",
                "selection_sha256": selection_sha, "modality_source": "fully_synthetic",
                "selection_used_biovil": False, "clinical_truth_available": False,
                "routing_or_calibration_update_allowed": False,
                "text_policy": "full_report_no_silent_truncation_overlength_is_na"}
            rp = write_private_json(request_dir/"request.json", request)
            write_private_json(request_dir/"manifest.json", {"schema_version": REQUEST_SCHEMA,
                "artifacts": {"request.json": {"sha256": sha256_file(rp)}}})
            all_runs = plan["baseline_report_runs"]+[str(new_run)]
            argv = [plan["biovil_python"], str(Path(__file__)), "biovil-worker",
                "--request-run", str(request_dir), "--model-path", plan["biovil_model"],
                "--output-file", str(root/"secondary.json")]
            for path in plan["cxr_runs"]: argv += ["--cxr-run", path]
            for path in all_runs: argv += ["--report-run", path]
            child("biovil", argv, {"image_encoder_samples": 4, "text_encoder_samples": 8}, 180)
            endpoint = read_json(root/"secondary.json")
            if (sha256_file(selection_path) != selection_sha or endpoint["request_sha256"] != sha256_file(rp)):
                raise ValueError("frozen selection or endpoint request changed")
            comparison = secondary_comparison(selection, rows, endpoint)
            event({"stage": "biovil", "status": "validated", "counts": endpoint["counts"]})
            write_private_json(root/"comparison.json", comparison)
            write_private_text(root/"score_table.csv", csv_table(rows, endpoint))
            check_pins(plan["source_pins"]); check_pins(plan["artifact_pins"])
            check_pins(maira["asset_pins"]); check_pins(chex["asset_pins"]); check_pins(plan["biovil_asset_pins"])
            journal.flush(); os.fsync(journal.fileno())
            files = ("score_rows.json", "selection.json", "secondary.json", "comparison.json", "score_table.csv", "cost_journal.jsonl")
            summary = {"schema_version": "tricompose-fixed-image-report-control-v1",
                "status": "completed_same_image_report_control_unvalidated", "fixed_ehr_cases": 2,
                "fixed_cxr_candidates": 4, "report_candidates": 8, "new_report_samples": 4,
                "new_chexbert_scored_samples": 4, "new_cxr_or_ehr_samples": 0,
                "source_charged_single_sample_calls": 16, "secondary_counts": endpoint["counts"],
                "batch_cost_accounting_not_single_case_router_ledger": True,
                "worker_costs": costs, "wall_seconds_including_startup_io": round(time.monotonic()-started, 3),
                "plan_manifest_sha256": args.plan_manifest_sha256, "selection_sha256_before_endpoint": selection_sha,
                "artifacts": {n: {"sha256": sha256_file(root/n)} for n in files},
                "clinical_acceptance": False, "adaptive_repair_executed": False, "gpu_seconds": None,
                "same_image_reports_are_independent_votes": False, "biovil_used_for_selection": False,
                "original_ehr_cxr_or_winners_changed": False}
            write_private_json(root/"manifest.json", summary)
        except Exception as exc:
            event({"status": "failed_or_interrupted_retained", "error_type": type(exc).__name__, "automatic_resume": False})
            write_private_json(root/"failure_manifest.json", {"status": "failed_or_interrupted_retained",
                "error_type": type(exc).__name__, "clinical_acceptance": False, "automatic_resume": False})
            raise
        finally:
            normalize_owned_modes(root)
    return root, summary


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="mode", required=True)
    launch = sub.add_parser("run")
    for name in ("plan-run", "plan-manifest-sha256", "output-root", "run-id"):
        launch.add_argument("--"+name, required=True)
    worker = sub.add_parser("biovil-worker")
    for name in ("request-run", "model-path", "output-file"): worker.add_argument("--"+name, required=True)
    for name in ("cxr-run", "report-run"): worker.add_argument("--"+name, action="append", required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        if args.mode == "biovil-worker":
            require_gpu_slurm()
            sys.path.insert(0, str(WORKSPACE/"runtime/vendor/hi_ml_multimodal_0_2_2"))
            from score_automatic_replay_biovil import score
            target = require_inside(args.output_file, PROTECTED_ROOT, must_exist=False)
            write_private_json(target, score(args)); return 0
        root, result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": result["status"], "runtime_seconds": result["wall_seconds_including_startup_io"],
        "manifest_sha256": sha256_file(root/"manifest.json")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
