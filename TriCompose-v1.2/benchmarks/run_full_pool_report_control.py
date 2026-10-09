#!/usr/bin/env python3
"""Approved GPU Slurm: fresh full-pool XRV/CheXbert, sealed report choice.

No generators or API. Historical BioViL values are consumed after choice only.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (PROTECTED_ROOT, RUN_ID_PATTERN, require_inside, read_json, sha256_file,
    load_cxr_candidates, load_report_candidates, private_directory, write_private_json, write_private_text)
from compare_frozen_report_paths import flatten
from run_report_nbest import single_view
from run_fixed_image_reports import normalize_owned_modes
from run_automatic_proxy_replay import render_csv
from tricompose_v12.full_pool_report_control import SCHEMA, POLICY, freeze, summarize, PAIR_FIELDS
from tricompose_v12.live_workers import require_gpu_slurm, check_pins, run_private_process, CHEXBERT_BERT
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.invariant_verification import _digest


def load(args):
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / "manifest.json") != args.plan_manifest_sha256:
        raise ValueError("approved full-pool plan manifest changed")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA or sha256_file(root / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("approved full-pool plan body changed")
    plan = read_json(root / "plan.json")
    if (plan.get("schema_version") != SCHEMA or plan.get("policy") != POLICY
            or any(plan.get(k) is not False for k in ("factory_instantiated", "source_bodies_parsed",
                "secondary_values_read", "legacy_image_or_report_labels_reused", "clinical_acceptance"))
            or plan.get("new_model_calls") != 0 or len(plan["anchors"]) != 80 or len(plan["pairs"]) != 960
            or [r["opaque_source_index"] for r in plan["anchors"]] != list(range(80))
            or len(plan["cxr_runs"]) != 3 or len(plan["report_runs"]) != 8
            or plan["runtime_limits"] != {"xrv_seconds": 420, "chexbert_seconds": 420, "chexbert_batch_size": 16}
            or plan["planned_counts"] != {"fixed_ehr_cases": 80, "fixed_images": 240, "report_candidates": 960,
                "new_xrv_scored_images": 240, "new_chexbert_scored_reports": 960,
                "reused_secondary_pairs": 960, "new_generation_calls": 0, "new_biovil_encodings": 0}):
        raise ValueError("exact frozen full-pool scope required")
    for case in plan["anchors"]:
        if anchor_from_record(case["anchor"]).sha256 != case["ehr_anchor_sha256"]:
            raise ValueError("fixed case anchor changed")
    check_pins(plan["source_pins"])
    check_pins(plan["artifact_pins"])
    return plan


def build_rows(plan, cxrs, reports, image_labels, report_labels, structures, *, image_labels_sha256, report_labels_sha256):
    """Pure metadata recomputation, shared by runtime and subsequent CPU audit."""
    if (len(cxrs) != 240 or len(reports) != 960
            or image_labels["counts"] != {"cxr_candidates": 240, "model_calls": 240}
            or report_labels["counts"] != {"reports": 960, "model_calls": 960}
            or set(structures) != set(reports)):
        raise ValueError("complete fresh label/structure inventory required")
    anchors = {r["anchor"]["case_id"]: anchor_from_record(r["anchor"]) for r in plan["anchors"]}
    partials, rows = {}, []
    for image in cxrs.values():
        anchor = anchors[image["case_id"]]
        iv = single_view(image_labels, "cxr_candidates", "cxr_candidate_id", image["candidate_id"])
        partials[image["candidate_id"]] = image_receipt(anchor, image, iv, label_sha256=image_labels_sha256,
            thresholds_sha256=plan["xrv"]["thresholds_sha256"], checkpoint_sha256=plan["xrv"]["checkpoint_sha256"])
    for pair in plan["pairs"]:
        image = cxrs[pair["cxr_candidate_id"]]
        report = reports[pair["report_candidate_id"]]
        anchor = anchors[pair["case_id"]]
        if (image["case_id"] != pair["case_id"] or report["case_id"] != pair["case_id"]
                or image["artifact"]["sha256"] != pair["cxr_sha256"]
                or report["artifact"]["sha256"] != pair["report_sha256"]
                or pair["ehr_sha256"] != anchor.ehr_sha256 or pair["ehr_facts_sha256"] != anchor.ehr_facts_sha256):
            raise ValueError("immutable original pair/EHR changed")
        receipt = completed_receipt(anchor, partials[image["candidate_id"]], image,
            single_view(image_labels, "cxr_candidates", "cxr_candidate_id", image["candidate_id"]), report,
            single_view(report_labels, "reports", "report_candidate_id", report["candidate_id"]),
            image_labels_sha256=image_labels_sha256, report_labels_sha256=report_labels_sha256,
            thresholds_sha256=plan["xrv"]["thresholds_sha256"],
            xrv_checkpoint_sha256=plan["xrv"]["checkpoint_sha256"], chexbert_checkpoint_sha256=plan["chexbert"]["checkpoint_sha256"])
        rows.append({**pair, "cxr_model_id": image["model_id"], "report_model_id": report["model_id"],
            "seed": image["seed"], "receipt": receipt, "structure": structures[report["candidate_id"]],
            "raw_edge_readouts": receipt["raw_edge_readouts"]})
    rows.sort(key=lambda r: r["triple_candidate_id"])
    return rows, [partials[i] for i in sorted(partials)]


def table_rows(rows, endpoint):
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    result = []
    for row in rows:
        score = scores[row["triple_candidate_id"]]
        result.append(flatten({k: v for k, v in row.items() if k not in ("receipt", "structure")}) | {
            "biovil_raw_cosine": score["biovil_raw_cosine"], "biovil_status": score["status"], "biovil_na_reason": score["reason"],
            "official_section_contract_pass": row["structure"]["section_contract_pass"],
            "unsupported_temporal_language": row["structure"]["unsupported_temporal_comparison_language"],
            "generic_report": row["structure"]["generic_report"]})
    return result


def run(args):
    require_gpu_slurm()
    plan = load(args)
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("opaque run ID required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True)
    root = require_inside(output / args.run_id, PROTECTED_ROOT, must_exist=False)
    private_directory(root)
    started, costs = time.monotonic(), []
    write_private_json(root / "start_manifest.json", {"schema_version": SCHEMA, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256, "new_generation_calls": 0, "clinical_acceptance": False})
    fd = os.open(root / "cost_journal.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o660)
    os.fchmod(fd, 0o660)
    with os.fdopen(fd, "w", encoding="utf-8") as journal:
        def event(value):
            journal.write(json.dumps(value, sort_keys=True) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        def child(stage, argv, units, timeout):
            event({"stage": stage, "status": "reserved_before_spawn", "maximum_sample_units": units,
                "argv_sha256": _digest(argv), "timeout_seconds": timeout})
            runtime = root / (stage + "_runtime")
            private_directory(runtime)
            began = time.monotonic()
            run_private_process(argv, runtime, timeout)
            costs.append({"stage": stage, "wall_seconds_including_startup_io": time.monotonic() - began,
                "maximum_sample_units": units})
            event({"stage": stage, "status": "process_completed_unvalidated"})
        try:
            for spec in (plan["xrv"], plan["chexbert"]):
                check_pins(spec["asset_pins"])
            cxrs = load_cxr_candidates(plan["cxr_runs"])
            reports = load_report_candidates(plan["report_runs"], cxr_candidates=cxrs)
            xrv = plan["xrv"]
            checkpoint = Path(xrv["checkpoint"])
            argv = [xrv["python"], xrv["script"], "--cache-dir", str(checkpoint.parent),
                "--weight-filename", checkpoint.name, "--model-name", "densenet121-res224-all",
                "--thresholds", xrv["thresholds_path"], "--output-root", str(root), "--run-id", "image_labels"]
            for path in plan["cxr_runs"]:
                argv += ["--cxr-run", path]
            child("xrv", argv, {"xrv_scored_images": 240}, plan["runtime_limits"]["xrv_seconds"])
            cb = plan["chexbert"]
            argv = [cb["python"], cb["script"], "--checkpoint", cb["checkpoint"], "--bert-path", str(CHEXBERT_BERT),
                "--batch-size", "16", "--output-root", str(root), "--run-id", "report_labels"]
            for path in plan["cxr_runs"]:
                argv += ["--cxr-run", path]
            for path in plan["report_runs"]:
                argv += ["--report-run", path]
            child("chexbert", argv, {"chexbert_scored_reports": 960}, plan["runtime_limits"]["chexbert_seconds"])
            ip = root / "image_labels/cxr_finding_labels.json"
            tp = root / "report_labels/report_finding_labels.json"
            il, tl = read_json(ip), read_json(tp)
            # Synthetic body inspection is internal to this explicitly approved
            # GPU controller; the CPU staging/audit never opens report text.
            from evaluate_report_structure import _record
            structures = {rid: _record(r, cxrs[r["parent_cxr_candidate_id"]], normalized_frequency={}) for rid, r in reports.items()}
            frequency = Counter(s["normalized_report_sha256"] for s in structures.values())
            for structure in structures.values():
                structure["normalized_template_frequency"] = frequency[structure["normalized_report_sha256"]]
            rows, partials = build_rows(plan, cxrs, reports, il, tl, structures,
                image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp))
            write_private_json(root / "image_receipts.json", {"records": partials})
            write_private_json(root / "structure_records.json", {"records": [structures[k] for k in sorted(structures)]})
            write_private_json(root / "score_rows.json", {"records": rows})
            selection = freeze(rows)
            sp = write_private_json(root / "selection.json", selection)
            selection_sha = sha256_file(sp)
            event({"stage": "choice", "status": "sealed_before_secondary_values",
                "selection_sha256": selection_sha, "endpoint_used_for_selection": False})
            # No BioViL model loading; exact same original IDs, hashes and full-
            # report semantics. This previously inspected pool is NOT a holdout.
            if sha256_file(plan["cached_secondary_path"]) != plan["cached_secondary_sha256"]:
                raise ValueError("cached secondary artifact changed")
            cached = read_json(plan["cached_secondary_path"])
            endpoint = {"records": cached["records"], "used_for_routing": False, "clinical_truth_available": False,
                "historical_pool_is_untouched_test": False, "new_biovil_encodings": 0,
                "source_sha256": plan["cached_secondary_sha256"], "selection_sha256": selection_sha}
            comparison = summarize(rows, selection, endpoint)
            if sha256_file(sp) != selection_sha:
                raise ValueError("sealed choice changed during measurement")
            write_private_json(root / "secondary.json", endpoint)
            write_private_json(root / "comparison.json", comparison)
            write_private_json(root / "summary.json", comparison["summary"])
            show = lambda values: render_csv([{k: "NA" if v is None else v for k, v in r.items()} for r in values])
            write_private_text(root / "score_table.csv", show(table_rows(rows, endpoint)))
            write_private_text(root / "model_comparison.csv", show([flatten(r) for r in comparison["model_comparison"]]))
            write_private_text(root / "case_outcomes.csv", show(comparison["case_outcomes"]))
            event({"stage": "secondary", "status": "cached_measurement_validated",
                "cached_pairs": 960, "new_biovil_encodings": 0})
            check_pins(plan["source_pins"])
            check_pins(plan["artifact_pins"])
            for spec in (plan["xrv"], plan["chexbert"]):
                check_pins(spec["asset_pins"])
            journal.flush()
            os.fsync(journal.fileno())
            files = ("image_receipts.json", "structure_records.json", "score_rows.json", "selection.json", "secondary.json",
                "comparison.json", "summary.json", "score_table.csv", "model_comparison.csv", "case_outcomes.csv", "cost_journal.jsonl")
            result = {"schema_version": SCHEMA, "status": "completed_full_pool_fresh_report_control_unvalidated",
                "plan_manifest_sha256": args.plan_manifest_sha256,
                "artifacts": {name: {"sha256": sha256_file(root / name)} for name in files},
                "selection_sha256_before_secondary": selection_sha,
                "new_generation_calls": 0, "new_xrv_scored_images": 240, "new_chexbert_scored_reports": 960,
                "requested_chexbert_batch_size": 16, "legacy_model_calls_are_sample_counts": True,
                "new_biovil_encodings": 0, "reused_secondary_pairs": 960,
                "xrv_labels_sha256": sha256_file(ip), "chexbert_labels_sha256": sha256_file(tp),
                "wall_seconds_including_startup_io": round(time.monotonic() - started, 3), "worker_costs": costs,
                "peak_vram_gib": {"xrv": il["peak_vram_gib"], "chexbert": tl["peak_vram_gib"]}, "gpu_seconds": None,
                "clinical_acceptance": False, "clinical_repair_success": False, "adaptive_repair_executed": False,
                "original_winners_changed": False, "biovil_used_for_selection": False,
                "historical_pool_is_untouched_test": False,
                "historical_generation_and_secondary_costs_are_not_zero": True}
            write_private_json(root / "manifest.json", result)
        except BaseException as exc:
            event({"status": "failed_or_interrupted_retained", "error_type": type(exc).__name__, "automatic_resume": False})
            write_private_json(root / "failure_manifest.json", {"status": "failed_or_interrupted_retained",
                "error_type": type(exc).__name__, "automatic_resume": False, "clinical_acceptance": False})
            raise
        finally:
            normalize_owned_modes(root)
    return root, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-run", "plan-manifest-sha256", "output-root", "run-id"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    os.umask(0o007)
    try:
        root, result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": result["status"], "runtime_seconds": result["wall_seconds_including_startup_io"],
        "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
