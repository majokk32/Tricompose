#!/usr/bin/env python3
"""CPU metadata/byte-hash replay, not clinical image/report adjudication.

Never parse EHR/prompt/report bodies, decode pixels, load weights, retry a job,
replace selections or change the already consumed generation/score contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import run_cxr_action_diversification as worker
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    sha256_file, write_private_json)
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_execution import validate_cxr_binding, validate_report_binding
from tricompose_v12.live_receipts import image_receipt, completed_receipt

VERSION = "tricompose-fixed-ehr-cxr-actions-metadata-postflight-v1"
require = worker.require
ARTIFACTS = {"summary.json", "completed_triplets.json", "slot_outcomes.json",
    "execution_summary.json", "score_rows.json", "paired_proxy_comparison.json",
    "score_table.csv", "start_manifest.json"}


def validate_totals(summary, triples, slots, books):
    """Keep four planned slots, all charges, unavailable answers and NA claims."""
    expected = {"fixed_ehr_cases": 2, "planned_image_slots": 4,
        "completed_triplets": len(triples), "charged_worker_requests": sum(b["charged_model_attempts"] for b in books),
        "validated_worker_calls": sum(b["completed_operations"] for b in books),
        "failed_worker_requests": sum(b["failed_attempts"] for b in books), "retries": 0,
        "new_planner_calls": 0, "external_api_calls": 0, "new_clinically_accepted_repairs": 0}
    require(len(books) == 2 and len(slots) == 4 and len(triples) <= 4
        and expected["charged_worker_requests"] <= 16
        and all(type(summary[k]) is int and summary[k] == v for k, v in expected.items()),
        "exact_four_slot_bounded_counts_required")
    require(sum(s["charged_worker_requests"] for s in slots) == expected["charged_worker_requests"]
        and sum(s["validated_worker_calls"] for s in slots) == expected["validated_worker_calls"]
        and sum(s["triple"] is not None for s in slots) == len(triples)
        and summary["slot_status_counts"] == dict(Counter(s["status"] for s in slots)),
        "no_dropped_slots_or_hidden_charges")
    status = "completed_candidate_expansion_unvalidated" if len(triples) == 4 else "partial_candidate_expansion_unvalidated"
    require(summary["schema_version"] == worker.VERSION and summary["status"] == status
        and all(summary[k] is False for k in ("training_performed", "clinical_acceptance", "selection_changed",
            "independent_endpoint_evaluated"))
        and all(summary[k] is None for k in ("clinical_accuracy", "measured_saved_model_calls"))
        and summary["same_call_budget_not_same_gpu_seconds"] is True
        and summary["historical_cost_is_shared_sunk_not_free"] is True,
        "unsupported_clinical_selection_or_cost_claim")
    wall = summary["runtime_seconds_includes_startup_and_io"]
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0,
        "finite_measured_wall_time_required")
    return expected


def candidate(reader, root, spec, request_root):
    """Read only candidate/manifest metadata; artifact bodies are byte-hashed."""
    root = reader.path(Path(root) / "manifest.json").parent
    manifest = reader.json(root / "manifest.json")
    require(manifest["frozen_model"] is True and manifest["model_id"] == spec["model_id"]
        and manifest["model_revision"] == spec["model_revision"] and manifest["model_audit"] == spec["model_audit"]
        and manifest["candidate_count"] == 1 and len(manifest["candidates"]) == 1
        and manifest["source_request_run_manifest_sha256"] == reader.hash(Path(request_root) / "manifest.json"),
        "same_single_frozen_worker_and_request_required")
    item = manifest["candidates"][0]
    path = reader.path(root / item["path"]); reader.hash(path, item["sha256"])
    result = reader.json(path)
    require(result["candidate_id"] == item["candidate_id"] and result["case_id"] == item["case_id"]
        and result["model_id"] == spec["model_id"] and result["frozen_model"] is True
        and result["cost"]["model_calls"] == 1, "single_candidate_identity_required")
    reader.hash(result["artifact"]["path"], result["artifact"]["sha256"])
    return result


def validate_triple(reader, case, triple, row, book, plan):
    anchor = worker.existing.gate.anchor_from_record(case["anchor"])
    keys = ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256",
        "cxr_model_id", "report_model_id", "seed", "raw_edge_readouts")
    require(all(triple[k] == row[k] for k in keys) and triple["clinical_acceptance"] is False
        and triple["selected_as_best"] is False and triple["ehr_anchor_sha256"] == anchor.sha256,
        "unpromoted_same_anchor_and_score_row_required")
    ctx = worker.existing.gate.context(case["anchor"], plan["workers"]["xrv"], plan["workers"]["chexbert"])
    worker.existing.gate.validate_observation(row, ctx)
    worker.existing.gate.completed_operation(row, book)
    slot = worker.SLOTS.index((triple["cxr_model_id"], triple["seed"]))
    cr = reader.root / "cases" / case["case_id"]
    ops = cr / "operations"
    require(Path(triple["ehr_path"]) == cr / "inputs/synthetic_ehr.json"
        and Path(triple["cxr_run"]) == ops / f"cxr_{slot}_a1/generated"
        and Path(triple["report_run"]) == ops / f"report_{slot}_0_a1/generated"
        and Path(triple["receipt_path"]) == ops / f"chexbert_{slot}_0_a1/completed_receipt.json",
        "exact_own_branch_paths_required")
    original = case["requests"][slot]
    image_pack, report_pack = ops / f"request_cxr_{slot}", ops / f"request_report_{slot}_0"
    image_manifest = reader.json(image_pack / "manifest.json")
    require(image_manifest["request_count"] == 1 and len(image_manifest["requests"]) == 1,
        "single_original_cxr_request_required")
    entry = image_manifest["requests"][0]
    rp = image_pack / entry["path"]; reader.hash(rp, entry["sha256"])
    require(reader.json(rp) == original["request"], "unchanged_final_prompt_request_required")
    image = candidate(reader, triple["cxr_run"], plan["workers"][triple["cxr_model_id"]], image_pack)
    report = candidate(reader, triple["report_run"], plan["workers"]["cxrmate_single"], report_pack)
    validate_cxr_binding(image, original, anchor)
    rm = reader.json(report_pack / "manifest.json")
    require(rm["request_count"] == 1 and len(rm["requests"]) == 1, "single_own_image_report_request_required")
    rr = rm["requests"][0]; reader.hash(report_pack / rr["path"], rr["sha256"])
    validate_report_binding(report, image, report_pack, anchor)
    require(image["candidate_id"] == row["cxr_candidate_id"] and report["candidate_id"] == row["report_candidate_id"]
        and image["artifact"]["path"] == triple["cxr_path"] and report["artifact"]["path"] == triple["report_path"],
        "candidate_artifact_pointer_binding_required")
    xp, cp = ops / f"xrv_{slot}_a1", ops / f"chexbert_{slot}_0_a1"
    receipt = row["receipt"]
    reader.hash(xp / "scored/cxr_finding_labels.json", receipt["xrv_labels_sha256"])
    reader.hash(cp / "scored/report_finding_labels.json", receipt["chexbert_labels_sha256"])
    xl = reader.json(xp / "scored/cxr_finding_labels.json")
    cl = reader.json(cp / "scored/report_finding_labels.json")
    xspec, cspec = plan["workers"]["xrv"], plan["workers"]["chexbert"]
    require(xl["thresholds"] == xspec["thresholds"]
        and xl["calibration"]["status"] == xspec["calibration_status"]
        and all(xl["producer"].get(k) == v for k, v in xspec["scorer_provenance"].items()),
        "unchanged_xrv_thresholds_and_preprocessing_required")
    partial = image_receipt(anchor, image, xl, label_sha256=receipt["xrv_labels_sha256"],
        thresholds_sha256=xspec["thresholds_sha256"], checkpoint_sha256=xspec["checkpoint_sha256"])
    require(reader.json(xp / "image_receipt.json") == partial, "image_receipt_numeric_replay_required")
    rebuilt = completed_receipt(anchor, partial, image, xl, report, cl,
        image_labels_sha256=receipt["xrv_labels_sha256"], report_labels_sha256=receipt["chexbert_labels_sha256"],
        thresholds_sha256=xspec["thresholds_sha256"], xrv_checkpoint_sha256=xspec["checkpoint_sha256"],
        chexbert_checkpoint_sha256=cspec["checkpoint_sha256"])
    reader.hash(triple["receipt_path"], triple["receipt_sha256"])
    require(reader.json(triple["receipt_path"]) == rebuilt == receipt
        and triple["receipt_id"] == receipt["receipt_id"], "completed_receipt_numeric_replay_required")


def audit(args):
    worker.existing.gate.cpu_guard()  # Before any source read or directory write.
    plan = worker.load_plan(args, weights=False)
    reader = worker.Reader(args.source_run)
    reader.hash(reader.root / "manifest.json", args.source_manifest_sha256)
    manifest = reader.json(reader.root / "manifest.json")
    require(manifest["schema_version"] == worker.VERSION and set(manifest["artifacts"]) == ARTIFACTS
        and manifest["plan_manifest_sha256"] == args.plan_manifest_sha256
        and manifest["clinical_acceptance"] is False and manifest["selection_changed"] is False
        and not (reader.root / "failure.json").exists(), "sealed_nonfailed_exact_run_required")
    for name, pin in manifest["artifacts"].items(): reader.hash(reader.root / name, pin["sha256"])
    start = reader.json(reader.root / "start_manifest.json")
    require(start["plan_manifest_sha256"] == args.plan_manifest_sha256 and start["slurm_job_id"] == args.job_id
        and start["automatic_resume"] is False and start["clinical_acceptance"] is False,
        "approved_job_and_plan_binding_required")
    triples = reader.json(reader.root / "completed_triplets.json")["records"]
    rows = reader.json(reader.root / "score_rows.json")["records"]
    books = reader.json(reader.root / "execution_summary.json")["case_ledgers"]
    pairs = reader.json(reader.root / "paired_proxy_comparison.json")["records"]
    summary = reader.json(reader.root / "summary.json")
    require([b["case_id"] for b in books] == [c["case_id"] for c in plan["cases"]]
        and len(rows) == len(triples) and len({r["triple_candidate_id"] for r in rows}) == len(rows),
        "same_fixed_case_order_and_unique_rows_required")
    slots = []; expected_pairs = []
    for case, book in zip(plan["cases"], books):
        case_triples = [t for t in triples if t["case_id"] == case["case_id"]]
        slots.extend(worker.slot_outcomes(case, case_triples, book))
        require(book["execution_mode"] == "approved_slurm_backend", "actual_worker_execution_ledger_required")
        cr = reader.root / "cases" / case["case_id"]
        require(reader.json(cr / "ledger_snapshot.json") == book
            and reader.journal(cr / "execution.journal.jsonl") == book["events"], "durable_journal_and_snapshot_replay_required")
        anchor = worker.existing.gate.anchor_from_record(case["anchor"])
        require(reader.json(cr / "ehr_anchor.json") == anchor.record(), "fixed_case_anchor_required")
        reader.hash(cr / "inputs/synthetic_ehr.json", anchor.ehr_sha256)
        reader.hash(cr / "inputs/ehr_facts.json", anchor.ehr_facts_sha256)
        for req in case["requests"]:
            reader.hash(cr / "inputs/cxr_prompts" / (req["request"]["model_id"] + ".txt"),
                req["request"]["inputs"]["final_prompt"]["sha256"])
        for event in book["events"]:
            if event["event"] != "attempt_reserved": continue
            request = event["request"]; spec = plan["workers"][request["model_id"]]
            require(request["frozen_model_audit_sha256"] == _digest(spec)
                and request["kind"] == spec["kind"], "exact_frozen_worker_receipt_identity_required")
        case_manifest = reader.json(cr / "case_manifest.json")
        require(case_manifest["completed_triplets"] == case_triples
            and case_manifest["charged_model_attempts"] == book["charged_model_attempts"], "case_export_inventory_required")
        for triple in case_triples:
            matched = [r for r in rows if (r["case_id"], r["cxr_model_id"], r["seed"])
                == (triple["case_id"], triple["cxr_model_id"], triple["seed"])]
            require(len(matched) == 1, "one_score_row_per_completed_branch_required")
            row = matched[0]; validate_triple(reader, case, triple, row, book, plan)
            expected_pairs.append({"case_id": case["case_id"], "cxr_model_id": row["cxr_model_id"], "seed": row["seed"],
                "baseline_row": case["baseline_row"], "new_row": row,
                "frozen_proxy_preservation_comparison": worker.compare(case["baseline_row"], row),
                "diagnostic_only_no_selection": True, "clinical_acceptance": False})
    require(reader.json(reader.root / "slot_outcomes.json")["records"] == slots and pairs == expected_pairs,
        "four_slot_inventory_and_comparison_exact_replay_required")
    totals = validate_totals(summary, triples, slots, books)
    require(manifest["status"] == summary["status"], "manifest_and_summary_status_required")
    reader.hash(reader.root / "score_table.csv", hashlib.sha256(worker.score_csv(triples).encode()).hexdigest())
    reader.recheck(); worker.load_plan(args, weights=False)
    result = {"schema_version": VERSION, "status": "completed_metadata_replay_not_clinical",
        "source_manifest_sha256": args.source_manifest_sha256, "plan_manifest_sha256": args.plan_manifest_sha256,
        "job_id": args.job_id, "totals": totals, "slot_status_counts": summary["slot_status_counts"],
        "metadata_hashes_checked": len(reader.pins), "source_hashes_checked": len(plan["source_pins"]),
        "input_hashes_checked": len(plan["artifact_pins"]), "model_asset_stats_checked": sum(len(w["asset_stats"]) for w in plan["workers"].values()),
        "ledger_and_numeric_receipts_replayed": True, "four_slots_retained": True,
        "csv_byte_hash_replayed": True, "protected_permissions_checked": True,
        "report_structure_independently_recomputed": False, "tokenizer_inputs_independently_recomputed": False,
        "source_bodies_parsed": False, "image_pixels_decoded": False, "new_model_calls": 0,
        "clinical_acceptance": False, "clinical_accuracy": None, "selection_changed": False,
        "reader_pins": reader.pins, "auditor_source_sha256": sha256_file(__file__)}
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(tmp / "audit.json", result)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION, "status": result["status"],
            "audit_sha256": sha256_file(path), "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(tmp, target)
    except BaseException: discard_atomic_run(tmp); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-manifest-sha256", "plan-manifest-sha256", "job-id", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        path = audit(args)
        print(json.dumps({"stage": VERSION, "status": "completed_cpu_only", "manifest_sha256": sha256_file(path / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
