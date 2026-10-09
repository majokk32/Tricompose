#!/usr/bin/env python3
"""CPU metadata postflight of fresh-observer-before-dispatch ordering.

Replays the sealed decision algebra and charged journals, not image inference.
Never reads EHR/report bodies, decodes pixels, starts a backend, changes a winner
or claims that agreement/abstention establishes clinical correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import run_guarded_image_probe as worker
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    sha256_file, write_private_json)
from tricompose_v12.invariant_verification import _digest

VERSION = "tricompose-guarded-image-execution-postflight-v1"
STATUS = "completed_fresh_verification_before_dispatch_unvalidated"
ARTIFACTS = {"predictions.json", "dispatch_results.json", "pending_requests.json", "summary.json"}
require = worker.require


def verify_payload(plan, cases, predictions, observer_journal, dispatch_payload,
                   pending, dispatch_journal, summary, predictions_sha256):
    """Pure replay; observations supplied by the sealed run, never invented."""
    require(plan["schema_version"] == worker.VERSION and plan["config"] == worker.CONFIG
        and plan["selection_change_allowed"] is False, "original_bounded_plan_required")
    require(predictions["schema_version"] == worker.VERSION and predictions["image_only"] is True
        and predictions["frozen"] is True
        and predictions["model_received_ehr_reports_ids_or_scores"] is False,
        "sealed_blind_frozen_predictions_required")
    images, records = plan["image_inputs"], predictions["records"]
    require(len(images) == len(records) == 2
        and len({r["cxr_sha256"] for r in records}) == 2,
        "exactly_two_distinct_observed_images_required")
    expected_observer = [{"event": "observer_load_attempt_reserved", "charged_load_attempts": 1}]
    for ordinal, (image, record) in enumerate(zip(images, records)):
        require(all(image[k] == record[k] for k in ("cxr_candidate_id", "cxr_sha256")),
            "observer_order_or_fixed_image_binding_changed")
        expected_observer.extend((
            {"event": "call_reserved", "ordinal": ordinal, "cxr_sha256": record["cxr_sha256"]},
            {"event": "call_finished", "ordinal": ordinal, "contract_status": record["contract_status"]}))
    require(_digest(observer_journal) == _digest(expected_observer),
        "charged_observer_reservations_or_completion_changed")
    expected_dispatch = [{"event": "fresh_observer_predictions_sealed", "sha256": predictions_sha256}]
    results = worker.dispatch_cases(cases, records, expected_dispatch.append)
    require(_digest(dispatch_journal) == _digest(expected_dispatch),
        "guard_not_reproducible_or_predictions_not_sealed_first")
    require(dispatch_payload["schema_version"] == pending["schema_version"] == worker.VERSION
        and _digest(dispatch_payload["records"]) == _digest(results),
        "sealed_dispatch_result_changed")
    expected_pending = [r for r in results
        if r["dispatch"]["status"] == "deferred_separately_approved_backend_required"]
    require(_digest(pending["records"]) == _digest(expected_pending)
        and pending["automatic_submission_allowed"] is False
        and pending["requires_complete_script_and_explicit_approval"] is True,
        "pending_request_inventory_or_authority_changed")
    expected = {"schema_version": worker.VERSION, "status": STATUS, "fixed_development_cases": 2,
        "fresh_observer_calls": 2, "charged_load_attempts": 1,
        "complete_responses": sum(r["contract_status"] == "complete" for r in records),
        "guard_status_counts": dict(sorted(Counter(r["guard"]["status"] for r in results).items())),
        "dispatch_status_counts": dict(sorted(Counter(r["dispatch"]["status"] for r in results).items())),
        "new_numeric_planner_calls": 0, "new_generator_or_primary_scorer_calls": 0,
        "new_backend_dispatch_attempts": 0, "numeric_intents_are_cached": True,
        "historical_cost_not_erased": True, "new_model_retries": 0,
        "clinical_acceptance": False, "clinical_accuracy": None, "measured_saved_model_calls": None,
        "historical_selection_changed": False, "llm_superiority_demonstrated": False,
        "training_performed": False}
    for name, value in expected.items():
        require(_digest(summary[name]) == _digest(value), "summary_counts_or_claims_changed")
    for name in ("runtime_seconds", "peak_torch_allocated_vram_gib"):
        value = summary[name]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "finite_nonnegative_resource_measurements_required")
    return {"fixed_images": 2, "fresh_observer_calls": 2, "charged_load_attempts": 1,
        "complete_responses": expected["complete_responses"],
        "guard_status_counts": expected["guard_status_counts"],
        "dispatch_status_counts": expected["dispatch_status_counts"],
        "pending_separately_approved_requests": len(expected_pending),
        "new_numeric_planner_calls": 0, "new_generator_or_primary_scorer_calls": 0,
        "new_backend_dispatch_attempts": 0, "execution_order_replay_pass": True,
        "selection_changed": False, "clinical_acceptance": False,
        "measured_saved_model_calls": None, "clinical_accuracy": None,
        "model_predictions_independently_reexecuted": False}


def audit(args):
    worker.cached.observer.source.previous.gate.cpu_guard()  # Before reads/writes.
    Reader = worker.cached.observer.source.previous.postflight.MetadataReader
    pr, sr = Reader(args.plan_run), Reader(args.source_run)
    pr.hash(pr.root / "manifest.json", args.plan_manifest_sha256)
    pm = pr.json(pr.root / "manifest.json")
    require(pm["schema_version"] == worker.VERSION
        and pm["status"] == "prepared_cpu_only_gpu_not_submitted", "authenticated_prepared_plan_required")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    plan = pr.json(pr.root / "plan.json")
    for field in ("source_pins", "artifact_pins"):
        worker.cached.observer.source.check_pins(plan[field])
    comparison_path = pr.path(plan["comparison_inputs_path"])
    pr.hash(comparison_path, plan["comparison_inputs_sha256"])
    comparison = pr.json(comparison_path)
    require(comparison["schema_version"] == worker.VERSION, "original_comparison_contract_required")
    sr.hash(sr.root / "manifest.json", args.run_manifest_sha256)
    manifest = sr.json(sr.root / "manifest.json")
    require(manifest["schema_version"] == worker.VERSION and manifest["status"] == STATUS
        and manifest["plan_manifest_sha256"] == args.plan_manifest_sha256
        and set(manifest["artifacts"]) == ARTIFACTS
        and type(manifest["fresh_observer_calls"]) is int and manifest["fresh_observer_calls"] == 2
        and type(manifest["new_generator_calls"]) is int and manifest["new_generator_calls"] == 0
        and manifest["clinical_acceptance"] is False, "complete_same_plan_unvalidated_run_required")
    for name, pin in manifest["artifacts"].items(): sr.hash(sr.root / name, pin["sha256"])
    payload = {name: sr.json(sr.root / name) for name in ARTIFACTS}
    counts = verify_payload(plan, comparison["cases"], payload["predictions.json"],
        sr.journal(sr.root / "observer_calls.journal.jsonl"), payload["dispatch_results.json"],
        payload["pending_requests.json"], sr.journal(sr.root / "dispatch.journal.jsonl"),
        payload["summary.json"], sr.hash(sr.root / "predictions.json"))
    pr.recheck(); sr.recheck()
    report = {"schema_version": VERSION, "status": "metadata_and_execution_order_verified_not_clinical",
        "source_manifest_sha256": args.run_manifest_sha256,
        "plan_manifest_sha256": args.plan_manifest_sha256, "counts": counts,
        "new_model_calls": 0, "raw_source_bodies_parsed": False, "image_pixels_decoded": False,
        "clinical_acceptance": False, "artifact_pins": {**pr.pins, **sr.pins},
        "source_pins": {**plan["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
            str(Path(__file__).resolve().parents[1] / "tests/test_guarded_image_postflight.py"):
                sha256_file(Path(__file__).resolve().parents[1] / "tests/test_guarded_image_postflight.py")}}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        artifact = write_private_json(temporary / "audit.json", report)
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": report["status"], "artifacts": {artifact.name: {"sha256": sha256_file(artifact)}},
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("run-manifest-sha256", "plan-manifest-sha256", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target = audit(args)
        print(json.dumps({"stage": VERSION, "status": "completed",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
