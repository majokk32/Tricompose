#!/usr/bin/env python3
"""Fresh image verification BEFORE dispatching two sealed historical intents.

This is a forward-ordered execution-boundary smoke on cached development inputs,
not fresh EHR/CXR/report generation, a held-out efficacy trial or LLM superiority.
Only two frozen image-observer calls are installed. No generation backend is
authorized: unblocked intentions become explicit pending requests for a separate
approved job. No external API, training, score retuning or winner replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import time

import build_image_disagreement_preview as cached
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    require_inside, sha256_file, write_private_json, write_private_text)
from tricompose_v12.runtime_dispatch import ProtectedJournal
from tricompose_llm.contracts import validate_decision, validate_public_state
from tricompose_llm.guarded_action_dispatch import GuardedActionDispatcher
from tricompose_llm.local_qwen import gpu_guard

VERSION = "tricompose-fresh-observer-before-dispatch-smoke-v1"
CONFIG = {**cached.observer.CONFIG, "maximum_image_calls": 2,
    "generation_backend_installed": False, "numeric_planner_installed": False}
require = cached.require


def two_reference_images(generation):
    require(generation["data_origin"] == "original_fully_synthetic_pool80"
        and len(generation["cases"]) == len({c["case_id"] for c in generation["cases"]}) == 2,
        "same_two_fixed_development_anchors_required")
    images = []
    for case in generation["cases"]:
        row = case["reference"]
        require(row["case_id"] == case["case_id"] and row["seed"] == 0 and row["cxr_model_id"] == "roentgen_v2"
            and len(case["cached_rows"]) == 4, "original_reference_image_not_best_probe_required")
        matches = [t for t in generation["cached_triplets"] if all(t[k] == row[k] for k in
            ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256",
             "cxr_model_id", "report_model_id", "seed"))]
        require(len(matches) == 1, "exact_original_reference_triplet_required")
        images.append({"cxr_candidate_id": row["cxr_candidate_id"],
            "cxr_sha256": row["cxr_sha256"], "path": matches[0]["cxr_path"]})
    require(len({i["cxr_candidate_id"] for i in images}) == len({i["cxr_sha256"] for i in images}) == 2,
            "two_distinct_reference_images_required")
    return sorted(images, key=lambda i: (i["cxr_sha256"], i["cxr_candidate_id"]))


def dispatch_cases(cases, records, sink):
    """No model/backend construction here. Fresh observations must be sealed."""
    index = {r["cxr_candidate_id"]: r for r in records}
    require(len(records) == len(index) == len(cases) == 2
        and set(index) == {c["reference"]["cxr_candidate_id"] for c in cases},
        "both_fresh_reference_observations_required")
    results = []
    for case in cases:
        row, state = case["reference"], case["numeric_state"]
        validate_public_state(state); validate_decision(case["intent"], state)
        current = next(e for e in state["evidence"] if e["candidate_id"] == state["current_candidate_id"])
        require(case["cached_rows"][int(current["candidate_id"][1:])]["triple_candidate_id"] == row["triple_candidate_id"],
                "sealed_intent_current_reference_changed")
        guard = cached.guard_for_image(index[row["cxr_candidate_id"]], case["cached_rows"],
            candidate_id=current["candidate_id"], evidence_id=current["evidence_id"])
        dispatcher = GuardedActionDispatcher(sink=lambda event: sink({"case_id": row["case_id"], **event}))
        result, payload = dispatcher.dispatch(case["intent"], state, guard, backend_factory=None)
        require(payload is None and result["backend_dispatch_attempts"] == 0
            and result["actual_worker_model_calls"] == 0, "no_generation_backend_in_this_verification_job")
        results.append({"case_id": row["case_id"], "reference_candidate_id": row["triple_candidate_id"],
            "cxr_sha256": row["cxr_sha256"], "ehr_sha256": row["ehr_sha256"],
            "ehr_facts_sha256": row["ehr_facts_sha256"], "guard": guard, "dispatch": result,
            "retained_reference_changed": False, "source_numeric_intent_is_cached": True,
            "clinical_acceptance": False})
    return results


def prepare(args):
    cached.observer.source.previous.gate.cpu_guard()
    Reader = cached.observer.source.previous.postflight.MetadataReader
    pr, gr, sr = [Reader(p) for p in (cached.PLAN, cached.observer.source.GENERATION_PLAN,
        cached.observer.source.SOURCE)]
    pr.hash(pr.root / "manifest.json", cached.PLAN_SHA)
    pm = pr.json(pr.root / "manifest.json")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    prior = pr.json(pr.root / "plan.json")
    require(prior["schema_version"] == cached.observer.VERSION
        and prior["config"] == cached.observer.CONFIG, "same_frozen_image_observer_required")
    for field in ("source_pins", "artifact_pins"):
        cached.observer.source.check_pins(prior[field])
    cached.observer.source.check_pins(prior["qwen"]["asset_pins"])
    gr.hash(gr.root / "manifest.json", cached.observer.source.GENERATION_PLAN_SHA)
    gm = gr.json(gr.root / "manifest.json")
    gr.hash(gr.root / "plan.json", gm["plan_sha256"])
    generation = gr.json(gr.root / "plan.json")
    images = two_reference_images(generation)
    sr.hash(sr.root / "manifest.json", cached.observer.source.SOURCE_SHA)
    cases = []
    for case in generation["cases"]:
        step = sr.root / "policy" / case["case_id"] / "step_0"
        for name in ("numeric_state.json", "policy_result.json"):
            path = str(step / name)
            require(path in prior["artifact_pins"], "historical_pending_intent_must_be_authenticated")
            sr.hash(path, prior["artifact_pins"][path])
        state = sr.json(step / "numeric_state.json")
        outcome = sr.json(step / "policy_result.json")
        validate_public_state(state)
        require(outcome["status"] == "completed"
            and outcome["state_sha256"] == sr.hash(step / "numeric_state.json"), "sealed_intent_state_binding_required")
        intent = validate_decision(outcome["decision"], state)
        require(intent["action"] == "regenerate_cxr", "both_original_image_intents_required")
        cases.append({"reference": case["reference"], "cached_rows": case["cached_rows"],
            "numeric_state": state, "intent": intent})
    for image in images:
        cached.observer.source.check_pins({image["path"]: image["cxr_sha256"]})
    for reader in (pr, gr, sr): reader.recheck()
    new_sources = {str(p): sha256_file(p) for p in (Path(__file__).resolve(),
        cached.observer.source.ROOT / "TriCompose-v1.2/agent/build_image_disagreement_preview.py",
        cached.observer.source.ROOT / "TriCompose-v1.2/agent/tricompose_llm/image_evidence_guard.py",
        cached.observer.source.ROOT / "TriCompose-v1.2/agent/tricompose_llm/guarded_action_dispatch.py",
        cached.observer.source.ROOT / "TriCompose-v1.2/tests/test_guarded_action_dispatch.py",
        cached.observer.source.ROOT / "TriCompose-v1.2/tests/test_guarded_image_probe.py")}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        compare = write_private_json(temporary / "comparison_inputs.json", {"schema_version": VERSION, "cases": cases})
        payload = {"schema_version": VERSION, "config": CONFIG, "image_inputs": images,
            "qwen": prior["qwen"], "image_prompt_sha256": prior["image_prompt_sha256"],
            "comparison_inputs_path": str(target / compare.name), "comparison_inputs_sha256": sha256_file(compare),
            "source_pins": {**prior["source_pins"], **new_sources},
            "artifact_pins": {**prior["artifact_pins"], **pr.pins, **gr.pins, **sr.pins,
                **{i["path"]: i["cxr_sha256"] for i in images}},
            "previous_image_observer_predictions_parsed": False, "source_bodies_parsed": False,
            "image_pixels_decoded": False, "new_model_calls": 0, "selection_change_allowed": False,
            "clinical_acceptance": False, "scope": "fresh_verification_gate_on_cached_development_intents"}
        cached.observer.source.check_pins(new_sources)
        plan = write_private_json(temporary / "plan.json", payload)
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(plan),
            "maximum_observer_calls": 2, "generation_backend_installed": False,
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def run(args):
    gpu_guard()  # Before metadata, model imports or writes.
    root = require_inside(args.plan_run, cached.observer.source.PROTECTED_ROOT, must_exist=True)
    require(sha256_file(root / "manifest.json") == args.plan_manifest_sha256, "reviewed_guarded_plan_changed")
    manifest = cached.observer.source.read_json(root / "manifest.json")
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only_gpu_not_submitted"
        and sha256_file(root / "plan.json") == manifest["plan_sha256"],
            "sealed_guarded_execution_plan_required")
    plan = cached.observer.source.read_json(root / "plan.json")
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG
        and len(plan["image_inputs"]) == 2 and plan["selection_change_allowed"] is False,
        "two_fresh_observers_no_generation_backend_required")
    for field in ("source_pins", "artifact_pins"): cached.observer.source.check_pins(plan[field])
    cached.observer.source.check_pins(plan["qwen"]["asset_pins"])
    require(sha256_file(plan["comparison_inputs_path"]) == plan["comparison_inputs_sha256"], "sealed_comparison_inputs_required")
    prompt = cached.observer.existing.image_interface.IMAGE_PROMPT.format(
        findings=", ".join(cached.observer.existing.image_interface.FINDINGS))
    require(cached.observer.existing.image_interface.digest_text(prompt) == plan["image_prompt_sha256"], "original_image_only_prompt_required")
    import torch
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= CONFIG["min_vram_gib"] * 1024**3,
            "at_least_24gib_gpu_required")
    from PIL import Image
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    records = []; charged = 0; started = time.monotonic()
    try:
        torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
        with ProtectedJournal(temporary / "observer_calls.journal.jsonl") as journal, open(os.devnull, "w") as sink, \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            journal.append({"event": "observer_load_attempt_reserved", "charged_load_attempts": 1})
            model, processor, _ = _load_model(Path(plan["qwen"]["model_path"]), torch,
                min_pixels=CONFIG["min_pixels"], max_pixels=CONFIG["max_pixels"])
            model.eval().requires_grad_(False)
            require(not model.training and not any(p.requires_grad for p in model.parameters()), "frozen_observer_required")
            for item in plan["image_inputs"]:
                path = require_inside(item["path"], cached.observer.source.PROTECTED_ROOT, must_exist=True)
                require(sha256_file(path) == item["cxr_sha256"], "fixed_reference_image_required")
                with Image.open(path) as handle:
                    require(64 <= handle.width <= 4096 and 64 <= handle.height <= 4096, "bounded_image_dimensions_required")
                    image = handle.convert("RGB").copy()
                require(charged < 2, "two_call_budget_exceeded")
                journal.append({"event": "call_reserved", "ordinal": charged, "cxr_sha256": item["cxr_sha256"]})
                charged += 1
                response, tokens = cached.observer.existing.image_interface.infer(
                    cached.observer.existing.image_interface.request_messages("image", image=image),
                    model, processor, torch, max_new_tokens=CONFIG["max_new_tokens"])
                record = {"cxr_candidate_id": item["cxr_candidate_id"], "cxr_sha256": item["cxr_sha256"],
                    "response_sha256": cached.observer.existing.image_interface.digest_text(response), **tokens,
                    **cached.observer.existing.sanitized_decode(response, token_limit_reached=tokens["token_limit_reached"])}
                records.append(record)
                journal.append({"event": "call_finished", "ordinal": charged - 1, "contract_status": record["contract_status"]})
        predictions = write_private_json(temporary / "predictions.json", {"schema_version": VERSION,
            "records": records, "image_only": True, "frozen": True,
            "model_received_ehr_reports_ids_or_scores": False})
        with predictions.open("rb") as handle: os.fsync(handle.fileno())
        # Fresh predictions are sealed before comparison and actual dispatch.
        require(sha256_file(plan["comparison_inputs_path"]) == plan["comparison_inputs_sha256"], "comparison_inputs_changed")
        cases = cached.observer.source.read_json(plan["comparison_inputs_path"])["cases"]
        with ProtectedJournal(temporary / "dispatch.journal.jsonl") as journal:
            journal.append({"event": "fresh_observer_predictions_sealed", "sha256": sha256_file(predictions)})
            results = dispatch_cases(cases, records, journal.append)
        executed = write_private_json(temporary / "dispatch_results.json", {"schema_version": VERSION, "records": results})
        pending = write_private_json(temporary / "pending_requests.json", {"schema_version": VERSION,
            "records": [r for r in results if r["dispatch"]["status"] == "deferred_separately_approved_backend_required"],
            "automatic_submission_allowed": False, "requires_complete_script_and_explicit_approval": True})
        summary = write_private_json(temporary / "summary.json", {"schema_version": VERSION,
            "status": "completed_fresh_verification_before_dispatch_unvalidated", "fixed_development_cases": 2,
            "fresh_observer_calls": charged, "charged_load_attempts": 1,
            "complete_responses": sum(r["contract_status"] == "complete" for r in records),
            "guard_status_counts": dict(sorted(Counter(r["guard"]["status"] for r in results).items())),
            "dispatch_status_counts": dict(sorted(Counter(r["dispatch"]["status"] for r in results).items())),
            "new_numeric_planner_calls": 0, "new_generator_or_primary_scorer_calls": 0,
            "new_backend_dispatch_attempts": 0, "numeric_intents_are_cached": True,
            "historical_cost_not_erased": True, "new_model_retries": 0,
            "clinical_acceptance": False, "clinical_accuracy": None, "measured_saved_model_calls": None,
            "historical_selection_changed": False, "llm_superiority_demonstrated": False,
            "training_performed": False, "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_torch_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
        for field in ("source_pins", "artifact_pins"): cached.observer.source.check_pins(plan[field])
        cached.observer.source.check_pins(plan["qwen"]["asset_pins"])
        files = (predictions, executed, pending, summary)
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": "completed_fresh_verification_before_dispatch_unvalidated",
            "plan_manifest_sha256": args.plan_manifest_sha256,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "fresh_observer_calls": charged, "new_generator_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        write_private_json(temporary / "failure.json", {"schema_version": VERSION,
            "status": "failed_attempts_retained", "charged_observer_calls": charged,
            "complete_prefix_records": records, "automatic_resume": False})
        commit_atomic_run(temporary, target); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep, execution = sub.add_parser("prepare"), sub.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for item in (prep, execution):
        item.add_argument("--output-root", type=Path, required=True)
        item.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
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
