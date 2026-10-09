#!/usr/bin/env python3
"""Fresh guarded Qwen requests for untried report experts on two existing probes.

CPU prepare authenticates metadata only. GPU run makes at most two NEW numeric
planning calls using CACHED blind image observations. No CXR/report generation,
score retuning, training, external API or replacement of original selected triples.
This is a development planning smoke, not held-out clinical repair or savings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import run_guarded_image_probe as previous
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    require_inside, sha256_file, write_private_json)
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.runtime_dispatch import ProtectedJournal
from tricompose_v12.report_expert_control import MODELS
from tricompose_llm.contracts import PUBLIC_SCHEMA, validate_public_state
from tricompose_llm.guard_aware_qwen import (GuardAwareQwenPlanner, VERSION as PACKET_VERSION,
    validate_packet, request_messages)
from tricompose_llm.guarded_action_dispatch import GuardedActionDispatcher
from tricompose_llm.live_report_bridge import FreshReportSession
from tricompose_llm.local_qwen import gpu_guard

VERSION = "tricompose-fresh-guarded-report-request-smoke-v1"
BASE = previous.cached.observer.source.BASE
CONFIG = {"maximum_policy_calls": 2, "maximum_steps_per_case": 1,
    "new_phase_budget_units_per_case": 3, "policy_units": 1, "report_and_chexbert_units": 2,
    "max_new_tokens": 384, "max_context_tokens": 8192, "do_sample": False,
    "min_vram_gib": 24, "generation_backend_installed": False, "model_retries": 0}
require = previous.require


def make_case(old, probe_row, cached_rows, observation, workers):
    """Same fixed EHR; seed-one trial image, not replacement of retained winner."""
    require(len(cached_rows) == 4 and {r["report_model_id"] for r in cached_rows} == set(MODELS)
        and probe_row["seed"] == 1 and probe_row["report_model_id"] == old["reference"]["report_model_id"]
        and probe_row["cxr_sha256"] != old["reference"]["cxr_sha256"],
        "two_original_seed_one_trial_branches_required")
    rows = cached_rows + [probe_row]
    require(all(r["case_id"] == old["case_id"] and all(r[k] == old["reference"][k]
        for k in ("ehr_sha256", "ehr_facts_sha256")) for r in rows), "fixed_ehr_lineage_required")
    context = previous.cached.observer.source.previous.gate.context(old["anchor"], workers["xrv"], workers["chexbert"])
    numeric = [FreshReportSession._numeric(SimpleNamespace(context=context), r, i) for i, r in enumerate(rows)]
    current_model = probe_row["report_model_id"]
    tools = [{"tool_id": f"t{100+i:04d}", "action": "regenerate_report", "model_id": f"m{i:04d}",
        "seed": 1, "cost_units": 2} for i, model in enumerate(MODELS) if model != current_model]
    state = validate_public_state({"schema_version": PUBLIC_SCHEMA, "step_id": "s0000",
        "current_candidate_id": "c0004", "evidence": numeric, "tools": tools,
        "budget": {"limit_units": 3, "spent_units": 1, "planner_units_per_call": 1}, "history": []})
    guard = previous.cached.guard_for_image(observation, [probe_row], candidate_id="c0004", evidence_id="e0004")
    packet = validate_packet({"schema_version": PACKET_VERSION, "observation": state, "image_guard": guard,
        "expert_history": [{"model_id": f"m{MODELS.index(r['report_model_id']):04d}",
            "candidate_id": f"c{i:04d}", "evidence_id": f"e{i:04d}"} for i, r in enumerate(cached_rows)],
        "candidate_image_groups": [{"candidate_id": f"c{i:04d}", "image_group_id": "i0001" if i == 4 else "i0000"}
            for i in range(5)], "current_report_model_id": f"m{MODELS.index(current_model):04d}",
        "generation_backend_installed": False})
    return {"case_id": old["case_id"], "ehr_sha256": probe_row["ehr_sha256"],
        "ehr_facts_sha256": probe_row["ehr_facts_sha256"], "trial_image_sha256": probe_row["cxr_sha256"],
        "trial_candidate_id": probe_row["triple_candidate_id"],
        "retained_original_candidate_id": old["reference"]["triple_candidate_id"], "packet": packet,
        "policy_input_sha256": _digest(packet), "request_messages_sha256": _digest(request_messages(packet)),
        "budget_in_packet_includes_prospective_policy_reservation": True,
        "tool_catalog": {f"t{100+i:04d}": {"model_id": model, "input_image_sha256": probe_row["cxr_sha256"],
            "seed": 1, "cost_units": 2, "cost_scope": "one_report_generator_and_one_chexbert",
            "backend_installed_in_this_job": False} for i, model in enumerate(MODELS) if model != current_model},
        "original_selection_change_allowed": False}


def prepare(args):
    previous.cached.observer.source.previous.gate.cpu_guard()
    source = previous.cached.observer.source
    Reader = source.previous.postflight.MetadataReader
    gr, sr, ir = [Reader(p) for p in (source.GENERATION_PLAN, source.SOURCE, previous.cached.OBSERVATIONS)]
    gr.hash(gr.root/"manifest.json", source.GENERATION_PLAN_SHA)
    gm = gr.json(gr.root/"manifest.json"); gr.hash(gr.root/"plan.json", gm["plan_sha256"])
    generation = gr.json(gr.root/"plan.json")
    require(generation["data_origin"] == "original_fully_synthetic_pool80"
        and len(generation["cases"]) == 2, "same_two_predeclared_synthetic_anchors_required")
    sr.hash(sr.root/"manifest.json", source.SOURCE_SHA)
    sm = sr.json(sr.root/"manifest.json")
    for name, pin in sm["artifacts"].items(): sr.hash(sr.root/name, pin["sha256"])
    rows = sr.json(sr.root/"score_rows.json")["records"]
    triples = sr.json(sr.root/"completed_triplets.json")
    require(len(rows) == 10 and len(triples["new_records"]) == 2, "all_original_observed_probes_required")
    ir.hash(ir.root/"manifest.json", previous.cached.OBSERVATIONS_SHA)
    im = ir.json(ir.root/"manifest.json")
    require(im["status"] == "completed_blind_image_observer_unvalidated", "completed_cached_image_observer_required")
    for name, pin in im["artifacts"].items(): ir.hash(ir.root/name, pin["sha256"])
    observations = ir.json(ir.root/"predictions.json")
    require(observations["frozen"] is True and observations["image_only"] is True
        and observations["model_received_ehr_reports_ids_or_scores"] is False,
        "cached_blind_observer_required")
    previous.cached.verify_calls(observations["records"], ir.journal(ir.root/"calls.journal.jsonl"))
    report_plan_path = [p for p in generation["artifact_pins"]
        if "/llm_fresh_report_plans/" in p and Path(p).name == "plan.json"]
    require(len(report_plan_path) == 1, "original_four_expert_registry_required")
    rr = Reader(Path(report_plan_path[0]).parent)
    rr.hash(report_plan_path[0], generation["artifact_pins"][report_plan_path[0]])
    report_plan = rr.json(report_plan_path[0])
    catalog = {m: report_plan["workers"][m] for m in MODELS}
    require(all(s["status"] == "preflighted" and s["factory_instantiated"] is False for s in catalog.values()),
        "existing_frozen_report_catalog_required")
    # Request-only menu: no report model loads or new stability claims here.
    cases = []
    for old in generation["cases"]:
        probes = [r for r in rows if r["case_id"] == old["case_id"] and r["seed"] == 1]
        require(len(probes) == 1, "exactly_one_existing_probe_per_fixed_anchor_required")
        probe = probes[0]
        images = [r for r in observations["records"] if r["cxr_candidate_id"] == probe["cxr_candidate_id"]]
        require(len(images) == 1 and images[0]["cxr_sha256"] == probe["cxr_sha256"], "probe_observer_identity_required")
        matched = [t for t in triples["new_records"] if all(t[k] == probe[k] for k in
            ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256", "report_model_id", "seed"))]
        require(len(matched) == 1, "probe_triplet_lineage_required")
        sr.hash(matched[0]["cxr_path"], probe["cxr_sha256"])
        require({r["report_model_id"] for r in rows if r["cxr_sha256"] == probe["cxr_sha256"]} == {probe["report_model_id"]},
            "future_experts_must_be_untried_in_sealed_image_inventory")
        cases.append(make_case(old, probe, old["cached_rows"], images[0], generation["workers"]))
    pin_files = [Path(__file__).resolve(), Path(__file__).parent/"tricompose_llm/guard_aware_qwen.py",
        Path(__file__).parent/"tricompose_llm/guarded_action_dispatch.py",
        Path(__file__).parent/"tricompose_llm/image_evidence_guard.py",
        Path(__file__).resolve().parents[1]/"tests/test_guard_aware_qwen.py",
        Path(__file__).resolve().parents[1]/"tests/test_guarded_fresh_policy.py"]
    pins = {**generation["source_pins"], **{str(p): sha256_file(p) for p in pin_files}}
    source.check_pins(pins); source.check_pins(generation["artifact_pins"])
    source.check_pins(generation["qwen"]["asset_pins"])
    for reader in (gr, sr, ir, rr): reader.recheck()
    plan = {"schema_version": VERSION, "config": CONFIG, "cases": cases, "qwen": generation["qwen"],
        "source_pins": pins, "artifact_pins": {**generation["artifact_pins"], **gr.pins, **sr.pins, **ir.pins, **rr.pins},
        "report_catalog": catalog, "data_origin": "original_fully_synthetic_pool80",
        "candidate_rule": "both_original_completed_seed_one_probes_no_outcome_selection",
        "cached_observer_calls": 4, "cached_observer_load_attempts": 1,
        "source_cxr_probe_worker_attempts": 8, "source_cxr_probe_policy_requests": 2,
        "earlier_stage_worker_attempts": generation["historical_worker_attempts"],
        "earlier_stage_policy_requests": generation["historical_policy_requests"],
        "historical_cost_scope": "shared_sunk_not_reestimated_or_zero",
        "historical_cost_not_erased": True, "source_bodies_parsed": False, "image_pixels_decoded": False,
        "new_model_calls": 0, "original_winner_change_allowed": False,
        "no_heldout_or_clinical_advantage_claim": True}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        artifact = write_private_json(temporary/"plan.json", plan)
        write_private_json(temporary/"manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(artifact),
            "maximum_new_policy_calls": 2, "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def run(args):
    gpu_guard()
    source = previous.cached.observer.source
    root = require_inside(args.plan_run, source.PROTECTED_ROOT, must_exist=True)
    require(sha256_file(root/"manifest.json") == args.plan_manifest_sha256, "reviewed_fresh_policy_plan_changed")
    manifest = source.read_json(root/"manifest.json")
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only_gpu_not_submitted"
        and sha256_file(root/"plan.json") == manifest["plan_sha256"], "sealed_new_planning_plan_required")
    plan = source.read_json(root/"plan.json")
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG and len(plan["cases"]) == 2,
        "same_two_bounded_fresh_requests_required")
    for name in ("source_pins", "artifact_pins"): source.check_pins(plan[name])
    source.check_pins(plan["qwen"]["asset_pins"])
    for case in plan["cases"]:
        validate_packet(case["packet"])
        require(case["policy_input_sha256"] == _digest(case["packet"])
            and case["request_messages_sha256"] == _digest(request_messages(case["packet"]))
            and len(case["packet"]["observation"]["tools"]) == 3, "three_real_untried_expert_requests_required")
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    results = []; charged = 0; started = time.monotonic()
    try:
        with ProtectedJournal(temporary/"policy_dispatch.journal.jsonl") as journal, open(os.devnull, "w") as sink, \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            journal.append({"event": "planner_load_reserved", "charged_load_attempts": 1})
            planner = GuardAwareQwenPlanner(plan["qwen"]["model_path"])
            for case in plan["cases"]:
                packet = case["packet"]
                journal.append({"event": "fresh_policy_reserved", "ordinal": charged,
                    "policy_input_sha256": case["policy_input_sha256"], "charged_policy_units": 1})
                charged += 1
                try:
                    decision = planner.propose_guarded(deepcopy(packet))
                except Exception:
                    record = {"case_id": case["case_id"], "status": "planner_failed_charged_no_retry",
                        "decision": None, "dispatch": None, "source": "fresh_llm_attempt_failed",
                        "clinical_acceptance": False, "original_winner_changed": False}
                    results.append(record)
                    journal.append({"event": "fresh_policy_failed_charged", "ordinal": charged-1})
                    continue
                journal.append({"event": "fresh_policy_validated", "ordinal": charged-1, "decision": decision})
                dispatcher = GuardedActionDispatcher(sink=lambda event: journal.append({"case_id": case["case_id"], **event}))
                dispatch, payload = dispatcher.dispatch(decision, packet["observation"], packet["image_guard"], backend_factory=None)
                require(payload is None and dispatch["actual_worker_model_calls"] == 0, "planning_only_no_generation_backend")
                results.append({"case_id": case["case_id"], "status": "fresh_llm_decision_validated_unverified",
                    "decision": decision, "dispatch": dispatch, "source": "new_frozen_qwen_generate_call",
                    "policy_input_sha256": case["policy_input_sha256"],
                    "response_metadata": deepcopy(planner.last_response_metadata),
                    "clinical_acceptance": False, "original_winner_changed": False})
            audit = planner.audit()
        output = write_private_json(temporary/"policy_results.json", {"schema_version": VERSION, "records": results})
        pending = write_private_json(temporary/"pending_requests.json", {"schema_version": VERSION,
            "records": [{"case_id": r["case_id"], "request": next(c for c in plan["cases"] if c["case_id"] == r["case_id"])["tool_catalog"][r["dispatch"]["effective_decision"]["target_id"]],
                "decision": r["decision"], "clinical_acceptance": False} for r in results if r["dispatch"] is not None
                and r["dispatch"]["status"] == "deferred_separately_approved_backend_required"],
            "automatic_submission_allowed": False, "requires_complete_script_and_explicit_approval": True})
        audit_file = write_private_json(temporary/"planner_audit.json", audit)
        summary = write_private_json(temporary/"summary.json", {"schema_version": VERSION,
            "status": "completed_fresh_guarded_numeric_planning_unvalidated", "fixed_development_cases": 2,
            "charged_new_policy_requests": charged, "fresh_generate_attempts": audit["generate_attempts"],
            "validated_fresh_decisions": sum(r["decision"] is not None for r in results),
            "decision_action_counts": dict(sorted(Counter(r["decision"]["action"] for r in results if r["decision"] is not None).items())),
            "charged_load_attempts": 1, "new_observer_calls": 0, "new_generator_or_primary_scorer_calls": 0,
            "new_backend_dispatch_attempts": 0, "image_observations_are_cached": True,
            "numeric_decisions_are_cached": False, "historical_cost_not_erased": True,
            "original_winner_changed": False, "clinical_acceptance": False,
            "clinical_accuracy": None, "measured_saved_model_calls": None,
            "training_performed": False, "runtime_seconds": round(time.monotonic()-started, 3),
            "peak_torch_allocated_vram_gib": round(audit["peak_gpu_memory_bytes"]/1024**3, 3)})
        for name in ("source_pins", "artifact_pins"): source.check_pins(plan[name])
        source.check_pins(plan["qwen"]["asset_pins"])
        write_private_json(temporary/"manifest.json", {"schema_version": VERSION,
            "status": "completed_fresh_guarded_numeric_planning_unvalidated", "plan_manifest_sha256": args.plan_manifest_sha256,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in (output, pending, audit_file, summary)},
            "charged_new_policy_requests": charged, "new_generator_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        write_private_json(temporary/"failure.json", {"schema_version": VERSION, "status": "failed_attempts_retained",
            "charged_policy_requests": charged, "charged_load_attempts": 1, "completed_prefix_records": results,
            "automatic_resume": False})
        commit_atomic_run(temporary, target); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep, execution = commands.add_parser("prepare"), commands.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    for command in (prep, execution):
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(target/"manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
