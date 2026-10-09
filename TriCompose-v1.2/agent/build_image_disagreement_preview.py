#!/usr/bin/env python3
"""Authenticated CPU-only preview of an optional image-evidence veto.

All four existing images remain represented. Only the two actual historical
planning states/decisions are replayed: the probe images are not invented future
policy requests. Post-hoc observations cannot prove prospective savings or
clinical correctness. No inference, artifact body/pixel reads or selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import verify_fresh_probe_images as observer
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       sha256_file, write_private_json, write_private_text)
from tricompose_llm.contracts import validate_public_state, validate_decision
from tricompose_llm.image_evidence_guard import build_guard, guarded_decision, guarded_state

VERSION = "tricompose-image-disagreement-counterfactual-preview-v1"
BASE = observer.source.BASE
OBSERVATIONS = BASE / "llm_probe_image_observers/probe_images4_12843415"
OBSERVATIONS_SHA = "a55e210f3c489b1dd9d7d691b885a6d979c56eac68a586a9e659c0aa041f6f8b"
PLAN = BASE / "llm_probe_image_plans/probe_images4_12817261_001"
PLAN_SHA = "4913be6d356954b46acbaabd6415e0f7f4aa390d727fb9fcebf036f1451a114f"
require = observer.require


def guard_for_image(record, rows, *, candidate_id, evidence_id):
    """Reports may differ; shared image/EHR states must not. No expert vote."""
    group = [r for r in rows if r["cxr_candidate_id"] == record["cxr_candidate_id"]]
    require(group and all(r["cxr_sha256"] == record["cxr_sha256"] for r in group),
            "existing_image_hash_and_lineage_required")
    known = [(f["finding"], f["ehr"], f["xrv"]) for f in group[0]["receipt"]["fact_states"]]
    require(all([(f["finding"], f["ehr"], f["xrv"]) for f in r["receipt"]["fact_states"]] == known
                for r in group), "shared_image_and_fixed_ehr_states_changed")
    facts = {f["finding"]: f for f in group[0]["receipt"]["fact_states"]}
    require(len(facts) == len(known) and set(observer.existing.image_interface.FINDINGS) <= set(facts),
            "complete_named_image_comparison_scope_required")
    return build_guard({n: facts[n]["ehr"] for n in observer.existing.image_interface.FINDINGS},
        {n: facts[n]["xrv"] for n in observer.existing.image_interface.FINDINGS}, record["states"],
        candidate_id=candidate_id, evidence_id=evidence_id, observer_status=record["contract_status"])


def verify_calls(records, journal):
    require(len(records) == 4 and len(journal) == 9
        and journal[0] == {"event": "observer_load_attempt_reserved", "charged_load_attempts": 1},
        "complete_four_call_reserved_observer_ledger_required")
    for ordinal, record in enumerate(records):
        require(journal[1 + 2 * ordinal] == {"event": "call_reserved", "ordinal": ordinal,
                "cxr_sha256": record["cxr_sha256"]}
            and journal[2 + 2 * ordinal] == {"event": "call_finished", "ordinal": ordinal,
                "contract_status": record["contract_status"]}, "charged_observer_ledger_changed")


def build(args):
    observer.source.previous.gate.cpu_guard()  # Before input reads or writes.
    Reader = observer.source.previous.postflight.MetadataReader
    pr, ir, sr, gr = [Reader(p) for p in (PLAN, OBSERVATIONS,
        observer.source.SOURCE, observer.source.GENERATION_PLAN)]
    pr.hash(pr.root / "manifest.json", PLAN_SHA)
    pm = pr.json(pr.root / "manifest.json")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    plan = pr.json(pr.root / "plan.json")
    require(plan["schema_version"] == observer.VERSION and plan["config"] == observer.CONFIG
        and plan["selection_change_allowed"] is False and plan["max_model_calls"] == 4,
        "exact_frozen_image_observer_plan_required")
    for field in ("source_pins", "artifact_pins"):
        observer.source.check_pins(plan[field])
    observer.source.check_pins(plan["qwen"]["asset_pins"])
    ir.hash(ir.root / "manifest.json", OBSERVATIONS_SHA)
    im = ir.json(ir.root / "manifest.json")
    require(im["schema_version"] == observer.VERSION
        and im["status"] == "completed_blind_image_observer_unvalidated"
        and im["plan_manifest_sha256"] == PLAN_SHA and im["clinical_acceptance"] is False,
        "completed_sealed_secondary_observer_required")
    for name, pin in im["artifacts"].items():
        ir.hash(ir.root / name, pin["sha256"])
    predictions = ir.json(ir.root / "predictions.json")
    require(predictions["image_only"] is True and predictions["frozen"] is True
        and predictions["model_received_ehr_reports_ids_or_scores"] is False,
        "blind_frozen_observer_required")
    records = predictions["records"]
    verify_calls(records, ir.journal(ir.root / "calls.journal.jsonl"))
    osummary = ir.json(ir.root / "summary.json")
    require(osummary["actual_model_calls"] == 4 and osummary["unique_image_finding_slots"] == 32
        and osummary["model_retries"] == osummary["new_generator_calls"] == 0
        and osummary["selection_changed"] is False and osummary["primary_metric_eligible"] is False,
        "unchanged_observer_scope_and_cost_required")
    sr.hash(sr.root / "manifest.json", observer.source.SOURCE_SHA)
    sm = sr.json(sr.root / "manifest.json")
    for name, pin in sm["artifacts"].items():
        sr.hash(sr.root / name, pin["sha256"])
    rows = sr.json(sr.root / "score_rows.json")["records"]
    sr.hash(sr.root / "score_rows.json", plan["comparison_rows_sha256"])
    triples = sr.json(sr.root / "completed_triplets.json")
    require(observer.inventory(rows, triples["cached_references"] + triples["new_records"]) == plan["image_inputs"],
            "unchanged_all_four_images_required")
    comparison = observer.comparisons(records, rows)
    require((ir.root / "image_observer_comparison.csv").read_bytes()
        == observer.source.previous.csv_text(comparison).encode("utf-8"), "exact_observer_table_replay_required")
    gr.hash(gr.root / "manifest.json", observer.source.GENERATION_PLAN_SHA)
    gm = gr.json(gr.root / "manifest.json")
    gr.hash(gr.root / "plan.json", gm["plan_sha256"])
    generation = gr.json(gr.root / "plan.json")
    require(generation["data_origin"] == "original_fully_synthetic_pool80"
        and len(generation["cases"]) == 2, "two_original_synthetic_development_anchors_required")
    index = {r["cxr_candidate_id"]: r for r in records}
    image_guards, previews = [], []
    for case in generation["cases"]:
        ref = case["reference"]
        step_root = sr.root / "policy" / case["case_id"] / "step_0"
        for name in ("numeric_state.json", "policy_result.json"):
            path = str(step_root / name)
            require(path in plan["artifact_pins"], "historical_policy_artifact_must_be_pinned")
            sr.hash(path, plan["artifact_pins"][path])
        state = sr.json(step_root / "numeric_state.json")
        outcome = sr.json(step_root / "policy_result.json")
        validate_public_state(state)
        require(outcome["status"] == "completed"
            and outcome["state_sha256"] == sr.hash(step_root / "numeric_state.json"),
            "existing_policy_outcome_state_binding_required")
        decision = validate_decision(outcome["decision"], state)
        current = next(e for e in state["evidence"] if e["candidate_id"] == state["current_candidate_id"])
        ordinal = int(current["candidate_id"][1:])
        require(case["cached_rows"][ordinal]["triple_candidate_id"] == ref["triple_candidate_id"],
                "historical_current_reference_binding_changed")
        reference_guard = guard_for_image(index[ref["cxr_candidate_id"]], rows,
            candidate_id=current["candidate_id"], evidence_id=current["evidence_id"])
        effective = guarded_decision(decision, state, reference_guard)
        previews.append({"case_id": case["case_id"], "reference_candidate_id": ref["triple_candidate_id"],
            "cxr_sha256": ref["cxr_sha256"], "ehr_sha256": ref["ehr_sha256"],
            "ehr_facts_sha256": ref["ehr_facts_sha256"], "guard": reference_guard,
            "original_numeric_state_sha256": sr.hash(step_root / "numeric_state.json"),
            "guarded_numeric_state": guarded_state(state, reference_guard), **effective,
            "counterfactual_only": True, "new_model_calls": 0, "historical_selection_changed": False})
        image_ids = sorted({r["cxr_candidate_id"] for r in rows if r["case_id"] == case["case_id"]})
        require(len(image_ids) == 2, "both_existing_seeds_retained")
        for iid in image_ids:
            group = [r for r in rows if r["cxr_candidate_id"] == iid]
            is_reference = iid == ref["cxr_candidate_id"]
            guard = reference_guard if is_reference else guard_for_image(index[iid], rows,
                candidate_id="c0004", evidence_id="e0004")
            image_guards.append({"case_id": case["case_id"], "cxr_candidate_id": iid,
                "cxr_sha256": index[iid]["cxr_sha256"], "ehr_sha256": group[0]["ehr_sha256"],
                "ehr_facts_sha256": group[0]["ehr_facts_sha256"], "seed": group[0]["seed"],
                "origin": "historical_retained_reference" if is_reference else "historical_rejected_probe",
                "observer_response_sha256": index[iid]["response_sha256"], "guard": guard,
                "has_historical_planning_request": is_reference,
                "clinical_fault_location": None, "clinical_acceptance": False})
    require(len(previews) == 2 and len(image_guards) == 4, "complete_two_request_four_image_preview_required")
    saved_summary = sr.json(sr.root / "summary.json")
    summary = {"schema_version": VERSION, "status": "completed_counterfactual_only_not_clinical",
        "fixed_development_cases": 2, "unique_existing_images": 4,
        "guard_status_counts": dict(sorted(Counter(i["guard"]["status"] for i in image_guards).items())),
        "historical_planning_requests": len(previews),
        "historical_image_requests": sum(p["raw_decision"]["action"] == "regenerate_cxr" for p in previews),
        "counterfactual_requests_withheld": sum(p["decision_withheld"] for p in previews),
        "new_model_calls": 0, "new_slurm_submissions": 0, "external_api_calls": 0,
        "source_job_charged_new_worker_attempts": saved_summary["charged_new_worker_attempts"],
        "source_job_charged_policy_requests": saved_summary["charged_policy_requests"],
        "source_observer_charged_calls": osummary["actual_model_calls"],
        "source_observer_charged_load_attempts": osummary["charged_load_attempts"],
        "measured_saved_model_calls": None, "clinical_accuracy": None,
        "clinical_acceptance": False, "clinical_fault_localization_validated": False,
        "historical_selection_changed": False, "training_performed": False,
        "prospective_guard_installed_in_consumed_runner": False,
        "same_observer_checkpoint_as_numeric_planner": True}
    for reader in (pr, ir, sr, gr):
        reader.recheck()
    new_sources = {str(p): sha256_file(p) for p in (Path(__file__).resolve(),
        observer.source.ROOT / "TriCompose-v1.2/agent/tricompose_llm/image_evidence_guard.py",
        observer.source.ROOT / "TriCompose-v1.2/tests/test_image_evidence_guard.py",
        observer.source.ROOT / "TriCompose-v1.2/tests/test_image_disagreement_preview.py")}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary / "image_guards.json", {"schema_version": VERSION, "records": image_guards}),
            write_private_json(temporary / "decision_preview.json", {"schema_version": VERSION, "records": previews}),
            write_private_json(temporary / "summary.json", summary)]
        csv_rows = [{"case_id": i["case_id"], "seed": i["seed"], "origin": i["origin"],
            **i["guard"]["counts"], "guard_status": i["guard"]["status"],
            "withhold_image_attribution": i["guard"]["withhold_image_attribution"],
            "has_historical_planning_request": i["has_historical_planning_request"]} for i in image_guards]
        files.append(write_private_text(temporary / "image_guard_table.csv", observer.source.previous.csv_text(csv_rows)))
        source_pins = {**plan["source_pins"], **new_sources}
        artifact_pins = {**plan["artifact_pins"], **pr.pins, **ir.pins, **sr.pins, **gr.pins}
        observer.source.check_pins(new_sources)
        for reader in (pr, ir, sr, gr):
            reader.recheck()
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION,
            "status": summary["status"], "observer_manifest_sha256": OBSERVATIONS_SHA,
            "source_pins": source_pins, "artifact_pins": artifact_pins,
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in files},
            "new_model_calls": 0, "historical_selection_changed": False,
            "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target, summary = build(args)
        print(json.dumps({"stage": VERSION, "status": summary["status"],
            "unique_existing_images": summary["unique_existing_images"],
            "historical_planning_requests": summary["historical_planning_requests"],
            "counterfactual_requests_withheld": summary["counterfactual_requests_withheld"],
            "new_model_calls": 0, "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
