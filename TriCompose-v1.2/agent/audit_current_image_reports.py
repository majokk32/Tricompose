#!/usr/bin/env python3
"""CPU-only numeric receipt, cost and gate replay for shared report execution.

No models, source clinical bodies, report text or image pixels are loaded.
All model attempts remain charged and both policies share one physical result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import run_current_image_requested_reports as worker
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    sha256_file, write_private_json)
from tricompose_llm.contracts import exact, require

VERSION = "tricompose-current-image-shared-report-postflight-v1"


def replay_case(case, workers, journal, snapshot, row, pair):
    """Replay the separate two-attempt phase, never extend/refund the old ledger."""
    require(len(journal) in (4, 6), "complete_serial_shared_phase_journal_required")
    first = {"event": "shared_dependencies_authenticated", "source_policy_manifest_sha256": worker.POLICY_RUN_SHA,
        "source_ledger_sha256": case["historical_ledger_sha256"],
        "cached_partial_receipt_id": case["partial_receipt"]["receipt_id"], "new_budget": 2,
        "policies": ["qwen", "rule"], "physical_report_attempts_reserved_by_plan": 1,
        "distinct_policy_outcomes_claimed": False, "clinical_acceptance": False}
    require(journal[0] == first and worker._digest(case["historical_ledger"]) == case["historical_ledger_sha256"],
        "authenticated_shared_dependencies_and_old_ledger_required")
    anchor = worker.previous.gate.anchor_from_record(case["anchor"])
    worker.previous.gate.validate_ledger(case["historical_ledger"], anchor)
    require(case["historical_ledger"]["charged_model_attempts"] == case["historical_ledger"]["call_budget"] == 4,
        "old_exhausted_budget_not_rewritten")
    charged = completed = failed = 0
    report_hash = None
    for ordinal in range((len(journal) - 2) // 2):
        require(not failed, "no_call_after_failed_shared_request")
        reservation, outcome = journal[1 + 2 * ordinal:3 + 2 * ordinal]
        kind, model = ("report_generator", "llavarad") if ordinal == 0 else ("chexbert", "chexbert")
        expected_request = worker.CallRequest("requested_report" if ordinal == 0 else "requested_chexbert",
            anchor.case_id, anchor.sha256, kind, model, worker._digest(workers[model]),
            case["baseline_row"]["seed"], case["cached_xrv_operation_id"] if ordinal == 0 else "requested_report",
            case["baseline_row"]["cxr_sha256"], report_hash if ordinal else None)
        require(reservation == {"event": "new_worker_reserved", "ordinal": ordinal,
            "request": expected_request.record(), "charged_worker_attempts": 1,
            "dependency_origin": "authenticated_cached_xrv" if ordinal == 0 else "new_validated_report",
            "historical_ledger_changed": False}, "exact_lazy_shared_worker_reservation_required")
        charged += 1
        elapsed = outcome.get("elapsed_seconds")
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 480,
            "finite_within_allocation_elapsed_required")
        if outcome.get("event") == "new_worker_failed_charged":
            exact(outcome, ("event", "ordinal", "error_code", "elapsed_seconds", "automatic_retry",
                "failure_is_clinical_contradiction"), "closed_charged_failure_required")
            require(outcome["ordinal"] == ordinal and outcome["error_code"] in ("timeout", "invalid_result", "runtime_exception")
                and outcome["automatic_retry"] is False and outcome["failure_is_clinical_contradiction"] is False,
                "failed_model_not_clinical_opposition_or_retry")
            failed += 1
        else:
            exact(outcome, ("event", "ordinal", "result", "elapsed_seconds", "clinical_acceptance"),
                "closed_completed_result_required")
            require(outcome["event"] == "new_worker_completed" and outcome["ordinal"] == ordinal
                and outcome["clinical_acceptance"] is False, "completed_unverified_worker_only")
            result = worker.CallResult(**outcome["result"])
            require((result.verification_receipt_id is not None) == (ordinal == 1), "phase_correct_result_binding")
            if ordinal == 0: report_hash = result.output_artifact_sha256
            else:
                require(row is not None and result.output_artifact_sha256 == row["receipt"]["chexbert_labels_sha256"]
                    and result.verification_receipt_id == row["receipt"]["receipt_id"], "new_chexbert_result_receipt_binding")
            completed += 1
    expected_snapshot = {"charged_new_worker_attempts": charged, "completed_new_worker_operations": completed,
        "failed_new_worker_attempts": failed, "pending_new_worker_attempts": 0,
        "new_worker_budget": 2, "automatic_retry": False, "historical_ledger_changed": False}
    require(snapshot == expected_snapshot and journal[-1] == {"event": "new_phase_sealed", **expected_snapshot,
        "clinical_acceptance": False}, "exact_sealed_shared_phase_costs_required")
    if row is not None:
        require(charged == completed == 2 and failed == 0 and row["report_sha256"] == report_hash
            and worker.previous.assess_pair(case, row, workers) == pair, "exact_verified_candidate_gate_replay_required")
    else:
        require(failed == 1 and pair == {"case_id": case["case_id"], "status": "failed_charged_no_retry",
            "branch_selected_candidate_id": case["baseline_row"]["triple_candidate_id"],
            "original_winner_changed": False, "clinical_acceptance": False}, "failure_cannot_be_hidden_or_selected")
    return {"case_id": case["case_id"], "journal_events": len(journal), **expected_snapshot,
        "exact_phase_replay_pass": True, "baseline_model": case["baseline_row"]["report_model_id"],
        "requested_model": "llavarad", "branch_accepted": bool(pair.get("comparison", {}).get("exploratory_gate_pass")),
        "rejection_reasons": pair.get("comparison", {}).get("reasons", ["failed_charged_no_retry"]),
        "baseline_scores": case["baseline_row"]["raw_edge_readouts"],
        "requested_scores": row["raw_edge_readouts"] if row is not None else None,
        "clinical_acceptance": False, "distinct_policy_outcomes_claimed": False}


def build(args):
    worker.previous.gate.cpu_guard()
    pr, sr = worker.previous.Reader(args.plan_run), worker.previous.Reader(args.source_run)
    pr.hash(pr.root / "manifest.json", args.plan_manifest_sha256)
    pm = pr.json(pr.root / "manifest.json")
    require(pm["schema_version"] == worker.VERSION and pm["status"] == "prepared_cpu_only_gpu_not_submitted",
        "reviewed_shared_report_plan_required")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    plan = pr.json(pr.root / "plan.json")
    require(plan["schema_version"] == worker.VERSION and plan["config"] == worker.CONFIG
        and plan["data_origin"] == "original_fully_synthetic_pool80" and len(plan["cases"]) == 2,
        "exact_two_fully_synthetic_shared_cases_required")
    for field in ("source_pins", "artifact_pins"): worker.check_pins(plan[field])
    sr.hash(sr.root / "manifest.json", args.source_manifest_sha256)
    sm = sr.json(sr.root / "manifest.json")
    require(sm["schema_version"] == worker.VERSION and sm["status"] == "completed_shared_report_comparison_unvalidated"
        and sm["plan_manifest_sha256"] == args.plan_manifest_sha256 and sm["clinical_acceptance"] is False,
        "completed_exact_shared_report_run_required")
    for name, pin in sm["artifacts"].items(): sr.hash(sr.root / name, pin["sha256"])
    rows = sr.json(sr.root / "score_rows.json")["records"]
    pairs = sr.json(sr.root / "paired_report_comparison.json")["records"]
    triples = sr.json(sr.root / "completed_triplets.json")["new_records"]
    execution = sr.json(sr.root / "execution_summary.json")
    require(len(pairs) == len(execution["records"]) == 2
        and len(rows) == 2 + len(triples) and len({r["triple_candidate_id"] for r in rows}) == len(rows)
        and {r["case_id"] for r in rows} == {c["case_id"] for c in plan["cases"]}
        and execution["historical_cost"] == plan["historical_cost"] and execution["historical_cost_not_erased"] is True,
        "complete_shared_inventory_and_sunk_costs_required")
    records = []; extra_readers = []
    for case in plan["cases"]:
        cid = case["case_id"]
        group = [r for r in rows if r["case_id"] == cid]
        base = [r for r in group if r["triple_candidate_id"] == case["baseline_row"]["triple_candidate_id"]]
        require(base == [case["baseline_row"]] and len(group) in (1, 2), "unchanged_baseline_and_one_new_row_required")
        proposed = [r for r in group if r not in base]
        row = proposed[0] if proposed else None
        pair = next(p for p in pairs if p["case_id"] == cid)
        snapshot = {k: v for k, v in next(p for p in execution["records"] if p["case_id"] == cid).items() if k != "case_id"}
        root = sr.root / "cases" / cid
        require(sr.json(root / "source_ledger.json") == case["historical_ledger"]
            and sr.json(root / "new_phase_snapshot.json") == snapshot, "persisted_old_and_new_costs_required")
        records.append(replay_case(case, plan["workers"], sr.journal(root / "new_phase.journal.jsonl"), snapshot, row, pair))
        if row is not None:
            ts = [t for t in triples if t["case_id"] == cid]
            require(len(ts) == 1 and all(ts[0][k] == row[k] for k in ("case_id", "ehr_sha256", "ehr_facts_sha256",
                "cxr_sha256", "report_sha256", "report_model_id", "seed")) and ts[0]["selected_as_best"] is False,
                "exact_new_unpromoted_triple_required")
            triple = ts[0]
            for modality in ("cxr", "report", "receipt"):
                path = Path(triple[modality + "_path"])
                reader = sr if path.is_relative_to(sr.root) else worker.previous.Reader(path.parent)
                reader.hash(path, triple[modality + "_sha256"])
                reader.recheck()
                if reader is not sr: extra_readers.append(reader)
            require(sr.json(triple["receipt_path"]) == row["receipt"], "stored_new_receipt_equals_numeric_row")
    csv = worker.score_csv([{**r, "verification_status": r["receipt"]["verification_status"]} for r in rows]).encode()
    path = sr.path(sr.root / "score_table.csv")
    require(path.read_bytes() == csv, "byte_exact_csv_replay_required")
    summary = sr.json(sr.root / "summary.json")
    require(summary["completed_report_label_pairs"] == len(triples)
        and summary["charged_new_worker_attempts"] == sum(r["charged_new_worker_attempts"] for r in records)
        and summary["completed_new_worker_operations"] == sum(r["completed_new_worker_operations"] for r in records)
        and summary["failed_new_worker_attempts"] == sum(r["failed_new_worker_attempts"] for r in records)
        and summary["proxy_preserving_branch_changes"] == sum(r["branch_accepted"] for r in records)
        and all(summary[k] == 0 for k in ("new_planner_calls", "new_cxr_calls", "new_xrv_calls", "new_image_observer_calls"))
        and all(summary[k] is False for k in ("training_performed", "original_winner_changed", "clinical_acceptance",
            "clinical_repair_success", "distinct_policy_outcomes_claimed", "llm_superiority_demonstrated"))
        and summary["clinical_accuracy"] is None and summary["measured_saved_model_calls"] is None,
        "exact_worker_totals_and_no_unsupported_claim_required")
    pr.recheck(); sr.recheck()
    extra_pins = {}
    for reader in extra_readers: reader.recheck(); extra_pins.update(reader.pins)
    audit = {"schema_version": VERSION, "plan_manifest_sha256": args.plan_manifest_sha256,
        "source_manifest_sha256": args.source_manifest_sha256, "records": records,
        "exact_new_phase_replay_pass": True, "csv_bytes_exact_replay": True, "protected_permissions_pass": True,
        "new_model_calls": 0, "source_bodies_parsed": False, "image_pixels_decoded": False,
        "distinct_policy_outcomes_claimed": False, "reader_pins": {**pr.pins, **sr.pins, **extra_pins},
        "auditor_source_sha256": sha256_file(__file__)}
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(tmp / "audit.json", audit)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "completed_cpu_only_shared_report_postflight", "audit_sha256": sha256_file(path), "new_model_calls": 0})
        commit_atomic_run(tmp, target)
    except BaseException: discard_atomic_run(tmp); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-run", "source-run", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-manifest-sha256", "source-manifest-sha256", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        target = build(args)
        print(json.dumps({"stage": VERSION, "status": "completed_cpu_only",
            "manifest_sha256": sha256_file(target / "manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
