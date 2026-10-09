#!/usr/bin/env python3
"""CPU-only numeric postflight; no model, clinical body or pixel inspection.

Recompute sealed-arm tables and report gate/embedding disagreement without
changing any choice. This validates metadata/math, not clinical correctness
or independently reexecuted embedding computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import run_fresh_report_secondary as secondary
import audit_fresh_report_agent as postflight
from contracts import (PROTECTED_ROOT, new_atomic_run, commit_atomic_run,
    discard_atomic_run, sha256_file, write_private_json, write_private_text)
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_workers import check_pins

VERSION = "tricompose-fresh-report-secondary-postflight-v1"
require = postflight.require


def verify_payload(plan, manifest, endpoint, summary):
    require(manifest["schema_version"] == secondary.VERSION
        and manifest["status"] == "completed_independent_endpoint_unvalidated"
        and manifest["charged_endpoint_worker_attempts"] == 1
        and manifest["new_generator_calls"] == 0
        and manifest["endpoint_used_for_selection"] is False
        and manifest["clinical_acceptance"] is False, "completed_secondary_only_manifest_required")
    require(endpoint["producer"]["frozen"] is True
        and endpoint["producer"]["checkpoint_sha256"] == secondary.MODEL_HASHES
        and endpoint["producer"]["text_policy"] == "full_report_no_silent_truncation_overlength_is_na"
        and endpoint["original_selection_changed"] is False
        and endpoint["primary_clinical_metric"] is False, "frozen_secondary_profile_required")
    table, pairs, expected = secondary.endpoint_tables(plan["rows"], plan["controls"], endpoint)
    require(_digest(expected) == _digest(summary), "summary_not_reproducible")
    available = [r for r in endpoint["records"] if r["biovil_raw_cosine"] is not None]
    counts = {"requested_pairs": len(plan["rows"]),
        "image_encoder_calls": len({r["cxr_candidate_id"] for r in available}),
        "text_encoder_calls": len({r["report_candidate_id"] for r in available}),
        "unavailable_reports": len(plan["rows"]) - len(available)}
    require(_digest(counts) == _digest(endpoint["counts"]) == _digest(manifest["encoding_counts"])
        and counts["image_encoder_calls"] <= plan["maximum_image_encodings"]
        and counts["text_encoder_calls"] <= plan["maximum_text_encodings"], "encoding_inventory_or_budget_changed")
    require(type(manifest["runtime_seconds_including_load_io"]) in (float, int)
        and math.isfinite(manifest["runtime_seconds_including_load_io"])
        and manifest["runtime_seconds_including_load_io"] >= 0
        and type(endpoint["peak_vram_gib"]) in (float, int)
        and math.isfinite(endpoint["peak_vram_gib"]) and endpoint["peak_vram_gib"] >= 0
        and manifest["peak_vram_gib"] == endpoint["peak_vram_gib"], "invalid_resource_measurement")
    index = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    choices = {r["case_id"]: r for r in plan["controls"]["choices"]}
    diagnostics = []
    for comparison in plan["controls"]["comparisons"]:
        choice = choices[comparison["case_id"]]
        alt = comparison["current_baseline_comparison"]["alternative_triple_id"]
        a, b = index[choice["llm_candidate_id"]]["biovil_raw_cosine"], index[alt]["biovil_raw_cosine"]
        diagnostics.append({"case_id": comparison["case_id"], "report_model": comparison["report_model"],
            "presealed_static_eligible": comparison["eligible"], "selected_by_llm": choice["llm_candidate_id"] == alt,
            "retained_raw_cosine": a, "alternative_raw_cosine": b,
            "higher_cosine_but_gate_blocked": bool(not comparison["eligible"] and a is not None and b is not None and b > a),
            "baseline_veto_reason_codes": ";".join(comparison["baseline_veto"]["rejection_reason_codes"]),
            "structure_fact_gate_reason_codes": ";".join(comparison["current_baseline_comparison"]["reasons"])})
    return table, pairs, diagnostics


def audit(args):
    secondary.gate.cpu_guard()
    pr, sr = postflight.MetadataReader(args.plan_run), postflight.MetadataReader(args.source_run)
    pr.hash(pr.root / "manifest.json", args.plan_manifest_sha256)
    pm = pr.json(pr.root / "manifest.json")
    require(pm["schema_version"] == secondary.VERSION, "reviewed_endpoint_plan_required")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    plan = pr.json(pr.root / "plan.json")
    require(plan["source_manifest_sha256"] == secondary.SOURCE_SHA
        and plan["source_audit_manifest_sha256"] == secondary.AUDIT_SHA
        and len(plan["rows"]) == 8 and len(plan["controls"]["choices"]) == 2,
        "exact_original_two_case_endpoint_required")
    for field in ("source_pins", "artifact_pins", "model_pins"):
        check_pins(plan[field])
    sr.hash(sr.root / "manifest.json", args.run_manifest_sha256)
    manifest = sr.json(sr.root / "manifest.json")
    require(manifest["plan_manifest_sha256"] == args.plan_manifest_sha256
        and manifest["source_manifest_sha256"] == secondary.SOURCE_SHA
        and set(manifest["artifacts"]) == {"scores.json", "summary.json", "score_table.csv", "case_comparison.csv"},
        "source_plan_and_output_inventory_changed")
    for name, pin in manifest["artifacts"].items():
        sr.hash(sr.root / name, pin["sha256"])
    endpoint, summary = sr.json(sr.root / "scores.json"), sr.json(sr.root / "summary.json")
    table, pairs, diagnostics = verify_payload(plan, manifest, endpoint, summary)
    for name, rows in (("score_table.csv", table), ("case_comparison.csv", pairs)):
        expected = hashlib.sha256(secondary.csv_text(rows).encode("utf-8")).hexdigest()
        sr.hash(sr.root / name, expected)
    reservation = sr.json(sr.root / "attempt_reserved.json")
    require(reservation == {"schema_version": secondary.VERSION, "charged_endpoint_worker_attempts": 1,
        "maximum_image_encodings": 2, "maximum_text_encodings": 8,
        "status": "reserved_before_load_no_free_failed_work"}, "durable_endpoint_reservation_changed")
    pr.recheck(); sr.recheck()
    for field in ("source_pins", "artifact_pins", "model_pins"):
        check_pins(plan[field])
    report = {"schema_version": VERSION, "status": "metadata_and_sealed_tables_verified_not_clinical",
        "source_manifest_sha256": args.run_manifest_sha256, "plan_manifest_sha256": args.plan_manifest_sha256,
        "summary": summary, "case_pairs": pairs, "encoding_counts": endpoint["counts"],
        "higher_cosine_blocked_alternative_count": sum(r["higher_cosine_but_gate_blocked"] for r in diagnostics),
        "choices_not_changed_after_scoring": True, "gate_thresholds_not_changed": True,
        "new_model_calls": 0, "clinical_accuracy": None,
        "embedding_computation_independently_reexecuted": False,
        "artifact_pins": {**pr.pins, **sr.pins},
        "source_pins": {**plan["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
            str(secondary.ROOT / "TriCompose-v1.2/tests/test_fresh_report_secondary_postflight.py"):
                sha256_file(secondary.ROOT / "TriCompose-v1.2/tests/test_fresh_report_secondary_postflight.py")}}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temporary / "audit.json", report)
        d = write_private_text(temporary / "veto_diagnostics.csv", secondary.csv_text(diagnostics))
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION, "status": report["status"],
            "artifacts": {p.name: {"sha256": sha256_file(p)}, d.name: {"sha256": sha256_file(d)}},
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("run-manifest-sha256", "plan-manifest-sha256", "run-id"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        target = audit(args)
        print(json.dumps({"stage": VERSION, "status": "completed",
            "manifest_sha256": sha256_file(target / "manifest.json")}))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"})); return 1


if __name__ == "__main__":
    raise SystemExit(main())
