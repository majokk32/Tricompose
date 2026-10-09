#!/usr/bin/env python3
"""Decompose an audited image-probe veto; never loosen or rerank the gate.

CPU metadata/byte hashes only. Neither classifier states nor BioViL cosine
establish clinical truth. This diagnostic identifies what needs verification,
not which modality is clinically wrong or a new regeneration authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import run_fresh_cxr_secondary as endpoint
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    sha256_file, write_private_json, write_private_text)
from tricompose_v12 import bounded_regeneration as image_gate
from tricompose_v12.invariant_verification import _digest, EXPLICIT
from tricompose_v12.report_nbest import RISK_FLAGS

VERSION = "tricompose-fresh-cxr-probe-diagnostic-v1"
ENDPOINT_PLAN = endpoint.BASE / "llm_fresh_cxr_endpoint_plans/fresh_cxr2_biovil_12817261_001"
ENDPOINT_PLAN_SHA = "914e06ae76e3788523c4910d52b696e1d0bb3a62a63c1b8a8a9f65dbfa11e917"
ENDPOINT_RUN = endpoint.BASE / "llm_fresh_cxr_endpoints/fresh_cxr2_biovil_12840340"
ENDPOINT_SHA = "035657ccb53c6a2937d32ab05ce33b07104b74ce5c1db94e0e20570a73c2f9e8"
require = endpoint.require


def decompose(before, probe, transition, before_cosine, probe_cosine):
    """Replay the frozen comparison and name its individual quality checks."""
    require(image_gate.compare(before, probe) == transition, "recorded_gate_must_replay")
    s, t = before["structure"], probe["structure"]
    quality = {"section_and_nonempty_pass": image_gate.structure_ok(probe),
        "risk_flags_not_worse": all(not t[k] or s[k] for k in RISK_FLAGS),
        "repeated_sentence_count_not_worse": t["repeated_sentence_count"] <= s["repeated_sentence_count"],
        "repeated_4gram_ratio_not_worse": t["repeated_4gram_ratio"] <= s["repeated_4gram_ratio"]}
    require(all(quality.values()) == transition["report_structure_and_risk_no_worse"],
        "quality_predicate_arithmetic_changed")
    checks = {"fixed_ehr_image_gain": bool(transition["gained_support_fact_ids"]["ehr_cxr"]
            or transition["removed_opposition_fact_ids"]["ehr_cxr"]),
        "previous_fact_sets_preserved": not any(v for e in transition["lost_fact_ids"].values() for v in e.values()),
        "no_new_proxy_opposition": not any(transition["new_opposition_fact_ids"].values()),
        "report_quality_not_worse": all(quality.values()), "image_not_duplicate": not transition["duplicate_image"]}
    require(all(checks.values()) == transition["exploratory_gate_pass"], "gate_predicate_arithmetic_changed")
    delta = probe_cosine - before_cosine if before_cosine is not None and probe_cosine is not None else None
    edges = {}
    for edge in ("ehr_cxr", "ehr_report", "cxr_report"):
        edges[edge] = {"before": before["raw_edge_readouts"][edge], "probe": probe["raw_edge_readouts"][edge],
            "lost_comparable_count": len(transition["lost_fact_ids"][edge]["comparable"]),
            "lost_support_count": sum(len(v) for k,v in transition["lost_fact_ids"][edge].items() if k != "comparable"),
            "new_opposition_count": len(transition["new_opposition_fact_ids"][edge]),
            "removed_opposition_count": len(transition["removed_opposition_fact_ids"][edge]),
            "opposition_silenced_by_lost_comparison_count": len(set(transition["removed_opposition_fact_ids"][edge])
                & set(transition["lost_fact_ids"][edge]["comparable"]))}
    return {"case_id": before["case_id"], "gate_checks": checks, "quality_checks": quality,
        "failed_checks": [k for k,v in checks.items() if not v],
        "actual_quality_failures": [k for k,v in quality.items() if not v],
        "before_4gram_ratio": s["repeated_4gram_ratio"], "probe_4gram_ratio": t["repeated_4gram_ratio"],
        "edges": edges, "before_cosine": before_cosine, "probe_cosine": probe_cosine,
        "cosine_delta": delta, "higher_cosine_but_vetoed": bool(delta is not None and delta > 0 and not all(checks.values())),
        "known_ehr_constraints": before["raw_edge_readouts"]["ehr_cxr"]["known_reference_facts"],
        "clinical_fault_location": None, "clinical_accuracy": None,
        "verification_requirement": "independent_image_side_evidence_before_claiming_fault_or_more_targeted_repair",
        "selection_changed": False, "thresholds_changed": False}


def label_readout(row, pins):
    """Read only the authenticated synthetic-image classifier metadata."""
    expected = row["receipt"]["xrv_labels_sha256"]
    paths = [Path(p) for p,h in pins.items() if h == expected and Path(p).name == "cxr_finding_labels.json"]
    require(len(paths) == 1, "one_authenticated_image_label_artifact_required")
    reader = endpoint.previous.postflight.MetadataReader(paths[0].parent)
    reader.hash(paths[0], expected)
    labels = reader.json(paths[0])
    require(labels["score_semantics"] == "xrv_op_norm_0_1" and labels["probability_semantics"] is False,
        "operating_point_scores_not_probabilities")
    records = labels["records"]
    matches = [r for r in records if r["cxr_candidate_id"] == row["cxr_candidate_id"]]
    require(len(matches) == 1 and matches[0]["image_sha256"] == row["cxr_sha256"], "classifier_image_lineage_changed")
    record = matches[0]
    result = []
    for i, fact in enumerate(row["receipt"]["fact_states"]):
        if fact["ehr"] not in EXPLICIT: continue
        name = fact["finding"]
        threshold = labels["thresholds"][name]
        value = record["finding_probabilities"][name]  # legacy key; not calibrated probabilities
        require(record["finding_states"][name] == fact["xrv"], "classifier_state_changed")
        result.append({"fact_id": f"f{i:04d}", "ehr_state": fact["ehr"], "xrv_state": fact["xrv"],
            "enabled": threshold["enabled"], "op_norm_score": value,
            "negative_max": threshold["negative_max"], "positive_min": threshold["positive_min"]})
    reader.recheck()
    return result, reader.pins


def diagnose(args):
    endpoint.previous.gate.cpu_guard()
    ar = endpoint.previous.postflight.MetadataReader(endpoint.AUDIT)
    sr = endpoint.previous.postflight.MetadataReader(endpoint.SOURCE)
    pr = endpoint.previous.postflight.MetadataReader(ENDPOINT_PLAN)
    er = endpoint.previous.postflight.MetadataReader(ENDPOINT_RUN)
    readers = (ar, sr, pr, er)
    for reader, wanted in ((ar, endpoint.AUDIT_SHA), (sr, endpoint.SOURCE_SHA), (pr, ENDPOINT_PLAN_SHA), (er, ENDPOINT_SHA)):
        reader.hash(reader.root / "manifest.json", wanted)
        manifest = reader.json(reader.root / "manifest.json")
        for name,pin in manifest.get("artifacts", {}).items(): reader.hash(reader.root/name, pin["sha256"])
    audit = ar.json(ar.root / "audit.json")
    require(audit["status"] == "metadata_and_ledger_verified_not_clinical"
        and audit["source_manifest_sha256"] == endpoint.SOURCE_SHA, "completed_numeric_source_audit_required")
    pm = pr.json(pr.root / "manifest.json")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"])
    plan = pr.json(pr.root / "plan.json")
    for field in ("source_pins", "artifact_pins", "model_pins"): endpoint.check_pins(plan[field])
    rows = sr.json(sr.root / "score_rows.json")["records"]
    require(rows == plan["rows"], "sealed_endpoint_candidate_inventory_changed")
    scores = er.json(er.root / "scores.json")
    table, pairs, summary = endpoint.endpoint_tables(rows, plan["controls"], scores)
    require(summary == er.json(er.root / "summary.json"), "endpoint_summary_changed")
    for name, data in (("score_table.csv",table), ("case_comparison.csv",pairs)):
        require(sha256_file(er.root/name) == hashlib.sha256(endpoint.previous.csv_text(data).encode()).hexdigest(),
            "endpoint_table_arithmetic_changed")
    selections = sr.json(sr.root / "selection.json")["records"]
    findings, label_pins = [], {}
    for selected in selections:
        cid = selected["case_id"]
        before = next(r for r in rows if r["triple_candidate_id"] == selected["reference_candidate_id"])
        probes = [r for r in rows[8:] if r["case_id"] == cid]
        require(len(probes) == 1, "diagnostic_requires_one_completed_probe_per_case")
        journal = sr.journal(sr.root / "policy" / cid / "live_policy.journal.jsonl")
        completed = [e for e in journal if e["event"] == "fresh_image_finished"]
        require(len(completed) == 1, "one_completed_feedback_required")
        pair = next(p for p in pairs if p["case_id"] == cid)
        result = decompose(before, probes[0], completed[0]["retained_reference_transition"],
            pair["before_raw_cosine"], pair["probe_raw_cosine"])
        for role,row in (("before",before), ("probe",probes[0])):
            result[role+"_classifier_readouts"], pins = label_readout(row, audit["artifact_pins"])
            label_pins.update(pins)
        result["recorded_baseline_veto_reasons"] = completed[0]["original_baseline_veto"]["rejection_reason_codes"]
        findings.append(result)
    for reader in readers: reader.recheck()
    endpoint.check_pins(audit["source_pins"]); endpoint.check_pins(audit["artifact_pins"])
    report = {"schema_version": VERSION, "status": "numeric_veto_decomposed_not_clinical_localization",
        "case_results": findings, "cases": len(findings), "new_model_calls": 0,
        "training_performed": False, "clinical_fault_localization_validated": False,
        "source_bodies_parsed": False, "image_pixels_decoded": False,
        "thresholds_changed": False, "selections_changed": False,
        "source_pins": {**plan["source_pins"], str(Path(__file__).resolve()): sha256_file(__file__),
            str(endpoint.ROOT / "TriCompose-v1.2/tests/test_probe_veto_diagnostic.py"):
                sha256_file(endpoint.ROOT / "TriCompose-v1.2/tests/test_probe_veto_diagnostic.py")},
        "artifact_pins": {**label_pins, **{p:h for r in readers for p,h in r.pins.items()}}}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temporary / "diagnosis.json", report)
        records = [{"case_id": r["case_id"], "failed_gate_checks": ";".join(r["failed_checks"]),
            "actual_quality_failures": ";".join(r["actual_quality_failures"]),
            "known_ehr_constraints": r["known_ehr_constraints"], "before_cosine": r["before_cosine"],
            "probe_cosine": r["probe_cosine"], "cosine_delta": r["cosine_delta"],
            "before_4gram_ratio": r["before_4gram_ratio"], "probe_4gram_ratio": r["probe_4gram_ratio"],
            "clinical_fault_location": None} for r in findings]
        csv = write_private_text(temporary / "veto_breakdown.csv", endpoint.previous.csv_text(records))
        write_private_json(temporary / "manifest.json", {"schema_version": VERSION, "status": report["status"],
            "artifacts": {f.name:{"sha256":sha256_file(f)} for f in (p,csv)}, "new_model_calls":0,
            "clinical_acceptance":False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--run-id", required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        target = diagnose(args)
        print(json.dumps({"stage":VERSION,"status":"completed","manifest_sha256":sha256_file(target/"manifest.json")},sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage":VERSION,"status":"failed_closed"},sort_keys=True)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
