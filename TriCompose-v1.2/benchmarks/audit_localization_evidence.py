#!/usr/bin/env python3
"""Audit direct EHR evidence available to a three-modality localizer.

CPU-only and protected. Existing XRV/CheXbert labels helped construct the
interventions, so this is a coverage diagnostic, NOT localization accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from build_intervention_smoke import file_sha256
from build_targeted_interventions import validate_items
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       new_atomic_run, require_inside, sha256_file,
                       write_private_json)

SCHEMA = "tricompose-v12-direct-ehr-localization-coverage-v1"
EHR_EDGES_SCHEMA = "tricompose-ehr-edge-crossmodal-evaluation-v1.1"
EXPLICIT = frozenset({"positive", "negative"})


def triad_pattern(ehr: dict[str, str], cxr: dict[str, str],
                  report: dict[str, str]) -> dict:
    """Count comparable direct facts; never infer negative from unknown."""
    known = [name for name, value in ehr.items() if value in EXPLICIT]
    both = [name for name in known if cxr.get(name) in EXPLICIT and
            report.get(name) in EXPLICIT]
    image_supported_report_opposed = sum(
        ehr[name] == cxr[name] != report[name] for name in both)
    report_supported_image_opposed = sum(
        ehr[name] == report[name] != cxr[name] for name in both)
    both_supported = sum(ehr[name] == cxr[name] == report[name] for name in both)
    both_opposed = sum(ehr[name] != cxr[name] == report[name] for name in both)
    if image_supported_report_opposed and report_supported_image_opposed:
        pattern = "conflicting_locus_signals"
    elif image_supported_report_opposed:
        pattern = "report_candidate_signal"
    elif report_supported_image_opposed:
        pattern = "cxr_candidate_signal"
    elif not known:
        pattern = "abstain_no_direct_ehr_fact"
    elif not both:
        pattern = "abstain_no_three_way_comparable_fact"
    elif both_opposed:
        pattern = "abstain_both_modalities_oppose_ehr"
    else:
        pattern = "no_asymmetric_direct_contradiction"
    return {"pattern": pattern, "known_direct_fact_count": len(known),
            "three_way_comparable_fact_count": len(both),
            "image_supported_report_opposed": image_supported_report_opposed,
            "report_supported_image_opposed": report_supported_image_opposed,
            "both_supported": both_supported,
            "both_opposed": both_opposed}


def build_evidence_index(details: dict) -> tuple[dict, dict, dict]:
    if details.get("schema_version") != EHR_EDGES_SCHEMA:
        raise ValueError("wrong EHR edge evidence schema")
    records = details.get("records", {})
    ehr, images, reports = {}, {}, {}
    for row in records.get("ehr_cxr", []):
        case, image_id = row["case_id"], row["cxr_candidate_id"]
        states = row["ehr_finding_states"]
        if case in ehr and ehr[case] != states:
            raise ValueError("same EHR case has inconsistent direct facts")
        if image_id in images:
            raise ValueError("duplicate CXR edge record")
        ehr[case] = states
        images[image_id] = {"sha256": row["image_sha256"],
                            "states": row["cxr_finding_states"]}
    for row in records.get("ehr_report", []):
        case, report_id = row["case_id"], row["report_candidate_id"]
        if case not in ehr or ehr[case] != row["ehr_finding_states"]:
            raise ValueError("report edge changed the EHR direct facts")
        if report_id in reports:
            raise ValueError("duplicate report edge record")
        reports[report_id] = {"sha256": row["report_sha256"],
                              "states": row["report_finding_states"]}
    if not ehr or not images or not reports:
        raise ValueError("empty EHR edge evidence")
    return ehr, images, reports


def audit_items(items: dict[str, list[dict]], ehr: dict, images: dict,
                reports: dict) -> dict:
    validated = validate_items(items)
    key = {row["item_id"]: row for row in items["intervention_key"]}
    records = []
    for row in items["resolver"]:
        case = row["case_id"]
        image = images.get(row["displayed_cxr_candidate_id"])
        report = reports.get(row["displayed_report_candidate_id"])
        if case not in ehr or image is None or report is None:
            raise ValueError("missing three-modality evidence")
        if (image["sha256"] != row["displayed_cxr_sha256"] or
            report["sha256"] != row["displayed_report_sha256"]):
            raise ValueError("EHR edge artifact hash mismatch")
        evidence = triad_pattern(ehr[case], image["states"], report["states"])
        records.append({"item_id": row["item_id"], "case_id": case,
                        "benchmark_split": row["benchmark_split"], **evidence})
    if len(records) != validated["items"]:
        raise AssertionError("not all benchmark items were audited")
    groups = {}
    for record in records:
        answer = key[record["item_id"]]
        for split in ("overall", record["benchmark_split"]):
            groups.setdefault(split, {}).setdefault(answer["intervention_type"], []).append(record)
    summary = {}
    for split, arms in groups.items():
        summary[split] = {}
        for arm, rows in arms.items():
            distribution = Counter(row["pattern"] for row in rows)
            summary[split][arm] = {
                "items": len(rows),
                "items_with_direct_ehr_fact": sum(row["known_direct_fact_count"] > 0
                                                  for row in rows),
                "items_with_three_way_comparable_fact": sum(
                    row["three_way_comparable_fact_count"] > 0 for row in rows),
                "pattern_counts": dict(sorted(distribution.items())),
            }
    return {"records": records, "summary": summary}


def run(args: argparse.Namespace) -> dict:
    bank = require_inside(args.bank_run, PROTECTED_ROOT, must_exist=True)
    edge_run = require_inside(args.ehr_edges_run, PROTECTED_ROOT, must_exist=True)
    bank_manifest = require_inside(bank / "manifest.json", bank, must_exist=True)
    bank_meta = json.loads(bank_manifest.read_text(encoding="utf-8"))
    if (bank_meta.get("source", {}).get("coverage", {}).get("split_strategy") !=
            "positive_pleural_effusion_coverage_v1"):
        raise ValueError("requires stratified split-locked intervention bank")
    items = {}
    for name in ("blind_items", "resolver", "intervention_key"):
        path = require_inside(bank / f"{name}.jsonl", bank, must_exist=True)
        if file_sha256(path) != bank_meta.get("artifact_sha256", {}).get(name):
            raise ValueError("intervention bank hash mismatch")
        items[name] = [json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines() if line]
    edge_manifest = require_inside(edge_run / "manifest.json", edge_run,
                                   must_exist=True)
    edge_meta = json.loads(edge_manifest.read_text(encoding="utf-8"))
    detail_path = require_inside(edge_run / "ehr_edge_details.json", edge_run,
                                 must_exist=True)
    if file_sha256(detail_path) != edge_meta.get("artifacts", {}).get(
            "ehr_edge_details.json", {}).get("sha256"):
        raise ValueError("EHR edge evidence hash mismatch")
    details = json.loads(detail_path.read_text(encoding="utf-8"))
    ehr, images, reports = build_evidence_index(details)
    result = audit_items(items, ehr, images, reports)
    return {"schema_version": SCHEMA,
            "status": "diagnostic_coverage_label_reuse_not_independent",
            "source_bank_manifest_sha256": file_sha256(bank_manifest),
            "source_ehr_edges_sha256": file_sha256(detail_path),
            "summary": result["summary"],
            "records": result["records"],
            "clinical_localization_accuracy": None,
            "independent_evidence_validated": False,
            "construction_label_reuse": True,
            "policy_should_abstain_without_asymmetric_direct_evidence": True,
            "interpretation": "Counts direct three-way comparable facts and asymmetric signals only; reused labelers make this unsuitable for accuracy claims."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-run", required=True)
    parser.add_argument("--ehr-edges-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    started = time.monotonic()
    temporary = None
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = run(args)
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output = write_private_json(temporary / "coverage.json", payload)
        output_hash = sha256_file(output)
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_diagnostic", "run_id": args.run_id,
                      "output_sha256": output_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
