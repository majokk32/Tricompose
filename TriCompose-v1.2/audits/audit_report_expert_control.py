#!/usr/bin/env python3
"""CPU Slurm audit of the fixed-image, four-cached-expert control.

Read only synthetic metadata, derived labels and hashes, never report bodies or
image pixels. Recompute receipts, decisions, secondary arithmetic and CSV cells.
The structure flags are authenticated worker outputs, not re-executed here.
"""
import argparse
from collections import Counter
import csv
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    load_cxr_candidates, load_report_candidates, new_atomic_run, write_private_json,
    commit_atomic_run, discard_atomic_run)
from run_report_expert_control import load
from run_report_nbest import single_view
from compare_frozen_report_paths import flatten
from score_automatic_replay_biovil import PAIR_FIELDS
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.live_workers import check_pins
from tricompose_v12.report_expert_control import SCHEMA, MODELS, freeze, measure
from tricompose_v12.runtime_dispatch import require_slurm


def aggregate(rows, endpoint):
    """Independently sum every raw edge; zero denominators remain unavailable."""
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    totals = []
    fields = ("known_reference_facts", "comparable_facts", "supported_positive",
              "supported_negative", "proxy_opposition_facts", "missing_comparisons")
    for model in MODELS:
        group = [r for r in rows if r["report_model_id"] == model]
        if len(group) != 2:
            raise ValueError("two reports per frozen expert required")
        edges = {}
        for name in ("ehr_cxr", "ehr_report", "cxr_report"):
            edge = {k: sum(r["raw_edge_readouts"][name][k] for r in group) for k in fields}
            known = edge["known_reference_facts"]
            edge.update(coverage_over_known=edge["comparable_facts"] / known if known else None,
                support_over_known=(edge["supported_positive"] + edge["supported_negative"]) / known if known else None,
                opposition_over_known=edge["proxy_opposition_facts"] / known if known else None)
            edges[name] = edge
        values = [scores[r["triple_candidate_id"]]["biovil_raw_cosine"] for r in group]
        totals.append({"report_model_id": model, "fixed_ehr_cases": 2, "report_candidates": 2,
            "biovil_available_pairs": sum(v is not None for v in values),
            "two_case_mean_biovil_raw_cosine": sum(values) / 2 if all(v is not None for v in values) else None,
            "official_structure_pass_count": sum(r["structure"]["section_contract_pass"] for r in group),
            "unsupported_temporal_language_count": sum(r["structure"]["unsupported_temporal_comparison_language"] for r in group),
            "raw_edge_totals": edges, "clinical_accuracy": None, "clinical_acceptance": False})
    return totals


def check_csv(path, expected, key):
    """Require the exact complete table, including every explicit NA cell."""
    actual = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    index = {r[key]: r for r in actual}
    if len(actual) != len(expected) or len(index) != len(expected):
        raise ValueError("CSV inventory differs")
    for row in expected:
        rendered = {k: "NA" if v is None else str(v) for k, v in row.items()}
        if index.get(str(row[key])) != rendered:
            raise ValueError("CSV value, rate, NA or lineage differs")


def audit(args):
    require_slurm()
    started = time.monotonic()
    root = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    mp = root / "manifest.json"
    manifest_sha = sha256_file(mp)
    m = read_json(mp)
    if (m.get("schema_version") != SCHEMA
            or m.get("status") != "completed_cached_four_expert_control_unvalidated"
            or any(m.get(k) is not False for k in ("clinical_acceptance", "adaptive_repair_executed",
                "original_winners_changed", "same_image_reports_are_independent_votes", "biovil_used_for_selection"))
            or m.get("new_generation_calls") != 0 or m.get("new_xrv_calls") != 0
            or m.get("reused_xrv_scored_images") != 2 or m.get("new_chexbert_scored_reports") != 8
            or m.get("chexbert_forward_batch_size") != 8
            or m.get("legacy_scorer_model_calls_are_sample_counts") is not True
            or m.get("report_generation_costs_are_historical_sunk_costs_not_zero") is not True):
        raise ValueError("completed cached-only, nonclinical control required")
    args.plan_manifest_sha256 = m["plan_manifest_sha256"]
    plan = load(args)
    for name, entry in m["artifacts"].items():
        if sha256_file(require_inside(root / name, root, must_exist=True)) != entry["sha256"]:
            raise ValueError("published artifact changed")
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or info.st_gid not in (96293, 65534)
                or stat.S_IMODE(info.st_mode) != (0o2770 if path.is_dir() else 0o660)):
            raise ValueError("project-private permission/group differs")
    for spec in (plan["xrv"], plan["chexbert"]):
        check_pins(spec["asset_pins"])
    check_pins(plan["biovil_asset_pins"])

    cxrs = load_cxr_candidates([plan["cxr_run"]])
    reports = load_report_candidates(plan["report_runs"], cxr_candidates=cxrs)
    ip = Path(plan["reused_xrv_labels"])
    tp = root / "labels/report_finding_labels.json"
    if sha256_file(tp) != m["chexbert_labels_sha256"]:
        raise ValueError("fresh report-label bundle changed")
    il, tl = read_json(ip), read_json(tp)
    if (len(cxrs) != 2 or len(reports) != 8
            or il["counts"] != {"cxr_candidates": 2, "model_calls": 2}
            or tl["counts"] != {"reports": 8, "model_calls": 8}
            or len(tl["records"]) != 8
            or {r["report_candidate_id"] for r in tl["records"]} != set(reports)):
        raise ValueError("two-image eight-report inventory differs")
    structure_rows = read_json(root / "structure_records.json")["records"]
    structures = {s["report_candidate_id"]: s for s in structure_rows}
    if len(structure_rows) != 8 or set(structures) != set(reports):
        raise ValueError("worker structure inventory differs")
    frequency = Counter(s["normalized_report_sha256"] for s in structure_rows)
    if any(s["normalized_template_frequency"] != frequency[s["normalized_report_sha256"]] for s in structure_rows):
        raise ValueError("normalized template frequency differs")
    partials = {r["cxr_candidate_id"]: r for r in plan["image_receipts"]}
    rebuilt = []
    for case in plan["fixed_cases"]:
        anchor = anchor_from_record(case["anchor"])
        image = cxrs[case["image"]["candidate_id"]]
        if image != case["image"]:
            raise ValueError("fixed original image metadata changed")
        iv = single_view(il, "cxr_candidates", "cxr_candidate_id", image["candidate_id"])
        partial = image_receipt(anchor, image, iv, label_sha256=sha256_file(ip),
            thresholds_sha256=plan["xrv"]["thresholds_sha256"], checkpoint_sha256=plan["xrv"]["checkpoint_sha256"])
        if partial != partials[image["candidate_id"]]:
            raise ValueError("reused fresh image receipt differs")
        for report in reports.values():
            if report["parent_cxr_candidate_id"] != image["candidate_id"]:
                continue
            receipt = completed_receipt(anchor, partial, image, iv, report,
                single_view(tl, "reports", "report_candidate_id", report["candidate_id"]),
                image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp),
                thresholds_sha256=plan["xrv"]["thresholds_sha256"],
                xrv_checkpoint_sha256=plan["xrv"]["checkpoint_sha256"],
                chexbert_checkpoint_sha256=plan["chexbert"]["checkpoint_sha256"])
            rebuilt.append({"triple_candidate_id": "expertpair_" + _digest([image["candidate_id"], report["candidate_id"]])[:32],
                "case_id": anchor.case_id, "cxr_candidate_id": image["candidate_id"],
                "report_candidate_id": report["candidate_id"], "report_model_id": report["model_id"],
                "ehr_sha256": anchor.ehr_sha256, "ehr_facts_sha256": anchor.ehr_facts_sha256,
                "cxr_sha256": image["artifact"]["sha256"], "report_sha256": report["artifact"]["sha256"],
                "receipt": receipt, "structure": structures[report["candidate_id"]],
                "raw_edge_readouts": receipt["raw_edge_readouts"]})
    rebuilt.sort(key=lambda r: r["triple_candidate_id"])
    if read_json(root / "score_rows.json") != {"records": rebuilt}:
        raise ValueError("fresh source-bound score rows differ")
    selection = read_json(root / "selection.json")
    endpoint = read_json(root / "secondary.json")
    comparison = measure(selection, rebuilt, endpoint)
    methods = aggregate(rebuilt, endpoint)
    request = read_json(root / "secondary_request/request.json")
    request_manifest = read_json(root / "secondary_request/manifest.json")
    request_sha = sha256_file(root / "secondary_request/request.json")
    if (freeze(rebuilt) != selection or comparison != read_json(root / "comparison.json")
            or {"records": methods} != read_json(root / "method_summary.json")
            or sha256_file(root / "selection.json") != m["selection_sha256_before_endpoint"]
            or request["selection_sha256"] != m["selection_sha256_before_endpoint"]
            or request_sha != endpoint["request_sha256"]
            or request_sha != request_manifest["artifacts"]["request.json"]["sha256"]
            or request["pairs"] != [{k: r[k] for k in PAIR_FIELDS} for r in rebuilt]
            or request["selection_used_biovil"] is not False
            or request["routing_or_calibration_update_allowed"] is not False
            or request["clinical_truth_available"] is not False):
        raise ValueError("sealed decision, endpoint arithmetic or source request differs")
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    table = []
    for row in rebuilt:
        s = scores[row["triple_candidate_id"]]
        table.append(flatten({k: v for k, v in row.items() if k not in ("receipt", "structure")}) | {
            "biovil_raw_cosine": s["biovil_raw_cosine"], "biovil_status": s["status"], "biovil_na_reason": s["reason"],
            "official_section_contract_pass": row["structure"]["section_contract_pass"],
            "impression_required": row["structure"]["impression_required_by_model_contract"],
            "unsupported_temporal_language": row["structure"]["unsupported_temporal_comparison_language"],
            "generic_report": row["structure"]["generic_report"]})
    check_csv(root / "score_table.csv", table, "triple_candidate_id")
    check_csv(root / "model_comparison.csv", [flatten(r) for r in methods], "report_model_id")
    counts = endpoint["counts"]
    if (counts != m["secondary_counts"] or counts["requested_pairs"] != 8
            or counts["unavailable_reports"] != sum(r["biovil_raw_cosine"] is None for r in endpoint["records"])
            or not 0 <= counts["image_encoder_calls"] <= 2 or not 0 <= counts["text_encoder_calls"] <= 8
            or m["peak_vram_gib"] != {"chexbert": tl["peak_vram_gib"], "biovil": endpoint["peak_vram_gib"]}):
        raise ValueError("bounded scorer sample/encoding/memory accounting differs")
    for name in ("wall_seconds_including_startup_io",):
        if type(m[name]) not in (int, float) or not math.isfinite(m[name]) or m[name] < 0:
            raise ValueError("invalid runtime metadata")
    costs = [json.loads(line) for line in (root / "cost_journal.jsonl").read_text().splitlines()]
    if [(e.get("stage"), e["status"]) for e in costs] != [
            (stage, status) for stage in ("chexbert", "biovil")
            for status in ("reserved_before_spawn", "process_completed_unvalidated", "validated")]:
        raise ValueError("bounded reserved/completed/validated stage journal differs")
    if ([r["stage"] for r in m["worker_costs"]] != ["chexbert", "biovil"]
            or any(e["timeout_seconds"] != 180 for e in costs if e["status"] == "reserved_before_spawn")):
        raise ValueError("owned worker timing/bounds differ")
    check_pins(plan["source_pins"])
    check_pins(plan["artifact_pins"])
    if sha256_file(mp) != manifest_sha:
        raise ValueError("source manifest changed during audit")
    result = {"schema_version": "tricompose-report-expert-control-audit-v1",
        "status": "metadata_hash_receipt_audit_passed", "source_manifest_sha256": manifest_sha,
        "source_plan_manifest_sha256": m["plan_manifest_sha256"], "fixed_ehr_cases": 2,
        "fixed_images": 2, "report_candidates": 8, "new_generation_calls": 0, "new_xrv_calls": 0,
        "fresh_chexbert_scored_reports": 8,
        "gate_passing_alternatives": sum(p["exploratory_gate_pass"] for p in selection["comparisons"]),
        "changed_selected_images": sum(c["baseline_triple_id"] != c["selected_triple_id"] for c in selection["choices"]),
        "direct_ehr_known_facts_per_baseline": [r["receipt"]["known_ehr_facts"] for r in rebuilt if r["report_model_id"] == "cxrmate_single"],
        "raw_edge_csv_and_model_totals_verified": True,
        "decisions_and_receipts_recomputed": True, "sealed_secondary_arithmetic_recomputed": True,
        "structure_scope": "authenticated_worker_metadata_not_body_reexecution",
        "source_bodies_or_image_pixels_opened": False, "new_model_calls": 0,
        "clinical_acceptance": False, "clinical_repair_success": False,
        "runtime_seconds": round(time.monotonic() - started, 3)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(temporary / "audit.json", result)
        write_private_json(temporary / "manifest.json", {"schema_version": result["schema_version"],
            "audit_sha256": sha256_file(path), "source_manifest_sha256": manifest_sha,
            "auditor_sha256": sha256_file(Path(__file__))})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "plan-run", "output-root", "run-id"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    os.umask(0o007)
    try:
        root, result = audit(args)
    except Exception as exc:
        print(json.dumps({"status": "audit_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"],
        "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
