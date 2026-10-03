#!/usr/bin/env python3
"""CPU Slurm post-run audit: metadata/hashes only, no model calls or body review."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json)
from tricompose_v12.live_plan import load_plan, anchor_from_record
from tricompose_v12.live_workers import check_pins, single_generated
from tricompose_v12.live_execution import validate_cxr_binding, validate_report_binding, score_csv
from tricompose_v12.live_receipts import image_receipt, completed_receipt, PROFILE
from tricompose_v12.execution_ledger import restore_ledger
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.runtime_dispatch import require_slurm


def private_modes(root):
    for p in (root, *root.rglob("*")):
        info = p.lstat()
        if (stat.S_ISLNK(info.st_mode) or info.st_gid not in (96293, 65534)
                or stat.S_IMODE(info.st_mode) != (0o2770 if p.is_dir() else 0o660)):
            raise ValueError("project-private modes/groups or symlink scope differ")


def run(args):
    require_slurm()  # Checkpoint rehashing happens only inside the CPU allocation.
    plan = load_plan(args.plan_run, args.plan_manifest_sha256)
    root = require_inside(args.run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/"manifest.json")
    if (manifest["schema_version"] != "tricompose-live-frozen-worker-smoke-v1"
            or manifest["status"] != "completed_fixed_paths_unvalidated"
            or manifest["plan_manifest_sha256"] != args.plan_manifest_sha256
            or manifest["completed_triplets"] != 4 or manifest["charged_model_attempts"] != 16
            or manifest["validated_model_calls"] != 16 or manifest["failed_attempts"] != 0
            or manifest["selection_or_adaptive_repair_executed"] is not False
            or manifest["clinical_acceptance"] is not False):
        raise ValueError("expected completed, non-adaptive four-triplet smoke")
    for name, record in manifest["artifacts"].items():
        p = require_inside(root/name, root, must_exist=True)
        if sha256_file(p) != record["sha256"]: raise ValueError("published artifact hash changed")
    private_modes(root)
    for spec in plan["workers"].values():
        if spec["status"] == "preflighted": check_pins(spec["asset_pins"])
    triples = [json.loads(line) for line in (root/"completed_triplets.jsonl").read_text().splitlines() if line]
    if len(triples) != 4 or len({r["receipt_id"] for r in triples}) != 4:
        raise ValueError("completed output inventory differs")
    if (root/"score_table.csv").read_bytes() != score_csv(triples).encode():
        raise ValueError("score CSV differs from raw receipt edges")
    counts = Counter(); memory = []; recomputed = {}
    for case in plan["cases"]:
        anchor = anchor_from_record(case["anchor"])
        cr = root/"cases"/anchor.case_id
        if read_json(cr/"ehr_anchor.json") != anchor.record(): raise ValueError("fixed EHR anchor changed")
        for name, expected in (("synthetic_ehr.json", anchor.ehr_sha256), ("ehr_facts.json", anchor.ehr_facts_sha256)):
            if sha256_file(cr/"inputs"/name) != expected: raise ValueError("fixed copied input changed")
        events = [json.loads(line) for line in (cr/"execution.journal.jsonl").read_text().splitlines() if line]
        book = restore_ledger(events, case_id=anchor.case_id, ehr_anchor_sha256=anchor.sha256,
            call_budget=plan["policy"]["call_budget_per_case"], max_retries=plan["policy"]["max_retries_per_operation"],
            execution_mode="approved_slurm_backend", sink=lambda e: None)
        if book.snapshot() != read_json(cr/"ledger_snapshot.json"):
            raise ValueError("durable journal restoration differs")
        if book.charged_attempts != 8 or book.snapshot()["pending_attempts"]:
            raise ValueError("unexpected charged or unfinished attempts")
        reserved = {e["reservation_id"]: e for e in events if e["event"] == "attempt_reserved"}
        state = {}
        for event in events:
            if event["event"] != "attempt_completed": continue
            start = reserved[event["reservation_id"]]; request = start["request"]
            spec = plan["workers"][request["model_id"]]
            if request["frozen_model_audit_sha256"] != _digest(spec): raise ValueError("charged frozen audit changed")
            op = cr/"operations"/(request["operation_id"]+"_a"+str(start["attempt_number"]))
            slot = int(request["operation_id"].split("_")[1])
            item = state.setdefault(slot, {})
            payload = {"output_root": str(op), "worker_audit_sha256": _digest(spec)}
            kind = request["kind"]; counts[kind] += 1
            if kind == "cxr_generator":
                pack = cr/"operations"/f"request_cxr_{slot}"
                image_run, cxr = single_generated(payload, spec, pack)
                validate_cxr_binding(cxr, case["requests"][slot], anchor)
                prompt = case["requests"][slot]["request"]["inputs"]["final_prompt"]
                if sha256_file(cr/"inputs/cxr_prompts"/(cxr["model_id"]+".txt")) != prompt["sha256"]:
                    raise ValueError("copied fixed prompt changed")
                if event["result"]["output_artifact_sha256"] != cxr["artifact"]["sha256"]:
                    raise ValueError("journal/image output hash differs")
                item.update(cxr=cxr, cxr_run=image_run)
                memory.append(cxr["cost"]["peak_vram_gib"])
            elif kind == "xrv":
                file = op/"scored/cxr_finding_labels.json"; labels = read_json(file)
                if (labels["thresholds"] != spec["thresholds"]
                        or labels["calibration"]["status"] != spec["calibration_status"]
                        or any(labels["producer"].get(k) != v for k,v in spec["scorer_provenance"].items())):
                    raise ValueError("fresh XRV protocol differs")
                receipt = image_receipt(anchor, item["cxr"], labels, label_sha256=sha256_file(file),
                    thresholds_sha256=spec["thresholds_sha256"], checkpoint_sha256=spec["checkpoint_sha256"])
                if (read_json(op/"image_receipt.json") != receipt
                        or event["result"]["verification_receipt_id"] != receipt["receipt_id"]
                        or event["result"]["output_artifact_sha256"] != sha256_file(file)):
                    raise ValueError("saved partial image receipt differs")
                item.update(image_labels=labels, image_labels_path=file, partial=receipt)
                memory.append(labels["peak_vram_gib"])
            elif kind == "report_generator":
                r_index = int(request["operation_id"].split("_")[2])
                pack = cr/"operations"/f"request_report_{slot}_{r_index}"
                report_run, report = single_generated(payload, spec, pack,
                    cxr_candidates={item["cxr"]["candidate_id"]: item["cxr"]})
                validate_report_binding(report, item["cxr"], pack, anchor)
                if event["result"]["output_artifact_sha256"] != report["artifact"]["sha256"]:
                    raise ValueError("journal/report output hash differs")
                item.update(report=report, report_run=report_run)
                memory.append(read_json(report_run/"manifest.json")["peak_vram_gib"])
            else:
                file = op/"scored/report_finding_labels.json"; labels = read_json(file)
                xrv = plan["workers"]["xrv"]
                receipt = completed_receipt(anchor, item["partial"], item["cxr"], item["image_labels"],
                    item["report"], labels, image_labels_sha256=sha256_file(item["image_labels_path"]),
                    report_labels_sha256=sha256_file(file), thresholds_sha256=xrv["thresholds_sha256"],
                    xrv_checkpoint_sha256=xrv["checkpoint_sha256"], chexbert_checkpoint_sha256=spec["checkpoint_sha256"])
                if (read_json(op/"completed_receipt.json") != receipt
                        or event["result"]["verification_receipt_id"] != receipt["receipt_id"]
                        or event["result"]["output_artifact_sha256"] != sha256_file(file)):
                    raise ValueError("saved completed receipt differs")
                recomputed[receipt["receipt_id"]] = receipt
                memory.append(labels["peak_vram_gib"])
    if counts != Counter({k: 4 for k in ("cxr_generator", "xrv", "report_generator", "chexbert")}):
        raise ValueError("per-worker invocation inventory differs")
    for row in triples:
        r = recomputed[row["receipt_id"]]
        if (row["raw_edge_readouts"] != r["raw_edge_readouts"] or row["profile"] != PROFILE
                or row["cxr_sha256"] != r["cxr_sha256"] or row["report_sha256"] != r["report_sha256"]
                or row["ehr_anchor_sha256"] != r["ehr_anchor_sha256"]
                or sha256_file(row["receipt_path"]) != row["receipt_sha256"]
                or row["clinical_acceptance"] is not False or row["selected_as_best"] is not False):
            raise ValueError("published triple/score lineage differs")
    summary = {"schema_version": "tricompose-live-smoke-postrun-audit-v1", "status": "metadata_hash_receipt_audit_passed",
        "source_run_manifest_sha256": sha256_file(root/"manifest.json"), "fixed_ehr_cases": 2,
        "completed_triplets": len(triples), "validated_model_calls": sum(counts.values()),
        "completed_by_kind": dict(counts), "failed_attempts": 0, "retries": 0,
        "maximum_worker_peak_allocated_vram_gib": max(memory),
        "runtime_seconds_includes_startup_and_io": manifest["runtime_seconds_includes_startup_and_io"],
        "receipt_status_counts": dict(Counter(r["verification_status"] for r in triples)),
        "direct_ehr_known_fact_count_distribution": dict(Counter(r["raw_edge_readouts"]["ehr_cxr"]["known_reference_facts"] for r in triples)),
        "private_modes_and_groups_passed": True, "journal_restore_passed": True,
        "receipts_independently_recomputed_from_same_proxy_evidence": True,
        "independent_clinical_truth_available": False, "new_audit_model_calls": 0,
        "report_bodies_or_image_pixels_reviewed": False, "clinical_acceptance": False,
        "adaptive_selection_or_repair_executed": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        result = write_private_json(temporary/"audit_summary.json", summary)
        write_private_json(temporary/"manifest.json", {"schema_version": summary["schema_version"],
            "source_run": str(root), "source_run_manifest_sha256": summary["source_run_manifest_sha256"],
            "audit_program_sha256": sha256_file(__file__), "plan_manifest_sha256": args.plan_manifest_sha256,
            "audit_summary_sha256": sha256_file(result), "new_model_calls": 0})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-run", "plan-manifest-sha256", "run", "output-root", "run-id"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007)
    try: target, summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "audit_failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds_includes_startup_and_io"],
        "peak_vram_gib": summary["maximum_worker_peak_allocated_vram_gib"],
        "manifest_sha256": sha256_file(target/"manifest.json")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
