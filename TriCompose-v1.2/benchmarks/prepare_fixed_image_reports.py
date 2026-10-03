#!/usr/bin/env python3
"""CPU Slurm: pin a same-image two-report control; no model instantiation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, write_private_json, discard_atomic_run)
from tricompose_v12.live_workers import registry, preflight_generator, source_pins, check_pins
from tricompose_v12.live_plan import load_plan
from tricompose_v12.fixed_image_reports import POLICY
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v11.report_contracts import prepare_report_request_run
from score_automatic_replay_biovil import MODEL_HASHES

SCHEMA = "tricompose-fixed-image-report-plan-v1"
BIOVIL_MODEL = WORKSPACE/"CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
BIOVIL_PYTHON = WORKSPACE/"CheXGenBench/metrics_venv/bin/python"


def prepare(args):
    require_slurm()
    original = load_plan(args.source_plan, args.source_plan_sha256)
    source = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    audit = require_inside(args.source_audit, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(source/"manifest.json")
    source_hash = sha256_file(source/"manifest.json")
    proof = read_json(audit/"audit_summary.json")
    if (manifest.get("status") != "completed_fixed_paths_unvalidated"
            or manifest.get("completed_triplets") != 4 or manifest.get("charged_model_attempts") != 16
            or manifest.get("clinical_acceptance") is not False
            or proof.get("source_run_manifest_sha256") != source_hash
            or proof.get("status") != "metadata_hash_receipt_audit_passed"):
        raise ValueError("audited completed source smoke required")
    am = read_json(audit/"manifest.json")
    if sha256_file(audit/"audit_summary.json") != am["audit_summary_sha256"]:
        raise ValueError("source audit hash differs")
    pins = dict(original["source_artifact_pins"])
    for name, entry in manifest["artifacts"].items():
        path = require_inside(source/name, source, must_exist=True)
        if sha256_file(path) != entry["sha256"]: raise ValueError("source artifact changed")
        pins[str(path)] = entry["sha256"]
    triples = [json.loads(x) for x in (source/"completed_triplets.jsonl").read_text().splitlines() if x]
    cxr_runs = [r["cxr_run"] for r in triples]; report_runs = [r["report_run"] for r in triples]
    if len(triples) != 4 or len(set(cxr_runs)) != 4 or len({r["case_id"] for r in triples}) != 2:
        raise ValueError("fixed two-case four-image source required")
    for row in triples:
        if row["report_model_id"] != "cxrmate_single": raise ValueError("fixed baseline differs")
        for key in ("cxr", "report", "receipt"):
            path = require_inside(row[key+"_path"], source, must_exist=True)
            if sha256_file(path) != row[key+"_sha256"]: raise ValueError("source lineage differs")
            pins[str(path)] = row[key+"_sha256"]
        receipt = Path(row["receipt_path"])
        # Exact files bound by the completed source audit; no EHR/report bodies opened.
        labels = receipt.parent/"scored/report_finding_labels.json"
        slot = receipt.parent.name.split("_")[1]
        if slot not in ("0", "1"): raise ValueError("unexpected original image slot")
        partial = source/"cases"/row["case_id"]/"operations"/("xrv_"+slot+"_a1")/"image_receipt.json"
        for path in (labels, partial):
            path = require_inside(path, source, must_exist=True)
            pins[str(path)] = sha256_file(path)
        row["baseline_label_path"] = str(labels); row["partial_receipt_path"] = str(partial)
    maira = preflight_generator(registry()["maira2"])
    chexbert = original["workers"]["chexbert"]; check_pins(chexbert["asset_pins"])
    biovil_pins = {}
    for name, digest in MODEL_HASHES.items():
        path = BIOVIL_MODEL/name
        if sha256_file(path) != digest: raise ValueError("frozen BioViL weight differs")
        biovil_pins[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}
    for path in BIOVIL_MODEL.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".txt", ".py"}:
            biovil_pins[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    vendor = WORKSPACE/"runtime/vendor/hi_ml_multimodal_0_2_2"
    vendor_sources = sorted(vendor.rglob("*.py"))
    if not vendor_sources: raise ValueError("local BioViL runtime source missing")
    for path in vendor_sources:
        biovil_pins[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    for python in (maira["python"], chexbert["python"], BIOVIL_PYTHON):
        if not os.access(python, os.X_OK): raise ValueError("required environment not executable")
    for path in (source/"manifest.json", audit/"manifest.json", audit/"audit_summary.json"):
        pins[str(path)] = sha256_file(path)
    requests = prepare_report_request_run(cxr_runs=cxr_runs, output_root=args.output_root,
        run_id=args.run_id+"_requests", model_ids=["maira2"])
    request_root = Path(requests["run_directory"])
    for path in request_root.rglob("*.json"): pins[str(path)] = sha256_file(path)
    sources = source_pins()
    payload = {"schema_version": SCHEMA, "policy": POLICY, "source_run": str(source),
        "source_run_manifest_sha256": source_hash, "source_audit_manifest_sha256": sha256_file(audit/"manifest.json"),
        "source_charged_calls": 16, "fixed_triplets": triples, "cxr_runs": cxr_runs, "baseline_report_runs": report_runs,
        "request_run": str(request_root), "maira": maira, "chexbert": chexbert,
        "biovil_model": str(BIOVIL_MODEL), "biovil_python": str(BIOVIL_PYTHON),
        "biovil_asset_pins": biovil_pins, "source_pins": sources, "artifact_pins": pins,
        "new_model_calls": 0, "factory_instantiated": False,
        "planned_new_report_samples": 4, "planned_new_chexbert_samples": 4,
        "planned_biovil_image_calls": 4, "planned_biovil_text_calls": 8,
        "new_cxr_or_ehr_samples": 0, "clinical_acceptance": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        file = write_private_json(temporary/"plan.json", payload)
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA,
            "plan_sha256": sha256_file(file), "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-plan", "source-plan-sha256", "source-run", "source-audit", "output-root", "run-id"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007); started = time.monotonic()
    try: root = prepare(args)
    except Exception as exc:
        print(json.dumps({"status": "preflight_failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": "prepared_fixed_image_report_control", "new_model_calls": 0,
        "runtime_seconds": round(time.monotonic()-started, 3), "manifest_sha256": sha256_file(root/"manifest.json")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
