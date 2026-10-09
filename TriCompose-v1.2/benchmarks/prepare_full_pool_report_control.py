#!/usr/bin/env python3
"""CPU Slurm preflight: all 80 fixed EHRs / 240 CXRs / 960 cached reports.

No body/pixel reads, model factories, inference, endpoint-value reads or API.
Only cached direct EHR states are retained from the legacy score registry.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file, load_cxr_candidates,
    load_report_candidates, new_atomic_run, write_private_json, commit_atomic_run, discard_atomic_run)
from run_report_expert_control import load as load_expert_plan
from compare_frozen_report_paths import load_bank
from score_automatic_replay_biovil import PAIR_FIELDS, MODEL_HASHES, validate_pairs
from tricompose_v12.invariant_verification import anchor_from_cached_candidate
from tricompose_v12.full_bank_secondary import SCHEMA as FULL_SCHEMA
from tricompose_v12.full_pool_report_control import SCHEMA, POLICY, MODELS, CXR_MODELS, CASE_COUNT
from tricompose_v12.live_workers import source_pins, check_pins
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v11.cxr_contracts import validate_cxr_request, canonical_json_sha256


def prepare(args):
    require_slurm()
    before = source_pins()
    prior = load_expert_plan(argparse.Namespace(plan_run=args.expert_plan,
        plan_manifest_sha256=args.expert_plan_manifest_sha256))
    for spec in (prior["xrv"], prior["chexbert"]):
        check_pins(spec["asset_pins"])
    full = require_inside(args.full_bank_run, PROTECTED_ROOT, must_exist=True)
    mp = full / "manifest.json"
    if sha256_file(mp) != args.full_bank_manifest_sha256:
        raise ValueError("fixed full-bank manifest changed")
    fm = read_json(mp)
    if (fm.get("schema_version") != FULL_SCHEMA or fm.get("original_selection_changed") is not False
            or fm.get("clinical_acceptance") is not False):
        raise ValueError("unchanged nonclinical synthetic registry required")
    pins = {str(mp): args.full_bank_manifest_sha256,
        str(Path(args.expert_plan).resolve() / "manifest.json"): args.expert_plan_manifest_sha256,
        str(Path(args.expert_plan).resolve() / "plan.json"): sha256_file(Path(args.expert_plan) / "plan.json")}
    cxr_runs, report_runs = [], []
    for key, value in fm["source_paths"].items():
        if key.startswith(("cxr_manifest_", "report_manifest_")):
            path = require_inside(value, PROTECTED_ROOT, must_exist=True)
            if sha256_file(path) != fm["source_sha256"][key]:
                raise ValueError("full generation registry changed")
            pins[str(path)] = sha256_file(path)
            (cxr_runs if key.startswith("cxr_") else report_runs).append(str(path.parent))
    cxrs = load_cxr_candidates(cxr_runs)
    reports = load_report_candidates(report_runs, cxr_candidates=cxrs)
    if (len(cxr_runs) != 3 or len(report_runs) != 8 or len(cxrs) != 240 or len(reports) != 960
            or {r["model_id"] for r in cxrs.values()} != set(CXR_MODELS)
            or {r["model_id"] for r in reports.values()} != set(MODELS)):
        raise ValueError("exact three-generator/four-expert inventory required")
    # Bind all canonical EHR/facts/final-prompt bytes without parsing any body.
    requests = {}
    for run in cxr_runs:
        cm = read_json(Path(run) / "manifest.json")
        request_root = require_inside(cm["source_request_run"], PROTECTED_ROOT, must_exist=True)
        request_mp = request_root / "manifest.json"
        if sha256_file(request_mp) != cm["source_request_run_manifest_sha256"]:
            raise ValueError("original CXR request registry changed")
        pins[str(request_mp)] = sha256_file(request_mp)
        for entry in read_json(request_mp)["requests"]:
            key = (entry["case_id"], entry["model_id"], entry["seed"])
            if key in requests:
                continue
            path = require_inside(request_root / entry["path"], request_root, must_exist=True)
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("source CXR request changed")
            request = read_json(path)
            validate_cxr_request(request)
            if (request["case_id"], request["model_id"], request["seed"]) != key:
                raise ValueError("source request index differs")
            requests[key] = request
            pins[str(path)] = entry["sha256"]
            for value in request["inputs"].values():
                pins[value["path"]] = value["sha256"]
    ep = require_inside(fm["source_paths"]["historical_endpoint_manifest"], PROTECTED_ROOT, must_exist=True)
    if sha256_file(ep) != fm["source_sha256"]["historical_endpoint_manifest"]:
        raise ValueError("cached EHR-anchor registry changed")
    bank, _, sources = load_bank(argparse.Namespace(endpoint_run=ep.parent))
    pins.update({str(p): sha256_file(p) for p in sources.values()})
    if len(bank) != CASE_COUNT:
        raise ValueError("all original eighty EHRs required")
    anchors, pairs = [], []
    for ordinal, (case, grid) in enumerate(sorted(bank.items())):
        if set(grid) != {(c, 0, r) for c in CXR_MODELS for r in MODELS}:
            raise ValueError("complete original per-case grid required")
        anchor = anchor_from_cached_candidate(next(iter(grid.values())))
        if any(anchor_from_cached_candidate(c) != anchor for c in grid.values()):
            raise ValueError("fixed cached EHR anchor differs across candidates")
        anchors.append({"opaque_source_index": ordinal, "anchor": anchor.record(), "ehr_anchor_sha256": anchor.sha256})
        for (cxr_model, seed, report_model), cached in sorted(grid.items()):
            record = cached["score_record"]
            lineage = record["lineage"]
            pair = {"case_id": case, "triple_candidate_id": record["triple_candidate_id"],
                **{k: lineage[k] for k in PAIR_FIELDS - {"case_id", "triple_candidate_id"}}}
            image, report = cxrs[pair["cxr_candidate_id"]], reports[pair["report_candidate_id"]]
            if (image["model_id"] != cxr_model or type(image["seed"]) is not int or image["seed"] != seed
                    or report["model_id"] != report_model
                    or pair["ehr_sha256"] != anchor.ehr_sha256 or pair["ehr_facts_sha256"] != anchor.ehr_facts_sha256
                    or report.get("model_input_signature") != "single_current_synthetic_cxr"
                    or report.get("structured_ehr_content_supplied_to_model") is not False
                    or report.get("source_report_or_real_target_supplied") is not False):
                raise ValueError("frozen fully synthetic case/image/report contract differs")
            request = requests[(case, cxr_model, seed)]
            if (canonical_json_sha256(request) != image["input_request_sha256"]
                    or request["inputs"]["synthetic_ehr"]["sha256"] != anchor.ehr_sha256
                    or request["inputs"]["ehr_facts"]["sha256"] != anchor.ehr_facts_sha256
                    or request["inputs"]["final_prompt"]["sha256"] != image["prompt_sha256"]):
                raise ValueError("unchanged EHR, prompt or actual generation request differs")
            pairs.append(pair)
    pairs.sort(key=lambda r: r["triple_candidate_id"])
    validate_pairs(pairs, cxrs, reports)
    if (len(pairs) != 960 or len({r["triple_candidate_id"] for r in pairs}) != 960
            or len({r["cxr_candidate_id"] for r in pairs}) != 240
            or len({r["report_candidate_id"] for r in pairs}) != 960):
        raise ValueError("complete unique original candidate pairing required")
    for artifact in (*cxrs.values(), *reports.values()):
        pins[artifact["artifact"]["path"]] = artifact["artifact"]["sha256"]
    scores_path = require_inside(full / "scores.json", full, must_exist=True)
    if sha256_file(scores_path) != fm["artifacts"]["scores.json"]["sha256"]:
        raise ValueError("sealed secondary cache changed")
    # Hash the cache without reading values. Values are loaded only after choice.
    pins[str(scores_path)] = sha256_file(scores_path)
    for name, digest in MODEL_HASHES.items():
        key = "model_" + name
        path = Path(fm["source_paths"][key])
        if fm["source_sha256"][key] != digest or sha256_file(path) != digest:
            raise ValueError("historical secondary checkpoint differs")
        pins[str(path)] = digest
    check_pins(before)
    check_pins(pins)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        payload = {"schema_version": SCHEMA, "policy": POLICY, "anchors": anchors, "pairs": pairs,
            "cxr_runs": cxr_runs, "report_runs": report_runs, "xrv": prior["xrv"], "chexbert": prior["chexbert"],
            "source_pins": before, "artifact_pins": pins, "full_bank_manifest_sha256": args.full_bank_manifest_sha256,
            "cached_secondary_path": str(scores_path), "cached_secondary_sha256": sha256_file(scores_path),
            "source_bodies_parsed": False, "secondary_values_read": False, "factory_instantiated": False,
            "legacy_image_or_report_labels_reused": False, "new_model_calls": 0,
            "direct_ehr_anchor_provenance": "cached_direct_state_and_source_category_not_new_raw_ehr_review",
            "planned_counts": {"fixed_ehr_cases": 80, "fixed_images": 240, "report_candidates": 960,
                "new_xrv_scored_images": 240, "new_chexbert_scored_reports": 960,
                "reused_secondary_pairs": 960, "new_generation_calls": 0, "new_biovil_encodings": 0},
            "runtime_limits": {"xrv_seconds": 420, "chexbert_seconds": 420, "chexbert_batch_size": 16},
            "clinical_acceptance": False}
        path = write_private_json(temporary / "plan.json", payload)
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA, "plan_sha256": sha256_file(path),
            "status": "prepared_full_pool_fresh_profile_scoring", "new_model_calls": 0,
            "planned_counts": payload["planned_counts"], "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expert-plan", "expert-plan-manifest-sha256", "full-bank-run",
                 "full-bank-manifest-sha256", "output-root", "run-id"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    os.umask(0o007)
    began = time.monotonic()
    try:
        root = prepare(args)
    except Exception as exc:
        print(json.dumps({"status": "preflight_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "prepared_full_pool_fresh_profile_scoring", "new_model_calls": 0,
        "runtime_seconds": round(time.monotonic() - began, 3), "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
