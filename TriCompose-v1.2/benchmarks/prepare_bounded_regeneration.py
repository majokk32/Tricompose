#!/usr/bin/env python3
"""CPU Slurm: seal a two-case, EHR-evidence-stratified real retry plan.

Only cached synthetic states, manifests and byte hashes; no image pixels,
EHR/prompt/report bodies, model factories, external API or generation.
"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json)
from tricompose_v12.bounded_regeneration import SCHEMA, POLICY, choose_cases, route, alternate_request
from tricompose_v12.full_pool_report_control import SCHEMA as SOURCE_SCHEMA, freeze
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v12.live_workers import registry, preflight_generator, source_pins, check_pins
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v11.cxr_contracts import validate_cxr_request, canonical_json_sha256
from run_full_pool_report_control import load as load_source_plan
from prepare_fixed_image_reports import BIOVIL_MODEL, BIOVIL_PYTHON
from score_automatic_replay_biovil import MODEL_HASHES

SOURCE_RUN = PROTECTED_ROOT / "tricompose_v1_2/full_pool_report_runs/pool80_fresh_12631194"
SOURCE_SHA = "f29d158088eb352426397b1c16458a2c71c1f9963e63ef5e5f92deaa044b2675"
SOURCE_PLAN = PROTECTED_ROOT / "tricompose_v1_2/full_pool_report_plans/pool80_fresh_12621834_001"
SOURCE_PLAN_SHA = "ffa4691cb2df67ba6d7c8e8a7bbd235fff1b592d82620837f6b26e1011ae65ca"


def prepare(args):
    require_slurm()
    started_sources = source_pins()
    if sha256_file(SOURCE_RUN / "manifest.json") != SOURCE_SHA:
        raise ValueError("exact completed full-pool source required")
    source = read_json(SOURCE_RUN / "manifest.json")
    if (source["schema_version"] != SOURCE_SCHEMA or source["status"] != "completed_full_pool_fresh_report_control_unvalidated"
            or source["clinical_acceptance"] is not False or source["original_winners_changed"] is not False
            or source["plan_manifest_sha256"] != SOURCE_PLAN_SHA):
        raise ValueError("completed immutable diagnostic source required")
    prior = load_source_plan(argparse.Namespace(plan_run=SOURCE_PLAN, plan_manifest_sha256=SOURCE_PLAN_SHA))
    pins = {**prior["artifact_pins"], str(SOURCE_RUN / "manifest.json"): SOURCE_SHA,
            str(SOURCE_PLAN / "manifest.json"): SOURCE_PLAN_SHA,
            str(SOURCE_PLAN / "plan.json"): sha256_file(SOURCE_PLAN / "plan.json")}
    for name in ("score_rows.json", "selection.json"):
        path = require_inside(SOURCE_RUN / name, SOURCE_RUN, must_exist=True)
        pins[str(path)] = source["artifacts"][name]["sha256"]
        if sha256_file(path) != pins[str(path)]:
            raise ValueError("fresh baseline states or sealed static choice changed")
    rows = read_json(SOURCE_RUN / "score_rows.json")["records"]
    static_selection = read_json(SOURCE_RUN / "selection.json")
    if freeze(rows) != static_selection:
        raise ValueError("static baseline is not recomputable from authenticated fresh states")
    enabled = {name for name, entry in prior["xrv"]["thresholds"].items() if entry["enabled"]}
    selected, cohort = choose_cases(prior["anchors"], enabled)
    indexed = {r["triple_candidate_id"]: r for r in rows}
    choices = {c["cxr_candidate_id"]: c for c in static_selection["choices"]}
    # This request source is authenticated through its original CXR manifest.
    request_roots = set()
    for run in prior["cxr_runs"]:
        cm = read_json(Path(run) / "manifest.json")
        if cm["model_id"] != POLICY["baseline_cxr_model"]:
            continue
        rp = require_inside(cm["source_request_run"], PROTECTED_ROOT, must_exist=True)
        if sha256_file(rp / "manifest.json") != cm["source_request_run_manifest_sha256"]:
            raise ValueError("original image request inventory changed")
        pins[str(rp / "manifest.json")] = cm["source_request_run_manifest_sha256"]
        request_roots.add(rp)
    if len(request_roots) != 1:
        raise ValueError("unique original RoentGen request source required")
    request_root = next(iter(request_roots))
    inventory = read_json(request_root / "manifest.json")["requests"]
    cases = []
    for item in selected:
        anchor = anchor_from_record(item["anchor"])
        matches = [r for r in rows if r["case_id"] == anchor.case_id
                   and r["cxr_model_id"] == POLICY["baseline_cxr_model"]
                   and r["report_model_id"] == POLICY["report_model"] and r["seed"] == 0]
        if len(matches) != 1:
            raise ValueError("unique fixed baseline image/report required")
        baseline = matches[0]
        static = indexed[choices[baseline["cxr_candidate_id"]]["selected_triple_id"]]
        records = [r for r in inventory if r["case_id"] == anchor.case_id
                   and r["model_id"] == POLICY["baseline_cxr_model"] and r["seed"] == 0]
        if len(records) != 1:
            raise ValueError("unique original seed-zero prompt request required")
        original_row = records[0]
        path = require_inside(request_root / original_row["path"], request_root, must_exist=True)
        if sha256_file(path) != original_row["sha256"]:
            raise ValueError("original request digest changed")
        original = read_json(path)
        validate_cxr_request(original)
        if (original["inputs"]["synthetic_ehr"]["sha256"] != anchor.ehr_sha256
                or original["inputs"]["ehr_facts"]["sha256"] != anchor.ehr_facts_sha256
                or baseline["receipt"]["ehr_anchor_sha256"] != anchor.sha256):
            raise ValueError("immutable EHR/request/baseline anchor differs")
        request = alternate_request(original)
        validate_cxr_request(request)
        pins[str(path)] = original_row["sha256"]
        cases.append({**item, "case_id": anchor.case_id, "baseline": baseline, "static": static,
            "frozen_action": route(baseline), "original_request": original,
            "requests": [{"request": request, "source_path": str(path),
                "source_sha256": original_row["sha256"], "canonical_request_sha256": canonical_json_sha256(request)}]})
    entries = registry()
    workers = {model: preflight_generator(entries[model])
               for model in (POLICY["baseline_cxr_model"], POLICY["report_model"])}
    workers.update(xrv=prior["xrv"], chexbert=prior["chexbert"])
    for spec in workers.values():
        check_pins(spec["asset_pins"])
    biovil_pins = {}
    for path in sorted(BIOVIL_MODEL.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".model", ".py", ".pt", ".bin"}:
            biovil_pins[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    for name, expected in MODEL_HASHES.items():
        if biovil_pins[str(BIOVIL_MODEL / name)]["sha256"] != expected:
            raise ValueError("independent secondary checkpoint changed")
    for path in sorted((WORKSPACE / "runtime/vendor/hi_ml_multimodal_0_2_2").rglob("*.py")):
        biovil_pins[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    if not os.access(BIOVIL_PYTHON, os.X_OK):
        raise ValueError("local endpoint environment unavailable")
    check_pins(started_sources)
    check_pins(pins)
    payload = {"schema_version": SCHEMA, "policy": POLICY, "cases": cases, "cohort": cohort,
        "workers": workers, "source_pins": started_sources, "artifact_pins": pins,
        "source_run": str(SOURCE_RUN), "source_manifest_sha256": SOURCE_SHA,
        "source_plan_run": str(SOURCE_PLAN), "source_plan_manifest_sha256": SOURCE_PLAN_SHA,
        "source_cxr_runs": prior["cxr_runs"], "source_report_runs": prior["report_runs"],
        "biovil_python": str(BIOVIL_PYTHON), "biovil_model": str(BIOVIL_MODEL),
        "biovil_asset_pins": biovil_pins, "endpoint_timeout_seconds": 150,
        "planned_counts": {"maximum_new_images": 2, "maximum_new_reports": 2,
            "maximum_generation_verification_attempts": 8,
            "maximum_endpoint_pairs": 6, "maximum_endpoint_image_encodings": 4,
            "maximum_endpoint_text_encodings": 6},
        "factory_instantiated": False, "source_bodies_parsed": False, "secondary_values_read": False,
        "new_model_calls": 0, "clinical_acceptance": False, "historical_pool_is_untouched_test": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(temporary / "plan.json", payload)
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA,
            "status": "cpu_preflight_complete_gpu_retry_not_run", "plan_sha256": sha256_file(path),
            "cohort": cohort, "planned_counts": payload["planned_counts"],
            "routed_cxr_retry_cases": sum(c["frozen_action"]["action"] == "regenerate_cxr" for c in cases),
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.umask(0o007)
    started = time.monotonic()
    try:
        root = prepare(args)
    except Exception as exc:
        import json
        print(json.dumps({"status": "preflight_failed", "error_type": type(exc).__name__}))
        return 1
    import json
    print(json.dumps({"status": "cpu_preflight_complete_gpu_retry_not_run", "new_model_calls": 0,
        "runtime_seconds": round(time.monotonic() - started, 3), "manifest_sha256": sha256_file(root / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
