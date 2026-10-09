#!/usr/bin/env python3
"""CPU Slurm preflight for two fixed synthetic images and native beam candidates.

No model imports/instantiation, report-body reads, or image-pixel reads.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (WORKSPACE, PROTECTED_ROOT, require_inside, read_json, sha256_file, load_cxr_candidates,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, private_directory)
from tricompose_v12.report_nbest import SCHEMA, POLICY
from tricompose_v12.runtime_dispatch import require_slurm
from tricompose_v12.invariant_verification import anchor_from_cached_candidate
from tricompose_v12.live_workers import registry, preflight_generator, source_pins, check_pins
from tricompose_v12.live_plan import preflight_scorers
from tricompose_v12.full_bank_secondary import SCHEMA as FULL_SCHEMA
from compare_frozen_report_paths import load_bank
from score_automatic_replay_biovil import MODEL_HASHES
from prepare_fixed_image_reports import BIOVIL_MODEL, BIOVIL_PYTHON


def generation_source_check(spec):
    """Static support check, NOT a claim that GPU inference has passed."""
    checkpoint = Path(spec["model_dir"])
    source = checkpoint/"modelling_single.py"
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SingleCXREncoderDecoderModel"]
    if (len(classes) != 1 or not any(isinstance(n, ast.Name) and n.id == "VisionEncoderDecoderModel" for n in classes[0].bases)
            or any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "generate" for n in classes[0].body)):
        raise ValueError("unexpected custom generation override")
    sites = list((Path(spec["python"]).parent.parent/"lib").glob("python*/site-packages/transformers"))
    if len(sites) != 1: raise ValueError("ambiguous transformers installation")
    generation = sites[0]/"generation"
    config_text = (generation/"configuration_utils.py").read_text()
    if "self.num_return_sequences > self.num_beams" not in config_text:
        raise ValueError("native n-best validation was not found")
    files = [generation/"configuration_utils.py", generation/"utils.py", sites[0]/"models/vision_encoder_decoder/modeling_vision_encoder_decoder.py"]
    return {str(p): {"sha256": sha256_file(p), "size_bytes": p.stat().st_size} for p in files}


def prepare(args):
    require_slurm()
    before = source_pins()
    full = require_inside(args.full_bank_run, PROTECTED_ROOT, must_exist=True)
    fm_path = full/"manifest.json"
    if sha256_file(fm_path) != args.full_bank_manifest_sha256:
        raise ValueError("approved full-bank source manifest changed")
    fm = read_json(fm_path)
    if (fm.get("schema_version") != FULL_SCHEMA or fm.get("original_selection_changed") is not False
            or fm.get("clinical_acceptance") is not False):
        raise ValueError("frozen synthetic source registry required")
    endpoint = Path(fm["source_paths"]["historical_endpoint_manifest"]).parent
    ep = endpoint/"manifest.json"
    if sha256_file(ep) != fm["source_sha256"]["historical_endpoint_manifest"]:
        raise ValueError("historical source changed")
    bank, _, sources = load_bank(argparse.Namespace(endpoint_run=endpoint))
    if len(bank) != 80: raise ValueError("original eighty-case synthetic source required")
    pins = {str(p): sha256_file(p) for p in sources.values()}
    pins[str(fm_path)] = args.full_bank_manifest_sha256
    paths = []
    for key, value in fm["source_paths"].items():
        if key.startswith("cxr_manifest_"):
            path = require_inside(value, PROTECTED_ROOT, must_exist=True)
            if sha256_file(path) != fm["source_sha256"][key]: raise ValueError("CXR registry changed")
            if read_json(path)["model_id"] == POLICY["fixed_image_model"]: paths.append(path)
    if len(paths) != 1: raise ValueError("unique predeclared Sana run required")
    original_mp = paths[0]; original = read_json(original_mp)
    cxrs = load_cxr_candidates([original_mp.parent])
    source_entries = {r["candidate_id"]: r for r in original["candidates"]}
    fixed = []
    for index in POLICY["opaque_case_indices"]:
        case = sorted(bank)[index]
        cached = bank[case][(POLICY["fixed_image_model"], POLICY["fixed_image_seed"], POLICY["report_model"])]
        anchor = anchor_from_cached_candidate(cached)
        if any(anchor_from_cached_candidate(c) != anchor for c in bank[case].values()):
            raise ValueError("fixed EHR source is inconsistent")
        lineage = cached["score_record"]["lineage"]
        image = cxrs[lineage["cxr_candidate_id"]]
        if (image["case_id"] != case or image["ehr_sha256"] != anchor.ehr_sha256
                or image["ehr_facts_sha256"] != anchor.ehr_facts_sha256
                or image["artifact"]["sha256"] != lineage["cxr_sha256"] or image["seed"] != 0):
            raise ValueError("fixed source EHR/image hashes differ")
        candidate_path = require_inside(original_mp.parent/source_entries[image["candidate_id"]]["path"], original_mp.parent, must_exist=True)
        pins.update({str(candidate_path): sha256_file(candidate_path),
                     str(Path(image["artifact"]["path"])): image["artifact"]["sha256"]})
        fixed.append({"opaque_source_index": index, "anchor": anchor.record(),
            "ehr_anchor_sha256": anchor.sha256, "image": image,
            "original_candidate_path": str(candidate_path), "original_candidate_sha256": sha256_file(candidate_path),
            "historical_cxrmate_report_sha256": lineage["report_sha256"]})
    pins[str(original_mp)] = sha256_file(original_mp)
    entries = registry()
    generator = preflight_generator(entries["cxrmate_single"])
    dependency_pins = generation_source_check(generator)
    thresholds = require_inside(args.thresholds, PROTECTED_ROOT, must_exist=True)
    entries = preflight_scorers(entries, thresholds)
    pins[str(thresholds)] = sha256_file(thresholds)
    biovil_pins = {str(BIOVIL_MODEL/name): {"sha256": sha256_file(BIOVIL_MODEL/name), "size_bytes": (BIOVIL_MODEL/name).stat().st_size}
                  for name in MODEL_HASHES}
    if any(biovil_pins[str(BIOVIL_MODEL/n)]["sha256"] != h for n, h in MODEL_HASHES.items()):
        raise ValueError("independent frozen endpoint changed")
    for path in sorted(BIOVIL_MODEL.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".model", ".py"}:
            biovil_pins[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    for path in sorted((WORKSPACE/"runtime/vendor/hi_ml_multimodal_0_2_2").rglob("*.py")):
        biovil_pins[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    if not os.access(BIOVIL_PYTHON, os.X_OK): raise ValueError("endpoint environment missing")
    check_pins(before); check_pins(pins)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        subset = temporary/"fixed_images"; private_directory(subset)
        subset_rows = []
        for case in fixed:
            image = case["image"]; name = image["candidate_id"]+".json"
            path = write_private_json(subset/name, image)
            subset_rows.append({"candidate_id": image["candidate_id"], "case_id": image["case_id"], "path": name, "sha256": sha256_file(path)})
        write_private_json(subset/"manifest.json", {"schema_version": "tricompose-cxr-candidate-run-v1.1",
            "model_id": original["model_id"], "model_revision": original["model_revision"], "frozen_model": True,
            "candidate_count": 2, "candidates": subset_rows, "metadata_copy_only": True,
            "source_manifest_sha256": sha256_file(original_mp), "new_cxr_generation_calls": 0,
            "historical_tokenizer_trace_verified": False})
        for path in subset.glob("*.json"): pins[str(target/"fixed_images"/path.name)] = sha256_file(path)
        payload = {"schema_version": SCHEMA, "policy": POLICY, "fixed_cases": fixed,
            "cxr_run": str(target/"fixed_images"), "generator": generator, "dependency_pins": dependency_pins,
            "xrv": entries["xrv"], "chexbert": entries["chexbert"], "biovil_model": str(BIOVIL_MODEL),
            "biovil_python": str(BIOVIL_PYTHON), "biovil_asset_pins": biovil_pins,
            "source_pins": before, "artifact_pins": pins,
            "planned_counts": {"native_generate_invocations": 2, "returned_sequences": 6,
                "xrv_scored_images": 2, "chexbert_scored_reports": 6,
                "maximum_biovil_image_encodings": 2, "maximum_biovil_text_encodings": 6},
            "source_bodies_parsed": False, "factory_instantiated": False, "new_model_calls": 0,
            "new_ehr_or_cxr_samples": 0, "clinical_acceptance": False,
            "gpu_native_nbest_verified": False, "duplicate_calls_are_diversity": False}
        path = write_private_json(temporary/"plan.json", payload)
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA, "plan_sha256": sha256_file(path),
            "status": "cpu_preflight_complete_gpu_interface_unverified", "new_model_calls": 0,
            "planned_counts": payload["planned_counts"], "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("full-bank-run", "full-bank-manifest-sha256", "thresholds", "output-root", "run-id"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args(); os.umask(0o007); started = time.monotonic()
    try: root = prepare(args)
    except Exception as exc:
        print(json.dumps({"status": "preflight_failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": "cpu_preflight_complete_gpu_interface_unverified", "new_model_calls": 0,
        "runtime_seconds": round(time.monotonic()-started, 3), "manifest_sha256": sha256_file(root/"manifest.json")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
