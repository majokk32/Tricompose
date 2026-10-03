"""Hash-bound CPU preflight; consumes synthetic request metadata, not bodies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from contracts import (WORKSPACE, PROTECTED_ROOT, read_json, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json)
from .invariant_verification import EHRAnchor, anchor_from_cached_candidate, _digest
from .legacy_replay_adapter import FINDINGS
from .live_workers import (registry, preflight_generator, source_pins, check_pins,
    XRV_PYTHON, XRV_WEIGHT, CHEXBERT_WEIGHT, CHEXBERT_BERT)
from .runtime_dispatch import require_slurm

SCHEMA = "tricompose-live-worker-plan-v1"


def validate_policy(policy):
    if (policy.get("schema_version") != "tricompose-live-frozen-worker-smoke-policy-v1"
            or policy.get("opaque_case_indices") != [0, 1]
            or policy.get("image_slots") != [["roentgen_v2", 0], ["chexgenbench_sana", 0]]
            or policy.get("report_models") != ["cxrmate_single"]
            or type(policy.get("call_budget_per_case")) is not int or policy["call_budget_per_case"] != 10
            or type(policy.get("max_retries_per_operation")) is not int or policy["max_retries_per_operation"] != 1
            or type(policy.get("timeout_seconds_per_worker_process")) is not int or policy["timeout_seconds_per_worker_process"] != 180
            or policy.get("retains_underconditioned_cases") is not True
            or any(policy.get(k) is not False for k in ("changes_existing_ehr_or_prompts",
                "selection_or_adaptive_repair_enabled", "clinical_acceptance_allowed", "new_training_allowed"))):
        raise ValueError("only the fixed two-case operational smoke is implemented")
    expected = {"schema_version", "source_request_run", "source_endpoint_run", "opaque_case_indices", "image_slots",
        "report_models", "call_budget_per_case", "max_retries_per_operation", "timeout_seconds_per_worker_process",
        "thresholds_path", "retains_underconditioned_cases", "changes_existing_ehr_or_prompts",
        "selection_or_adaptive_repair_enabled", "clinical_acceptance_allowed", "new_training_allowed"}
    if set(policy) != expected: raise ValueError("unexpected policy field")


def anchor_from_record(record):
    anchor = EHRAnchor(record["case_id"], record["ehr_sha256"], record["ehr_facts_sha256"],
        tuple((r["finding"], r["state"], tuple(r["source_categories"])) for r in record["findings"]))
    if anchor.record() != record: raise ValueError("fixed anchor provenance changed")
    return anchor


def frozen_requests(source, cases, slots):
    from tricompose_v11.cxr_contracts import validate_cxr_request, CASE_ID_PATTERN, canonical_json_sha256
    source = require_inside(source, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(source/"manifest.json")
    ids = manifest.get("case_ids", [])
    if (manifest.get("schema_version") != "tricompose-cxr-request-run-v1.1" or len(ids) != 80
            or len(set(ids)) != len(ids) or set(ids) != set(cases)
            or any(not CASE_ID_PATTERN.fullmatch(c) for c in ids)
            or manifest.get("request_count") != len(manifest.get("requests", []))):
        raise ValueError("original fully synthetic eighty-case request inventory differs")
    rows = {}
    for row in manifest["requests"]:
        key = (row["case_id"], row["model_id"], row["seed"])
        if key in rows: raise ValueError("duplicate source request")
        rows[key] = row
    selected = []
    for index in (0, 1):
        case = sorted(ids)[index]
        anchor = anchor_from_cached_candidate(next(iter(cases[case].values())))
        if any(anchor_from_cached_candidate(v) != anchor for v in cases[case].values()):
            raise ValueError("cached candidate bank changed fixed EHR within a case")
        requests = []
        for model, seed in slots:
            row = rows[(case, model, seed)]
            path = require_inside(source/row["path"], source, must_exist=True)
            if sha256_file(path) != row["sha256"]: raise ValueError("source request SHA differs")
            request = read_json(path)
            validate_cxr_request(request)  # File hashes only, no canonical EHR/prompt body parsing.
            if (request["case_id"] != case or request["model_id"] != model or request["seed"] != seed
                    or request["inputs"]["synthetic_ehr"]["sha256"] != anchor.ehr_sha256
                    or request["inputs"]["ehr_facts"]["sha256"] != anchor.ehr_facts_sha256):
                raise ValueError("generation source differs from immutable EHR anchor")
            requests.append({"request": request, "source_path": str(path), "source_sha256": row["sha256"],
                "canonical_request_sha256": canonical_json_sha256(request)})
        selected.append({"opaque_source_index": index, "case_id": case, "anchor": anchor.record(),
            "ehr_anchor_sha256": anchor.sha256, "requests": requests})
    return selected


def preflight_scorers(entries, thresholds):
    from extract_cxr_labels_xrv import XRV_LABELS
    from xrv_calibration import validate_bundle
    xrv_env = XRV_PYTHON.parent.parent
    candidates = list((xrv_env/"lib").glob("python*/site-packages/torchxrayvision/models.py"))
    if len(candidates) != 1: raise ValueError("ambiguous local XRV source")
    models = candidates[0]; datasets = models.with_name("datasets.py")
    fingerprint = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
    protocol = {"image": "PIL_L_float32", "normalize_maxval": 255, "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
        "xrv_models_source_sha256": sha256_file(models), "xrv_datasets_source_sha256": sha256_file(datasets)}
    provenance = {"score_space": "xrv_op_norm_0_1", "preprocessing_sha256": fingerprint(protocol),
        "finding_mapping_sha256": fingerprint(XRV_LABELS)}
    weight_sha = sha256_file(XRV_WEIGHT)
    bundle = read_json(thresholds)
    validate_bundle(bundle, checkpoint_sha256=weight_sha, expected_provenance=provenance)
    if bundle["calibration_status"] != "thresholds_fitted_pending_independent_review":
        raise ValueError("expected frozen diagnostic threshold profile")
    files = {"xrv": [XRV_WEIGHT, models, datasets], "chexbert": [CHEXBERT_WEIGHT,
        CHEXBERT_BERT/"config.json", CHEXBERT_BERT/"tokenizer_config.json", CHEXBERT_BERT/"vocab.txt",
        WORKSPACE/"cxrmate/tools/chexbert.py"]}
    for name, paths in files.items():
        spec = entries[name]
        if not os.access(spec["python"], os.X_OK): raise ValueError("missing executable scorer environment")
        pins = {str(p): {"sha256": sha256_file(p), "size_bytes": p.stat().st_size} for p in paths}
        entries[name] = {**spec, "status": "preflighted", "asset_pins": pins,
            "checkpoint_sha256": pins[spec["checkpoint"]]["sha256"], "factory_instantiated": False}
    if entries["chexbert"]["checkpoint_sha256"] != "6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1":
        raise ValueError("existing frozen CheXbert checkpoint changed")
    entries["xrv"].update(scorer_provenance=provenance,
        thresholds_sha256=sha256_file(thresholds), thresholds=bundle["findings"],
        thresholds_path=str(thresholds), calibration_status=bundle["calibration_status"])
    return entries


def prepare(args):
    require_slurm()  # CPU allocation required for cohort joins and large checkpoint hashing.
    from diagnose_automatic_discrepancy import load
    started_sources = source_pins()
    config_path = require_inside(args.config, WORKSPACE, must_exist=True)
    policy = read_json(config_path, root=WORKSPACE); validate_policy(policy)
    endpoint = require_inside(WORKSPACE/policy["source_endpoint_run"], PROTECTED_ROOT, must_exist=True)
    _, bank, _, sources = load(argparse.Namespace(endpoint_run=endpoint))
    original = require_inside(WORKSPACE/policy["source_request_run"], PROTECTED_ROOT, must_exist=True)
    cases = frozen_requests(original, bank, policy["image_slots"])
    thresholds = require_inside(WORKSPACE/policy["thresholds_path"], PROTECTED_ROOT, must_exist=True)
    entries = {k: {**v, "status": "registered_not_preflighted"} for k, v in registry().items()}
    active = [m for m, _ in policy["image_slots"]]+policy["report_models"]
    for name in active: entries[name] = preflight_generator(entries[name])
    preflight_scorers(entries, thresholds)
    check_pins(started_sources)
    lineage = {str(p): sha256_file(p) for p in sources.values()}
    lineage.update({str(config_path): sha256_file(config_path), str(original/"manifest.json"): sha256_file(original/"manifest.json"),
        str(thresholds): sha256_file(thresholds)})
    payload = {"schema_version": SCHEMA, "policy": policy, "cases": cases, "workers": entries,
        "source_pins": started_sources, "source_artifact_pins": lineage,
        "counts": {"fixed_ehr_cases": 2, "planned_new_cxrs": 4, "planned_new_reports": 4,
            "normal_model_attempts": 16, "maximum_charged_model_attempts": 20},
        "data_origin": "original_fully_synthetic_pool80", "source_bodies_parsed": False,
        "new_model_calls": 0, "gpu_inference_used": False, "factory_instantiated": False,
        "clinical_acceptance": False, "policy_is_adaptive": False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        file = write_private_json(temporary/"plan.json", payload)
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "plan_sha256": sha256_file(file), "counts": payload["counts"], "new_model_calls": 0,
            "registered_workers": len(entries), "preflighted_workers": sum(v["status"]=="preflighted" for v in entries.values()),
            "source_pins_sha256": _digest(started_sources), "clinical_acceptance": False})
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary); raise
    return target, payload


def load_plan(root, expected_manifest_sha256):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root/"manifest.json") != expected_manifest_sha256: raise ValueError("approved plan manifest changed")
    manifest = read_json(root/"manifest.json")
    if manifest.get("schema_version") != SCHEMA or sha256_file(root/"plan.json") != manifest["plan_sha256"]:
        raise ValueError("approved live plan changed")
    plan = read_json(root/"plan.json"); validate_policy(plan["policy"])
    if (plan.get("schema_version") != SCHEMA or plan.get("data_origin") != "original_fully_synthetic_pool80"
            or plan.get("source_bodies_parsed") is not False or plan.get("new_model_calls") != 0
            or plan.get("policy_is_adaptive") is not False or len(plan["cases"]) != 2):
        raise ValueError("unexpected live smoke provenance")
    check_pins(plan["source_pins"]); check_pins(plan["source_artifact_pins"])
    for case in plan["cases"]:
        if anchor_from_record(case["anchor"]).sha256 != case["ehr_anchor_sha256"]:
            raise ValueError("fixed EHR anchor changed")
        for row in case["requests"]:
            if read_json(row["source_path"]) != row["request"] or sha256_file(row["source_path"]) != row["source_sha256"]:
                raise ValueError("original final input request changed")
    return plan
