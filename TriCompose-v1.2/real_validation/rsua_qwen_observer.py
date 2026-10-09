#!/usr/bin/env python3
"""Frozen image-only Qwen on the existing RSUA50 cohort; proxy evidence only.

Preparation parses no source pixels/cohort labels. Approved GPU execution seals
all predictions before offline reference evaluation. No model/gate revision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import verify_fresh_probe_images as observer
import image_validity_guard as guard
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.runtime_dispatch import ProtectedJournal

Reader = observer.source.previous.postflight.MetadataReader
require = observer.require
VERSION = "tricompose-rsua-frozen-qwen-image-observer-v1"
BASE = PROTECTED_ROOT / "tricompose_v1_2"
PNG_SOURCE = BASE / "real_validation/rsua_biovil_pilots/rsua_biovil50_12637081"
PNG_SOURCE_SHA = "a8eed9db9e20146cb1bc7ca597aed78f8bfb5cd1683b29e82dde99497b08992a"
XRV_SOURCE = BASE / "real_validation/rsua_xrv_pilots/rsua_xrv50_12636566"
XRV_SOURCE_SHA = "92a64874818f37e0d363e8406064d4619cb9edefa01ccab43881cf1a1df7d0e0"
COHORT = BASE / "real_validation/rsua_pilot_cohorts/cohort50_12625457_001/cohort.json"
COHORT_SHA = "ed33c7708ac737aa8e270da2219457def542186f02bcc085c7188c7a135779c1"
OBSERVER_PLAN = BASE / "llm_probe_image_plans/probe_images4_12817261_001"
OBSERVER_PLAN_SHA = "4913be6d356954b46acbaabd6415e0f7f4aa390d727fb9fcebf036f1451a114f"
CONFIG = {**observer.CONFIG, "case_count": 50, "maximum_model_calls": 50,
    "maximum_model_loads": 1, "reference_finding": "pneumonia"}
PROFILES = {"xrv_default_0_5": .5, "xrv_unchanged_weak_reference_transport": .55457607}
EXPECTED_IDS = [f"case_{i:04d}" for i in range(50)]


def manifest(root, pin):
    r = Reader(root); r.hash(r.root / "manifest.json", pin)
    return r, r.json(r.root / "manifest.json")


def artifact(r, m, name, *, parse=True):
    pin = m["artifacts"][name]
    r.hash(r.root / name, pin["sha256"] if isinstance(pin, dict) else pin)
    return r.json(r.root / name) if parse else None


def public_inputs(records, *, stat_path):
    """Only derived hashes/opaque order/stat, never labels or old model scores."""
    require([r["case_id"] for r in records] == EXPECTED_IDS, "fixed_fifty_source_order_required")
    result = []
    for row in records:
        require(guard.HASH.fullmatch(row["png_sha256"]) and guard.HASH.fullmatch(row["source_image_sha256"]),
            "existing_lossless_png_and_bmp_hashes_required")
        path = PNG_SOURCE / "inputs" / (row["case_id"] + ".png")
        stats = stat_path(path)
        require(len(stats) == 2 and type(stats[0]) is int and 1 <= stats[0] <= guard.MAX_BYTES,
            "bounded_existing_opaque_png_required")
        result.append({"case_id": row["case_id"], "path": str(path), "png_sha256": row["png_sha256"],
            "original_bmp_sha256": row["source_image_sha256"], "file_stats": stats})
    require(len({r["png_sha256"] for r in result}) == len({r["original_bmp_sha256"] for r in result}) == 50,
        "all_fifty_distinct_images_no_substitution")
    return result


def prepare(args):
    observer.source.previous.gate.cpu_guard()
    r, m = manifest(PNG_SOURCE, PNG_SOURCE_SHA)
    summary = artifact(r, m, "summary.json")
    require(summary["frozen"] is True and summary["selection_changed"] is False
        and summary["primary_metric_eligible"] is False and summary["independent_clinical_adjudication"] is False
        and summary["execution"]["source_cases"] == summary["execution"]["lossless_png_checked"] == 50,
        "existing_fifty_lossless_public_source_images_required")
    records = artifact(r, m, "scores.json")["records"]  # No reference labels in this derived file.
    inputs = public_inputs(records, stat_path=lambda p: [r.path(p).stat().st_size, r.path(p).stat().st_mtime_ns])
    xr, xm = manifest(XRV_SOURCE, XRV_SOURCE_SHA)
    xs = artifact(xr, xm, "summary.json")
    for name in ("scores.json", "score_table.csv"): artifact(xr, xm, name, parse=False)
    require(xs["cohort_sha256"] == COHORT_SHA and xs["model_calls"] == 50 and xs["frozen"] is True
        and xs["selection_changed"] is False and xs["thresholds_fitted_on_this_cohort"] is False,
        "same_unchanged_xrv_cohort_and_execution_required")
    cr = Reader(COHORT.parent); cr.hash(COHORT, COHORT_SHA)  # NO cohort-label parse in prepare.
    qr, qm = manifest(OBSERVER_PLAN, OBSERVER_PLAN_SHA)
    qr.hash(qr.root / "plan.json", qm["plan_sha256"])
    native = qr.json(qr.root / "plan.json")
    require(native["schema_version"] == observer.VERSION and native["config"] == observer.CONFIG
        and native["model_received_ehr_reports_ids_or_scores"] is False,
        "same_unmodified_image_observer_required")
    qwen = native["qwen"]
    asset_stats = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in qwen["asset_pins"]}
    prompt = observer.existing.image_interface.IMAGE_PROMPT.format(
        findings=", ".join(observer.existing.image_interface.FINDINGS))
    require(observer.existing.image_interface.digest_text(prompt) == native["image_prompt_sha256"],
        "exact_frozen_eight_finding_prompt_required")
    files = [Path(__file__), Path(observer.__file__), Path(observer.existing.__file__),
        Path(observer.existing.image_interface.__file__), Path(guard.__file__),
        ROOT / "tests/test_rsua_qwen_observer.py", ROOT.parent / "docs/rsua_qwen_observer_protocol.md"]
    sources = {**native["source_pins"], **{str(p.resolve()): sha256_file(p) for p in files}}
    observer.source.check_pins(sources)
    readers = (r, xr, cr, qr); pins = {}
    for reader in readers: reader.recheck(); pins.update(reader.pins)
    plan = {"schema_version": VERSION, "config": CONFIG, "inputs": inputs, "qwen": qwen,
        "model_asset_stats": asset_stats, "image_prompt_sha256": native["image_prompt_sha256"],
        "finding_order": list(observer.existing.image_interface.FINDINGS),
        "cohort_sha256": COHORT_SHA, "cohort_path": str(COHORT), "source_pins": sources,
        "artifact_pins": pins, "reference_finding": "pneumonia", "xrv_profiles": PROFILES,
        "input_plan_contains_labels_or_scores": False, "reference_labels_parsed_in_prepare": False,
        "source_pixels_opened_in_prepare": False, "new_model_calls": 0,
        "primary_metric_eligible": False, "selection_changed": False}
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        path = write_private_json(tmp / "plan.json", plan)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(path),
            "maximum_model_calls": 50, "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(tmp, target)
    except BaseException: discard_atomic_run(tmp); raise
    return target


def validate_records(records, references):
    ids = [r["case_id"] for r in records]
    require(records and len(set(ids)) == len(ids) and set(ids) == set(references)
        and all(v in ("positive", "negative") for v in references.values()), "complete_unique_proxy_reference_scope_required")
    for row in records:
        status, states = row["contract_status"], row["states"]
        require((status == "complete" and row["model_called"] is True and isinstance(states, dict)
            and set(states) == set(guard.HEADS) and all(v in guard.STATES for v in states.values()))
            or (status == "failed_unavailable" and row["model_called"] is True and states is None)
            or (status in ("blocked_before_model_call", "not_attempted_after_runtime_failure")
                and row["model_called"] is False and states is None), "complete_four_state_or_unavailable_not_invented_negative")
        require(type(row["model_called"]) is bool and not (status in
            ("blocked_before_model_call", "not_attempted_after_runtime_failure") and row["model_called"]),
            "availability_and_actual_model_call_must_be_distinct")


def summarize(records, references):
    validate_records(records, references)
    groups = {}
    for ref in ("positive", "negative"):
        rows = [r for r in records if references[r["case_id"]] == ref]
        c = Counter(r["states"]["pneumonia"] if r["contract_status"] == "complete" else r["contract_status"] for r in rows)
        correct = c[ref]; explicit = c["positive"] + c["negative"]
        unresolved = len(rows) - explicit
        groups[ref] = {"reference_slots": len(rows), "positive": c["positive"], "negative": c["negative"],
            "uncertain": c["uncertain"], "unknown": c["unknown"], "failed_unavailable": c["failed_unavailable"],
            "blocked_before_model_call": c["blocked_before_model_call"],
            "not_attempted_after_runtime_failure": c["not_attempted_after_runtime_failure"],
            "explicit_slots": explicit, "correct_explicit_slots": correct, "unresolved_slots": unresolved,
            "explicit_coverage": explicit / len(rows) if rows else None,
            "correct_explicit_support_over_all": correct / len(rows) if rows else None,
            "conditional_support_over_explicit": correct / explicit if explicit else None,
            "missingness_correctness_lower": correct / len(rows) if rows else None,
            "missingness_correctness_upper": (correct + unresolved) / len(rows) if rows else None}
    all_slots = len(records); explicit = sum(g["explicit_slots"] for g in groups.values())
    correct = sum(g["correct_explicit_slots"] for g in groups.values())
    return {"reference_kind": "published_pneumonia_vs_paper_described_normal_cohort_classes",
        "reference_finding": "pneumonia", "reference_groups": groups, "reference_slots": all_slots,
        "explicit_slots": explicit, "explicit_coverage": explicit / all_slots,
        "correct_explicit_support_over_all": correct / all_slots,
        "conditional_support_over_explicit": correct / explicit if explicit else None,
        "balanced_correct_explicit_support_over_all": sum(g["correct_explicit_support_over_all"] for g in groups.values()) / 2
            if all(g["reference_slots"] for g in groups.values()) else None,
        "confusion_explicit_only": {"tp": groups["positive"]["positive"], "fn": groups["positive"]["negative"],
            "tn": groups["negative"]["negative"], "fp": groups["negative"]["positive"]},
        "unresolved_not_counted_as_negative": True, "auroc": None, "average_precision": None,
        "probability_metrics": None, "metric_na_reason": "discrete_states_not_calibrated_scores",
        "other_seven_findings_reference_unavailable": True, "clinical_accuracy": None,
        "independent_clinical_adjudication": False, "primary_metric_eligible": False,
        "clinical_acceptance": False, "regeneration_authorized": False}


def compare_xrv(records, references, source_rows, inputs):
    validate_records(records, references)
    ids = {r["case_id"] for r in records}
    require(len(source_rows) == len(inputs) == len(ids) and {r["case_id"] for r in source_rows} == ids
        and {r["case_id"] for r in inputs} == ids and len({r["png_sha256"] for r in inputs}) == len(inputs),
        "same_original_xrv_images_and_references_required")
    original = {r["case_id"]: r for r in inputs}
    for row in source_rows:
        require(row["image_sha256"] == original[row["case_id"]]["original_bmp_sha256"]
            and row["reference_state"] == references[row["case_id"]]
            and type(row["pneumonia_score"]) in (int, float) and math.isfinite(row["pneumonia_score"])
            and 0 <= row["pneumonia_score"] <= 1, "unchanged_xrv_bmp_reference_and_score_lineage_required")
    output = {}
    for name, threshold in PROFILES.items():
        states = {r["case_id"]: "positive" if r["pneumonia_score"] >= threshold else "negative" for r in source_rows}
        fixed = [{"case_id": cid, "model_called": True, "contract_status": "complete",
            "states": {**dict.fromkeys(guard.HEADS, "unknown"), "pneumonia": states[cid]}} for cid in sorted(ids)]
        comparable = [r for r in records if r["contract_status"] == "complete"
            and r["states"]["pneumonia"] in ("positive", "negative")]
        same = sum(r["states"]["pneumonia"] == states[r["case_id"]] for r in comparable)
        output[name] = {"unchanged_threshold": threshold, "xrv_proxy_readout": summarize(fixed, references),
            "qwen_xrv_comparable": len(comparable), "same_explicit_state": same,
            "opposite_explicit_state": len(comparable) - same, "qwen_unresolved": len(records) - len(comparable),
            "same_state_is_clinical_truth": False, "new_xrv_calls": 0}
    return output


def invoke_inputs(inputs, callback, *, journal, inspect, maximum_calls=50):
    """Lazy checked-pixel calls. Any runtime failure stops, never auto retries."""
    require(inputs and len({i["case_id"] for i in inputs}) == len(inputs)
        and len({i["png_sha256"] for i in inputs}) == len(inputs)
        and type(maximum_calls) is int and maximum_calls >= len(inputs), "fixed_unique_input_and_call_budget_required")
    records = []; charged = 0; failed = False
    for ordinal, item in enumerate(inputs):
        base = {"case_id": item["case_id"], "png_sha256": item["png_sha256"],
            "source_image_sha256": item["original_bmp_sha256"]}
        if failed:
            records.append({**base, "contract_status": "not_attempted_after_runtime_failure", "states": None,
                "model_called": False, "guard": None, "failure_reason": "prior_runtime_failure"})
            continue
        current, image = inspect(item)
        guard.validate_receipt(current)
        require(current["artifact_sha256"] == item["png_sha256"], "checked_current_image_hash_required")
        if not current["basic_comparison_permitted"]:
            records.append({**base, "contract_status": "blocked_before_model_call", "states": None,
                "model_called": False, "guard": current, "failure_reason": current["reason"]})
            journal({"event": "blocked_before_model_call", "ordinal": ordinal, "png_sha256": item["png_sha256"]})
            continue
        require(image is not None and guard.normalized_pixel_sha(image) == current["normalized_pixel_sha256"],
            "callback_receives_exact_checked_in_memory_pixels")
        require(charged < maximum_calls, "no_call_beyond_fixed_budget")
        journal({"event": "call_reserved", "ordinal": ordinal, "charged_model_attempts": 1,
            "png_sha256": item["png_sha256"]}); charged += 1
        started = time.monotonic()
        try:
            raw = callback(image)
        except Exception:
            failed = True
            raw = {"contract_status": "failed_unavailable", "states": None, "failure_reason": "runtime_exception",
                "response_sha256": None, "input_tokens": None, "output_tokens": None, "token_limit_reached": None}
        elapsed = time.monotonic() - started
        require(isinstance(raw, dict) and set(raw) == {"contract_status", "states", "failure_reason",
            "response_sha256", "input_tokens", "output_tokens", "token_limit_reached"}, "sanitized_closed_callback_result_required")
        validate_records([{**base, **raw, "model_called": True}], {item["case_id"]: "positive"})
        require((raw["contract_status"] == "complete" and raw["token_limit_reached"] is False
            and raw["failure_reason"] is None) or raw["contract_status"] == "failed_unavailable",
            "token_cap_cannot_be_complete_finding_response")
        require(raw["response_sha256"] is None or (isinstance(raw["response_sha256"], str)
            and guard.HASH.fullmatch(raw["response_sha256"])), "sanitized_response_hash_required")
        records.append({**base, **raw, "model_called": True, "guard": current, "elapsed_seconds": elapsed})
        journal({"event": "call_finished", "ordinal": ordinal, "contract_status": raw["contract_status"],
            "runtime_failure_stop": failed})
    require(charged == sum(r["model_called"] for r in records), "charged_attempts_equal_actual_invocations")
    return records, charged, failed


def run(args):
    observer.gpu_guard()  # Before source reads, torch import or writes.
    require(args.allow_public_reference_eval is True, "explicit_public_reference_pixel_evaluation_required")
    pr, pm = manifest(args.plan_run, args.plan_manifest_sha256)
    require(pm["schema_version"] == VERSION and pm["status"] == "prepared_cpu_only_gpu_not_submitted",
        "exact_reviewed_public_reference_plan_required")
    pr.hash(pr.root / "plan.json", pm["plan_sha256"]); plan = pr.json(pr.root / "plan.json")
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG and len(plan["inputs"]) == 50
        and [i["case_id"] for i in plan["inputs"]] == EXPECTED_IDS and plan["selection_changed"] is False
        and plan["cohort_sha256"] == COHORT_SHA and plan["cohort_path"] == str(COHORT)
        and plan["input_plan_contains_labels_or_scores"] is False and plan["xrv_profiles"] == PROFILES,
        "fixed_fifty_blind_image_inputs_only")
    for field in ("source_pins", "artifact_pins"): observer.source.check_pins(plan[field])
    observer.source.check_pins(plan["qwen"]["asset_pins"])
    require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == v for p, v in plan["model_asset_stats"].items()),
        "unchanged_qwen_asset_metadata_required")
    prompt = observer.existing.image_interface.IMAGE_PROMPT.format(findings=", ".join(guard.HEADS))
    require(observer.existing.image_interface.digest_text(prompt) == plan["image_prompt_sha256"]
        and plan["finding_order"] == list(guard.HEADS), "unchanged_image_only_prompt_and_inventory_required")
    import torch
    from PIL import Image, ImageFile
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= CONFIG["min_vram_gib"] * 1024**3
        and ImageFile.LOAD_TRUNCATED_IMAGES is False, "supported_gpu_and_strict_png_decoder_required")
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic(); loaded = {}; load_attempts = 0; generate_attempts = 0
    try:
        torch.set_num_threads(2); torch.manual_seed(CONFIG["seed"]); torch.cuda.reset_peak_memory_stats()
        with ProtectedJournal(tmp / "calls.journal.jsonl") as journal, open(os.devnull, "w") as quiet, \
                contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            def callback(image):
                nonlocal load_attempts, generate_attempts
                if not loaded:
                    require(load_attempts == 0, "one_lazy_load_no_retry_required")
                    journal.append({"event": "observer_load_reserved", "charged_load_attempts": 1}); load_attempts += 1
                    model, processor, kind = _load_model(Path(plan["qwen"]["model_path"]), torch,
                        min_pixels=CONFIG["min_pixels"], max_pixels=CONFIG["max_pixels"])
                    model.eval().requires_grad_(False)
                    require(not model.training and not any(p.requires_grad for p in model.parameters()), "frozen_qwen_required")
                    loaded.update(model=model, processor=processor, kind=kind)
                journal.append({"event": "generate_attempt_started", "ordinal": generate_attempts}); generate_attempts += 1
                response, tokens = observer.existing.image_interface.infer(
                    observer.existing.image_interface.request_messages("image", image=image), loaded["model"],
                    loaded["processor"], torch, max_new_tokens=CONFIG["max_new_tokens"])
                decoded = observer.existing.sanitized_decode(response, token_limit_reached=tokens["token_limit_reached"])
                return {**tokens, "response_sha256": observer.existing.image_interface.digest_text(response),
                    **{k: decoded[k] for k in ("contract_status", "states", "failure_reason")}}

            def inspect(item):
                path = require_inside(item["path"], PNG_SOURCE, must_exist=True)
                require([path.stat().st_size, path.stat().st_mtime_ns] == item["file_stats"], "same_existing_png_stats_required")
                return guard.inspect_png(path, item["png_sha256"], Image=Image)

            records, charged, runtime_failed = invoke_inputs(plan["inputs"], callback, journal=journal.append, inspect=inspect)
        pp = write_private_json(tmp / "predictions.json", {"schema_version": VERSION, "records": records,
            "frozen": True, "image_only": True, "labels_or_old_scores_supplied_to_model": False,
            "actual_model_attempts": charged, "actual_generate_attempts": generate_attempts, "charged_load_attempts": load_attempts})
        with pp.open("rb") as handle: os.fsync(handle.fileno())
        # Patient-derived public reference rows stay transient and private, after predictions.
        cr = Reader(COHORT.parent); cr.hash(COHORT, COHORT_SHA)
        cohort = cr.json(COHORT)
        references = {r["case_id"]: r["reference_state"] for r in cohort["records"]}
        require(len(cohort["records"]) == len(references) == 50
            and Counter(references.values()) == {"positive": 25, "negative": 25}, "unchanged_twenty_five_plus_twenty_five_references")
        xr, xm = manifest(XRV_SOURCE, XRV_SOURCE_SHA)
        source_rows = artifact(xr, xm, "scores.json")["records"]
        primary = summarize(records, references)
        comparisons = compare_xrv(records, references, source_rows, plan["inputs"])
        summary = {"schema_version": VERSION, "status": "completed_proxy_reference_diagnostic",
            "case_count": 50, "pneumonia_proxy_metrics": primary, "xrv_same_image_diagnostics": comparisons,
            "actual_model_attempts": charged, "actual_generate_attempts": generate_attempts, "charged_load_attempts": load_attempts,
            "runtime_exception_stop": runtime_failed, "complete_responses": sum(r["contract_status"] == "complete" for r in records),
            "model_retries": 0, "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_torch_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "gpu_name": torch.cuda.get_device_name(0), "model_dtype": str(next(loaded["model"].parameters()).dtype) if loaded else None,
            "raw_mimic_inputs_opened": False, "reference_labels_read_after_predictions_sealed": True,
            "selection_changed": False, "training_performed": False, "new_generation_calls": 0,
            "new_planner_calls": 0, "external_api_calls": 0, "clinical_repair_success": False,
            "clinical_fault_location": None, "clinical_acceptance": False, "measured_saved_model_calls": None}
        sp = write_private_json(tmp / "summary.json", summary)
        rp = write_private_text(tmp / "RESULTS_CN_EN.md", "# 冻结图像观察器：公开肺炎代理参考验证\n\n"
            "Same fixed RSUA50 cohort. Published class proxy, not image-adjudicated clinical truth.\n\n"
            "unknown/uncertain/不可用均不视作阴性；本结果不会修改原有 gate、阈值或 winner。\n\n"
            "```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
        for item in plan["inputs"]: require(sha256_file(item["path"]) == item["png_sha256"], "unchanged_source_png_required")
        for field in ("source_pins", "artifact_pins"): observer.source.check_pins(plan[field])
        require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == v for p, v in plan["model_asset_stats"].items()),
            "read_only_model_assets_unchanged_after_run")
        pr.recheck(); cr.recheck(); xr.recheck()
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "plan_manifest_sha256": args.plan_manifest_sha256, "source_cohort_sha256": COHORT_SHA,
            "prediction_sha256_before_reference_eval": sha256_file(pp), "actual_model_attempts": charged,
            "source_pins": plan["source_pins"], "source_png_sha256": {i["case_id"]: i["png_sha256"] for i in plan["inputs"]},
            "artifacts": {p.name: {"sha256": sha256_file(p)} for p in (pp, sp, rp, tmp / "calls.journal.jsonl")},
            "clinical_acceptance": False, "selection_changed": False})
        commit_atomic_run(tmp, target)
    except BaseException:
        # Retain all already fsynced reservations; a failed run is never a success.
        write_private_json(tmp / "failure.json", {"schema_version": VERSION, "status": "failed_closed_attempts_retained",
            "charged_load_attempts": load_attempts, "automatic_retry": False, "clinical_acceptance": False})
        commit_atomic_run(tmp, target); raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    preparation, execution = sub.add_parser("prepare"), sub.add_parser("run")
    execution.add_argument("--plan-run", type=Path, required=True)
    execution.add_argument("--plan-manifest-sha256", required=True)
    execution.add_argument("--allow-public-reference-eval", action="store_true")
    for p in (preparation, execution):
        p.add_argument("--output-root", type=Path, required=True); p.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        path = prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(path / "manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
