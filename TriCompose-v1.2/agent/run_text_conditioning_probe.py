#!/usr/bin/env python3
"""Invented-text paired interface diagnostic; never an EHR repair experiment.

CPU prepare only; generation/scoring require separately approved GPU Slurm.
Reuse frozen native runtimes/settings. No real inputs, reports, API or fitting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for relative in ("src", "TriCompose-v1.0/src", "TriCompose-v1.1/src", "TriCompose-v1.2/src",
    "TriCompose-v1.2/agent", "TriCompose-v1.2/tools", "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT / relative))

from audit_fresh_report_agent import MetadataReader
from contracts import (PROTECTED_ROOT, private_directory, new_atomic_run, commit_atomic_run,
    discard_atomic_run, require_inside, sha256_file, write_private_json, write_private_text, RUN_ID_PATTERN)
from score_free_random_control import cpu_guard
from tricompose_v12.live_workers import (require_gpu_slurm, check_pins, load_script,
    run_private_process, registry)
from tricompose_v12.runtime_dispatch import ProtectedJournal

VERSION = "tricompose-invented-text-conditioning-probe-v1"
BASE = PROTECTED_ROOT / "tricompose_v1_2"
ASSET_SOURCE = BASE / "cxr_action_diversification_plans/cxr_actions2_12851223_001"
ASSET_SOURCE_SHA = "da1e819bbd9f0cf901388e5e0a5cf1fcf02cb25701a34a3fbf96db1147a51a16"
MODELS = ("roentgen_v2", "chexgenbench_sana")
ARMS = ("present", "absent")
SEEDS = (0, 1)
PROMPTS = {
    "present": "Chest radiograph. There is focal airspace consolidation consistent with pneumonia.",
    "absent": "Chest radiograph. There is no focal airspace consolidation or evidence of pneumonia.",
}
TARGETS = ("pneumonia", "consolidation")
CONFIG = {"data_origin": "authored_invented_text_not_ehr_or_real_anchor",
    "models": list(MODELS), "arms": list(ARMS), "seeds": list(SEEDS),
    "prompt_changes_diffusion_negative_prompt": False,
    "native_inference_settings_changed": False, "generation_slots": 8, "classifier_slots": 8,
    "maximum_inference_attempts": 16, "maximum_model_load_attempts": 3,
    "model_retries": 0, "minimum_gpu_vram_gib": 16,
    "worker_timeout_seconds": {"roentgen_v2": 240, "chexgenbench_sana": 240, "xrv": 120},
    "report_generation": False, "clinical_acceptance": False,
    "original_ehr_prompt_scores_or_selections_changed": False}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def gpu_guard():
    require_gpu_slurm()
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and f"/job_{job}/" in Path("/proc/self/cgroup").read_text(),
        "actual_gpu_slurm_cgroup_required")


def slots():
    return [{"slot_id": f"probe_{model}_s{seed}_{arm}", "model": model,
        "seed": seed, "arm": arm, "prompt_sha256": hashlib.sha256(PROMPTS[arm].encode()).hexdigest()}
        for model in MODELS for seed in SEEDS for arm in ARMS]


def validate_plan(plan, *, weights=False):
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG
        and plan["prompts"] == PROMPTS and plan["slots"] == slots()
        and set(plan["workers"]) == {*MODELS, "xrv"}
        and plan["asset_source_manifest_sha256"] == ASSET_SOURCE_SHA
        and plan["new_model_calls"] == 0, "fixed_invented_control_plan_required")
    check_pins(plan["source_pins"])
    for name, spec in plan["workers"].items():
        require(spec["status"] == "preflighted" and spec["factory_instantiated"] is False
            and spec["model_id"] == name and os.access(spec["python"], os.X_OK),
            "existing_frozen_worker_required")
        registered = registry()[name]
        require(all(spec[k] == v for k, v in registered.items()), "unchanged_native_registry_required")
        require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == value
            for p, value in spec["asset_stats"].items()), "unchanged_model_asset_stats_required")
        if weights: check_pins(spec["asset_pins"])


def prepare(args):
    cpu_guard()  # Before any source inspection or write; no model factory.
    reader = MetadataReader(ASSET_SOURCE)
    reader.hash(reader.root / "manifest.json", ASSET_SOURCE_SHA)
    manifest = reader.json(reader.root / "manifest.json")
    reader.hash(reader.root / "plan.json", manifest["plan_sha256"])
    previous = reader.json(reader.root / "plan.json")
    workers = {name: deepcopy(previous["workers"][name]) for name in (*MODELS, "xrv")}
    for spec in workers.values():
        spec["asset_stats"] = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in spec["asset_pins"]}
        require(all(spec["asset_stats"][p][0] == pin["size_bytes"] for p, pin in spec["asset_pins"].items()),
            "authenticated_asset_sizes_required")
    sources = {**previous["source_pins"], **{str(p): sha256_file(p) for p in (
        Path(__file__).resolve(), ROOT / "TriCompose-v1.2/tests/test_text_conditioning_probe.py",
        ROOT / "docs/text_conditioning_probe_protocol.md")}}
    plan = {"schema_version": VERSION, "config": CONFIG, "prompts": PROMPTS, "slots": slots(),
        "workers": workers, "source_pins": sources,
        "asset_source_manifest_sha256": ASSET_SOURCE_SHA, "asset_receipt_pins": reader.pins,
        "new_model_calls": 0, "clinical_acceptance": False}
    validate_plan(plan); reader.recheck()
    tmp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(tmp / "plan.json", plan)
        write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
            "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": sha256_file(tmp / "plan.json"),
            "new_model_calls": 0, "clinical_acceptance": False})
        commit_atomic_run(tmp, target)
    except BaseException:
        discard_atomic_run(tmp); raise
    return target


def load_plan(args):
    reader = MetadataReader(args.plan_run)
    reader.hash(reader.root / "manifest.json", args.plan_manifest_sha256)
    manifest = reader.json(reader.root / "manifest.json")
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only_gpu_not_submitted",
        "reviewed_cpu_plan_required")
    reader.hash(reader.root / "plan.json", manifest["plan_sha256"])
    plan = reader.json(reader.root / "plan.json")
    validate_plan(plan); reader.recheck()
    return plan


def freeze_components(runtime):
    """Explicit eval/grad flags on this diagnostic only, never optimizer updates."""
    torch = runtime._torch
    modules = [m for m in runtime.pipe.components.values() if isinstance(m, torch.nn.Module)]
    require(bool(modules), "native_pipeline_modules_required")
    for module in modules:
        module.eval().requires_grad_(False)
    require(all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in modules),
        "frozen_native_components_required")
    return len(modules)


def paired_readout(generation, predictions):
    require(len(generation) == 8 and {r["slot_id"] for r in generation} == {s["slot_id"] for s in slots()},
        "all_eight_generation_slots_retained")
    generated = {r["slot_id"]: r for r in generation}
    for slot in slots():
        require(all(generated[slot["slot_id"]][k] == v for k,v in slot.items()),
            "same_declared_model_seed_arm_and_prompt_required")
    require(len(predictions) == len({r["slot_id"] for r in predictions})
        and all(r["slot_id"] in generated for r in predictions), "unique_bound_predictions_required")
    scored = {r["slot_id"]: r for r in predictions}
    for sid, row in scored.items():
        image = generated[sid]
        require(image["status"] == "completed" and row["image_sha256"] == image["image_sha256"],
            "classifier_uses_own_completed_image")
        for target in TARGETS:
            value = row["scores"][target]
            require(value is None or type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                "finite_operating_point_scores_required")
    result = []
    for model in MODELS:
        for seed in SEEDS:
            a, b = (generated[f"probe_{model}_s{seed}_{arm}"] for arm in ARMS)
            complete = a["status"] == b["status"] == "completed"
            row = {"model": model, "seed": seed, "present_status": a["status"], "absent_status": b["status"],
                "tokenizer_ids_differ": a["input_ids_sha256"] != b["input_ids_sha256"] if complete else None,
                "image_bytes_differ": a["image_sha256"] != b["image_sha256"] if complete else None,
                "clinical_accuracy": None, "clinical_acceptance": False,
                "score_semantics": "xrv_op_norm_0_1_not_probability"}
            for target in TARGETS:
                values = [scored.get(r["slot_id"], {}).get("scores", {}).get(target) for r in (a, b)]
                row["present_" + target + "_score"], row["absent_" + target + "_score"] = values
                row[target + "_score_delta_present_minus_absent"] = values[0] - values[1] if all(v is not None for v in values) else None
            result.append(row)
    return result


def run_worker(args, plan):
    """One model load; reservations are fsynced before model/forward attempts."""
    gpu_guard()
    model = args.model
    require(model in (*MODELS, "xrv"), "fixed_local_worker_required")
    reader = MetadataReader(args.source_run)
    require(reader.root.parent == BASE / "text_conditioning_probe_runs", "new_diagnostic_run_boundary_required")
    start = reader.json(reader.root / "start_manifest.json")
    require(start["schema_version"] == VERSION and start["plan_manifest_sha256"] == args.plan_manifest_sha256
        and start["status"] == "started" and start["automatic_resume"] is False,
        "parent_started_this_exact_fixture_plan_required")
    root = reader.root / model
    private_directory(root)
    work = [s for s in plan["slots"] if s["model"] == model] if model in MODELS else []
    if model == "xrv":
        for name in MODELS:
            path = reader.root / name / "outcomes.json"
            if path.exists():
                # Authored fixture outputs only; no EHR/report/source images.
                outcomes = reader.json(path)
                expected = [s for s in plan["slots"] if s["model"] == name]
                require(outcomes["schema_version"] == VERSION and outcomes["model"] == name
                    and len(outcomes["records"]) == 4
                    and {r["slot_id"] for r in outcomes["records"]} == {s["slot_id"] for s in expected},
                    "exact_declared_generator_outcomes_required")
                declared = {s["slot_id"]: s for s in expected}
                require(all(all(r[k] == v for k,v in declared[r["slot_id"]].items()) for r in outcomes["records"]),
                    "same_declared_generator_metadata_required")
                work.extend(r for r in outcomes["records"] if r["status"] == "completed")
        require(len(work) <= 8 and len({r["slot_id"] for r in work}) == len(work), "bounded_eight_classifier_slots_required")
    records = []; runtime = None; failed = False
    with ProtectedJournal(root / "calls.journal.jsonl") as journal:
        if work:
            journal.append({"event": "model_load_reserved", "model": model})
            try:
                spec = plan["workers"][model]
                check_pins(spec["asset_pins"])
                module = load_script(spec["script"], "_probe_native_" + model)
                if model in MODELS:
                    precision = "float16" if model == "roentgen_v2" else None
                    revision, audit, factory = module._model_components(model, spec["model_dir"], precision)
                    require(revision == spec["model_revision"] and audit == spec["model_audit"], "native_runtime_audit_changed")
                    runtime = factory(); frozen_modules = freeze_components(runtime)
                else:
                    weight = Path(spec["checkpoint"])
                    runtime = module.FrozenXRVRuntime(cache_dir=weight.parent,
                        weight_filename=weight.name, model_name="densenet121-res224-all")
                    require(not runtime.model.training and all(not p.requires_grad for p in runtime.model.parameters()),
                        "frozen_xrv_required")
                    checkpoint_sha = sha256_file(weight)
                    require(checkpoint_sha == spec["checkpoint_sha256"], "same_classifier_checkpoint_required")
                    preprocessing = {"image": "PIL_L_float32", "normalize_maxval": 255,
                        "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
                        "xrv_models_source_sha256": sha256_file(Path(runtime.xrv.models.__file__)),
                        "xrv_datasets_source_sha256": sha256_file(Path(runtime.xrv.datasets.__file__))}
                    fingerprint = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                    require(fingerprint(preprocessing) == spec["scorer_provenance"]["preprocessing_sha256"],
                        "same_xrv_preprocessing_required")
                journal.append({"event": "model_load_completed", "model": model})
            except Exception:
                journal.append({"event": "model_load_failed", "model": model, "automatic_retry": False})
                failed = True
        for slot in work:
            if failed:
                if model in MODELS: records.append({**slot, "status": "blocked_after_failure"})
                continue
            sid = slot["slot_id"]
            journal.append({"event": "inference_reserved", "slot_id": sid, "model": model})
            started = time.monotonic()
            try:
                if model in MODELS:
                    from tricompose_v11.tokenizer_trace import observe_pipeline_tokenizer
                    prompt = PROMPTS[slot["arm"]]
                    with observe_pipeline_tokenizer(runtime) as observer:
                        batch = runtime.generate_batch([prompt], [slot["seed"]])
                    trace = observer.candidate_payload(0, [prompt])
                    require(len(batch.images) == 1 and trace["supplied_prompt_sha256"] == slot["prompt_sha256"]
                        and trace["official_generation_arguments_changed"] is False,
                        "single_image_and_exact_observed_main_prompt_required")
                    edge = 512 if model == "roentgen_v2" else 1024
                    require(batch.images[0].size == (edge, edge), "native_fixed_dimensions_required")
                    image_path = root / (sid + ".png")
                    fd = os.open(image_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,"O_NOFOLLOW",0), 0o660)
                    with os.fdopen(fd,"wb") as image_handle:
                        batch.images[0].save(image_handle, format="PNG")
                    trace_path = write_private_json(root / (sid + "_tokenizer.json"), trace)
                    freeze_components(runtime)
                    record = {**slot, "status": "completed", "image_path": str(image_path),
                        "image_sha256": sha256_file(image_path), "image_dimensions": list(batch.images[0].size),
                        "tokenizer_trace_path": str(trace_path), "tokenizer_trace_sha256": sha256_file(trace_path),
                        "input_ids_sha256": trace["positive_input_ids_sha256"],
                        "tokenizer_text_sha256": trace["positive_tokenizer_text_sha256"],
                        "runtime_observed": trace["runtime_observed"], "pipeline_changed_text": trace["pipeline_changed_text"],
                        "prompt_token_count": batch.prompt_token_counts[0], "peak_vram_gib": batch.peak_vram_gib,
                        "explicitly_frozen_modules": frozen_modules}
                else:
                    path = reader.path(slot["image_path"]); reader.hash(path, slot["image_sha256"])
                    scores = runtime.predict(path)
                    record = {"slot_id": sid, "image_sha256": slot["image_sha256"],
                        "scores": {target: scores.get(target) for target in TARGETS},
                        "checkpoint_sha256": checkpoint_sha,
                        "preprocessing_sha256": spec["scorer_provenance"]["preprocessing_sha256"]}
                record["worker_wall_seconds_including_io"] = round(time.monotonic() - started, 6)
                records.append(record)
                journal.append({"event": "inference_completed", "slot_id": sid, "model": model})
            except Exception:
                journal.append({"event": "inference_failed", "slot_id": sid, "model": model, "automatic_retry": False})
                if model in MODELS: records.append({**slot, "status": "failed_charged"})
                failed = True
    reader.recheck()
    write_private_json(root / "outcomes.json", {"schema_version": VERSION, "model": model, "records": records,
        "automatic_retry": False, "clinical_acceptance": False})


def run(args):
    gpu_guard()  # Before reading plan, creating outputs or spawning.
    plan = load_plan(args)
    import torch
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16 * 1024**3,
        "registered_16gib_planning_memory_required")
    require(RUN_ID_PATTERN.fullmatch(args.run_id), "opaque_run_required")
    output = require_inside(args.output_root, BASE, must_exist=False)
    require(output == BASE / "text_conditioning_probe_runs", "separate_diagnostic_output_required")
    private_directory(output, exist_ok=True)
    root = output / args.run_id; private_directory(root)
    write_private_json(root / "start_manifest.json", {"schema_version": VERSION,
        "plan_manifest_sha256": args.plan_manifest_sha256, "status": "started", "automatic_resume": False})
    generated = []; predictions = []; attempts = []; events = []
    for model in (*MODELS, "xrv"):
        runtime_root = root / (model + "_runtime"); private_directory(runtime_root)
        command = [plan["workers"][model]["python"], str(Path(__file__).resolve()), "worker",
            "--model", model, "--plan-run", str(args.plan_run), "--plan-manifest-sha256", args.plan_manifest_sha256,
            "--source-run", str(root)]
        with ProtectedJournal(root / (model + "_dispatch.journal.jsonl")) as journal:
            journal.append({"event": "worker_process_reserved", "model": model})
            try:
                run_private_process(command, runtime_root, CONFIG["worker_timeout_seconds"][model])
                status = "completed_process"
            except Exception:
                status = "failed_process_charged_no_retry"
            journal.append({"event": status, "model": model, "automatic_retry": False})
        attempts.append({"model": model, "status": status})
        reader = MetadataReader(root)
        journal_path = root / model / "calls.journal.jsonl"
        if journal_path.exists(): events.extend(reader.journal(journal_path))
        outcomes = root / model / "outcomes.json"
        if outcomes.exists():
            records = reader.json(outcomes)["records"]
            if model in MODELS: generated.extend(records)
            else: predictions.extend(records)
        if model in MODELS:
            present = {r["slot_id"] for r in generated}
            generated.extend({**s, "status": "unavailable_failed_worker"} for s in plan["slots"]
                if s["model"] == model and s["slot_id"] not in present)
        reader.recheck()
    paired = paired_readout(generated, predictions)
    count = lambda event: sum(e["event"] == event for e in events)
    charged = count("inference_reserved")
    require(charged <= 16 and count("model_load_reserved") <= 3, "fixed_attempt_caps_required")
    full = all(r["status"] == "completed" for r in generated) and len(predictions) == 8
    summary = {"schema_version": VERSION, "status": "completed_interface_diagnostic_unvalidated" if full
        else "partial_interface_diagnostic_unvalidated",
        "planned_generation_slots": 8, "completed_images": sum(r["status"] == "completed" for r in generated),
        "completed_classifier_slots": len(predictions), "worker_process_attempts": len(attempts),
        "model_load_attempts": count("model_load_reserved"), "charged_inference_attempts": charged,
        "completed_inference_attempts": count("inference_completed"), "failed_inference_attempts": count("inference_failed"),
        "pending_reserved_inference_attempts": charged - count("inference_completed") - count("inference_failed"),
        "automatic_retry": False, "clinical_accuracy": None, "clinical_acceptance": False,
        "primary_clinical_metric_eligible": False, "original_ehr_prompt_scores_or_selections_changed": False}
    output_buffer = io.StringIO(); writer = csv.DictWriter(output_buffer, fieldnames=list(paired[0]))
    writer.writeheader()
    for row in paired: writer.writerow({k: "NA" if v is None else v for k, v in row.items()})
    for name, value in (("generation_slots.json", {"records": generated}), ("predictions.json", {"records": predictions}),
        ("paired_readout.json", {"records": paired}), ("summary.json", summary), ("dispatch_outcomes.json", {"records": attempts})):
        write_private_json(root / name, value)
    write_private_text(root / "paired_readout.csv", output_buffer.getvalue())
    validate_plan(plan)
    artifacts = {str(p.relative_to(root)): {"sha256": sha256_file(p)} for p in root.rglob("*")
        if p.is_file() and not p.relative_to(root).parts[0].endswith("_runtime")}
    write_private_json(root / "manifest.json", {**summary, "plan_manifest_sha256": args.plan_manifest_sha256,
        "artifacts": artifacts})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep, execution, worker = (commands.add_parser(name) for name in ("prepare", "run", "worker"))
    for sub in (prep, execution):
        sub.add_argument("--output-root", type=Path, required=True); sub.add_argument("--run-id", required=True)
    for sub in (execution, worker):
        sub.add_argument("--plan-run", type=Path, required=True); sub.add_argument("--plan-manifest-sha256", required=True)
    worker.add_argument("--model", choices=(*MODELS, "xrv"), required=True)
    worker.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        if args.command == "prepare": root = prepare(args)
        elif args.command == "run": root = run(args)
        else:
            gpu_guard(); run_worker(args, load_plan(args)); return 0
        print(json.dumps({"stage": VERSION, "status": "prepared_cpu_only" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(root / "manifest.json")}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed_private_prefix_retained"}, sort_keys=True))
        return 1


if __name__ == "__main__": raise SystemExit(main())
