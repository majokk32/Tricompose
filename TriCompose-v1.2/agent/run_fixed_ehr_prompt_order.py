#!/usr/bin/env python3
"""Fixed synthetic EHRs, original clauses in a different order; no repair claim.

CPU preparation renders existing synthetic facts privately. All actual model
loads/forwards require a separately approved GPU Slurm allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

import run_text_conditioning_probe as native
from tricompose_v11.prompts import render_v11_prompt
from tricompose_v11.facts import validate_v11_facts

VERSION = "tricompose-fixed-synthetic-ehr-prompt-order-v1"
BASE = native.BASE
MODELS = native.MODELS
CASES = ("case_009", "case_018")
SEEDS = (0, 1)
ARMS = ("original", "findings_first")
TARGETS = native.TARGETS
LEGACY_FACTS = ["pneumonia", "congestive_heart_failure"]
LEGACY_RULES = ["pneumonia_to_airspace_opacity_prior_v1", "congestive_heart_failure_to_cxr_prior_v1"]
LEGACY_RENDERERS = {"roentgen_v2": "roentgen_v2.ehr_context_prompt.v1_1_3",
    "chexgenbench_sana": "chexgenbench_sana.ehr_context_prompt.v1_1"}
CONFIG = {"data_origin": "original_fully_synthetic_pool80",
    "case_ids": list(CASES), "models": list(MODELS), "seeds": list(SEEDS), "arms": list(ARMS),
    "changed_only_existing_clause_order": True, "generation_slots": 16, "classifier_slots": 16,
    "maximum_inference_attempts": 32, "maximum_model_load_attempts": 3, "model_retries": 0,
    "worker_timeout_seconds": {"roentgen_v2": 300, "chexgenbench_sana": 300, "xrv": 120},
    "minimum_gpu_vram_gib": 16, "report_generation": False, "external_api_calls": 0,
    "training_allowed": False, "clinical_acceptance": False,
    "sealed_legacy_prompts_not_current_renderer_reconstruction": True,
    "legacy_chf_image_prior_preserved_as_unvalidated": True,
    "old_ehr_facts_prompts_scores_or_selections_changed": False}
require = native.require
Reader = native.MetadataReader
sha256_file = native.sha256_file
write_json = native.write_private_json
write_text = native.write_private_text


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reorder_rendering(rendering):
    """Move the EXACT existing final finding block; no clinical reinterpretation."""
    model = rendering["model_id"]
    require(model in MODELS and rendering["renderer_version"] == LEGACY_RENDERERS[model]
        and rendering["is_final_model_input"] is True
        and rendering["context_is_not_a_radiographic_assertion"] is True,
        "existing_final_context_separated_rendering_required")
    ids = rendering["included_direct_fact_ids"]
    require(bool(ids) and len(ids) == len(set(ids)), "existing_positive_finding_block_required")
    original = rendering["text"]
    marker = "Findings:" if model == "roentgen_v2" else "Radiographic findings:"
    require(text_hash(original) == rendering["prompt_sha256"] and original.count(marker) == 1,
        "exact_original_last_finding_block_required")
    prefix, separator, suffix = original.partition(marker)
    prefix = prefix.rstrip(); block = separator + suffix
    require(bool(prefix) and prefix.endswith(".") and bool(suffix.strip()) and block.endswith(".")
        and original == prefix + " " + block, "exact_existing_complete_clause_boundary_required")
    reordered = block + " " + prefix
    require(original != reordered and Counter(original.split()) == Counter(reordered.split()),
        "only_exact_clause_order_may_change")
    return {"original": original, "findings_first": reordered,
        "finding_block": block, "unchanged_nonfinding_prefix": prefix,
        "same_word_multiset": True, "same_fact_context_and_derived_rule_ids": True}


def grounded_proof(facts, rendering):
    """Existing synthetic evidence references only, never a new EHR extractor."""
    proof = {k: deepcopy(rendering[k]) for k in ("clinical_intent_sha256",
        "included_direct_fact_ids", "included_context_ids", "omitted_context_ids",
        "available_context_ids", "derived_rule_ids", "renderer_version")}
    require(rendering["included_direct_fact_ids"] == LEGACY_FACTS
        and rendering["derived_rule_ids"] == LEGACY_RULES,
        "same_declared_legacy_facts_and_unvalidated_priors_required")
    evidence = {}
    for fact in rendering["included_direct_fact_ids"]:
        row = facts["direct_facts"][fact]
        require(row["state"] == "positive" and bool(row["evidence"]) and bool(row["source_fields"]),
            "unchanged_positive_grounded_fact_required")
        evidence[fact] = {k: deepcopy(row[k]) for k in ("state", "evidence", "source_fields")}
    for context in rendering["included_context_ids"]:
        row = facts["clinical_contexts"][context]
        require(row["status"] == "documented" and bool(row["evidence"]) and bool(row["source_fields"]),
            "unchanged_documented_grounded_context_required")
    return {**proof, "direct_fact_evidence": evidence,
        "context_is_not_a_radiographic_assertion": True,
        "inherited_derived_rule_not_new_image_evidence": True,
        "legacy_chf_is_clinical_context_not_independent_image_evidence": True,
        "legacy_image_assertion_validity_established": False,
        "reorder_version": VERSION, "same_word_multiset": True}


def read_text(reader, path, expected):
    p = reader.path(path)
    require(p.stat().st_size <= 65536, "bounded_synthetic_prompt_required")
    reader.hash(p, expected)
    text = p.read_text(encoding="utf-8")
    require(text_hash(text) == expected, "exact_utf8_final_prompt_required")
    reader.hash(p, expected)
    return text


def slots(cases):
    require(tuple(c["case_id"] for c in cases) == CASES, "same_two_predeclared_ehrs_required")
    result = []
    for case in cases:
        for model in MODELS:
            spec = case["models"][model]
            for seed in SEEDS:
                for arm in ARMS:
                    prompt = spec["prompts"][arm]
                    result.append({"slot_id": f"order_{case['case_id']}_{model}_s{seed}_{arm}",
                        "case_id": case["case_id"], "model": model, "seed": seed, "arm": arm,
                        "ehr_sha256": case["ehr_sha256"], "ehr_facts_sha256": case["ehr_facts_sha256"],
                        "clinical_intent_sha256": spec["proof"]["clinical_intent_sha256"],
                        "prompt_path": prompt["path"], "prompt_sha256": prompt["sha256"]})
    return result


def validate_plan(plan):
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG
        and plan["source_plan_manifest_sha256"] == native.ASSET_SOURCE_SHA
        and plan["new_model_calls"] == 0 and plan["real_inputs_or_targets_opened"] is False
        and plan["canonical_ehr_body_parsed"] is False
        and set(plan["workers"]) == {*MODELS, "xrv"} and plan["slots"] == slots(plan["cases"]),
        "exact_reviewed_fixed_ehr_order_plan_required")
    native.check_pins(plan["source_pins"])
    native.check_pins(plan["input_pins"])
    for name, spec in plan["workers"].items():
        require(spec["status"] == "preflighted" and spec["factory_instantiated"] is False
            and spec["model_id"] == name and os.access(spec["python"], os.X_OK)
            and all(spec[k] == v for k,v in native.registry()[name].items()),
            "same_preflighted_native_worker_required")
        require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == v
            for p,v in spec["asset_stats"].items()), "unchanged_existing_asset_stats_required")
    for case in plan["cases"]:
        specs = [case["models"][m] for m in MODELS]
        require(len({s["proof"]["clinical_intent_sha256"] for s in specs}) == 1
            and all(s["proof"]["same_word_multiset"] is True for s in specs)
            and all(s["proof"]["included_direct_fact_ids"] == LEGACY_FACTS
                and s["proof"]["derived_rule_ids"] == LEGACY_RULES
                and s["proof"]["renderer_version"] == LEGACY_RENDERERS[m]
                and s["proof"]["legacy_image_assertion_validity_established"] is False
                for m,s in zip(MODELS,specs,strict=True)),
            "same_legacy_diagnosis_intent_not_independent_image_truth_required")


def prepare(args):
    native.cpu_guard()  # Before source bodies, writes, factories or models.
    reader = Reader(native.ASSET_SOURCE)
    reader.hash(reader.root / "manifest.json", native.ASSET_SOURCE_SHA)
    manifest = reader.json(reader.root / "manifest.json")
    reader.hash(reader.root / "plan.json", manifest["plan_sha256"])
    source = reader.json(reader.root / "plan.json")
    require(source["data_origin"] == "original_fully_synthetic_pool80"
        and tuple(c["case_id"] for c in source["cases"]) == CASES,
        "known_fully_synthetic_development_anchors_required")
    native.check_pins(source["source_pins"])
    workers = {m: deepcopy(source["workers"][m]) for m in (*MODELS, "xrv")}
    for spec in workers.values():
        spec["asset_stats"] = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in spec["asset_pins"]}
        require(all(spec["asset_stats"][p][0] == pin["size_bytes"] for p,pin in spec["asset_pins"].items()),
            "authenticated_existing_asset_sizes_required")
    output = native.require_inside(args.output_root, BASE, must_exist=False)
    require(output == BASE / "fixed_ehr_prompt_order_plans", "separate_order_plan_boundary_required")
    tmp, target = native.new_atomic_run(output, args.run_id)
    readers = [reader]; cases = []
    try:
        for old in source["cases"]:
            case = {"case_id": old["case_id"], "ehr_sha256": old["anchor"]["ehr_sha256"],
                "ehr_facts_sha256": old["anchor"]["ehr_facts_sha256"], "models": {}}
            for model in MODELS:
                matches = [r["request"] for r in old["requests"] if r["request"]["model_id"] == model]
                require(len(matches) == 1, "one_authenticated_original_request_per_model_required")
                request = matches[0]; inputs = request["inputs"]
                require(inputs["synthetic_ehr"]["sha256"] == case["ehr_sha256"]
                    and inputs["ehr_facts"]["sha256"] == case["ehr_facts_sha256"],
                    "same_existing_ehr_and_facts_required")
                body_reader = Reader(Path(inputs["ehr_facts"]["path"]).parent); readers.append(body_reader)
                body_reader.hash(inputs["synthetic_ehr"]["path"], case["ehr_sha256"])
                body_reader.hash(inputs["ehr_facts"]["path"], case["ehr_facts_sha256"])
                facts = body_reader.json(inputs["ehr_facts"]["path"])
                require(facts["case_id"] == case["case_id"], "same_opaque_synthetic_fact_case_required")
                validate_v11_facts(facts)
                prompt = inputs["final_prompt"]
                original = read_text(body_reader, prompt["path"], prompt["sha256"])
                source_manifest = body_reader.json(Path(prompt["path"]).parent / "prompt_manifest.json")
                require(source_manifest["case_id"] == case["case_id"]
                    and source_manifest["ehr_facts_sha256"] == case["ehr_facts_sha256"]
                    and source_manifest["one_shared_clinical_intent"] is True
                    and source_manifest["adapter_must_not_add_prefix"] is True,
                    "matching_existing_staging_metadata_required")
                rendering = deepcopy(source_manifest["models"][model])
                require(rendering["prompt_sha256"] == prompt["sha256"]
                    and all(rendering[k] == prompt[k] for k in ("clinical_intent_sha256", "included_direct_fact_ids",
                        "available_context_ids", "included_context_ids", "omitted_context_ids", "renderer_version")),
                    "sealed_request_and_source_renderer_metadata_must_match")
                rendering.update({"model_id": model, "text": original})
                reordered = reorder_rendering(rendering)
                require(original == reordered["original"], "unchanged_original_final_text_required")
                prefix = Path("cases") / case["case_id"] / model
                native.private_directory(tmp / prefix)
                prompts = {}
                for arm in ARMS:
                    relative = prefix / (arm + ".txt")
                    write_text(tmp / relative, reordered[arm])
                    prompts[arm] = {"path": str(relative), "sha256": text_hash(reordered[arm]),
                        "is_final_model_input": True, "adapter_must_not_add_prefix": True}
                proof = grounded_proof(facts, rendering)
                current = render_v11_prompt(facts, model)
                proof["current_renderer_reconstruction_matches_legacy"] = current["prompt_sha256"] == prompt["sha256"]
                proof["current_renderer_used_for_generation"] = False
                case["models"][model] = {"proof": proof, "prompts": prompts,
                    "original_source_prompt_sha256": prompt["sha256"],
                    "original_source_prompt_path": prompt["path"]}
                write_json(tmp / prefix / "prompt_manifest.json", {"schema_version": VERSION,
                    "case_id": case["case_id"], "ehr_sha256": case["ehr_sha256"],
                    "ehr_facts_sha256": case["ehr_facts_sha256"], "model": model,
                    "proof": proof, "prompts": prompts})
            cases.append(case)
        inputs_pins = {**source["artifact_pins"]}
        for r in readers: r.recheck(); inputs_pins.update(r.pins)
        own = (Path(__file__).resolve(), ROOT / "TriCompose-v1.2/tests/test_fixed_ehr_prompt_order.py",
            ROOT / "docs/fixed_ehr_prompt_order_protocol.md")
        sources = {**source["source_pins"], str(Path(native.__file__)): sha256_file(native.__file__),
            **{str(p): sha256_file(p) for p in own}}
        plan = {"schema_version": VERSION, "config": CONFIG, "cases": cases, "slots": slots(cases),
            "workers": workers, "source_pins": sources, "input_pins": inputs_pins,
            "source_plan_manifest_sha256": native.ASSET_SOURCE_SHA, "new_model_calls": 0,
            "real_inputs_or_targets_opened": False, "canonical_ehr_body_parsed": False,
            "synthetic_facts_and_original_prompts_rendered_privately": True,
            "historical_cost_is_not_free": True, "clinical_acceptance": False}
        validate_plan(plan)
        write_json(tmp / "plan.json", plan)
        artifacts = {str(p.relative_to(tmp)): {"sha256": sha256_file(p)} for p in tmp.rglob("*") if p.is_file()}
        write_json(tmp / "manifest.json", {"schema_version": VERSION, "status": "prepared_cpu_only",
            "plan_sha256": sha256_file(tmp / "plan.json"), "artifacts": artifacts,
            "new_model_calls": 0, "clinical_acceptance": False})
        native.commit_atomic_run(tmp, target)
    except BaseException:
        native.discard_atomic_run(tmp); raise
    return target


def load_plan(args):
    reader = Reader(args.plan_run)
    require(reader.root.parent == BASE / "fixed_ehr_prompt_order_plans", "fixed_plan_boundary_required")
    reader.hash(reader.root / "manifest.json", args.plan_manifest_sha256)
    manifest = reader.json(reader.root / "manifest.json")
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only",
        "reviewed_fixed_order_plan_required")
    reader.hash(reader.root / "plan.json", manifest["plan_sha256"])
    for relative,pin in manifest["artifacts"].items(): reader.hash(reader.root / relative, pin["sha256"])
    plan = reader.json(reader.root / "plan.json")
    validate_plan(plan); reader.recheck()
    return plan


def paired_readout(plan, generation, predictions):
    declared = {s["slot_id"]: s for s in plan["slots"]}
    require(len(declared) == len(generation) == 16
        and {r["slot_id"] for r in generation} == set(declared), "all_sixteen_slots_retained")
    generated = {r["slot_id"]: r for r in generation}
    require(all(all(generated[sid][k] == v for k,v in s.items()) for sid,s in declared.items()),
        "same_declared_ehr_fact_prompt_arm_model_and_seed_required")
    require(len(predictions) == len({r["slot_id"] for r in predictions}), "unique_image_score_bindings_required")
    scored = {}
    for row in predictions:
        image = generated.get(row["slot_id"])
        require(image is not None and image["status"] == "completed" and image["image_sha256"] == row["image_sha256"],
            "score_must_use_its_own_completed_image")
        for finding in TARGETS:
            value = row["scores"][finding]
            require(value is None or type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1,
                "finite_raw_operating_point_score_required")
        scored[row["slot_id"]] = row
    pairs = []
    for case in CASES:
        for model in MODELS:
            for seed in SEEDS:
                a,b = (generated[f"order_{case}_{model}_s{seed}_{arm}"] for arm in ARMS)
                complete = a["status"] == b["status"] == "completed"
                row = {"case_id": case, "model": model, "seed": seed,
                    "original_status": a["status"], "findings_first_status": b["status"],
                    "tokenizer_ids_differ": a["input_ids_sha256"] != b["input_ids_sha256"] if complete else None,
                    "image_bytes_differ": a["image_sha256"] != b["image_sha256"] if complete else None,
                    "clinical_accuracy": None, "clinical_acceptance": False,
                    "score_semantics": "xrv_op_norm_0_1_not_probability"}
                for finding in TARGETS:
                    values = [scored.get(r["slot_id"], {}).get("scores", {}).get(finding) for r in (a,b)]
                    row["original_" + finding + "_score"], row["findings_first_" + finding + "_score"] = values
                    row[finding + "_delta_findings_first_minus_original"] = values[1]-values[0] if all(v is not None for v in values) else None
                pairs.append(row)
    return pairs


def run_worker(args, plan):
    native.gpu_guard()
    model = args.model
    require(model in (*MODELS, "xrv"), "fixed_native_model_worker_required")
    reader = Reader(args.source_run)
    require(reader.root.parent == BASE / "fixed_ehr_prompt_order_runs", "new_protected_order_run_required")
    start = reader.json(reader.root / "start_manifest.json")
    require(start["schema_version"] == VERSION and start["plan_manifest_sha256"] == args.plan_manifest_sha256
        and start["automatic_resume"] is False, "parent_started_same_reviewed_plan_required")
    root = reader.root / model; native.private_directory(root)
    work = [s for s in plan["slots"] if s["model"] == model] if model in MODELS else []
    if model == "xrv":
        for generator in MODELS:
            path = reader.root / generator / "outcomes.json"
            if path.is_file():
                outcome = reader.json(path)
                expected = [s for s in plan["slots"] if s["model"] == generator]
                declared = {s["slot_id"]: s for s in expected}
                require(outcome["schema_version"] == VERSION and outcome["model"] == generator
                    and len(outcome["records"]) == 8
                    and {r["slot_id"] for r in outcome["records"]} == set(declared)
                    and all(all(r[k] == v for k,v in declared[r["slot_id"]].items()) for r in outcome["records"]),
                    "all_original_declared_generator_records_required")
                work.extend(r for r in outcome["records"] if r["status"] == "completed")
        require(len(work) <= 16 and len({r["slot_id"] for r in work}) == len(work), "bounded_unique_classifier_slots_required")
    records = []; failed = False; runtime = None
    prompt_reader = Reader(args.plan_run)
    with native.ProtectedJournal(root / "calls.journal.jsonl") as journal:
        if work:
            journal.append({"event": "model_load_reserved", "model": model})
            try:
                spec = plan["workers"][model]; native.check_pins(spec["asset_pins"])
                module = native.load_script(spec["script"], "_order_native_" + model)
                if model in MODELS:
                    revision,audit,factory = module._model_components(model, spec["model_dir"], "float16" if model == "roentgen_v2" else None)
                    require(revision == spec["model_revision"] and audit == spec["model_audit"], "unchanged_native_runtime_audit_required")
                    runtime = factory(); frozen_modules = native.freeze_components(runtime)
                else:
                    weight = Path(spec["checkpoint"])
                    runtime = module.FrozenXRVRuntime(cache_dir=weight.parent, weight_filename=weight.name, model_name="densenet121-res224-all")
                    require(not runtime.model.training and all(not p.requires_grad for p in runtime.model.parameters()), "frozen_xrv_required")
                    checkpoint_sha = sha256_file(weight)
                    require(checkpoint_sha == spec["checkpoint_sha256"], "unchanged_classifier_checkpoint_required")
                    pre = {"image": "PIL_L_float32", "normalize_maxval": 255, "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
                        "xrv_models_source_sha256": sha256_file(Path(runtime.xrv.models.__file__)),
                        "xrv_datasets_source_sha256": sha256_file(Path(runtime.xrv.datasets.__file__))}
                    digest = hashlib.sha256(json.dumps(pre, sort_keys=True).encode()).hexdigest()
                    require(digest == spec["scorer_provenance"]["preprocessing_sha256"], "unchanged_xrv_preprocessing_required")
                journal.append({"event": "model_load_completed", "model": model})
            except Exception:
                journal.append({"event": "model_load_failed", "model": model, "automatic_retry": False}); failed = True
        for slot in work:
            if failed:
                if model in MODELS: records.append({**slot, "status": "blocked_after_failure"})
                continue
            journal.append({"event": "inference_reserved", "slot_id": slot["slot_id"], "model": model})
            started = time.monotonic()
            try:
                if model in MODELS:
                    from tricompose_v11.tokenizer_trace import observe_pipeline_tokenizer
                    prompt = read_text(prompt_reader, prompt_reader.root / slot["prompt_path"], slot["prompt_sha256"])
                    with observe_pipeline_tokenizer(runtime) as observer: batch = runtime.generate_batch([prompt], [slot["seed"]])
                    trace = observer.candidate_payload(0, [prompt])
                    edge = 512 if model == "roentgen_v2" else 1024
                    require(len(batch.images) == 1 and batch.images[0].size == (edge,edge)
                        and trace["supplied_prompt_sha256"] == slot["prompt_sha256"]
                        and trace["official_generation_arguments_changed"] is False, "native_dimensions_and_exact_final_input_required")
                    path = root / (slot["slot_id"] + ".png")
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o660)
                    with os.fdopen(fd,"wb") as handle: batch.images[0].save(handle, format="PNG")
                    trace_path = write_json(root / (slot["slot_id"] + "_tokenizer.json"), trace)
                    native.freeze_components(runtime)
                    record = {**slot, "status": "completed", "image_path": str(path), "image_sha256": sha256_file(path),
                        "image_dimensions": list(batch.images[0].size), "input_ids_sha256": trace["positive_input_ids_sha256"],
                        "tokenizer_trace_path": str(trace_path), "tokenizer_trace_sha256": sha256_file(trace_path),
                        "tokenizer_text_sha256": trace["positive_tokenizer_text_sha256"],
                        "padded_token_count": trace["padded_token_count"], "attention_token_count": trace["attention_token_count"],
                        "prompt_token_count": batch.prompt_token_counts[0], "runtime_observed": trace["runtime_observed"],
                        "pipeline_changed_text": trace["pipeline_changed_text"], "peak_vram_gib": batch.peak_vram_gib,
                        "explicitly_frozen_modules": frozen_modules}
                else:
                    path = reader.path(slot["image_path"]); reader.hash(path, slot["image_sha256"])
                    scores = runtime.predict(path)
                    record = {"slot_id": slot["slot_id"], "image_sha256": slot["image_sha256"],
                        "scores": {t: scores.get(t) for t in TARGETS}, "checkpoint_sha256": checkpoint_sha,
                        "preprocessing_sha256": spec["scorer_provenance"]["preprocessing_sha256"]}
                record["worker_wall_seconds_including_io"] = round(time.monotonic()-started, 6); records.append(record)
                journal.append({"event": "inference_completed", "slot_id": slot["slot_id"], "model": model})
            except Exception:
                journal.append({"event": "inference_failed", "slot_id": slot["slot_id"], "model": model, "automatic_retry": False})
                if model in MODELS: records.append({**slot, "status": "failed_charged"})
                failed = True
    reader.recheck(); prompt_reader.recheck()
    write_json(root / "outcomes.json", {"schema_version": VERSION, "model": model, "records": records, "automatic_retry": False})


def run(args):
    native.gpu_guard()
    plan = load_plan(args)
    import torch
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16 * 1024**3,
        "native_planning_gpu_memory_required")
    output = native.require_inside(args.output_root, BASE, must_exist=False)
    require(output == BASE / "fixed_ehr_prompt_order_runs" and native.RUN_ID_PATTERN.fullmatch(args.run_id), "new_exclusive_run_boundary_required")
    native.private_directory(output, exist_ok=True)
    root = output / args.run_id; native.private_directory(root)
    write_json(root / "start_manifest.json", {"schema_version": VERSION, "plan_manifest_sha256": args.plan_manifest_sha256,
        "automatic_resume": False, "clinical_acceptance": False})
    generation = []; predictions = []; attempts = []; events = []
    for model in (*MODELS, "xrv"):
        cache = root / (model + "_runtime"); native.private_directory(cache)
        command = [plan["workers"][model]["python"], str(Path(__file__).resolve()), "worker", "--model", model,
            "--plan-run", str(args.plan_run), "--plan-manifest-sha256", args.plan_manifest_sha256, "--source-run", str(root)]
        with native.ProtectedJournal(root / (model + "_dispatch.journal.jsonl")) as journal:
            journal.append({"event": "worker_process_reserved", "model": model})
            try:
                native.run_private_process(command, cache, CONFIG["worker_timeout_seconds"][model]); status = "completed_process"
            except Exception: status = "failed_process_charged_no_retry"
            journal.append({"event": status, "model": model, "automatic_retry": False})
        attempts.append({"model": model, "status": status})
        reader = Reader(root)
        path = root / model / "calls.journal.jsonl"
        if path.is_file(): events.extend(reader.journal(path))
        path = root / model / "outcomes.json"
        if path.is_file():
            payload = reader.json(path)
            require(payload["schema_version"] == VERSION and payload["model"] == model, "own_model_outcome_required")
            if model in MODELS: generation.extend(payload["records"])
            else: predictions.extend(payload["records"])
        if model in MODELS:
            observed = {r["slot_id"] for r in generation}
            generation.extend({**s, "status": "unavailable_failed_worker"} for s in plan["slots"]
                if s["model"] == model and s["slot_id"] not in observed)
        reader.recheck()
    pairs = paired_readout(plan, generation, predictions)
    count = lambda name: sum(e["event"] == name for e in events)
    charged = count("inference_reserved")
    require(charged <= 32 and count("model_load_reserved") <= 3, "fixed_no_retry_attempt_caps_required")
    complete = sum(r["status"] == "completed" for r in generation)
    summary = {"schema_version": VERSION, "status": "completed_order_diagnostic_unvalidated" if complete == len(predictions) == 16
        else "partial_order_diagnostic_unvalidated", "planned_generation_slots": 16, "paired_rows": 8,
        "completed_images": complete, "completed_classifier_slots": len(predictions), "worker_process_attempts": len(attempts),
        "charged_inference_attempts": charged, "model_load_attempts": count("model_load_reserved"),
        "completed_inference_attempts": count("inference_completed"), "failed_inference_attempts": count("inference_failed"),
        "pending_reserved_inference_attempts": charged-count("inference_completed")-count("inference_failed"),
        "automatic_retry": False, "clinical_accuracy": None, "clinical_acceptance": False,
        "old_ehr_facts_prompts_scores_or_selections_changed": False}
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=list(pairs[0])); writer.writeheader()
    for row in pairs: writer.writerow({k: "NA" if v is None else v for k,v in row.items()})
    for name,value in (("summary.json",summary), ("generation_slots.json",{"records":generation}),
        ("predictions.json",{"records":predictions}), ("paired_readout.json",{"records":pairs}),
        ("dispatch_outcomes.json",{"records":attempts})): write_json(root / name, value)
    write_text(root / "paired_readout.csv", out.getvalue())
    validate_plan(plan)
    artifacts = {str(p.relative_to(root)): {"sha256": sha256_file(p)} for p in root.rglob("*")
        if p.is_file() and not p.relative_to(root).parts[0].endswith("_runtime")}
    write_json(root / "manifest.json", {**summary, "plan_manifest_sha256": args.plan_manifest_sha256, "artifacts": artifacts})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    prep,execution,worker = (sub.add_parser(name) for name in ("prepare","run","worker"))
    for p in (prep,execution): p.add_argument("--output-root", type=Path, required=True); p.add_argument("--run-id", required=True)
    for p in (execution,worker): p.add_argument("--plan-run", type=Path, required=True); p.add_argument("--plan-manifest-sha256", required=True)
    worker.add_argument("--model", choices=(*MODELS,"xrv"), required=True); worker.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        if args.command == "prepare": root = prepare(args)
        elif args.command == "run": root = run(args)
        else: native.gpu_guard(); run_worker(args, load_plan(args)); return 0
        print(json.dumps({"stage": VERSION, "status": "prepared_cpu_only" if args.command == "prepare" else "completed",
            "manifest_sha256": sha256_file(root / "manifest.json")}, sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed_private_prefix_retained"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
