#!/usr/bin/env python3
"""Prospective clause-order diagnostic on three fixed remaining synthetic cases.

No clinical gold, threshold fitting, winner replacement, report or planner.
CPU preparation is private and atomic; all loads/forwards require GPU Slurm.
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
    "TriCompose-v1.2/agent", "TriCompose-v1.2/tools", "TriCompose-v1.2/benchmarks",
    "TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0, str(ROOT / relative))

import run_text_conditioning_probe as native
import select_explicit_ehr_cohort as screen_rules
from tricompose_v1.ehr_bridge import canonicalize_synehrgy_case
from tricompose_v11.facts import extract_v11_facts, validate_v11_facts
from tricompose_v11.prompts import render_v11_prompt, validate_v11_prompt, RENDERER_VERSIONS_V11

VERSION = "tricompose-new-synthetic-case-prompt-order-v1"
BASE = native.BASE
MODELS = native.MODELS
SEEDS = (0, 1)
ARMS = ("original", "findings_first")
CASES = ("case_079", "case_080", "case_082")
FINDINGS = {"case_079": "pleural_effusion", "case_080": "pneumonia", "case_082": "pneumonia"}
SCORE_HEADS = {"pleural_effusion": "effusion", "pneumonia": "pneumonia"}
SCREEN = BASE / "ehr_cohorts/v12_explicit_ehr_screen_20261001_001"
SCREEN_SHA = "0e939fc3315f0a3a872477eeaa607b4bf21df670e062207f67e56b06bc1561b5"
SOURCE_SHA = "b38796886fb0a625ba6dd900d36224981a16a370063f5f19b8e976939c36efed"
ASSETS = BASE / "fixed_ehr_prompt_order_plans/prompt_order16_12879534_003"
ASSETS_SHA = "f45718f3a2917ec24fd06c4b8551fd84b958528e9790511d4262c924d3bc06c7"
CONFIG = {"data_origin": "fully_synthetic_pool100_remaining_fixed_eligible_cases",
    "case_ids": list(CASES), "models": list(MODELS), "seeds": list(SEEDS), "arms": list(ARMS),
    "source_cases": 100, "source_structurally_valid": 74, "source_eligible": 5,
    "source_previously_selected": 2, "all_remaining_eligible_retained": True,
    "generation_slots": 24, "classifier_slots": 24, "paired_rows": 12,
    "maximum_inference_attempts": 48, "maximum_model_load_attempts": 3,
    "worker_timeout_seconds": {"roentgen_v2": 300, "chexgenbench_sana": 300, "xrv": 120},
    "minimum_gpu_vram_gib": 16, "model_retries": 0, "report_generation": False,
    "planner_calls": 0, "external_api_calls": 0, "training_allowed": False,
    "changed_only_existing_clause_order_within_pair": True,
    "renderer_is_current_v1_1_4_not_previous_legacy": True,
    "legacy_chf_image_prior_used": False, "new_clinical_gold_available": False,
    "paper_primary_eligible": False, "clinical_acceptance": False,
    "old_ehrs_prompts_scorers_thresholds_and_winners_changed": False}
Reader = native.MetadataReader
require = native.require
sha256_file = native.sha256_file
write_json = native.write_private_json
write_text = native.write_private_text


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_remaining(manifest, rows):
    require(manifest["source_count"] == len(rows) == 100 and manifest["eligible_count"] == 5
        and manifest["selected_count"] == len(manifest["selected_case_ids"]) == 2
        and manifest["rule_version"] == screen_rules.RULE_VERSION
        and manifest["source_run_sha256"] == SOURCE_SHA
        and len({r["case_id"] for r in rows}) == 100
        and sum(r["eligible"] is True for r in rows) == 5,
        "unchanged_complete_historical_screen_required")
    counts = Counter(r["screen_reason"] for r in rows)
    require(counts == Counter(manifest["screen_reason_counts"])
        and counts["structurally_invalid"] == 26, "full_original_pool_denominator_required")
    chosen = [r for r in rows if r["eligible"] and r["case_id"] not in manifest["selected_case_ids"]]
    require(tuple(r["case_id"] for r in chosen) == CASES
        and all(r["eligible_fact_ids"] == [FINDINGS[r["case_id"]]] for r in chosen),
        "all_three_remaining_cases_no_quality_selection_or_replacement")
    return chosen


def reorder(rendering, facts):
    validate_v11_prompt(rendering, facts)
    model = rendering["model_id"]
    require(model in MODELS and rendering["renderer_version"] == RENDERER_VERSIONS_V11[model]
        and rendering["included_direct_fact_ids"] == [FINDINGS[facts["case_id"]]]
        and "congestive_heart_failure_to_cxr_prior_v1" not in rendering["derived_rule_ids"],
        "current_single_grounded_finding_no_legacy_chf_image_prior_required")
    original = rendering["text"]
    marker = "Findings:" if model == "roentgen_v2" else "Radiographic findings:"
    require(original.count(marker) == 1 and text_hash(original) == rendering["prompt_sha256"],
        "exact_single_final_finding_block_required")
    prefix, separator, suffix = original.partition(marker)
    prefix = prefix.rstrip(); block = separator + suffix
    require(prefix.endswith(".") and suffix.strip() and block.endswith(".")
        and original == prefix + " " + block, "complete_existing_clause_boundary_required")
    moved = block + " " + prefix
    require(original != moved and Counter(original.split()) == Counter(moved.split()),
        "only_word_order_may_change")
    return {"original": original, "findings_first": moved}


def slots(cases):
    require(tuple(c["case_id"] for c in cases) == CASES, "fixed_three_case_order_required")
    rows = []
    for case in cases:
        for model in MODELS:
            spec = case["models"][model]
            for seed in SEEDS:
                for arm in ARMS:
                    prompt = spec["prompts"][arm]
                    rows.append({"slot_id": f"neworder_{case['case_id']}_{model}_s{seed}_{arm}",
                        "case_id": case["case_id"], "model": model, "seed": seed, "arm": arm,
                        "reference_finding": FINDINGS[case["case_id"]],
                        "ehr_sha256": case["ehr_sha256"], "ehr_facts_sha256": case["ehr_facts_sha256"],
                        "clinical_intent_sha256": spec["proof"]["clinical_intent_sha256"],
                        "prompt_path": prompt["path"], "prompt_sha256": prompt["sha256"]})
    return rows


def validate_plan(plan):
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG
        and plan["screen_manifest_sha256"] == SCREEN_SHA and plan["source_manifest_sha256"] == SOURCE_SHA
        and plan["asset_receipt_manifest_sha256"] == ASSETS_SHA
        and plan["new_model_calls"] == 0 and plan["real_inputs_or_targets_opened"] is False
        and set(plan["workers"]) == {*MODELS, "xrv"} and plan["slots"] == slots(plan["cases"]),
        "exact_reviewed_new_case_order_plan_required")
    native.check_pins(plan["source_pins"]); native.check_pins(plan["input_pins"])
    require(len({c["ehr_sha256"] for c in plan["cases"]}) == 3
        and not {c["ehr_sha256"] for c in plan["cases"]} & set(plan["excluded_ehr_sha256"]),
        "distinct_canonical_records_disjoint_from_previous_order_cases")
    for name,spec in plan["workers"].items():
        require(spec["status"] == "preflighted" and spec["factory_instantiated"] is False
            and os.access(spec["python"], os.X_OK)
            and all(spec[k] == v for k,v in native.registry()[name].items()),
            "same_existing_frozen_native_worker_required")
        require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == v
            for p,v in spec["asset_stats"].items()), "unchanged_asset_inventory_required")
    for case in plan["cases"]:
        specs = [case["models"][m] for m in MODELS]
        require(len({s["proof"]["clinical_intent_sha256"] for s in specs}) == 1
            and all(s["proof"]["included_direct_fact_ids"] == [FINDINGS[case["case_id"]]]
                and s["proof"]["renderer_version"] == RENDERER_VERSIONS_V11[m]
                and s["proof"]["context_is_not_a_radiographic_assertion"] is True
                and s["proof"]["same_word_multiset"] is True
                and s["proof"]["image_truth_established"] is False
                for m,s in zip(MODELS,specs,strict=True)),
            "same_grounded_diagnosis_intent_not_image_gold_required")


def prepare(args):
    native.cpu_guard()  # Before protected source reads, writes or model factories.
    screen = Reader(SCREEN); screen.hash(screen.root / "screen_manifest.json", SCREEN_SHA)
    manifest = screen.json(screen.root / "screen_manifest.json")
    rows = screen.journal(screen.root / "screen_rows.jsonl")
    chosen = choose_remaining(manifest, rows)
    source = Reader(Path(manifest["source_run"]))
    source.hash(source.root / "run.json", SOURCE_SHA); run = source.json(source.root / "run.json")
    require(run["privacy"]["real_patient_input_used"] is False
        and run["schema"] == screen_rules.SOURCE_SCHEMA
        and run["generator"]["variant"] == "qwen2-40bins"
        and run["generation"]["input"] == "bos_token_only"
        and run["generation"]["base_seed"] == 5200
        and run["generation"]["count"] == len(run["cases"]) == 100,
        "fixed_fully_synthetic_unconditional_pool_required")
    require(all(row["source_index"] == i and row["case_id"] == entry["case_id"]
        and row["source_sha256"] == entry["sha256"]
        for i,(row,entry) in enumerate(zip(rows,run["cases"],strict=True))),
        "complete_screen_rows_bound_to_original_source_manifest")
    assets = Reader(ASSETS); assets.hash(assets.root / "manifest.json", ASSETS_SHA)
    am = assets.json(assets.root / "manifest.json"); assets.hash(assets.root / "plan.json", am["plan_sha256"])
    old = assets.json(assets.root / "plan.json")
    native.check_pins(old["source_pins"]); native.check_pins(old["input_pins"])
    workers = deepcopy(old["workers"])
    for spec in workers.values():
        require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == v
            for p,v in spec["asset_stats"].items()), "sealed_native_asset_stats_required")
    output = native.require_inside(args.output_root, BASE, must_exist=False)
    require(output == BASE / "new_case_prompt_order_plans", "separate_new_case_plan_boundary_required")
    tmp,target = native.new_atomic_run(output, args.run_id)
    cases = []
    try:
        for row in chosen:
            entries = [e for e in run["cases"] if e["case_id"] == row["case_id"]]
            require(len(entries) == 1, "one_immutable_synthetic_source_case_required")
            entry = entries[0]
            path = source.path(source.root / entry["relative_path"])
            require(path == source.root / "cases" / (row["case_id"] + ".json")
                and entry["sha256"] == row["source_sha256"], "matching_fixed_source_record_required")
            source.hash(path, entry["sha256"]); raw = source.json(path)
            require(raw["case_id"] == row["case_id"] and raw["validation"]["strict_valid"] is True,
                "same_structurally_valid_synthetic_record_required")
            canonical = canonicalize_synehrgy_case(raw, source_model_id="synehrgy_qwen2_40bins")
            facts = extract_v11_facts(canonical); validate_v11_facts(facts)
            require(screen_rules.eligible_positive_facts(facts) == (FINDINGS[row["case_id"]],),
                "same_predeclared_eligible_finding_required")
            prefix = Path("cases") / row["case_id"]; native.private_directory(tmp / prefix)
            ehr_path = write_json(tmp / prefix / "synthetic_ehr.json", canonical)
            facts_path = write_json(tmp / prefix / "ehr_facts.json", facts)
            case = {"case_id": row["case_id"], "source_sha256": entry["sha256"],
                "ehr_sha256": sha256_file(ehr_path), "ehr_facts_sha256": sha256_file(facts_path),
                "reference_finding": FINDINGS[row["case_id"]], "models": {}}
            write_json(tmp / prefix / "source_manifest.json", {"source_generator": "synehrgy_qwen2_40bins",
                "source_path": str(path), "source_sha256": entry["sha256"], "canonical_ehr_sha256": case["ehr_sha256"],
                "source_is_fully_synthetic": True, "old_source_changed": False,
                "diagnosis_is_not_independent_image_truth": True})
            for model in MODELS:
                rendering = render_v11_prompt(facts, model); ordered = reorder(rendering, facts)
                proof = {k: deepcopy(rendering[k]) for k in ("clinical_intent_sha256", "renderer_version",
                    "included_direct_fact_ids", "included_context_ids", "omitted_context_ids", "derived_rule_ids")}
                fact = facts["direct_facts"][case["reference_finding"]]
                require(fact["state"] == "positive" and fact["evidence"] and fact["source_fields"],
                    "unchanged_evidence_backed_positive_diagnosis_required")
                proof.update({"direct_evidence": {k: deepcopy(fact[k]) for k in ("state", "evidence", "source_fields")},
                    "same_word_multiset": True, "context_is_not_a_radiographic_assertion": True,
                    "image_truth_established": False, "view_is_fixed_protocol_not_ehr_evidence": True})
                native.private_directory(tmp / prefix / model); prompts = {}
                for arm in ARMS:
                    relative = prefix / model / (arm + ".txt"); write_text(tmp / relative, ordered[arm])
                    prompts[arm] = {"path": str(relative), "sha256": text_hash(ordered[arm]),
                        "is_final_model_input": True, "adapter_must_not_add_prefix": True}
                case["models"][model] = {"proof": proof, "prompts": prompts}
                write_json(tmp / prefix / model / "prompt_manifest.json", {"schema_version": VERSION,
                    "case_id": case["case_id"], "ehr_sha256": case["ehr_sha256"],
                    "ehr_facts_sha256": case["ehr_facts_sha256"], "model": model, "proof": proof, "prompts": prompts})
            cases.append(case)
        pins = {**old["input_pins"]}
        for reader in (screen,source,assets): reader.recheck(); pins.update(reader.pins)
        own = (Path(__file__).resolve(), ROOT / "TriCompose-v1.2/tests/test_new_case_prompt_order.py",
            ROOT / "docs/new_case_prompt_order_protocol.md", Path(screen_rules.__file__))
        sources = {**old["source_pins"], **{str(p): sha256_file(p) for p in own}}
        plan = {"schema_version": VERSION, "config": CONFIG, "cases": cases, "slots": slots(cases),
            "workers": workers, "source_pins": sources, "input_pins": pins,
            "excluded_ehr_sha256": [c["ehr_sha256"] for c in old["cases"]],
            "screen_manifest_sha256": SCREEN_SHA, "source_manifest_sha256": SOURCE_SHA,
            "asset_receipt_manifest_sha256": ASSETS_SHA, "new_model_calls": 0,
            "real_inputs_or_targets_opened": False, "synthetic_ehr_bodies_processed_privately": True,
            "independent_clinical_gold_available": False, "clinical_acceptance": False}
        validate_plan(plan); write_json(tmp / "plan.json", plan)
        artifacts = {str(p.relative_to(tmp)): {"sha256": sha256_file(p)} for p in tmp.rglob("*") if p.is_file()}
        write_json(tmp / "manifest.json", {"schema_version": VERSION, "status": "prepared_cpu_only",
            "plan_sha256": sha256_file(tmp / "plan.json"), "artifacts": artifacts,
            "new_model_calls": 0, "clinical_acceptance": False})
        native.commit_atomic_run(tmp, target)
    except BaseException:
        native.discard_atomic_run(tmp); raise
    return target


def load_plan(args):
    r = Reader(args.plan_run)
    require(r.root.parent == BASE / "new_case_prompt_order_plans", "new_case_plan_boundary_required")
    r.hash(r.root / "manifest.json", args.plan_manifest_sha256); manifest = r.json(r.root / "manifest.json")
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only",
        "reviewed_new_case_plan_required")
    r.hash(r.root / "plan.json", manifest["plan_sha256"])
    for relative,pin in manifest["artifacts"].items(): r.hash(r.root / relative, pin["sha256"])
    plan = r.json(r.root / "plan.json"); validate_plan(plan); r.recheck()
    return plan


def paired_readout(plan, generation, predictions):
    declared = {s["slot_id"]: s for s in plan["slots"]}
    require(len(declared) == len(generation) == 24 and {r["slot_id"] for r in generation} == set(declared),
        "all_twenty_four_slots_retained")
    images = {r["slot_id"]: r for r in generation}
    require(all(all(images[sid][k] == v for k,v in s.items()) for sid,s in declared.items()),
        "same_declared_ehr_finding_prompt_model_and_seed_required")
    require(len(predictions) == len({r["slot_id"] for r in predictions}), "unique_image_score_binding_required")
    scored = {}
    for row in predictions:
        image = images.get(row["slot_id"])
        require(image is not None and image["status"] == "completed" and row["image_sha256"] == image["image_sha256"],
            "scores_must_use_their_own_completed_image")
        require(set(row["scores"]) == set(SCORE_HEADS), "fixed_two_named_head_inventory_required")
        for value in row["scores"].values():
            require(value is None or type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1,
                "finite_raw_operating_point_score_required")
        scored[row["slot_id"]] = row
    pairs = []
    for case in CASES:
        for model in MODELS:
            for seed in SEEDS:
                a,b = (images[f"neworder_{case}_{model}_s{seed}_{arm}"] for arm in ARMS)
                complete = a["status"] == b["status"] == "completed"
                finding = FINDINGS[case]
                values = [scored.get(r["slot_id"],{}).get("scores",{}).get(finding) for r in (a,b)]
                pairs.append({"case_id": case, "model": model, "seed": seed, "reference_finding": finding,
                    "original_status": a["status"], "findings_first_status": b["status"],
                    "original_condition_score": values[0], "findings_first_condition_score": values[1],
                    "condition_delta_findings_first_minus_original": values[1]-values[0] if all(v is not None for v in values) else None,
                    "tokenizer_ids_differ": a["input_ids_sha256"] != b["input_ids_sha256"] if complete else None,
                    "image_bytes_differ": a["image_sha256"] != b["image_sha256"] if complete else None,
                    "reference_is_diagnosis_not_image_gold": True,
                    "score_semantics": "xrv_op_norm_0_1_not_probability",
                    "clinical_accuracy": None, "clinical_acceptance": False})
    return pairs


def run_worker(args, plan):
    native.gpu_guard()
    model = args.model; require(model in (*MODELS,"xrv"), "fixed_native_worker_required")
    reader = Reader(args.source_run)
    require(reader.root.parent == BASE / "new_case_prompt_order_runs", "new_exclusive_run_boundary_required")
    start = reader.json(reader.root / "start_manifest.json")
    require(start["schema_version"] == VERSION and start["plan_manifest_sha256"] == args.plan_manifest_sha256
        and start["automatic_resume"] is False, "same_parent_plan_required")
    root = reader.root / model; native.private_directory(root)
    work = [s for s in plan["slots"] if s["model"] == model] if model in MODELS else []
    if model == "xrv":
        for generator in MODELS:
            path = reader.root / generator / "outcomes.json"
            if not path.is_file(): continue
            outcome = reader.json(path)
            expected = {s["slot_id"]: s for s in plan["slots"] if s["model"] == generator}
            require(outcome["schema_version"] == VERSION and outcome["model"] == generator
                and len(outcome["records"]) == 12 and {r["slot_id"] for r in outcome["records"]} == set(expected)
                and all(all(r[k] == v for k,v in expected[r["slot_id"]].items()) for r in outcome["records"]),
                "complete_fixed_generator_outcome_inventory_required")
            work.extend(r for r in outcome["records"] if r["status"] == "completed")
        require(len(work) <= 24 and len({r["slot_id"] for r in work}) == len(work), "bounded_unique_xrv_slots_required")
    records = []; failed = False; prompts = Reader(args.plan_run)
    with native.ProtectedJournal(root / "calls.journal.jsonl") as journal:
        if work:
            journal.append({"event": "model_load_reserved", "model": model})
            try:
                spec = plan["workers"][model]; native.check_pins(spec["asset_pins"])
                module = native.load_script(spec["script"], "_neworder_native_" + model)
                if model in MODELS:
                    revision,audit,factory = module._model_components(model,spec["model_dir"],"float16" if model == "roentgen_v2" else None)
                    require(revision == spec["model_revision"] and audit == spec["model_audit"], "unchanged_native_audit_required")
                    runtime = factory(); frozen = native.freeze_components(runtime)
                else:
                    weight = Path(spec["checkpoint"])
                    runtime = module.FrozenXRVRuntime(cache_dir=weight.parent,weight_filename=weight.name,model_name="densenet121-res224-all")
                    require(not runtime.model.training and all(not p.requires_grad for p in runtime.model.parameters()), "frozen_xrv_required")
                    checkpoint_sha = sha256_file(weight); require(checkpoint_sha == spec["checkpoint_sha256"], "same_xrv_checkpoint_required")
                    preprocessing = {"image": "PIL_L_float32", "normalize_maxval": 255, "crop": "XRayCenterCrop", "resize": "XRayResizer_224",
                        "xrv_models_source_sha256": sha256_file(Path(runtime.xrv.models.__file__)),
                        "xrv_datasets_source_sha256": sha256_file(Path(runtime.xrv.datasets.__file__))}
                    digest = hashlib.sha256(json.dumps(preprocessing,sort_keys=True).encode()).hexdigest()
                    require(digest == spec["scorer_provenance"]["preprocessing_sha256"], "same_xrv_preprocessing_required")
                journal.append({"event": "model_load_completed", "model": model})
            except Exception:
                journal.append({"event": "model_load_failed", "model": model, "automatic_retry": False}); failed = True
        for slot in work:
            if failed:
                if model in MODELS: records.append({**slot,"status":"blocked_after_failure"})
                continue
            journal.append({"event":"inference_reserved","slot_id":slot["slot_id"],"model":model})
            started = time.monotonic()
            try:
                if model in MODELS:
                    from tricompose_v11.tokenizer_trace import observe_pipeline_tokenizer
                    path = prompts.path(prompts.root / slot["prompt_path"]); prompts.hash(path,slot["prompt_sha256"])
                    require(path.stat().st_size <= 65536, "bounded_synthetic_prompt_required")
                    prompt = path.read_text(encoding="utf-8"); require(text_hash(prompt) == slot["prompt_sha256"], "exact_final_utf8_prompt_required")
                    with observe_pipeline_tokenizer(runtime) as observer: batch = runtime.generate_batch([prompt],[slot["seed"]])
                    trace = observer.candidate_payload(0,[prompt]); edge = 512 if model == "roentgen_v2" else 1024
                    require(len(batch.images) == 1 and batch.images[0].size == (edge,edge)
                        and trace["supplied_prompt_sha256"] == slot["prompt_sha256"] and trace["runtime_observed"] is True
                        and trace["official_generation_arguments_changed"] is False
                        and batch.prompt_token_counts[0] <= (77 if model == "roentgen_v2" else 300),
                        "native_final_input_dimensions_token_budget_and_arguments_required")
                    image_path = root / (slot["slot_id"] + ".png")
                    fd = os.open(image_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o660)
                    with os.fdopen(fd,"wb") as handle: batch.images[0].save(handle,format="PNG")
                    tp = write_json(root / (slot["slot_id"] + "_tokenizer.json"),trace); native.freeze_components(runtime)
                    record = {**slot,"status":"completed","image_path":str(image_path),"image_sha256":sha256_file(image_path),
                        "image_dimensions":list(batch.images[0].size),"input_ids_sha256":trace["positive_input_ids_sha256"],
                        "tokenizer_trace_path":str(tp),"tokenizer_trace_sha256":sha256_file(tp),
                        "runtime_observed":True,"pipeline_changed_text":trace["pipeline_changed_text"],
                        "prompt_token_count":batch.prompt_token_counts[0],"peak_vram_gib":batch.peak_vram_gib,
                        "explicitly_frozen_modules":frozen}
                else:
                    path = reader.path(slot["image_path"]); reader.hash(path,slot["image_sha256"])
                    values = runtime.predict(path)
                    record = {"slot_id":slot["slot_id"],"image_sha256":slot["image_sha256"],
                        "scores":{finding:values.get(head) for finding,head in SCORE_HEADS.items()},
                        "checkpoint_sha256":checkpoint_sha,"preprocessing_sha256":digest,
                        "named_head_mapping":SCORE_HEADS}
                record["worker_wall_seconds_including_io"] = round(time.monotonic()-started,6); records.append(record)
                journal.append({"event":"inference_completed","slot_id":slot["slot_id"],"model":model})
            except Exception:
                journal.append({"event":"inference_failed","slot_id":slot["slot_id"],"model":model,"automatic_retry":False})
                if model in MODELS: records.append({**slot,"status":"failed_charged"})
                failed = True
    reader.recheck(); prompts.recheck()
    write_json(root / "outcomes.json",{"schema_version":VERSION,"model":model,"records":records,"automatic_retry":False})


def run(args):
    native.gpu_guard(); plan = load_plan(args)
    import torch
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16*1024**3,
        "native_gpu_memory_required")
    output = native.require_inside(args.output_root,BASE,must_exist=False)
    require(output == BASE / "new_case_prompt_order_runs" and native.RUN_ID_PATTERN.fullmatch(args.run_id),
        "new_exclusive_order_run_required")
    native.private_directory(output,exist_ok=True); root = output / args.run_id; native.private_directory(root)
    write_json(root / "start_manifest.json",{"schema_version":VERSION,"plan_manifest_sha256":args.plan_manifest_sha256,"automatic_resume":False})
    generation = []; predictions = []; events = []; dispatch = []
    for model in (*MODELS,"xrv"):
        cache = root / (model + "_runtime"); native.private_directory(cache)
        command = [plan["workers"][model]["python"],str(Path(__file__).resolve()),"worker","--model",model,
            "--plan-run",str(args.plan_run),"--plan-manifest-sha256",args.plan_manifest_sha256,"--source-run",str(root)]
        with native.ProtectedJournal(root / (model + "_dispatch.journal.jsonl")) as journal:
            journal.append({"event":"worker_process_reserved","model":model})
            try: native.run_private_process(command,cache,CONFIG["worker_timeout_seconds"][model]); status = "completed_process"
            except Exception: status = "failed_process_charged_no_retry"
            journal.append({"event":status,"model":model,"automatic_retry":False})
        dispatch.append({"model":model,"status":status}); r = Reader(root)
        if (root/model/"calls.journal.jsonl").is_file(): events.extend(r.journal(root/model/"calls.journal.jsonl"))
        if (root/model/"outcomes.json").is_file():
            outcome = r.json(root/model/"outcomes.json")
            require(outcome["schema_version"] == VERSION and outcome["model"] == model, "own_worker_outcome_required")
            if model in MODELS: generation.extend(outcome["records"])
            else: predictions.extend(outcome["records"])
        if model in MODELS:
            observed = {r["slot_id"] for r in generation}
            generation.extend({**s,"status":"unavailable_failed_worker"} for s in plan["slots"]
                if s["model"] == model and s["slot_id"] not in observed)
        r.recheck()
    pairs = paired_readout(plan,generation,predictions); count = lambda event: sum(e["event"] == event for e in events)
    require(count("inference_reserved") <= 48 and count("model_load_reserved") <= 3, "fixed_attempt_caps_required")
    completed = sum(r["status"] == "completed" for r in generation)
    summary = {"schema_version":VERSION,"status":"completed_new_case_order_unvalidated" if completed == len(predictions) == 24 else "partial_new_case_order_unvalidated",
        "planned_images":24,"completed_images":completed,"completed_classifier_slots":len(predictions),"paired_rows":12,
        "source_cases":100,"source_eligible":5,"retained_new_cases":3,"charged_inference_attempts":count("inference_reserved"),
        "model_load_attempts":count("model_load_reserved"),"completed_inference_attempts":count("inference_completed"),
        "failed_inference_attempts":count("inference_failed"),
        "pending_reserved_inference_attempts":count("inference_reserved")-count("inference_completed")-count("inference_failed"),
        "worker_process_attempts":len(dispatch),"automatic_retry":False,"clinical_accuracy":None,"clinical_acceptance":False,
        "old_ehrs_prompts_scorers_thresholds_and_winners_changed":False}
    out = io.StringIO(); writer = csv.DictWriter(out,fieldnames=list(pairs[0])); writer.writeheader()
    for row in pairs: writer.writerow({k:"NA" if v is None else v for k,v in row.items()})
    for name,value in (("summary.json",summary),("generation_slots.json",{"records":generation}),
        ("predictions.json",{"records":predictions}),("paired_readout.json",{"records":pairs}),
        ("dispatch_outcomes.json",{"records":dispatch})): write_json(root/name,value)
    write_text(root/"paired_readout.csv",out.getvalue()); validate_plan(plan)
    artifacts = {str(p.relative_to(root)):{"sha256":sha256_file(p)} for p in root.rglob("*")
        if p.is_file() and not p.relative_to(root).parts[0].endswith("_runtime")}
    write_json(root/"manifest.json",{**summary,"plan_manifest_sha256":args.plan_manifest_sha256,"artifacts":artifacts})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command",required=True)
    prep,execution,worker = (sub.add_parser(name) for name in ("prepare","run","worker"))
    for p in (prep,execution): p.add_argument("--output-root",type=Path,required=True); p.add_argument("--run-id",required=True)
    for p in (execution,worker): p.add_argument("--plan-run",type=Path,required=True); p.add_argument("--plan-manifest-sha256",required=True)
    worker.add_argument("--model",choices=(*MODELS,"xrv"),required=True); worker.add_argument("--source-run",type=Path,required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        if args.command == "prepare": root = prepare(args)
        elif args.command == "run": root = run(args)
        else: native.gpu_guard(); run_worker(args,load_plan(args)); return 0
        print(json.dumps({"stage":VERSION,"status":"prepared_cpu_only" if args.command == "prepare" else "completed",
            "manifest_sha256":sha256_file(root/"manifest.json")},sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage":VERSION,"status":"failed_closed_private_prefix_retained"},sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
