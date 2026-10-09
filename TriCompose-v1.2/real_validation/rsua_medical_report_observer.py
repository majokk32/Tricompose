#!/usr/bin/env python3
"""Same RSUA50: unchanged CheXagent SRRG -> frozen CheXbert, proxy only.

CPU preparation reads receipts/stats, never images or reference rows. Separate
approved GPU subprocesses release the report model before the labeler loads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import gc
import json
import os
from pathlib import Path
import sys
import time

import rsua_qwen_observer as prior
from contracts import CHEXPERT_FINDINGS, private_directory
from tricompose_v12.live_workers import (
    CHEXBERT_BERT, check_pins, require_gpu_slurm, run_private_process)

VERSION = "tricompose-rsua-official-srrg-chexbert-observer-v1"
BASE = prior.BASE
require = prior.require
NATIVE = BASE / "llm_fresh_report_plans/fresh_report2_fullassets_12799642_001"
NATIVE_SHA = "01516d6965c3b0e287acff715da1735086580ddc8aa8d9ef39525ea8a9533e1f"
INPUT_PLAN = BASE / "real_validation/rsua_qwen_plans/rsua_qwen50_12851223_001"
INPUT_PLAN_SHA = "e8a38e289adc778bdc4fc89a91d31cdf4fc832414226db5261932ce3d5a9d6b3"
CONFIG = {"case_count": 50, "seed": 42, "precision": "float32",
    "max_new_tokens": 512, "min_vram_gib": 24, "retries": 0,
    "maximum_report_requests": 50, "maximum_label_requests": 50,
    "maximum_report_loads": 1, "maximum_label_loads": 1,
    "report_process_timeout_seconds": 420, "label_process_timeout_seconds": 60}


def record_base(item):
    return {"case_id": item["case_id"], "png_sha256": item["png_sha256"],
        "source_image_sha256": item["original_bmp_sha256"]}


def generate_records(inputs, *, inspect, generate, save, journal):
    """No source labels, retries, substitutions or dropped denominators."""
    require(0 < len(inputs) <= CONFIG["maximum_report_requests"]
        and len({i["case_id"] for i in inputs}) == len(inputs)
        and len({i["png_sha256"] for i in inputs}) == len(inputs), "unique_fixed_inputs_required")
    records = []; stopped = False
    for ordinal, item in enumerate(inputs):
        base = {**record_base(item), "model_called": False, "guard": None,
            "report_path": None, "report_sha256": None, "report_metadata": None}
        if stopped:
            records.append({**base, "report_status": "not_attempted_after_runtime_failure"})
            continue
        receipt = inspect(item)
        prior.guard.validate_receipt(receipt)
        require(receipt["artifact_sha256"] == item["png_sha256"], "same_checked_image_required")
        base["guard"] = receipt
        if not receipt["basic_comparison_permitted"]:
            records.append({**base, "report_status": "blocked_before_model_call"})
            journal({"event": "report_blocked", "ordinal": ordinal})
            continue
        journal({"event": "report_request_reserved", "ordinal": ordinal,
            "png_sha256": item["png_sha256"]})
        base["model_called"] = True
        try:
            generated = generate(item)
        except Exception:
            stopped = True
            records.append({**base, "report_status": "runtime_failed_unavailable"})
            journal({"event": "report_request_finished", "ordinal": ordinal, "status": "runtime_failed_unavailable"})
            continue
        require(isinstance(generated, dict) and set(generated) == {"text", "metadata", "tokens"}
            and isinstance(generated["text"], str) and len(generated["text"].encode()) <= 65536,
            "bounded_official_report_required")
        tokens = generated["tokens"]
        require(set(tokens) == {"input_tokens", "output_tokens", "token_limit_reached"}
            and type(tokens["input_tokens"]) is int and tokens["input_tokens"] > 0
            and type(tokens["output_tokens"]) is int and 0 <= tokens["output_tokens"] <= 512
            and tokens["token_limit_reached"] is (tokens["output_tokens"] >= 512), "actual_generation_token_receipt_required")
        saved = save(item, generated["text"])
        require(set(saved) == {"report_path", "report_sha256"}
            and prior.guard.HASH.fullmatch(saved["report_sha256"]), "protected_report_hash_required")
        status = ("empty_unavailable" if not generated["text"].strip() else
            "token_cap_unavailable" if tokens["token_limit_reached"] else "generated_available")
        records.append({**base, **saved, "report_status": status,
            "report_metadata": generated["metadata"], "tokens": tokens})
        journal({"event": "report_request_finished", "ordinal": ordinal, "status": status})
    return records


def normalized_chexbert_input(text):
    # Exact literal replacements in the unchanged cxrmate/tools/chexbert.py.
    return text.strip().replace("\n", " ").replace("\\s+", " ").replace("\\s+(?=[\\.,])", "").strip()


def label_records(reports, *, load, length, infer, journal):
    """Labeler unavailable and unmentioned finding unknown are different."""
    require(0 < len(reports) <= CONFIG["maximum_label_requests"]
        and len({r["case_id"] for r in reports}) == len(reports), "unique_bounded_label_scope_required")
    ready = [r for r in reports if r["report_status"] == "generated_available"]
    loaded = False; stopped = False; load_failed = False
    if ready:
        journal({"event": "label_load_reserved"})
        try:
            load(); loaded = True
        except Exception:
            load_failed = True
    records = []
    for ordinal, report in enumerate(reports):
        row = {**report, "states": None, "finding_states": None, "label_called": False,
            "label_input_tokens": None, "label_status": "not_applicable_report_unavailable"}
        if report["report_status"] != "generated_available":
            row["contract_status"] = (report["report_status"] if not report["model_called"] else "failed_unavailable")
        elif not loaded:
            row.update(contract_status="failed_unavailable", label_status="load_failed_unavailable")
        elif stopped:
            row.update(contract_status="failed_unavailable", label_status="not_attempted_after_label_runtime_failure")
        else:
            token_count, maximum = length(report)
            require(type(token_count) is int and type(maximum) is int and token_count > 0 and maximum > 0,
                "exact_untruncated_label_token_bound_required")
            row["label_input_tokens"] = token_count
            if token_count > maximum:
                row.update(contract_status="failed_unavailable", label_status="overlength_unavailable_no_truncation")
            else:
                journal({"event": "label_request_reserved", "ordinal": ordinal,
                    "report_sha256": report["report_sha256"]})
                row["label_called"] = True
                try:
                    states = infer(report)
                except Exception:
                    stopped = True
                    row.update(contract_status="failed_unavailable", label_status="runtime_failed_unavailable")
                else:
                    require(isinstance(states, dict) and set(states) == set(CHEXPERT_FINDINGS)
                        and set(states.values()) <= prior.guard.STATES,
                        "complete_named_four_state_chexbert_vector_required")
                    row.update(contract_status="complete", label_status="complete", finding_states=states,
                        states={k: states[k] for k in prior.guard.HEADS})
                journal({"event": "label_request_finished", "ordinal": ordinal, "status": row["label_status"]})
        records.append(row)
    return records, load_failed or stopped


def tracked_official_generate(runtime, path):
    """Observe lengths; forward every original adapter argument unchanged."""
    original = runtime._model.generate; capture = {}
    def tracked(ids, **kwargs):
        output = original(ids, **kwargs)
        n = int(output[0].shape[-1]) - int(ids.shape[-1])
        capture.update(input_tokens=int(ids.shape[-1]), output_tokens=n, token_limit_reached=n >= 512)
        return output
    runtime._model.generate = tracked
    try:
        result = runtime.generate(path)
    finally:
        runtime._model.generate = original
    require(bool(capture), "actual_official_generate_call_required")
    return {"text": result.canonical_text, "metadata": result.metadata, "tokens": capture}


def load_plan(args, *, weights=False):
    reader, manifest = prior.manifest(args.plan_run, args.plan_manifest_sha256)
    require(manifest["schema_version"] == VERSION and manifest["status"] == "prepared_cpu_only_gpu_not_submitted",
        "reviewed_medical_observer_plan_required")
    reader.hash(reader.root / "plan.json", manifest["plan_sha256"])
    plan = reader.json(reader.root / "plan.json")
    require(plan["schema_version"] == VERSION and plan["config"] == CONFIG
        and [i["case_id"] for i in plan["inputs"]] == prior.EXPECTED_IDS
        and plan["cohort_sha256"] == prior.COHORT_SHA
        and plan["cohort_labels_in_model_inputs"] is False, "unchanged_fifty_blind_inputs_required")
    for key in ("source_pins", "artifact_pins"): check_pins(plan[key])
    for spec in plan["workers"].values():
        if weights: check_pins(spec["asset_pins"])
        require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == v
            for p, v in spec["asset_stats"].items()), "unchanged_readonly_assets_required")
    reader.recheck()
    return plan


def prepare(args):
    prior.observer.source.previous.gate.cpu_guard()
    ir, im = prior.manifest(INPUT_PLAN, INPUT_PLAN_SHA)
    ir.hash(ir.root / "plan.json", im["plan_sha256"]); inputs = ir.json(ir.root / "plan.json")
    nr, nm = prior.manifest(NATIVE, NATIVE_SHA)
    nr.hash(nr.root / "plan.json", nm["plan_sha256"]); native = nr.json(nr.root / "plan.json")
    workers = {name: dict(native["workers"][name]) for name in ("chexagent2", "chexbert")}
    require(workers["chexagent2"]["extra_args"][-2:] == ["--precision", "float32"]
        and workers["chexagent2"]["model_revision"] == "9f7225fc382ddd1297ade1aa796da660237940bc",
        "unchanged_deployed_srrg_revision_and_precision_required")
    for name, spec in workers.items():
        require(spec["status"] == "preflighted" and spec["factory_instantiated"] is False
            and os.access(spec["python"], os.X_OK)
            and set(native["supplemental_asset_inventory"][name]) <= set(spec["asset_pins"]),
            "existing_full_asset_receipt_required")
        spec["asset_stats"] = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in spec["asset_pins"]}
        require(all(v[0] == spec["asset_pins"][p]["size_bytes"] for p, v in spec["asset_stats"].items()),
            "same_pinned_weight_sizes_required")
    check_pins(inputs["source_pins"])
    check_pins(inputs["artifact_pins"])
    sources = {**inputs["source_pins"], **{str(p.resolve()): prior.sha256_file(p) for p in (
        Path(__file__), prior.ROOT / "tests/test_rsua_medical_report_observer.py",
        prior.ROOT.parent / "docs/rsua_medical_report_observer_protocol.md")}}
    for reader in (ir, nr): reader.recheck()
    plan = {"schema_version": VERSION, "config": CONFIG, "inputs": inputs["inputs"], "workers": workers,
        "source_pins": sources, "artifact_pins": {**inputs["artifact_pins"], **ir.pins, **nr.pins},
        "cohort_path": str(prior.COHORT), "cohort_sha256": prior.COHORT_SHA,
        "cohort_labels_in_model_inputs": False, "source_pixels_opened_in_prepare": False,
        "reference_labels_parsed_in_prepare": False, "selection_changed": False,
        "new_model_calls": 0, "primary_metric_eligible": False, "clinical_acceptance": False}
    tmp, target = prior.new_atomic_run(args.output_root, args.run_id)
    prior.write_private_json(tmp / "plan.json", plan)
    prior.write_private_json(tmp / "manifest.json", {"schema_version": VERSION,
        "status": "prepared_cpu_only_gpu_not_submitted", "plan_sha256": prior.sha256_file(tmp / "plan.json"),
        "new_model_calls": 0, "clinical_acceptance": False})
    prior.commit_atomic_run(tmp, target)
    return target


def phase(args):
    require_gpu_slurm(); prior.observer.gpu_guard()
    require(args.allow_public_reference_eval is True, "separate_public_image_job_approval_required")
    plan = load_plan(args)
    root = prior.require_inside(args.run_directory, BASE, must_exist=True)
    start = prior.Reader(root).json(root / "start_manifest.json")
    require(start == {"schema_version": VERSION, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "run_directory": str(root)},
        "new_job_owned_phase_directory_required")
    import torch
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 24 * 1024**3,
        "registered_srrg_memory_class_required")
    torch.set_num_threads(2)
    if args.command == "reports":
        from PIL import Image, ImageFile
        require(ImageFile.LOAD_TRUNCATED_IMAGES is False, "strict_image_decoder_required")
        sys.path.insert(0, str(prior.ROOT.parent / "CheXagent-2/src"))
        from tricompose_chexagent2_report.model import FrozenCheXagent2Runtime
        spec = plan["workers"]["chexagent2"]; loaded = {}; load_attempts = 0
        private_directory(root / "reports")
        with prior.ProtectedJournal(root / "reports.journal.jsonl") as journal:
            def inspect(item):
                path = prior.require_inside(item["path"], prior.PNG_SOURCE, must_exist=True)
                require([path.stat().st_size, path.stat().st_mtime_ns] == item["file_stats"], "original_png_stats_required")
                receipt, image = prior.guard.inspect_png(path, item["png_sha256"], Image=Image)
                del image
                return receipt
            def generate(item):
                nonlocal load_attempts
                if not loaded:
                    require(load_attempts == 0, "one_report_load_no_retry")
                    journal.append({"event": "report_load_reserved"}); load_attempts += 1
                    loaded["runtime"] = FrozenCheXagent2Runtime(model_dir=spec["model_dir"],
                        vision_dir=spec["extra_args"][1], precision="float32")
                require(prior.sha256_file(item["path"]) == item["png_sha256"], "checked_image_bytes_before_official_decoder")
                journal.append({"event": "report_generate_started", "png_sha256": item["png_sha256"]})
                result = tracked_official_generate(loaded["runtime"], Path(item["path"]))
                require(prior.sha256_file(item["path"]) == item["png_sha256"], "unchanged_image_bytes_after_official_decoder")
                return result
            def save(item, text):
                path = prior.write_private_text(root / "reports" / (item["case_id"] + ".txt"), text)
                return {"report_path": str(path.relative_to(root)), "report_sha256": prior.sha256_file(path)}
            records = generate_records(plan["inputs"], inspect=inspect, generate=generate, save=save, journal=journal.append)
        prior.write_private_json(root / "report_records.json", {"records": records, "load_attempts": load_attempts,
            "peak_torch_allocated_vram_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3)})
    else:
        sys.path.insert(0, str(prior.ROOT.parent / "cxrmate"))
        from tools.chexbert import CheXbert
        from extract_report_labels_chexbert import CHEXBERT_ORDER, CHEXBERT_CLASS_TO_STATE
        rr = prior.Reader(root); reports = rr.json(root / "report_records.json")["records"]
        model = {}; load_attempts = 0; texts = {}
        with prior.ProtectedJournal(root / "labels.journal.jsonl") as journal:
            def load():
                nonlocal load_attempts
                load_attempts += 1
                checkpoint = Path(plan["workers"]["chexbert"]["checkpoint"])
                model["runtime"] = CheXbert(ckpt_dir=str(checkpoint.parent), bert_path=str(CHEXBERT_BERT),
                    checkpoint_path=checkpoint.name, device=torch.device("cuda:0")).to("cuda:0").eval().requires_grad_(False)
                require(not model["runtime"].training and not any(p.requires_grad for p in model["runtime"].parameters()),
                    "frozen_labeler_required")
                torch.cuda.reset_peak_memory_stats()
            def length(report):
                path = prior.require_inside(root / report["report_path"], root, must_exist=True)
                rr.hash(path, report["report_sha256"])
                require(path.stat().st_size <= 65536, "bounded_generated_report_required")
                text = path.read_text()
                texts[report["case_id"]] = text
                runtime = model["runtime"]
                ids = runtime.tokenizer(normalized_chexbert_input(text), truncation=False, add_special_tokens=True)["input_ids"]
                return len(ids), runtime.bert.config.max_position_embeddings
            def infer(report):
                journal.append({"event": "label_forward_started", "report_sha256": report["report_sha256"]})
                with torch.inference_mode():
                    values = model["runtime"]([texts.pop(report["case_id"])]).detach().cpu().tolist()
                require(len(values) == 1 and len(values[0]) == 14 and all(type(v) is int and v in CHEXBERT_CLASS_TO_STATE
                    for v in values[0]), "closed_categorical_label_output_required")
                return {k: CHEXBERT_CLASS_TO_STATE[v] for k, v in zip(CHEXBERT_ORDER, values[0], strict=True)}
            records, stopped = label_records(reports, load=load, length=length, infer=infer, journal=journal.append)
        rr.recheck()
        predictions = prior.write_private_json(root / "predictions.json", {"records": records,
            "label_load_attempts": load_attempts, "label_runtime_stop": stopped,
            "peak_torch_allocated_vram_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3),
            "frozen": True, "labels_or_old_scores_supplied_to_models": False})
        with predictions.open("rb") as handle: os.fsync(handle.fileno())
    gc.collect()
    return root


def run(args):
    require_gpu_slurm(); prior.observer.gpu_guard()
    require(args.allow_public_reference_eval is True, "explicit_public_reference_job_approval_required")
    plan = load_plan(args, weights=True)
    tmp, target = prior.new_atomic_run(args.output_root, args.run_id)
    started = time.monotonic()
    try:
        prior.write_private_json(tmp / "start_manifest.json", {"schema_version": VERSION, "status": "in_progress",
            "plan_manifest_sha256": args.plan_manifest_sha256, "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "run_directory": str(tmp)})
        for command, model, timeout in (("reports", "chexagent2", 420), ("labels", "chexbert", 60)):
            runtime = tmp / (command + "_runtime"); private_directory(runtime)
            run_private_process([plan["workers"][model]["python"], str(Path(__file__).resolve()), command,
                "--plan-run", str(args.plan_run), "--plan-manifest-sha256", args.plan_manifest_sha256,
                "--run-directory", str(tmp), "--allow-public-reference-eval"], runtime, timeout)
        reader = prior.Reader(tmp)
        predictions = reader.json(tmp / "predictions.json"); records = predictions["records"]
        prediction_sha = reader.hash(tmp / "predictions.json")
        reports = reader.json(tmp / "report_records.json")
        rjournal = reader.journal(tmp / "reports.journal.jsonl")
        ljournal = reader.journal(tmp / "labels.journal.jsonl")
        # No reference-label parse occurs in either inference process.
        cr = prior.Reader(prior.COHORT.parent); cr.hash(prior.COHORT, prior.COHORT_SHA)
        cohort = cr.json(prior.COHORT)
        refs = {r["case_id"]: r["reference_state"] for r in cohort["records"]}
        require(len(cohort["records"]) == 50 and Counter(refs.values()) == {"positive": 25, "negative": 25},
            "fixed_proxy_reference_groups_required")
        xr, xm = prior.manifest(prior.XRV_SOURCE, prior.XRV_SOURCE_SHA)
        xrv_rows = prior.artifact(xr, xm, "scores.json")["records"]
        summary = {"schema_version": VERSION, "status": "completed_medical_pipeline_proxy_diagnostic",
            "case_count": 50, "pneumonia_proxy_metrics": prior.summarize(records, refs),
            "xrv_same_image_diagnostics": prior.compare_xrv(records, refs, xrv_rows, plan["inputs"]),
            "report_status_counts": dict(Counter(r["report_status"] for r in records)),
            "label_status_counts": dict(Counter(r["label_status"] for r in records)),
            "report_request_attempts": sum(e["event"] == "report_request_reserved" for e in rjournal),
            "actual_report_generate_attempts": sum(e["event"] == "report_generate_started" for e in rjournal),
            "report_load_attempts": reports["load_attempts"],
            "label_request_attempts": sum(e["event"] == "label_request_reserved" for e in ljournal),
            "actual_label_forward_attempts": sum(e["event"] == "label_forward_started" for e in ljournal),
            "label_load_attempts": predictions["label_load_attempts"], "model_retries": 0,
            "report_peak_torch_allocated_vram_gib": reports["peak_torch_allocated_vram_gib"],
            "label_peak_torch_allocated_vram_gib": predictions["peak_torch_allocated_vram_gib"],
            "runtime_seconds": round(time.monotonic()-started, 3), "frozen": True,
            "reference_labels_read_after_predictions_sealed": True,
            "reference_reports_or_mimic_inputs_opened": False, "clinical_acceptance": False,
            "selection_changed": False, "training_performed": False, "new_cxr_calls": 0,
            "new_planner_calls": 0, "external_api_calls": 0, "measured_saved_model_calls": None,
            "not_independent_of_candidate_chexbert_or_chexagent_vision": True}
        prior.write_private_json(tmp / "summary.json", summary)
        prior.write_private_text(tmp / "RESULTS_CN_EN.md", "# Official SRRG + CheXbert proxy diagnostic\n\n"
            "Published cohort proxy, not independently adjudicated clinical truth.\n"
            "Unknown/uncertain/unavailable are not negative. Existing winners and gates unchanged.\n\n"
            "```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
        load_plan(args)
        for item in plan["inputs"]: require(prior.sha256_file(item["path"]) == item["png_sha256"], "source_png_unchanged")
        reader.recheck(); cr.recheck(); xr.recheck()
        artifacts = {str(p.relative_to(tmp)): {"sha256": prior.sha256_file(p)}
            for p in sorted(tmp.rglob("*")) if p.is_file() and "_runtime" not in str(p.relative_to(tmp))}
        prior.write_private_json(tmp / "manifest.json", {"schema_version": VERSION, "status": summary["status"],
            "plan_manifest_sha256": args.plan_manifest_sha256, "prediction_sha256_before_reference_eval": prediction_sha,
            "artifacts": artifacts, "clinical_acceptance": False, "selection_changed": False})
    except BaseException:
        prior.write_private_json(tmp / "failure.json", {"schema_version": VERSION,
            "status": "failed_closed_charged_journals_retained", "automatic_retry": False, "clinical_acceptance": False})
        prior.commit_atomic_run(tmp, target)
        raise
    prior.commit_atomic_run(tmp, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "reports", "labels"):
        command = commands.add_parser(name)
        if name != "prepare":
            command.add_argument("--plan-run", type=Path, required=True)
            command.add_argument("--plan-manifest-sha256", required=True)
            command.add_argument("--allow-public-reference-eval", action="store_true")
        if name in ("reports", "labels"):
            command.add_argument("--run-directory", type=Path, required=True)
        else:
            command.add_argument("--output-root", type=Path, required=True)
            command.add_argument("--run-id", required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            target = prepare(args) if args.command == "prepare" else run(args) if args.command == "run" else phase(args)
        print(json.dumps({"stage": VERSION, "status": "prepared" if args.command == "prepare" else "completed",
            "manifest_sha256": prior.sha256_file(target / "manifest.json") if args.command in ("prepare", "run") else None}, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"stage": VERSION, "status": "failed_closed"}, sort_keys=True)); return 1


if __name__ == "__main__": raise SystemExit(main())
