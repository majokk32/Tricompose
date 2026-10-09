#!/usr/bin/env python3
"""Approved GPU Slurm only: two native beam calls, fresh labels, sealed choice.

Sequential private children release GPU memory between models. This is a batch
n-best engineering control, NOT the adaptive single-case execution ledger.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src", "TriCompose-v1.2/benchmarks", "TriCompose-v1.1/src",
                 "TriCompose-v1.0/src", "src", "TriCompose-v1.0/eval/report_v1_1",
                 "experiments/cxrmate_single_report/src"):
    sys.path.insert(0, str(ROOT.parent/relative))
from contracts import (WORKSPACE, PROTECTED_ROOT, RUN_ID_PATTERN, require_inside, read_json, sha256_file,
    load_cxr_candidates, load_report_candidates, read_report_text, private_directory, write_private_json, write_private_text)
from tricompose_v12.report_nbest import SCHEMA, POLICY, validate_policy, canonical_report, freeze, measure
from tricompose_v12.live_workers import require_gpu_slurm, check_pins, run_private_process, argv_for, CHEXBERT_BERT
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.invariant_verification import _digest
from run_fixed_image_reports import normalize_owned_modes
from run_automatic_proxy_replay import render_csv
from compare_frozen_report_paths import flatten
from score_automatic_replay_biovil import REQUEST_SCHEMA, PAIR_FIELDS


def load(args):
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root/"manifest.json") != args.plan_manifest_sha256:
        raise ValueError("approved n-best plan manifest changed")
    manifest = read_json(root/"manifest.json")
    if manifest.get("schema_version") != SCHEMA or sha256_file(root/"plan.json") != manifest["plan_sha256"]:
        raise ValueError("approved n-best plan changed")
    plan = read_json(root/"plan.json"); validate_policy(plan["policy"])
    if (plan.get("schema_version") != SCHEMA or plan.get("factory_instantiated") is not False
            or plan.get("new_model_calls") != 0 or plan.get("source_bodies_parsed") is not False
            or plan.get("new_ehr_or_cxr_samples") != 0 or len(plan["fixed_cases"]) != 2
            or [c["opaque_source_index"] for c in plan["fixed_cases"]] != [0, 1]
            or plan["generator"].get("model_id") != "cxrmate_single"
            or plan["generator"].get("status") != "preflighted"):
        raise ValueError("unsupported frozen two-case n-best plan")
    check_pins(plan["source_pins"]); check_pins(plan["artifact_pins"]); check_pins(plan["dependency_pins"])
    for c in plan["fixed_cases"]:
        anchor = anchor_from_record(c["anchor"])
        if (anchor.sha256 != c["ehr_anchor_sha256"] or c["image"]["case_id"] != anchor.case_id
                or c["image"]["ehr_sha256"] != anchor.ehr_sha256
                or c["image"]["ehr_facts_sha256"] != anchor.ehr_facts_sha256
                or read_json(c["original_candidate_path"]) != c["image"]):
            raise ValueError("immutable original image/EHR anchor differs")
    return plan


def native_kwargs(runtime):
    """Official native settings; only return count is extended from one to three."""
    return {"bos_token_id": runtime.tokenizer.bos_token_id, "eos_token_id": runtime.tokenizer.eos_token_id,
        "pad_token_id": runtime.tokenizer.pad_token_id, "special_token_ids": [runtime.tokenizer.sep_token_id],
        "return_dict_in_generate": True, "use_cache": True, "num_beams": POLICY["num_beams"],
        "max_length": POLICY["max_length"], "do_sample": False, "num_return_sequences": POLICY["num_return_sequences"]}


def generate(args):
    require_gpu_slurm()
    plan = load(args); spec = plan["generator"]
    check_pins(spec["asset_pins"])
    cxrs = load_cxr_candidates([plan["cxr_run"]])
    if len(cxrs) != 2: raise ValueError("exact two fixed images required")
    root = require_inside(args.output_run, PROTECTED_ROOT, must_exist=False)
    private_directory(root); private_directory(root/"candidates")
    # Import/model initialization exclusively in this approved GPU child.
    from tricompose_cxrmate_single_report.model import FrozenCXRMateSingleRuntime
    from evaluate_report_structure import _record
    from PIL import Image
    began = time.monotonic(); runtime = FrozenCXRMateSingleRuntime(spec["model_dir"])
    if runtime.audit != spec["model_audit"]: raise ValueError("native model audit changed")
    load_seconds = time.monotonic()-began
    descriptor = os.open(root/"native_call_journal.jsonl", os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o660)
    os.fchmod(descriptor, 0o660)
    candidates, calls = [], []
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        def event(value):
            handle.write(json.dumps(value, sort_keys=True)+"\n"); handle.flush(); os.fsync(handle.fileno())
        for source in plan["fixed_cases"]:
            image = cxrs[source["image"]["candidate_id"]]
            if image != source["image"]: raise ValueError("historical image metadata changed")
            image_path = require_inside(image["artifact"]["path"], PROTECTED_ROOT, must_exist=True)
            if sha256_file(image_path) != image["artifact"]["sha256"]: raise ValueError("fixed image changed")
            group = "beamcall_" + _digest(image["candidate_id"])[:24]
            runtime._assert_frozen()
            with Image.open(image_path) as source_handle: pixels = source_handle.convert("RGB").copy()
            tensor = runtime.transform(pixels).unsqueeze(0).to("cuda")
            event({"invocation_id": group, "status": "reserved_before_generate", "native_generate_invocations": 1,
                   "maximum_returned_sequences": 3, "num_beams": 4})
            runtime._torch.cuda.synchronize(); started = time.monotonic()
            with runtime._torch.inference_mode():
                output = runtime.model.generate(pixel_values=tensor, **native_kwargs(runtime))
            runtime._torch.cuda.synchronize(); elapsed = time.monotonic()-started
            runtime._assert_frozen()
            if output.sequences.shape[0] != 3: raise ValueError("native interface did not return three sequences")
            findings, impressions = runtime.model.split_and_decode_sections(output.sequences,
                [runtime.tokenizer.sep_token_id, runtime.tokenizer.eos_token_id], runtime.tokenizer)
            if len(findings) != 3 or len(impressions) != 3: raise ValueError("native section inventory differs")
            hashes, normalized = [], []
            for rank in range(3):
                cid = "nbestreport_" + _digest([image["candidate_id"], rank])[:32]
                directory = root/"candidates"/cid; private_directory(directory)
                text = canonical_report(findings[rank], impressions[rank])
                path = write_private_text(directory/"synthetic_report.txt", text)
                report = {"schema_version": "tricompose-report-candidate-v1.1", "modality": "report",
                    "candidate_id": cid, "case_id": image["case_id"], "model_id": "cxrmate_single",
                    "model_revision": spec["model_revision"], "frozen_model": True,
                    "model_input_signature": "single_current_synthetic_cxr", "structured_ehr_content_supplied_to_model": False,
                    "source_report_or_real_target_supplied": False, "classifier_labels_supplied_to_generator": False,
                    "parent_cxr_candidate_id": image["candidate_id"], "input_cxr": dict(image["artifact"]),
                    "input_cxr_candidate_sha256": source["original_candidate_sha256"],
                    "ehr_sha256_retained_for_lineage": image["ehr_sha256"],
                    "ehr_facts_sha256_retained_for_lineage": image["ehr_facts_sha256"],
                    "artifact": {"path": str(path), "sha256": sha256_file(path), "mime_type": "text/plain"},
                    "beam_rank": rank, "native_invocation_id": group,
                    "decoding": {"num_beams": 4, "num_return_sequences": 3, "max_length": 256, "do_sample": False, "dtype": "float32"},
                    "cost": {"model_calls": 1 if rank == 0 else 0, "shared_invocation_id": group,
                        "accounting": "charge_shared_generate_only_to_rank_zero_not_independent_calls",
                        "shared_invocation_runtime_seconds": elapsed},
                    "clinical_acceptance": False}
                meta = write_private_json(directory/"candidate.json", report)
                structure = _record(report, image, normalized_frequency={})
                write_private_json(directory/"structure.json", structure)
                candidates.append({"candidate_id": cid, "case_id": image["case_id"], "parent_cxr_candidate_id": image["candidate_id"],
                    "path": str(meta.relative_to(root)), "sha256": sha256_file(meta)})
                hashes.append(report["artifact"]["sha256"]); normalized.append(structure["normalized_report_sha256"])
            calls.append({"native_invocation_id": group, "native_generate_invocations": 1, "returned_sequences": 3,
                "runtime_seconds_generate_only": elapsed, "unique_exact_reports": len(set(hashes)),
                "unique_normalized_reports": len(set(normalized)),
                "historical_top1_sha256_equal": hashes[0] == source["historical_cxrmate_report_sha256"]})
            event({"invocation_id": group, "status": "validated", "returned_sequences": 3,
                   "runtime_seconds_generate_only": elapsed})
            if sha256_file(image_path) != image["artifact"]["sha256"]: raise ValueError("fixed image changed during inference")
        check_pins(spec["asset_pins"]); check_pins(plan["dependency_pins"])
        write_private_json(root/"manifest.json", {"schema_version": "tricompose-report-candidate-run-v1.1",
            "model_id": "cxrmate_single", "model_revision": spec["model_revision"], "frozen_model": True,
            "model_audit": runtime.audit, "candidate_count": 6, "candidates": candidates,
            "plan_manifest_sha256": args.plan_manifest_sha256, "native_calls": calls,
            "native_generate_invocations": 2, "returned_sequences": 6,
            "model_load_seconds": load_seconds, "peak_vram_gib": runtime.peak_vram_gib,
            "same_beam_candidates_are_independent_votes": False,
            "native_call_journal_sha256": sha256_file(root/"native_call_journal.jsonl")})


def single_view(bundle, name, id_field, candidate_id):
    """A per-record validation view; original full-bundle SHA remains the receipt.

The legacy producer's model_calls field counts scored samples, not batches.
This view neither invokes a model nor charges/claims an extra call.
"""
    records = bundle["records"]
    if (bundle["counts"] != {name: len(records), "model_calls": len(records)}
            or len({r[id_field] for r in records}) != len(records)):
        raise ValueError("unique authenticated scorer sample inventory required")
    matches = [r for r in records if r[id_field] == candidate_id]
    if len(matches) != 1: raise ValueError("exact scorer sample required")
    return {**bundle, "records": matches, "counts": {name: 1, "model_calls": 1}}


def run(args):
    require_gpu_slurm(); plan = load(args)
    if not RUN_ID_PATTERN.fullmatch(args.run_id): raise ValueError("opaque run ID required")
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False); private_directory(output, exist_ok=True)
    root = require_inside(output/args.run_id, PROTECTED_ROOT, must_exist=False); private_directory(root)
    began = time.monotonic(); costs = []
    write_private_json(root/"start_manifest.json", {"schema_version": SCHEMA, "status": "in_progress",
        "plan_manifest_sha256": args.plan_manifest_sha256, "clinical_acceptance": False})
    descriptor = os.open(root/"cost_journal.jsonl", os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o660); os.fchmod(descriptor, 0o660)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        def event(value):
            handle.write(json.dumps(value, sort_keys=True)+"\n"); handle.flush(); os.fsync(handle.fileno())
        def child(stage, argv, units, timeout):
            event({"stage": stage, "status": "reserved_before_spawn", "maximum_sample_units": units,
                   "argv_sha256": _digest(argv), "timeout_seconds": timeout})
            runtime = root/(stage+"_runtime"); private_directory(runtime)
            started = time.monotonic(); run_private_process(argv, runtime, timeout)
            costs.append({"stage": stage, "wall_seconds_including_startup_io": time.monotonic()-started,
                          "maximum_sample_units": units})
            event({"stage": stage, "status": "process_completed_unvalidated"})
        try:
            for spec in (plan["generator"], plan["xrv"], plan["chexbert"]): check_pins(spec["asset_pins"])
            check_pins(plan["biovil_asset_pins"])
            reports_root = root/"reports"
            child("generate", [plan["generator"]["python"], str(Path(__file__)), "generate",
                "--plan-run", args.plan_run, "--plan-manifest-sha256", args.plan_manifest_sha256,
                "--output-run", str(reports_root)], {"native_generate_invocations": 2, "maximum_returned_sequences": 6}, 600)
            cxrs = load_cxr_candidates([plan["cxr_run"]]); reports = load_report_candidates([reports_root], cxr_candidates=cxrs)
            gm = read_json(reports_root/"manifest.json")
            if (len(reports) != 6 or gm["native_generate_invocations"] != 2 or gm["returned_sequences"] != 6
                    or gm["model_audit"] != plan["generator"]["model_audit"]
                    or gm["plan_manifest_sha256"] != args.plan_manifest_sha256
                    or sum(r["cost"]["model_calls"] for r in reports.values()) != 2):
                raise ValueError("native beam inventory/call accounting differs")
            event({"stage": "generate", "status": "validated", "native_generate_invocations": 2, "returned_sequences": 6})
            private_directory(root/"xrv")
            child("xrv", argv_for(plan["xrv"], cxr_run=plan["cxr_run"], output_root=root/"xrv",
                thresholds=plan["xrv"]["thresholds_path"]), {"xrv_scored_samples": 2}, 180)
            private_directory(root/"chexbert")
            cb = plan["chexbert"]
            child("chexbert", [cb["python"], cb["script"], "--cxr-run", plan["cxr_run"], "--report-run", str(reports_root),
                "--checkpoint", cb["checkpoint"], "--bert-path", str(CHEXBERT_BERT), "--batch-size", "6",
                "--output-root", str(root/"chexbert"), "--run-id", "scored"], {"chexbert_scored_samples": 6}, 180)
            ip = root/"xrv/scored/cxr_finding_labels.json"; tp = root/"chexbert/scored/report_finding_labels.json"
            il, tl = read_json(ip), read_json(tp)
            if il["counts"]["cxr_candidates"] != 2 or tl["counts"]["reports"] != 6:
                raise ValueError("fresh scorer inventory differs")
            rows = []
            for source in plan["fixed_cases"]:
                anchor = anchor_from_record(source["anchor"]); image = cxrs[source["image"]["candidate_id"]]
                image_labels = single_view(il, "cxr_candidates", "cxr_candidate_id", image["candidate_id"])
                partial = image_receipt(anchor, image, image_labels, label_sha256=sha256_file(ip),
                    thresholds_sha256=plan["xrv"]["thresholds_sha256"], checkpoint_sha256=plan["xrv"]["checkpoint_sha256"])
                for report in sorted(reports.values(), key=lambda r: r["candidate_id"]):
                    if report["parent_cxr_candidate_id"] != image["candidate_id"]: continue
                    receipt = completed_receipt(anchor, partial, image, image_labels, report,
                        single_view(tl, "reports", "report_candidate_id", report["candidate_id"]),
                        image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp),
                        thresholds_sha256=plan["xrv"]["thresholds_sha256"], xrv_checkpoint_sha256=plan["xrv"]["checkpoint_sha256"],
                        chexbert_checkpoint_sha256=cb["checkpoint_sha256"])
                    directory = Path(report["artifact"]["path"]).parent
                    structure = read_json(directory/"structure.json")
                    if (structure["report_sha256"] != receipt["report_sha256"]
                            or structure["report_candidate_id"] != report["candidate_id"]
                            or structure["image_sha256"] != receipt["cxr_sha256"]
                            or structure["case_id"] != anchor.case_id): raise ValueError("structure lineage differs")
                    rows.append({"triple_candidate_id": "nbestpair_" + _digest([image["candidate_id"], report["candidate_id"]])[:32],
                        "case_id": anchor.case_id, "cxr_candidate_id": image["candidate_id"], "report_candidate_id": report["candidate_id"],
                        "ehr_sha256": anchor.ehr_sha256, "ehr_facts_sha256": anchor.ehr_facts_sha256,
                        "cxr_sha256": image["artifact"]["sha256"], "report_sha256": report["artifact"]["sha256"],
                        "beam_rank": report["beam_rank"], "receipt": receipt, "structure": structure,
                        "raw_edge_readouts": receipt["raw_edge_readouts"]})
            rows.sort(key=lambda r: r["triple_candidate_id"]); selection = freeze(rows)
            write_private_json(root/"score_rows.json", {"records": rows})
            selected = write_private_json(root/"selection.json", selection); selected_sha = sha256_file(selected)
            event({"stage": "xrv_chexbert", "status": "validated", "scored_image_samples": 2, "scored_report_samples": 6})
            request_dir = root/"secondary_request"; private_directory(request_dir)
            request = {"schema_version": REQUEST_SCHEMA, "pairs": [{k: r[k] for k in PAIR_FIELDS} for r in rows],
                "experiment": SCHEMA, "selection_sha256": selected_sha, "modality_source": "fully_synthetic",
                "selection_used_biovil": False, "routing_or_calibration_update_allowed": False,
                "clinical_truth_available": False, "text_policy": "full_report_no_silent_truncation_overlength_is_na"}
            rp = write_private_json(request_dir/"request.json", request)
            write_private_json(request_dir/"manifest.json", {"schema_version": REQUEST_SCHEMA, "artifacts": {"request.json": {"sha256": sha256_file(rp)}}})
            child("biovil", [plan["biovil_python"], str(Path(__file__)), "biovil-worker", "--request-run", str(request_dir),
                "--cxr-run", plan["cxr_run"], "--report-run", str(reports_root), "--model-path", plan["biovil_model"],
                "--output-file", str(root/"secondary.json")], {"maximum_image_encodings": 2, "maximum_text_encodings": 6}, 180)
            endpoint = read_json(root/"secondary.json")
            if sha256_file(selected) != selected_sha or endpoint["request_sha256"] != sha256_file(rp):
                raise ValueError("selection/secondary request changed")
            comparison = measure(selection, rows, endpoint)
            write_private_json(root/"comparison.json", comparison)
            scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
            table = []
            for row in rows:
                table.append(flatten({k: v for k, v in row.items() if k not in ("receipt", "structure")} ) | {
                    "biovil_raw_cosine": scores[row["triple_candidate_id"]]["biovil_raw_cosine"],
                    "biovil_status": scores[row["triple_candidate_id"]]["status"],
                    "biovil_na_reason": scores[row["triple_candidate_id"]]["reason"],
                    "known_ehr_facts": row["receipt"]["known_ehr_facts"],
                    "unknown_image_positive_report_facts": sum(f["xrv"] not in {"positive", "negative"} and f["chexbert"] == "positive" for f in row["receipt"]["fact_states"]),
                    **{k: row["structure"][k] for k in ("section_contract_pass", "unsupported_temporal_comparison_language", "generic_report")}})
            write_private_text(root/"score_table.csv", render_csv([{k: "NA" if v is None else v for k, v in r.items()} for r in table]))
            event({"stage": "biovil", "status": "validated", "counts": endpoint["counts"]})
            check_pins(plan["source_pins"]); check_pins(plan["artifact_pins"]); check_pins(plan["dependency_pins"])
            for spec in (plan["generator"], plan["xrv"], plan["chexbert"]): check_pins(spec["asset_pins"])
            check_pins(plan["biovil_asset_pins"])
            handle.flush(); os.fsync(handle.fileno())
            files = ("score_rows.json", "selection.json", "secondary.json", "comparison.json", "score_table.csv", "cost_journal.jsonl")
            result = {"schema_version": SCHEMA, "status": "completed_engineering_nbest_control_unvalidated",
                "plan_manifest_sha256": args.plan_manifest_sha256, "selection_sha256_before_endpoint": selected_sha,
                "artifacts": {n: {"sha256": sha256_file(root/n)} for n in files},
                "generator_manifest_sha256": sha256_file(reports_root/"manifest.json"),
                "xrv_labels_sha256": sha256_file(ip), "chexbert_labels_sha256": sha256_file(tp),
                "native_generate_invocations": 2, "returned_sequences": 6, "new_ehr_or_cxr_samples": 0,
                "native_calls": gm["native_calls"], "model_load_seconds": gm["model_load_seconds"],
                "peak_vram_gib": {"generator": gm["peak_vram_gib"], "xrv": il["peak_vram_gib"], "chexbert": tl["peak_vram_gib"], "biovil": endpoint["peak_vram_gib"]},
                "legacy_scorer_model_calls_are_sample_counts_not_batched_invocations": True,
                "new_xrv_scored_samples": 2, "new_chexbert_scored_samples": 6, "secondary_counts": endpoint["counts"],
                "batch_cost_accounting_not_adaptive_single_case_ledger": True, "worker_costs": costs,
                "wall_seconds_including_startup_io": round(time.monotonic()-began, 3), "gpu_seconds": None,
                "clinical_acceptance": False, "adaptive_repair_executed": False, "same_beam_candidates_are_independent_votes": False,
                "biovil_used_for_selection": False, "original_ehr_cxr_or_winners_changed": False}
            write_private_json(root/"manifest.json", result)
        except BaseException as exc:
            event({"status": "failed_or_interrupted_retained", "error_type": type(exc).__name__, "automatic_resume": False})
            write_private_json(root/"failure_manifest.json", {"status": "failed_or_interrupted_retained",
                "error_type": type(exc).__name__, "automatic_resume": False, "clinical_acceptance": False})
            raise
        finally:
            normalize_owned_modes(root)
    return root, result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="mode", required=True)
    launch = sub.add_parser("run"); native = sub.add_parser("generate")
    for command in (launch, native):
        command.add_argument("--plan-run", required=True); command.add_argument("--plan-manifest-sha256", required=True)
    launch.add_argument("--output-root", required=True); launch.add_argument("--run-id", required=True)
    native.add_argument("--output-run", required=True)
    worker = sub.add_parser("biovil-worker")
    for name in ("request-run", "model-path", "output-file"): worker.add_argument("--"+name, required=True)
    for name in ("cxr-run", "report-run"): worker.add_argument("--"+name, action="append", required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        if args.mode == "biovil-worker":
            require_gpu_slurm(); sys.path.insert(0, str(WORKSPACE/"runtime/vendor/hi_ml_multimodal_0_2_2"))
            from score_automatic_replay_biovil import score
            write_private_json(require_inside(args.output_file, PROTECTED_ROOT, must_exist=False), score(args)); return 0
        if args.mode == "generate": generate(args); return 0
        root, result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__})); return 1
    print(json.dumps({"status": result["status"], "runtime_seconds": result["wall_seconds_including_startup_io"],
        "manifest_sha256": sha256_file(root/"manifest.json")})); return 0


if __name__ == "__main__": raise SystemExit(main())
