#!/usr/bin/env python3
"""Frozen image-only Qwen observer on all four existing probe/reference CXRs.

No EHR/report/model identity/score is sent to the model. Predictions are sealed
before source label comparison. This is a secondary observer, not independent
clinical truth, a new primary scorer or authority to change the winners.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

import run_fresh_cxr_secondary as source
import verify_cached_image_findings as existing
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
    require_inside, sha256_file, write_private_json, write_private_text)
from tricompose_v12.runtime_dispatch import ProtectedJournal
from tricompose_llm.local_qwen import gpu_guard

VERSION = "tricompose-fresh-probe-blind-image-observer-v1"
CONFIG = {"min_pixels": 200704, "max_pixels": 401408, "max_new_tokens":384,
    "model_retries":0, "seed":0, "do_sample":False, "min_vram_gib":24}
require = source.require


def inventory(rows, triples):
    require(len(rows) == len(triples) == 10 and len({r["case_id"] for r in rows}) == 2,
        "all_ten_audited_probe_and_reference_pairs_required")
    images = {}
    for row in rows:
        matches = [t for t in triples if all(t[k] == row[k] for k in
            ("case_id","ehr_sha256","ehr_facts_sha256","cxr_sha256","report_sha256","cxr_model_id","report_model_id","seed"))]
        require(len(matches) == 1, "one_existing_triplet_lineage_required")
        t = matches[0]
        image = {"cxr_candidate_id":row["cxr_candidate_id"], "cxr_sha256":row["cxr_sha256"], "path":t["cxr_path"]}
        require(images.setdefault(image["cxr_candidate_id"], image) == image, "shared_image_changed")
    require(len(images) == len({i["cxr_sha256"] for i in images.values()}) == 4
        and all({r["seed"] for r in rows if r["case_id"] == cid} == {0,1} for cid in {r["case_id"] for r in rows}),
        "both_seeds_on_both_fixed_cases_no_cherry_picking")
    return sorted(images.values(), key=lambda i:(i["cxr_sha256"],i["cxr_candidate_id"]))


def comparisons(records, rows):
    expected = {r["cxr_candidate_id"] for r in rows}
    index = {r["cxr_candidate_id"]:r for r in records}
    require(len(index) == len(records) == 4 and set(index) == expected, "all_four_image_responses_required")
    output = []
    for iid, record in index.items():
        group = [r for r in rows if r["cxr_candidate_id"] == iid]
        row = group[0]
        require(record["cxr_sha256"] == row["cxr_sha256"], "image_hash_changed")
        known = [(f["finding"],f["ehr"],f["xrv"]) for f in row["receipt"]["fact_states"]]
        require(all([(f["finding"],f["ehr"],f["xrv"]) for f in r["receipt"]["fact_states"]] == known for r in group),
            "same_image_source_label_states_changed")
        complete = record["contract_status"] == "complete"
        require((complete and isinstance(record["states"],dict)
                and set(record["states"]) == set(existing.image_interface.FINDINGS)
                and set(record["states"].values()) <= existing.STATES)
            or (record["contract_status"] == "failed_unavailable" and record["states"] is None),
            "full_eight_named_states_or_explicit_unavailable_required")
        facts = {f["finding"]:f for f in row["receipt"]["fact_states"]}
        for name in existing.image_interface.FINDINGS:
            qwen = record["states"][name] if complete else None
            fact = facts[name]
            output.append({"case_id":row["case_id"], "cxr_candidate_id":iid, "seed":row["seed"],
                "finding":name, "ehr_weak_state":fact["ehr"], "xrv_proxy_state":fact["xrv"], "qwen_image_state":qwen,
                "xrv_qwen_relation":existing.relation(fact["xrv"],qwen) if complete else "observer_unavailable",
                "ehr_qwen_relation":existing.relation(fact["ehr"],qwen) if complete else "observer_unavailable",
                "clinical_fault_location":None, "clinical_acceptance":False})
    return output


def prepare(args):
    source.previous.gate.cpu_guard()
    ar, sr, pr = [source.previous.postflight.MetadataReader(p) for p in (source.AUDIT,source.SOURCE,source.GENERATION_PLAN)]
    ar.hash(ar.root/"manifest.json",source.AUDIT_SHA)
    am=ar.json(ar.root/"manifest.json"); ar.hash(ar.root/"audit.json",am["artifacts"]["audit.json"]["sha256"])
    audit=ar.json(ar.root/"audit.json")
    require(audit["status"] == "metadata_and_ledger_verified_not_clinical" and audit["source_manifest_sha256"] == source.SOURCE_SHA,
        "exact_completed_probe_audit_required")
    sr.hash(sr.root/"manifest.json",source.SOURCE_SHA)
    sm=sr.json(sr.root/"manifest.json")
    for name,pin in sm["artifacts"].items(): sr.hash(sr.root/name,pin["sha256"])
    pr.hash(pr.root/"manifest.json",source.GENERATION_PLAN_SHA)
    pm=pr.json(pr.root/"manifest.json"); pr.hash(pr.root/"plan.json",pm["plan_sha256"])
    generation=pr.json(pr.root/"plan.json")
    rows=sr.json(sr.root/"score_rows.json")["records"]
    t=sr.json(sr.root/"completed_triplets.json")
    images=inventory(rows,t["cached_references"]+t["new_records"])
    for image in images: sr.hash(image["path"],image["cxr_sha256"]) if Path(image["path"]).is_relative_to(sr.root) else source.check_pins({image["path"]:image["cxr_sha256"]})
    prompt=existing.image_interface.IMAGE_PROMPT.format(findings=", ".join(existing.image_interface.FINDINGS))
    sources={**audit["source_pins"],str(Path(__file__).resolve()):sha256_file(__file__),
        str(source.ROOT/"TriCompose-v1.2/tests/test_fresh_probe_image_observer.py"):
            sha256_file(source.ROOT/"TriCompose-v1.2/tests/test_fresh_probe_image_observer.py"),
        str(Path(existing.__file__).resolve()):sha256_file(existing.__file__),
        str(Path(existing.image_interface.__file__).resolve()):sha256_file(existing.image_interface.__file__)}
    pins={**audit["artifact_pins"],**ar.pins,**sr.pins,**pr.pins,
        **{i["path"]:i["cxr_sha256"] for i in images}}
    source.check_pins(sources); source.check_pins(pins); source.check_pins(generation["qwen"]["asset_pins"])
    for reader in (ar,sr,pr): reader.recheck()
    plan={"schema_version":VERSION,"config":CONFIG,"image_inputs":images,"max_model_calls":4,
        "qwen":generation["qwen"],"image_prompt_sha256":existing.image_interface.digest_text(prompt),
        "prompt_version":existing.image_interface.PROMPT_VERSION,"source_manifest_sha256":source.SOURCE_SHA,
        "comparison_rows_path":str(sr.root/"score_rows.json"),"comparison_rows_sha256":sr.hash(sr.root/"score_rows.json"),
        "source_pins":sources,"artifact_pins":pins,"new_model_calls":0,"source_bodies_parsed":False,
        "image_pixels_decoded":False,"model_received_ehr_reports_ids_or_scores":False,
        "selection_change_allowed":False,"primary_metric_eligible":False}
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    try:
        p=write_private_json(temporary/"plan.json",plan)
        write_private_json(temporary/"manifest.json",{"schema_version":VERSION,"status":"prepared_cpu_only_gpu_not_submitted",
            "plan_sha256":sha256_file(p),"max_model_calls":4,"new_model_calls":0,"clinical_acceptance":False})
        commit_atomic_run(temporary,target)
    except BaseException: discard_atomic_run(temporary); raise
    return target


def run(args):
    gpu_guard()
    root=require_inside(args.plan_run,source.PROTECTED_ROOT,must_exist=True)
    require(sha256_file(root/"manifest.json") == args.plan_manifest_sha256,"reviewed_plan_changed")
    manifest=source.read_json(root/"manifest.json")
    require(manifest["schema_version"] == VERSION and sha256_file(root/"plan.json") == manifest["plan_sha256"],"sealed_observer_plan_required")
    plan=source.read_json(root/"plan.json")
    require(plan["config"] == CONFIG and plan["max_model_calls"] == len(plan["image_inputs"]) == 4
        and plan["source_manifest_sha256"] == source.SOURCE_SHA and plan["selection_change_allowed"] is False,
        "four_blind_calls_with_original_settings_only")
    for field in ("source_pins","artifact_pins"): source.check_pins(plan[field])
    source.check_pins(plan["qwen"]["asset_pins"])
    import torch
    require(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= CONFIG["min_vram_gib"]*1024**3,
        "at_least_24gib_gpu_required")
    from PIL import Image
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    temporary,target=new_atomic_run(args.output_root,args.run_id)
    records=[]; charged=0; started=time.monotonic()
    try:
        torch.manual_seed(CONFIG["seed"]); torch.cuda.reset_peak_memory_stats()
        with ProtectedJournal(temporary/"calls.journal.jsonl") as journal, open(os.devnull,"w") as sink, \
             contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            journal.append({"event":"observer_load_attempt_reserved","charged_load_attempts":1})
            model,processor,model_type=_load_model(Path(plan["qwen"]["model_path"]),torch,
                min_pixels=CONFIG["min_pixels"],max_pixels=CONFIG["max_pixels"])
            model.eval().requires_grad_(False)
            require(not model.training and not any(p.requires_grad for p in model.parameters()),"frozen_observer_required")
            for item in plan["image_inputs"]:
                path=require_inside(item["path"],source.PROTECTED_ROOT,must_exist=True)
                require(sha256_file(path) == item["cxr_sha256"],"checked_existing_synthetic_image_required")
                with Image.open(path) as handle:
                    require(64 <= handle.width <= 4096 and 64 <= handle.height <= 4096,"bounded_image_dimensions_required")
                    image=handle.convert("RGB").copy()
                require(charged < 4,"fixed_call_budget_exceeded")
                journal.append({"event":"call_reserved","ordinal":charged,"cxr_sha256":item["cxr_sha256"]}); charged+=1
                response,tokens=existing.image_interface.infer(existing.image_interface.request_messages("image",image=image),
                    model,processor,torch,max_new_tokens=CONFIG["max_new_tokens"])
                record={"cxr_candidate_id":item["cxr_candidate_id"],"cxr_sha256":item["cxr_sha256"],
                    "response_sha256":existing.image_interface.digest_text(response),**tokens,
                    **existing.sanitized_decode(response,token_limit_reached=tokens["token_limit_reached"])}
                records.append(record); journal.append({"event":"call_finished","ordinal":charged-1,"contract_status":record["contract_status"]})
        predictions=write_private_json(temporary/"predictions.json",{"schema_version":VERSION,"records":records,
            "image_only":True,"frozen":True,"model_received_ehr_reports_ids_or_scores":False})
        with predictions.open("rb") as handle: os.fsync(handle.fileno())
        # Read the comparison table only after the model observations are sealed.
        require(sha256_file(plan["comparison_rows_path"]) == plan["comparison_rows_sha256"],"comparison_table_changed")
        rows=source.read_json(plan["comparison_rows_path"])["records"]
        table=comparisons(records,rows)
        csv=write_private_text(temporary/"image_observer_comparison.csv",source.previous.csv_text(table))
        summary=write_private_json(temporary/"summary.json",{"schema_version":VERSION,"actual_model_calls":charged,
            "complete_responses":sum(r["contract_status"] == "complete" for r in records),"unique_image_finding_slots":len(table),
            "new_generator_calls":0,"model_retries":0,"clinical_fault_location":None,"clinical_acceptance":False,
            "charged_load_attempts":1,
            "selection_changed":False,"primary_metric_eligible":False,"same_qwen_checkpoint_as_numeric_planner":True,
            "runtime_seconds":round(time.monotonic()-started,3),"peak_torch_allocated_vram_gib":round(torch.cuda.max_memory_allocated()/1024**3,3)})
        for field in ("source_pins","artifact_pins"): source.check_pins(plan[field])
        source.check_pins(plan["qwen"]["asset_pins"])
        write_private_json(temporary/"manifest.json",{"schema_version":VERSION,"status":"completed_blind_image_observer_unvalidated",
            "plan_manifest_sha256":args.plan_manifest_sha256,"actual_model_calls":charged,"clinical_acceptance":False,
            "artifacts":{f.name:{"sha256":sha256_file(f)} for f in (predictions,csv,summary)}})
        commit_atomic_run(temporary,target)
    except BaseException:
        write_private_json(temporary/"failure.json",{"schema_version":VERSION,"status":"failed_attempts_retained",
            "charged_model_calls":charged,"complete_prefix_records":records,"automatic_resume":False})
        commit_atomic_run(temporary,target); raise
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    prep,execution=sub.add_parser("prepare"),sub.add_parser("run")
    execution.add_argument("--plan-run",type=Path,required=True); execution.add_argument("--plan-manifest-sha256",required=True)
    for parser in (prep,execution):
        parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--run-id",required=True)
    args=p.parse_args(); os.umask(0o007)
    try:
        target=prepare(args) if args.command == "prepare" else run(args)
        print(json.dumps({"stage":VERSION,"status":"prepared" if args.command == "prepare" else "completed",
            "manifest_sha256":sha256_file(target/"manifest.json")},sort_keys=True)); return 0
    except Exception:
        print(json.dumps({"stage":VERSION,"status":"failed_closed"},sort_keys=True)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
