#!/usr/bin/env python3
"""Approved GPU Slurm: verify eight cached reports, seal choice, score endpoint.

No EHR/CXR/report generation. Existing outputs and winners remain immutable.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for relative in ("TriCompose-v1.2/src","TriCompose-v1.2/benchmarks","TriCompose-v1.1/src",
                 "TriCompose-v1.0/src","src","TriCompose-v1.0/eval/report_v1_1"):
    sys.path.insert(0,str(ROOT.parent/relative))
from contracts import (PROTECTED_ROOT,WORKSPACE,RUN_ID_PATTERN,require_inside,read_json,sha256_file,
    load_cxr_candidates,load_report_candidates,private_directory,write_private_json,write_private_text)
from tricompose_v12.report_expert_control import SCHEMA,POLICY,MODELS,freeze,measure
from tricompose_v12.live_workers import require_gpu_slurm,check_pins,run_private_process,CHEXBERT_BERT
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v12.live_receipts import completed_receipt
from tricompose_v12.invariant_verification import _digest
from run_report_nbest import single_view
from run_fixed_image_reports import normalize_owned_modes
from run_automatic_proxy_replay import render_csv
from compare_frozen_report_paths import flatten
from score_automatic_replay_biovil import REQUEST_SCHEMA,PAIR_FIELDS


def load(args):
    root=require_inside(args.plan_run,PROTECTED_ROOT,must_exist=True)
    if sha256_file(root/"manifest.json")!=args.plan_manifest_sha256: raise ValueError("approved expert plan changed")
    manifest=read_json(root/"manifest.json")
    if manifest.get("schema_version")!=SCHEMA or sha256_file(root/"plan.json")!=manifest["plan_sha256"]:
        raise ValueError("approved expert plan body changed")
    plan=read_json(root/"plan.json")
    if (plan.get("schema_version")!=SCHEMA or plan.get("policy")!=POLICY
            or plan.get("factory_instantiated") is not False or plan.get("source_bodies_parsed") is not False
            or plan.get("legacy_xrv_or_report_labels_used") is not False or plan.get("new_model_calls")!=0
            or plan.get("planned_new_generation_calls")!=0 or len(plan["fixed_cases"])!=2
            or [c["opaque_source_index"] for c in plan["fixed_cases"]]!=[0,1]
            or len(plan["report_runs"])!=4 or len(plan["image_receipts"])!=2):
        raise ValueError("unsupported two-image cached-expert scope")
    check_pins(plan["source_pins"]);check_pins(plan["artifact_pins"])
    return plan


def method_summary(rows,endpoint):
    scores={r["triple_candidate_id"]:r for r in endpoint["records"]};result=[]
    for model in MODELS:
        group=[r for r in rows if r["report_model_id"]==model]
        values=[scores[r["triple_candidate_id"]]["biovil_raw_cosine"] for r in group]
        edges={}
        for edge in ("ehr_cxr","ehr_report","cxr_report"):
            total={k:sum(r["raw_edge_readouts"][edge][k] for r in group) for k in (
                "known_reference_facts","comparable_facts","supported_positive","supported_negative","proxy_opposition_facts","missing_comparisons")}
            known=total["known_reference_facts"]
            total.update(coverage_over_known=total["comparable_facts"]/known if known else None,
                support_over_known=(total["supported_positive"]+total["supported_negative"])/known if known else None,
                opposition_over_known=total["proxy_opposition_facts"]/known if known else None)
            edges[edge]=total
        result.append({"report_model_id":model,"fixed_ehr_cases":2,"report_candidates":2,
            "biovil_available_pairs":sum(v is not None for v in values),
            "two_case_mean_biovil_raw_cosine":sum(values)/2 if all(v is not None for v in values) else None,
            "official_structure_pass_count":sum(r["structure"]["section_contract_pass"] for r in group),
            "unsupported_temporal_language_count":sum(r["structure"]["unsupported_temporal_comparison_language"] for r in group),
            "raw_edge_totals":edges,"clinical_accuracy":None,"clinical_acceptance":False})
    return result


def run(args):
    require_gpu_slurm();plan=load(args)
    if not RUN_ID_PATTERN.fullmatch(args.run_id): raise ValueError("opaque run ID required")
    output=require_inside(args.output_root,PROTECTED_ROOT,must_exist=False);private_directory(output,exist_ok=True)
    root=require_inside(output/args.run_id,PROTECTED_ROOT,must_exist=False);private_directory(root)
    started=time.monotonic();costs=[]
    write_private_json(root/"start_manifest.json",{"schema_version":SCHEMA,"status":"in_progress",
        "plan_manifest_sha256":args.plan_manifest_sha256,"new_generation_calls":0,"clinical_acceptance":False})
    fd=os.open(root/"cost_journal.jsonl",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o660);os.fchmod(fd,0o660)
    with os.fdopen(fd,"w",encoding="utf-8") as journal:
        def event(value):
            journal.write(json.dumps(value,sort_keys=True)+"\n");journal.flush();os.fsync(journal.fileno())
        def child(stage,argv,units,timeout):
            event({"stage":stage,"status":"reserved_before_spawn","maximum_sample_units":units,
                   "argv_sha256":_digest(argv),"timeout_seconds":timeout})
            runtime=root/(stage+"_runtime");private_directory(runtime)
            began=time.monotonic();run_private_process(argv,runtime,timeout)
            costs.append({"stage":stage,"wall_seconds_including_startup_io":time.monotonic()-began,"maximum_sample_units":units})
            event({"stage":stage,"status":"process_completed_unvalidated"})
        try:
            for spec in (plan["xrv"],plan["chexbert"]):check_pins(spec["asset_pins"])
            check_pins(plan["biovil_asset_pins"])
            cxrs=load_cxr_candidates([plan["cxr_run"]]);reports=load_report_candidates(plan["report_runs"],cxr_candidates=cxrs)
            if len(cxrs)!=2 or len(reports)!=8:raise ValueError("exact cached image/report inventory required")
            cb=plan["chexbert"]
            argv=[cb["python"],cb["script"],"--cxr-run",plan["cxr_run"],"--checkpoint",cb["checkpoint"],
                "--bert-path",str(CHEXBERT_BERT),"--batch-size","8","--output-root",str(root),"--run-id","labels"]
            for path in plan["report_runs"]:argv += ["--report-run",path]
            child("chexbert",argv,{"chexbert_scored_reports":8},180)
            label_path=root/"labels/report_finding_labels.json";labels=read_json(label_path)
            if (labels["counts"]!={"reports":8,"model_calls":8}
                    or {r["report_candidate_id"] for r in labels["records"]}!=set(reports)):
                raise ValueError("fresh report-label inventory differs")
            # Body-consuming structure evaluation is restricted to this approved
            # GPU controller. Never print generated report text in public output.
            from evaluate_report_structure import _record
            structures={rid:_record(r,cxrs[r["parent_cxr_candidate_id"]],normalized_frequency={}) for rid,r in reports.items()}
            frequency=Counter(s["normalized_report_sha256"] for s in structures.values())
            for s in structures.values():s["normalized_template_frequency"]=frequency[s["normalized_report_sha256"]]
            write_private_json(root/"structure_records.json",{"records":[structures[k] for k in sorted(structures)]})
            image_labels=read_json(plan["reused_xrv_labels"]);rows=[]
            if image_labels["counts"]!={"cxr_candidates":2,"model_calls":2}:raise ValueError("exact fresh cached image labels required")
            partials={p["cxr_candidate_id"]:p for p in plan["image_receipts"]}
            for case in plan["fixed_cases"]:
                anchor=anchor_from_record(case["anchor"]);image=cxrs[case["image"]["candidate_id"]]
                if image!=case["image"]:raise ValueError("unchanged historical image required")
                iv=single_view(image_labels,"cxr_candidates","cxr_candidate_id",image["candidate_id"])
                for report in reports.values():
                    if report["parent_cxr_candidate_id"]!=image["candidate_id"]:continue
                    receipt=completed_receipt(anchor,partials[image["candidate_id"]],image,iv,report,
                        single_view(labels,"reports","report_candidate_id",report["candidate_id"]),
                        image_labels_sha256=sha256_file(plan["reused_xrv_labels"]),report_labels_sha256=sha256_file(label_path),
                        thresholds_sha256=plan["xrv"]["thresholds_sha256"],xrv_checkpoint_sha256=plan["xrv"]["checkpoint_sha256"],
                        chexbert_checkpoint_sha256=cb["checkpoint_sha256"])
                    rows.append({"triple_candidate_id":"expertpair_"+_digest([image["candidate_id"],report["candidate_id"]])[:32],
                        "case_id":anchor.case_id,"cxr_candidate_id":image["candidate_id"],"report_candidate_id":report["candidate_id"],
                        "report_model_id":report["model_id"],"ehr_sha256":anchor.ehr_sha256,"ehr_facts_sha256":anchor.ehr_facts_sha256,
                        "cxr_sha256":image["artifact"]["sha256"],"report_sha256":report["artifact"]["sha256"],
                        "receipt":receipt,"structure":structures[report["candidate_id"]],"raw_edge_readouts":receipt["raw_edge_readouts"]})
            rows.sort(key=lambda r:r["triple_candidate_id"]);selection=freeze(rows)
            write_private_json(root/"score_rows.json",{"records":rows})
            sp=write_private_json(root/"selection.json",selection);selection_sha=sha256_file(sp)
            event({"stage":"chexbert","status":"validated","scored_report_samples":8})
            request_dir=root/"secondary_request";private_directory(request_dir)
            request={"schema_version":REQUEST_SCHEMA,"experiment":SCHEMA,"pairs":[{k:r[k] for k in PAIR_FIELDS} for r in rows],
                "selection_sha256":selection_sha,"modality_source":"fully_synthetic","selection_used_biovil":False,
                "clinical_truth_available":False,"routing_or_calibration_update_allowed":False,
                "text_policy":"full_report_no_silent_truncation_overlength_is_na"}
            rp=write_private_json(request_dir/"request.json",request)
            write_private_json(request_dir/"manifest.json",{"schema_version":REQUEST_SCHEMA,"artifacts":{"request.json":{"sha256":sha256_file(rp)}}})
            argv=[plan["biovil_python"],str(Path(__file__)),"biovil-worker","--request-run",str(request_dir),
                "--cxr-run",plan["cxr_run"],"--model-path",plan["biovil_model"],"--output-file",str(root/"secondary.json")]
            for path in plan["report_runs"]:argv += ["--report-run",path]
            child("biovil",argv,{"maximum_image_encodings":2,"maximum_text_encodings":8},180)
            endpoint=read_json(root/"secondary.json")
            if sha256_file(sp)!=selection_sha or endpoint["request_sha256"]!=sha256_file(rp):raise ValueError("sealed decision/request changed")
            comparison=measure(selection,rows,endpoint);methods=method_summary(rows,endpoint)
            write_private_json(root/"comparison.json",comparison);write_private_json(root/"method_summary.json",{"records":methods})
            scores={r["triple_candidate_id"]:r for r in endpoint["records"]};table=[]
            for row in rows:
                table.append(flatten({k:v for k,v in row.items() if k not in ("receipt","structure")})|{
                    "biovil_raw_cosine":scores[row["triple_candidate_id"]]["biovil_raw_cosine"],
                    "biovil_status":scores[row["triple_candidate_id"]]["status"],"biovil_na_reason":scores[row["triple_candidate_id"]]["reason"],
                    "official_section_contract_pass":row["structure"]["section_contract_pass"],
                    "impression_required":row["structure"]["impression_required_by_model_contract"],
                    "unsupported_temporal_language":row["structure"]["unsupported_temporal_comparison_language"],
                    "generic_report":row["structure"]["generic_report"]})
            show=lambda rs:render_csv([{k:"NA" if v is None else v for k,v in r.items()} for r in rs])
            write_private_text(root/"score_table.csv",show(table));write_private_text(root/"model_comparison.csv",show([flatten(r) for r in methods]))
            event({"stage":"biovil","status":"validated","counts":endpoint["counts"]})
            check_pins(plan["source_pins"]);check_pins(plan["artifact_pins"])
            for spec in (plan["xrv"],plan["chexbert"]):check_pins(spec["asset_pins"])
            check_pins(plan["biovil_asset_pins"]);journal.flush();os.fsync(journal.fileno())
            files=("score_rows.json","structure_records.json","selection.json","secondary.json","comparison.json","method_summary.json",
                   "score_table.csv","model_comparison.csv","cost_journal.jsonl")
            result={"schema_version":SCHEMA,"status":"completed_cached_four_expert_control_unvalidated",
                "plan_manifest_sha256":args.plan_manifest_sha256,"source_nbest_manifest_sha256":plan["source_nbest_manifest_sha256"],
                "artifacts":{n:{"sha256":sha256_file(root/n)} for n in files},"selection_sha256_before_endpoint":selection_sha,
                "new_generation_calls":0,"new_xrv_calls":0,"reused_xrv_scored_images":2,"new_chexbert_scored_reports":8,
                "chexbert_labels_sha256":sha256_file(label_path),"chexbert_forward_batch_size":8,
                "legacy_scorer_model_calls_are_sample_counts":True,"secondary_counts":endpoint["counts"],
                "wall_seconds_including_startup_io":round(time.monotonic()-started,3),"worker_costs":costs,"gpu_seconds":None,
                "peak_vram_gib":{"chexbert":labels["peak_vram_gib"],"biovil":endpoint["peak_vram_gib"]},
                "clinical_acceptance":False,"adaptive_repair_executed":False,"original_winners_changed":False,
                "same_image_reports_are_independent_votes":False,"biovil_used_for_selection":False,
                "report_generation_costs_are_historical_sunk_costs_not_zero":True}
            write_private_json(root/"manifest.json",result)
        except BaseException as exc:
            event({"status":"failed_or_interrupted_retained","error_type":type(exc).__name__,"automatic_resume":False})
            write_private_json(root/"failure_manifest.json",{"status":"failed_or_interrupted_retained",
                "error_type":type(exc).__name__,"automatic_resume":False,"clinical_acceptance":False})
            raise
        finally:normalize_owned_modes(root)
    return root,result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="mode",required=True)
    launch=sub.add_parser("run")
    for name in ("plan-run","plan-manifest-sha256","output-root","run-id"):launch.add_argument("--"+name,required=True)
    worker=sub.add_parser("biovil-worker")
    for name in ("request-run","model-path","output-file"):worker.add_argument("--"+name,required=True)
    for name in ("cxr-run","report-run"):worker.add_argument("--"+name,action="append",required=True)
    args=p.parse_args();os.umask(0o007)
    try:
        if args.mode=="biovil-worker":
            require_gpu_slurm();sys.path.insert(0,str(WORKSPACE/"runtime/vendor/hi_ml_multimodal_0_2_2"))
            from score_automatic_replay_biovil import score
            write_private_json(require_inside(args.output_file,PROTECTED_ROOT,must_exist=False),score(args));return 0
        root,result=run(args)
    except Exception as exc:
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}));return 1
    print(json.dumps({"status":result["status"],"runtime_seconds":result["wall_seconds_including_startup_io"],
        "manifest_sha256":sha256_file(root/"manifest.json")}));return 0


if __name__=="__main__":raise SystemExit(main())
