#!/usr/bin/env python3
"""Offline positive-focused policy ablation for an existing scored smoke pool.

No model inference and no changes to the original score/selection runs.
Negative agreement is retained for audit but is not a positive reward.
"""
from __future__ import annotations
import argparse, csv, io, json, os
from collections import defaultdict
from pathlib import Path
from contracts import PROTECTED_ROOT, require_inside, read_json, sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text

FINDINGS=("atelectasis","cardiomegaly","consolidation","edema","enlarged_cardiomediastinum","fracture","lung_lesion","lung_opacity","pleural_effusion","pleural_other","pneumonia","pneumothorax","support_devices","no_finding")

def parts(edge):
    image, report=edge["cxr_finding_states"], edge["report_finding_states"]
    positive=negative=contradiction=known=0
    for f in FINDINGS:
        a,b=image[f],report[f]
        if a in ("positive","negative"): known+=1
        if a==b=="positive": positive+=1
        if a==b=="negative": negative+=1
        if a in ("positive","negative") and b in ("positive","negative") and a!=b: contradiction+=1
    return {"known":known,"positive_support":positive,"negative_support":negative,"contradictions":contradiction,"positive_recall":None if not known else round(positive/known,8)}

def key(row, edge):
    s=row["scoring"]; p=parts(edge)
    return (s["selection"]["hard_gate_failure_count"], s["clinical_totals"]["total_hard_contradiction_count"], -p["positive_support"], -p["positive_recall"] if p["positive_recall"] is not None else 1e9, -s["modality_quality"]["report_structure_quality_score_0_1"], s["cost"]["known_runtime_seconds"] if s["cost"]["known_runtime_seconds"] is not None else 1e9, row["triple_candidate_id"])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--selection-run",required=True); p.add_argument("--crossmodal",required=True); p.add_argument("--output-root",required=True); p.add_argument("--run-id",required=True); a=p.parse_args()
    selection=require_inside(a.selection_run,PROTECTED_ROOT,must_exist=True); cross=read_json(require_inside(a.crossmodal,PROTECTED_ROOT,must_exist=True)/"crossmodal_details.json")
    rows=[json.loads(x) for x in (selection/"candidate_score_table.jsonl").read_text().splitlines() if x.strip()]; edges={x["report_candidate_id"]:x for x in cross["records"]}
    groups=defaultdict(list)
    for row in rows: groups[row["case_id"]].append(row)
    chosen=[]; records=[]
    for case,values in sorted(groups.items()):
        ranked=sorted(values,key=lambda r:key(r,edges[r["triple_candidate_id"]])); winner=ranked[0]; chosen.append(winner)
        for rank,row in enumerate(ranked,1):
            metric=parts(edges[row["triple_candidate_id"]]); records.append({"case_id":case,"candidate_id":row["triple_candidate_id"],"cxr_model":row["lineage"]["cxr_model_id"],"seed":row["lineage"]["cxr_seed"],"report_model":row["lineage"]["report_model_id"],"rank":rank,"selected":rank==1,"positive_support":metric["positive_support"],"negative_support":metric["negative_support"],"contradictions":metric["contradictions"],"positive_recall":metric["positive_recall"]})
    fixed=[r for r in rows if r["lineage"]["cxr_model_id"]=="chexgenbench_sana" and r["lineage"]["cxr_seed"]==0 and r["lineage"]["report_model_id"]=="maira2"]
    def agg(values): return {"count":len(values),"positive_support":sum(x["positive_support"] for x in values),"negative_support":sum(x["negative_support"] for x in values),"contradictions":sum(x["contradictions"] for x in values),"mean_positive_recall":None if not values else round(sum((x["positive_recall"] or 0) for x in values)/len(values),8)}
    chosen_records=[x for x in records if x["selected"]]; fixed_records=[]
    for row in fixed: fixed_records.append({**parts(edges[row["triple_candidate_id"]]),"case_id":row["case_id"]})
    payload={"schema_version":"tricompose-positive-focused-selection-v1","policy":"positive_support_then_contradiction_then_structure_then_runtime_then_id","candidate_rows":len(rows),"cases":len(groups),"selected_triples":len(chosen),"gpu_inference_used":False,"original_selection_modified":False,"comparison":{"fixed":agg(fixed_records),"positive_focused":agg(chosen_records)},"selections":[{"case_id":r["case_id"],"candidate_id":r["triple_candidate_id"],"cxr_model":r["lineage"]["cxr_model_id"],"seed":r["lineage"]["cxr_seed"],"report_model":r["lineage"]["report_model_id"]} for r in chosen],"records":records}
    tmp,target=new_atomic_run(a.output_root,a.run_id)
    try:
        table=io.StringIO(); fields=list(records[0]); w=csv.DictWriter(table,fieldnames=fields); w.writeheader(); w.writerows(records); t=write_private_text(tmp/"positive_focused_candidates.csv",table.getvalue())
        s=write_private_json(tmp/"positive_focused_selection.json",payload)
        md=["# Positive-focused selection ablation","","This is a diagnostic offline policy ablation. Original scoring and selection are unchanged.","","| Method | Positive support | Negative support | Contradictions | Mean positive recall |","|---|---:|---:|---:|---:|"]
        for name,value in payload["comparison"].items(): md.append(f"| {name} | {value['positive_support']} | {value['negative_support']} | {value['contradictions']} | {value['mean_positive_recall']} |")
        md += ["","The policy does not reward negative agreement. Unknown remains unknown; XRV is uncalibrated; this is not clinical validation.","", "Selected candidates:"]+[f"- {x['case_id']}: {x['cxr_model']} seed {x['seed']} + {x['report_model']}" for x in payload["selections"]]
        m=write_private_text(tmp/"positive_focused_summary.md","\n".join(md)+"\n")
        write_private_json(tmp/"manifest.json",{"schema_version":payload["schema_version"],"run_id":a.run_id,"artifacts":{x.name:{"sha256":sha256_file(x),"size_bytes":x.stat().st_size} for x in (t,s,m)}}); commit_atomic_run(tmp,target)
    except Exception: discard_atomic_run(tmp); raise
    print(json.dumps({"status":"completed","output_directory":str(target),"comparison":payload["comparison"],"selections":payload["selections"]},sort_keys=True))
if __name__=="__main__": main()
