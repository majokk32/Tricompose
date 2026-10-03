#!/usr/bin/env python3
"""Write a protected fixed-vs-selected synthetic output review."""
from __future__ import annotations
import csv, hashlib, json, os, shutil
from collections import defaultdict
from pathlib import Path
from contracts import PROTECTED_ROOT, require_inside, read_json, read_report_text, sha256_file, load_cxr_candidates, load_report_candidates, new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text

FINDINGS=("atelectasis","cardiomegaly","consolidation","edema","enlarged_cardiomediastinum","fracture","lung_lesion","lung_opacity","pleural_effusion","pleural_other","pneumonia","pneumothorax","support_devices","no_finding")

def decompose(image, report):
    out={"positive_support":[],"negative_support":[],"contradictions":[],"positive_unmentioned":[]}
    for f in FINDINGS:
        a,b=image[f],report[f]
        if a==b=="positive": out["positive_support"].append(f)
        if a==b=="negative": out["negative_support"].append(f)
        if a in ("positive","negative") and b in ("positive","negative") and a!=b: out["contradictions"].append(f)
        if a=="positive" and b in ("unknown","uncertain"): out["positive_unmentioned"].append(f)
    return out

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--scoring",required=True); p.add_argument("--structure",required=True); p.add_argument("--cxr-run",action="append",required=True); p.add_argument("--report-run",action="append",required=True); p.add_argument("--output-root",required=True); p.add_argument("--run-id",required=True)
    a=p.parse_args(); scoring=require_inside(a.scoring,PROTECTED_ROOT,must_exist=True)
    selection=read_json(scoring/"selection/selection_results.json")
    rows=[json.loads(x) for x in (scoring/"selection/candidate_score_table.jsonl").read_text().splitlines() if x.strip()]
    structures=read_json(a.structure); structure={x["report_candidate_id"]:x for x in structures["records"]}
    cross=read_json(scoring/"crossmodal/crossmodal_details.json"); cross={x["report_candidate_id"]:x for x in cross["records"]}
    cxrs=load_cxr_candidates(a.cxr_run); reports=load_report_candidates(a.report_run,cxr_candidates=cxrs)
    byid={x["triple_candidate_id"]:x for x in rows}; fixed=selection["policy"]["operational_fixed_baseline"]
    selected={x["case_id"]:x["triple_candidate_id"] for x in selection["static_reranking"]["selections"]}
    arms=[]; allrows=[]
    for case,chosen in sorted(selected.items()):
        fixedrow=[x for x in rows if x["case_id"]==case and x["lineage"]["cxr_model_id"]==fixed["cxr_model_id"] and x["lineage"]["cxr_seed"]==fixed["cxr_seed"] and x["lineage"]["report_model_id"]==fixed["report_model_id"]][0]
        fixedid=fixedrow["triple_candidate_id"]; chosenrow=byid[chosen]
        controls={}
        for role,parent,model in (("fixed_image_selected_report",fixedrow["lineage"]["cxr_candidate_id"],chosenrow["lineage"]["report_model_id"]),("selected_image_fixed_report",chosenrow["lineage"]["cxr_candidate_id"],fixed["report_model_id"])):
            controls[role]=[x["triple_candidate_id"] for x in rows if x["lineage"]["cxr_candidate_id"]==parent and x["lineage"]["report_model_id"]==model][0]
        ids={"fixed":fixedid,"selected":chosen,**controls}; arms.append({"case_id":case,"ids":ids})
        for role,rid in ids.items():
            e=cross[rid]; d=decompose(e["cxr_finding_states"],e["report_finding_states"]); s=structure[rid]
            row={"case_id":case,"role":role,"report_id":rid,"cxr_id":byid[rid]["lineage"]["cxr_candidate_id"],"cxr_model":byid[rid]["lineage"]["cxr_model_id"],"seed":byid[rid]["lineage"]["cxr_seed"],"report_model":byid[rid]["lineage"]["report_model_id"],"tokens":s["token_count"],"positive_support":len(d["positive_support"]),"negative_support":len(d["negative_support"]),"contradictions":len(d["contradictions"]),"positive_unmentioned":len(d["positive_unmentioned"]),"biovil":e["secondary_scores"]["biovil_report_cxr"]}
            allrows.append(row)
    selected_rows=[x for x in allrows if x["role"]=="selected"]
    fixed_rows=[x for x in allrows if x["role"]=="fixed"]
    tmp,target=new_atomic_run(a.output_root,a.run_id)
    try:
        (tmp/"images").mkdir(mode=0o2770); os.chmod(tmp/"images",0o2770)
        lines=["# TriCompose fixed vs selected review", "", "仅包含 synthetic CXR/report。该文档是事后审查，不是临床真值或盲法专家评审。", "", "## 核心结论", "", "当前择优分数不能直接解释为临床质量提升。支持率增加主要需要拆成阳性支持和阴性支持；XRV 未校准，BioViL 只是辅助分数。", "", f"固定路径：{fixed['cxr_model_id']} seed {fixed['cxr_seed']} + {fixed['report_model_id']}", "", f"阳性支持（fixed → selected）：{sum(x['positive_support'] for x in fixed_rows)} → {sum(x['positive_support'] for x in selected_rows)}", f"阴性支持（fixed → selected）：{sum(x['negative_support'] for x in fixed_rows)} → {sum(x['negative_support'] for x in selected_rows)}", f"矛盾信号（fixed → selected）：{sum(x['contradictions'] for x in fixed_rows)} → {sum(x['contradictions'] for x in selected_rows)}", "", "规则把阳性和阴性一致都计入 support；unknown 不计为矛盾，也不计为支持。", ""]
        fields=list(allrows[0]); out=__import__('io').StringIO(); w=csv.DictWriter(out,fieldnames=fields); w.writeheader(); w.writerows(allrows)
        write_private_text(tmp/"candidate_review.csv",out.getvalue())
        for item in arms:
            case=item["case_id"]; lines += [f"## {case}", "", "| 路径 | CXR模型/seed | 报告模型 | token数 | 阳性支持 | 阴性支持 | 矛盾 | 阳性未表达 | BioViL |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
            for role,rid in item["ids"].items():
                r=next(x for x in allrows if x["report_id"]==rid); lines.append(f"| {role} | {r['cxr_model']} / {r['seed']} | {r['report_model']} | {r['tokens']} | {r['positive_support']} | {r['negative_support']} | {r['contradictions']} | {r['positive_unmentioned']} | {r['biovil']:.6f} |")
                cxrid=byid[rid]["lineage"]["cxr_candidate_id"]; src=Path(cxrs[cxrid]["artifact"]["path"]); dst=tmp/"images"/(cxrid+".png")
                if not dst.exists(): shutil.copyfile(src,dst); os.chmod(dst,0o660)
                lines += ["",f"### {role}","",f"![Synthetic CXR]({dst.relative_to(tmp)})","", "```text", read_report_text(reports[rid]).strip(), "```", ""]
            lines += ["同一 CXR 的不同报告模型对照有助于区分图像变化和报告变化；但所有报告仍依赖同一张合成图。", ""]
        lines += ["## 解释", "", "这两个病例的 EHR 没有直接 radiographic positive fact，因此 EHR–CXR 与 EHR–Report 保持 NA。当前结果只说明静态规则能在小样本中偏向某些图像/报告组合，不能证明全队列临床有效。", "", "需要下一步：独立校准 XRV 阈值、扩大有直接 radiographic facts 的病例、人工盲评，并比较固定/随机/静态择优。"]
        write_private_text(tmp/"comparison.md","\n".join(lines)+"\n")
        write_private_json(tmp/"review_summary.json",{"schema_version":"tricompose-selection-review-v1","cases":len(arms),"candidate_rows":len(rows),"gpu_inference_used":False,"original_selection_modified":False,"fixed_positive_support":sum(x["positive_support"] for x in fixed_rows),"selected_positive_support":sum(x["positive_support"] for x in selected_rows),"fixed_negative_support":sum(x["negative_support"] for x in fixed_rows),"selected_negative_support":sum(x["negative_support"] for x in selected_rows),"fixed_contradictions":sum(x["contradictions"] for x in fixed_rows),"selected_contradictions":sum(x["contradictions"] for x in selected_rows)})
        files=[x for x in tmp.rglob('*') if x.is_file()]; write_private_json(tmp/"manifest.json",{"schema_version":"tricompose-selection-review-v1","run_id":a.run_id,"artifacts":{str(x.relative_to(tmp)):{"sha256":sha256_file(x),"size_bytes":x.stat().st_size} for x in files}}); commit_atomic_run(tmp,target)
    except Exception: discard_atomic_run(tmp); raise
    print(json.dumps({"status":"completed","output_directory":str(target),"cases":len(arms),"candidate_rows":len(rows)}))
if __name__=="__main__": main()
