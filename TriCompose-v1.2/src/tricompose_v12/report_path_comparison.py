"""Fixed-image/frozen-key report paths, with secondary endpoints attached later.

This is a cached development comparison. No endpoint oracle, new policy,
clinical acceptance, model execution, image selection or source mutation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics

from .automatic_replay import candidate_key, validate_policy
from .complete_endpoint_inventory import PAIR_FIELDS
from .invariant_verification import _candidate_facts, _edge, anchor_from_cached_candidate
from .legacy_replay_adapter import FINDINGS, PROFILE

SCHEMA = "tricompose-frozen-report-path-comparison-v1"
STATIC = "same_image_static_source_key"


def _fixed_name(model):
    return "fixed_"+model


def _mean(values):
    values=list(values)
    return statistics.mean(values) if values else None


def _states(candidate):
    facts=_candidate_facts(candidate)
    return [{"finding":name,**facts[name]["states"]} for name in FINDINGS]


def freeze_choices(bank,policy):
    """Only source scores/labels/lineage are arguments; no secondary endpoint."""
    validate_policy(policy)
    if policy["routing_evidence"]!=PROFILE or not 1<=len(bank)<=80:
        raise ValueError("bounded historical development profile required")
    expected={(*image,report) for image in policy["image_order"] for report in policy["report_order"]}
    choices=[];seen=set()
    for case,grid in sorted(bank.items()):
        if set(grid)!=expected:raise ValueError("complete registered image/report inventory required")
        anchor=None
        for image_model,seed in policy["image_order"]:
            group=[grid[(image_model,seed,model)] for model in policy["report_order"]]
            image_identity=None;image_vector=None
            for c in group:
                row=c["score_record"];lineage=row["lineage"]
                current=anchor_from_cached_candidate(c)
                if row["case_id"]!=case or current.case_id!=case:
                    raise ValueError("fixed case differs from source")
                if anchor is None:anchor=current
                elif current!=anchor:raise ValueError("fixed EHR states/hash/category changed")
                identity=tuple(lineage[k] for k in ("cxr_candidate_id","cxr_sha256"))
                vector=tuple(f["xrv"] for f in _states(c))
                if image_identity is None:image_identity,image_vector=identity,vector
                elif identity!=image_identity or vector!=image_vector:
                    raise ValueError("same-image reference changed between report experts")
                if (lineage["cxr_model_id"]!=image_model or lineage["cxr_seed"]!=seed
                        or lineage["report_model_id"] not in policy["report_order"]):
                    raise ValueError("model/seed slot lineage differs")
                cid=row["triple_candidate_id"]
                if cid in seen:raise ValueError("duplicate source candidate")
                gate=row["scoring"]["selection"]["hard_gate_failure_count"]
                if type(gate) is not int or gate<0:
                    raise ValueError("nonnegative integer artifact gate count required")
                seen.add(cid);candidate_key(c)
            if [c["score_record"]["lineage"]["report_model_id"] for c in group]!=policy["report_order"]:
                raise ValueError("report model slot differs")
            eligible=[c for c in group if c["score_record"]["scoring"]["selection"]["hard_gate_failure_count"]==0]
            selected=min(eligible,key=candidate_key) if eligible else None
            methods=[(_fixed_name(model),c) for model,c in zip(policy["report_order"],group)]+[(STATIC,selected)]
            for method,c in methods:
                row=c["score_record"] if c is not None else None
                choices.append({"case_id":case,"cxr_model_id":image_model,"cxr_seed":seed,
                    "cxr_candidate_id":image_identity[0],"cxr_sha256":image_identity[1],
                    "ehr_sha256":anchor.ehr_sha256,"ehr_facts_sha256":anchor.ehr_facts_sha256,
                    "method":method,"selected_candidate_id":None if row is None else row["triple_candidate_id"],
                    "selected_report_model":None if row is None else row["lineage"]["report_model_id"],
                    "source_key":None if c is None else [None if isinstance(v,float) and not math.isfinite(v) else v for v in candidate_key(c)],
                    "source_gate_failure_count":None if row is None else row["scoring"]["selection"]["hard_gate_failure_count"],
                    "status":"no_eligible_report_unresolved" if c is None else "cached_source_path_not_clinically_validated",
                    "simulated_generator_scorer_calls":10 if method==STATIC else 4,
                    "simulated_incremental_report_scorer_calls":8 if method==STATIC else 2,
                    "direct_ehr_known_facts":sum(s in {"positive","negative"} for _,s,_ in anchor.findings),
                    "clinical_acceptance":False,"selection_used_biovil":False})
    return choices


def attach_endpoint(choices,bank,policy,records):
    """Assert frozen decisions, exact whole-bank endpoint lineage, then measure."""
    if choices!=freeze_choices(bank,policy):raise ValueError("choices changed before measurement")
    candidates={c["score_record"]["triple_candidate_id"]:c for grid in bank.values() for c in grid.values()}
    endpoint={r["triple_candidate_id"]:r for r in records}
    if len(endpoint)!=len(records) or set(endpoint)!=set(candidates):
        raise ValueError("complete unique secondary inventory required")
    for cid,r in endpoint.items():
        row=candidates[cid]["score_record"]
        expected={"case_id":row["case_id"],"triple_candidate_id":cid,
            **{k:row["lineage"][k] for k in PAIR_FIELDS if k not in {"case_id","triple_candidate_id"}}}
        if any(r.get(k)!=v for k,v in expected.items()):raise ValueError("endpoint parent/case/hash differs")
        v=r["biovil_raw_cosine"]
        if v is None:
            if r.get("status")!="not_available" or not r.get("reason"):
                raise ValueError("NA endpoint requires reason/status")
        elif (type(v) not in (int,float) or not math.isfinite(v) or not -1.01<=v<=1.01
                or r.get("reason") is not None or r.get("status")!="computed_secondary_uncalibrated"):
            raise ValueError("finite raw endpoint required, not a probability")
        if r.get("calibrated") is not False:raise ValueError("raw cosine is not calibrated")
    rows=[]
    for choice in choices:
        cid=choice["selected_candidate_id"]
        # Keep reference evidence even when no eligible report was selected.
        reference=bank[choice["case_id"]][(choice["cxr_model_id"],choice["cxr_seed"],policy["report_order"][0])]
        states=_states(reference if cid is None else candidates[cid])
        edges={"ehr_cxr":_edge(states,"ehr","xrv"),
            "ehr_report":None if cid is None else _edge(states,"ehr","chexbert"),
            "cxr_report":None if cid is None else _edge(states,"xrv","chexbert")}
        rows.append({**choice,"biovil_raw_cosine":None if cid is None else endpoint[cid]["biovil_raw_cosine"],
            "endpoint_unavailable_reason":"no_eligible_selected_report" if cid is None else endpoint[cid]["reason"],
            "raw_edge_readouts":edges,"image_positive_reference_facts":sum(f["xrv"]=="positive" for f in states),
            "image_negative_reference_facts":sum(f["xrv"]=="negative" for f in states),
            "clinical_accuracy":None})
    return rows


def _case_values(rows):
    groups=defaultdict(list)
    for row in rows:groups[row["case_id"]].append(row)
    return {case:_mean(r["biovil_raw_cosine"] for r in group) if all(r["biovil_raw_cosine"] is not None for r in group) else None
            for case,group in sorted(groups.items())}


def aggregate(rows):
    """Each EHR averages its three images before cohort comparison; NA stays NA."""
    groups=defaultdict(list)
    for row in rows:
        subgroup="direct_cached_ehr_fact" if row["direct_ehr_known_facts"] else "no_direct_cached_ehr_fact"
        for image in ("all_image_generators",row["cxr_model_id"]):
            for scope in ("all",subgroup):groups[(row["method"],image,scope)].append(row)
    tables=[];pair_cases=[];contrasts=[]
    for (method,image,scope),group in sorted(groups.items()):
        values=_case_values(group);available=[v for v in values.values() if v is not None]
        edge_totals={}
        for name in ("ehr_cxr","ehr_report","cxr_report"):
            all_edges=[r["raw_edge_readouts"][name] for r in group]
            edges=[e for e in all_edges if e is not None]
            totals={key:sum(e[key] for e in edges) for key in ("known_reference_facts","comparable_facts","supported_positive","supported_negative","proxy_opposition_facts")}
            known=totals["known_reference_facts"]
            totals.update(unavailable_selected_edge_slots=len(all_edges)-len(edges),
                coverage_over_known=totals["comparable_facts"]/known if known and len(edges)==len(all_edges) else None,
                opposition_over_known=totals["proxy_opposition_facts"]/known if known and len(edges)==len(all_edges) else None)
            edge_totals[name]=totals
        ce=edge_totals["cxr_report"]
        positive=sum(r["image_positive_reference_facts"] for r in group)
        negative=sum(r["image_negative_reference_facts"] for r in group)
        table={"method":method,"cxr_group":image,"ehr_evidence_subgroup":scope,
            "fixed_ehr_cases":len(values),"fixed_image_slots":len(group),"endpoint_available_ehr_cases":len(available),
            "full_cohort_ehr_mean_biovil":_mean(available) if len(available)==len(values) else None,
            "available_ehr_mean_biovil_not_full_cohort":_mean(available),
            "image_positive_support_over_known":ce["supported_positive"]/positive if positive and not ce["unavailable_selected_edge_slots"] else None,
            "image_negative_support_over_known":ce["supported_negative"]/negative if negative and not ce["unavailable_selected_edge_slots"] else None,
            "cxr_report_proxy_opposition_over_known":ce["opposition_over_known"],
            "cxr_report_coverage_over_known":ce["coverage_over_known"],
            "simulated_calls_per_fixed_image":_mean(r["simulated_generator_scorer_calls"] for r in group),
            "simulated_incremental_report_calls_per_fixed_image":_mean(r["simulated_incremental_report_scorer_calls"] for r in group),
            "source_gate_failed_path_slots":sum(r["source_gate_failure_count"] is not None and r["source_gate_failure_count"]>0 for r in group),
            "unresolved_selected_report_slots":sum(r["selected_candidate_id"] is None for r in group),
            "raw_edge_totals":edge_totals,"clinical_accuracy":None,"clinical_acceptance":False}
        tables.append(table)
    for (method,image,scope),group in sorted(groups.items()):
        if method!=STATIC:continue
        avalues=_case_values(group)
        for baseline in sorted({k[0] for k in groups if k[0]!=STATIC and k[1:]==(image,scope)}):
            other=groups[(baseline,image,scope)];bvalues=_case_values(other)
            if set(avalues)!=set(bvalues):raise ValueError("fixed EHR inventory differs across methods")
            ai={(r["case_id"],r["cxr_candidate_id"]):r for r in group}
            bi={(r["case_id"],r["cxr_candidate_id"]):r for r in other}
            if set(ai)!=set(bi) or len(ai)!=len(group) or len(bi)!=len(other):
                raise ValueError("same-image paired inventory differs")
            for key in ai:
                if any(ai[key][k]!=bi[key][k] for k in ("cxr_sha256","ehr_sha256","ehr_facts_sha256")):
                    raise ValueError("paired comparison changed image/EHR")
            ds=[]
            for case in sorted(avalues):
                a,b=avalues[case],bvalues[case]
                delta=a-b if a is not None and b is not None else None
                if delta is not None:ds.append(delta)
                pair_cases.append({"case_id":case,"cxr_group":image,"ehr_evidence_subgroup":scope,
                    "baseline":baseline,"method":STATIC,"baseline_case_mean_biovil":b,
                    "selected_case_mean_biovil":a,"delta_selected_minus_baseline":delta,
                    "all_required_images_available":delta is not None,"clinical_accuracy":None})
            contrasts.append({"baseline":baseline,"method":STATIC,"cxr_group":image,"ehr_evidence_subgroup":scope,
                "fixed_ehr_cases":len(avalues),"paired_available_ehr_cases":len(ds),
                "full_cohort_mean_biovil_delta":_mean(ds) if len(ds)==len(avalues) else None,
                "available_case_mean_biovil_delta":_mean(ds),"positive_delta_cases":sum(d>0 for d in ds),
                "zero_delta_cases":sum(d==0 for d in ds),"negative_delta_cases":sum(d<0 for d in ds),
                "simulated_call_delta_per_fixed_image":_mean(r["simulated_generator_scorer_calls"] for r in group)-_mean(r["simulated_generator_scorer_calls"] for r in other),
                "clinical_accuracy":None,"significance_test_performed":False})
    frequency=[]
    for image in ("all_image_generators",*sorted({r["cxr_model_id"] for r in rows})):
        selected=[r for r in rows if r["method"]==STATIC and (image=="all_image_generators" or r["cxr_model_id"]==image)]
        for model,count in sorted(Counter(r["selected_report_model"] or "unresolved" for r in selected).items()):
            frequency.append({"cxr_group":image,"report_model":model,"selected_image_slots":count,
                "fixed_ehr_cases":len({r["case_id"] for r in selected}),"clinical_model_rank":None})
    return {"method_comparison":tables,"paired_comparison":contrasts,"paired_case_rows":pair_cases,
        "selection_frequencies":frequency,"new_model_calls":0,"original_selection_changed":False,
        "policy_or_thresholds_changed":False,"selection_used_biovil":False,"actual_gpu_savings":None,
        "profile":PROFILE,"clinical_acceptance":False,"clinical_accuracy":None}


def verify_previous_sana(choices,previous,maximum_budget):
    """Same source-key projection must reproduce the archived Sana-only control."""
    now={r["case_id"]:r for r in choices if r["method"]==STATIC and r["cxr_model_id"]=="chexgenbench_sana"}
    old=[r for r in previous if r["method"]=="report_only_static" and r["model_call_budget"]==maximum_budget]
    prior={r["case_id"]:r for r in old}
    if len(prior)!=len(old) or set(now)!=set(prior):
        raise ValueError("archived Sana cohort differs")
    for case,r in now.items():
        p=prior[case]
        if (p["selected_candidate_id"]!=r["selected_candidate_id"]
                or p["control_fixed_image_sha256"]!=r["cxr_sha256"]
                or p["selected_ehr_sha256"]!=r["ehr_sha256"]):
            raise ValueError("archived Sana choice/image/EHR changed")
        snapshot=p.get("selected_snapshot")
        if snapshot is not None and (snapshot["artifact_hashes"]["ehr_facts_sha256"]!=r["ehr_facts_sha256"]
                or snapshot["report_model_id"]!=r["selected_report_model"]):
            raise ValueError("archived Sana report/EHR-facts lineage changed")
    return {"archived_sana_cases":len(prior),"exact_same_choices":len(now),"archived_choices_changed":False}
