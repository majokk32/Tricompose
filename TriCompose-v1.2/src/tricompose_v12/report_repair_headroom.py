"""Same-image label-preserving opportunity analysis, NOT a repair/selection policy.

Freeze every directed report pair before attaching secondary endpoints. No
scalar weights, new operating points, clinical truth or model execution.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import math
import statistics

from .invariant_verification import _candidate_facts
from .legacy_replay_adapter import FINDINGS,PROFILE
from .report_path_comparison import freeze_choices,attach_endpoint

SCHEMA="tricompose-report-repair-headroom-v1"
EXPLICIT={"positive","negative"}
SET_FIELDS=("image_positive_support","ehr_direct_support","image_comparable","ehr_comparable",
            "image_opposition","ehr_opposition")


def signature(candidate):
    """Fact IDs, not just counts, plus existing metadata-quality availability."""
    row=candidate["score_record"];facts=_candidate_facts(candidate)
    states={name:facts[name]["states"] for name in FINDINGS}
    out={k:[] for k in SET_FIELDS}
    for name,s in states.items():
        if s["chexbert"] not in EXPLICIT:continue
        if s["xrv"] in EXPLICIT:
            out["image_comparable"].append(name)
            if s["xrv"]!=s["chexbert"]:out["image_opposition"].append(name)
            elif s["xrv"]=="positive":out["image_positive_support"].append(name)
        if s["ehr"] in EXPLICIT:
            out["ehr_comparable"].append(name)
            out["ehr_opposition" if s["ehr"]!=s["chexbert"] else "ehr_direct_support"].append(name)
    q=row["scoring"]["modality_quality"]["report_structure_quality_score_0_1"]
    if q is not None and (type(q) not in (int,float) or not math.isfinite(q) or not 0<=q<=1):
        raise ValueError("finite existing report-structure score or explicit NA required")
    image_valid=row["scoring"]["modality_quality"]["cxr_basic_validity_pass"]
    gate=row["scoring"]["selection"]["hard_gate_failure_count"]
    if type(image_valid) is not bool or type(gate) is not int or gate<0:
        raise ValueError("typed existing artifact metadata required")
    return {"case_id":row["case_id"],"triple_candidate_id":row["triple_candidate_id"],
        "lineage":dict(row["lineage"]),**{k:sorted(v) for k,v in out.items()},
        "image_reference_states":{name:s["xrv"] for name,s in states.items()},
        "ehr_reference_states":{name:s["ehr"] for name,s in states.items()},
        "ehr_source_categories":{name:sorted(facts[name]["source_categories"]) for name in FINDINGS},
        "report_structure_quality":q,"source_gate_failure_count":gate,"source_image_validity":image_valid,
        "direct_ehr_known_facts":sum(s["ehr"] in EXPLICIT for s in states.values()),
        "image_known_facts":sum(s["xrv"] in EXPLICIT for s in states.values()),
        "report_positive_unknown_image":[name for name,s in states.items() if s["chexbert"]=="positive" and s["xrv"] not in EXPLICIT],
        "clinical_accuracy":None,"clinical_acceptance":False}


def compare(base,alternative):
    """Strict set preservation: silence cannot count as correcting opposition."""
    fixed=("cxr_candidate_id","cxr_sha256","ehr_sha256","ehr_facts_sha256","cxr_model_id","cxr_seed")
    if (base["case_id"]!=alternative["case_id"]
            or base["triple_candidate_id"]==alternative["triple_candidate_id"]
            or any(base["lineage"][k]!=alternative["lineage"][k] for k in fixed)
            or any(base[k]!=alternative[k] for k in ("image_reference_states","ehr_reference_states","ehr_source_categories"))):
        raise ValueError("same-image and fixed-EHR source scope required")
    a={k:set(base[k]) for k in SET_FIELDS};b={k:set(alternative[k]) for k in SET_FIELDS}
    losses={k:sorted(a[k]-b[k]) for k in SET_FIELDS[:4]}
    new_opposition={k:sorted(b[k]-a[k]) for k in SET_FIELDS[4:]}
    gains={k:sorted(b[k]-a[k]) for k in SET_FIELDS[:4]}
    removed={k:sorted(a[k]-b[k]) for k in SET_FIELDS[4:]}
    silence={edge:sorted(set(removed[edge+"_opposition"]) & set(losses[edge+"_comparable"])) for edge in ("image","ehr")}
    preserved=not any(losses.values()) and not any(new_opposition.values())
    strict=preserved and (any(gains.values()) or any(removed.values()))
    available=(base["report_structure_quality"] is not None and alternative["report_structure_quality"] is not None)
    quality_ok=available and alternative["report_structure_quality"]>=base["report_structure_quality"]
    artifacts_ok=all(r["source_gate_failure_count"]==0 and r["source_image_validity"] for r in (base,alternative))
    assessable=available and artifacts_ok
    reasons=[]
    if not artifacts_ok:reasons.append("source_artifact_metadata_failed")
    if not available:reasons.append("existing_structure_quality_unavailable")
    elif not quality_ok:reasons.append("existing_structure_quality_lower")
    if any(losses.values()):reasons.append("loses_existing_supported_or_comparable_fact_ids")
    if any(new_opposition.values()):reasons.append("introduces_new_explicit_proxy_opposition")
    if any(silence.values()):reasons.append("opposition_removed_by_unknown_or_uncertain_not_repair")
    if preserved and not strict:reasons.append("no_strict_finding_evidence_improvement")
    return {"case_id":base["case_id"],"cxr_model_id":base["lineage"]["cxr_model_id"],
        "cxr_seed":base["lineage"]["cxr_seed"],"cxr_candidate_id":base["lineage"]["cxr_candidate_id"],
        "cxr_sha256":base["lineage"]["cxr_sha256"],"ehr_sha256":base["lineage"]["ehr_sha256"],
        "ehr_facts_sha256":base["lineage"]["ehr_facts_sha256"],
        "baseline_candidate_id":base["triple_candidate_id"],"alternative_candidate_id":alternative["triple_candidate_id"],
        "baseline_report_model":base["lineage"]["report_model_id"],"alternative_report_model":alternative["lineage"]["report_model_id"],
        "direct_ehr_known_facts":base["direct_ehr_known_facts"],"image_known_facts":base["image_known_facts"],
        "preserved_fact_ids":preserved,"strict_label_dominance_without_quality_gate":bool(strict),
        "headroom_assessable":assessable,"strict_label_preserving_headroom":bool(strict and quality_ok and artifacts_ok),
        "lost_fact_ids":losses,"new_opposition_fact_ids":new_opposition,
        "gained_fact_ids":gains,"removed_opposition_fact_ids":removed,"opposition_silenced_fact_ids":silence,
        "baseline_structure_quality":base["report_structure_quality"],"alternative_structure_quality":alternative["report_structure_quality"],
        "baseline_raw_opposition_count":sum(len(a[k]) for k in SET_FIELDS[4:]),
        "alternative_raw_opposition_count":sum(len(b[k]) for k in SET_FIELDS[4:]),
        "alternative_positive_unknown_image":alternative["report_positive_unknown_image"],
        "block_or_equivalence_reasons":reasons,"clinical_repair_success":False,"clinical_accuracy":None}


def prepare(bank,policy,choices):
    """All report-pair opportunities computed without secondary endpoint access."""
    if choices!=freeze_choices(bank,policy):raise ValueError("archived source-only choices changed")
    evidence={c["score_record"]["triple_candidate_id"]:signature(c) for grid in bank.values() for c in grid.values()}
    pairs=[];graphs=[]
    for case,grid in sorted(bank.items()):
        for image,seed in policy["image_order"]:
            ids=[grid[(image,seed,m)]["score_record"]["triple_candidate_id"] for m in policy["report_order"]]
            image_pairs=[compare(evidence[base],evidence[alt]) for base in ids for alt in ids if base!=alt]
            pairs.extend(image_pairs)
            dominated={p["baseline_candidate_id"] for p in image_pairs if p["strict_label_preserving_headroom"]}
            eligible=[cid for cid in ids if evidence[cid]["source_gate_failure_count"]==0 and evidence[cid]["source_image_validity"]]
            graphs.append({"case_id":case,"cxr_model_id":image,"cxr_seed":seed,
                "cxr_candidate_id":evidence[ids[0]]["lineage"]["cxr_candidate_id"],
                "registered_candidates":ids,"metadata_eligible_candidates":eligible,
                "unassessable_quality_candidates":[cid for cid in eligible if evidence[cid]["report_structure_quality"] is None],
                "not_dominated_under_frozen_diagnostic":[cid for cid in eligible if cid not in dominated],
                "strict_dominance_directed_pairs":sum(p["strict_label_preserving_headroom"] for p in image_pairs),
                "clinical_pareto_optimality":None})
    bybase=defaultdict(list)
    for pair in pairs:bybase[pair["baseline_candidate_id"]].append(pair)
    paths=[]
    for c in choices:
        cid=c["selected_candidate_id"];options=bybase[cid] if cid is not None else []
        allowed=[p["alternative_candidate_id"] for p in options if p["strict_label_preserving_headroom"]]
        paths.append({**{k:c[k] for k in ("case_id","cxr_model_id","cxr_seed","cxr_candidate_id","cxr_sha256","ehr_sha256","ehr_facts_sha256","method","direct_ehr_known_facts")},
            "baseline_candidate_id":cid,"label_preserving_alternative_ids":allowed,
            "registered_alternative_count":len(options),"assessable_alternative_count":sum(p["headroom_assessable"] for p in options),
            "strict_label_preserving_alternative_count":len(allowed),
            "new_selected_candidate_id":None,"actual_regeneration_executed":False})
    return {"schema_version":SCHEMA,"profile":PROFILE,"candidate_evidence":[evidence[k] for k in sorted(evidence)],
        "directed_pairs":pairs,"image_graphs":graphs,"path_opportunities":paths,
        "secondary_endpoint_used_for_opportunity":False,"new_policy_executed":False,
        "original_choices_changed":False,"clinical_acceptance":False,"new_model_calls":0}


def attach(plan,bank,policy,choices,records):
    """Measurement only; full endpoint parent validation and frozen-plan equality."""
    if plan!=prepare(bank,policy,choices):raise ValueError("sealed diagnostic scope/pairs changed")
    attach_endpoint(choices,bank,policy,records)  # Validate exact full inventory/lineage/NA semantics.
    endpoint={r["triple_candidate_id"]:r for r in records}
    pairs=[]
    for pair in plan["directed_pairs"]:
        base,alt=(endpoint[pair[k]] for k in ("baseline_candidate_id","alternative_candidate_id"))
        a,b=base["biovil_raw_cosine"],alt["biovil_raw_cosine"]
        pairs.append({**pair,"baseline_biovil_raw_cosine":a,"alternative_biovil_raw_cosine":b,
            "biovil_delta_alternative_minus_baseline":b-a if a is not None and b is not None else None,
            "endpoint_na_reasons":{"baseline":base["reason"],"alternative":alt["reason"]},
            "endpoint_is_secondary_not_clinical_truth":True})
    bypair={(p["baseline_candidate_id"],p["alternative_candidate_id"]):p for p in pairs}
    paths=[]
    for p in plan["path_opportunities"]:
        ds=[bypair[(p["baseline_candidate_id"],alt)]["biovil_delta_alternative_minus_baseline"] for alt in p["label_preserving_alternative_ids"]]
        paths.append({**p,"eligible_alternative_endpoint_deltas":ds,
            "mean_all_eligible_alternative_delta_not_selection":statistics.mean(ds) if ds and all(d is not None for d in ds) else None,
            "endpoint_available_eligible_alternatives":sum(d is not None for d in ds)})
    return {**plan,"directed_pairs":pairs,"path_opportunities":paths}


def summarize(measured):
    """All baseline cases stay; conditional opportunity denominators are explicit."""
    groups=defaultdict(list)
    for p in measured["path_opportunities"]:
        subgroup="direct_cached_ehr_fact" if p["direct_ehr_known_facts"] else "no_direct_cached_ehr_fact"
        for image in ("all_image_generators",p["cxr_model_id"]):
            for scope in ("all",subgroup):groups[(p["method"],image,scope)].append(p)
    tables=[]
    for (method,image,scope),paths in sorted(groups.items()):
        opportunities=[p for p in paths if p["strict_label_preserving_alternative_count"]]
        case_groups=defaultdict(list)
        for p in opportunities:case_groups[p["case_id"]].append(p)
        case_values=[]
        for ps in case_groups.values():
            values=[p["mean_all_eligible_alternative_delta_not_selection"] for p in ps]
            case_values.append(statistics.mean(values) if all(v is not None for v in values) else None)
        available=[v for v in case_values if v is not None]
        deltas=[d for p in opportunities for d in p["eligible_alternative_endpoint_deltas"]]
        tables.append({"method":method,"cxr_group":image,"ehr_evidence_subgroup":scope,
            "fixed_ehr_cases":len({p["case_id"] for p in paths}),"fixed_image_slots":len(paths),
            "image_slots_with_label_preserving_alternative":len(opportunities),
            "ehr_cases_with_any_label_preserving_alternative":len(case_groups),
            "image_slots_without_existing_alternative":sum(p["strict_label_preserving_alternative_count"]==0 for p in paths),
            "image_slots_with_no_assessable_alternative":sum(p["assessable_alternative_count"]==0 for p in paths),
            "dominating_alternative_pair_count":len(deltas),"endpoint_available_dominating_pairs":sum(d is not None for d in deltas),
            "dominating_pairs_biovil_higher":sum(d is not None and d>0 for d in deltas),
            "dominating_pairs_biovil_equal":sum(d==0 for d in deltas),
            "dominating_pairs_biovil_lower":sum(d is not None and d<0 for d in deltas),
            "conditional_opportunity_ehr_cases_with_complete_endpoints":len(available),
            "conditional_ehr_balanced_mean_all_alternative_delta_not_policy":statistics.mean(available) if available and len(available)==len(case_values) else None,
            "available_opportunity_case_mean_not_full_denominator":statistics.mean(available) if available else None,
            "clinical_accuracy":None,"clinical_repair_success":False,"new_model_calls":0})
    pairs=measured["directed_pairs"]
    headroom=[p for p in pairs if p["strict_label_preserving_headroom"]]
    reductions=[p for p in pairs if p["alternative_raw_opposition_count"]<p["baseline_raw_opposition_count"]]
    reasons=Counter(reason for p in pairs for reason in p["block_or_equivalence_reasons"])
    transitions=Counter((p["baseline_report_model"],p["alternative_report_model"]) for p in headroom)
    return {"method_comparison":tables,"model_pair_opportunities":[{"baseline_report_model":a,"alternative_report_model":b,
        "strict_label_preserving_pairs":n,"clinical_model_rank":None} for (a,b),n in sorted(transitions.items())],
        "counts":{"fixed_ehr_cases":len({p["case_id"] for p in measured["image_graphs"]}),
            "fixed_image_slots":len(measured["image_graphs"]),"report_candidates":len(measured["candidate_evidence"]),
            "directed_report_pairs":len(pairs),"baseline_path_slots":len(measured["path_opportunities"]),
            "strict_label_preserving_directed_pairs":len(headroom),"pairs_with_raw_opposition_reduction":len(reductions),
            "opposition_reducing_pairs_rejected_by_preservation_or_quality":sum(not p["strict_label_preserving_headroom"] for p in reductions),
            "opposition_reducing_pairs_silencing_existing_conflict":sum(any(p["opposition_silenced_fact_ids"].values()) for p in reductions)},
        "overlapping_pair_block_reason_counts":dict(sorted(reasons.items())),
        "profile":PROFILE,"no_new_winner_or_policy":True,"secondary_endpoint_used_for_opportunity":False,
        "clinical_accuracy":None,"clinical_repair_success":False,"new_model_calls":0}
