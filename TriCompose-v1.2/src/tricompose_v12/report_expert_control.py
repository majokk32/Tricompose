"""Four cached experts with one compatible fresh image-label profile.

No generation, learning, endpoint-based decisions or clinical truth. This is a
separately declared cross-expert control, not a modification of the n-best gate.
"""
from collections import defaultdict
import math

from .invariant_verification import _digest
from .live_receipts import PROFILE
from .report_nbest import evidence_sets, SET_FIELDS, RISK_FLAGS

SCHEMA = "tricompose-fixed-image-four-expert-control-v1"
MODELS = ("cxrmate_single", "maira2", "llavarad", "chexagent2")
SECTION_CONTRACTS = {m: {"findings": True, "impression": m == "cxrmate_single"} for m in MODELS}
POLICY = {"schema_version": SCHEMA, "fixed_case_scope": "same_two_opaque_ordinals_as_nbest",
    "fixed_image_model": "chexgenbench_sana", "fixed_seed": 0, "baseline_model": "cxrmate_single",
    "expert_order": list(MODELS), "choice": "first_strict_fact_preserving_eligible_expert",
    "structure_contract": "per_model_official_sections_with_common_risk_nonregression_v1",
    "uses_biovil": False, "changes_original_winners": False, "new_generation_allowed": False,
    "adaptive_repair_executed": False, "clinical_acceptance": False}


def structure_ok(row):
    model, s = row["report_model_id"], row["structure"]
    if model not in MODELS: raise ValueError("undeclared report expert")
    required = SECTION_CONTRACTS[model]
    for key in ("findings_complete", "impression_complete", "section_contract_pass") + RISK_FLAGS:
        if type(s[key]) is not bool: raise ValueError("typed structure flags required")
    expected = s["findings_complete"] and (s["impression_complete"] or not required["impression"])
    if (s["section_contract_pass"] != expected
            or s["impression_required_by_model_contract"] != required["impression"]):
        raise ValueError("declared official section contract differs")
    if (type(s["repeated_sentence_count"]) is not int or s["repeated_sentence_count"] < 0
            or type(s["repeated_4gram_ratio"]) not in (int,float)
            or not math.isfinite(s["repeated_4gram_ratio"]) or not 0 <= s["repeated_4gram_ratio"] <= 1):
        raise ValueError("valid repetition metadata required")
    for key, value in (("report_candidate_id",row["report_candidate_id"]), ("report_sha256",row["report_sha256"]),
                       ("image_sha256",row["cxr_sha256"]), ("case_id",row["case_id"]), ("report_model_id",model)):
        if s[key] != value: raise ValueError("structure lineage differs")
    return expected and not s["empty"]


def compare(base, alt):
    a, b = base["receipt"], alt["receipt"]
    fixed = ("case_id", "ehr_anchor_sha256", "cxr_candidate_id", "cxr_sha256", "thresholds_sha256",
             "xrv_checkpoint_sha256", "chexbert_checkpoint_sha256", "profile")
    if (any(a[k] != b[k] for k in fixed) or a["report_candidate_id"] == b["report_candidate_id"]
            or [(f["finding"],f["ehr"],f["xrv"]) for f in a["fact_states"]]
              != [(f["finding"],f["ehr"],f["xrv"]) for f in b["fact_states"]]):
        raise ValueError("fixed EHR/image/profile scope required")
    left, right = evidence_sets(a), evidence_sets(b)
    lost = {k:sorted(left[k]-right[k]) for k in SET_FIELDS[:4]}
    new = {k:sorted(right[k]-left[k]) for k in SET_FIELDS[4:]}
    gained = {k:sorted(right[k]-left[k]) for k in SET_FIELDS[:4]}
    removed = {k:sorted(left[k]-right[k]) for k in SET_FIELDS[4:]}
    silenced = {e:sorted(set(removed[e+"_opposition"]) & set(lost[e+"_comparable"])) for e in ("image","ehr")}
    valid = structure_ok(base) and structure_ok(alt)
    s,t=base["structure"],alt["structure"]
    quality = (valid and all(not t[k] or s[k] for k in RISK_FLAGS)
        and t["repeated_sentence_count"] <= s["repeated_sentence_count"]
        and t["repeated_4gram_ratio"] <= s["repeated_4gram_ratio"])
    duplicate = a["report_sha256"] == b["report_sha256"] or s["normalized_report_sha256"] == t["normalized_report_sha256"]
    strict = not any(lost.values()) and not any(new.values()) and (any(gained.values()) or any(removed.values()))
    reasons=[]
    if duplicate: reasons.append("duplicate_not_expert_diversity")
    if not quality: reasons.append("official_structure_failed_or_common_risk_worse")
    if any(lost.values()): reasons.append("lost_supported_or_comparable_fact_ids")
    if any(new.values()): reasons.append("new_explicit_proxy_opposition")
    if any(silenced.values()): reasons.append("conflict_silenced_not_corrected")
    if not any(gained.values()) and not any(removed.values()): reasons.append("no_strict_evidence_improvement")
    return {"baseline_triple_id":base["triple_candidate_id"],"alternative_triple_id":alt["triple_candidate_id"],
        "baseline_model":base["report_model_id"],"alternative_model":alt["report_model_id"],
        "lost_fact_ids":lost,"new_opposition_fact_ids":new,"gained_fact_ids":gained,
        "removed_opposition_fact_ids":removed,"silenced_fact_ids":silenced,
        "per_model_structure_and_common_risk_no_worse":quality,"duplicate":duplicate,
        "exploratory_gate_pass":bool(strict and quality and not duplicate),"reasons":reasons,
        "clinical_repair_success":False}


def freeze(rows):
    if len(rows)!=8 or len({r["triple_candidate_id"] for r in rows})!=8:
        raise ValueError("exact two-image eight-report control required")
    groups=defaultdict(list)
    for row in rows:
        evidence_sets(row["receipt"]); structure_ok(row)
        for field in ("case_id","cxr_candidate_id","report_candidate_id","cxr_sha256","report_sha256"):
            if row[field]!=row["receipt"][field]: raise ValueError("row/receipt lineage differs")
        if row["triple_candidate_id"]!="expertpair_"+_digest([row["cxr_candidate_id"],row["report_candidate_id"]])[:32]:
            raise ValueError("opaque pair ID differs")
        groups[row["cxr_candidate_id"]].append(row)
    if len(groups)!=2 or len({r["case_id"] for r in rows})!=2: raise ValueError("fixed two EHRs/images required")
    choices,pairs=[],[]
    for iid,group in sorted(groups.items()):
        if len(group)!=4 or {r["report_model_id"] for r in group}!=set(MODELS): raise ValueError("one report per declared expert required")
        for field in ("case_id","ehr_sha256","ehr_facts_sha256","cxr_sha256"):
            if len({r[field] for r in group})!=1: raise ValueError("fixed row anchor differs")
        base=next(r for r in group if r["report_model_id"]==POLICY["baseline_model"])
        bymodel={r["report_model_id"]:r for r in group}
        comparisons=[compare(base,bymodel[m]) for m in MODELS if m!=POLICY["baseline_model"]]
        pairs.extend(comparisons)
        eligible={p["alternative_triple_id"] for p in comparisons if p["exploratory_gate_pass"]}
        chosen=next((bymodel[m] for m in MODELS if bymodel[m]["triple_candidate_id"] in eligible),base)
        choices.append({"case_id":base["case_id"],"cxr_candidate_id":iid,"baseline_triple_id":base["triple_candidate_id"],
            "selected_triple_id":chosen["triple_candidate_id"],"selected_model":chosen["report_model_id"],
            "eligible_alternative_ids":sorted(eligible),"status":"exploratory_gate_pass" if eligible else "unresolved_baseline_retained"})
    return {"schema_version":SCHEMA,"policy":POLICY,"profile":PROFILE,"rows_sha256":_digest(rows),
        "choices":choices,"comparisons":pairs,"used_biovil":False,"clinical_acceptance":False,
        "new_generation_calls":0,"adaptive_repair_executed":False,"original_winners_changed":False}


def measure(selection,rows,endpoint):
    if freeze(rows)!=selection: raise ValueError("selection changed after independent endpoint")
    index={r["triple_candidate_id"]:r for r in rows}; scores={r["triple_candidate_id"]:r for r in endpoint["records"]}
    if (len(scores)!=8 or len(endpoint["records"])!=8 or set(scores)!=set(index)
            or endpoint.get("used_for_routing") is not False or endpoint.get("clinical_truth_available") is not False):
        raise ValueError("complete secondary-only endpoint required")
    for cid,s in scores.items():
        if (any(s[k]!=index[cid][k] for k in ("case_id","cxr_candidate_id","report_candidate_id","ehr_sha256","ehr_facts_sha256","cxr_sha256","report_sha256"))
                or s.get("calibrated") is not False): raise ValueError("secondary lineage/calibration differs")
        value=s["biovil_raw_cosine"]
        if value is None:
            if not s.get("reason") or s["status"]!="not_available": raise ValueError("explicit NA required")
        elif (type(value) not in (int,float) or not math.isfinite(value) or not -1.01<=value<=1.01
              or s["status"]!="computed_secondary_uncalibrated" or s.get("reason") is not None): raise ValueError("invalid raw cosine")
    pairs=[]
    for c in selection["choices"]:
        a,b=(scores[c[k]]["biovil_raw_cosine"] for k in ("baseline_triple_id","selected_triple_id"))
        pairs.append({**c,"baseline_raw_cosine":a,"selected_raw_cosine":b,"delta":b-a if a is not None and b is not None else None})
    values=[p["delta"] for p in pairs]
    return {"schema_version":SCHEMA,"case_pairs":pairs,"two_case_mean_delta":sum(values)/2 if all(v is not None for v in values) else None,
        "available_pairs":sum(v is not None for v in values),"total_case_pairs":2,"endpoint_used_for_selection":False,
        "clinical_accuracy":None,"clinical_repair_success":False,"new_generation_calls":0,
        "interpretation":"cached_four_expert_fresh_profile_control_not_online_repair_or_efficacy"}
