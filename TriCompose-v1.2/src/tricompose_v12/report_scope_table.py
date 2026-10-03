"""Candidate evidence availability, not a selector or clinical fault policy.

The four-finding frozen syntax gate only withdraws report proposals. All eight
cached findings remain in the denominator. No new clinical rules or weights.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re

from .report_assertions import FINDINGS as SCOPE_FINDINGS, STATES, digest, gate_assertion

FINDINGS = ("atelectasis", "cardiomegaly", "consolidation", "edema",
    "lung_opacity", "pleural_effusion", "pneumonia", "pneumothorax")
EDGES = ("ehr_cxr", "ehr_report", "cxr_report")
EXPLICIT = frozenset({"positive", "negative"})
RELATIONS = ("support", "opposition", "unknown", "not_comparable")
SCHEMA = "tricompose-candidate-report-scope-availability-v1"


def relation(left, right, *, available=True):
    if left not in STATES or right not in STATES:
        raise ValueError("invalid four-state comparison")
    if not available or "uncertain" in (left,right):
        return "not_comparable"
    if "unknown" in (left,right):
        return "unknown"
    return "support" if left == right else "opposition"


def edge_summary(rows, edge, stage):
    counts=Counter(row["relations"][stage][edge] for row in rows)
    reference="ehr" if edge.startswith("ehr_") else "xrv"
    known=sum(row["states"][reference] in EXPLICIT for row in rows)
    comparable=counts["support"]+counts["opposition"]
    return {"inventory_facts":len(rows),"explicit_reference_facts":known,
        **{name:counts[name] for name in RELATIONS},"comparable_facts":comparable,
        "coverage_over_inventory":comparable/len(rows) if rows else None,
        "coverage_over_explicit_reference":comparable/known if known else None,
        "conditional_support_fraction":counts["support"]/comparable if comparable else None,
        "conditional_opposition_fraction":counts["opposition"]/comparable if comparable else None,
        "clinical_accuracy":None}


def validate_inputs(raw_facts, texts):
    if not raw_facts or len(raw_facts)>48*len(FINDINGS):
        raise ValueError("bounded nonempty eight-finding pilot required")
    by_candidate=defaultdict(list)
    keys, evidence_ids, cases, image_states, ehr_states, report_states=set(),set(),{}, {},{},{}
    report_hashes=set()
    for row in raw_facts:
        key=(row["triple_candidate_id"],row["finding"])
        if row["finding"] not in FINDINGS or key in keys or row["evidence_id"] in evidence_ids:
            raise ValueError("duplicate or foreign fact inventory")
        keys.add(key);evidence_ids.add(row["evidence_id"])
        hashes=row["artifact_hashes"]
        if (set(hashes)!={"ehr_sha256","ehr_facts_sha256","cxr_sha256","report_sha256"} or
                any(not isinstance(h,str) or not re.fullmatch(r"[0-9a-f]{64}",h) for h in hashes.values())):
            raise ValueError("invalid artifact hash inventory")
        if (set(row["states"])!={"ehr","xrv","chexbert","qwen_image","qwen_report"} or
                any(state not in STATES for state in row["states"].values())):
            raise ValueError("invalid source states")
        provenance=row["ehr_provenance"]
        if provenance.get("weak_clinical_context_promoted") is not False:
            raise ValueError("weak EHR context cannot become a direct fact")
        if row["states"]["ehr"] != "unknown" and (
                not provenance.get("evidence_ids") or not provenance.get("source_fields")):
            raise ValueError("asserted EHR state lacks direct provenance")
        case=row["case_id"];finding=row["finding"]
        fixed=(hashes["ehr_sha256"],hashes["ehr_facts_sha256"])
        if cases.setdefault(case,fixed)!=fixed:
            raise ValueError("fixed EHR changed within case")
        for index,k,value in (
                (ehr_states,(case,finding),(row["states"]["ehr"],provenance)),
                (image_states,(case,hashes["cxr_sha256"],finding),
                    (row["states"]["xrv"],row["states"]["qwen_image"])),
                (report_states,(hashes["report_sha256"],finding),
                    (row["states"]["chexbert"],row["states"]["qwen_report"]))):
            if index.setdefault(k,value)!=value:
                raise ValueError("shared artifact has inconsistent cached evidence")
        by_candidate[row["triple_candidate_id"]].append(row)
        report_hashes.add(hashes["report_sha256"])
    if len(cases)>2 or report_hashes!=set(texts):
        raise ValueError("fixed two-case source text inventory differs")
    for h,text in texts.items():
        if not isinstance(text,str) or not 1<=len(text)<=8192 or digest(text)!=h:
            raise ValueError("original synthetic report text/hash differs")
    for rows in by_candidate.values():
        identity={(row["case_id"],row["cxr_candidate_id"],row["report_candidate_id"],
            tuple(sorted(row["artifact_hashes"].items()))) for row in rows}
        if len(rows)!=len(FINDINGS) or len(identity)!=1:
            raise ValueError("candidate must retain one complete aligned finding inventory")
    return by_candidate


def build_scope_table(raw_facts, texts, scope_checker):
    by_candidate=validate_inputs(raw_facts,texts)
    assertions,rows={},[]
    for original in sorted(raw_facts,key=lambda row:(row["triple_candidate_id"],FINDINGS.index(row["finding"]))):
        finding=original["finding"];h=original["artifact_hashes"]["report_sha256"]
        supported=finding in SCOPE_FINDINGS
        if supported:
            key=(h,finding)
            if key not in assertions:
                assertions[key]=gate_assertion(texts[h],finding,original["states"]["chexbert"],scope_checker)
            scoped=assertions[key]
            state,decision=scoped["state"],scoped["decision"]
            available=decision in {"scope_commit","no_model_assertion"}
            reason=scoped["reason"]
        else:
            state,decision,available,reason="unknown","outside_scope_inventory",False,"frozen_guard_has_no_head"
        if state not in {original["states"]["chexbert"],"unknown"}:
            raise ValueError("gate cannot flip or manufacture a model label")
        states=dict(original["states"])
        raw={"ehr_cxr":relation(states["ehr"],states["xrv"]),
            "ehr_report":relation(states["ehr"],states["chexbert"]),
            "cxr_report":relation(states["xrv"],states["chexbert"])}
        gated={"ehr_cxr":raw["ehr_cxr"],
            "ehr_report":relation(states["ehr"],state,available=available),
            "cxr_report":relation(states["xrv"],state,available=available)}
        rows.append({"evidence_id":original["evidence_id"],"case_id":original["case_id"],
            "triple_candidate_id":original["triple_candidate_id"],
            "cxr_candidate_id":original["cxr_candidate_id"],
            "report_candidate_id":original["report_candidate_id"],
            "report_model_id":original.get("report_model_id"),
            "artifact_hashes":dict(original["artifact_hashes"]),"states":states,
            "ehr_provenance":original["ehr_provenance"],
            "finding":finding,"report_scope_state":state,"report_scope_decision":decision,
            "report_scope_reason":reason,"report_scope_supported_finding":supported,
            "report_scope_available":available,"relations":{"raw":raw,"scoped":gated},
            "report_scope_evidence_key":f"{h}:{finding}" if supported else None,
            "image_dependency_group":original["artifact_hashes"]["cxr_sha256"],
            "clinical_error_confirmed":False,"confirmed_faulty_modality":None,
            "automatic_repair_eligible":False})
    candidate_rows=[]
    for cid in sorted(by_candidate):
        facts=[row for row in rows if row["triple_candidate_id"]==cid]
        first=facts[0]
        candidate_rows.append({"case_id":first["case_id"],"triple_candidate_id":cid,
            "cxr_candidate_id":first["cxr_candidate_id"],"report_candidate_id":first["report_candidate_id"],
            "report_model_id":first["report_model_id"],"artifact_hashes":first["artifact_hashes"],
            "finding_evidence_ids":[row["evidence_id"] for row in facts],
            "source_bound_scope_commits":sum(row["report_scope_decision"]=="scope_commit" for row in facts),
            "scope_abstentions":sum(row["report_scope_decision"]=="abstain" for row in facts),
            "findings_without_scope_head":sum(not row["report_scope_supported_finding"] for row in facts),
            "edges":{stage:{edge:edge_summary(facts,edge,stage) for edge in EDGES} for stage in ("raw","scoped")},
            "withdrawn_support_signals":{edge:sum(row["relations"]["raw"][edge]=="support" and
                row["relations"]["scoped"][edge]!="support" for row in facts) for edge in EDGES},
            "withdrawn_opposition_signals":{edge:sum(row["relations"]["raw"][edge]=="opposition" and
                row["relations"]["scoped"][edge]!="opposition" for row in facts) for edge in EDGES},
            "clinical_selection_score":None,"next_step":"independent_evidence_required",
            "automatic_repair_eligible":False})
    group_facts=defaultdict(dict)
    for row in rows:
        key=(row["case_id"],row["artifact_hashes"]["cxr_sha256"],row["finding"])
        group_facts[key].setdefault(row["artifact_hashes"]["report_sha256"],row)
    groups=[]
    for (case,ihash,finding),reports in sorted(group_facts.items()):
        raw=Counter(row["states"]["chexbert"] for row in reports.values())
        gated=Counter(row["report_scope_state"] for row in reports.values())
        groups.append({"case_id":case,"cxr_sha256":ihash,"finding":finding,
            "distinct_report_texts":len(reports),"raw_report_state_counts":dict(raw),
            "scope_report_state_counts":dict(gated),
            "raw_positive_negative_disagreement":raw["positive"]>0 and raw["negative"]>0,
            "scope_positive_negative_disagreement":gated["positive"]>0 and gated["negative"]>0,
            "independent_votes":False,"confirmed_faulty_modality":None})
    image_rows={}
    for row in rows:
        image_rows.setdefault((row["case_id"],row["artifact_hashes"]["cxr_sha256"],row["finding"]),row)
    summary={"schema_version":SCHEMA,"candidate_triples":len(candidate_rows),"fact_rows":len(rows),
        "independent_ehr_cases":len({row["case_id"] for row in rows}),
        "unique_image_finding_pairs":len(image_rows),"finding_order":list(FINDINGS),
        "scope_finding_order":list(SCOPE_FINDINGS),
        "report_scope_decision_counts":dict(Counter(row["report_scope_decision"] for row in rows)),
        "relations":{stage:{edge:edge_summary(list(image_rows.values()) if edge=="ehr_cxr" else rows,edge,stage)
            for edge in EDGES} for stage in ("raw","scoped")},
        "cross_path_image_finding_groups":len(groups),
        "raw_report_disagreement_groups":sum(row["raw_positive_negative_disagreement"] for row in groups),
        "scope_report_disagreement_groups":sum(row["scope_positive_negative_disagreement"] for row in groups),
        "model_calls":0,"primary_metric_eligible":False,"selection_changed":False,
        "regeneration_authorized":False,"clinical_accuracy":None,"clinical_localization_accuracy":None,
        "independent_human_labels_used":False,"report_coverage_is_syntax_not_clinical_truth":True,
        "interpretation":"Withdrawn comparisons mean lost evidence, not corrected reports or clinical improvement. Frozen report scope covers four of eight findings; the rest remain explicitly unavailable. Shared-CXR reports are not independent votes."}
    return rows,candidate_rows,groups,summary,{f"{h}:{finding}":value for (h,finding),value in sorted(assertions.items())}
