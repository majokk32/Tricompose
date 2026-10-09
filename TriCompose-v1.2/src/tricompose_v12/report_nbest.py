"""Prospective same-image n-best control; labels are proxies, not clinical truth.

Pure functions only. No model, filesystem, endpoint-based selection, or training.
Fresh receipts are deliberately not converted to the historical 14-head profile.
"""
from collections import defaultdict
import math

from .invariant_verification import _digest
from .legacy_replay_adapter import FINDINGS, STATES
from .live_receipts import PROFILE

SCHEMA = "tricompose-fixed-image-report-nbest-v1"
POLICY = {
    "schema_version": SCHEMA,
    "report_model": "cxrmate_single",
    "fixed_image_model": "chexgenbench_sana",
    "fixed_image_seed": 0,
    "opaque_case_indices": [0, 1],
    "num_beams": 4,
    "num_return_sequences": 3,
    "max_length": 256,
    "do_sample": False,
    "baseline_beam_rank": 0,
    "gate": "strict_fact_id_preservation_and_no_worse_structure_v1",
    "eligible_choice": "lowest_native_beam_rank",
    "uses_biovil": False,
    "clinical_acceptance_allowed": False,
    "adaptive_repair_executed": False,
}
EXPLICIT = {"positive", "negative"}
SET_FIELDS = ("image_positive_support", "ehr_direct_support", "image_comparable",
              "ehr_comparable", "image_opposition", "ehr_opposition")
PASS_FLAGS = ("findings_complete", "impression_complete", "section_contract_pass")
RISK_FLAGS = ("empty", "generic_report", "unsupported_temporal_comparison_language")


def validate_policy(policy):
    # Reject extra options, unsupported stochastic decoding and post-hoc changes.
    if policy != POLICY:
        raise ValueError("only the predeclared two-case deterministic n-best control is supported")


def canonical_report(findings, impression):
    sections = []
    for heading, content in (("FINDINGS", findings), ("IMPRESSION", impression)):
        text = str(content).strip()
        if text:
            sections.append(heading + ":\n" + text)
    if not sections:
        raise ValueError("empty n-best report")
    return "\n\n".join(sections) + "\n"


def evidence_sets(receipt):
    if (receipt.get("schema_version") != "tricompose-fresh-completed-receipt-v1"
            or receipt.get("profile") != PROFILE
            or receipt.get("primary_clinical_metric_eligible") is not False
            or receipt.get("clinical_acceptance") is not False):
        raise ValueError("compatible fresh diagnostic receipt required")
    if receipt.get("receipt_id") != _digest({k: v for k, v in receipt.items() if k != "receipt_id"}):
        raise ValueError("fresh receipt digest changed")
    facts = receipt["fact_states"]
    if ([f["finding"] for f in facts] != list(FINDINGS)
            or any(f[k] not in STATES for f in facts for k in ("ehr", "xrv", "chexbert"))):
        raise ValueError("complete four-state named facts required")
    sets = {k: set() for k in SET_FIELDS}
    for f in facts:
        name, report = f["finding"], f["chexbert"]
        if report not in EXPLICIT:
            continue
        for modality, edge in (("xrv", "image"), ("ehr", "ehr")):
            ref = f[modality]
            if ref not in EXPLICIT:
                continue
            sets[edge + "_comparable"].add(name)
            if ref != report:
                sets[edge + "_opposition"].add(name)
            elif edge == "ehr":
                sets["ehr_direct_support"].add(name)
            elif ref == "positive":
                sets["image_positive_support"].add(name)
    return sets


def compare(base, other):
    a, b = base["receipt"], other["receipt"]
    fixed = ("case_id", "ehr_anchor_sha256", "cxr_candidate_id", "cxr_sha256",
             "thresholds_sha256", "xrv_checkpoint_sha256", "chexbert_checkpoint_sha256", "profile")
    if (any(a[k] != b[k] for k in fixed) or a["report_candidate_id"] == b["report_candidate_id"]
            or [(f["finding"], f["ehr"], f["xrv"]) for f in a["fact_states"]]
               != [(f["finding"], f["ehr"], f["xrv"]) for f in b["fact_states"]]):
        raise ValueError("same immutable EHR/image/scorer profile required")
    left, right = evidence_sets(a), evidence_sets(b)
    lost = {k: sorted(left[k] - right[k]) for k in SET_FIELDS[:4]}
    new_opposition = {k: sorted(right[k] - left[k]) for k in SET_FIELDS[4:]}
    gains = {k: sorted(right[k] - left[k]) for k in SET_FIELDS[:4]}
    removed = {k: sorted(left[k] - right[k]) for k in SET_FIELDS[4:]}
    silenced = {e: sorted(set(removed[e + "_opposition"]) & set(lost[e + "_comparable"]))
                for e in ("image", "ehr")}
    for row in (base, other):
        s = row["structure"]
        if (any(type(s[k]) is not bool for k in PASS_FLAGS + RISK_FLAGS)
                or type(s["repeated_sentence_count"]) is not int or s["repeated_sentence_count"] < 0
                or type(s["repeated_4gram_ratio"]) not in (int, float)
                or not math.isfinite(s["repeated_4gram_ratio"]) or not 0 <= s["repeated_4gram_ratio"] <= 1):
            raise ValueError("typed existing structure checks required")
    s, t = base["structure"], other["structure"]
    quality_ok = (not t["empty"] and t["section_contract_pass"]
        and all(not s[k] or t[k] for k in PASS_FLAGS)
        and all(not t[k] or s[k] for k in RISK_FLAGS)
        and t["repeated_sentence_count"] <= s["repeated_sentence_count"]
        and t["repeated_4gram_ratio"] <= s["repeated_4gram_ratio"])
    duplicate = (a["report_sha256"] == b["report_sha256"]
                 or s["normalized_report_sha256"] == t["normalized_report_sha256"])
    strict = not any(lost.values()) and not any(new_opposition.values()) and (any(gains.values()) or any(removed.values()))
    reasons = []
    if duplicate: reasons.append("duplicate_not_new_diversity")
    if not quality_ok: reasons.append("structure_missing_or_worse")
    if any(lost.values()): reasons.append("lost_supported_or_comparable_fact_ids")
    if any(new_opposition.values()): reasons.append("new_explicit_proxy_opposition")
    if any(silenced.values()): reasons.append("conflict_silenced_not_corrected")
    if not any(gains.values()) and not any(removed.values()): reasons.append("no_strict_evidence_improvement")
    return {"baseline_triple_id": base["triple_candidate_id"], "alternative_triple_id": other["triple_candidate_id"],
        "lost_fact_ids": lost, "new_opposition_fact_ids": new_opposition, "gained_fact_ids": gains,
        "removed_opposition_fact_ids": removed, "silenced_fact_ids": silenced,
        "structure_no_worse": quality_ok, "duplicate": duplicate,
        "exploratory_gate_pass": bool(strict and quality_ok and not duplicate),
        "reasons": reasons, "clinical_repair_success": False}


def freeze(rows):
    groups = defaultdict(list)
    if len(rows) != 6 or len({r["triple_candidate_id"] for r in rows}) != 6:
        raise ValueError("exact two-image six-sequence control required")
    for row in rows:
        evidence_sets(row["receipt"])
        if row["triple_candidate_id"] != "nbestpair_" + _digest([row["cxr_candidate_id"], row["report_candidate_id"]])[:32]:
            raise ValueError("candidate pair ID changed")
        for k in ("case_id", "cxr_candidate_id", "report_candidate_id", "cxr_sha256", "report_sha256"):
            if row[k] != row["receipt"][k]: raise ValueError("row/receipt lineage differs")
        groups[row["cxr_candidate_id"]].append(row)
    if len(groups) != 2 or len({r["case_id"] for r in rows}) != 2:
        raise ValueError("fixed two EHRs and two images required")
    choices, comparisons = [], []
    for iid, group in sorted(groups.items()):
        if {r["beam_rank"] for r in group} != {0, 1, 2} or len(group) != 3:
            raise ValueError("exact native beam ranks required")
        if any(type(r["beam_rank"]) is not int for r in group): raise ValueError("integer native beam ranks required")
        for k in ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256"):
            if len({r[k] for r in group}) != 1: raise ValueError("fixed row anchor changed")
        base = next(r for r in group if r["beam_rank"] == 0)
        pairs = [compare(base, r) for r in sorted(group, key=lambda r: r["beam_rank"]) if r is not base]
        comparisons.extend(pairs)
        eligible = {p["alternative_triple_id"] for p in pairs if p["exploratory_gate_pass"]}
        chosen = next((r for r in sorted(group, key=lambda r: r["beam_rank"]) if r["triple_candidate_id"] in eligible), base)
        choices.append({"case_id": base["case_id"], "cxr_candidate_id": iid,
            "baseline_triple_id": base["triple_candidate_id"], "selected_triple_id": chosen["triple_candidate_id"],
            "eligible_alternative_ids": sorted(eligible), "status": "exploratory_gate_pass" if eligible else "unresolved_baseline_retained"})
    return {"schema_version": SCHEMA, "policy": POLICY, "profile": PROFILE,
        "rows_sha256": _digest(rows), "choices": choices, "comparisons": comparisons,
        "used_biovil": False, "adaptive_repair_executed": False, "clinical_acceptance": False,
        "original_winners_changed": False}


def measure(selection, rows, endpoint):
    if freeze(rows) != selection: raise ValueError("selection changed after endpoint")
    index = {r["triple_candidate_id"]: r for r in rows}
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    if (len(scores) != 6 or len(endpoint["records"]) != 6 or set(scores) != set(index)
            or endpoint.get("used_for_routing") is not False or endpoint.get("clinical_truth_available") is not False):
        raise ValueError("complete secondary-only endpoint required")
    fields = ("case_id", "cxr_candidate_id", "report_candidate_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")
    for cid, score in scores.items():
        if any(score[k] != index[cid][k] for k in fields) or score.get("calibrated") is not False:
            raise ValueError("endpoint lineage/calibration differs")
        value = score["biovil_raw_cosine"]
        if value is None:
            if not score.get("reason") or score["status"] != "not_available": raise ValueError("explicit NA required")
        elif (type(value) not in (int, float) or not math.isfinite(value) or not -1.01 <= value <= 1.01
              or score["status"] != "computed_secondary_uncalibrated" or score.get("reason") is not None):
            raise ValueError("invalid raw secondary cosine")
    pairs = []
    for choice in selection["choices"]:
        a, b = (scores[choice[k]]["biovil_raw_cosine"] for k in ("baseline_triple_id", "selected_triple_id"))
        pairs.append({**choice, "baseline_raw_cosine": a, "selected_raw_cosine": b,
            "delta": b-a if a is not None and b is not None else None})
    values = [p["delta"] for p in pairs]
    return {"schema_version": SCHEMA, "case_pairs": pairs,
        "two_case_mean_delta": sum(values)/2 if all(v is not None for v in values) else None,
        "available_pairs": sum(v is not None for v in values), "total_case_pairs": 2,
        "endpoint_used_for_selection": False, "clinical_accuracy": None, "clinical_repair_success": False,
        "interpretation": "engineering_nbest_control_not_online_targeted_repair_or_efficacy"}
