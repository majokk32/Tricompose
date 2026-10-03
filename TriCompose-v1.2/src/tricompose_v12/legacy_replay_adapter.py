"""Read-only adapter for the historical, uncalibrated synthetic candidate bank.

Only cached four-state vectors and artifact lineage are used. This does not
invent a syntax guard, recover clinical truth, or promote weak EHR priors.
The original global No-Finding adjustment is checked as a legacy source metric;
unknown report states remain unknown in the actual routing evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re

from .automatic_replay import candidate_key, validate_policy
from .report_scope_table import EXPLICIT, relation

FINDINGS = (
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
    "pleural_effusion", "pleural_other", "pneumonia", "pneumothorax",
    "support_devices", "no_finding")
STATES = {"positive", "negative", "uncertain", "unknown"}
HASH_FIELDS = ("ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")
PROFILE = "legacy_fourteen_raw_uncalibrated_xrv_chexbert_states"
SCHEMA = "tricompose-legacy-synthetic-cache-replay-v1"


def vector(value):
    if (not isinstance(value, dict) or set(value) != set(FINDINGS)
            or any(state not in STATES for state in value.values())):
        raise ValueError("complete four-state fourteen-finding vector required")
    return {name: value[name] for name in FINDINGS}


def counts(reference, candidate):
    known = sum(state in EXPLICIT for state in reference.values())
    comparable = [(reference[name], candidate[name]) for name in FINDINGS
                  if reference[name] in EXPLICIT and candidate[name] in EXPLICIT]
    return {"known_reference_fact_count": known,
            "support_count": sum(left == right for left, right in comparable),
            "contradiction_count": sum(left != right for left, right in comparable)}


def metric_counts(ehr, image, report):
    adjusted = dict(report)
    if report["no_finding"] == "positive":
        for name in FINDINGS:
            if (name not in {"no_finding", "support_devices"}
                    and ehr[name] == "positive" and report[name] not in EXPLICIT):
                adjusted[name] = "negative"
    return {"ehr_cxr": counts(ehr, image), "ehr_report": counts(ehr, adjusted),
            "report_cxr": counts(image, report)}


def _index(rows, id_field):
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty cached edge records required")
    result = {}
    for row in rows:
        key = (row["case_id"], row[id_field])
        if key in result:
            raise ValueError("duplicate cached artifact evidence")
        result[key] = row
    return result


def _fixed(index, key, value):
    if index.setdefault(key, value) != value:
        raise ValueError("shared artifact or fixed EHR evidence changed")


def make_legacy_bank(scores, details, policy):
    validate_policy(policy)
    if policy["routing_evidence"] != PROFILE:
        raise ValueError("legacy cache requires its explicit uncalibrated profile")
    if (details.get("schema_version") != "tricompose-ehr-edge-crossmodal-evaluation-v1.1"
            or details.get("evaluation_scope") != {
                "cohort": "fully_synthetic_cold_start_non_longitudinal",
                "real_reference_supplied": False, "unknown_is_negative": False,
                "weak_prior_is_hard_label": False}):
        raise ValueError("synthetic-only, unknown-preserving source scope required")
    if not isinstance(scores, list) or not 1 <= len(scores) <= 960:
        raise ValueError("bounded legacy score inventory required")
    images = _index(details["records"]["ehr_cxr"], "cxr_candidate_id")
    reports = _index(details["records"]["ehr_report"], "report_candidate_id")
    bank, fixed_ehr, shared_images, shared_reports = {}, {}, {}, {}
    used_images, used_reports, triples = set(), set(), set()
    expected_slots = {(model, seed, report) for model, seed in policy["image_order"]
                      for report in policy["report_order"]}
    for row in scores:
        if row.get("schema_version") != "tricompose-edge-specific-selection-v1.1":
            raise ValueError("legacy selection schema differs")
        if row["triple_candidate_id"] in triples:
            raise ValueError("duplicate candidate triple")
        triples.add(row["triple_candidate_id"])
        case, lineage = row["case_id"], row["lineage"]
        hashes = {name: lineage[name] for name in HASH_FIELDS}
        if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()):
            raise ValueError("invalid frozen artifact hash")
        ikey, rkey = (case, lineage["cxr_candidate_id"]), (case, lineage["report_candidate_id"])
        if ikey not in images or rkey not in reports or rkey in used_reports:
            raise ValueError("cached edge/score inventories differ")
        image, report = images[ikey], reports[rkey]
        if (image["image_sha256"] != hashes["cxr_sha256"]
                or report["report_sha256"] != hashes["report_sha256"]
                or image["cxr_model_id"] != lineage["cxr_model_id"]
                or report["source_cxr_model_id"] != lineage["cxr_model_id"]
                or report["report_model_id"] != lineage["report_model_id"]):
            raise ValueError("cached artifact/model lineage differs")
        ehr = vector(image["ehr_finding_states"])
        if ehr != vector(report["ehr_finding_states"]):
            raise ValueError("EHR edge sources disagree")
        source_vectors = report.get("ehr_finding_states_by_source", {})
        if not source_vectors:
            raise ValueError("direct EHR source category evidence required")
        for values in source_vectors.values():
            source = vector(values)
            if any(state != "unknown" and state != ehr[name] for name, state in source.items()):
                raise ValueError("direct EHR source category differs")
        if any(state != "unknown" and not any(v[name] == state for v in source_vectors.values())
               for name, state in ehr.items()):
            raise ValueError("asserted EHR proxy lacks cached source category")
        xrv, chexbert = vector(image["cxr_finding_states"]), vector(report["report_finding_states"])
        _fixed(fixed_ehr, case, (hashes["ehr_sha256"], hashes["ehr_facts_sha256"], ehr))
        _fixed(shared_images, hashes["cxr_sha256"], xrv)
        _fixed(shared_reports, hashes["report_sha256"], chexbert)
        edges = metric_counts(ehr, xrv, chexbert)
        scoring = row["scoring"]
        if set(scoring["edge_metrics"]) != set(edges):
            raise ValueError("source edge inventory differs")
        for name, values in edges.items():
            for field, value in values.items():
                source = scoring["edge_metrics"][name][field]
                if type(source) is not int or source != value:
                    raise ValueError("cached proxy edge counts differ from source states")
        totals = scoring["clinical_totals"]
        expected_totals = {
            "total_hard_contradiction_count": sum(x["contradiction_count"] for x in edges.values()),
            "total_direct_support_count": sum(x["support_count"] for x in edges.values()),
            "total_known_reference_fact_count": sum(x["known_reference_fact_count"] for x in edges.values()),
            "ehr_direct_support_count": sum(edges[name]["support_count"] for name in ("ehr_cxr", "ehr_report"))}
        if any(type(totals[name]) is not int or totals[name] != value for name, value in expected_totals.items()):
            raise ValueError("cached proxy totals differ")
        known = expected_totals["total_known_reference_fact_count"]
        expected_balance = (50 * (1 + (expected_totals["total_direct_support_count"]
                            - expected_totals["total_hard_contradiction_count"]) / known)) if known else None
        balance = totals["clinical_balance_score_0_100"]
        if ((expected_balance is None) != (balance is None)
                or balance is not None and (not isinstance(balance, (int, float))
                    or isinstance(balance, bool) or not math.isfinite(balance)
                    or not math.isclose(balance, expected_balance, abs_tol=1e-6))):
            raise ValueError("cached proxy balance differs")
        gate = scoring["selection"]["hard_gate_failure_count"]
        if type(gate) is not int or gate < 0 or type(scoring["modality_quality"]["cxr_basic_validity_pass"]) is not bool:
            raise ValueError("invalid cached artifact gate")
        facts = []
        for finding in FINDINGS:
            identity = json.dumps([case, row["triple_candidate_id"], hashes, finding], sort_keys=True)
            facts.append({"evidence_id": hashlib.sha256(identity.encode()).hexdigest(),
                "case_id": case, "triple_candidate_id": row["triple_candidate_id"],
                "cxr_candidate_id": lineage["cxr_candidate_id"],
                "report_candidate_id": lineage["report_candidate_id"],
                "artifact_hashes": dict(hashes), "finding": finding,
                "states": {"ehr": ehr[finding], "xrv": xrv[finding], "chexbert": chexbert[finding]},
                "relations": {"raw": {
                    "ehr_cxr": relation(ehr[finding], xrv[finding]),
                    "ehr_report": relation(ehr[finding], chexbert[finding]),
                    "cxr_report": relation(xrv[finding], chexbert[finding])}},
                "weak_context_promoted": False,
                "source_categories": [name for name, v in source_vectors.items() if v[finding] != "unknown"],
                "provenance_resolution": "cached_direct_state_and_source_category_not_new_raw_ehr_review",
                "report_scope_available": None, "clinical_truth_verified": False})
        candidate = {"score_record": row, "facts": facts, "alternate_scope_readout_available": False}
        candidate_key(candidate)
        slot = (lineage["cxr_model_id"], lineage["cxr_seed"], lineage["report_model_id"])
        if slot not in expected_slots or slot in bank.setdefault(case, {}):
            raise ValueError("foreign or duplicate model slot")
        bank[case][slot] = candidate
        used_images.add(ikey); used_reports.add(rkey)
    if (not bank or len(bank) > 80 or any(set(grid) != expected_slots for grid in bank.values())
            or used_images != set(images) or used_reports != set(reports)):
        raise ValueError("all fixed cases require the complete registered legacy grid")
    actual_counts = {"cases": len(bank), "cxr_candidates": len(images), "report_candidates": len(reports)}
    if any(details["counts"].get(name) != count for name, count in actual_counts.items()):
        raise ValueError("source cohort counts differ")
    return bank


def inventory_summary(bank):
    """Count all input cases, including the ones with no direct EHR constraints."""
    counts_by_case = Counter()
    states = {name: Counter() for name in FINDINGS}
    for grid in bank.values():
        facts = next(iter(grid.values()))["facts"]
        counts_by_case["explicit_fact_proxy" if any(f["states"]["ehr"] in EXPLICIT for f in facts)
                       else "no_direct_comparable_ehr_facts"] += 1
        for fact in facts:
            states[fact["finding"]][fact["states"]["ehr"]] += 1
    return {"all_input_cases_retained": True, "case_counts_by_ehr_evidence": dict(counts_by_case),
            "ehr_finding_state_counts_one_vector_per_case": {
                name: {state: values[state] for state in sorted(STATES)} for name, values in states.items()},
            "weak_priors_used_for_hard_routing": False,
            "report_scope_check_status": "not_computed_for_legacy_cohort",
            "legacy_classifier_profile": "historical_uncalibrated_xrv_not_the_new_eight_head_profile"}
