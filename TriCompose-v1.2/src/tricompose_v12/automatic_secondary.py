"""Hash-bound secondary readout, never a routing/selection update.

Random replicates are averaged within a fixed EHR only when ALL predeclared
replicates have an endpoint score. Missing endpoints remain unavailable.
The original proxy readouts and selected candidate IDs are left unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math
import statistics

from .automatic_replay import METHODS, validate_policy

REQUEST_SCHEMA = "tricompose-automatic-replay-secondary-request-v1"
SCORE_SCHEMA = "tricompose-automatic-replay-secondary-biovil-v1"
PAIR_FIELDS = {"case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id",
               "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256"}
REASONS = {"empty_report", "full_report_exceeds_text_context_no_truncation",
           "unsupported_tokenizer_special_token_in_report"}


def overlay(outcomes, request, scores):
    if (request.get("schema_version") != REQUEST_SCHEMA
            or request.get("modality_source") != "fully_synthetic"
            or request.get("selection_used_biovil") is not False
            or request.get("clinical_truth_available") is not False
            or request.get("routing_or_calibration_update_allowed") is not False
            or request.get("text_policy") != "full_report_no_silent_truncation_overlength_is_na"
            or scores.get("schema_version") != SCORE_SCHEMA
            or scores.get("status") != "completed_secondary_biovil"
            or scores.get("used_for_routing") is not False
            or scores.get("primary_clinical_metric") is not False
            or scores.get("original_selection_changed") is not False
            or scores.get("clinical_truth_available") is not False
            or scores.get("producer", {}).get("frozen") is not True
            or scores.get("producer", {}).get("model_id") != "biovil_t"
            or scores.get("producer", {}).get("text_policy") != request.get("text_policy")):
        raise ValueError("frozen secondary-only synthetic contract required")
    expected, index = {}, {}
    for pair in request["pairs"]:
        cid = pair["triple_candidate_id"]
        if set(pair) != PAIR_FIELDS or cid in expected:
            raise ValueError("invalid secondary request inventory")
        expected[cid] = pair
    for record in scores["records"]:
        cid = record["triple_candidate_id"]
        if (set(record) != PAIR_FIELDS | {"biovil_raw_cosine", "status", "reason", "calibrated"}
                or cid in index or cid not in expected
                or any(record[name] != expected[cid][name] for name in PAIR_FIELDS)
                or record["calibrated"] is not False):
            raise ValueError("secondary artifact lineage differs")
        value = record["biovil_raw_cosine"]
        if value is None:
            if record["status"] != "not_available" or record["reason"] not in REASONS:
                raise ValueError("unavailable endpoint must retain reason")
        elif (not isinstance(value, (int, float)) or isinstance(value, bool)
              or not math.isfinite(value) or not -1.01 <= value <= 1.01
              or record["status"] != "computed_secondary_uncalibrated" or record["reason"] is not None):
            raise ValueError("invalid secondary cosine")
        index[cid] = record
    selected = {r["selected_candidate_id"] for r in outcomes if r["selected_candidate_id"] is not None}
    if not selected or selected != set(expected) or selected != set(index):
        raise ValueError("complete selected-union endpoints required, no cherry picking")
    result = []
    for source in outcomes:
        row = copy.deepcopy(source)
        cid = row["selected_candidate_id"]
        if cid is not None:
            record = index[cid]; snapshot = row["selected_snapshot"]
            if (row["case_id"] != record["case_id"]
                    or row["selected_ehr_sha256"] != record["ehr_sha256"]
                    or any(value != record[name] for name, value in snapshot["artifact_hashes"].items())):
                raise ValueError("selected endpoint hash/case differs")
            if snapshot["biovil_cosine_secondary_not_routing"] is not None:
                raise ValueError("do not overwrite an existing endpoint score")
            snapshot["biovil_cosine_secondary_not_routing"] = record["biovil_raw_cosine"]
            snapshot["secondary_endpoint_status"] = record["status"]
            snapshot["secondary_endpoint_unavailable_reason"] = record["reason"]
        result.append(row)
    return result


def paired_case_comparisons(outcomes, policy):
    """Descriptive paired deltas, not a significance/clinical accuracy claim."""
    validate_policy(policy)
    cases = {r["case_id"] for r in outcomes}
    groups = defaultdict(dict)
    fixed = {}
    for row in outcomes:
        case = row["case_id"]
        identity = (row["selected_ehr_sha256"], row["input_ehr_assessment_scope"])
        if fixed.setdefault(case, identity) != identity:
            raise ValueError("fixed EHR/hash scope changed across policies")
        key = (row["method"], row["model_call_budget"], case)
        seed = row["random_seed"]
        if seed in groups[key]:
            raise ValueError("duplicate case/seed endpoint trial")
        groups[key][seed] = row
    expected_keys = {(m, b, c) for m in METHODS for b in policy["model_call_budgets"] for c in cases}
    if set(groups) != expected_keys:
        raise ValueError("complete case/method/budget endpoint inventory required")
    endpoints = {}
    for key, rows in groups.items():
        expected_seeds = set(policy["random_seeds"]) if key[0] == "random" else {None}
        if set(rows) != expected_seeds:
            raise ValueError("all predeclared random replicates required")
        values = [None if r["selected_snapshot"] is None else r["selected_snapshot"]["biovil_cosine_secondary_not_routing"]
                  for r in rows.values()]
        endpoints[key] = {"cosine": None if any(v is None for v in values) else statistics.mean(values),
                          "calls": statistics.mean(r["simulated_model_calls"] for r in rows.values())}
    comparisons = []
    scopes = ["all", "explicit_fact_proxy", "no_direct_comparable_ehr_facts"]
    for scope in scopes:
        subset = sorted(c for c in cases if scope == "all" or fixed[c][1] == scope)
        for budget in policy["model_call_budgets"]:
            for baseline in ("fixed", "random", "static_rerank"):
                available = [c for c in subset if endpoints[("targeted_heuristic", budget, c)]["cosine"] is not None
                             and endpoints[(baseline, budget, c)]["cosine"] is not None]
                deltas = [endpoints[("targeted_heuristic", budget, c)]["cosine"] - endpoints[(baseline, budget, c)]["cosine"]
                          for c in available]
                comparisons.append({"ehr_evidence_subgroup": scope, "model_call_budget": budget,
                    "method": "targeted_heuristic", "baseline": baseline,
                    "all_fixed_ehr_cases": len(subset), "paired_available_ehr_cases": len(available),
                    "unavailable_case_pairs": len(subset) - len(available),
                    "distinct_ehr_sha256_in_available_pairs": len({fixed[c][0] for c in available}),
                    "mean_biovil_delta_targeted_minus_baseline": statistics.mean(deltas) if deltas else None,
                    "paired_wins": sum(d > 0 for d in deltas), "paired_ties": sum(d == 0 for d in deltas),
                    "paired_losses": sum(d < 0 for d in deltas),
                    "mean_simulated_call_delta_all_fixed_cases": statistics.mean(
                        endpoints[("targeted_heuristic", budget, c)]["calls"] - endpoints[(baseline, budget, c)]["calls"]
                        for c in subset) if subset else None,
                    "random_endpoint_averages_all_five_predeclared_replicates": True,
                    "significance_test_performed": False, "clinical_accuracy": None})
    return comparisons
