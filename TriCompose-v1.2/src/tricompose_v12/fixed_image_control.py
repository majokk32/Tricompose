"""Exploratory frozen-image cache control, not clinical repair or new inference.

Report selection never receives alternate endpoint scores. Missing cached
endpoints are attached AFTER selection and never cause candidate substitution.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import statistics

from .automatic_discrepancy import point
from .automatic_replay import action_signal, candidate_key, evaluation_snapshot, validate_policy
from .automatic_secondary import PAIR_FIELDS, paired_case_comparisons

SCHEMA = "tricompose-fixed-image-report-control-v1"
POLICY_SCHEMA = "tricompose-fixed-image-report-control-policy-v1"
REPORT_METHODS = ("report_only_static", "report_only_targeted")
BASELINES = ("fixed", "targeted_heuristic", "static_rerank")


def validate_control(control, policy):
    validate_policy(policy)
    expected = {
        "schema_version": POLICY_SCHEMA,
        "control_id": "pool80_sana0_report_only_development_v1",
        "fixed_image_slot": ["chexgenbench_sana", 0],
        "report_order": policy["report_order"],
        "model_call_budgets": policy["model_call_budgets"],
        "report_methods": list(REPORT_METHODS),
        "candidate_selection_key": "unchanged_existing_v11_lexicographic",
        "image_selection_used_report_or_endpoint_scores": False,
        "secondary_scores_used_for_selection": False,
        "missing_endpoints": "retain_na_never_substitute_scored_candidate",
        "clinical_accuracy_claim_allowed": False,
        "cohort_role": "previously_inspected_development_not_held_out"}
    if (control != expected or policy["image_order"][0] != control["fixed_image_slot"]
            or policy["routing_evidence"] != "legacy_fourteen_raw_uncalibrated_xrv_chexbert_states"):
        raise ValueError("unchanged frozen legacy fixed-image control required")


def replay_fixed_image(case_bank, policy, control, method, budget):
    validate_control(control, policy)
    if method not in REPORT_METHODS or type(budget) is not int or budget < 0:
        raise ValueError("invalid fixed-image method or budget")
    image = tuple(control["fixed_image_slot"])
    slots = [(*image, report) for report in control["report_order"]]
    if any(slot not in case_bank for slot in slots):
        raise ValueError("complete registered fixed-image report inventory required")
    initial = case_bank[slots[0]]
    lineage = initial["score_record"]["lineage"]
    identity = tuple(lineage[k] for k in ("ehr_sha256", "ehr_facts_sha256", "cxr_sha256"))
    case = initial["score_record"]["case_id"]
    # Structural checks are not observations of another report's scores.
    for slot in slots:
        row = case_bank[slot]["score_record"]
        if (row["case_id"] != case or tuple(row["lineage"][k] for k in
                ("ehr_sha256", "ehr_facts_sha256", "cxr_sha256")) != identity):
            raise ValueError("fixed EHR/image changed across report slots")
    observed, trace, calls = [], [], Counter()
    terminal = "fixed_image_report_inventory_exhausted"
    for slot in slots:
        cost = 4 if not observed else 2
        if sum(calls.values()) + cost > budget:
            terminal = "model_call_budget_exhausted"
            break
        if not observed:
            calls.update(cxr_generator=1, xrv=1)
        calls.update(report_generator=1, chexbert=1)
        candidate = case_bank[slot]
        observed.append(candidate)
        trace.append({"step": len(trace), "action": "generate_candidate" if len(trace) == 0 else "switch_report_model",
            "request_slot": list(slot), "observed_candidate_id": candidate["score_record"]["triple_candidate_id"],
            "reason_codes": ["frozen_fixed_image_report_order"],
            "charged_model_calls": cost, "cumulative_model_calls": sum(calls.values())})
        if method == "report_only_targeted":
            action, reasons, ids = action_signal(candidate)
            trace[-1]["next_heuristic_signal"] = action
            trace[-1]["signal_reason_codes"] = reasons
            trace[-1]["signal_evidence_ids"] = ids
            if action == "stop_proxy_satisfied":
                terminal = action
                break
            if action == "regenerate_cxr":
                terminal = "fixed_image_cxr_change_blocked"
                break
    eligible = [c for c in observed if c["score_record"]["scoring"]["selection"]["hard_gate_failure_count"] == 0]
    selected = (observed[-1] if terminal == "stop_proxy_satisfied"
                else min(eligible, key=candidate_key) if eligible else None)
    return {"case_id": case, "method": method, "random_seed": None, "model_call_budget": budget,
        "simulated_calls": dict(calls), "simulated_model_calls": sum(calls.values()),
        "observed_candidates": len(observed), "observed_images": int(bool(observed)),
        "terminal_reason": terminal,
        "selected_candidate_id": selected["score_record"]["triple_candidate_id"] if selected else None,
        "selected_ehr_sha256": lineage["ehr_sha256"],
        "selected_snapshot": evaluation_snapshot(selected) if selected else None,
        "selected_proxy_stop_conditions_met": selected is not None and action_signal(selected)[0] == "stop_proxy_satisfied",
        "action_trace": trace, "control_fixed_image_sha256": lineage["cxr_sha256"],
        "image_selection_used_report_or_endpoint_scores": False,
        "actual_regeneration_executed": False, "clinical_fault_confirmed": False,
        "clinical_acceptance": False}


def _pair(candidate):
    row = candidate["score_record"]
    return {"case_id": row["case_id"], "triple_candidate_id": row["triple_candidate_id"],
        **{k: row["lineage"][k] for k in PAIR_FIELDS - {"case_id", "triple_candidate_id"}}}


def build_trials(original, bank, policy, control):
    validate_control(control, policy)
    paired_case_comparisons(original, policy)  # Exact old case/method/budget/seed inventory.
    if {r["case_id"] for r in original} != set(bank):
        raise ValueError("entire original case inventory required")
    index = {c["score_record"]["triple_candidate_id"]: c for grid in bank.values() for c in grid.values()}
    if len(index) != sum(len(grid) for grid in bank.values()):
        raise ValueError("candidate IDs must be globally unique")
    endpoints = {}
    for row in original:
        cid = row["selected_candidate_id"]
        if cid is None:
            continue
        if cid not in index:
            raise ValueError("original selected candidate absent from frozen bank")
        point(row, index[cid])  # Check selected states, hashes and original totals.
        snap = row["selected_snapshot"]
        endpoint = {"biovil_cosine_secondary_not_routing": snap["biovil_cosine_secondary_not_routing"],
            "secondary_endpoint_status": snap["secondary_endpoint_status"],
            "secondary_endpoint_unavailable_reason": snap["secondary_endpoint_unavailable_reason"]}
        if endpoints.setdefault(cid, endpoint) != endpoint:
            raise ValueError("inconsistent cached endpoints for the same candidate")
    rows = [copy.deepcopy(r) for r in original if r["method"] in BASELINES]
    pending = {}
    for case, grid in sorted(bank.items()):
        for budget in control["model_call_budgets"]:
            for method in REPORT_METHODS:
                row = replay_fixed_image(grid, policy, control, method, budget)
                row["input_ehr_assessment_scope"] = evaluation_snapshot(
                    grid[(*control["fixed_image_slot"], control["report_order"][0])])["ehr_assessment_scope"]
                cid = row["selected_candidate_id"]
                if cid is not None:
                    snap = row["selected_snapshot"]
                    if cid in endpoints:
                        snap.update(copy.deepcopy(endpoints[cid]))
                    else:
                        snap.update(biovil_cosine_secondary_not_routing=None,
                            secondary_endpoint_status="not_scored_in_frozen_endpoint_union",
                            secondary_endpoint_unavailable_reason="new_control_choice_not_previously_scored")
                        pending[cid] = _pair(index[cid])
                rows.append(row)
    rows.sort(key=lambda r: (r["case_id"], r["model_call_budget"], r["method"]))
    return rows, [pending[cid] for cid in sorted(pending)]


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def contrasts(rows, bank, control):
    expected = {(case, budget, method) for case in bank for budget in control["model_call_budgets"]
                for method in (*BASELINES, *REPORT_METHODS)}
    index = {(r["case_id"], r["model_call_budget"], r["method"]): r for r in rows}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError("complete case/budget/control inventory required")
    candidate_index = {c["score_record"]["triple_candidate_id"]: c for grid in bank.values() for c in grid.values()}
    views = {key: point(row, candidate_index[row["selected_candidate_id"]])
             for key, row in index.items() if row["selected_candidate_id"] is not None}
    output = []
    for case in sorted(bank):
        for budget in control["model_call_budgets"]:
            for method in REPORT_METHODS:
                for baseline in BASELINES:
                    a, b = index[(case, budget, baseline)], index[(case, budget, method)]
                    if (a["selected_ehr_sha256"], a["input_ehr_assessment_scope"]) != (
                            b["selected_ehr_sha256"], b["input_ehr_assessment_scope"]):
                        raise ValueError("fixed EHR/hash scope differs across controls")
                    va, vb = views.get((case, budget, baseline)), views.get((case, budget, method))
                    record = {"case_id": case, "model_call_budget": budget, "method": method, "baseline": baseline,
                        "ehr_evidence_subgroup": b["input_ehr_assessment_scope"],
                        "baseline_candidate_id": a["selected_candidate_id"], "control_candidate_id": b["selected_candidate_id"],
                        "simulated_call_delta": b["simulated_model_calls"] - a["simulated_model_calls"],
                        "selected_pair_available": va is not None and vb is not None,
                        "same_image_reference": None, "clinical_fault_confirmed": False,
                        "control_endpoint_status": None if vb is None else b["selected_snapshot"]["secondary_endpoint_status"],
                        "deltas": None}
                    if va is not None and vb is not None:
                        ah, bh = a["selected_snapshot"]["artifact_hashes"], b["selected_snapshot"]["artifact_hashes"]
                        if any(ah[k] != bh[k] for k in ("ehr_sha256", "ehr_facts_sha256")):
                            raise ValueError("EHR changed in paired control")
                        record["same_image_reference"] = ah["cxr_sha256"] == bh["cxr_sha256"]
                        if record["same_image_reference"] and va["polarity_by_edge"]["ehr_cxr"] != vb["polarity_by_edge"]["ehr_cxr"]:
                            raise ValueError("same-image classifier reference changed")
                        record["deltas"] = {field: None if va[field] is None or vb[field] is None else vb[field]-va[field]
                            for field in ("biovil_cosine_secondary", "clinical_balance_source_proxy", "raw_support_positive",
                                "raw_support_negative", "source_opposition_count", "xrv_positive_pathology_count")}
                    output.append(record)
    groups = defaultdict(list)
    for row in output:
        for scope in ("all", row["ehr_evidence_subgroup"]):
            groups[(scope, row["method"], row["model_call_budget"], row["baseline"])].append(row)
    aggregates = []
    for (scope, method, budget, baseline), group in sorted(groups.items()):
        valid = [r for r in group if r["selected_pair_available"]]
        paired = [r for r in valid if r["deltas"]["biovil_cosine_secondary"] is not None]
        ds = [r["deltas"]["biovil_cosine_secondary"] for r in paired]
        aggregates.append({"ehr_evidence_subgroup": scope, "method": method, "model_call_budget": budget,
            "baseline": baseline, "all_fixed_ehr_cases": len(group), "selected_pair_cases": len(valid),
            "paired_available_biovil_cases": len(paired), "unavailable_biovil_case_pairs": len(group)-len(paired),
            "mean_biovil_delta_on_paired_available_cases": _mean(ds),
            "paired_biovil_wins": sum(d > 0 for d in ds), "paired_biovil_ties": sum(d == 0 for d in ds),
            "paired_biovil_losses": sum(d < 0 for d in ds),
            "mean_source_proxy_balance_delta_all_selected_pairs": _mean(r["deltas"]["clinical_balance_source_proxy"] for r in valid),
            "mean_source_proxy_balance_delta_same_biovil_pairs": _mean(r["deltas"]["clinical_balance_source_proxy"] for r in paired),
            "sum_positive_support_delta_all_selected_pairs": sum(r["deltas"]["raw_support_positive"] for r in valid),
            "sum_negative_support_delta_all_selected_pairs": sum(r["deltas"]["raw_support_negative"] for r in valid),
            "same_image_reference_pairs": sum(r["same_image_reference"] for r in valid),
            "mean_simulated_call_delta_all_fixed_cases": _mean(r["simulated_call_delta"] for r in group),
            "clinical_accuracy": None, "significance_test_performed": False})
    return output, aggregates
