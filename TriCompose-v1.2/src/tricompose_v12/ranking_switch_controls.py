"""Frozen-history ranking ablation and invariant cached-EHR accounting.

No policy execution, model calls, clinical verdict or old artifact mutation.
Alternate endpoint values/availability are attached after ranking only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import statistics

from .automatic_discrepancy import point
from .automatic_replay import action_signal, candidate_key, evaluation_snapshot
from .fixed_image_control import _pair
from .legacy_replay_adapter import FINDINGS, STATES

SCHEMA = "tricompose-ranking-invariant-switch-control-v1"
POLICY_SCHEMA = "tricompose-ranking-switch-control-policy-v1"
METHODS = ("fixed", "report_only_static", "report_only_targeted", "static_rerank", "targeted_heuristic")
EXPLICIT = {"positive", "negative"}


def validate_control(config):
    if config != {
        "schema_version": POLICY_SCHEMA, "control_id": "pool80_same_observations_remove_runtime_v1",
        "source_methods": list(METHODS), "model_call_budgets": [4, 8, 12, 20, 30],
        "runtime_ablation": "remove_runtime_only_keep_original_clinical_prefix_and_id",
        "candidate_visibility": "same_original_observed_slots_only",
        "original_proxy_stop_selection": "retain_current_candidate_and_terminal_reason",
        "routing_or_stopping_changed": False, "secondary_scores_used_for_selection": False,
        "missing_endpoints": "retain_na_never_substitute_scored_candidate",
        "image_switch_reference": "unchanged_explicit_cached_ehr_states",
        "unknown_uncertain_are_negative": False, "weak_context_is_hard_constraint": False,
        "clinical_accuracy_claim_allowed": False,
        "cohort_role": "previously_inspected_development_not_held_out"}:
        raise ValueError("frozen same-history ranking/invariant-reference control required")


def no_runtime_key(candidate):
    key = candidate_key(candidate)
    return (*key[:5], key[6])


def endpoint_index(original, bank):
    index = {c["score_record"]["triple_candidate_id"]: c for grid in bank.values() for c in grid.values()}
    endpoints = {}
    for row in original:
        cid = row["selected_candidate_id"]
        if cid is None: continue
        if cid not in index: raise ValueError("endpoint candidate absent")
        point(row, index[cid])
        snap = row["selected_snapshot"]
        data = {k: snap[k] for k in ("biovil_cosine_secondary_not_routing", "secondary_endpoint_status",
                                     "secondary_endpoint_unavailable_reason")}
        if endpoints.setdefault(cid, data) != data:
            raise ValueError("shared endpoint evidence differs")
    return endpoints


def observed_candidates(row, grid):
    index = {c["score_record"]["triple_candidate_id"]: c for c in grid.values()}
    observed, images, ledger = [], set(), Counter()
    for step, event in enumerate(row["action_trace"]):
        cid = event["observed_candidate_id"]
        if cid not in index or cid in {c["score_record"]["triple_candidate_id"] for c in observed}:
            raise ValueError("foreign/duplicate observed candidate")
        candidate = index[cid]; lineage = candidate["score_record"]["lineage"]
        slot = [lineage["cxr_model_id"], lineage["cxr_seed"], lineage["report_model_id"]]
        image = tuple(slot[:2]); cost = 2 + (0 if image in images else 2)
        if event["request_slot"] != slot or event["step"] != step or event["charged_model_calls"] != cost:
            raise ValueError("observed slot/cost differs")
        if image not in images: ledger.update(cxr_generator=1, xrv=1)
        images.add(image); ledger.update(report_generator=1, chexbert=1)
        if event["cumulative_model_calls"] != sum(ledger.values()):
            raise ValueError("cumulative observed cost differs")
        observed.append(candidate)
    if (row["observed_candidates"] != len(observed) or row["observed_images"] != len(images)
            or row["simulated_calls"] != dict(ledger) or row["simulated_model_calls"] != sum(ledger.values())
            or sum(ledger.values()) > row["model_call_budget"]):
        raise ValueError("original observed inventory/ledger differs")
    return observed


def ablate_trial(row, grid, endpoints):
    observed = observed_candidates(row, grid)
    eligible = [c for c in observed if c["score_record"]["scoring"]["selection"]["hard_gate_failure_count"] == 0]
    old_id = row["selected_candidate_id"]
    old = next((c for c in observed if c["score_record"]["triple_candidate_id"] == old_id), None)
    if old_id is not None and old is None: raise ValueError("original winner was not observed")
    if row["terminal_reason"] == "stop_proxy_satisfied":
        if old is None or old is not observed[-1] or action_signal(old)[0] != "stop_proxy_satisfied":
            raise ValueError("original proxy-stop choice differs")
        selected = old
    else:
        baseline = min(eligible, key=candidate_key) if eligible else None
        if (baseline is None) != (old is None) or baseline is not None and baseline is not old:
            raise ValueError("original best-observed choice differs")
        selected = min(eligible, key=no_runtime_key) if eligible else None
    if old is not None:
        point(row, old)
        if candidate_key(old)[:5] != candidate_key(selected)[:5]:
            raise ValueError("runtime ablation changed earlier clinical prefix")
    result = copy.deepcopy(row)
    result.update(method=row["method"] + "_no_runtime", source_method=row["method"],
        original_selected_candidate_id=old_id,
        selected_candidate_id=selected["score_record"]["triple_candidate_id"] if selected else None,
        selected_snapshot=evaluation_snapshot(selected) if selected else None,
        selected_proxy_stop_conditions_met=selected is not None and action_signal(selected)[0] == "stop_proxy_satisfied",
        routing_or_stopping_changed=False, same_observed_history=True)
    if selected is not None:
        cid = result["selected_candidate_id"]
        result["selected_snapshot"].update(copy.deepcopy(endpoints.get(cid, {
            "biovil_cosine_secondary_not_routing": None,
            "secondary_endpoint_status": "not_scored_in_frozen_endpoint_union",
            "secondary_endpoint_unavailable_reason": "new_control_choice_not_previously_scored"})))
    return result


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def ranking_ablation(rows, bank, original_endpoints, config):
    validate_control(config)
    expected = {(case, m, b) for case in bank for m in METHODS for b in config["model_call_budgets"]}
    if len(rows) != len(expected) or {(r["case_id"], r["method"], r["model_call_budget"]) for r in rows} != expected:
        raise ValueError("complete fixed cohort/control inventory required")
    endpoints = endpoint_index(original_endpoints, bank)
    output, comparisons, pending = [], [], {}
    for row in sorted(rows, key=lambda r: (r["case_id"], r["method"], r["model_call_budget"])):
        grid = bank[row["case_id"]]
        after = ablate_trial(row, grid, endpoints)
        output.append(after)
        index = {c["score_record"]["triple_candidate_id"]: c for c in grid.values()}
        old_id, new_id = row["selected_candidate_id"], after["selected_candidate_id"]
        contrast = {"case_id": row["case_id"], "source_method": row["method"],
            "model_call_budget": row["model_call_budget"], "ehr_evidence_subgroup": row["input_ehr_assessment_scope"],
            "original_candidate_id": old_id, "no_runtime_candidate_id": new_id,
            "winner_changed": old_id != new_id, "simulated_call_delta": 0,
            "same_image_reference": None, "selected_pair_available": old_id is not None and new_id is not None,
            "biovil_delta": None, "known_runtime_delta": None, "old_runtime_available": False,
            "new_runtime_available": False, "earlier_quality_prefix_unchanged": True,
            "clinical_accuracy": None}
        if old_id is not None and new_id is not None:
            a, b = point(row, index[old_id]), point(after, index[new_id])
            x, y = a["biovil_cosine_secondary"], b["biovil_cosine_secondary"]
            contrast["biovil_delta"] = None if x is None or y is None else y-x
            la, lb = index[old_id]["score_record"]["lineage"], index[new_id]["score_record"]["lineage"]
            contrast["same_image_reference"] = la["cxr_sha256"] == lb["cxr_sha256"]
            ta, tb = (index[cid]["score_record"]["scoring"]["cost"]["known_runtime_seconds"] for cid in (old_id, new_id))
            contrast.update(old_runtime_available=ta is not None, new_runtime_available=tb is not None,
                known_runtime_delta=None if ta is None or tb is None else tb-ta)
            if new_id not in endpoints: pending[new_id] = _pair(index[new_id])
        comparisons.append(contrast)
    groups = defaultdict(list)
    for r in comparisons:
        for scope in ("all", r["ehr_evidence_subgroup"]):
            groups[(scope, r["source_method"], r["model_call_budget"])].append(r)
    summary = []
    for (scope, method, budget), group in sorted(groups.items()):
        paired = [r for r in group if r["biovil_delta"] is not None]
        changed = [r for r in group if r["winner_changed"]]
        summary.append({"ehr_evidence_subgroup": scope, "source_method": method, "model_call_budget": budget,
            "all_fixed_ehr_cases": len(group), "changed_winner_cases": len(changed),
            "changed_image_cases": sum(r["same_image_reference"] is False for r in group),
            "paired_available_biovil_cases": len(paired), "unavailable_biovil_case_pairs": len(group)-len(paired),
            "mean_biovil_delta_no_runtime_minus_original": _mean(r["biovil_delta"] for r in paired),
            "paired_biovil_wins": sum(r["biovil_delta"] > 0 for r in paired),
            "paired_biovil_losses": sum(r["biovil_delta"] < 0 for r in paired),
            "paired_biovil_ties": sum(r["biovil_delta"] == 0 for r in paired),
            "changed_winners_with_paired_runtime": sum(r["known_runtime_delta"] is not None for r in changed),
            "mean_known_runtime_delta_changed_pairs": _mean(r["known_runtime_delta"] for r in changed),
            "simulated_call_delta": 0, "same_observed_history": True,
            "routing_or_stopping_changed": False, "clinical_accuracy": None})
    return output, comparisons, summary, [pending[k] for k in sorted(pending)]


def invariant_profile(candidate):
    facts = candidate["facts"]
    if len(facts) != len(FINDINGS) or {f["finding"] for f in facts} != set(FINDINGS):
        raise ValueError("complete invariant finding inventory required")
    counts = Counter()
    vector = {}
    for f in facts:
        ehr, image, report = (f["states"][k] for k in ("ehr", "xrv", "chexbert"))
        if any(s not in STATES for s in (ehr, image, report)): raise ValueError("invalid invariant four-state evidence")
        vector[f["finding"]] = ehr
        known = ehr in EXPLICIT
        if known:
            counts["known_ehr_facts"] += 1
            for name, value in (("image", image), ("report", report)):
                state = "support" if value == ehr else "opposition" if value in EXPLICIT else "missing"
                counts[f"direct_{name}_{state}"] += 1
            counts["all_three_support"] += image == report == ehr
        if image in EXPLICIT and image == report:
            counts[f"report_cxr_{image}_support_" + ("with_direct_ehr" if known else "without_direct_ehr")] += 1
    names = ["known_ehr_facts", "all_three_support",
        *(f"direct_{side}_{state}" for side in ("image", "report") for state in ("support", "opposition", "missing")),
        *(f"report_cxr_{s}_support_{coverage}" for s in ("positive", "negative")
          for coverage in ("with_direct_ehr", "without_direct_ehr"))]
    profile = {name: counts[name] for name in names}
    for side in ("image", "report"):
        if sum(profile[f"direct_{side}_{state}"] for state in ("support", "opposition", "missing")) != profile["known_ehr_facts"]:
            raise ValueError("invariant reference denominator differs")
    return vector, profile


def invariant_switches(rows, bank, config):
    validate_control(config)
    index = {(r["case_id"], r["method"], r["model_call_budget"]): r for r in rows}
    expected = {(c, m, b) for c in bank for m in METHODS for b in config["model_call_budgets"]}
    if len(index) != len(rows) or set(index) != expected: raise ValueError("complete invariant control inventory required")
    output = []
    for case, grid in sorted(bank.items()):
        candidates = {c["score_record"]["triple_candidate_id"]: c for c in grid.values()}
        for budget in config["model_call_budgets"]:
            before = index[(case, "fixed", budget)]
            for method in ("static_rerank", "targeted_heuristic"):
                after = index[(case, method, budget)]
                entry = {"case_id": case, "method": method, "model_call_budget": budget,
                    "ehr_evidence_subgroup": after["input_ehr_assessment_scope"], "image_changed": None,
                    "selected_pair_available": False, "known_ehr_facts": None, "before": None, "after": None,
                    "deltas": None, "biovil_delta": None, "source_balance_delta": None,
                    "invariant_reference_support_rate_before": None, "invariant_reference_support_rate_after": None,
                    "clinical_fault_confirmed": False}
                if before["selected_candidate_id"] is not None and after["selected_candidate_id"] is not None:
                    a, b = candidates[before["selected_candidate_id"]], candidates[after["selected_candidate_id"]]
                    pa, pb = point(before, a), point(after, b)
                    ha, hb = a["score_record"]["lineage"], b["score_record"]["lineage"]
                    if any(ha[k] != hb[k] for k in ("ehr_sha256", "ehr_facts_sha256")):
                        raise ValueError("invariant EHR hashes changed")
                    va, ca = invariant_profile(a); vb, cb = invariant_profile(b)
                    if va != vb: raise ValueError("invariant cached EHR vector changed")
                    entry.update(image_changed=ha["cxr_sha256"] != hb["cxr_sha256"], selected_pair_available=True,
                        known_ehr_facts=ca["known_ehr_facts"], before=ca, after=cb,
                        deltas={k: cb[k]-ca[k] for k in ca})
                    if not entry["image_changed"] and any(entry["deltas"][f"direct_image_{k}"] for k in ("support", "opposition", "missing")):
                        raise ValueError("same image has different classifier reference")
                    known = ca["known_ehr_facts"]
                    if known:
                        entry.update(invariant_reference_support_rate_before=ca["all_three_support"]/known,
                            invariant_reference_support_rate_after=cb["all_three_support"]/known)
                    for field, key in (("biovil_delta", "biovil_cosine_secondary"), ("source_balance_delta", "clinical_balance_source_proxy")):
                        entry[field] = None if pa[key] is None or pb[key] is None else pb[key]-pa[key]
                output.append(entry)
    groups = defaultdict(list)
    for r in output:
        for scope in ("all", r["ehr_evidence_subgroup"]):
            for changed in ("all", "changed" if r["image_changed"] else "unchanged_or_unavailable"):
                groups[(scope, r["method"], r["model_call_budget"], changed)].append(r)
    summary = []
    for (scope, method, budget, change), group in sorted(groups.items()):
        valid = [r for r in group if r["selected_pair_available"]]
        known = [r for r in valid if r["known_ehr_facts"]]
        record = {"ehr_evidence_subgroup": scope, "method": method, "model_call_budget": budget,
            "image_change_group": change, "fixed_ehr_cases": len(group), "selected_pair_cases": len(valid),
            "cases_with_direct_ehr_constraints": len(known), "cases_without_direct_ehr_constraints": len(valid)-len(known),
            "sum_invariant_ehr_reference_facts": sum(r["known_ehr_facts"] for r in known),
            "image_switch_with_direct_support_gain_cases": sum(r["deltas"]["direct_image_support"] > 0 for r in known),
            "image_switch_with_direct_support_loss_cases": sum(r["deltas"]["direct_image_support"] < 0 for r in known),
            "sum_direct_image_support_delta_constrained_cases": sum(r["deltas"]["direct_image_support"] for r in known) if known else None,
            "sum_direct_image_opposition_delta_constrained_cases": sum(r["deltas"]["direct_image_opposition"] for r in known) if known else None,
            "sum_direct_report_support_delta_constrained_cases": sum(r["deltas"]["direct_report_support"] for r in known) if known else None,
            "sum_all_three_support_delta_constrained_cases": sum(r["deltas"]["all_three_support"] for r in known) if known else None,
            "mean_invariant_reference_support_rate_before_constrained_cases": _mean(r["invariant_reference_support_rate_before"] for r in known),
            "mean_invariant_reference_support_rate_after_constrained_cases": _mean(r["invariant_reference_support_rate_after"] for r in known),
            "biovil_paired_available_cases": sum(r["biovil_delta"] is not None for r in valid),
            "mean_biovil_delta_on_available_pairs": _mean(r["biovil_delta"] for r in valid),
            "mean_source_balance_delta_all_selected_pairs": _mean(r["source_balance_delta"] for r in valid),
            "clinical_accuracy": None, "natural_faulty_modality": None}
        for state in ("positive", "negative"):
            for coverage in ("with_direct_ehr", "without_direct_ehr"):
                key = f"report_cxr_{state}_support_{coverage}"
                record["sum_" + key + "_delta_all_selected_pairs"] = sum(r["deltas"][key] for r in valid)
        summary.append(record)
    return output, summary
