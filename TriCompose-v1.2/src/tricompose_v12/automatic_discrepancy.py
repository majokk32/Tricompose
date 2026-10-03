"""Descriptive frozen-cache diagnostics, not clinical fault localization.

Polarity arithmetic splits explicit positive support from explicit negative
support; unknown/uncertain never become negative. Actual choices/rules do not
change. Same-image report contrasts control the image, not clinical truth.
All common-image pairs are from the endpoint-SCORED selected union, not the
whole bank; reports/pairs sharing an image are dependent observations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
import statistics

from .automatic_replay import candidate_key
from .automatic_secondary import paired_case_comparisons
from .legacy_replay_adapter import FINDINGS, STATES

EDGES = {"ehr_cxr": ("ehr", "xrv"), "ehr_report": ("ehr", "chexbert"),
         "cxr_report": ("xrv", "chexbert")}
KEY_COMPONENTS = ("artifact_gate", "source_proxy_opposition", "direct_ehr_support",
                  "report_classifier_support", "report_structure", "known_runtime", "id_tie_break")
SCHEMA = "tricompose-frozen-automatic-policy-discrepancy-v1"


def polarity_profile(facts, edge):
    if edge not in EDGES or len(facts) != len(FINDINGS) or {f["finding"] for f in facts} != set(FINDINGS):
        raise ValueError("complete legacy finding inventory required")
    left_key, right_key = EDGES[edge]
    count = Counter()
    for fact in facts:
        left, right = fact["states"][left_key], fact["states"][right_key]
        if left not in STATES or right not in STATES:
            raise ValueError("invalid four-state polarity evidence")
        count[f"reference_{left}"] += 1
        if left in {"positive", "negative"}:
            if right == left:
                count[f"support_{left}"] += 1
            elif right in {"positive", "negative"}:
                count[f"opposition_{left}_reference"] += 1
            else:
                count[f"missing_candidate_on_{left}_reference"] += 1
        elif right == "positive":
            count[f"candidate_positive_on_{left}_reference_not_hallucination"] += 1
    fields = [*(f"reference_{s}" for s in sorted(STATES)),
              "support_positive", "support_negative", "opposition_positive_reference",
              "opposition_negative_reference", "missing_candidate_on_positive_reference",
              "missing_candidate_on_negative_reference",
              "candidate_positive_on_unknown_reference_not_hallucination",
              "candidate_positive_on_uncertain_reference_not_hallucination"]
    output = {name: count[name] for name in fields}
    for state in ("positive", "negative"):
        if output[f"reference_{state}"] != sum(output[name] for name in (
                f"support_{state}", f"opposition_{state}_reference", f"missing_candidate_on_{state}_reference")):
            raise ValueError("polarity count arithmetic differs")
    return output


def point(outcome, candidate):
    row, facts, snapshot = candidate["score_record"], candidate["facts"], outcome["selected_snapshot"]
    if (row["case_id"] != outcome["case_id"] or row["triple_candidate_id"] != outcome["selected_candidate_id"]
            or any(value != row["lineage"][name] for name, value in snapshot["artifact_hashes"].items())
            or snapshot["optimization_proxy"] != {key: row["scoring"]["clinical_totals"][key]
                for key in snapshot["optimization_proxy"]}):
        raise ValueError("endpoint and raw diagnostic evidence lineage differ")
    profiles = {edge: polarity_profile(facts, edge) for edge in EDGES}
    support_positive = sum(p["support_positive"] for p in profiles.values())
    support_negative = sum(p["support_negative"] for p in profiles.values())
    raw_opposition = sum(p["opposition_positive_reference"] + p["opposition_negative_reference"] for p in profiles.values())
    totals = row["scoring"]["clinical_totals"]
    if support_positive + support_negative != totals["total_direct_support_count"]:
        raise ValueError("support polarity differs from source metric")
    global_normal_adjustment = totals["total_hard_contradiction_count"] - raw_opposition
    if global_normal_adjustment < 0:
        raise ValueError("legacy contradiction adjustment cannot be negative")
    cosine = snapshot["biovil_cosine_secondary_not_routing"]
    if cosine is not None and (isinstance(cosine, bool) or not isinstance(cosine, (int, float)) or not math.isfinite(cosine)):
        raise ValueError("invalid secondary endpoint")
    return {"clinical_balance_source_proxy": totals["clinical_balance_score_0_100"],
        "source_support_count": totals["total_direct_support_count"],
        "source_opposition_count": totals["total_hard_contradiction_count"],
        "source_known_reference_count": totals["total_known_reference_fact_count"],
        "raw_support_positive": support_positive, "raw_support_negative": support_negative,
        "raw_opposition": raw_opposition, "legacy_global_normal_adjustment": global_normal_adjustment,
        "negative_share_of_explicit_support": support_negative / (support_positive + support_negative)
            if support_positive + support_negative else None,
        "xrv_positive_pathology_count": sum(f["states"]["xrv"] == "positive" for f in facts
            if f["finding"] not in {"no_finding", "support_devices"}),
        "report_positive_pathology_count": sum(f["states"]["chexbert"] == "positive" for f in facts
            if f["finding"] not in {"no_finding", "support_devices"}),
        "report_no_finding_positive": int(next(f for f in facts if f["finding"] == "no_finding")["states"]["chexbert"] == "positive"),
        "direct_ehr_reference_count": sum(f["states"]["ehr"] in {"positive", "negative"} for f in facts),
        "biovil_cosine_secondary": cosine, "polarity_by_edge": profiles}


def artifact_change(before, after):
    old, new = before["selected_snapshot"]["artifact_hashes"], after["selected_snapshot"]["artifact_hashes"]
    if (old["ehr_sha256"], old["ehr_facts_sha256"]) != (new["ehr_sha256"], new["ehr_facts_sha256"]):
        raise ValueError("fixed EHR changed in contrast")
    image = old["cxr_sha256"] != new["cxr_sha256"]
    report = old["report_sha256"] != new["report_sha256"]
    return ("both_image_and_report_changed" if image and report else "image_changed_same_report_artifact" if image
            else "same_image_report_changed" if report else "same_artifacts")


def _delta(before, after, field):
    return None if before[field] is None or after[field] is None else after[field] - before[field]


def _mean(values):
    available = [v for v in values if v is not None]
    return statistics.mean(available) if available else None


def analyze(outcomes, bank, policy):
    paired_case_comparisons(outcomes, policy)  # Complete fixed case/method/seed/budget invariants.
    index = {(case, c["score_record"]["triple_candidate_id"]): c for case, grid in bank.items() for c in grid.values()}
    points, candidates, groups = {}, {}, defaultdict(dict)
    for outcome in outcomes:
        groups[(outcome["case_id"], outcome["model_call_budget"])][(outcome["method"], outcome["random_seed"])] = outcome
        cid = outcome["selected_candidate_id"]
        if cid is None:
            continue
        key = (outcome["case_id"], cid)
        if key not in index:
            raise ValueError("selected diagnostic candidate absent")
        view = point(outcome, index[key])
        if points.setdefault(key, view) != view:
            raise ValueError("shared candidate has inconsistent endpoint evidence")
        candidates[key] = index[key]
    contrasts = []
    for (case, budget), trials in sorted(groups.items()):
        target = trials[("targeted_heuristic", None)]
        for baseline in ("fixed", "static_rerank"):
            before = trials[(baseline, None)]
            entry = {"case_id": case, "ehr_evidence_subgroup": target["input_ehr_assessment_scope"],
                "model_call_budget": budget, "baseline": baseline,
                "baseline_candidate_id": before["selected_candidate_id"], "targeted_candidate_id": target["selected_candidate_id"],
                "simulated_call_delta": target["simulated_model_calls"] - before["simulated_model_calls"],
                "artifact_change": "unavailable_selected_candidate", "baseline_view": None, "targeted_view": None,
                "clinical_fault_confirmed": False, "natural_faulty_modality": None}
            if before["selected_candidate_id"] is not None and target["selected_candidate_id"] is not None:
                a, b = points[(case, before["selected_candidate_id"])], points[(case, target["selected_candidate_id"])]
                entry.update(artifact_change=artifact_change(before, target), baseline_view=a, targeted_view=b,
                    baseline_models={key: before["selected_snapshot"][key] for key in ("cxr_model_id", "report_model_id")},
                    targeted_models={key: target["selected_snapshot"][key] for key in ("cxr_model_id", "report_model_id")})
                entry["deltas"] = {field: _delta(a, b, field) for field in a if field != "polarity_by_edge"}
                entry["delta_support_positive_plus_negative_verified"] = (
                    entry["deltas"]["raw_support_positive"] + entry["deltas"]["raw_support_negative"]
                    == entry["deltas"]["source_support_count"])
            contrasts.append(entry)
    slices = []
    sliced = defaultdict(list)
    for row in contrasts:
        for scope in ("all", row["ehr_evidence_subgroup"]):
            for change in ("all_changes", row["artifact_change"]):
                sliced[(row["baseline"], row["model_call_budget"], scope, change)].append(row)
    for (baseline, budget, scope, change), rows in sorted(sliced.items()):
        valid = [r for r in rows if r["targeted_view"] is not None]
        secondary = [r for r in valid if r["deltas"]["biovil_cosine_secondary"] is not None]
        slices.append({"baseline": baseline, "model_call_budget": budget,
            "ehr_evidence_subgroup": scope, "artifact_change": change,
            "fixed_ehr_cases": len(rows), "selected_pair_cases": len(valid),
            "biovil_paired_available_cases": len(secondary), "biovil_unavailable_cases": len(rows)-len(secondary),
            "mean_biovil_delta_on_available_pairs": _mean(r["deltas"]["biovil_cosine_secondary"] for r in secondary),
            "mean_source_proxy_balance_delta_all_selected_pairs": _mean(r["deltas"]["clinical_balance_source_proxy"] for r in valid),
            "mean_source_proxy_balance_delta_on_same_biovil_pairs": _mean(r["deltas"]["clinical_balance_source_proxy"] for r in secondary),
            "sum_raw_positive_support_delta_all_selected_pairs": sum(r["deltas"]["raw_support_positive"] for r in valid),
            "sum_raw_negative_support_delta_all_selected_pairs": sum(r["deltas"]["raw_support_negative"] for r in valid),
            "sum_source_opposition_delta_all_selected_pairs": sum(r["deltas"]["source_opposition_count"] for r in valid),
            "sum_xrv_positive_pathology_delta_all_selected_pairs": sum(r["deltas"]["xrv_positive_pathology_count"] for r in valid),
            "sum_report_positive_pathology_delta_all_selected_pairs": sum(r["deltas"]["report_positive_pathology_count"] for r in valid),
            "mean_baseline_negative_support_share": _mean(r["baseline_view"]["negative_share_of_explicit_support"] for r in valid),
            "mean_targeted_negative_support_share": _mean(r["targeted_view"]["negative_share_of_explicit_support"] for r in valid),
            "proxy_balance_gain_and_biovil_loss_cases": sum(
                r["deltas"]["clinical_balance_source_proxy"] is not None
                and r["deltas"]["clinical_balance_source_proxy"] > 0
                and r["deltas"]["biovil_cosine_secondary"] < 0 for r in secondary),
            "clinical_accuracy": None, "causal_error_attribution": None})
    # Observational controls: both reports describe the SAME synthetic image.
    images = defaultdict(list)
    for key, c in candidates.items():
        images[(key[0], c["score_record"]["lineage"]["cxr_sha256"])].append(key)
    same_image_pairs = []
    for (case, image_hash), keys in sorted(images.items()):
        for a, b in combinations(sorted(keys), 2):
            ka, kb = candidate_key(candidates[a]), candidate_key(candidates[b])
            preferred, other = (a, b) if ka < kb else (b, a)
            kp, ko = candidate_key(candidates[preferred]), candidate_key(candidates[other])
            component = next((KEY_COMPONENTS[i] for i in range(len(kp)) if kp[i] != ko[i]), "equal_key")
            p, o = points[preferred], points[other]
            lp, lo = candidates[preferred]["score_record"]["lineage"], candidates[other]["score_record"]["lineage"]
            same_image_pairs.append({"case_id": case, "cxr_sha256": image_hash,
                "preferred_candidate_id": preferred[1], "other_candidate_id": other[1],
                "preferred_report_model": lp["report_model_id"], "other_report_model": lo["report_model_id"],
                "same_report_artifact": lp["report_sha256"] == lo["report_sha256"],
                "key_deciding_component": component,
                "biovil_delta_preferred_minus_other": _delta(o, p, "biovil_cosine_secondary"),
                "raw_positive_support_delta": _delta(o, p, "raw_support_positive"),
                "raw_negative_support_delta": _delta(o, p, "raw_support_negative"),
                "source_opposition_delta": _delta(o, p, "source_opposition_count"),
                "clinical_fault_confirmed": False, "pair_is_independent_patient": False})
    controls = []
    by_component = defaultdict(list)
    for row in same_image_pairs:
        by_component["all_components"].append(row)
        by_component[row["key_deciding_component"]].append(row)
    for component, rows in sorted(by_component.items()):
        valid = [r for r in rows if r["biovil_delta_preferred_minus_other"] is not None]
        case_values = defaultdict(list)
        for r in valid: case_values[r["case_id"]].append(r["biovil_delta_preferred_minus_other"])
        controls.append({"key_deciding_component": component,
            "observed_same_image_report_pairs": len(rows), "available_pair_readouts": len(valid),
            "distinct_ehr_cases": len({r["case_id"] for r in rows}),
            "distinct_case_image_groups": len({(r["case_id"], r["cxr_sha256"]) for r in rows}),
            "available_cases": len(case_values),
            "case_mean_then_cohort_mean_biovil_delta": _mean(statistics.mean(v) for v in case_values.values()),
            "preferred_has_lower_biovil_pair_count": sum(r["biovil_delta_preferred_minus_other"] < 0 for r in valid),
            "duplicate_report_artifact_pair_count": sum(r["same_report_artifact"] for r in rows),
            "sampling_scope": "scored_selected_union_not_exhaustive_bank",
            "pair_independence_assumed": False, "clinical_accuracy": None})
    freq = defaultdict(list)
    for row in outcomes: freq[(row["method"], row["model_call_budget"])].append(row)
    frequencies = []
    for (method, budget), rows in sorted(freq.items()):
        counter = Counter((r["selected_snapshot"]["cxr_model_id"], r["selected_snapshot"]["report_model_id"])
                          if r["selected_snapshot"] else ("unavailable", "unavailable") for r in rows)
        for (image, report), count in sorted(counter.items()):
            frequencies.append({"method": method, "model_call_budget": budget,
                "cxr_model": image, "report_model": report, "selection_trials": count,
                "all_trials": len(rows), "fixed_ehr_cases": len({r["case_id"] for r in rows}),
                "selection_trial_fraction": count / len(rows), "is_clinical_model_ranking": False})
    return {"schema_version": SCHEMA, "contrasts": contrasts, "action_slices": slices,
        "same_image_pairs": same_image_pairs, "same_image_controls": controls,
        "model_selection_frequencies": frequencies, "new_model_calls": 0,
        "rules_or_winners_changed": False, "independent_clinical_truth_available": False}
