"""No-human, frozen-proxy policy experiment on an already generated bank.

This is an offline heuristic search replay, not independently verified clinical
localization or actual regeneration. The policy observes only requested slots.
Raw XRV/CheXbert drive routing; alternate readouts never alter the policy.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import product
import math

from .decision_preview import validate_fact_inventory
from .report_scope_table import EDGES, EXPLICIT, FINDINGS, edge_summary

SCHEMA = "tricompose-automatic-proxy-policy-replay-v1"
POLICY_SCHEMA = "tricompose-automatic-proxy-replay-policy-v1"
METHODS = ("fixed", "random", "static_rerank", "targeted_heuristic")


def validate_policy(policy):
    legacy = policy.get("routing_evidence") == "legacy_fourteen_raw_uncalibrated_xrv_chexbert_states"
    if (policy.get("schema_version") != POLICY_SCHEMA
            or policy.get("candidate_selection_key") != "existing_v11_lexicographic_not_full_bank_rank"
            or policy.get("routing_evidence") not in {
                "existing_eight_raw_xrv_chexbert_finding_states",
                "legacy_fourteen_raw_uncalibrated_xrv_chexbert_states"}
            or policy.get("raw_report_states_are_unverified_proxies") is not True
            or policy.get("secondary_scores_used_for_routing") is not False
            or policy.get("requires_human_feedback_for_replay") is not False
            or policy.get("clinical_accuracy_claim_allowed") is not False):
        raise ValueError("unsupported automatic diagnostic policy")
    images = policy.get("image_order", [])
    reports = policy.get("report_order", [])
    expected_images = set(product(("chexgenbench_sana", "chexgenbench_pixart", "roentgen_v2"), (0,) if legacy else (0, 1)))
    if (len(images) != len(expected_images) or any(not isinstance(x, list) or len(x) != 2 for x in images)
            or {tuple(x) for x in images} != expected_images
            or images[0] != ["chexgenbench_sana", 0]
            or any(type(x[1]) is not int for x in images)
            or len(reports) != 4 or set(reports) != {"maira2", "cxrmate_single", "llavarad", "chexagent2"}
            or reports[0] != "maira2"):
        raise ValueError("complete registered profile-specific image/four-report order required")
    for name in ("model_call_budgets", "random_seeds"):
        values = policy.get(name, [])
        if (not isinstance(values, list) or not values or len(values) != len(set(values))
                or any(type(x) is not int or x < 0 for x in values)):
            raise ValueError("invalid fixed budget/seed inventory")
    if policy.get("cost_contract") != {
            "new_cxr_generator_calls": 1, "new_cxr_xrv_calls": 1,
            "new_report_generator_calls": 1, "new_report_chexbert_calls": 1,
            "unit": "simulated_single_case_model_invocations",
            "ehr_generation_is_shared_sunk_cost": True,
            "secondary_evaluation_cost_not_free": True, "prospective_gpu_seconds": None}:
        raise ValueError("unsupported replay cost contract")


def _number(value, *, missing=None):
    if value is None:
        return missing
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("invalid cached numeric evidence")
    return float(value)


def candidate_key(candidate):
    """Existing V1.1 order, applied only to OBSERVED candidates; no bank ranks."""
    s = candidate["score_record"]["scoring"]
    totals = s["clinical_totals"]
    return (s["selection"]["hard_gate_failure_count"],
            totals["total_hard_contradiction_count"],
            -totals["ehr_direct_support_count"],
            -_number(s["edge_metrics"]["report_cxr"]["support_recall"], missing=-math.inf),
            -_number(s["modality_quality"]["report_structure_quality_score_0_1"], missing=-math.inf),
            _number(s["cost"]["known_runtime_seconds"], missing=math.inf),
            candidate["score_record"]["triple_candidate_id"])


def make_bank(score_records, fact_rows, policy):
    validate_policy(policy)
    if policy["routing_evidence"] != "existing_eight_raw_xrv_chexbert_finding_states":
        raise ValueError("legacy profile requires its separate cache adapter")
    grouped = validate_fact_inventory(fact_rows)
    index = {row["triple_candidate_id"]: row for row in score_records}
    if len(index) != len(score_records) or set(index) != set(grouped):
        raise ValueError("score/fact candidate inventories differ")
    bank, fixed = {}, {}
    for cid, facts in grouped.items():
        row = index[cid]; lineage = row["lineage"]
        if row.get("schema_version") != "tricompose-edge-specific-selection-v1.1":
            raise ValueError("completed V1.1 diagnostic score records required")
        hashes = facts[0]["artifact_hashes"]
        if (any(hashes[name] != lineage[name] for name in hashes)
                or row["case_id"] != facts[0]["case_id"]
                or lineage["cxr_candidate_id"] != facts[0]["cxr_candidate_id"]
                or lineage["report_candidate_id"] != facts[0]["report_candidate_id"]):
            raise ValueError("score/fact lineage differs")
        identity = (hashes["ehr_sha256"], hashes["ehr_facts_sha256"])
        if fixed.setdefault(row["case_id"], identity) != identity:
            raise ValueError("fixed EHR changed")
        slot = (lineage["cxr_model_id"], lineage["cxr_seed"], lineage["report_model_id"])
        case_bank = bank.setdefault(row["case_id"], {})
        if slot in case_bank:
            raise ValueError("duplicate model/seed slot")
        candidate = {"score_record": row, "facts": sorted(facts, key=lambda x: FINDINGS.index(x["finding"]))}
        candidate_key(candidate)  # Validate numerics, not a policy observation.
        # The legacy EHR-report edge also has an explicit global No-Finding
        # proxy adjustment. Preserve it as a SOURCE METRIC; do not silently
        # turn eight cached unknown finding states into negative states.
        source_edges = row["scoring"]["edge_metrics"]
        for name in ("ehr_cxr", "ehr_report", "report_cxr"):
            edge = source_edges[name]
            for field in ("support_count", "contradiction_count", "known_reference_fact_count"):
                if type(edge[field]) is not int or edge[field] < 0:
                    raise ValueError("invalid source-edge count")
        for source_name, fact_name in (("ehr_cxr", "ehr_cxr"), ("report_cxr", "cxr_report")):
            if (source_edges[source_name]["support_count"] != sum(f["relations"]["raw"][fact_name] == "support" for f in facts)
                    or source_edges[source_name]["contradiction_count"] != sum(f["relations"]["raw"][fact_name] == "opposition" for f in facts)):
                raise ValueError("shared finding counts differ from source edge")
        expected = {
            "total_hard_contradiction_count": sum(edge["contradiction_count"] for edge in source_edges.values()),
            "total_direct_support_count": sum(edge["support_count"] for edge in source_edges.values()),
            "ehr_direct_support_count": sum(source_edges[name]["support_count"] for name in ("ehr_cxr", "ehr_report")),
            "total_known_reference_fact_count": sum(edge["known_reference_fact_count"] for edge in source_edges.values())}
        for key in expected:
            value = row["scoring"]["clinical_totals"][key]
            if type(value) is not int or value != expected[key]:
                raise ValueError("cached proxy counts differ from source edge totals")
        gate_count = row["scoring"]["selection"]["hard_gate_failure_count"]
        if type(gate_count) is not int or gate_count < 0:
            raise ValueError("invalid artifact gate count")
        case_bank[slot] = candidate
    expected = {(model, seed, report) for model, seed in policy["image_order"] for report in policy["report_order"]}
    if not bank or any(set(values) != expected for values in bank.values()):
        raise ValueError("complete six-image/four-report bank required per fixed EHR")
    return bank


def action_signal(candidate):
    """Heuristic routing, NOT truth attribution. No unseen candidate is inspected."""
    row = candidate["score_record"]
    if not row["scoring"]["modality_quality"]["cxr_basic_validity_pass"]:
        return "regenerate_cxr", ["invalid_image_metadata"], []
    if row["scoring"]["selection"]["hard_gate_failure_count"]:
        return "switch_report_model", ["candidate_artifact_gate_failed"], []
    facts = candidate["facts"]
    image_conflicts = [f["evidence_id"] for f in facts if f["relations"]["raw"]["ehr_cxr"] == "opposition"]
    if image_conflicts:
        return "regenerate_cxr", ["direct_ehr_xrv_opposition_heuristic_not_confirmed_fault"], image_conflicts
    report_conflicts = [f["evidence_id"] for f in facts
                       if any(f["relations"]["raw"][edge] == "opposition" for edge in ("ehr_report", "cxr_report"))]
    omissions = [f["evidence_id"] for f in facts
                if (f["states"]["ehr"] in EXPLICIT or f["states"]["xrv"] == "positive")
                and f["states"]["chexbert"] not in EXPLICIT]
    if report_conflicts or omissions:
        return "switch_report_model", ["report_opposition_or_missing_known_fact_heuristic"], sorted(set(report_conflicts + omissions))
    ehr_known = [f for f in facts if f["states"]["ehr"] in EXPLICIT]
    image_known = [f for f in facts if f["states"]["xrv"] in EXPLICIT]
    comparable = [f for f in image_known if f["states"]["chexbert"] in EXPLICIT]
    ehr_complete = all(f["states"]["ehr"] == f["states"]["xrv"] == f["states"]["chexbert"] for f in ehr_known)
    if image_known and comparable and ehr_complete and row["scoring"]["clinical_totals"]["total_hard_contradiction_count"] == 0:
        return "stop_proxy_satisfied", ["all_observed_explicit_constraints_satisfied_not_clinical_acceptance"], []
    return "regenerate_cxr", ["missing_comparable_image_evidence"], []


def replay_case(case_bank, policy, method, budget, *, random_seed=0):
    if method not in METHODS or type(budget) is not int or budget < 0:
        raise ValueError("invalid replay method/budget")
    images = [tuple(x) for x in policy["image_order"]]
    reports = list(policy["report_order"])
    ordered = [(*image, report) for image in images for report in reports]
    case = next(iter(case_bank.values()))["score_record"]["case_id"]
    if method == "random":
        ordered.sort(key=lambda slot: hashlib.sha256(json_slot(case, random_seed, slot).encode()).hexdigest())
    observed, charged_images, trace, calls = {}, set(), [], Counter()
    terminal = "candidate_inventory_exhausted"

    def observe(slot, action, reasons, evidence_ids):
        image = slot[:2]
        cost = 2 + (0 if image in charged_images else 2)
        if sum(calls.values()) + cost > budget:
            return False
        candidate = case_bank[slot]
        if image not in charged_images:
            calls.update(cxr_generator=1, xrv=1)
            charged_images.add(image)
        calls.update(report_generator=1, chexbert=1)
        observed[slot] = candidate
        trace.append({"step": len(trace), "action": action, "request_slot": list(slot),
            "observed_candidate_id": candidate["score_record"]["triple_candidate_id"],
            "reason_codes": list(reasons), "trigger_evidence_ids": list(evidence_ids),
            "charged_model_calls": cost, "cumulative_model_calls": sum(calls.values())})
        return True

    if method != "targeted_heuristic":
        for slot in ordered[:1] if method == "fixed" else ordered:
            if not observe(slot, "generate_candidate", ["predeclared_order"], []):
                terminal = "model_call_budget_exhausted"; break
        if method == "fixed" and observed: terminal = "fixed_path_complete"
    else:
        slot, action, reasons, ids = ordered[0], "generate_candidate", ["predeclared_initial_path"], []
        while True:
            if not observe(slot, action, reasons, ids):
                terminal = "model_call_budget_exhausted"; break
            action, reasons, ids = action_signal(observed[slot])
            if action == "stop_proxy_satisfied":
                terminal = action; break
            current_image = slot[:2]
            if action == "switch_report_model":
                available = [(*current_image, report) for report in reports if (*current_image, report) not in observed]
                if available:
                    slot = available[0]; continue
                reasons = [*reasons, "report_inventory_exhausted_try_another_image"]
            unseen_images = [image for image in images if image not in charged_images]
            if not unseen_images:
                terminal = "targeted_neighborhood_exhausted"; break
            slot, action = (*unseen_images[0], reports[0]), "regenerate_cxr"
    eligible = [candidate for candidate in observed.values()
                if candidate["score_record"]["scoring"]["selection"]["hard_gate_failure_count"] == 0]
    selected = (observed[slot] if method == "targeted_heuristic" and terminal == "stop_proxy_satisfied"
                else min(eligible, key=candidate_key) if eligible else None)
    return {"case_id": case, "method": method, "random_seed": random_seed if method == "random" else None,
        "model_call_budget": budget, "simulated_calls": dict(calls), "simulated_model_calls": sum(calls.values()),
        "observed_candidates": len(observed), "observed_images": len(charged_images),
        "terminal_reason": terminal, "selected_candidate_id": None if selected is None else selected["score_record"]["triple_candidate_id"],
        "selected_proxy_stop_conditions_met": selected is not None and action_signal(selected)[0] == "stop_proxy_satisfied",
        "selected_ehr_sha256": next(iter(case_bank.values()))["score_record"]["lineage"]["ehr_sha256"],
        "selected_snapshot": None if selected is None else evaluation_snapshot(selected),
        "action_trace": trace, "actual_regeneration_executed": False,
        "clinical_fault_confirmed": False, "clinical_acceptance": False}


def json_slot(case, seed, slot):
    # Locale-independent, score-free random ordering. No answer key is used.
    return f"automatic-replay-v1|{case}|{seed}|{slot[0]}|{slot[1]}|{slot[2]}"


def evaluation_snapshot(candidate):
    row, facts = candidate["score_record"], candidate["facts"]
    totals = row["scoring"]["clinical_totals"]
    secondary = row.get("secondary_scores", {}).get("biovil_report_cxr", {
        "status": "not_available", "calibrated": False, "raw_cosine": None})
    if (not isinstance(secondary, dict) or set(secondary) != {"status", "calibrated", "raw_cosine"}
            or secondary["calibrated"] is not False
            or secondary["status"] not in {"not_available", "computed_secondary_uncalibrated"}
            or (secondary["status"] == "not_available") != (secondary["raw_cosine"] is None)):
        raise ValueError("invalid secondary BioViL cache contract")
    scope_available = candidate.get("alternate_scope_readout_available", True)
    return {"artifact_hashes": dict(facts[0]["artifact_hashes"]),
        "cxr_model_id": row["lineage"]["cxr_model_id"], "cxr_seed": row["lineage"]["cxr_seed"],
        "report_model_id": row["lineage"]["report_model_id"],
        "optimization_proxy": {key: totals[key] for key in ("total_hard_contradiction_count", "total_direct_support_count", "total_known_reference_fact_count", "clinical_balance_score_0_100")},
        "report_structure_quality": row["scoring"]["modality_quality"]["report_structure_quality_score_0_1"],
        "biovil_cosine_secondary_not_routing": _number(secondary["raw_cosine"]),
        "raw_edge_readouts_optimization_proxy": {edge: edge_summary(facts, edge, "raw") for edge in EDGES},
        "source_edge_metrics_optimization_proxy": {
            edge: {key: row["scoring"]["edge_metrics"][source][key]
                   for key in ("support_count", "contradiction_count", "known_reference_fact_count")}
            for edge, source in (("ehr_cxr", "ehr_cxr"), ("ehr_report", "ehr_report"), ("cxr_report", "report_cxr"))},
        "guarded_edge_readouts_not_routing": {
            edge: edge_summary(facts, edge, "scoped") if scope_available else None for edge in EDGES},
        "guarded_edge_readout_status": "cached_scope_available" if scope_available else "not_computed_for_legacy_cohort",
        "ehr_assessment_scope": "explicit_fact_proxy" if any(f["states"]["ehr"] in EXPLICIT for f in facts) else "no_direct_comparable_ehr_facts",
        "raw_report_labels_unverified": True, "independent_clinical_truth_available": False}
