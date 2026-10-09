"""One prospective image retry with fixed EHR/text and diagnostic proxies.

Pure policy only. An EHR/classifier disagreement triggers a HEURISTIC retry,
not a diagnosis of which modality is clinically wrong. Report agreement is
never counted as independent evidence that the image is wrong.
"""
from copy import deepcopy

from .invariant_verification import _digest, _edge, EXPLICIT
from .legacy_replay_adapter import FINDINGS
from .live_plan import anchor_from_record
from .report_expert_control import structure_ok
from .report_nbest import evidence_sets, RISK_FLAGS

SCHEMA = "tricompose-bounded-cxr-regeneration-smoke-v1"
POLICY = {
    "schema_version": SCHEMA,
    "cohort_rule": "first_two_sorted_ehr_anchors_with_explicit_enabled_head_facts",
    "cohort_selection_uses_image_report_or_endpoint_values": False,
    "baseline_cxr_model": "roentgen_v2",
    "baseline_seed": 0,
    "retry_seed": 1,
    "report_model": "cxrmate_single",
    "call_budget_per_case": 4,
    "max_retries_per_operation": 0,
    "timeout_seconds_per_worker_process": 120,
    "report_models": ["cxrmate_single"],
    "maximum_new_images_per_case": 1,
    "maximum_new_reports_per_case": 1,
    "route": "direct_ehr_xrv_explicit_opposition_only_heuristic",
    "gate": "fixed_ehr_image_gain_all_edge_fact_preservation_quality_nonregression_v1",
    "changes_existing_ehr_or_prompts": False,
    "uses_biovil": False,
    "clinical_acceptance": False,
    "clinical_fault_localization_validated": False,
    "training_allowed": False,
}


def validate_policy(policy):
    if policy != POLICY:
        raise ValueError("exact preregistered bounded retry policy required")


def choose_cases(anchor_rows, enabled_heads):
    """EHR/availability stratification only; no image/report/score arguments."""
    if (len(anchor_rows) != 80 or len({a["anchor"]["case_id"] for a in anchor_rows}) != 80
            or set(enabled_heads) - set(FINDINGS) or not enabled_heads):
        raise ValueError("complete eighty-EHR cohort and declared scorer mask required")
    eligible, direct = [], []
    for row in sorted(anchor_rows, key=lambda r: r["anchor"]["case_id"]):
        anchor = anchor_from_record(row["anchor"])
        if anchor.sha256 != row["ehr_anchor_sha256"]:
            raise ValueError("immutable EHR anchor digest differs")
        explicit = {name for name, state, _ in anchor.findings if state in EXPLICIT}
        if explicit:
            direct.append(anchor.case_id)
        if explicit & set(enabled_heads):
            eligible.append(row)
    if len(eligible) < 2:
        raise ValueError("not enough explicit enabled-head EHR anchors; never pad facts")
    return eligible[:2], {
        "original_ehr_cases": 80, "direct_ehr_cases": len(direct),
        "enabled_head_ehr_cases": len(eligible), "smoke_cases": 2,
        "other_ehr_cases_not_regenerated": 78,
        "case_selection_uses_outcomes": False,
        "scope": "development_ehr_evidence_stratum_not_representative_efficacy_cohort",
    }


def validate_row(row):
    evidence_sets(row["receipt"])
    structure_ok(row)
    for key in ("case_id", "cxr_candidate_id", "report_candidate_id", "cxr_sha256", "report_sha256"):
        if row[key] != row["receipt"][key]:
            raise ValueError("row/receipt identity differs")
    facts = row["receipt"]["fact_states"]
    expected = {"ehr_cxr": _edge(facts, "ehr", "xrv"),
                "ehr_report": _edge(facts, "ehr", "chexbert"),
                "cxr_report": _edge(facts, "xrv", "chexbert")}
    if (row["raw_edge_readouts"] != expected or row["receipt"]["raw_edge_readouts"] != expected
            or row["receipt"]["known_ehr_facts"] != expected["ehr_cxr"]["known_reference_facts"]):
        raise ValueError("raw edge readouts must be recomputed from source-bound states")
    return facts


def edge_sets(row):
    facts = validate_row(row)
    result = {}
    for edge, left, right in (("ehr_cxr", "ehr", "xrv"), ("ehr_report", "ehr", "chexbert"),
                              ("cxr_report", "xrv", "chexbert")):
        known = {f["finding"] for f in facts if f[left] in EXPLICIT}
        comparable = {f["finding"] for f in facts if f[left] in EXPLICIT and f[right] in EXPLICIT}
        support = {f["finding"] for f in facts if f["finding"] in comparable and f[left] == f[right]}
        positive = {f["finding"] for f in facts if f["finding"] in support and f[left] == "positive"}
        result[edge] = {"known": known, "comparable": comparable, "support": support,
                        "positive_support": positive, "opposition": comparable - support}
    return result


def route(baseline):
    edges = edge_sets(baseline)
    image = edges["ehr_cxr"]
    if image["opposition"]:
        action, reason = "regenerate_cxr", "explicit_direct_ehr_xrv_proxy_opposition"
    elif not image["known"]:
        action, reason = "abstain", "no_direct_ehr_facts"
    elif image["known"] - image["comparable"]:
        action, reason = "abstain", "missing_direct_ehr_image_comparison"
    else:
        action, reason = "stop", "no_direct_ehr_image_opposition_detected"
    return {"action": action, "reason": reason,
            "trigger_finding_ids": sorted(image["opposition"]),
            "clinical_fault_location": None, "clinical_repair_success": False,
            "correlated_reports_used_as_independent_votes": False,
            "endpoint_used": False}


def alternate_request(original):
    if (original["model_id"] != POLICY["baseline_cxr_model"]
            or type(original["seed"]) is not int or original["seed"] != POLICY["baseline_seed"]):
        raise ValueError("declared original RoentGen seed-zero request required")
    result = deepcopy(original)
    result["seed"] = POLICY["retry_seed"]
    result["request_id"] = f"cxrreq_{result['case_id']}_{result['model_id']}_s{result['seed']:06d}"
    # Every other field, including final-tokenizer text hash, remains identical.
    if {k: v for k, v in result.items() if k not in ("seed", "request_id")} != {
            k: v for k, v in original.items() if k not in ("seed", "request_id")}:
        raise ValueError("retry changed fixed EHR/prompt or generation contract")
    return result


def compare(baseline, alternative):
    """Cross-image gate; never call same-image comparison with a changed image.

Require actual improvement of the fixed EHR/image edge, with no lost known
comparisons, no new opposition on ANY edge, preserved direct-EHR support and
previous positive report/image support. Baseline image labels are conservative
constraints, not assumed ground truth. Negative support alone is not a gain.
"""
    left, right = edge_sets(baseline), edge_sets(alternative)
    a, b = baseline["receipt"], alternative["receipt"]
    fixed = ("case_id", "ehr_anchor_sha256", "thresholds_sha256", "xrv_checkpoint_sha256",
             "chexbert_checkpoint_sha256", "profile")
    if (any(a[k] != b[k] for k in fixed)
            or any(baseline[k] != alternative[k] for k in ("ehr_sha256", "ehr_facts_sha256"))
            or [(f["finding"], f["ehr"]) for f in a["fact_states"]]
               != [(f["finding"], f["ehr"]) for f in b["fact_states"]]):
        raise ValueError("fixed EHR/scorer profile changed during regeneration")
    duplicate = a["cxr_sha256"] == b["cxr_sha256"] or a["cxr_candidate_id"] == b["cxr_candidate_id"]
    lost, new, gained, removed = {}, {}, {}, {}
    for edge in left:
        # EHR edges preserve all support; image/report preserves positive
        # support, not arbitrary negative agreement from a possibly bad image.
        required_support = "positive_support" if edge == "cxr_report" else "support"
        lost[edge] = {key: sorted(left[edge][key] - right[edge][key])
                      for key in ("comparable", required_support)}
        new[edge] = sorted(right[edge]["opposition"] - left[edge]["opposition"])
        gained[edge] = sorted(right[edge]["support"] - left[edge]["support"])
        removed[edge] = sorted(left[edge]["opposition"] - right[edge]["opposition"])
    s, t = baseline["structure"], alternative["structure"]
    quality = (structure_ok(alternative) and all(not t[k] or s[k] for k in RISK_FLAGS)
               and t["repeated_sentence_count"] <= s["repeated_sentence_count"]
               and t["repeated_4gram_ratio"] <= s["repeated_4gram_ratio"])
    gain = bool(gained["ehr_cxr"] or removed["ehr_cxr"])
    loss = any(values for edge in lost.values() for values in edge.values())
    reasons = []
    if duplicate: reasons.append("duplicate_image_not_regeneration_diversity")
    if loss: reasons.append("lost_supported_or_comparable_fact_ids")
    if any(new.values()): reasons.append("new_explicit_proxy_opposition")
    if not quality: reasons.append("official_report_structure_failed_or_common_risk_worse")
    if not gain: reasons.append("no_strict_fixed_ehr_image_evidence_gain")
    return {"baseline_triple_id": baseline["triple_candidate_id"],
            "alternative_triple_id": alternative["triple_candidate_id"],
            "lost_fact_ids": lost, "new_opposition_fact_ids": new,
            "gained_support_fact_ids": gained, "removed_opposition_fact_ids": removed,
            "report_structure_and_risk_no_worse": quality, "duplicate_image": duplicate,
            "exploratory_gate_pass": bool(gain and not loss and not any(new.values()) and quality and not duplicate),
            "reasons": reasons, "clinical_repair_success": False, "endpoint_used": False}


def decision(baseline, static, alternative=None, failure_reason=None):
    """Retain the static result unless retry strictly passes BOTH references."""
    validate_row(baseline)
    validate_row(static)
    fixed = ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_candidate_id", "cxr_sha256")
    if (any(baseline[k] != static[k] for k in fixed)
            or any(baseline["receipt"][k] != static["receipt"][k] for k in
                   ("ehr_anchor_sha256", "profile", "thresholds_sha256", "xrv_checkpoint_sha256", "chexbert_checkpoint_sha256"))
            or [(f["finding"], f["ehr"], f["xrv"]) for f in baseline["receipt"]["fact_states"]]
               != [(f["finding"], f["ehr"], f["xrv"]) for f in static["receipt"]["fact_states"]]):
        raise ValueError("static reference must use the same EHR/image/scorer profile")
    action = route(baseline)
    comparisons = []
    if alternative is not None:
        if action["action"] != "regenerate_cxr":
            raise ValueError("unrequested candidate cannot be accepted")
        comparisons = [compare(baseline, alternative)]
        if static["triple_candidate_id"] != baseline["triple_candidate_id"]:
            comparisons.append(compare(static, alternative))
    passed = bool(comparisons) and all(c["exploratory_gate_pass"] for c in comparisons)
    selected = alternative if passed else static
    return {"schema_version": SCHEMA, "case_id": baseline["case_id"],
            "action": action, "baseline_triple_id": baseline["triple_candidate_id"],
            "static_triple_id": static["triple_candidate_id"],
            "selected_triple_id": selected["triple_candidate_id"],
            "alternative_triple_id": alternative["triple_candidate_id"] if alternative else None,
            "status": "exploratory_retry_gate_pass" if passed else "unresolved_static_retained",
            "failure_reason": failure_reason, "comparisons": comparisons,
            "selection_uses_secondary_endpoint": False, "clinical_repair_success": False,
            "original_winners_changed": False}
