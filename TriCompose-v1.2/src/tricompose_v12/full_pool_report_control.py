"""Fixed full-pool report selection; no generation, fitting or endpoint oracle.

The existing strict cross-expert gate is unchanged. This protocol expands the
inventory, not the clinical claim: direct EHR missingness remains explicit.
"""
from collections import defaultdict
import math

from paired_case_bootstrap import paired_case_bootstrap
from .invariant_verification import _digest, _edge
from .live_receipts import PROFILE
from .report_expert_control import MODELS, compare, structure_ok
from .report_nbest import evidence_sets

SCHEMA = "tricompose-full-pool-fresh-report-control-v1"
CXR_MODELS = ("chexgenbench_sana", "chexgenbench_pixart", "roentgen_v2")
CASE_COUNT = 80
POLICY = {"schema_version": SCHEMA, "case_scope": "all_original_eighty_cases_no_filter",
    "cxr_models": list(CXR_MODELS), "cxr_seed": 0, "report_models": list(MODELS),
    "baseline_model": "cxrmate_single", "expert_order": list(MODELS),
    "choice": "first_strict_fact_preserving_eligible_expert_on_same_image",
    "uses_biovil": False, "original_winners_changed": False,
    "new_generation_allowed": False, "adaptive_repair_executed": False,
    "clinical_acceptance": False, "bootstrap_unit": "synthetic_ehr_case",
    "bootstrap_seed": 0, "bootstrap_repetitions": 2000,
    "historical_pool_is_untouched_test": False}
PAIR_FIELDS = ("case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id",
               "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")


def inventory(rows, *, expected_cases=CASE_COUNT):
    """Require the complete Cartesian grid and one fixed EHR anchor per case."""
    if (type(expected_cases) is not int or expected_cases < 1 or len(rows) != expected_cases * 12
            or len({r["triple_candidate_id"] for r in rows}) != len(rows)
            or len({r["report_candidate_id"] for r in rows}) != len(rows)):
        raise ValueError("complete unique twelve-candidate grid per fixed EHR required")
    cases, images = defaultdict(list), defaultdict(list)
    profile = set()
    for row in rows:
        if (row["cxr_model_id"] not in CXR_MODELS or row["report_model_id"] not in MODELS
                or type(row["seed"]) is not int or row["seed"] != 0):
            raise ValueError("predeclared frozen model/seed grid required")
        receipt = row["receipt"]
        evidence_sets(receipt)
        structure_ok(row)
        states = receipt["fact_states"]
        expected_edges = {"ehr_cxr": _edge(states, "ehr", "xrv"), "ehr_report": _edge(states, "ehr", "chexbert"),
            "cxr_report": _edge(states, "xrv", "chexbert")}
        if (receipt["raw_edge_readouts"] != expected_edges
                or receipt["known_ehr_facts"] != expected_edges["ehr_cxr"]["known_reference_facts"]):
            raise ValueError("receipt raw-edge arithmetic differs from its four-state facts")
        if (any(row[k] != receipt[k] for k in ("case_id", "cxr_candidate_id", "report_candidate_id", "cxr_sha256", "report_sha256"))
                or row["raw_edge_readouts"] != receipt["raw_edge_readouts"]
                or row["structure"]["source_cxr_model_id"] != row["cxr_model_id"]
                or row["structure"]["parent_cxr_candidate_id"] != row["cxr_candidate_id"]):
            raise ValueError("row, receipt and structure lineage differs")
        profile.add(tuple(receipt[k] for k in ("profile", "thresholds_sha256", "xrv_checkpoint_sha256", "chexbert_checkpoint_sha256")))
        cases[row["case_id"]].append(row)
        images[row["cxr_candidate_id"]].append(row)
    if len(cases) != expected_cases or len(images) != expected_cases * 3 or len(profile) != 1:
        raise ValueError("fixed cohort, image inventory or scorer profile differs")
    if next(iter(profile))[0] != PROFILE:
        raise ValueError("legacy classifier profile cannot enter fresh control")
    for group in cases.values():
        if (len(group) != 12 or {(r["cxr_model_id"], r["report_model_id"]) for r in group}
                != {(c, r) for c in CXR_MODELS for r in MODELS}
                or any(len({r[k] for r in group}) != 1 for k in ("ehr_sha256", "ehr_facts_sha256"))
                or len({r["receipt"]["ehr_anchor_sha256"] for r in group}) != 1
                or len({_digest([(f["finding"], f["ehr"]) for f in r["receipt"]["fact_states"]]) for r in group}) != 1):
            raise ValueError("fixed EHR anchor or complete case grid differs")
    for group in images.values():
        if (len(group) != 4 or {r["report_model_id"] for r in group} != set(MODELS)
                or any(len({r[k] for r in group}) != 1 for k in ("case_id", "cxr_model_id", "cxr_sha256"))
                or len({_digest([(f["finding"], f["xrv"]) for f in r["receipt"]["fact_states"]]) for r in group}) != 1):
            raise ValueError("fixed four-expert same-image grid differs")
    return cases, images


def freeze(rows, *, expected_cases=CASE_COUNT):
    cases, images = inventory(rows, expected_cases=expected_cases)
    choices, comparisons = [], []
    for iid, group in sorted(images.items()):
        models = {r["report_model_id"]: r for r in group}
        base = models[POLICY["baseline_model"]]
        alternatives = [compare(base, models[m]) for m in MODELS if m != POLICY["baseline_model"]]
        eligible = {r["alternative_triple_id"] for r in alternatives if r["exploratory_gate_pass"]}
        selected = next((models[m] for m in MODELS if models[m]["triple_candidate_id"] in eligible), base)
        comparisons.extend(alternatives)
        choices.append({"case_id": base["case_id"], "cxr_candidate_id": iid, "cxr_model_id": base["cxr_model_id"],
            "baseline_triple_id": base["triple_candidate_id"], "selected_triple_id": selected["triple_candidate_id"],
            "selected_model": selected["report_model_id"], "eligible_alternative_ids": sorted(eligible),
            "status": "exploratory_gate_pass" if eligible else "unresolved_baseline_retained"})
    return {"schema_version": SCHEMA, "policy": POLICY, "profile": PROFILE, "rows_sha256": _digest(rows),
        "fixed_ehr_cases": len(cases), "fixed_images": len(images), "report_candidates": len(rows),
        "choices": choices, "comparisons": comparisons, "used_biovil": False,
        "clinical_acceptance": False, "adaptive_repair_executed": False, "original_winners_changed": False}


def validate_endpoint(rows, endpoint):
    index = {r["triple_candidate_id"]: r for r in rows}
    scores = {r["triple_candidate_id"]: r for r in endpoint["records"]}
    if (len(scores) != len(endpoint["records"]) or set(scores) != set(index)
            or endpoint.get("used_for_routing") is not False
            or endpoint.get("clinical_truth_available") is not False
            or endpoint.get("historical_pool_is_untouched_test") is not False):
        raise ValueError("complete historical secondary-only score inventory required")
    for cid, score in scores.items():
        if any(score[k] != index[cid][k] for k in PAIR_FIELDS) or score["calibrated"] is not False:
            raise ValueError("secondary artifact lineage or calibration differs")
        value = score["biovil_raw_cosine"]
        if value is None:
            if score["status"] != "not_available" or not score["reason"]:
                raise ValueError("explicit endpoint NA required")
        elif (type(value) not in (int, float) or not math.isfinite(value) or not -1.01 <= value <= 1.01
                or score["status"] != "computed_secondary_uncalibrated" or score["reason"] is not None):
            raise ValueError("valid finite raw cosine required")
    return scores


def edge_totals(group, edge):
    fields = ("known_reference_facts", "comparable_facts", "supported_facts", "supported_positive",
              "supported_negative", "proxy_opposition_facts", "missing_comparisons")
    totals = {k: sum(r["raw_edge_readouts"][edge][k] for r in group) for k in fields}
    known = totals["known_reference_facts"]
    totals.update(coverage_over_known=totals["comparable_facts"] / known if known else None,
        support_over_known=totals["supported_facts"] / known if known else None,
        opposition_over_known=totals["proxy_opposition_facts"] / known if known else None)
    return totals


def summarize(rows, selection, endpoint, *, expected_cases=CASE_COUNT):
    if freeze(rows, expected_cases=expected_cases) != selection:
        raise ValueError("frozen selection changed before secondary measurement")
    scores = validate_endpoint(rows, endpoint)
    index = {r["triple_candidate_id"]: r for r in rows}
    selections = [index[c["selected_triple_id"]] for c in selection["choices"]]
    models = []
    for cxr in CXR_MODELS:
        for report in (*MODELS, "first_eligible_expert"):
            group = [r for r in (selections if report == "first_eligible_expert" else rows)
                     if r["cxr_model_id"] == cxr and (report == "first_eligible_expert" or r["report_model_id"] == report)]
            values = [scores[r["triple_candidate_id"]]["biovil_raw_cosine"] for r in group]
            models.append({"cxr_model_id": cxr, "report_policy": report, "fixed_ehr_cases": len(group),
                "report_candidates": len(group), "direct_ehr_cases": sum(r["receipt"]["known_ehr_facts"] > 0 for r in group),
                "no_direct_ehr_cases": sum(r["receipt"]["known_ehr_facts"] == 0 for r in group),
                "biovil_available_cases": sum(v is not None for v in values),
                "mean_biovil_raw_cosine": math.fsum(values) / len(values) if all(v is not None for v in values) else None,
                "official_structure_pass_count": sum(r["structure"]["section_contract_pass"] for r in group),
                "unsupported_temporal_language_count": sum(r["structure"]["unsupported_temporal_comparison_language"] for r in group),
                "raw_edge_totals": {edge: edge_totals(group, edge) for edge in ("ehr_cxr", "ehr_report", "cxr_report")},
                "clinical_accuracy": None, "clinical_acceptance": False})
    pairs = []
    for choice in selection["choices"]:
        a, b = (scores[choice[k]]["biovil_raw_cosine"] for k in ("baseline_triple_id", "selected_triple_id"))
        pairs.append({**choice, "baseline_raw_cosine": a, "selected_raw_cosine": b,
            "delta": b - a if a is not None and b is not None else None})
    case_rows, bootstraps = [], []
    for cxr in (*CXR_MODELS, "equal_weight_three_fixed_images"):
        grouped = defaultdict(list)
        for pair in pairs:
            if cxr == "equal_weight_three_fixed_images" or pair["cxr_model_id"] == cxr:
                grouped[pair["case_id"]].append(pair)
        output = []
        for case, values in sorted(grouped.items()):
            expected = 3 if cxr == "equal_weight_three_fixed_images" else 1
            if len(values) != expected:
                raise ValueError("complete per-case image outcomes required")
            a, b = ([r[k] for r in values] for k in ("baseline_raw_cosine", "selected_raw_cosine"))
            output.append({"case_id": case, "group_id": case,
                "baseline": math.fsum(a) / expected if all(v is not None for v in a) else None,
                "method": math.fsum(b) / expected if all(v is not None for v in b) else None})
        bootstraps.append({"cxr_model_id": cxr, "paired_case_bootstrap": paired_case_bootstrap(output,
            seed=POLICY["bootstrap_seed"], repetitions=POLICY["bootstrap_repetitions"]),
            "interpretation": "exploratory_historical_pool_cosine_not_untouched_clinical_efficacy"})
        case_rows.extend({"cxr_model_id": cxr, **r} for r in output)
    baseline = [r for r in rows if r["report_model_id"] == POLICY["baseline_model"] and r["cxr_model_id"] == CXR_MODELS[0]]
    summary = {"schema_version": SCHEMA, "fixed_ehr_cases": expected_cases, "fixed_images": len(selection["choices"]),
        "report_candidates": len(rows), "alternative_comparisons": len(selection["comparisons"]),
        "gate_passing_alternatives": sum(p["exploratory_gate_pass"] for p in selection["comparisons"]),
        "changed_images": sum(c["baseline_triple_id"] != c["selected_triple_id"] for c in selection["choices"]),
        "unresolved_baseline_images": sum(c["status"] == "unresolved_baseline_retained" for c in selection["choices"]),
        "direct_ehr_cases": sum(r["receipt"]["known_ehr_facts"] > 0 for r in baseline),
        "no_direct_ehr_cases": sum(r["receipt"]["known_ehr_facts"] == 0 for r in baseline),
        "available_secondary_pairs": sum(r["biovil_raw_cosine"] is not None for r in endpoint["records"]),
        "new_generation_calls": 0, "clinical_repair_success": False, "clinical_acceptance": False,
        "original_winners_changed": False, "endpoint_used_for_selection": False,
        "historical_pool_is_untouched_test": False, "independent_sampling_unit": "synthetic_ehr_case"}
    return {"summary": summary, "model_comparison": models, "image_pairs": pairs,
        "case_outcomes": case_rows, "paired_case_bootstraps": bootstraps}
