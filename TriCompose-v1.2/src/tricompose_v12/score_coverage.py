"""Metadata-only score coverage and same-image tie diagnostics, never selection.

The fresh eight-enabled-head and historical fourteen-head profiles MUST be
analyzed separately. These counts cannot establish clinical correctness.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
import statistics

from .invariant_verification import _edge, _HASH, _ID
from .legacy_replay_adapter import FINDINGS, PROFILE as LEGACY_PROFILE, STATES
from .live_receipts import PROFILE as FRESH_PROFILE

SCHEMA = "tricompose-score-coverage-diagnostic-v1"
PREFIX_FIELDS = {
    FRESH_PROFILE: ("opposition_plus_missing_positive", "minus_supported_positive"),
    LEGACY_PROFILE: ("artifact_gate", "source_opposition", "minus_ehr_support",
                    "minus_source_image_report_support_recall"),
}
HASH_FIELDS = ("ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")
ID_FIELDS = ("case_id", "cxr_candidate_id", "report_candidate_id", "report_model_id", "triple_candidate_id")


def _mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def _fixed(index, key, value):
    if index.setdefault(key, value) != value:
        raise ValueError("shared artifact or fixed EHR evidence differs")


def analyze(records, profile):
    """Return private diagnostic tables; do not choose, filter or modify a row."""
    if profile not in PREFIX_FIELDS or not records:
        raise ValueError("one explicitly named nonempty scorer profile required")
    if len(records) > 960:
        raise ValueError("bounded cached inventory required")
    ehr, images, reports, image_hash_states, report_hash_states, groups = {}, {}, {}, {}, {}, defaultdict(list)
    seen, rows = set(), []
    for record in sorted(records, key=lambda r: (r["case_id"], r["triple_candidate_id"])):
        if (record.get("profile") != profile
                or any(not isinstance(record.get(k), str) or not _ID.fullmatch(record[k]) for k in ID_FIELDS)
                or any(not isinstance(record.get(k), str) or not _HASH.fullmatch(record[k]) for k in HASH_FIELDS)):
            raise ValueError("single-profile opaque IDs and artifact hashes required")
        cid = record["triple_candidate_id"]
        if cid in seen:
            raise ValueError("duplicate candidate pair")
        seen.add(cid)
        states = record["fact_states"]
        if (len(states) != len(FINDINGS) or [f["finding"] for f in states] != list(FINDINGS)
                or any(f.get(k) not in STATES for f in states for k in ("ehr", "xrv", "chexbert"))):
            raise ValueError("complete ordered four-state evidence required")
        prefix = record["source_primary_prefix"]
        if (not isinstance(prefix, (list, tuple)) or len(prefix) != len(PREFIX_FIELDS[profile])
                or any(x is not None and (type(x) not in (int, float) or not math.isfinite(x)) for x in prefix)):
            raise ValueError("explicit finite source prefix or unavailable component required")
        cosine = record["biovil_raw_cosine"]
        if cosine is None:
            if not isinstance(record.get("endpoint_unavailable_reason"), str) or not record["endpoint_unavailable_reason"]:
                raise ValueError("unavailable endpoint requires a reason")
        elif (type(cosine) not in (int, float) or not math.isfinite(cosine) or not -1.01 <= cosine <= 1.01
              or record.get("endpoint_unavailable_reason") is not None):
            raise ValueError("finite uncalibrated endpoint required, not probability")
        case, image, report = (record[k] for k in ("case_id", "cxr_candidate_id", "report_candidate_id"))
        ev, iv, rv = (tuple(f[k] for f in states) for k in ("ehr", "xrv", "chexbert"))
        _fixed(ehr, case, (record["ehr_sha256"], record["ehr_facts_sha256"], ev))
        _fixed(images, (case, image), (record["cxr_sha256"], iv))
        _fixed(image_hash_states, record["cxr_sha256"], iv)
        _fixed(reports, (case, report), (image, record["report_sha256"], rv))
        _fixed(report_hash_states, record["report_sha256"], rv)
        edges = {"ehr_cxr": _edge(states, "ehr", "xrv"),
                 "ehr_report": _edge(states, "ehr", "chexbert"),
                 "cxr_report": _edge(states, "xrv", "chexbert")}
        edge = edges["cxr_report"]
        missing_positive = sum(f["xrv"] == "positive" and f["chexbert"] in {"unknown", "uncertain"} for f in states)
        if profile == FRESH_PROFILE and list(prefix) != [edge["proxy_opposition_facts"] + missing_positive,
                                                       -edge["supported_positive"]]:
            raise ValueError("fresh diagnostic prefix differs from frozen source rule")
        rows.append({**{k: record[k] for k in (*ID_FIELDS, *HASH_FIELDS)},
            "profile": profile, "source_primary_prefix": list(prefix), "raw_edge_readouts": edges,
            "image_positive_reference_facts": iv.count("positive"),
            "report_explicit_facts": rv.count("positive") + rv.count("negative"),
            "missing_positive_image_facts": missing_positive,
            "biovil_raw_cosine": cosine, "endpoint_unavailable_reason": record["endpoint_unavailable_reason"],
            "clinical_accuracy": None})
        groups[(case, image)].append(rows[-1])
    pairs = []
    for (case, image), members in sorted(groups.items()):
        if len({r["report_candidate_id"] for r in members}) != len(members):
            raise ValueError("duplicate report within image group")
        for a, b in combinations(members, 2):
            available = a["biovil_raw_cosine"] is not None and b["biovil_raw_cosine"] is not None
            prefix_available = all(x is not None for r in (a, b) for x in r["source_primary_prefix"])
            pairs.append({"case_id": case, "cxr_candidate_id": image,
                "left_triple_id": a["triple_candidate_id"], "right_triple_id": b["triple_candidate_id"],
                "left_report_model": a["report_model_id"], "right_report_model": b["report_model_id"],
                "source_primary_prefix_tied": a["source_primary_prefix"] == b["source_primary_prefix"] if prefix_available else None,
                "same_report_artifact": a["report_sha256"] == b["report_sha256"],
                "both_endpoints_available": available,
                "absolute_biovil_gap": abs(a["biovil_raw_cosine"] - b["biovil_raw_cosine"]) if available else None,
                "clinical_difference_confirmed": False, "independent_patient_pair": False})
    tied = [p for p in pairs if p["source_primary_prefix_tied"]]
    available_tied = [p for p in tied if p["both_endpoints_available"]]
    case_gaps = defaultdict(list)
    for p in available_tied:
        case_gaps[p["case_id"]].append(p["absolute_biovil_gap"])
    edge_rows = [r["raw_edge_readouts"]["cxr_report"] for r in rows]
    model_rows = []
    for model in sorted({r["report_model_id"] for r in rows}):
        members = [r for r in rows if r["report_model_id"] == model]
        model_rows.append({"report_model": model, "report_candidates": len(members),
            "no_comparable_image_report_candidates": sum(r["raw_edge_readouts"]["cxr_report"]["comparable_facts"] == 0 for r in members),
            "explicit_report_facts": sum(r["report_explicit_facts"] for r in members),
            "endpoint_available_candidates": sum(r["biovil_raw_cosine"] is not None for r in members),
            "clinical_accuracy": None})
    def state_counts(vectors):
        count = Counter(state for vector in vectors for state in vector)
        return {state: count[state] for state in sorted(STATES)}
    summary = {"schema_version": SCHEMA, "profile": profile,
        "fixed_ehr_cases": len(ehr), "image_groups": len(images), "report_candidates": len(rows),
        "unique_image_artifact_hashes": len(image_hash_states), "unique_report_artifact_hashes": len(report_hash_states),
        "no_direct_ehr_cases": sum(not any(s in {"positive", "negative"} for s in v[2]) for v in ehr.values()),
        "images_with_no_explicit_reference": sum(not any(s in {"positive", "negative"} for s in v[1]) for v in images.values()),
        "images_with_explicit_labels_all_negative": sum("negative" in v[1] and "positive" not in v[1] for v in images.values()),
        "images_with_positive_reference": sum("positive" in v[1] for v in images.values()),
        "ehr_state_counts_one_vector_per_case": state_counts(v[2] for v in ehr.values()),
        "image_state_counts_one_vector_per_image_id": state_counts(v[1] for v in images.values()),
        "report_state_counts_one_vector_per_report_id": state_counts(v[2] for v in reports.values()),
        "no_comparable_image_report_candidates": sum(e["comparable_facts"] == 0 for e in edge_rows),
        "zero_opposition_but_no_comparable_candidates": sum(e["comparable_facts"] == 0 and e["proxy_opposition_facts"] == 0 for e in edge_rows),
        "same_image_report_pairs": len(pairs), "primary_prefix_tied_pairs": len(tied),
        "primary_prefix_unavailable_pairs": sum(p["source_primary_prefix_tied"] is None for p in pairs),
        "both_endpoint_available_pairs": sum(p["both_endpoints_available"] for p in pairs),
        "endpoint_available_tied_pairs": len(available_tied),
        "endpoint_available_tied_cases": len(case_gaps),
        "case_mean_absolute_biovil_gap_among_available_ties": _mean(_mean(v) for v in case_gaps.values()),
        "endpoint_available_candidates": sum(r["biovil_raw_cosine"] is not None for r in rows),
        "endpoint_unavailable_reasons": dict(Counter(r["endpoint_unavailable_reason"] for r in rows if r["biovil_raw_cosine"] is None)),
        "source_primary_prefix_fields": list(PREFIX_FIELDS[profile]),
        "new_model_calls": 0, "source_choices_changed": False, "thresholds_or_masks_changed": False,
        "clinical_accuracy": None, "clinical_acceptance": False,
        "endpoint_missingness_is_not_zero": True, "significance_test_performed": False,
        "all_input_cases_retained": True}
    return {"summary": summary, "rows": rows, "same_image_pairs": pairs, "model_coverage": model_rows}
