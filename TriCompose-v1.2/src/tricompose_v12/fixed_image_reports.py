"""Same-image report choice, followed by a routing-independent endpoint.

Pure metadata/state functions. Labels are imperfect proxies, not clinical truth.
The policy penalizes missing positive image evidence as well as explicit label
opposition; it never rewards a blanket negative or promotes unknown to negative.
"""
from collections import defaultdict
import math

from .invariant_verification import _digest, _edge, _HASH, _ID
from .legacy_replay_adapter import FINDINGS, STATES

POLICY = {
    "schema_version": "tricompose-fixed-image-report-policy-v1",
    "candidate_models": ["cxrmate_single", "maira2"],
    "baseline_model": "cxrmate_single",
    "key": ["explicit_opposition_plus_missing_positive", "minus_supported_positive",
            "prefer_fixed_baseline_on_tie", "opaque_candidate_id"],
    "uses_biovil": False, "uses_negative_agreement_reward": False,
    "clinical_acceptance_allowed": False, "changes_ehr_or_cxr": False,
}


def report_row(image, report, partial, labels, *, labels_sha256, checkpoint_sha256):
    """One row from a hash-authenticated raw CheXbert bundle and image receipt."""
    if (report["case_id"] != image["case_id"] or partial["case_id"] != image["case_id"]
            or partial["cxr_candidate_id"] != image["candidate_id"]
            or partial["cxr_sha256"] != image["artifact"]["sha256"]
            or report["parent_cxr_candidate_id"] != image["candidate_id"]
            or report["input_cxr"]["sha256"] != image["artifact"]["sha256"]
            or report["ehr_sha256_retained_for_lineage"] != image["ehr_sha256"]
            or report["ehr_facts_sha256_retained_for_lineage"] != image["ehr_facts_sha256"]
            or report.get("frozen_model") is not True
            or report.get("model_input_signature") != "single_current_synthetic_cxr"
            or report.get("structured_ehr_content_supplied_to_model") is not False
            or report.get("source_report_or_real_target_supplied") is not False
            or report["model_id"] not in POLICY["candidate_models"]):
        raise ValueError("fixed synthetic image/report lineage differs")
    if (labels.get("schema_version") != "tricompose-report-finding-labels-v1.1"
            or labels.get("producer", {}).get("frozen") is not True
            or labels["producer"]["checkpoint_sha256"] != checkpoint_sha256
            or labels.get("finding_order") != list(FINDINGS)
            or labels.get("calibration", {}).get("unknown_is_negative") is not False
            or labels.get("counts") != {"reports": len(labels["records"]), "model_calls": len(labels["records"])}):
        raise ValueError("raw frozen CheXbert profile differs")
    index = {r["report_candidate_id"]: r for r in labels["records"]}
    if len(index) != len(labels["records"]): raise ValueError("duplicate label record")
    label = index[report["candidate_id"]]
    if label["report_sha256"] != report["artifact"]["sha256"]:
        raise ValueError("label/report artifact hash differs")
    text = label["finding_states"]
    if set(text) != set(FINDINGS) or any(v not in STATES for v in text.values()):
        raise ValueError("complete four-state labels required")
    facts = partial["fact_states"]
    if ([f["finding"] for f in facts] != list(FINDINGS)
            or any(f[k] not in STATES for f in facts for k in ("ehr", "xrv"))):
        raise ValueError("complete fixed image/EHR states required")
    states = [{**f, "chexbert": text[f["finding"]]} for f in facts]
    edges = {"ehr_cxr": _edge(states, "ehr", "xrv"),
             "ehr_report": _edge(states, "ehr", "chexbert"),
             "cxr_report": _edge(states, "xrv", "chexbert")}
    missing_positive = sum(f["xrv"] == "positive" and f["chexbert"] in {"unknown", "uncertain"}
                           for f in states)
    row = {"case_id": image["case_id"], "cxr_candidate_id": image["candidate_id"],
        "report_candidate_id": report["candidate_id"], "report_model_id": report["model_id"],
        "ehr_sha256": image["ehr_sha256"], "ehr_facts_sha256": image["ehr_facts_sha256"],
        "cxr_sha256": image["artifact"]["sha256"], "report_sha256": report["artifact"]["sha256"],
        "chexbert_labels_sha256": labels_sha256, "chexbert_checkpoint_sha256": checkpoint_sha256,
        "image_receipt_id": partial["receipt_id"], "fact_states": states, "raw_edge_readouts": edges,
        "missing_positive_image_facts": missing_positive,
        "proxy_penalty": edges["cxr_report"]["proxy_opposition_facts"] + missing_positive,
        "clinical_accuracy": None, "clinical_acceptance": False,
        "same_image_reports_are_independent_votes": False, "selection_used_biovil": False}
    if (any(not isinstance(row[k], str) or not _HASH.fullmatch(row[k]) for k in
            ("ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256", "chexbert_labels_sha256",
             "chexbert_checkpoint_sha256", "image_receipt_id"))
            or any(not _ID.fullmatch(row[k]) for k in ("case_id", "cxr_candidate_id", "report_candidate_id"))):
        raise ValueError("opaque IDs and SHA256 lineage required")
    row["triple_candidate_id"] = "fixedpair_" + _digest([row["cxr_candidate_id"], row["report_candidate_id"]])[:32]
    return row


def selection_key(row):
    edge = row["raw_edge_readouts"]["cxr_report"]
    return (row["proxy_penalty"], -edge["supported_positive"],
            row["report_model_id"] != POLICY["baseline_model"], row["report_candidate_id"])


def freeze_selection(rows):
    groups = defaultdict(list); seen = set()
    for row in rows:
        if row["triple_candidate_id"] in seen: raise ValueError("duplicate pair")
        seen.add(row["triple_candidate_id"]); groups[row["cxr_candidate_id"]].append(row)
    if not groups: raise ValueError("nonempty paired image pool required")
    choices = []
    for iid, group in sorted(groups.items()):
        if len(group) != 2 or {r["report_model_id"] for r in group} != set(POLICY["candidate_models"]):
            raise ValueError("exactly both frozen report experts per fixed image required")
        for field in ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "image_receipt_id"):
            if len({r[field] for r in group}) != 1: raise ValueError("same-image comparison changed anchor")
        if group[0]["raw_edge_readouts"]["ehr_cxr"] != group[1]["raw_edge_readouts"]["ehr_cxr"]:
            raise ValueError("same-image comparison changed image evidence")
        baseline = next(r for r in group if r["report_model_id"] == POLICY["baseline_model"])
        chosen = min(group, key=selection_key)
        choices.append({"case_id": chosen["case_id"], "cxr_candidate_id": iid,
            "baseline_triple_id": baseline["triple_candidate_id"],
            "selected_triple_id": chosen["triple_candidate_id"],
            "baseline_key": list(selection_key(baseline)), "selected_key": list(selection_key(chosen)),
            "image_known_proxy_facts": chosen["raw_edge_readouts"]["cxr_report"]["known_reference_facts"],
            "direct_ehr_known_facts": chosen["raw_edge_readouts"]["ehr_cxr"]["known_reference_facts"],
            "clinical_acceptance": False})
    payload = {"schema_version": "tricompose-fixed-image-report-selection-v1", "policy": POLICY,
        "score_rows_sha256": _digest(rows), "choices": choices, "used_biovil": False,
        "ehr_cxr_regeneration_calls": 0, "adaptive_repair_executed": False, "clinical_acceptance": False}
    return {**payload, "selection_id": _digest(payload)}


def secondary_comparison(selection, rows, endpoint):
    """Pair by fixed image, average within EHR, never fill missing with zero."""
    if freeze_selection(rows) != selection: raise ValueError("selection changed after endpoint")
    pairs = {r["triple_candidate_id"]: r for r in rows}
    records = endpoint["records"]; scored = {r["triple_candidate_id"]: r for r in records}
    if (len(scored) != len(records) or set(scored) != set(pairs)
            or endpoint.get("used_for_routing") is not False
            or endpoint.get("clinical_truth_available") is not False):
        raise ValueError("complete independent secondary endpoint required")
    for cid, r in scored.items():
        for name in ("case_id", "cxr_candidate_id", "report_candidate_id", "ehr_sha256",
                     "ehr_facts_sha256", "cxr_sha256", "report_sha256"):
            if r[name] != pairs[cid][name]: raise ValueError("endpoint lineage differs")
        value = r["biovil_raw_cosine"]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not -1.01 <= value <= 1.01):
            raise ValueError("invalid raw cosine")
        if value is None and not r.get("reason"): raise ValueError("missing endpoint requires reason")
    by_case = defaultdict(list); image_pairs = []
    for choice in selection["choices"]:
        left = scored[choice["baseline_triple_id"]]["biovil_raw_cosine"]
        right = scored[choice["selected_triple_id"]]["biovil_raw_cosine"]
        delta = right - left if left is not None and right is not None else None
        image_pairs.append({"case_id": choice["case_id"], "cxr_candidate_id": choice["cxr_candidate_id"],
                            "baseline_raw_cosine": left, "selected_raw_cosine": right, "delta": delta})
        by_case[choice["case_id"]].append(delta)
    case_pairs = [{"case_id": case, "fixed_images": len(values),
                   "available_image_pairs": sum(v is not None for v in values),
                   "delta": sum(values) / len(values) if all(v is not None for v in values) else None}
                  for case, values in sorted(by_case.items())]
    deltas = [r["delta"] for r in case_pairs]
    return {"schema_version": "tricompose-fixed-image-secondary-comparison-v1",
        "selection_id": selection["selection_id"], "image_pairs": image_pairs, "case_pairs": case_pairs,
        "mean_case_delta": sum(deltas) / len(deltas) if all(v is not None for v in deltas) else None,
        "interpretation": "small_same_image_proxy_selection_check_not_clinical_accuracy_or_repair",
        "independent_patients": None, "clinical_truth_available": False,
        "endpoint_used_for_selection": False, "significance_test_performed": False}
