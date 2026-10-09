"""Lossless full-inventory secondary-score merge; never reranking or routing."""
from __future__ import annotations

import math
from .complete_endpoint_inventory import missing_pairs, pair

SCHEMA = "tricompose-full-bank-secondary-coverage-v1"


def validate_bundle(bundle, inventory, expected_pairs, model_hashes):
    if (bundle.get("used_for_routing") is not False or bundle.get("primary_clinical_metric") is not False
            or bundle.get("clinical_truth_available") is not False
            or bundle.get("original_selection_changed") is not False
            or bundle.get("producer",{}).get("frozen") is not True
            or bundle["producer"].get("checkpoint_sha256") != model_hashes
            or bundle["producer"].get("text_policy") != "full_report_no_silent_truncation_overlength_is_na"):
        raise ValueError("frozen secondary-only full-report scores required")
    records = bundle["records"]
    missing_pairs(inventory,records)  # Validate unique IDs, hashes, four-state NA status.
    returned = sorted((pair(r) for r in records),key=lambda p:p["triple_candidate_id"])
    if returned != sorted(expected_pairs,key=lambda p:p["triple_candidate_id"]):
        raise ValueError("exact planned endpoint pair inventory required")
    available = [r for r in records if r["biovil_raw_cosine"] is not None]
    counts = {"requested_pairs":len(records),
        "image_encoder_calls":len({r["cxr_candidate_id"] for r in available}),
        "text_encoder_calls":len({r["report_candidate_id"] for r in available}),
        "unavailable_reports":len({r["report_candidate_id"] for r in records if r["biovil_raw_cosine"] is None})}
    if bundle.get("counts") != counts:
        raise ValueError("missingness or encoder accounting differs")
    for field in ("peak_vram_gib","runtime_seconds"):
        v=bundle.get(field)
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:
            raise ValueError("finite nonnegative runtime/memory required")


def merge(inventory, old, new, request, model_hashes):
    pending,counts=missing_pairs(inventory,old["records"])
    if pending != request["pairs"] or counts != request["inventory_counts"]:
        raise ValueError("frozen set-difference request changed")
    validate_bundle(old,inventory,[pair(r) for r in old["records"]],model_hashes)
    validate_bundle(new,inventory,pending,model_hashes)
    records=sorted([*old["records"],*new["records"]],key=lambda r:r["triple_candidate_id"])
    remaining,_=missing_pairs(inventory,records)
    if remaining:raise ValueError("complete nominal coverage required")
    return records,{"schema_version":SCHEMA,"candidate_pairs":len(records),
        "old_records_retained_exactly":len(old["records"]),"new_records":len(new["records"]),
        "numerically_available_pairs":sum(r["biovil_raw_cosine"] is not None for r in records),
        "na_pairs":sum(r["biovil_raw_cosine"] is None for r in records),
        "new_evaluation_counts":new["counts"],"new_evaluation_runtime_seconds":new["runtime_seconds"],
        "new_peak_vram_gib":new["peak_vram_gib"],"old_record_values_changed":False,
        "source_choices_changed":False,"clinical_accuracy":None,"clinical_acceptance":False,
        "gpu_generation_savings":None,"full_report_context_retokenized_by_merge":False}
