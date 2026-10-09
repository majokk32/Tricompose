"""Set-difference request for an exhaustive cached synthetic endpoint inventory.

Scores, models, states and winners cannot affect which missing pairs are requested.
"""
from __future__ import annotations

import math
from .invariant_verification import _HASH, _ID
from .legacy_replay_adapter import PROFILE

PAIR_FIELDS = ("case_id", "triple_candidate_id", "cxr_candidate_id", "report_candidate_id",
               "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256")
SCHEMA = "tricompose-complete-bank-secondary-plan-v1"
EXPERIMENT = "historical_development_full_bank_endpoint_completion_not_policy_replay"


def pair(row):
    result = {k:row[k] for k in PAIR_FIELDS}
    if (any(not isinstance(result[k],str) or not _HASH.fullmatch(result[k]) for k in PAIR_FIELDS if k.endswith("sha256"))
            or any(not isinstance(result[k],str) or not _ID.fullmatch(result[k]) for k in PAIR_FIELDS if not k.endswith("sha256"))):
        raise ValueError("opaque synthetic pair IDs and hashes required")
    return result


def missing_pairs(inventory, previous):
    if not inventory or len(inventory)>960:
        raise ValueError("bounded complete inventory required")
    full, old = {}, {}
    for row in inventory:
        if row.get("profile") != PROFILE:
            raise ValueError("historical profile required; never mix fresh heads")
        p = pair(row); cid = p["triple_candidate_id"]
        if cid in full:raise ValueError("duplicate inventory pair")
        full[cid] = p
    for record in previous:
        p = pair(record); cid = p["triple_candidate_id"]
        if cid in old or full.get(cid) != p:
            raise ValueError("previous endpoint lineage differs or duplicates")
        value = record.get("biovil_raw_cosine")
        if value is None:
            if record.get("status") != "not_available" or not record.get("reason"):
                raise ValueError("old unavailable endpoint requires reason")
        elif (type(value) not in (int,float) or not math.isfinite(value) or not -1.01<=value<=1.01
              or record.get("status") != "computed_secondary_uncalibrated" or record.get("reason") is not None):
            raise ValueError("valid raw prior endpoint required")
        if record.get("calibrated") is not False:raise ValueError("cosine is not calibrated")
        old[cid] = p
    pending = [full[cid] for cid in sorted(set(full)-set(old))]
    return pending, {"all_candidate_pairs":len(full), "already_scored_pairs_including_na":len(old),
        "missing_candidate_pairs":len(pending),
        "maximum_new_image_encoder_samples":len({p["cxr_candidate_id"] for p in pending}),
        "maximum_new_text_encoder_samples":len({p["report_candidate_id"] for p in pending}),
        "old_na_endpoints_not_retried":sum(r["biovil_raw_cosine"] is None for r in previous),
        "endpoint_values_used_to_choose_missing_pairs":False, "original_selection_changed":False}
