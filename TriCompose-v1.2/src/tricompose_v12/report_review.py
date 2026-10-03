"""Blinded report-only annotation contracts, not image or clinical gold.

Pending is distinct from reviewed unknown. No extractor output is used to
prefill labels, break reader ties, or authorize regeneration.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from .report_assertions import FINDINGS, STATES, digest, validate_span

VERSION = "report-only-four-state-human-review-v1"
SCHEMA = "tricompose-blinded-report-annotation-v1"
STATUS = frozenset({"pending", "reviewed", "unassessable"})
REASONS = frozenset({"explicit_assertion", "not_mentioned",
    "historical_or_hypothetical_only", "qualified_or_conflicting", "unassessable"})
ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def object_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")))


def make_queue(candidates):
    if not candidates or len(candidates) > 48:
        raise ValueError("bounded nonempty report inventory required")
    ids = [row["candidate_id"] for row in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate inventory")
    hashes = {row["report_sha256"] for row in candidates}
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
        raise ValueError("invalid source report hash")
    # No score, winner flag, text editability or finding prevalence affects order.
    ordered = sorted(hashes, key=lambda h: hashlib.sha256((VERSION+"|"+h).encode()).digest())
    opaque = {h: f"report_{i:04d}" for i, h in enumerate(ordered)}
    items = [{"item_id": opaque[h], "report_sha256": h,
        "report_relative_path": f"reports/{opaque[h]}.txt"} for h in ordered]
    resolver = [{**row, "item_id": opaque[row["report_sha256"]]}
        for row in sorted(candidates, key=lambda row: row["candidate_id"])]
    return items, resolver


def annotation_template(items):
    return {"schema_version": SCHEMA, "protocol_version": VERSION,
        "inventory_sha256": object_hash(items), "reviewer_alias": None,
        "reviewer_role": None, "independence_attestation": None,
        "input_scope": "report_only_no_images_ehr_predictions_or_winner_flags",
        "records": [{"item_id": row["item_id"], "report_sha256": row["report_sha256"],
            "finding": name, "status": "pending", "state": None,
            "reason": None, "evidence": []} for row in items for name in FINDINGS]}


def validate_annotations(items, annotation, *, read_source=None):
    expected = annotation_template(items)
    for field in ("schema_version", "protocol_version", "inventory_sha256", "input_scope"):
        if annotation.get(field) != expected[field]:
            raise ValueError("annotation header binding differs")
    index = {(row["item_id"], row["finding"]): row for row in annotation["records"]}
    required = {(row["item_id"], row["finding"]): row for row in expected["records"]}
    if len(index) != len(annotation["records"]) or set(index) != set(required):
        raise ValueError("missing or duplicate annotation rows")
    nonpending = any(row.get("status") != "pending" for row in index.values())
    if nonpending:
        alias = annotation.get("reviewer_alias")
        if not isinstance(alias, str) or not ALIAS.fullmatch(alias):
            raise ValueError("opaque reviewer alias required")
        if annotation.get("reviewer_role") not in {"human_domain_annotator", "clinician", "radiologist", "non_expert"}:
            raise ValueError("human reviewer role required")
        if annotation.get("independence_attestation") != {
                "model_predictions_seen": False, "winner_flags_seen": False,
                "labels_generated_by_model": False}:
            raise ValueError("independent human annotation attestation required")
    texts = {}
    for key, row in index.items():
        if set(row) != set(required[key]) or row["report_sha256"] != required[key]["report_sha256"]:
            raise ValueError("annotation row/source binding differs")
        if row["status"] not in STATUS or not isinstance(row["evidence"], list):
            raise ValueError("invalid annotation status/evidence")
        if row["status"] == "pending":
            if row["state"] is not None or row["reason"] is not None or row["evidence"]:
                raise ValueError("pending is not a labeled unknown")
            continue
        if row["reason"] not in REASONS:
            raise ValueError("review reason required")
        if row["status"] == "unassessable":
            if row["state"] is not None or row["reason"] != "unassessable":
                raise ValueError("unassessable cannot supply a state")
        elif row["state"] not in STATES:
            raise ValueError("reviewed four-state label required")
        elif row["reason"] == "unassessable":
            raise ValueError("reviewed state cannot be unassessable")
        elif row["state"] != "unknown" and not row["evidence"]:
            raise ValueError("asserted label requires source evidence")
        if row["reason"] == "not_mentioned" and (row["state"] != "unknown" or row["evidence"]):
            raise ValueError("not mentioned is reviewed unknown without fabricated evidence")
        if read_source is None:
            raise ValueError("completed review requires approved source verification")
        h = row["report_sha256"]
        if h not in texts:
            texts[h] = read_source(h)
            if not isinstance(texts[h], str) or digest(texts[h]) != h:
                raise ValueError("review source text hash differs")
        for span in row["evidence"]:
            validate_span(texts[h], span)
    return index


def reader_agreement(items, first, second, *, read_source=None):
    left = validate_annotations(items, first, read_source=read_source)
    right = validate_annotations(items, second, read_source=read_source)
    if (first.get("reviewer_alias") is not None and
            first["reviewer_alias"] == second.get("reviewer_alias")):
        raise ValueError("distinct human readers required")
    comparable = [key for key in left if left[key]["status"] == right[key]["status"] == "reviewed"]
    matches = sum(left[key]["state"] == right[key]["state"] for key in comparable)
    count = len(comparable)
    observed = matches/count if count else None
    kappa = None
    if count:
        a = Counter(left[key]["state"] for key in comparable)
        b = Counter(right[key]["state"] for key in comparable)
        chance = sum(a[name]*b[name] for name in STATES)/(count*count)
        if chance < 1:
            kappa = (observed-chance)/(1-chance)
    return {"annotation_rows_per_reader": len(left), "both_reviewed": count,
        "both_reviewed_coverage": count/len(left), "matching_states": matches,
        "disagreements": count-matches, "exact_reader_agreement": observed,
        "cohens_kappa_nominal_four_states": kappa,
        "reader_status_counts": [dict(Counter(row["status"] for row in data.values())) for data in (left,right)],
        "clinical_accuracy": None, "adjudicated_gold_created": False,
        "model_or_rules_used_to_break_ties": False, "regeneration_authorized": False,
        "independence_attestation_externally_verified": False}
