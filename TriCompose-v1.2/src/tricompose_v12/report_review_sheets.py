"""Human CSV -> unchanged annotation contract; no label inference or gold.

Only source offsets/hashes are computed. A quote with multiple occurrences
requires an explicit zero-based occurrence index from the human reviewer.
"""
from __future__ import annotations

import copy
import csv
import io
import re

from .report_assertions import checked_span, digest
from .report_review import annotation_template, validate_annotations

IDENTITY_SCHEMA = "tricompose-human-review-sheet-identity-v1"
FIELDS = ("item_id", "report_sha256", "finding", "status", "state", "reason",
    "quote_1", "quote_1_occurrence", "quote_2", "quote_2_occurrence")


def identity_template(items):
    template = annotation_template(items)
    return {"schema_version": IDENTITY_SCHEMA,
        "inventory_sha256": template["inventory_sha256"],
        "reviewer_alias": None, "reviewer_role": None,
        "independence_attestation": None}


def sheet_template(items):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in annotation_template(items)["records"]:
        writer.writerow({**{key: row[key] for key in FIELDS[:6]},
            **dict.fromkeys(FIELDS[6:], "")})
    return stream.getvalue()


def locate_human_quote(text, quote, occurrence):
    if not quote:
        if occurrence:
            raise ValueError("quote occurrence supplied without quote")
        return None
    if not isinstance(quote, str) or not 1 <= len(quote) <= 8192:
        raise ValueError("invalid bounded human quote")
    matches, start = [], 0
    while True:
        index = text.find(quote, start)
        if index < 0:
            break
        matches.append(index)
        start = index+1
    if not matches:
        raise ValueError("human quote not in exact source")
    if occurrence == "":
        if len(matches) != 1:
            raise ValueError("ambiguous quote needs explicit occurrence")
        selected = 0
    else:
        if not isinstance(occurrence, str) or not re.fullmatch(r"0|[1-9][0-9]*", occurrence):
            raise ValueError("quote occurrence must be zero-based integer")
        selected = int(occurrence)
        if selected >= len(matches):
            raise ValueError("human quote occurrence outside source")
    left = matches[selected]
    return checked_span(text, left, left+len(quote))


def import_sheet(items, text, identity, *, read_source=None):
    expected_identity = identity_template(items)
    if (set(identity) != set(expected_identity) or
            identity["schema_version"] != IDENTITY_SCHEMA or
            identity["inventory_sha256"] != expected_identity["inventory_sha256"]):
        raise ValueError("sheet identity binding differs")
    reader = csv.DictReader(io.StringIO(text.removeprefix("\ufeff"), newline=""))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError("sheet columns differ or are duplicated")
    rows = list(reader)
    annotation = annotation_template(items)
    required = {(row["item_id"], row["finding"]): row for row in annotation["records"]}
    index = {(row.get("item_id"), row.get("finding")): row for row in rows}
    if len(index) != len(rows) or set(index) != set(required):
        raise ValueError("missing duplicate or foreign sheet rows")
    annotation.update({key: copy.deepcopy(identity[key]) for key in
        ("reviewer_alias", "reviewer_role", "independence_attestation")})
    sources = {}
    result = []
    for key, expected in required.items():
        row = index[key]
        if set(row) != set(FIELDS) or any(value is None for value in row.values()):
            raise ValueError("malformed sheet row")
        if row["report_sha256"] != expected["report_sha256"]:
            raise ValueError("sheet source report hash differs")
        value = {**expected, "status": row["status"], "state": row["state"] or None,
            "reason": row["reason"] or None, "evidence": []}
        for number in (1, 2):
            quote, occurrence = row[f"quote_{number}"], row[f"quote_{number}_occurrence"]
            if not quote:
                if occurrence:
                    raise ValueError("quote occurrence supplied without quote")
                continue
            if read_source is None:
                raise ValueError("human quotes require approved source access")
            h = value["report_sha256"]
            if h not in sources:
                sources[h] = read_source(h)
                if not isinstance(sources[h], str) or digest(sources[h]) != h:
                    raise ValueError("human sheet source hash differs")
            span = locate_human_quote(sources[h], quote, occurrence)
            if span in value["evidence"]:
                raise ValueError("duplicate human evidence span")
            value["evidence"].append(span)
        result.append(value)
    annotation["records"] = result
    validate_annotations(items, annotation, read_source=read_source)
    return annotation


def report_book(items, texts):
    """Protected Markdown only; exact original text inside inert code fences."""
    lines = ["# Synthetic report-only review / 合成报告文字审核", "",
        "Use opaque report IDs to fill your own CSV. No image/EHR/model scores are included.",
        "This is a two-EHR development pilot, not clinical gold or an untouched test.", ""]
    for item in items:
        h = item["report_sha256"]
        if digest(texts[h]) != h:
            raise ValueError("review book source hash differs")
        runs = [len(match.group()) for match in re.finditer(r"`+", texts[h])]
        fence = "`"*max(3, 1+max(runs, default=0))
        lines.extend([f"## {item['item_id']}", "", fence+"text", texts[h], fence, ""])
    return "\n".join(lines)
