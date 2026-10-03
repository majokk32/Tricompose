#!/usr/bin/env python3
"""Stage a bounded synthetic report-polarity diagnostic, never clinical gold.

Only two hash-bound synthetic reports are interpreted. Source reports/images,
fixed EHRs, scores and winners remain immutable. No inference or external API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from audit_selected_disagreements import load as load_audited_findings
from contracts import (
    PROTECTED_ROOT, commit_atomic_run, discard_atomic_run, new_atomic_run,
    private_directory, read_json, require_inside, sha256_file,
    write_private_json, write_private_text,
)

SCHEMA = "tricompose-synthetic-report-polarity-diagnostic-v1"
RESOLVER_FIELDS = frozenset({"item_id", "image_path", "image_sha256", "report_path", "report_sha256"})
PATTERNS = {"pleural_effusion": r"\b(?:pleural\s+)?effusions?\b",
            "consolidation": r"\bconsolidations?\b"}
NEGATIVES = {"pleural_effusion": "No pleural effusion.",
             "consolidation": "No consolidation."}
MODIFIERS = frozenset("there is are a an the and of seen noted present identified evident small moderate large trace minimal mild slight bilateral left right sided basilar basal bibasilar layering pleural effusion effusions consolidation consolidations".split())


def minimal_flip(text, finding):
    """Refuse ambiguity; negate one isolated explicit affirmative sentence.

    This narrow whitelist is a construction contract, not a clinical extractor.
    It does not infer consolidation from opacity or clinical truth from text.
    """
    if finding not in PATTERNS:
        return None, "unsupported_target_finding"
    mentions = list(re.finditer(PATTERNS[finding], text, re.I))
    if not mentions:
        return None, "no_explicit_target_phrase"
    if len(mentions) != 1:
        return None, "multiple_target_mentions"
    sentences = [match for match in re.finditer(r"[^.!?\n]+[.!?]?", text)
                 if re.search(PATTERNS[finding], match.group(), re.I)]
    if len(sentences) != 1:
        return None, "target_sentence_unavailable"
    match = sentences[0]
    sentence = match.group()
    header = re.match(r"\s*(?:findings|impression)\s*:\s*", sentence, re.I)
    start = match.start() + (header.end() if header else len(sentence) - len(sentence.lstrip()))
    statement = text[start:match.end()]
    words = re.findall(r"[a-z]+", statement.lower())
    allowed = MODIFIERS - ({"consolidation", "consolidations"} if finding == "pleural_effusion"
                          else {"pleural", "effusion", "effusions"})
    if not words or not set(words) <= allowed or not re.fullmatch(r"[A-Za-z\s.,-]+", statement):
        return None, "not_an_isolated_unambiguous_affirmative_sentence"
    replacement = NEGATIVES[finding]
    changed = text[:start] + replacement + text[match.end():]
    # Explicit prefix/suffix and reversible-edit checks preserve unrelated text.
    if (changed[:start] != text[:start] or changed[start + len(replacement):] != text[match.end():]
            or changed[:start] + statement + changed[start + len(replacement):] != text):
        raise ValueError("minimal edit failed reversibility")
    return {"text": changed, "span_start": start, "span_end": match.end(),
            "source_statement": statement, "replacement_statement": replacement,
            "source_text_assertion": "positive", "edited_text_assertion": "negative"}, None


def whitespace_control(text):
    changed = text.replace(" ", "  ")
    if changed.split() != text.split():
        raise ValueError("whitespace control changed a token")
    return changed


def build_records(selected, source_texts):
    if len(selected) != 2 or len({row["case_id"] for row in selected}) != 2:
        raise ValueError("requires the two fixed selected opposition cases")
    records, rejected = [], []
    for row in sorted(selected, key=lambda item: item["case_id"]):
        text = source_texts[row["case_id"]]
        if not text.strip() or len(text) > 8192:
            raise ValueError("source report outside bounded synthetic contract")
        flip, reason = minimal_flip(text, row["finding"])
        variants = [("unchanged", text, None), ("whitespace_only", whitespace_control(text), None)]
        if flip:
            variants.append(("minimal_polarity_flip", flip["text"], flip))
        else:
            rejected.append({"case_id": row["case_id"], "finding": row["finding"],
                             "intervention_type": "minimal_polarity_flip", "reason": reason})
        for kind, output, edit in variants:
            records.append({"case_id": row["case_id"], "finding": row["finding"],
                "evidence_id": row["evidence_id"], "intervention_type": kind, "text": output,
                "source": row, "edit": edit, "clinical_mismatch_verified": False})
    # Opaque item IDs hide construction order, but this is not blinded human review.
    records.sort(key=lambda row: hashlib.sha256(
        f"polarity-v1|{row['evidence_id']}|{row['intervention_type']}".encode()).hexdigest())
    return records, rejected


def load_blind_bank(bank_path):
    """Scorer reads only manifest metadata and resolver, never the answer key."""
    root = require_inside(bank_path, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema_version") != SCHEMA or manifest.get("synthetic_only") is not True
            or manifest.get("primary_metric_eligible") is not False):
        raise ValueError("unexpected polarity diagnostic scope")
    path = root / "resolver.jsonl"
    if sha256_file(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("resolver hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not 4 <= len(rows) <= 6 or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("unexpected bounded item inventory")
    for row in rows:
        if set(row) != RESOLVER_FIELDS:
            raise ValueError("resolver fields leak construction labels or are missing")
        for kind in ("image", "report"):
            artifact = require_inside(row[f"{kind}_path"], PROTECTED_ROOT, must_exist=True)
            if sha256_file(artifact) != row[f"{kind}_sha256"]:
                raise ValueError("polarity artifact hash mismatch")
        report = require_inside(row["report_path"], root, must_exist=True)
        if report.stat().st_size > 32768:
            raise ValueError("report exceeds bounded diagnostic size")
    if len({row["image_sha256"] for row in rows}) != 2:
        raise ValueError("expected two fixed images")
    return rows, {"manifest_sha256": sha256_file(root / "manifest.json"),
                  "resolver_sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finding-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        selected, _, sources = load_audited_findings(args.finding_run)
        texts = {}
        for row in selected:
            path = require_inside(row["report_path"], PROTECTED_ROOT, must_exist=True)
            if sha256_file(path) != row["artifact_hashes"]["report_sha256"]:
                raise ValueError("source synthetic report changed")
            if path.stat().st_size > 32768:
                raise ValueError("source synthetic report exceeds pilot bound")
            texts[row["case_id"]] = path.read_bytes().decode("utf-8")
            sources[f"original_report_{row['case_id']}"] = path
        records, rejected = build_records(selected, texts)
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        private_directory(temporary / "reports")
        resolver, key, files = [], [], []
        for number, record in enumerate(records):
            item_id = f"item_{number:04d}"
            row = record["source"]
            relative = Path("reports") / f"{item_id}.txt"
            artifact = write_private_text(temporary / relative, record["text"])
            files.append(artifact)
            resolver.append({"item_id": item_id, "image_path": row["cxr_path"],
                "image_sha256": row["artifact_hashes"]["cxr_sha256"],
                "report_path": str(target / relative), "report_sha256": sha256_file(artifact)})
            key.append({"item_id": item_id, "case_id": record["case_id"],
                "finding": record["finding"], "evidence_id": record["evidence_id"],
                "intervention_type": record["intervention_type"], "edit": record["edit"],
                "fixed_ehr_sha256": row["artifact_hashes"]["ehr_sha256"],
                "source_report_sha256": row["artifact_hashes"]["report_sha256"],
                "clinical_mismatch_verified": False, "independent_review_status": "pending"})
        counts = {"fixed_cases": 2, "reports": len(records),
                  "unchanged_controls": 2, "whitespace_controls": 2,
                  "eligible_polarity_flips": len(records) - 4,
                  "unavailable_polarity_flips": len(rejected)}
        summary = {"schema_version": SCHEMA, "counts": counts, "rejections": rejected,
            "synthetic_only": True, "primary_metric_eligible": False,
            "targeted_repair_approved": False, "original_artifacts_or_winners_changed": False,
            "scored": False, "clinical_localization_accuracy": None}
        for name, rows in (("resolver", resolver), ("intervention_key", key)):
            files.append(write_private_text(temporary / f"{name}.jsonl", "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in rows)))
        files.append(write_private_json(temporary / "summary.json", summary))
        files.append(write_private_text(temporary / "summary.md", "\n".join([
            "# Synthetic report polarity diagnostic / 报告最小否定测试", "",
            f"Fixed cases: 2; unchanged controls: 2; whitespace controls: 2; polarity flips: {len(records)-4}.",
            f"Unavailable target edits: {len(rejected)}; reasons are retained in summary.json.",
            "Only isolated, explicitly affirmative finding sentences are negated; unrelated text and images stay fixed.",
            "Consolidation is NOT inferred from opacity to invent a mutable statement.",
            "This tests extraction/metric sensitivity, not which original image/report is clinically correct.",
            "Controls may already be clinically wrong. BioViL score decreases are NOT assumed to be correct.",
            "Intervention labels are mechanical; human/clinical adjudication remains pending.", ""])))
        write_private_json(temporary / "manifest.json", {**{k: v for k, v in summary.items() if k != "rejections"}, "run_id": args.run_id,
            "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "program_sha256": sha256_file(__file__),
            "artifacts": {str(path.relative_to(temporary)): {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "prepared_synthetic_polarity_diagnostic", "counts": counts,
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
