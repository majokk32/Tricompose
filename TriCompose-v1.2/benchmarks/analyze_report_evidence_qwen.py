#!/usr/bin/env python3
"""Cached development evidence alignment, not clinical gold or new scores.

Only this post-inference analysis reads the synthetic construction key. The
verifier itself sees report text only. Exact quote offsets are traceability,
not independent semantic verification or proof that other assertions are absent.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from contracts import (
    PROTECTED_ROOT, commit_atomic_run, discard_atomic_run, new_atomic_run,
    read_json, require_inside, sha256_file, write_private_json, write_private_text,
)
from diagnose_chexbert_polarity import SCHEMA as CONTEXT_SCHEMA, sentence_bindings
from verify_report_evidence_qwen import (
    FINDINGS, POLARITIES, SCHEMA as REVIEW_SCHEMA, decode_evidence, digest_text,
    load_report_texts, summarize as review_summary,
)

SCHEMA = "tricompose-qwen-report-evidence-alignment-diagnostic-v1"


def span_alignment(record, finding, start, end, expected):
    if finding not in FINDINGS or expected not in {"positive", "negative"} or not 0 <= start < end:
        raise ValueError("invalid bounded target span or text assertion")
    if record["contract_status"] != "complete":
        return {"contract_status": record["contract_status"], "state": "unknown",
            "expected_target_quote_recorded": None, "opposite_outside_target_recorded": None,
            "uncertain_outside_target_recorded": None, "span_counts": None}
    row = record["findings"][finding]
    counts = {relation: dict.fromkeys(POLARITIES, 0) for relation in ("within_target", "crossing_target", "outside_target")}
    for polarity in POLARITIES:
        for span in row["evidence"][polarity]:
            left, right = span["char_start"], span["char_end"]
            if not 0 <= left < right:
                raise ValueError("invalid evidence offsets")
            if start <= left and right <= end:
                relation = "within_target"
            elif right <= start or left >= end:
                relation = "outside_target"
            else:
                relation = "crossing_target"
            counts[relation][polarity] += 1
    opposite = "negative" if expected == "positive" else "positive"
    return {"contract_status": "complete", "state": row["state"],
        "expected_target_quote_recorded": counts["within_target"][expected] > 0,
        "opposite_outside_target_recorded": counts["outside_target"][opposite] > 0,
        "uncertain_outside_target_recorded": counts["outside_target"]["uncertain"] > 0,
        "span_counts": counts}


def annotate_pair(pair, edit, evidence, full_states):
    finding = pair["finding"]
    source_hash, edited_hash = pair["original_report_sha256"], pair["edited_report_sha256"]
    start = edit["span_start"]
    source = span_alignment(evidence[source_hash], finding, start, edit["span_end"], pair["expected_source_text_state"])
    target = span_alignment(evidence[edited_hash], finding, start,
        start+len(edit["replacement_statement"]), pair["expected_edited_text_state"])
    source_state = full_states[source_hash][finding]["state"]
    target_state = full_states[edited_hash][finding]["state"]
    both_complete = source["contract_status"] == target["contract_status"] == "complete"
    target_quotes_recorded = both_complete and source["expected_target_quote_recorded"] and target["expected_target_quote_recorded"]
    if not both_complete:
        category = "contract_unavailable"
    elif not target_quotes_recorded:
        category = "target_polarity_quote_not_recorded_in_both_reports"
    elif source["opposite_outside_target_recorded"] or target["opposite_outside_target_recorded"]:
        category = "target_and_opposite_outside_quotes_recorded"
    elif source["uncertain_outside_target_recorded"] or target["uncertain_outside_target_recorded"]:
        category = "target_and_uncertain_outside_quotes_recorded"
    else:
        category = "target_recorded_without_other_opposite_or_uncertain_quote"
    return {**pair, "source_review": source, "edited_review": target,
        "chexbert_full_source_state": source_state, "chexbert_full_edited_state": target_state,
        "chexbert_full_both_text_states_extracted": source_state == pair["expected_source_text_state"] and target_state == pair["expected_edited_text_state"],
        "qwen_both_contracts_complete": both_complete,
        "qwen_target_quotes_recorded_in_both_reports": bool(target_quotes_recorded) if both_complete else None,
        "quoted_evidence_pattern": category, "semantic_correctness_independently_verified": False,
        "clinical_mismatch_verified": False}


def summarize_details(details):
    if not details or len(details) > 97:
        raise ValueError("bounded pair inventory required")

    def counts(rows):
        misses = [row for row in rows if not row["chexbert_full_both_text_states_extracted"]]
        return {"pairs": len(rows), "chexbert_full_both_text_states_extracted": len(rows)-len(misses),
            "qwen_both_contracts_complete": sum(row["qwen_both_contracts_complete"] for row in rows),
            "qwen_contract_unavailable_pairs": sum(not row["qwen_both_contracts_complete"] for row in rows),
            "qwen_target_quotes_recorded_in_both_reports": sum(row["qwen_target_quotes_recorded_in_both_reports"] is True for row in rows),
            "quoted_evidence_patterns": dict(Counter(row["quoted_evidence_pattern"] for row in rows)),
            "chexbert_miss_quoted_evidence_patterns": dict(Counter(row["quoted_evidence_pattern"] for row in misses))}

    unique = {(row["finding"], row["original_report_sha256"], row["edited_report_sha256"]): row for row in details}
    return {"case_linked": counts(details), "unique_finding_text_pairs": counts(list(unique.values())),
        "per_finding": {finding: counts([row for row in details if row["finding"] == finding]) for finding in FINDINGS},
        "interpretation": "Model-attributed, exact-source quotes only. Non-recorded evidence is not evidence of absence; capped quotes may omit assertions. No clinical correctness, causal localization or replacement scores."}


def checked_artifact(run_root, filename, schema):
    root = require_inside(run_root, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/"manifest.json")
    path = root/filename
    if manifest.get("schema_version") != schema or sha256_file(path) != manifest["artifacts"][filename]["sha256"]:
        raise ValueError("diagnostic source schema/hash mismatch")
    return read_json(path), path


def analyze(args):
    texts, resolver, source = load_report_texts(args.bank_run)
    bank = require_inside(args.bank_run, PROTECTED_ROOT, must_exist=True)
    bank_manifest = read_json(bank/"manifest.json")
    key_path = bank/"intervention_key.jsonl"
    if sha256_file(key_path) != bank_manifest["artifacts"][key_path.name]["sha256"]:
        raise ValueError("construction key hash mismatch")
    key = [json.loads(line) for line in key_path.read_text().splitlines() if line]
    resolver_index = {row["item_id"]: row for row in resolver}
    key_index = {row["item_id"]: row for row in key}
    if len(key_index) != len(key) or set(key_index) != set(resolver_index):
        raise ValueError("key/resolver inventory differs")
    _, pairs = sentence_bindings(key, {item: texts[row["report_sha256"]] for item, row in resolver_index.items()})
    if len(pairs) != 97 or {pair["finding"] for pair in pairs} != set(FINDINGS):
        raise ValueError("fixed pair/finding scope differs")
    review, review_path = checked_artifact(args.review_run, "evidence.json", REVIEW_SCHEMA)
    raw, raw_path = checked_artifact(args.review_run, "raw_responses.json", REVIEW_SCHEMA)
    context, context_path = checked_artifact(args.context_run, "diagnostic.json", CONTEXT_SCHEMA)
    if (review["source"] != source or review.get("finding_order") != list(FINDINGS)
            or review.get("primary_metric_eligible") is not False or review.get("producer", {}).get("frozen") is not True
            or review.get("verifier_received_ehr_images_scores_or_answer_key") is not False
            or context["source"]["manifest_sha256"] != source["bank_manifest_sha256"]
            or context["source"]["resolver_sha256"] != source["resolver_sha256"]
            or context["source_sha256"]["intervention_key"] != sha256_file(key_path)
            or context.get("primary_metric_eligible") is not False):
        raise ValueError("review/context provenance differs")
    records = {row["report_sha256"]: row for row in review["records"]}
    responses = {row["report_sha256"]: row for row in raw["records"]}
    if (len(records) != len(review["records"]) or len(responses) != len(raw["records"])
            or set(records) != set(texts) or set(responses) != set(texts)
            or len(records) != 100 or set(context["full_batch8_logits"]) != set(texts)):
        raise ValueError("distinct-text inventory differs")
    expected_links = sorted((row["item_id"], row["report_sha256"]) for row in resolver)
    if sorted((row["item_id"], row["report_sha256"]) for row in review["item_links"]) != expected_links:
        raise ValueError("review item links differ")
    for fingerprint, row in records.items():
        response = responses[fingerprint]
        if digest_text(response["response"]) != row["response_sha256"]:
            raise ValueError("raw response hash differs")
        if any(row[name] != response[name] for name in ("input_tokens", "output_tokens", "token_limit_reached")):
            raise ValueError("response token metadata differs")
        replay = decode_evidence(response["response"], texts[fingerprint], token_limit_reached=response["token_limit_reached"])
        if any(row[name] != replay[name] for name in ("contract_status", "contract_failure_reason", "findings")):
            raise ValueError("evidence parser replay differs")
    if review_summary(review["records"]) != review["summary"]:
        raise ValueError("review aggregate replay differs")
    details = [annotate_pair(pair, key_index[pair["item_id"]]["edit"], records, context["full_batch8_logits"]) for pair in pairs]
    controls = {row["case_id"]: row for row in key if row["intervention_type"] == "unchanged"}
    spacing = [row for row in key if row["intervention_type"] == "whitespace_only"]
    unavailable = changed = 0
    for row in spacing:
        old = records[resolver_index[controls[row["case_id"]]["item_id"]]["report_sha256"]]
        new = records[resolver_index[row["item_id"]]["report_sha256"]]
        if old["contract_status"] != "complete" or new["contract_status"] != "complete":
            unavailable += 1
        elif any(old["findings"][name]["state"] != new["findings"][name]["state"] for name in FINDINGS):
            changed += 1
    summary = {"schema_version": SCHEMA, "benchmark_split": "development", "source": source,
        "model_calls_in_analysis": 0, "raw_response_reparse_verified": True, "review": review["summary"],
        **summarize_details(details), "format_controls": {"case_linked_controls": len(spacing),
            "unavailable": unavailable, "state_vectors_changed": changed},
        "primary_metric_eligible": False, "targeted_repair_approved": False, "selection_changed": False,
        "clinical_localization_accuracy": None, "independent_clinical_ground_truth": False}
    sources = {"bank_manifest": bank/"manifest.json", "construction_key": key_path,
        "review": review_path, "raw_responses": raw_path, "chexbert_context": context_path}
    return summary, details, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-run", "review-run", "context-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    temporary = None
    try:
        summary, details, sources = analyze(args)
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        files = [write_private_json(temporary/"summary.json", summary),
            write_private_json(temporary/"pair_alignment.json", {"records": details})]
        lines = ["# Secondary report evidence alignment / 带证据的全文核查", "",
            "Only the post-inference analysis reads the construction key; model requests are report-only.",
            "Counts reflect model-attributed verbatim quotes, not independently verified semantics.", "",
            "| Finding | Case-linked pairs | Complete Qwen pairs | Target quotes in both reports | CheXbert full both states |",
            "|---|---:|---:|---:|---:|---:|"]
        for finding, row in summary["per_finding"].items():
            lines.append(f"| {finding} | {row['pairs']} | {row['qwen_both_contracts_complete']} | {row['qwen_target_quotes_recorded_in_both_reports']} | {row['chexbert_full_both_text_states_extracted']} |")
        lines += ["", "Failed/absent evidence stays unknown; all denominators and duplicate-aware counts remain.",
            "Each polarity is capped at two quotes. No quote recorded does not prove absence of a conflicting assertion.",
            "No source artifact, primary metric, threshold, selection or repair rule is changed.", ""]
        files.append(write_private_text(temporary/"summary.md", "\n".join(lines)))
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_secondary_evidence_alignment_no_selection_change"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
