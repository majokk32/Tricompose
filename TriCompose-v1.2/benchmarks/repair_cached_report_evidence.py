#!/usr/bin/env python3
"""Secondary cache-only quote alignment and conservative polarity veto.

No model loading, inference, answer-key-assisted repair or primary scoring.
Whitespace alignment restores exact original offsets. Narrow scope rules may
veto an assertion, but never move it into another polarity or create evidence.
This is not an official NegEx, ConText or NegBio implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import Counter
from pathlib import Path

from contracts import (
    PROTECTED_ROOT, commit_atomic_run, discard_atomic_run, new_atomic_run,
    sha256_file, write_private_json, write_private_text,
)
from analyze_report_evidence_qwen import (
    analyze as analyze_strict, annotate_pair, checked_artifact, summarize_details,
)
from verify_report_evidence_qwen import (
    FINDINGS, POLARITIES, SCHEMA as REVIEW_SCHEMA, EvidenceContractError,
    decode_evidence, digest_text, load_report_texts, unique_object,
)

SCHEMA = "tricompose-cached-report-evidence-repair-diagnostic-v1"
ALIGNMENT_VERSION = "unicode-whitespace-unique-offset-v1"
SCOPE_VERSION = "literal-four-finding-veto-v2-original-line-scope"

# Deliberately limited literal vocabulary, not a clinical synonym extractor.
MENTIONS = {
    "cardiomegaly": r"\bcardiomegaly\b",
    "consolidation": r"\bconsolidations?\b",
    "pleural_effusion": r"\b(?:pleural\s+)?effusions?\b",
    "pneumothorax": r"\b(?:pneumothora(?:x|ces)|ptx)\b",
}
BOUNDARY = re.compile(r"(?<!\d)[.!?;](?!\d)|\b(?:but|however|although|yet)\b", re.I)
NEG_PRE = re.compile(r"\b(?:no|without|negative\s+for|free\s+of|neither)\b", re.I)
NEG_POST = re.compile(r"^\s*(?:(?:is|are|was|were)\s+)?(?:absent\b|not\s+(?:seen|present|identified|evident|demonstrated|noted|detected)\b)", re.I)
POS_POST = re.compile(r"^\s*(?:is|are|was|were)\s+(?:present|seen|identified|evident|demonstrated|noted)\b", re.I)
UNCERTAIN_PRE = re.compile(r"\b(?:possible|possibly|suspected|questionable|may|might|cannot\s+(?:exclude|rule\s+out)|can\s+not\s+(?:exclude|rule\s+out)|not\s+excluded)\b", re.I)
UNCERTAIN_POST = re.compile(r"^\s*(?:(?:is|are|was|were)\s+)?(?:not\s+excluded|cannot\s+be\s+excluded|may\s+be|might\s+be)\b", re.I)
PSEUDO = re.compile(r"\b(?:no\s+(?:(?:interval|significant)\s+)?(?:change|increase)|not\s+only|without\s+contrast)\b", re.I)
CONTEXT = re.compile(r"\b(?:history\s+of|previous|prior|resolved|rule\s+out|evaluate\s+for)\b", re.I)
QUALIFIERS = frozenset({"large", "small", "moderate", "massive", "significant", "sizable", "sizeable", "focal", "left", "right", "bilateral", "apical", "basilar"})
NEG_GAP_WORDS = frozenset({"a", "an", "any", "evidence", "of", "sign", "signs", "and", "or", "nor", "the"}) | QUALIFIERS


def fold_with_offsets(text):
    """Collapse Unicode whitespace only; map each folded char to source chars."""
    if not isinstance(text, str):
        raise EvidenceContractError("invalid_source_type")
    chars, offsets, index = [], [], 0
    while index < len(text):
        end = index+1
        if text[index].isspace():
            while end < len(text) and text[end].isspace():
                end += 1
            chars.append(" ")
        else:
            chars.append(text[index])
        offsets.append((index, end))
        index = end
    return "".join(chars), offsets


def occurrences(text, fragment):
    start, found = 0, []
    while True:
        index = text.find(fragment, start)
        if index < 0:
            return found
        found.append(index)
        start = index+1


def align_quote(report, quote):
    if not isinstance(quote, str) or not 3 <= len(quote) <= 256 or quote != quote.strip():
        raise EvidenceContractError("invalid_quote_length_or_type")
    folded, offsets = fold_with_offsets(report)
    normalized, _ = fold_with_offsets(quote)
    candidates = occurrences(folded, normalized)
    # Check normalized uniqueness even when one exact hit exists; a second
    # whitespace-equivalent location must not be silently disambiguated.
    if len(candidates) > 1:
        raise EvidenceContractError("quote_location_ambiguous")
    if not candidates:
        raise EvidenceContractError("quote_not_in_source_even_after_whitespace")
    left = candidates[0]
    start, end = offsets[left][0], offsets[left+len(normalized)-1][1]
    source_quote = report[start:end]
    if fold_with_offsets(source_quote)[0] != normalized:
        raise EvidenceContractError("original_offset_reconstruction_failed")
    return {"quote": source_quote, "char_start": start, "char_end": end,
        "quote_sha256": digest_text(source_quote), "offset_unit": "unicode_codepoint",
        "returned_quote": quote, "returned_quote_sha256": digest_text(quote),
        "alignment_mode": "exact" if source_quote == quote else "whitespace_equivalent",
        "alignment_version": ALIGNMENT_VERSION}


def finding_record(spans):
    positive, negative, uncertain = (bool(spans[name]) for name in POLARITIES)
    conflict = positive and negative
    state = "uncertain" if conflict or uncertain else "positive" if positive else "negative" if negative else "unknown"
    return {"state": state, "opposed_quoted_assertions": conflict, "evidence": spans,
        "semantic_correctness_independently_verified": False}


def decode_aligned(response, report, *, token_limit_reached=False):
    try:
        if token_limit_reached:
            raise EvidenceContractError("token_limit_reached")
        if not isinstance(response, str) or len(response) > 32768:
            raise EvidenceContractError("invalid_response_type_or_length")
        if not isinstance(report, str) or not report.strip() or len(report) > 8192:
            raise EvidenceContractError("invalid_bounded_report")
        text = response.strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[8:-3].strip()
        elif text.startswith("```\n") and text.endswith("```"):
            text = text[4:-3].strip()
        try:
            payload = json.loads(text, object_pairs_hook=unique_object)
        except (TypeError, ValueError):
            raise EvidenceContractError("invalid_json_or_duplicate_key") from None
        if not isinstance(payload, dict) or set(payload) != set(FINDINGS):
            raise EvidenceContractError("finding_inventory_mismatch")
        findings = {}
        for finding in FINDINGS:
            obj = payload[finding]
            if not isinstance(obj, dict) or set(obj) != set(POLARITIES):
                raise EvidenceContractError("polarity_inventory_mismatch")
            spans, assigned = {}, set()
            for polarity in POLARITIES:
                quotes = obj[polarity]
                if not isinstance(quotes, list) or len(quotes) > 2:
                    raise EvidenceContractError("invalid_quote_list")
                spans[polarity] = []
                for quote in quotes:
                    span = align_quote(report, quote)
                    key = (span["char_start"], span["char_end"])
                    if key in assigned:
                        raise EvidenceContractError("duplicate_or_conflicting_quote")
                    assigned.add(key)
                    spans[polarity].append(span)
            findings[finding] = finding_record(spans)
        return {"contract_status": "complete", "contract_failure_reason": None, "findings": findings}
    except EvidenceContractError as exc:
        return {"contract_status": "failed_unavailable", "contract_failure_reason": str(exc),
            "findings": {name: finding_record({polarity: [] for polarity in POLARITIES}) for name in FINDINGS}}


def _list_gap(text):
    """Accept only a small disease-list syntax between cue and finding."""
    stripped = text
    for pattern in MENTIONS.values():
        stripped = re.sub(pattern, " ", stripped, flags=re.I)
    words = re.findall(r"\b\w+\b", stripped.lower())
    return len(words) <= 12 and set(words) <= NEG_GAP_WORDS and not re.search(r"[^\w\s,/-]", stripped)


def _mention_scope(text, mention, original, offsets):
    start, end = mention.span()
    boundaries = list(BOUNDARY.finditer(text))
    left = max((boundary.end() for boundary in boundaries if boundary.end() <= start), default=0)
    right = min((boundary.start() for boundary in boundaries if boundary.start() >= end), default=len(text))
    before, after = text[left:start], text[end:right]
    # Explicit independent clause: "no A, and B is present". Do not apply
    # this termination to "no A or B", which shares a negation scope.
    if POS_POST.search(after) and re.search(r",\s*and\s*$", before, re.I):
        before = ""
    # Alignment folding must not erase a separate assertion's original line
    # boundary. Terminate only for an explicit new presence clause after a
    # prior literal finding, not a wrapped "no\nB" or "no A or\nB" list.
    original_left = offsets[left][0] if left < len(offsets) else len(original)
    prefix = original[original_left:offsets[start][0]]
    newline = max(prefix.rfind("\n"), prefix.rfind("\r"))
    if POS_POST.search(after) and newline >= 0 and not prefix[newline+1:].strip():
        previous = prefix[:newline]
        if (any(re.search(pattern, previous, re.I) for pattern in MENTIONS.values())
                and not re.search(r"\b(?:no|without|or|and|nor)\s*$|[,/]\s*$", previous, re.I)):
            before = ""
    # A cue about another assertion cannot jump across a copula/independent
    # clause because _list_gap rejects its verb and other non-list words.
    if (UNCERTAIN_POST.search(after) or any(_list_gap(before[cue.end():]) for cue in UNCERTAIN_PRE.finditer(before))):
        return "uncertain", "explicit_uncertainty"
    if CONTEXT.search(before) or re.match(r"\s*(?:has\s+)?resolved\b", after, re.I):
        return None, "noncurrent_or_indication_context"
    if PSEUDO.search(before):
        return None, "pseudo_negation_scope_not_checked"
    cues = [cue for cue in NEG_PRE.finditer(before) if _list_gap(before[cue.end():])]
    if cues:
        gap_words = set(re.findall(r"\b\w+\b", before[cues[-1].end():].lower()))
        if gap_words & QUALIFIERS:
            return "uncertain", "qualified_absence_not_global_negative"
        return "negative", "explicit_pre_negation"
    if NEG_POST.search(after):
        return "negative", "explicit_post_negation"
    if POS_POST.search(after):
        return "positive", "explicit_presence_copula"
    return None, "scope_not_checked"


def scope_check(report, span, finding, model_polarity):
    """Inspect original source scope only; no key, expected state or score."""
    if finding not in FINDINGS or model_polarity not in POLARITIES:
        raise ValueError("invalid scope inventory")
    start, end = span["char_start"], span["char_end"]
    if (not 0 <= start < end <= len(report) or report[start:end] != span["quote"]
            or digest_text(span["quote"]) != span["quote_sha256"]):
        raise ValueError("source offset/hash mismatch")
    text, offsets = fold_with_offsets(report)
    decisions = []
    for mention in re.finditer(MENTIONS[finding], text, re.I):
        mstart, mend = mention.span()
        if not (start <= offsets[mstart][0] and offsets[mend-1][1] <= end):
            continue
        # Bare effusion with an explicitly non-pleural modifier is outside
        # this literal guard; do not equate it with pleural effusion.
        if finding == "pleural_effusion" and re.search(r"\b(?:pericardial|joint)\s*$", text[max(0, mstart-30):mstart], re.I):
            decisions.append((None, "nonpleural_effusion_not_checked"))
        else:
            decisions.append(_mention_scope(text, mention, report, offsets))
    blockers = {reason for _, reason in decisions if reason in {"noncurrent_or_indication_context", "nonpleural_effusion_not_checked"}}
    determinate = {state for state, _ in decisions if state is not None}
    reasons = sorted({reason for _, reason in decisions}) or ["literal_finding_not_covered"]
    if blockers:
        veto, reason = True, "unsupported_current_finding_context"
    elif len(determinate) > 1:
        veto, reason = True, "conflicting_scopes_in_one_quote"
    elif determinate == {"uncertain"} and model_polarity != "uncertain":
        veto, reason = True, "uncertainty_or_qualified_absence_veto"
    elif model_polarity == "positive" and determinate == {"negative"}:
        veto, reason = True, "positive_quote_explicitly_negated"
    elif model_polarity == "negative" and determinate == {"positive"}:
        veto, reason = True, "negative_quote_explicitly_present"
    else:
        veto, reason = False, None
    return {"version": SCOPE_VERSION, "veto": veto, "veto_reason": reason,
        "rule_suggested_states": sorted(determinate), "scope_reasons": reasons,
        "literal_mentions_checked": len(decisions), "independent_clinical_validation": False}


def guard_record(aligned, report):
    guarded = copy.deepcopy(aligned)
    for finding in FINDINGS:
        row = guarded["findings"][finding]
        accepted = {polarity: [] for polarity in POLARITIES}
        rejected = []
        for polarity in POLARITIES:
            for span in row["evidence"][polarity]:
                decision = scope_check(report, span, finding, polarity)
                annotated = {**span, "scope_guard": decision}
                if decision["veto"]:
                    rejected.append({**annotated, "model_polarity": polarity})
                else:
                    accepted[polarity].append(annotated)
        guarded["findings"][finding] = {**finding_record(accepted), "vetoed_evidence": rejected,
            "state_before_guard": row["state"], "scope_guard_version": SCOPE_VERSION}
        # Deleting a veto must not manufacture a determinate opposite state.
        # A rejected pos/neg assertion may indicate an unresolved conflict.
        if rejected and guarded["findings"][finding]["state"] in {"positive", "negative"}:
            guarded["findings"][finding]["state"] = "unknown"
            guarded["findings"][finding]["unresolved_veto_prevents_determinate_state"] = True
    return guarded


def summarize_stage(records):
    complete = [row for row in records if row["contract_status"] == "complete"]
    return {"distinct_texts": len(records), "complete_quote_contracts": len(complete),
        "failed_unavailable": len(records)-len(complete),
        "failure_reasons": dict(Counter(row["contract_failure_reason"] for row in records if row["contract_status"] != "complete")),
        "state_counts_all_records": {name: dict(Counter(row["findings"][name]["state"] for row in records)) for name in FINDINGS},
        "states_not_unknown": sum(row["findings"][name]["state"] != "unknown" for row in records for name in FINDINGS),
        "finding_record_denominator": len(records)*len(FINDINGS)}


def format_controls(key, resolver, records):
    by_id = {row["item_id"]: row["report_sha256"] for row in resolver}
    originals = {row["case_id"]: row for row in key if row["intervention_type"] == "unchanged"}
    result = Counter(case_linked_controls=0, unavailable=0, comparable=0, state_vectors_changed=0)
    distinct_pairs = set()
    for row in key:
        if row["intervention_type"] != "whitespace_only":
            continue
        old_hash, new_hash = by_id[originals[row["case_id"]]["item_id"]], by_id[row["item_id"]]
        old, new = records[old_hash], records[new_hash]
        distinct_pairs.add((old_hash, new_hash))
        result["case_linked_controls"] += 1
        if old["contract_status"] != "complete" or new["contract_status"] != "complete":
            result["unavailable"] += 1
        else:
            result["comparable"] += 1
            result["state_vectors_changed"] += any(old["findings"][name]["state"] != new["findings"][name]["state"] for name in FINDINGS)
    return {**dict(result), "unique_text_pairs": len(distinct_pairs)}


def load_blinded_cache(args):
    texts, resolver, source = load_report_texts(args.bank_run)
    review, review_path = checked_artifact(args.review_run, "evidence.json", REVIEW_SCHEMA)
    raw, raw_path = checked_artifact(args.review_run, "raw_responses.json", REVIEW_SCHEMA)
    if (review["source"] != source or review.get("finding_order") != list(FINDINGS)
            or review.get("producer", {}).get("frozen") is not True
            or review.get("producer", {}).get("model_file_sha256", {}).get("model.safetensors") != "26fa644c8e61aa8185da5055b61aa28d6897fee102c2f43173acab657c4257a1"
            or review.get("primary_metric_eligible") is not False
            or review.get("verifier_received_ehr_images_scores_or_answer_key") is not False):
        raise ValueError("frozen blinded cache provenance differs")
    records = {row["report_sha256"]: row for row in review["records"]}
    responses = {row["report_sha256"]: row for row in raw["records"]}
    if (len(records) != len(review["records"]) or len(responses) != len(raw["records"])
            or set(records) != set(texts) or set(responses) != set(texts) or len(records) != 100):
        raise ValueError("fixed distinct-text inventory differs")
    if sorted((row["item_id"], row["report_sha256"]) for row in review["item_links"]) != sorted((row["item_id"], row["report_sha256"]) for row in resolver):
        raise ValueError("item link inventory differs")
    for fingerprint, row in records.items():
        response = responses[fingerprint]
        if digest_text(response["response"]) != row["response_sha256"]:
            raise ValueError("cached response hash differs")
        if any(row[name] != response[name] for name in ("input_tokens", "output_tokens", "token_limit_reached")):
            raise ValueError("response token metadata differs")
        strict = decode_evidence(response["response"], texts[fingerprint], token_limit_reached=response["token_limit_reached"])
        if any(row[name] != strict[name] for name in ("contract_status", "contract_failure_reason", "findings")):
            raise ValueError("strict cache replay differs")
    return texts, resolver, records, responses, source, {"review": review_path, "raw_responses": raw_path,
        "review_manifest": review_path.parent/"manifest.json", "bank_manifest": Path(args.bank_run)/"manifest.json"}


def run(args):
    texts, resolver, strict, responses, source, sources = load_blinded_cache(args)
    initial_hashes = {name: sha256_file(path) for name, path in sources.items()}
    stages = {"strict": strict, "whitespace_aligned": {}, "scope_guarded": {}}
    alignment_counts, veto_reasons = Counter(), Counter()
    veto_by_finding, changed_states, promotions, recovered, lost = Counter(), Counter(), 0, 0, 0
    # Blinded pass completes before the post-hoc analyzer loads the key.
    for fingerprint, report in sorted(texts.items()):
        raw = responses[fingerprint]
        aligned = decode_aligned(raw["response"], report, token_limit_reached=raw["token_limit_reached"])
        guarded = guard_record(aligned, report)
        meta = {"report_sha256": fingerprint, "response_sha256": digest_text(raw["response"])}
        stages["whitespace_aligned"][fingerprint] = {**meta, **aligned}
        stages["scope_guarded"][fingerprint] = {**meta, **guarded}
        recovered += strict[fingerprint]["contract_status"] != "complete" and aligned["contract_status"] == "complete"
        lost += strict[fingerprint]["contract_status"] == "complete" and aligned["contract_status"] != "complete"
        for finding in FINDINGS:
            row = aligned["findings"][finding]
            final = guarded["findings"][finding]
            for polarity in POLARITIES:
                alignment_counts.update(span["alignment_mode"] for span in row["evidence"][polarity])
            for span in final["vetoed_evidence"]:
                veto_reasons[span["scope_guard"]["veto_reason"]] += 1
                veto_by_finding[finding] += 1
            if row["state"] != final["state"]:
                changed_states[f"{row['state']}->{final['state']}"] += 1
            promotions += row["state"] == "unknown" and final["state"] != "unknown"
    if promotions or any(key not in {"positive->unknown", "negative->unknown", "uncertain->unknown"} for key in changed_states):
        raise ValueError("guard promoted evidence or flipped polarity")
    if initial_hashes != {name: sha256_file(path) for name, path in sources.items()}:
        raise ValueError("source changed during blinded repair")

    # Evaluation only: the existing analyzer verifies all construction/context
    # provenance and gives the fixed 97 pairs. No key reaches the repair pass.
    strict_analysis, pairs, evaluation_sources = analyze_strict(args)
    sources.update(evaluation_sources)
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    key_path = evaluation_sources["construction_key"]
    key = [json.loads(line) for line in key_path.read_text().splitlines() if line]
    by_id = {row["item_id"]: row for row in key}
    context, _ = checked_artifact(args.context_run, "diagnostic.json", "tricompose-chexbert-polarity-context-diagnostic-v1")
    details, comparisons = {}, {}
    for stage, records in stages.items():
        rows = [annotate_pair(pair, by_id[pair["item_id"]]["edit"], records, context["full_batch8_logits"]) for pair in pairs]
        details[stage] = rows
        comparisons[stage] = {"quote_contract": summarize_stage(list(records.values())),
            "posthoc_mechanical_assertion_checks": summarize_details(rows),
            "format_controls": format_controls(key, resolver, records)}
    if comparisons["strict"]["posthoc_mechanical_assertion_checks"]["case_linked"] != strict_analysis["case_linked"]:
        raise ValueError("strict baseline comparison replay differs")
    if source_hashes != {name: sha256_file(path) for name, path in sources.items()}:
        raise ValueError("immutable source changed during analysis")
    payload = {"schema_version": SCHEMA, "scope": "secondary_development_cache_interface_diagnostic",
        "benchmark_split": "development", "source": source, "model_calls": 0,
        "source_sha256": source_hashes, "alignment_version": ALIGNMENT_VERSION, "scope_version": SCOPE_VERSION,
        "stages": comparisons, "recovered_quote_contracts": recovered,
        "lost_quote_contracts": lost,
        "aligned_quote_modes": dict(alignment_counts), "vetoed_quote_reasons": dict(veto_reasons),
        "vetoed_quotes_per_finding": dict(veto_by_finding), "guard_state_transitions": dict(changed_states),
        "unknown_promotions": promotions, "repair_received_answer_key": False,
        "repair_blinded_pass_completed_before_key_read": True,
        "posthoc_evaluation_used_construction_key": True, "raw_response_strict_replay_verified": True,
        "primary_metric_eligible": False, "targeted_repair_approved": False, "selection_changed": False,
        "clinical_localization_accuracy": None, "independent_clinical_ground_truth": False,
        "external_api_used": False, "reserved_cases_opened": 0,
        "interpretation": "Quote-contract recovery is not semantic accuracy. Rules only veto; unchecked retained quotes remain unvalidated. Mechanical edits and shared templates are not independent clinical gold."}
    return payload, {name: list(records.values()) for name, records in stages.items() if name != "strict"}, details, sources


def render_summary(summary):
    lines = ["# Cached evidence interface repair / 缓存证据接口修复", "",
        "Secondary diagnostic only. Zero model calls; no generation, primary score or selection changed.", "",
        "| Stage | Complete quote contracts / 100 | Unavailable | Non-unknown states / 400 | Target quotes in both / 97 | Unique target pairs / 48 | Comparable spacing controls / 48 | Changed state vectors (comparable only) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, stage in summary["stages"].items():
        contract, checks, controls = stage["quote_contract"], stage["posthoc_mechanical_assertion_checks"], stage["format_controls"]
        lines.append(f"| {name} | {contract['complete_quote_contracts']} | {contract['failed_unavailable']} | {contract['states_not_unknown']} | {checks['case_linked']['qwen_target_quotes_recorded_in_both_reports']} | {checks['unique_finding_text_pairs']['qwen_target_quotes_recorded_in_both_reports']} | {controls['comparable']} | {controls['state_vectors_changed']} |")
    lines += ["", "Whitespace is collapsed only to locate a unique source span; stored quote/offsets always refer to unchanged original text.",
        "No case, punctuation, unit, number, negation or paraphrase changes are accepted.",
        "Negation/uncertainty rules veto unsafe evidence, never relabel it. Coverage losses remain visible.",
        "Rules use source scope only; only the later evaluation reads the intervention key.",
        "Counts on repeated mechanical edits are NOT independent clinical accuracy, repair success or primary scores.",
        "All 100 texts, 97 case-linked pairs, 48 unique finding/text pairs and unavailable controls remain in denominators.",
        "Uncovered synonyms and unsupported grammar remain explicitly unvalidated. No quote proves exhaustive evidence.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-run", "review-run", "context-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        summary, records, details, sources = run(args)
        files = [write_private_json(temporary/"summary.json", summary),
            write_private_json(temporary/"repaired_evidence.json", {"schema_version": SCHEMA, "stages": records}),
            write_private_json(temporary/"pair_alignment.json", {"stages": details}),
            write_private_text(temporary/"summary.md", render_summary(summary))]
        if summary["source_sha256"] != {name: sha256_file(path) for name, path in sources.items()}:
            raise ValueError("source changed before commit")
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "source_sha256": summary["source_sha256"],
            "primary_metric_eligible": False, "selection_changed": False,
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        # Never emit cached text, JSON responses or exception details.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_cache_evidence_interface_diagnostic", "model_calls": 0,
        "recovered_quote_contracts": summary["recovered_quote_contracts"],
        "vetoed_quotes": sum(summary["vetoed_quote_reasons"].values()), "selection_changed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
