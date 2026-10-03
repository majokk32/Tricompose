#!/usr/bin/env python3
"""Line-preserving horizontal-whitespace interface, exact frozen cache only.

This is an engineering input/cache contract, not independent model robustness
or clinical validation. No model imports, inference, API or primary scoring.
Original reports and original-format cached responses remain immutable.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from contracts import (
    commit_atomic_run, discard_atomic_run, new_atomic_run, read_json,
    sha256_file, write_private_json, write_private_text,
)
from analyze_report_evidence_qwen import analyze as analyze_strict, annotate_pair, checked_artifact, summarize_details
from repair_cached_report_evidence import (
    SCHEMA as REPAIR_SCHEMA, align_quote, decode_aligned, format_controls,
    guard_record, load_blinded_cache, summarize_stage,
)
from verify_report_evidence_qwen import FINDINGS, POLARITIES, PROMPT, digest_text, request_messages

SCHEMA = "tricompose-canonical-report-evidence-cache-interface-v1"
NORMALIZER_VERSION = "horizontal-whitespace-preserve-original-lines-v1"
VERTICAL_WHITESPACE = frozenset("\r\n\v\f\x1c\x1d\x1e\x1f\u0085\u2028\u2029")


class CacheMissError(ValueError):
    """A new inference needs separate Slurm disclosure/approval, never fallback."""


def digest_object(value):
    return digest_text(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")))


def horizontal(char):
    return char.isspace() and char not in VERTICAL_WHITESPACE


def canonicalize_report(report):
    """Collapse horizontal runs to one ASCII space; never strip or flatten lines."""
    request_messages(report)  # Same bounded text contract as the old verifier.
    chars, offsets, index = [], [], 0
    while index < len(report):
        end = index+1
        if horizontal(report[index]):
            while end < len(report) and horizontal(report[end]):
                end += 1
            chars.append(" ")
        else:
            chars.append(report[index])
        offsets.append([index, end])
        index = end
    canonical = "".join(chars)
    if ("".join(char for char in canonical if not horizontal(char)) !=
            "".join(char for char in report if not horizontal(char))):
        raise ValueError("non-horizontal-whitespace content changed")
    return {"text": canonical, "offsets": offsets,
        "source_sha256": digest_text(report), "canonical_sha256": digest_text(canonical),
        "normalizer_version": NORMALIZER_VERSION, "offset_unit": "unicode_codepoint"}


def build_input_plan(texts):
    if not texts or len(texts) > 100:
        raise ValueError("bounded nonempty source inventory required")
    inputs, mappings = {}, {}
    for fingerprint, report in sorted(texts.items()):
        canonical = canonicalize_report(report)
        if fingerprint != canonical["source_sha256"]:
            raise ValueError("source text hash differs")
        target_hash = canonical["canonical_sha256"]
        if target_hash in inputs and inputs[target_hash]["text"] != canonical["text"]:
            raise ValueError("canonical hash collision")
        inputs.setdefault(target_hash, {"text": canonical["text"], "canonical_sha256": target_hash,
            "request_sha256": digest_object(request_messages(canonical["text"])), "source_sha256s": []})
        inputs[target_hash]["source_sha256s"].append(fingerprint)
        mappings[fingerprint] = {key: value for key, value in canonical.items() if key != "text"}
    return inputs, mappings


def group_size_distribution(inputs):
    # JSON object names are strings. Keep the in-memory and persisted contract
    # identical rather than relying on json.dump's integer-key conversion.
    return {str(size): count for size, count in sorted(Counter(len(row["source_sha256s"]) for row in inputs.values()).items())}


def checked_cache_identity(args, sources):
    """Pin the frozen request/producer; reuse is within this historical run only."""
    review = read_json(sources["review"])
    manifest = read_json(sources["review_manifest"])
    producer = review["producer"]
    if (producer.get("prompt_sha256") != digest_text(PROMPT)
            or producer.get("max_new_tokens") != 512 or producer.get("do_sample") is not False
            or producer.get("frozen") is not True
            or manifest.get("program_sha256") != sha256_file(Path(request_messages.__code__.co_filename))):
        raise ValueError("cached model request or producer code differs")
    identity = {"producer": producer, "execution": review["execution"],
        "source_review_sha256": sha256_file(sources["review"]),
        "source_review_manifest_sha256": sha256_file(sources["review_manifest"]),
        "scope": "report_only_same_completed_frozen_run_cache"}
    return {"identity": identity, "identity_sha256": digest_object(identity)}


def project_response(response, report, canonical_input, mapping, *, token_limit_reached=False):
    """Restore quotes to the unchanged original source and verify inverse mapping."""
    if (mapping["source_sha256"] != digest_text(report)
            or mapping["canonical_sha256"] != digest_text(canonical_input["text"])
            or mapping != {key: value for key, value in canonicalize_report(report).items() if key != "text"}):
        raise ValueError("canonical/original coordinate map differs")
    aligned = decode_aligned(response, report, token_limit_reached=token_limit_reached)
    for finding in FINDINGS:
        for polarity in POLARITIES:
            for span in aligned["findings"][finding]["evidence"][polarity]:
                canonical_span = align_quote(canonical_input["text"], span["returned_quote"])
                left, right = canonical_span["char_start"], canonical_span["char_end"]
                original_left, original_right = mapping["offsets"][left][0], mapping["offsets"][right-1][1]
                if (original_left, original_right) != (span["char_start"], span["char_end"]):
                    raise ValueError("inverse quote mapping differs")
                span.update(canonical_char_start=left, canonical_char_end=right,
                    canonical_quote_sha256=canonical_span["quote_sha256"],
                    canonical_input_sha256=mapping["canonical_sha256"], inverse_offset_verified=True)
    return aligned, guard_record(aligned, report)


def materialize_blinded(texts, responses, cache_identity):
    """No answer key or expected state is accepted by this interface."""
    inputs, mappings = build_input_plan(texts)
    missing = set(inputs)-set(responses)
    if missing:
        raise CacheMissError("exact canonical inputs require separately approved inference")
    stages = {"canonical_input_aligned": {}, "canonical_input_guarded": {}}
    for fingerprint, report in sorted(texts.items()):
        mapping = mappings[fingerprint]
        target_hash = mapping["canonical_sha256"]
        canonical_input = inputs[target_hash]
        raw = responses[target_hash]
        if raw["report_sha256"] != target_hash:
            raise ValueError("cache response is not bound to exact model input")
        aligned, guarded = project_response(raw["response"], report, canonical_input, mapping,
            token_limit_reached=raw["token_limit_reached"])
        cache_key = {"identity_sha256": cache_identity["identity_sha256"],
            "request_sha256": canonical_input["request_sha256"], "canonical_input_sha256": target_hash}
        meta = {"report_sha256": fingerprint, "canonical_input_sha256": target_hash,
            "response_sha256": digest_text(raw["response"]), "cache_key_sha256": digest_object(cache_key),
            "cache_origin_report_sha256": target_hash, "new_model_call": False}
        stages["canonical_input_aligned"][fingerprint] = {**meta, **aligned}
        stages["canonical_input_guarded"][fingerprint] = {**meta, **guarded}
    # Equality is a required engineering contract, not a clinical test result.
    for stage in stages.values():
        for row in inputs.values():
            states = {tuple(stage[h]["findings"][name]["state"] for name in FINDINGS) for h in row["source_sha256s"]}
            statuses = {stage[h]["contract_status"] for h in row["source_sha256s"]}
            if len(states) != 1 or len(statuses) != 1:
                raise ValueError("equivalent inputs have inconsistent source projection")
    return inputs, mappings, stages


def run(args):
    texts, resolver, _, responses, source, sources = load_blinded_cache(args)
    initial_hashes = {name: sha256_file(path) for name, path in sources.items()}
    identity = checked_cache_identity(args, sources)
    inputs, mappings, stages = materialize_blinded(texts, responses, identity)
    if initial_hashes != {name: sha256_file(path) for name, path in sources.items()}:
        raise ValueError("source changed during blinded materialization")

    # Only now access construction/context evidence for post-hoc diagnostics.
    _, pairs, evaluation_sources = analyze_strict(args)
    sources.update(evaluation_sources)
    previous, previous_path = checked_artifact(args.repair_run, "summary.json", REPAIR_SCHEMA)
    if (previous["source"] != source or previous.get("primary_metric_eligible") is not False
            or previous.get("selection_changed") is not False
            or read_json(previous_path.parent/"manifest.json")["program_sha256"] != sha256_file(Path(guard_record.__code__.co_filename))):
        raise ValueError("prior repair diagnostic provenance/code differs")
    sources["previous_repair_summary"] = previous_path
    sources["previous_repair_manifest"] = previous_path.parent/"manifest.json"
    context, _ = checked_artifact(args.context_run, "diagnostic.json", "tricompose-chexbert-polarity-context-diagnostic-v1")
    key = [json.loads(line) for line in sources["construction_key"].read_text().splitlines() if line]
    by_id = {row["item_id"]: row for row in key}
    details, comparisons = {}, dict(previous["stages"])
    for stage, records in stages.items():
        rows = [annotate_pair(pair, by_id[pair["item_id"]]["edit"], records, context["full_batch8_logits"]) for pair in pairs]
        details[stage] = rows
        canonical_records = [records[inputs[fingerprint]["source_sha256s"][0]] for fingerprint in sorted(inputs)]
        comparisons[stage] = {"quote_contract": summarize_stage(list(records.values())),
            "distinct_canonical_input_quote_contract": summarize_stage(canonical_records),
            "posthoc_mechanical_assertion_checks": summarize_details(rows),
            "format_controls": {**format_controls(key, resolver, records),
                "interpretation": "shared identical cached input; NOT independently measured model invariance"}}
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    summary = {"schema_version": SCHEMA, "normalizer_version": NORMALIZER_VERSION,
        "scope": "secondary_development_input_cache_engineering", "benchmark_split": "development",
        "source": source, "source_sha256": source_hashes, "distinct_source_texts": len(texts),
        "distinct_canonical_inputs": len(inputs), "exact_input_cache_hits": len(inputs), "model_calls": 0,
        "group_size_distribution": group_size_distribution(inputs),
        "cache_identity": identity, "stages": comparisons, "horizontal_whitespace_only": True,
        "original_line_breaks_preserved": True, "inverse_quote_offsets_verified": True,
        "format_equalities_by_construction": True, "independent_format_robustness_accuracy": None,
        "repair_materialized_before_key_read": True, "repair_received_answer_key": False,
        "primary_metric_eligible": False, "targeted_repair_approved": False, "selection_changed": False,
        "clinical_localization_accuracy": None, "independent_clinical_ground_truth": False,
        "reserved_cases_opened": 0, "external_api_used": False,
        "interpretation": "Input de-duplication restores reproducibility only. Zero format differences reuse the same historical response and do not show clinical improvement or independent model invariance."}
    return summary, inputs, mappings, {name: list(rows.values()) for name, rows in stages.items()}, details, sources


def render_summary(summary):
    lines = ["# Canonical report input/cache interface / 统一报告输入与缓存接口", "",
        "Only horizontal whitespace is normalized; every original line break and source file is preserved.",
        "All canonical inputs exactly match inputs in the completed frozen Qwen run. New model calls: 0.", "",
        "| Stage | Complete source projections / 100 | Non-unknown states / 400 | Target quotes in both / 97 | Comparable format controls / 48 | Format state differences |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in ("strict", "whitespace_aligned", "scope_guarded", "canonical_input_aligned", "canonical_input_guarded"):
        stage = summary["stages"][name]
        c, checks, controls = stage["quote_contract"], stage["posthoc_mechanical_assertion_checks"], stage["format_controls"]
        lines.append(f"| {name} | {c['complete_quote_contracts']} | {c['states_not_unknown']} | {checks['case_linked']['qwen_target_quotes_recorded_in_both_reports']} | {controls['comparable']} | {controls['state_vectors_changed']} |")
    lines += ["", "IMPORTANT: identical canonical inputs share one cached response. Format equality is by construction, NOT independent robustness or clinical accuracy.",
        "Nonmatching or missing canonical inputs are refused; no approximate response reuse or model fallback is allowed.",
        "Failed contracts and missing findings remain unknown. Existing original-format responses are retained as the observed instability baseline.",
        "Quotes retain original Unicode offsets plus verified inverse canonical offsets. The evidence guard never flips polarity.",
        "This does not establish clinical correctness, reliable error localization, repair success or better selected triples.",
        "No EHR/CXR/report regeneration, primary score update, threshold tuning, reserved case access or Slurm submission.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-run", "review-run", "context-run", "repair-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    temporary = None
    os.umask(0o007)
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        summary, inputs, mappings, records, details, sources = run(args)
        files = [write_private_json(temporary/"summary.json", summary),
            write_private_text(temporary/"summary.md", render_summary(summary)),
            write_private_json(temporary/"canonical_inputs.json", {"normalizer_version": NORMALIZER_VERSION, "inputs": inputs}),
            write_private_json(temporary/"source_mappings.json", {"mappings": mappings}),
            write_private_json(temporary/"evidence.json", {"stages": records}),
            write_private_json(temporary/"pair_alignment.json", {"stages": details})]
        if summary["source_sha256"] != {name: sha256_file(path) for name, path in sources.items()}:
            raise ValueError("immutable sources changed before commit")
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "source_sha256": summary["source_sha256"],
            "normalizer_version": NORMALIZER_VERSION, "model_calls": 0,
            "primary_metric_eligible": False, "selection_changed": False,
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "refused_new_inference_needed" if isinstance(exc, CacheMissError) else "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_canonical_input_cache_interface", "model_calls": 0,
        "canonical_inputs": summary["distinct_canonical_inputs"], "selection_changed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
