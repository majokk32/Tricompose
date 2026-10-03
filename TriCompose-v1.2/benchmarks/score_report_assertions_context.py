#!/usr/bin/env python3
"""Official frozen medspaCy ConText diagnostic, approved CPU Slurm only.

Uses a blank English tokenizer and official sentence/context rules; no
learned NLP/CXR/EHR model, downloaded checkpoint, custom negation rule or key.
The same limited literal finding vocabulary is reused and disclosed.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tricompose_v12.report_assertions import (
    FINDINGS, CONTEXT_FLAGS, checked_span, context_state, aggregate_context_mentions,
)
from contracts import sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json
from report_assertion_challenge import load_inputs, SCORE_SCHEMA, FROZEN_GUARD_SHA

VERSIONS = {"medspacy": "1.3.1", "spacy": "3.7.5", "PyRuSH": "1.0.12"}


def serialize_document(doc, text):
    if doc.text != text:
        raise ValueError("parser changed original text")
    records = {name: {"mentions": []} for name in FINDINGS}
    for entity in doc.ents:
        if entity.label_ not in FINDINGS:
            raise ValueError("finding inventory differs")
        flags = {name: getattr(entity._, name) for name in CONTEXT_FLAGS}
        state = context_state(flags)
        modifiers = []
        for modifier in entity._.modifiers:
            start, end = modifier.modifier_span
            left, right = modifier.scope_span
            cue, scope = doc[start:end], doc[left:right]
            modifiers.append({"category": modifier.category, "direction": modifier.direction,
                "cue": checked_span(text, cue.start_char, cue.end_char),
                "scope": checked_span(text, scope.start_char, scope.end_char)})
        records[entity.label_]["mentions"].append({"span": checked_span(text, entity.start_char, entity.end_char),
            "flags": flags, "context_state": state, "modifiers": modifiers,
            "unmodified_mention_is_not_verified_positive": not modifiers,
            "independent_clinical_validation": False})
    for row in records.values():
        row["state"] = aggregate_context_mentions(row["mentions"])
    return records


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved CPU Slurm required before parser/input access")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError("pinned parser versions differ")
    resolver, texts, source = load_inputs(args.bank_run)
    installed = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        import medspacy
        from medspacy.context.context import ConText
        from medspacy.context.context_modifier import ConTextModifier
        from repair_cached_report_evidence import MENTIONS, scope_check
        guard = Path(scope_check.__code__.co_filename)
        if sha256_file(guard) != FROZEN_GUARD_SHA:
            raise ValueError("frozen literal vocabulary/guard changed")
        nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_context"])
        if nlp.pipe_names != ["medspacy_pyrush", "medspacy_context"] or nlp.vocab.vectors.size:
            raise ValueError("unexpected trained NLP component or vectors")
        context = nlp.get_pipe("medspacy_context")
        package_root = Path(medspacy.__file__).resolve().parents[1]
        assets = {"context_rules": Path(context.DEFAULT_RULES_FILEPATH),
            "sentence_rules": package_root/"resources/en/rush_rules.tsv",
            "context_code": Path(ConText.__init__.__code__.co_filename),
            "modifier_code": Path(ConTextModifier.__init__.__code__.co_filename),
            "literal_vocabulary": guard}
        hashes = {name: sha256_file(path) for name, path in assets.items()}
        initial_rules = [rule.to_dict() for rule in context.rules]

        def forward():
            evidence = {}
            for h in sorted(texts):
                text = texts[h]
                doc = nlp.make_doc(text)
                spans = []
                for finding in FINDINGS:
                    for mention in re.finditer(MENTIONS[finding], text, re.I):
                        span = doc.char_span(*mention.span(), label=finding, alignment_mode="strict")
                        if span is None:
                            raise ValueError("literal/token span alignment differs")
                        spans.append(span)
                spans.sort(key=lambda span: (span.start, span.end, span.label_))
                if any(left.end > right.start for left, right in zip(spans, spans[1:])):
                    raise ValueError("overlapping literal spans")
                doc.ents = spans
                evidence[h] = serialize_document(nlp(doc), text)
            return evidence

        primary, replay = forward(), forward()
    if initial_rules != [rule.to_dict() for rule in context.rules] or hashes != {name: sha256_file(path) for name, path in assets.items()}:
        raise ValueError("official rules/code changed during diagnostic")
    changed = sum(primary[h] != replay[h] for h in texts)
    records = [{"item_id": row["item_id"], "report_sha256": row["report_sha256"], "status": "complete",
        "finding_states": {name: primary[row["report_sha256"]][name]["state"] for name in FINDINGS},
        "finding_evidence": primary[row["report_sha256"]]} for row in resolver]
    return {"schema_version": SCORE_SCHEMA, "source": source, "records": records,
        "producer": {"model_id": "medspacy_context", "frozen": True, "versions": versions,
            "installed_distribution_versions": installed,
            "asset_sha256": hashes, "official_source": "https://github.com/medspacy/medspacy/tree/1.3.1",
            "pipeline": nlp.pipe_names, "literal_vocabulary_shared_with_old_guard": True,
            "trained_models_loaded": False, "context_rule_count": len(initial_rules)},
        "counts": {"authored_texts": 56, "documents_including_replay": 112},
        "batch_replay": {"rerun_changed_texts": changed, "batching_not_applicable": True},
        "scorer_read_reference_key": False, "model_received_reference_states": False,
        "official_rules_modified": False, "primary_metric_eligible": False,
        "selection_changed": False, "targeted_repair_approved": False,
        "independent_clinical_accuracy": None, "expert_reviewed": False,
        "unmodified_mention_default": "positive_baseline_not_verified_clinical_assertion",
        "evidence_dependency": "same_report_text_not_independent_clinical_votes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    temporary = None
    os.umask(0o007)
    started = time.monotonic()
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = run(args)
        payload["elapsed_seconds"] = round(time.monotonic()-started, 3)
        path = write_private_json(temporary/"predictions.json", payload)
        library = Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_assertions.py"
        write_private_json(temporary/"manifest.json", {"schema_version": SCORE_SCHEMA, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "assertion_library_sha256": sha256_file(library),
            "source": payload["source"], "artifacts": {path.name: {"sha256": sha256_file(path)}},
            "primary_metric_eligible": False})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_official_context_diagnostic", "elapsed_seconds": payload["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
