#!/usr/bin/env python3
"""Development-only context/batch diagnosis with unchanged frozen CheXbert.

Not a replacement evaluator: sentence-only arms remove context deliberately.
No source edits, clinical gold, thresholds, selection, training or repair.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

from contracts import (
    PROTECTED_ROOT, commit_atomic_run, discard_atomic_run, new_atomic_run,
    read_json, require_inside, sha256_file, write_private_json, write_private_text,
)
from extract_report_labels_chexbert import CHEXBERT_CLASS_TO_STATE, CHEXBERT_ORDER
from prepare_expanded_polarity import load_expanded_bank
from score_report_polarity import report_text

SCHEMA = "tricompose-chexbert-polarity-context-diagnostic-v1"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_wrapper(text):
    return text.strip().replace("\n", " ").replace("\\s+", " ").replace("\\s+(?=[\\.,])", "").strip()


def normalize_official_csv(text):
    # Stanford src/bert_tokenizer.py:get_impressions_from_csv, no section selection.
    return re.sub(r"\s+", " ", text.strip().replace("\n", " ")).strip()


def sentence_bindings(key, texts):
    """Reconstruct every original exactly; no failure-based example selection."""
    controls = {row["case_id"]: row["item_id"] for row in key if row["intervention_type"] == "unchanged"}
    if len(controls) != sum(row["intervention_type"] == "unchanged" for row in key):
        raise ValueError("duplicate unchanged control")
    sentences, pairs, seen = {}, [], set()
    for row in key:
        if row["intervention_type"] in {"unchanged", "whitespace_only"}:
            continue
        if row["intervention_type"] != "minimal_polarity_flip" or row["finding"] not in CHEXBERT_ORDER:
            raise ValueError("unsupported diagnostic intervention")
        group = (row["case_id"], row["finding"])
        if group in seen:
            raise ValueError("duplicate case/finding intervention")
        seen.add(group)
        old_id = controls[row["case_id"]]
        old, new, edit = texts[old_id], texts[row["item_id"]], row["edit"]
        start, end = edit["span_start"], edit["span_end"]
        before, after = edit["source_statement"], edit["replacement_statement"]
        if not 0 <= start < end <= len(old) or old[start:end] != before:
            raise ValueError("source edit span differs")
        if new != old[:start] + after + old[end:]:
            raise ValueError("unrelated bytes changed")
        source, target = edit["source_text_assertion"], edit["edited_text_assertion"]
        if {source, target} != {"positive", "negative"}:
            raise ValueError("unsupported expected text polarity")
        for text in (before, after):
            sentences[text_hash(text)] = text
        pairs.append({"case_id": row["case_id"], "item_id": row["item_id"], "finding": row["finding"],
            "original_report_sha256": text_hash(old), "edited_report_sha256": text_hash(new),
            "original_sentence_sha256": text_hash(before), "edited_sentence_sha256": text_hash(after),
            "expected_source_text_state": source, "expected_edited_text_state": target})
    return sentences, pairs


def decode_heads(logits):
    if len(logits) != len(CHEXBERT_ORDER):
        raise ValueError("wrong head inventory")
    result = {}
    for index, (finding, values) in enumerate(zip(CHEXBERT_ORDER, logits, strict=True)):
        if len(values) != (2 if index == 13 else 4) or not all(math.isfinite(value) for value in values):
            raise ValueError("invalid diagnostic logits")
        winner = max(range(len(values)), key=values.__getitem__)
        ordered = sorted(values, reverse=True)
        result[finding] = {"state": CHEXBERT_CLASS_TO_STATE[winner], "class_index": winner,
            "raw_logits": values, "top_two_raw_logit_margin": ordered[0]-ordered[1]}
    return result


def diagnose_pairs(pairs, full, sentences):
    details = []
    for pair in pairs:
        finding = pair["finding"]
        old = full[pair["original_report_sha256"]][finding]
        new = full[pair["edited_report_sha256"]][finding]
        source = sentences[pair["original_sentence_sha256"]][finding]
        target = sentences[pair["edited_sentence_sha256"]][finding]
        expected_source, expected_target = pair["expected_source_text_state"], pair["expected_edited_text_state"]
        details.append({**pair, "full_source_state": old["state"], "full_edited_state": new["state"],
            "sentence_source_state": source["state"], "sentence_edited_state": target["state"],
            "full_both_text_states_extracted": old["state"] == expected_source and new["state"] == expected_target,
            "sentence_both_text_states_extracted": source["state"] == expected_source and target["state"] == expected_target,
            "full_edited_mismatch_sentence_match": new["state"] != expected_target and target["state"] == expected_target,
            "edited_mismatch_both_contexts": new["state"] != expected_target and target["state"] != expected_target,
            "full_edited_top_two_raw_logit_margin": new["top_two_raw_logit_margin"],
            "sentence_edited_top_two_raw_logit_margin": target["top_two_raw_logit_margin"],
            "clinical_mismatch_verified": False})
    summary = {}
    for finding in sorted({pair["finding"] for pair in pairs}):
        rows = [row for row in details if row["finding"] == finding]
        unique = {(row["original_report_sha256"], row["edited_report_sha256"]): row for row in rows}
        metrics = ("full_both_text_states_extracted", "sentence_both_text_states_extracted",
                   "full_edited_mismatch_sentence_match", "edited_mismatch_both_contexts")
        summary[finding] = {"case_linked_pairs": len(rows), "unique_text_pairs": len(unique),
            "direction_counts": dict(Counter(row["expected_source_text_state"]+"->"+row["expected_edited_text_state"] for row in rows)),
            "case_linked_counts": {name: sum(row[name] for row in rows) for name in metrics},
            "deduplicated_counts": {name: sum(row[name] for row in unique.values()) for name in metrics}}
    return summary, details


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm required before data/model access")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    bank_root = require_inside(args.bank_run, PROTECTED_ROOT, must_exist=True)
    resolver, source = load_expanded_bank(bank_root)
    bank_manifest = read_json(bank_root/"manifest.json")
    key_path = bank_root/"intervention_key.jsonl"
    if sha256_file(key_path) != bank_manifest["artifacts"][key_path.name]["sha256"]:
        raise ValueError("diagnostic key changed")
    key = [json.loads(line) for line in key_path.read_text().splitlines() if line]
    if {row["item_id"] for row in key} != {row["item_id"] for row in resolver}:
        raise ValueError("key/resolver inventory mismatch")
    texts_by_id = {row["item_id"]: report_text(row) for row in resolver}
    full_texts = {text_hash(text): text for text in texts_by_id.values()}
    sentence_texts, pairs = sentence_bindings(key, texts_by_id)
    if len(resolver) != 193 or len(full_texts) != 100 or len(pairs) != 97 or len({row["case_id"] for row in pairs}) > 48:
        raise ValueError("fixed development diagnostic inventory changed")
    cache_root = require_inside(args.cached_run, PROTECTED_ROOT, must_exist=True)
    cached_path = cache_root/"scores.json"
    if sha256_file(cached_path) != read_json(cache_root/"manifest.json")["artifacts"][cached_path.name]["sha256"]:
        raise ValueError("cached CheXbert hash mismatch")
    cached = read_json(cached_path)
    if (cached.get("mode") != "chexbert" or cached["source"] != source
            or cached.get("scorer_read_intervention_key") is not False or cached["producer"]["frozen"] is not True):
        raise ValueError("cached frozen CheXbert provenance differs")
    cached_records = {row["item_id"]: row for row in cached["records"]}
    if len(cached_records) != len(resolver) or set(cached_records) != set(texts_by_id):
        raise ValueError("cached item inventory differs")
    for row in resolver:
        if any(cached_records[row["item_id"]][name] != row[name] for name in ("report_sha256", "image_sha256")):
            raise ValueError("cached item hash differs")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        from tools.chexbert import CheXbert
        checkpoint = Path(args.checkpoint).resolve(strict=True)
        bert = Path(args.bert_path).resolve(strict=True)
        producer = cached["producer"]
        for path, name in ((checkpoint, "checkpoint_sha256"), (bert/"config.json", "bert_config_sha256"),
                           (bert/"vocab.txt", "tokenizer_vocab_sha256"), (Path(CheXbert.__init__.__code__.co_filename), "model_adapter_sha256")):
            if sha256_file(path) != producer[name]:
                raise ValueError("model/tokenizer/adapter changed since cached scoring")
        if producer["finding_order"] != list(CHEXBERT_ORDER) or producer["class_mapping"] != {str(k): v for k, v in CHEXBERT_CLASS_TO_STATE.items()}:
            raise ValueError("cached head order/class mapping differs")
        model = CheXbert(ckpt_dir=str(checkpoint.parent), bert_path=str(bert), checkpoint_path=checkpoint.name,
                        device=torch.device("cuda:0")).to("cuda:0")
        model.eval().requires_grad_(False)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("CheXbert not frozen")
        batches, examples, token_parity_mismatches, token_lengths = 0, 0, 0, []

        def forward(text_map, batch_size, check_wrapper=False):
            nonlocal batches, examples, token_parity_mismatches
            output = {}
            ordered = sorted(text_map)
            for start in range(0, len(ordered), batch_size):
                hashes = ordered[start:start+batch_size]
                values = [normalize_wrapper(text_map[h]) for h in hashes]
                tokenized = model.tokenizer(values, padding="longest", truncation=False, return_tensors="pt")
                lengths = tokenized["attention_mask"].sum(dim=1).tolist()
                if max(lengths) > model.bert.config.max_position_embeddings:
                    raise ValueError("token truncation would occur")
                token_lengths.extend(lengths)
                for h, ids, mask in zip(hashes, tokenized["input_ids"].tolist(), tokenized["attention_mask"].tolist(), strict=True):
                    official = model.tokenizer.encode_plus(model.tokenizer.tokenize(normalize_official_csv(text_map[h])))["input_ids"]
                    token_parity_mismatches += official != [value for value, present in zip(ids, mask, strict=True) if present]
                device_inputs = {name: value.to("cuda:0") for name, value in tokenized.items()}
                with torch.inference_mode():
                    cls = model.dropout(model.bert(**device_inputs)[0][:, 0, :])
                    logits = [head(cls).detach().cpu().tolist() for head in model.linear_heads]
                    batches += 1
                    examples += len(values)
                    for index, h in enumerate(hashes):
                        output[h] = decode_heads([head[index] for head in logits])
                    if check_wrapper:
                        predictions = model(list(values)).detach().cpu().tolist()
                        batches += 1
                        examples += len(values)
                        for h, prediction in zip(hashes, predictions, strict=True):
                            if prediction != [output[h][finding]["class_index"] for finding in CHEXBERT_ORDER]:
                                raise ValueError("diagnostic logit path differs from existing wrapper")
            return output

        full_batch8 = forward(full_texts, 8, check_wrapper=True)
        full_batch1 = forward(full_texts, 1)
        sentences = forward(sentence_texts, 1)
    torch.cuda.synchronize()
    batch_changed = [{"report_sha256": h, "changed_findings": [finding for finding in CHEXBERT_ORDER
        if full_batch8[h][finding]["state"] != full_batch1[h][finding]["state"]]} for h in sorted(full_texts)]
    cached_changes = [{"item_id": row["item_id"], "changed_findings": [finding for finding in CHEXBERT_ORDER
        if full_batch8[row["report_sha256"]][finding]["state"] != cached_records[row["item_id"]]["finding_states"][finding]]} for row in resolver]
    per_finding, details = diagnose_pairs(pairs, full_batch8, sentences)
    return {"schema_version": SCHEMA, "benchmark_split": "development", "source": source,
        "counts": {"items": 193, "unique_full_texts": 100, "eligible_pairs": 97,
            "unique_sentence_texts": len(sentence_texts), "forward_batches": batches, "encoder_examples": examples},
        "checks": {"wrapper_logit_path_argmax_parity": True,
            "official_csv_token_parity_mismatches": token_parity_mismatches,
            "token_count_min": min(token_lengths), "token_count_max": max(token_lengths),
            "batch1_vs_batch8_changed_unique_texts": sum(bool(row["changed_findings"]) for row in batch_changed),
            "cached_replay_changed_items": sum(bool(row["changed_findings"]) for row in cached_changes)},
        "per_finding": per_finding, "pair_details": details,
        "batch_changes": batch_changed, "cached_changes": cached_changes,
        "full_batch8_logits": full_batch8, "full_batch1_logits": full_batch1, "sentence_logits": sentences,
        "producer": producer, "execution": {"gpu_name": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__), "seed": 0},
        "source_sha256": {"cached_scores": sha256_file(cached_path), "intervention_key": sha256_file(key_path)},
        "peak_allocated_vram_including_load_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3),
        "diagnostic_reads_construction_key": True, "expected_states_passed_to_model": False,
        "primary_metric_eligible": False, "clinical_localization_accuracy": None,
        "targeted_repair_approved": False, "original_artifacts_or_winners_changed": False,
        "interpretation": "Context ablation/batch diagnostics only. Isolated sentences are not replacement report scores; raw logit margins are not calibrated clinical confidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-run", "cached-run", "checkpoint", "bert-path", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    temporary = None
    started = time.monotonic()
    os.umask(0o007)
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = run(args)
        payload["elapsed_seconds"] = round(time.monotonic()-started, 3)
        scores = write_private_json(temporary/"diagnostic.json", payload)
        summary = {key: value for key, value in payload.items() if key not in {
            "pair_details", "batch_changes", "cached_changes", "full_batch8_logits", "full_batch1_logits", "sentence_logits"}}
        summary_path = write_private_json(temporary/"summary.json", summary)
        lines = ["# CheXbert context/batch diagnostic / 上下文与批次诊断", "",
            "Development only; original EHR/CXR/reports/winners remain unchanged.",
            "Sentence-only predictions are diagnostic ablations, not new primary scores.", "",
            "| Finding | Case-linked pairs | Full both states | Isolated both states | Full edited mismatch / isolated match | Both edited contexts mismatch |",
            "|---|---:|---:|---:|---:|---:|"]
        for finding, row in payload["per_finding"].items():
            counts = row["case_linked_counts"]
            lines.append(f"| {finding} | {row['case_linked_pairs']} | {counts['full_both_text_states_extracted']} | {counts['sentence_both_text_states_extracted']} | {counts['full_edited_mismatch_sentence_match']} | {counts['edited_mismatch_both_contexts']} |")
        lines += ["", "Deduplicated counts, batch invariance and raw-logit diagnostics are in summary.json.",
            "Original and edited text assertions are not image truth; no clinical fault localization is established.",
            "No thresholds, selection, repair permissions or current metric implementations are changed.", ""]
        markdown = write_private_text(temporary/"summary.md", "\n".join(lines))
        write_private_json(temporary/"manifest.json", {"schema_version": SCHEMA, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "source": payload["source"], "source_sha256": payload["source_sha256"],
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in (scores, summary_path, markdown)}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_development_context_diagnostic", "elapsed_seconds": payload["elapsed_seconds"],
        "peak_allocated_vram_gib": payload["peak_allocated_vram_including_load_gib"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
