#!/usr/bin/env python3
"""Frozen CheXbert authored-language diagnostic; approved Slurm inference only.

The scorer reads blinded text/hash/opaque IDs, never the authored reference
key. No new training, truncation, current score replacement or generation.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

from contracts import commit_atomic_run, discard_atomic_run, new_atomic_run, sha256_file, write_private_json
from extract_report_labels_chexbert import CHEXBERT_CLASS_TO_STATE, CHEXBERT_ORDER
from report_assertion_challenge import FINDINGS, SCORE_SCHEMA, load_inputs

CHECKPOINT_SHA = "6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1"
ADAPTER_SHA = "5d131d8dc8253211f127f48d8ac61fc689373e64dc92b9ec00a794e2331eafe6"


def decode_predictions(values):
    if len(values) != 14:
        raise ValueError("CheXbert head inventory differs")
    for index, value in enumerate(values):
        if type(value) is not int or value not in (range(2) if index == 13 else range(4)):
            raise ValueError("invalid categorical class index")
    all_states = dict(zip(CHEXBERT_ORDER, (CHEXBERT_CLASS_TO_STATE[value] for value in values), strict=True))
    return {finding: all_states[finding] for finding in FINDINGS}


def normalize_wrapper(text):
    # Exactly the audited cxrmate/tools/chexbert.py text preprocessing; the
    # existing wrapper still executes its own preprocessing, unchanged.
    return text.strip().replace("\n", " ").replace("\\s+", " ").replace("\\s+(?=[\\.,])", "").strip()


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm required before model/input access")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    resolver, texts, source = load_inputs(args.bank_run)
    checkpoint, bert = Path(args.checkpoint).resolve(strict=True), Path(args.bert_path).resolve(strict=True)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        from tools.chexbert import CheXbert
        adapter = Path(CheXbert.__init__.__code__.co_filename)
        fingerprints = {"checkpoint_sha256": sha256_file(checkpoint), "bert_config_sha256": sha256_file(bert/"config.json"),
            "tokenizer_vocab_sha256": sha256_file(bert/"vocab.txt"), "model_adapter_sha256": sha256_file(adapter)}
        if fingerprints["checkpoint_sha256"] != CHECKPOINT_SHA or fingerprints["model_adapter_sha256"] != ADAPTER_SHA:
            raise ValueError("frozen CheXbert checkpoint/adapter differs")
        assets = (checkpoint, bert/"config.json", bert/"vocab.txt", adapter)
        stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in assets}
        model = CheXbert(ckpt_dir=str(checkpoint.parent), bert_path=str(bert), checkpoint_path=checkpoint.name,
            device=torch.device("cuda:0")).to("cuda:0")
        model.eval().requires_grad_(False)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("model not frozen")
        batches, examples, lengths = 0, 0, []

        def forward(batch_size):
            nonlocal batches, examples
            states = {}
            ordered = sorted(texts)
            for start in range(0, len(ordered), batch_size):
                hashes = ordered[start:start+batch_size]
                values = [texts[h] for h in hashes]
                encoded = model.tokenizer([normalize_wrapper(text) for text in values], truncation=False)
                token_lengths = [len(ids) for ids in encoded["input_ids"]]
                if max(token_lengths) > model.bert.config.max_position_embeddings:
                    raise ValueError("input would be truncated")
                lengths.extend(token_lengths)
                with torch.inference_mode():
                    outputs = model(list(values)).detach().cpu().tolist()
                if len(outputs) != len(hashes):
                    raise ValueError("prediction batch size differs")
                batches += 1
                examples += len(hashes)
                for h, values in zip(hashes, outputs, strict=True):
                    states[h] = decode_predictions(values)
            return states

        primary, replay = forward(8), forward(1)
    torch.cuda.synchronize()
    if any((path.stat().st_size, path.stat().st_mtime_ns) != stats[str(path)] for path in assets):
        raise ValueError("frozen assets changed during diagnostic")
    records = [{"item_id": row["item_id"], "report_sha256": row["report_sha256"], "status": "complete",
        "finding_states": primary[row["report_sha256"]]} for row in resolver]
    changes = [{"report_sha256": h, "changed_findings": [name for name in FINDINGS if primary[h][name] != replay[h][name]]} for h in sorted(texts)]
    return {"schema_version": SCORE_SCHEMA, "source": source, "records": records,
        "producer": {"model_id": "chexbert", "frozen": True, **fingerprints,
            "class_mapping": {str(key): value for key, value in CHEXBERT_CLASS_TO_STATE.items()},
            "model_head_order": list(CHEXBERT_ORDER), "evaluated_findings": list(FINDINGS)},
        "counts": {"authored_texts": 56, "encoder_examples_including_replay": examples, "forward_batches": batches},
        "batch_replay": {"batch8_vs_batch1_changed_texts": sum(bool(row["changed_findings"]) for row in changes),
            "changes": changes, "independent_clinical_validation": False},
        "token_count_min": min(lengths), "token_count_max": max(lengths), "truncation_used": False,
        "scorer_read_reference_key": False, "model_received_reference_states": False,
        "primary_metric_eligible": False, "selection_changed": False, "targeted_repair_approved": False,
        "independent_clinical_accuracy": None, "expert_reviewed": False,
        "execution": {"gpu_name": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__), "seed": 0},
        "peak_allocated_vram_including_load_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank-run", "checkpoint", "bert-path", "output-root", "run-id"):
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
        write_private_json(temporary/"manifest.json", {"schema_version": SCORE_SCHEMA, "run_id": args.run_id,
            "program_sha256": sha256_file(__file__), "source": payload["source"],
            "artifacts": {path.name: {"sha256": sha256_file(path)}}, "primary_metric_eligible": False})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_authored_language_predictions", "elapsed_seconds": payload["elapsed_seconds"],
        "peak_allocated_vram_gib": payload["peak_allocated_vram_including_load_gib"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
