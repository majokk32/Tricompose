#!/usr/bin/env python3
"""Frozen report-polarity diagnostic scoring (Slurm), then small cached analysis.

Inference reads a blind resolver, not the intervention key. No clinical gold,
automatic localization, new thresholds, ranking, repair or model training.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

from build_report_polarity_benchmark import SCHEMA as BANK_SCHEMA, load_blind_bank
from contracts import (
    CHEXPERT_FINDINGS, PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
    new_atomic_run, read_json, require_inside, sha256_file,
    write_private_json, write_private_text,
)
from extract_report_labels_chexbert import CHEXBERT_CLASS_TO_STATE, CHEXBERT_ORDER
from score_report_cxr_biovil import IMAGE_WEIGHT, _load_runtime

SCHEMA = "tricompose-synthetic-report-polarity-scores-v1"


def report_text(row):
    path = require_inside(row["report_path"], PROTECTED_ROOT, must_exist=True)
    if sha256_file(path) != row["report_sha256"]:
        raise ValueError("report changed before frozen inference")
    return path.read_bytes().decode("utf-8")


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm required before any model/data access")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    resolver, bank = load_blind_bank(args.bank_run)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    records = []
    base = lambda row: {key: row[key] for key in ("item_id", "image_sha256", "report_sha256")}
    # Never pass item IDs, image paths or intervention labels to the report model.
    texts = {row["report_sha256"]: report_text(row) for row in resolver}
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        if args.mode == "chexbert":
            from tools.chexbert import CheXbert
            checkpoint = Path(args.checkpoint).resolve(strict=True)
            bert = Path(args.bert_path).resolve(strict=True)
            model = CheXbert(ckpt_dir=str(checkpoint.parent), bert_path=str(bert),
                checkpoint_path=checkpoint.name, device=torch.device("cuda:0")).to("cuda:0")
            model.eval().requires_grad_(False)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError("CheXbert not frozen")
            ordered = sorted(texts)
            tokenized = model.tokenizer([texts[key] for key in ordered], truncation=False)
            if any(len(tokens) > model.bert.config.max_position_embeddings for tokens in tokenized["input_ids"]):
                raise ValueError("diagnostic report would be truncated by CheXbert")
            with torch.inference_mode():
                output = model([texts[key] for key in ordered]).detach().cpu().tolist()
            if len(output) != len(ordered):
                raise ValueError("CheXbert output inventory mismatch")
            states = {}
            for fingerprint, values in zip(ordered, output, strict=True):
                if len(values) != len(CHEXBERT_ORDER):
                    raise ValueError("CheXbert width mismatch")
                states[fingerprint] = dict(zip(CHEXBERT_ORDER, (CHEXBERT_CLASS_TO_STATE[value] for value in values), strict=True))
            records = [{**base(row), "finding_states": {name: states[row["report_sha256"]][name]
                       for name in CHEXPERT_FINDINGS}} for row in resolver]
            producer = {"model_id": "chexbert", "frozen": True,
                "checkpoint_sha256": sha256_file(checkpoint),
                "bert_config_sha256": sha256_file(bert / "config.json"),
                "tokenizer_vocab_sha256": sha256_file(bert / "vocab.txt"),
                "model_adapter_sha256": sha256_file(Path(CheXbert.__init__.__code__.co_filename)),
                "finding_order": list(CHEXBERT_ORDER), "class_mapping": CHEXBERT_CLASS_TO_STATE}
            calls = {"image_encoder_calls": 0, "report_encoder_calls": len(texts)}
        elif args.mode == "biovil":
            model_path = Path(args.model_path).resolve(strict=True)
            image_engine, text_engine = _load_runtime(model_path, torch.device("cuda:0"))
            images, embeddings = {}, {}
            with torch.inference_mode():
                for row in resolver:
                    ihash = row["image_sha256"]
                    if ihash not in images:
                        if sha256_file(row["image_path"]) != ihash:
                            raise ValueError("image changed before frozen inference")
                        images[ihash] = image_engine.get_projected_global_embedding(Path(row["image_path"]))
                for fingerprint, text in sorted(texts.items()):
                    embeddings[fingerprint] = text_engine.get_embeddings_from_prompt([text], normalize=True, verbose=False)
                for row in resolver:
                    score = float((embeddings[row["report_sha256"]] @ images[row["image_sha256"]]).detach().cpu().item())
                    if not math.isfinite(score) or not -1.01 <= score <= 1.01:
                        raise ValueError("invalid BioViL cosine")
                    records.append({**base(row), "biovil_raw_cosine": round(score, 8)})
            producer = {"model_id": "biovil_t", "frozen": True,
                "image_checkpoint_sha256": sha256_file(model_path / IMAGE_WEIGHT),
                "text_checkpoint_sha256": sha256_file(model_path / "pytorch_model.bin"),
                "loader_sha256": sha256_file(Path(_load_runtime.__code__.co_filename))}
            calls = {"image_encoder_calls": len(images), "report_encoder_calls": len(texts)}
        else:
            raise ValueError("unsupported inference mode")
    torch.cuda.synchronize()
    return {"schema_version": SCHEMA, "mode": args.mode, "source": bank,
        "producer": producer, "records": records, "counts": {"items": len(records), **calls},
        "execution": {"torch_version": str(torch.__version__), "gpu_name": torch.cuda.get_device_name(0),
                      "model_parameters_frozen": True, "seed": 0},
        "primary_metric_eligible": False, "scorer_read_intervention_key": False,
        "clinical_localization_accuracy": None, "targeted_repair_approved": False,
        "peak_allocated_vram_including_load_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}


def summarize(key, chexbert, biovil):
    if len(key) > 6 or not key:
        raise ValueError("bounded diagnostic required")
    by_key = {row["item_id"]: row for row in key}
    indices = [{row["item_id"]: row for row in scores["records"]} for scores in (chexbert, biovil)]
    if len(by_key) != len(key) or any(set(index) != set(by_key) or len(index) != len(scores["records"])
        for index, scores in zip(indices, (chexbert, biovil), strict=True)):
        raise ValueError("analysis inventory mismatch")
    cindex, bindex = indices
    for item in by_key:
        if any(cindex[item][field] != bindex[item][field] for field in ("image_sha256", "report_sha256")):
            raise ValueError("scorer input hashes differ")
    cases = []
    for case in sorted({row["case_id"] for row in key}):
        items = [row for row in key if row["case_id"] == case]
        grouped = {row["intervention_type"]: row for row in items}
        if len(grouped) != len(items) or not {"unchanged", "whitespace_only"} <= set(grouped):
            raise ValueError("missing or duplicate control")
        control, spaces = grouped["unchanged"], grouped["whitespace_only"]
        cid, sid = control["item_id"], spaces["item_id"]
        finding = control["finding"]
        if any(row["finding"] != finding for row in items) or any(bindex[row["item_id"]]["image_sha256"] != bindex[cid]["image_sha256"] for row in items):
            raise ValueError("fixed finding/image changed")
        old, same = cindex[cid]["finding_states"], cindex[sid]["finding_states"]
        record = {"case_id": case, "finding": finding,
            "whitespace_extraction_identical": old == same,
            "whitespace_changed_heads": [name for name in CHEXPERT_FINDINGS if old[name] != same[name]],
            "whitespace_biovil_cosine_delta": round(bindex[sid]["biovil_raw_cosine"] - bindex[cid]["biovil_raw_cosine"], 8),
            "polarity_edit_available": "minimal_polarity_flip" in grouped}
        if record["polarity_edit_available"]:
            flip = grouped["minimal_polarity_flip"]
            fid = flip["item_id"]
            if flip["edit"]["source_text_assertion"] != "positive" or flip["edit"]["edited_text_assertion"] != "negative":
                raise ValueError("unsupported mechanical polarity expectation")
            new = cindex[fid]["finding_states"]
            record.update({"original_chexbert_target_state": old[finding],
                "edited_chexbert_target_state": new[finding],
                "explicit_positive_to_negative_detected": old[finding] == "positive" and new[finding] == "negative",
                "non_target_changed_heads": [name for name in CHEXPERT_FINDINGS if name != finding and old[name] != new[name]],
                "polarity_biovil_cosine_delta": round(bindex[fid]["biovil_raw_cosine"] - bindex[cid]["biovil_raw_cosine"], 8)})
        cases.append(record)
    return {"cases": cases, "fixed_cases": len(cases),
        "polarity_pairs": sum(row["polarity_edit_available"] for row in cases),
        "primary_metric_eligible": False, "clinical_localization_accuracy": None,
        "clinical_mismatch_verified": False, "targeted_repair_approved": False,
        "interpretation": "Mechanical text polarity and whitespace sensitivity only; cosine direction is not clinical correctness."}


def analyze(args):
    root = require_inside(args.bank_run, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root / "manifest.json")
    if manifest["schema_version"] != BANK_SCHEMA:
        raise ValueError("unsupported bank")
    key_path = root / "intervention_key.jsonl"
    if sha256_file(key_path) != manifest["artifacts"][key_path.name]["sha256"]:
        raise ValueError("intervention key hash mismatch")
    key = [json.loads(line) for line in key_path.read_text().splitlines() if line]
    scores, sources = [], {"bank_manifest": root / "manifest.json", "intervention_key": key_path}
    for mode, run_path in (("chexbert", args.chexbert_run), ("biovil", args.biovil_run)):
        run_root = require_inside(run_path, PROTECTED_ROOT, must_exist=True)
        path = run_root / "scores.json"
        if sha256_file(path) != read_json(run_root / "manifest.json")["artifacts"][path.name]["sha256"]:
            raise ValueError("scorer output hash mismatch")
        payload = read_json(path)
        if (payload.get("schema_version") != SCHEMA or payload.get("mode") != mode
                or payload.get("primary_metric_eligible") is not False
                or payload.get("scorer_read_intervention_key") is not False
                or payload["source"]["manifest_sha256"] != sha256_file(root / "manifest.json")
                or payload["source"]["resolver_sha256"] != sha256_file(root / "resolver.jsonl")):
            raise ValueError("scorer source/scope mismatch")
        scores.append(payload)
        sources[mode] = path
    summary = summarize(key, *scores)
    return summary, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("chexbert", "biovil", "analyze"), required=True)
    for name in ("bank-run", "output-root", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("checkpoint", "bert-path", "model-path", "chexbert-run", "biovil-run"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    if args.mode != "analyze" and not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    temporary = None
    os.umask(0o007)
    started = time.monotonic()
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload, sources = analyze(args) if args.mode == "analyze" else (run(args), {})
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        files = [write_private_json(temporary / ("summary.json" if args.mode == "analyze" else "scores.json"), payload)]
        if args.mode == "analyze":
            files.append(write_private_text(temporary / "summary.md", "# Report polarity diagnostic / 报告最小否定测试\n\n"
                + "```json\n" + json.dumps(payload, indent=2) + "\n```\n\n"
                + "No clinical gold, localization accuracy, new winner, calibration or repair.\n"))
        write_private_json(temporary / "manifest.json", {"schema_version": SCHEMA,
            "mode": args.mode, "run_id": args.run_id, "program_sha256": sha256_file(__file__),
            "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
            "artifacts": {path.name: {"sha256": sha256_file(path)} for path in files}})
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed_polarity_diagnostic", "elapsed_seconds": payload["elapsed_seconds"],
                      "manifest_sha256": sha256_file(target / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
