#!/usr/bin/env python3
"""Score blinded synthetic intervention pairs with frozen BioViL-T.

Slurm only. The intervention key is neither read nor passed to this process.
Raw synthetic report text and images remain in protected memory and files.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

from build_intervention_smoke import file_sha256
from contracts import (PROTECTED_ROOT, commit_atomic_run, discard_atomic_run,
                       load_cxr_candidates, load_report_candidates,
                       new_atomic_run, read_report_text, require_inside,
                       sha256_file, write_private_json)
from score_report_cxr_biovil import IMAGE_WEIGHT, _load_runtime

SCHEMA = "tricompose-v12-intervention-biovil-blind-scores-v1"
BANK_SCHEMA = "tricompose-v12-intervention-smoke-v1"
RESOLVER_FIELDS = frozenset({
    "item_id", "case_id", "benchmark_split", "ehr_sha256",
    "displayed_cxr_candidate_id", "displayed_cxr_sha256",
    "displayed_report_candidate_id", "displayed_report_sha256",
})


def load_blind_bank(run_path: str | Path) -> tuple[list[dict], dict]:
    run = require_inside(run_path, PROTECTED_ROOT, must_exist=True)
    manifest_path = require_inside(run / "manifest.json", run, must_exist=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BANK_SCHEMA:
        raise ValueError("unsupported intervention bank schema")
    source = manifest.get("source", {})
    if (source.get("construction_uses_frozen_labels") is not True or
        source.get("coverage", {}).get("split_strategy") !=
            "positive_pleural_effusion_coverage_v1"):
        raise ValueError("expected the split-locked stratified benchmark")
    files = {}
    for name in ("blind_items", "resolver"):
        path = require_inside(run / f"{name}.jsonl", run, must_exist=True)
        if file_sha256(path) != manifest.get("artifact_sha256", {}).get(name):
            raise ValueError("blinded benchmark artifact hash mismatch")
        files[name] = [json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines() if line]
    blind, resolver = files["blind_items"], files["resolver"]
    if not blind or len(blind) != len(resolver):
        raise ValueError("incomplete blinded benchmark")
    blind_ids = [row.get("item_id") for row in blind]
    resolver_ids = [row.get("item_id") for row in resolver]
    if (any(set(row) != {"item_id"} for row in blind) or
        any(set(row) != RESOLVER_FIELDS for row in resolver) or
        len(set(blind_ids)) != len(blind_ids) or
        len(set(resolver_ids)) != len(resolver_ids) or
        set(blind_ids) != set(resolver_ids)):
        raise ValueError("blind/resolver IDs or fields invalid")
    return resolver, {"bank_manifest_sha256": file_sha256(manifest_path),
                      "blind_items_sha256": file_sha256(run / "blind_items.jsonl"),
                      "resolver_sha256": file_sha256(run / "resolver.jsonl"),
                      "items": len(resolver)}


def validate_artifacts(resolver: list[dict], cxrs: dict,
                       reports: dict) -> tuple[list[str], list[str]]:
    images, texts = set(), set()
    for row in resolver:
        image_id = row["displayed_cxr_candidate_id"]
        report_id = row["displayed_report_candidate_id"]
        image = cxrs.get(image_id)
        report = reports.get(report_id)
        if image is None or report is None:
            raise ValueError("intervention artifact absent from supplied frozen runs")
        if (image["artifact"]["sha256"] != row["displayed_cxr_sha256"] or
            report["artifact"]["sha256"] != row["displayed_report_sha256"]):
            raise ValueError("intervention artifact hash mismatch")
        images.add(image_id)
        texts.add(report_id)
    return sorted(images), sorted(texts)


def score_pairs(resolver: list[dict], cxrs: dict, reports: dict,
                image_engine, text_engine, torch) -> tuple[list[dict], dict]:
    image_ids, report_ids = validate_artifacts(resolver, cxrs, reports)
    image_embeddings = {}
    text_embeddings = {}
    with torch.inference_mode(), open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            for image_id in image_ids:
                image_embeddings[image_id] = image_engine.get_projected_global_embedding(
                    Path(cxrs[image_id]["artifact"]["path"]))
            for report_id in report_ids:
                text = read_report_text(reports[report_id]).strip()
                if not text:
                    raise ValueError("empty synthetic report")
                text_embeddings[report_id] = text_engine.get_embeddings_from_prompt(
                    [text], normalize=True, verbose=False)
    records = []
    for row in sorted(resolver, key=lambda value: value["item_id"]):
        image_id = row["displayed_cxr_candidate_id"]
        report_id = row["displayed_report_candidate_id"]
        score = float((text_embeddings[report_id] @
                       image_embeddings[image_id]).detach().float().cpu().item())
        if not math.isfinite(score) or not -1.01 <= score <= 1.01:
            raise ValueError("invalid BioViL-T cosine")
        records.append({"item_id": row["item_id"], "case_id": row["case_id"],
                        "benchmark_split": row["benchmark_split"],
                        "displayed_cxr_sha256": row["displayed_cxr_sha256"],
                        "displayed_report_sha256": row["displayed_report_sha256"],
                        "biovil_raw_cosine": round(score, 8)})
    return records, {"pairs": len(records), "image_encoder_calls": len(image_ids),
                     "text_encoder_calls": len(report_ids)}


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("BioViL-T inference requires approved Slurm")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    model = Path(args.model_path).resolve(strict=True)
    if not (model / IMAGE_WEIGHT).is_file() or not (model / "pytorch_model.bin").is_file():
        raise ValueError("BioViL-T checkpoint incomplete")
    resolver, bank = load_blind_bank(args.bank_run)
    cxrs = load_cxr_candidates(args.cxr_run)
    reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
    torch.cuda.reset_peak_memory_stats()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            image_engine, text_engine = _load_runtime(model, torch.device("cuda:0"))
    records, counts = score_pairs(resolver, cxrs, reports,
                                  image_engine, text_engine, torch)
    if counts["pairs"] != bank["items"]:
        raise AssertionError("not all blinded items were scored")
    return {
        "schema_version": SCHEMA,
        "status": "completed_blind_secondary_score_uncalibrated",
        "source": bank,
        "producer": {"model_id": "biovil_t", "frozen": True,
                     "image_checkpoint_sha256": sha256_file(model / IMAGE_WEIGHT),
                     "text_checkpoint_sha256": sha256_file(model / "pytorch_model.bin"),
                     "program_sha256": sha256_file(Path(__file__))},
        "counts": counts,
        "records": records,
        "scorer_read_intervention_key": False,
        "clinical_localization_accuracy": None,
        "probability_calibration_performed": False,
        "interpretation": "Blinded raw image-report cosine. It detects pair mismatch, not the faulty modality or clinical truth.",
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-run", required=True)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--report-run", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        print(json.dumps({"status": "refused", "reason": "approved_slurm_required"}))
        return 2
    os.umask(0o007)
    started = time.monotonic()
    temporary = None
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        payload = run(args)
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output = write_private_json(temporary / "scores.json", payload)
        output_hash = sha256_file(output)
        commit_atomic_run(temporary, target)
    except Exception as exc:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({"status": "completed", "run_id": args.run_id,
                      "counts": payload["counts"], "output_sha256": output_hash,
                      "peak_vram_gib": payload["peak_vram_gib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
