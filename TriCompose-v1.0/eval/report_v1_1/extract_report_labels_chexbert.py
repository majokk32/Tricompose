#!/usr/bin/env python3
"""Extract 14 finding states from V1.1 reports with frozen CheXbert.

This is GPU inference and must run only through an approved Slurm allocation.
No report text is written to the output bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from contracts import (
    CHEXPERT_FINDINGS,
    commit_atomic_run,
    discard_atomic_run,
    load_cxr_candidates,
    load_report_candidates,
    new_atomic_run,
    read_report_text,
    sha256_file,
    write_private_json,
)


SCHEMA_VERSION = "tricompose-report-finding-labels-v1.1"
CHEXBERT_ORDER = (
    "enlarged_cardiomediastinum",
    "cardiomegaly",
    "lung_opacity",
    "lung_lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural_effusion",
    "pleural_other",
    "fracture",
    "support_devices",
    "no_finding",
)
# This class mapping follows cxrmate/tools/metrics/chexbert.py.
CHEXBERT_CLASS_TO_STATE = {
    0: "unknown",
    1: "positive",
    2: "negative",
    3: "uncertain",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--report-run", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from tools.chexbert import CheXbert

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run CheXbert through Slurm")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    bert_path = Path(args.bert_path).resolve(strict=True)
    if not checkpoint.is_file() or not bert_path.is_dir():
        raise ValueError("CheXbert checkpoint or BERT directory is missing")
    cxrs = load_cxr_candidates(args.cxr_run)
    reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
    ordered = [report for _, report in sorted(reports.items())]
    device = torch.device("cuda:0")
    model = CheXbert(
        ckpt_dir=str(checkpoint.parent),
        bert_path=str(bert_path),
        checkpoint_path=checkpoint.name,
        device=device,
    ).to(device)
    model.eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CheXbert is not frozen")

    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(ordered), args.batch_size):
            batch = ordered[start : start + args.batch_size]
            texts = [read_report_text(report).strip() for report in batch]
            if any(not text for text in texts):
                raise ValueError("empty report reached CheXbert")
            outputs = model(list(texts)).detach().cpu().tolist()
            if len(outputs) != len(batch):
                raise ValueError("CheXbert batch output length mismatch")
            for report, values in zip(batch, outputs, strict=True):
                if len(values) != len(CHEXBERT_ORDER):
                    raise ValueError("CheXbert finding width mismatch")
                ordered_states = {
                    finding: CHEXBERT_CLASS_TO_STATE[int(value)]
                    for finding, value in zip(CHEXBERT_ORDER, values, strict=True)
                }
                records.append(
                    {
                        "report_candidate_id": report["candidate_id"],
                        "report_sha256": report["artifact"]["sha256"],
                        "finding_states": {
                            finding: ordered_states[finding]
                            for finding in CHEXPERT_FINDINGS
                        },
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": {
            "model_id": "chexbert",
            "frozen": True,
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "class_mapping_source": "cxrmate/tools/metrics/chexbert.py",
        },
        "primary_metric_eligible": True,
        "calibration": {
            "status": "categorical_checkpoint_outputs",
            "unknown_is_negative": False,
        },
        "finding_order": list(CHEXPERT_FINDINGS),
        "records": records,
        "counts": {"reports": len(records), "model_calls": len(records)},
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o007)
    started = time.monotonic()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        payload = run(args)
        payload["run_id"] = args.run_id
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output = write_private_json(temporary / "report_finding_labels.json", payload)
        output_hash = sha256_file(output)
        commit_atomic_run(temporary, target)
    except Exception:
        discard_atomic_run(temporary)
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": args.run_id,
                "report_count": payload["counts"]["reports"],
                "output_sha256": output_hash,
                "peak_vram_gib": payload["peak_vram_gib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
