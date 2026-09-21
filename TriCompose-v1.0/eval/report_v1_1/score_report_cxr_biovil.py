#!/usr/bin/env python3
"""Secondary BioViL-T cosine for V1.1 report-CXR pairs.

This program performs frozen GPU inference and must run through approved
Slurm. Raw cosine is not converted into a clinical probability.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from contracts import (
    commit_atomic_run,
    discard_atomic_run,
    load_cxr_candidates,
    load_report_candidates,
    new_atomic_run,
    read_report_text,
    sha256_file,
    write_private_json,
)


SCHEMA_VERSION = "tricompose-report-cxr-biovil-scores-v1.1"
IMAGE_WEIGHT = "biovil_t_image_model_proj_size_128.pt"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxr-run", action="append", required=True)
    parser.add_argument("--report-run", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _load_runtime(model_path: Path, device: Any) -> tuple[Any, Any]:
    from health_multimodal.image.data.transforms import (
        create_chest_xray_transform_for_inference,
    )
    from health_multimodal.image.inference_engine import ImageInferenceEngine
    from health_multimodal.image.model.model import ImageModel
    from health_multimodal.image.model.types import ImageEncoderType
    from health_multimodal.text.inference_engine import TextInferenceEngine
    from health_multimodal.text.model import CXRBertModel, CXRBertTokenizer

    tokenizer = CXRBertTokenizer.from_pretrained(model_path, local_files_only=True)
    text_model = CXRBertModel.from_pretrained(model_path, local_files_only=True)
    text_model.eval().requires_grad_(False)
    text_engine = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)
    text_engine.model.to(device)
    image_model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
        joint_feature_size=128,
        pretrained_model_path=model_path / IMAGE_WEIGHT,
    )
    image_model.eval().requires_grad_(False)
    transform = create_chest_xray_transform_for_inference(
        resize=512, center_crop_size=448
    )
    image_engine = ImageInferenceEngine(image_model=image_model, transform=transform)
    image_engine.model.to(device)
    if image_engine.model.training or text_engine.model.training:
        raise RuntimeError("BioViL-T entered training mode")
    if any(parameter.requires_grad for parameter in image_engine.model.parameters()):
        raise RuntimeError("BioViL-T image encoder is not frozen")
    if any(parameter.requires_grad for parameter in text_engine.model.parameters()):
        raise RuntimeError("BioViL-T text encoder is not frozen")
    return image_engine, text_engine


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run BioViL-T through Slurm")
    model_path = Path(args.model_path).resolve(strict=True)
    required = [model_path / IMAGE_WEIGHT, model_path / "pytorch_model.bin"]
    if any(not path.is_file() for path in required):
        raise ValueError("BioViL-T checkpoint is incomplete")
    cxrs = load_cxr_candidates(args.cxr_run)
    reports = load_report_candidates(args.report_run, cxr_candidates=cxrs)
    reports_by_cxr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports.values():
        reports_by_cxr[str(report["parent_cxr_candidate_id"])].append(report)
    device = torch.device("cuda:0")
    image_engine, text_engine = _load_runtime(model_path, device)
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for cxr_id, cxr in sorted(cxrs.items()):
            parents = sorted(
                reports_by_cxr.get(cxr_id, []), key=lambda row: str(row["candidate_id"])
            )
            if not parents:
                raise ValueError("CXR candidate has no report candidates")
            texts = [read_report_text(report).strip() for report in parents]
            if any(not text for text in texts):
                raise ValueError("empty report reached BioViL-T")
            image_embedding = image_engine.get_projected_global_embedding(
                Path(cxr["artifact"]["path"])
            )
            text_embeddings = text_engine.get_embeddings_from_prompt(
                texts, normalize=True, verbose=False
            )
            scores = (text_embeddings @ image_embedding).detach().float().cpu().tolist()
            for report, score in zip(parents, scores, strict=True):
                records.append(
                    {
                        "report_candidate_id": report["candidate_id"],
                        "parent_cxr_candidate_id": cxr_id,
                        "report_sha256": report["artifact"]["sha256"],
                        "image_sha256": cxr["artifact"]["sha256"],
                        "biovil_raw_cosine": round(float(score), 8),
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_uncalibrated_secondary_metric",
        "producer": {
            "model_id": "biovil_t",
            "frozen": True,
            "image_checkpoint_sha256": sha256_file(model_path / IMAGE_WEIGHT),
            "text_checkpoint_sha256": sha256_file(model_path / "pytorch_model.bin"),
        },
        "interpretation": (
            "Raw image-report cosine only; calibration against matched and hard "
            "negative pairs is required before thresholding."
        ),
        "records": records,
        "counts": {"report_cxr_pairs": len(records), "image_encoder_calls": len(cxrs)},
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
        output = write_private_json(temporary / "biovil_scores.json", payload)
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
                "counts": payload["counts"],
                "output_sha256": output_hash,
                "peak_vram_gib": payload["peak_vram_gib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
