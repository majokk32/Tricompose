#!/usr/bin/env python3
"""Score V1 CXR-prompt and CXR-report alignment with frozen BioViL-T.

This is GPU inference and must run only through an approved Slurm job.  It
reads synthetic artifacts only and writes all results below the protected root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tricompose.privacy import require_private_file, sha256_file, write_private_json
from tricompose_v1.scoring import (
    BIOVIL_SCORE_SCHEMA,
    CALIBRATION_STATUS,
    commit_atomic_protected_run,
    discard_atomic_protected_run,
    load_candidate_bank,
    new_atomic_protected_run,
)


BIOVIL_T_IMAGE_WEIGHT = "biovil_t_image_model_proj_size_128.pt"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _load_runtime(model_path: Path, device: Any) -> tuple[Any, Any, Any]:
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
        pretrained_model_path=model_path / BIOVIL_T_IMAGE_WEIGHT,
    )
    image_model.eval().requires_grad_(False)
    transform = create_chest_xray_transform_for_inference(
        resize=512,
        center_crop_size=448,
    )
    image_engine = ImageInferenceEngine(image_model=image_model, transform=transform)
    image_engine.model.to(device)
    if image_engine.model.training or text_engine.model.training:
        raise RuntimeError("BioViL-T entered training mode")
    if any(parameter.requires_grad for parameter in image_engine.model.parameters()):
        raise RuntimeError("BioViL-T image encoder is not frozen")
    if any(parameter.requires_grad for parameter in text_engine.model.parameters()):
        raise RuntimeError("BioViL-T text encoder is not frozen")
    return image_engine, text_engine, tokenizer


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this program through Slurm")
    bank = load_candidate_bank(args.candidate_bank)
    model_path = Path(args.model_path).resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("BioViL-T model path is not a directory")
    required = [model_path / BIOVIL_T_IMAGE_WEIGHT, model_path / "pytorch_model.bin"]
    if any(not path.is_file() for path in required):
        raise ValueError("BioViL-T frozen checkpoint is incomplete")

    device = torch.device("cuda:0")
    image_engine, text_engine, _ = _load_runtime(model_path, device)
    torch.cuda.reset_peak_memory_stats()

    reports_by_cxr: dict[str, list[dict[str, Any]]] = {}
    for report in bank.report_candidates:
        reports_by_cxr.setdefault(str(report["parent_ids"][1]), []).append(report)

    cxr_records: list[dict[str, Any]] = []
    report_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for cxr in sorted(bank.cxr_candidates, key=lambda row: str(row["candidate_id"])):
            cxr_id = str(cxr["candidate_id"])
            image_ref = cxr["artifact"]
            image_path = require_private_file(image_ref["path"])
            if sha256_file(image_path) != image_ref["sha256"]:
                raise ValueError("CXR image hash changed after bank finalization")

            prompt_path = require_private_file(
                bank.staging_root
                / "cases"
                / str(cxr["case_id"])
                / "cxr_prompts"
                / f"{cxr['model_id']}.txt"
            )
            if sha256_file(prompt_path) != cxr["prompt_sha256"]:
                raise ValueError("CXR prompt hash does not match candidate lineage")
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not prompt:
                raise ValueError("CXR prompt is empty")

            reports = sorted(
                reports_by_cxr.get(cxr_id, []), key=lambda row: str(row["candidate_id"])
            )
            if len(reports) != 4:
                raise ValueError("each CXR must have four report candidates")
            report_texts: list[str] = []
            for report in reports:
                report_ref = report["artifact"]
                report_path = require_private_file(report_ref["path"])
                if sha256_file(report_path) != report_ref["sha256"]:
                    raise ValueError("report hash changed after bank finalization")
                text = report_path.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    raise ValueError("report candidate is empty")
                report_texts.append(text)

            # Compute the image embedding once, then score the prompt and all
            # four reports as one text batch.
            image_embedding = image_engine.get_projected_global_embedding(image_path)
            text_embeddings = text_engine.get_embeddings_from_prompt(
                [prompt, *report_texts], normalize=True, verbose=False
            )
            similarities = (text_embeddings @ image_embedding).detach().float().cpu().tolist()
            cxr_records.append(
                {
                    "cxr_candidate_id": cxr_id,
                    "image_sha256": image_ref["sha256"],
                    "prompt_sha256": cxr["prompt_sha256"],
                    "prompt_alignment_raw_cosine": round(float(similarities[0]), 8),
                }
            )
            for report, score in zip(reports, similarities[1:], strict=True):
                report_records.append(
                    {
                        "report_candidate_id": report["candidate_id"],
                        "parent_cxr_candidate_id": cxr_id,
                        "image_sha256": image_ref["sha256"],
                        "report_sha256": report["artifact"]["sha256"],
                        "image_report_alignment_raw_cosine": round(float(score), 8),
                    }
                )

    return {
        "schema_version": BIOVIL_SCORE_SCHEMA,
        "status": "completed_uncalibrated",
        "calibration_status": CALIBRATION_STATUS,
        "source_candidate_bank": {
            "path": str(bank.root),
            "manifest_sha256": bank.manifest_sha256,
        },
        "verifier": {
            "model_revision": model_path.name,
            "frozen": True,
            "image_checkpoint_sha256": sha256_file(required[0]),
            "text_checkpoint_sha256": sha256_file(required[1]),
            "score_contract": "raw BioViL-T cosine similarity; not calibrated",
        },
        "cxr_records": cxr_records,
        "report_records": report_records,
        "counts": {
            "cxr_candidates": len(cxr_records),
            "report_candidates": len(report_records),
            "image_encoder_calls": len(cxr_records),
            "text_items": len(cxr_records) + len(report_records),
        },
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def main() -> int:
    args = _parser().parse_args()
    os.umask(0o077)
    started = time.monotonic()
    temp, target = new_atomic_protected_run(args.output_root, args.run_id)
    try:
        payload = run(args)
        payload["run_id"] = args.run_id
        payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
        path = write_private_json(temp / "scores.json", payload)
        digest = sha256_file(path)
        commit_atomic_protected_run(temp, target)
    except Exception:
        discard_atomic_protected_run(temp)
        raise
    print(
        json.dumps(
            {
                "stage": "tricompose_v1_biovil_score",
                "status": "ok",
                "run_id": args.run_id,
                "cxr_candidate_count": payload["counts"]["cxr_candidates"],
                "report_candidate_count": payload["counts"]["report_candidates"],
                "artifact_sha256": digest,
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_vram_gib": payload["peak_vram_gib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
