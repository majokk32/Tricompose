"""Two-case MeDiM text-to-image diagnostic using the official paired recipe.

This adapter intentionally keeps the external MeDiM checkout read-only.  It
composes the repository's ``large_scale_train`` and
``paired_standalone_fid_eval`` Hydra configs, uses its ``ValDataset`` and
``Diffusion.sample_for_fid`` implementation, and supplies only the two small
compatibility hooks missing from the current upstream checkout.

All prompts and generated images are protected artifacts.  The public process
output contains only sanitized status and aggregate runtime information.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medim_ehr_prompt.common import (
    create_run_directory,
    enforce_private_directory_mode,
    enforce_private_file_mode,
    read_private_json,
    require_run_directory,
    sha256_file,
    validate_case_id,
    write_private_json,
    write_private_text,
)
SCHEMA_VERSION = "medim_ehr_prompt.official_paired_t2i.v1"
EXPECTED_SOURCE_SCHEMA = "medim_ehr_prompt.case_input.v1"
OFFICIAL_REPO = Path(
    "/project2/ruishanl_1185/MeDiM/MeDiM_original_infer"
)
CHECKPOINT = Path(
    "/project2/ruishanl_1185/MeDiM/pretrained_models/medim_ckpts/model.safetensors"
)
LLAMA_CHECKPOINT = Path(
    "/project2/ruishanl_1185/MeDiM/pretrained_models/Llama-2-7b-hf"
)
LIQUID_CHECKPOINT = Path(
    "/project2/ruishanl_1185/MeDiM/pretrained_models/Liquid_V1_7B"
)
VQGAN_CONFIG = Path(
    "/project2/ruishanl_1185/MeDiM/pretrained_models/chameleon/vqgan.yaml"
)
VQGAN_CHECKPOINT = Path(
    "/project2/ruishanl_1185/MeDiM/pretrained_models/chameleon/vqgan.ckpt"
)

OFFICIAL_CHEST_PREFIX = (
    "The image is a radiograph of the chest, showing the thoracic cavity "
    "structures."
)
SYNTHETIC_REPORT = (
    "Portable frontal AP chest radiograph. "
    "No focal airspace opacity, pleural effusion, or pneumothorax. "
    "The cardiomediastinal silhouette is within normal limits."
)
EXPECTED_OFFICIAL_COMMIT = "ce1ea9936be1043cc28b034b19c5094d953d2023"
ALLOWED_MISSING_CHECKPOINT_KEYS = frozenset({"lm_head.weight"})


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    enforce_private_directory_mode(path)
    return path


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strip_single_official_prefix(prompt: str) -> str:
    """Remove the prefix that official ``ValDataset`` adds for MIMIC text."""

    normalized = prompt.strip()
    if not normalized.startswith(OFFICIAL_CHEST_PREFIX):
        raise ValueError("protected EHR prompt lacks the expected official prefix")
    remainder = normalized[len(OFFICIAL_CHEST_PREFIX) :].lstrip()
    if not remainder:
        raise ValueError("protected EHR prompt is empty after prefix removal")
    if remainder.startswith(OFFICIAL_CHEST_PREFIX):
        raise ValueError("protected EHR prompt contains more than one prefix")
    return remainder


def build_official_style_ehr_prompt(facts: dict[str, Any]) -> str:
    """Serialize only report-like, radiographically relevant EHR context."""

    age_group = str(facts.get("age_group", "adult"))
    sex = str(facts.get("sex", "unspecified-sex"))
    diagnoses = [str(value) for value in facts.get("positive_diagnoses", [])]
    devices = [
        str(value) for value in facts.get("positive_support_devices", [])
    ]
    sentences = [
        "Portable frontal AP chest radiograph.",
        f"Clinical history: {age_group} {sex} patient.",
    ]
    if diagnoses:
        sentences.append(
            "Clinical indication: evaluation for "
            + ", ".join(diagnoses)
            + "."
        )
    if devices:
        sentences.append("Support devices: " + ", ".join(devices) + ".")
    if not diagnoses and not devices:
        sentences.append(
            "No additional radiographically relevant clinical context is "
            "provided."
        )
    return " ".join(sentences)


def _write_neutral_image(path: Path) -> None:
    image = Image.new("RGB", (512, 512), color=(122, 116, 104))
    image.save(path, format="JPEG", quality=95)
    enforce_private_file_mode(path)


def stage_two_conditions(
    *, source_run_id: str, source_case_id: str, output_run_id: str
) -> Path:
    """Stage one synthetic-report condition and one protected-EHR condition."""

    os.umask(0o077)
    validate_case_id(source_case_id)
    source_run = require_run_directory(source_run_id)
    source = read_private_json(
        source_run / "cases" / source_case_id / "input.json"
    )
    if source.get("schema_version") != EXPECTED_SOURCE_SCHEMA:
        raise ValueError("unsupported protected source case")
    ehr_facts = source.get("ehr_facts")
    if not isinstance(ehr_facts, dict):
        raise ValueError("protected source case lacks serialized EHR facts")
    # Rebuild from protected facts using a concise MIMIC-report-like template.
    # Labs and vitals remain in the source artifact but are intentionally not
    # passed to the report-conditioned generator.
    ehr_prompt = build_official_style_ehr_prompt(ehr_facts)

    output_run = create_run_directory(output_run_id)
    dataset_root = _private_dir(output_run / "dataset")
    mimic_root = _private_dir(dataset_root / "mimic")
    _private_dir(dataset_root / "path")
    cases_root = _private_dir(output_run / "cases")
    _private_dir(output_run / "logs")
    _private_dir(output_run / "work")

    conditions = [
        ("case_000", "synthetic_radiology_report", SYNTHETIC_REPORT),
        ("case_001", "ehr_deterministic_prompt", ehr_prompt),
    ]
    records: list[dict[str, Any]] = []
    for ordinal, (case_id, condition, prompt) in enumerate(conditions):
        case_root = _private_dir(cases_root / case_id)
        stem = f"sample_{ordinal:03d}"
        text_path = mimic_root / f"{stem}.txt"
        image_path = mimic_root / f"{stem}.jpg"
        write_private_text(text_path, prompt)
        _write_neutral_image(image_path)
        prompt_hash = _hash_text(prompt)
        write_private_json(
            case_root / "input.json",
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "condition": condition,
                "dataset_stem": stem,
                "prompt_sha256": prompt_hash,
                "seed": 30403,
            },
        )
        records.append(
            {
                "case_id": case_id,
                "condition": condition,
                "dataset_stem": stem,
                "prompt_sha256": prompt_hash,
                "seed": 30403,
            }
        )

    write_private_json(
        output_run / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": output_run_id,
            "source_run_id": source_run_id,
            "source_case_id": source_case_id,
            "official_repo_commit": EXPECTED_OFFICIAL_COMMIT,
            "official_experiments": [
                "large_scale_train",
                "paired_standalone_fid_eval",
            ],
            "official_method": "Diffusion.sample_for_fid",
            "neutral_image_is_condition": False,
            "cases": records,
        },
    )
    return output_run


def _prepend_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _verify_official_checkout() -> None:
    git_prefix = [
        "git",
        "-c",
        f"safe.directory={OFFICIAL_REPO}",
        "-C",
        str(OFFICIAL_REPO),
    ]
    revision = subprocess.run(
        [*git_prefix, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != EXPECTED_OFFICIAL_COMMIT:
        raise RuntimeError("official MeDiM checkout is at an unexpected commit")
    required_paths = [
        "model.py",
        "model_eval.py",
        "models/datasets/all_dataset.py",
        "configs/config.yaml",
        "configs/experiments/large_scale_train.yaml",
        "configs/experiments/paired_standalone_fid_eval.yaml",
    ]
    clean = subprocess.run(
        [*git_prefix, "diff", "--quiet", "HEAD", "--", *required_paths],
        check=False,
    )
    if clean.returncode != 0:
        raise RuntimeError("required official MeDiM source files are modified")


def _compose_official_config(output_run: Path) -> Any:
    _prepend_import_path(OFFICIAL_REPO / "third_party" / "hydra_submitit_launcher")
    _prepend_import_path(OFFICIAL_REPO)

    import utils
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    utils.set_omega_conf_resolvers()
    overrides = [
        "+experiments=[large_scale_train,paired_standalone_fid_eval]",
        "devices=1",
        "loader.batch_size=1",
        "loader.eval_batch_size=1",
        "trainer.precision=fp16",
        "trainer.compile=false",
        "trainer.fsdp=false",
        "trainer.ema=0.0",
        "trainer.use_custom_ema=false",
        "wandb.mode=disabled",
        f"model.llama_ckpt={LLAMA_CHECKPOINT}",
        f"model.liquid_ckpt={LIQUID_CHECKPOINT}",
        f"model.vqgan_config={VQGAN_CONFIG}",
        f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
        f"data.data_path_dir_train={output_run / 'dataset' / 'path'}",
        f"data.data_path_dir_val={output_run / 'dataset' / 'path'}",
        f"data.data_mimic_dir_train={output_run / 'dataset' / 'mimic'}",
        f"data.data_mimic_dir_val={output_run / 'dataset' / 'mimic'}",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str(OFFICIAL_REPO / "configs")
    ):
        config = compose(config_name="config", overrides=overrides)
    with open_dict(config):
        config.output_dir = str(output_run / "work")
        config.logging_dir = str(output_run / "logs")
        config.trainer.dtype = "torch.float16"
    OmegaConf.resolve(config)

    expected = {
        "mode": "eval",
        "txt_conditional_fid": True,
        "cfg": 5,
        "length": 1280,
        "txt_length": 256,
        "img_length": 1024,
        "sampling_steps": 1280,
        "max_sampling_steps": 1280,
    }
    actual = {
        "mode": str(config.mode),
        "txt_conditional_fid": bool(config.eval.txt_conditional_fid),
        "cfg": int(config.eval.cfg),
        "length": int(config.model.length),
        "txt_length": int(config.model.txt_length),
        "img_length": int(config.model.img_length),
        "sampling_steps": int(config.sampling.steps),
        "max_sampling_steps": int(config.sampling.max_sampling_steps),
    }
    if actual != expected:
        raise RuntimeError("resolved official paired configuration is unexpected")
    return config


def _get_cond_dict(_self: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Compatibility hook absent from upstream ``Diffusion``."""

    return {"batch": batch}


def _get_vae(self: Any) -> Any:
    """Expose the already loaded official Chameleon tokenizer to the decoder."""

    return self.image_tokenizer


def _save_official_image(image: Any, path: Path) -> list[int]:
    if path.exists():
        raise FileExistsError("generated image already exists")
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path, format="PNG")
        dimensions = [int(image.width), int(image.height)]
    else:
        import torch
        from model_utils import remap_image_torch

        if not isinstance(image, torch.Tensor):
            raise TypeError("official decoder returned an unsupported image type")
        tensor = remap_image_torch(image).detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("expected one generated image")
            tensor = tensor[0]
        array = tensor.permute(1, 2, 0).numpy()
        Image.fromarray(array).convert("RGB").save(path, format="PNG")
        dimensions = [int(tensor.shape[-1]), int(tensor.shape[-2])]
    enforce_private_file_mode(path)
    return dimensions


def run_official_paired_t2i(output_run_id: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this adapter through Slurm")

    _verify_official_checkout()
    output_run = require_run_directory(output_run_id)
    manifest = read_private_json(output_run / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported paired diagnostic manifest")
    records = manifest.get("cases")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("paired diagnostic must contain exactly two cases")

    config = _compose_official_config(output_run)
    _prepend_import_path(OFFICIAL_REPO)
    from evaluation.chameleon.inference.image_tokenizer import ImageTokenizer
    from model import Diffusion
    from models.datasets.all_dataset import ValDataset

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        str(LLAMA_CHECKPOINT), padding_side="right", local_files_only=True
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<boi>", "<eoi>", "<eos>"]}
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    image_tokenizer = ImageTokenizer(
        cfg_path=str(VQGAN_CONFIG),
        ckpt_path=str(VQGAN_CHECKPOINT),
        device=str(device),
    )
    model = Diffusion(config, tokenizer, image_tokenizer, device)
    state = load_file(str(CHECKPOINT), device="cpu")
    incompatible = model.backbone.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - ALLOWED_MISSING_CHECKPOINT_KEYS or unexpected:
        raise RuntimeError("MeDiM checkpoint schema mismatch")
    model.backbone.tie_weights()
    del state
    model.to(device)
    model.backbone.eval()
    model.backbone.requires_grad_(False)
    model.get_cond_dict = types.MethodType(_get_cond_dict, model)
    model.get_vae = types.MethodType(_get_vae, model)
    model.saved_tokens = defaultdict(list)
    model.save_image_text_pair = lambda *args, **kwargs: None
    torch.cuda.empty_cache()
    checkpoint_hash = sha256_file(CHECKPOINT)

    dataset = ValDataset(
        str(output_run / "dataset" / "mimic"),
        str(output_run / "dataset" / "path"),
        txt_tokenizer=tokenizer,
        max_length=int(config.model.txt_length),
    )
    expected_stems = [str(record["dataset_stem"]) for record in records]
    actual_stems = [Path(path).stem for path in dataset.all_images]
    if actual_stems != expected_stems:
        raise RuntimeError("official ValDataset order does not match manifest")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    completed: list[dict[str, Any]] = []
    for ordinal, (record, raw_batch) in enumerate(zip(records, loader, strict=True)):
        case_id = validate_case_id(str(record["case_id"]))
        seed = int(record["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Upstream's tracked update_batch assumes its caller has placed these
        # two tensors on the model device.
        raw_batch["image"] = raw_batch["image"].to(device)
        raw_batch["text"] = raw_batch["text"].to(device)
        raw_batch["mask"] = raw_batch["mask"].to(device)
        batch = model.update_batch(raw_batch)
        text_length = int(config.model.txt_length)
        image_length = int(config.model.img_length)
        modality = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        modality[:, text_length:] = 1
        batch["modality"] = modality
        batch["attention_mask"] = torch.cat(
            [batch["mask"], torch.ones_like(batch["img_mask"])], dim=-1
        )
        if tuple(batch["input_ids"].shape) != (1, text_length + image_length):
            raise RuntimeError("official token layout has an unexpected shape")

        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            generated, *_ = model.sample_for_fid(
                batch,
                ordinal,
                return_gt_img=False,
                return_gt_txt=False,
            )
        elapsed = time.monotonic() - started

        generation_dir = _private_dir(
            output_run / "cases" / case_id / "generation"
        )
        image_path = generation_dir / "generated_cxr.png"
        dimensions = _save_official_image(generated, image_path)
        if dimensions != [512, 512]:
            raise RuntimeError("generated CXR dimensions are not 512 by 512")
        image_hash = sha256_file(image_path)
        write_private_json(
            generation_dir / "generation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "condition": str(record["condition"]),
                "producer": "official_medim_sample_for_fid_compat",
                "official_repo_commit": EXPECTED_OFFICIAL_COMMIT,
                "official_experiments": [
                    "large_scale_train",
                    "paired_standalone_fid_eval",
                ],
                "official_method": "Diffusion.sample_for_fid",
                "compatibility_hooks": [
                    "get_cond_dict_returns_batch",
                    "get_vae_returns_loaded_image_tokenizer",
                    "caller_moves_val_batch_to_cuda",
                ],
                "checkpoint_sha256": checkpoint_hash,
                "prompt_sha256": str(record["prompt_sha256"]),
                "seed": seed,
                "cfg_scale": float(config.eval.cfg),
                "configured_sampling_steps": int(config.sampling.steps),
                "effective_sampling_steps": image_length,
                "text_token_count": text_length,
                "image_token_count": image_length,
                "generated_cxr_sha256": image_hash,
                "image_dimensions": dimensions,
                "elapsed_seconds": round(elapsed, 3),
                "peak_vram_gib": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                ),
            },
        )
        completed.append(
            {
                "case_id": case_id,
                "generated_cxr_sha256": image_hash,
                "elapsed_seconds": round(elapsed, 3),
                "peak_vram_gib": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                ),
            }
        )

    write_private_json(
        output_run / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": output_run_id,
            "status": "completed",
            "completed_case_count": len(completed),
            "cases": completed,
        },
    )
    return {
        "status": "completed",
        "completed_case_count": len(completed),
        "image_dimensions": [512, 512],
        "inference_seconds": round(
            sum(float(row["elapsed_seconds"]) for row in completed), 3
        ),
        "peak_vram_gib": max(float(row["peak_vram_gib"]) for row in completed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-case-id", default="case_000")
    parser.add_argument("--output-run-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.umask(0o077)
    phase = "staging"
    output_run: Path | None = None
    started = time.monotonic()
    try:
        output_run = stage_two_conditions(
            source_run_id=args.source_run_id,
            source_case_id=args.source_case_id,
            output_run_id=args.output_run_id,
        )
        phase = "official_paired_t2i"
        protected_log = output_run / "logs" / "official_paired_t2i.log"
        with protected_log.open("x", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                result = run_official_paired_t2i(args.output_run_id)
        enforce_private_file_mode(protected_log)
        result["task_elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        if output_run is not None:
            failure_path = output_run / "failure.json"
            if not failure_path.exists():
                write_private_json(
                    failure_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "failed",
                        "phase": phase,
                        "error_type": type(exc).__name__,
                    },
                )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "phase": phase,
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
