"""Legacy hand-built MeDiM adapter.

This module does not execute the complete official MeDiM inference entrypoint
and must not be used for claims labeled as official MeDiM inference. Use the
external official ``inference.py`` through ``official_native_io`` instead.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from medim_ehr_prompt.common import (
    create_case_stage,
    enforce_private_file_mode,
    load_config,
    read_private_json,
    require_run_directory,
    sha256_file,
    validate_case_id,
    write_private_json,
)


GENERATION_SCHEMA = "medim_ehr_prompt.generation.v1"
EXPECTED_CASE_SCHEMA = "medim_ehr_prompt.case_input.v1"
ALLOWED_MISSING_CHECKPOINT_KEYS = frozenset({"lm_head.weight"})
OFFICIAL_MASK_IMAGE_RECIPE = "official_mask_image_only"
LEGACY_MASK_IMAGE_RECIPE = "legacy_mask_image_only"


def validate_checkpoint_compatibility(
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> dict[str, int]:
    missing = set(missing_keys)
    unexpected = set(unexpected_keys)
    if missing - ALLOWED_MISSING_CHECKPOINT_KEYS or unexpected:
        raise RuntimeError("MeDiM checkpoint schema mismatch")
    return {
        "missing_key_count": len(missing),
        "allowed_missing_key_count": len(
            missing & ALLOWED_MISSING_CHECKPOINT_KEYS
        ),
        "unexpected_key_count": len(unexpected),
    }


def effective_image_sampling_steps(requested_steps: int, image_tokens: int) -> int:
    """Mirror MeDiM's sampler clamp for a fully masked image-token region."""
    if requested_steps < 1 or image_tokens < 1:
        raise ValueError("sampling steps and image tokens must be positive")
    return min(requested_steps, image_tokens)


def _prepend_import_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _build_medim_config(config: dict[str, Any], output_dir: Path) -> Any:
    medim = config["medim"]
    repo = Path(medim["repo"]).resolve(strict=True)
    os.environ["UNIDISC_DIR"] = str(repo)
    _prepend_import_path(repo / "image_utils" / "src")
    _prepend_import_path(repo)

    from omegaconf import OmegaConf
    from utils import set_omega_conf_resolvers

    set_omega_conf_resolvers()
    base = OmegaConf.load(repo / "configs" / "config.yaml")
    noise = OmegaConf.load(repo / "configs" / "noise" / "loglinear.yaml")
    model = OmegaConf.load(repo / "configs" / "model" / "extra_large.yaml")
    vq = OmegaConf.load(repo / "configs" / "experiments" / "vq16_t2i.yaml")
    experiment = OmegaConf.load(repo / "configs" / "experiments" / "large_scale_train.yaml")
    recipe = str(medim.get("sampling_recipe", LEGACY_MASK_IMAGE_RECIPE))
    if recipe in {OFFICIAL_MASK_IMAGE_RECIPE, LEGACY_MASK_IMAGE_RECIPE}:
        cfg = OmegaConf.merge(base, {"noise": noise}, {"model": model}, vq, experiment)
    else:
        raise ValueError("unsupported MeDiM sampling recipe")

    cfg.mode = "eval"
    cfg.debug = False
    cfg.backbone = "dit"
    cfg.parameterization = "subs"
    cfg.T = 0
    cfg.model.llama_ckpt = str(Path(medim["llama_checkpoint"]).resolve(strict=True))
    cfg.model.liquid_ckpt = str(Path(medim["liquid_checkpoint"]).resolve(strict=True))
    cfg.model.vqgan_config = str(Path(medim["vqgan_config"]).resolve(strict=True))
    cfg.model.vqgan_ckpt = str(Path(medim["vqgan_checkpoint"]).resolve(strict=True))
    cfg.model.txt_length = int(medim["text_tokens"])
    cfg.model.img_length = int(medim["image_tokens"])
    cfg.model.length = cfg.model.txt_length + cfg.model.img_length
    cfg.model.text_vocab_size = 32004

    cfg.trainer.precision = "bf16"
    cfg.trainer.ema = 0.0
    cfg.trainer.use_custom_ema = False
    cfg.trainer.compile = False
    cfg.trainer.fsdp = False
    cfg.trainer.use_gradient_checkpointing = False
    cfg.trainer.image_mode = "discrete"
    cfg.trainer.low_precision_params = False
    cfg.trainer.low_precision_loss = False
    cfg.trainer.disable_forward_autocast_during_eval = False
    cfg.trainer.disable_all_checkpointing = False
    cfg.trainer.output = str(output_dir)
    cfg.output_dir = str(output_dir)
    cfg.logging_dir = str(output_dir)

    cfg.sampling.steps = int(medim["sampling_steps"])
    cfg.sampling.noise_removal = True
    cfg.eval.generate_samples = False
    cfg.eval.task = "mask_image_only"
    cfg.eval.cfg = float(medim.get("cfg_scale", 5.0))
    cfg.eval.class_conditional_fid = False
    cfg.eval.unconditional_fid = False
    cfg.eval.txt_conditional_fid = False
    cfg.model.image_model_fid_eval = False
    cfg.wandb.mode = "disabled"
    return cfg


class FrozenMedimRuntime:
    """One frozen model instance reused for every case in an array shard."""

    def __init__(self, config: dict[str, Any], private_output_dir: Path) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run MeDiM through Slurm")
        self.torch = torch
        self.config_dict = config
        self.sampling_recipe = str(
            config["medim"].get("sampling_recipe", LEGACY_MASK_IMAGE_RECIPE)
        )
        self.device = torch.device("cuda:0")
        self.cfg = _build_medim_config(config, private_output_dir)

        medim = config["medim"]
        repo = Path(medim["repo"]).resolve(strict=True)
        _prepend_import_path(repo / "image_utils" / "src")
        _prepend_import_path(repo)

        from evaluation.chameleon.inference.image_tokenizer import ImageTokenizer
        from medim.tokenizers.image_tokenizers import decode_latents
        from model_inf import Diffusion

        self.decode_latents = decode_latents
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.llama_ckpt,
            padding_side="right",
            local_files_only=True,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<boi>", "<eoi>", "<eos>"]}
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        self.image_tokenizer = ImageTokenizer(
            cfg_path=self.cfg.model.vqgan_config,
            ckpt_path=self.cfg.model.vqgan_ckpt,
            device=str(self.device),
        )
        self.model = Diffusion(
            self.cfg,
            self.tokenizer,
            self.image_tokenizer,
            self.device,
        )
        checkpoint = Path(medim["checkpoint"]).resolve(strict=True)
        state = load_file(str(checkpoint), device="cpu")
        incompatible = self.model.backbone.load_state_dict(state, strict=False)
        compatibility = validate_checkpoint_compatibility(
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        )
        self.model.backbone.tie_weights()
        self.checkpoint_load = {
            **compatibility,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        }
        del state

        self.model.to(self.device)
        self.model.backbone.eval()
        self.model.backbone.requires_grad_(False)
        self.model.save_image_text_pair = lambda *args, **kwargs: None
        torch.cuda.empty_cache()

    def _special(self, token: str) -> Any:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError("MeDiM special token is unavailable")
        return self.torch.tensor([[token_id]], dtype=self.torch.long, device=self.device)

    def generate(self, prompt: str, *, seed: int, max_prompt_tokens: int) -> tuple[Any, dict[str, Any]]:
        torch = self.torch
        if not prompt.strip():
            raise ValueError("empty protected prompt")
        if max_prompt_tokens < 2 or max_prompt_tokens >= int(self.cfg.model.txt_length):
            raise ValueError("invalid prompt token limit")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        prompt_length = int(encoded.shape[1])
        if prompt_length > max_prompt_tokens:
            raise ValueError("protected prompt exceeds configured token limit")
        total_length = int(self.cfg.model.length)
        text_length = int(self.cfg.model.txt_length)
        image_length = int(self.cfg.model.img_length)

        official_text = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=text_length,
            truncation=True,
        )
        text_condition = official_text["input_ids"].to(self.device)
        text_attention_mask = official_text["attention_mask"].to(self.device)
        image_mask = torch.full(
            (1, image_length),
            fill_value=int(self.model.mask_index),
            dtype=torch.long,
            device=self.device,
        )
        x0 = torch.cat([text_condition, image_mask], dim=-1)
        if int(x0.shape[1]) != total_length:
            raise AssertionError("invalid MeDiM text-image token layout")
        x0_unmask = torch.zeros_like(x0, dtype=torch.bool)
        x0_unmask[:, :text_length] = True

        modality = torch.zeros_like(x0, dtype=torch.long)
        modality[:, text_length:] = 1

        condition_width = int(
            self.config_dict["medim"].get(
                "conditioning_width", self.config_dict["medim"]["image_size"]
            )
        )
        condition_height = int(
            self.config_dict["medim"].get(
                "conditioning_height", self.config_dict["medim"]["image_size"]
            )
        )
        size_condition = self.tokenizer(
            [
                "Image Size: Width is {} Height is {}.".format(
                    condition_width,
                    condition_height,
                )
            ],
            return_tensors="pt",
            padding="longest",
            max_length=text_length,
            truncation=True,
        )["input_ids"].to(self.device)
        batch = {
            "boi": self._special("<boi>"),
            "eoi": self._special("<eoi>"),
            "eos": self._special("<eos>"),
            "cond": size_condition,
            "mask": text_attention_mask,
            "img_mask": torch.ones((1, image_length), dtype=torch.long, device=self.device),
            "input_ids": x0,
            "attention_mask": torch.cat(
                [
                    text_attention_mask,
                    torch.ones(
                        (1, image_length),
                        dtype=text_attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            ),
            "modality": modality,
        }

        requested_steps = int(self.cfg.sampling.steps)
        effective_steps = effective_image_sampling_steps(requested_steps, image_length)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            _condition_text_tokens, image_tokens = self.model._sample(
                num_steps=effective_steps,
                text_only=False,
                x0=x0,
                x0_unmask=x0_unmask,
                sample_modality=modality,
                batch=batch,
            )
        elapsed = time.monotonic() - started
        image = self.decode_latents(
            self.cfg,
            self.image_tokenizer,
            image_tokens,
        )
        metadata = {
            "prompt_token_count": prompt_length,
            "condition_text_token_count": text_length,
            "image_token_count": int(image_tokens.shape[1]),
            "sampling_recipe": self.sampling_recipe,
            "requested_sampling_steps": requested_steps,
            "effective_sampling_steps": effective_steps,
            "cfg_scale": float(self.cfg.eval.cfg),
            "conditioning_dimensions": [condition_width, condition_height],
            "elapsed_seconds": round(elapsed, 3),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        }
        return image, metadata


def _save_generated_image(image: Any, path: Path) -> list[int]:
    if path.exists():
        raise FileExistsError("generated image path already exists")
    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            raise ValueError("expected one decoded image")
        image = image[0]

    if hasattr(image, "save") and hasattr(image, "size"):
        image.save(path, format="PNG")
        dimensions = [int(image.size[0]), int(image.size[1])]
    else:
        import torch
        from torchvision.utils import save_image

        if not isinstance(image, torch.Tensor):
            raise TypeError("unsupported decoded image type")
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError("decoded image tensor must be CHW")
        tensor = (tensor.clamp(-1, 1) + 1) / 2
        save_image(tensor, path)
        dimensions = [int(tensor.shape[-1]), int(tensor.shape[-2])]
    enforce_private_file_mode(path)
    return dimensions


def _case_records(selection: dict[str, Any], task_index: int, num_shards: int) -> list[dict[str, Any]]:
    if selection.get("schema_version") != "medim_ehr_prompt.selection.v1":
        raise ValueError("unsupported selection artifact")
    cases = selection.get("cases")
    if not isinstance(cases, list):
        raise TypeError("selection cases must be a list")
    if num_shards < 1 or task_index < 0 or task_index >= num_shards:
        raise ValueError("invalid shard selection")
    return [case for ordinal, case in enumerate(cases) if ordinal % num_shards == task_index]


def run_shard(
    *,
    config_path: str | Path,
    run_id: str,
    task_index: int,
    num_shards: int,
) -> dict[str, Any]:
    os.umask(0o077)
    config = load_config(config_path)
    expected_shards = int(config["execution"]["inference_shards"])
    if num_shards != expected_shards:
        raise ValueError("requested shard count does not match experiment config")
    run_dir = require_run_directory(run_id)
    selection = read_private_json(run_dir / "selection.json")
    records = _case_records(selection, task_index, num_shards)
    if not records:
        raise ValueError("selected inference shard is empty")

    setup_dir = run_dir / f"runtime_setup_{task_index:02d}"
    if setup_dir.exists():
        raise FileExistsError("runtime setup directory already exists")
    setup_dir.mkdir(mode=0o700, exist_ok=False)
    runtime = FrozenMedimRuntime(config, setup_dir)

    completed: list[dict[str, Any]] = []
    for record in records:
        case_id = validate_case_id(str(record["case_id"]))
        case_input_path = run_dir / "cases" / case_id / "input.json"
        case_input = read_private_json(case_input_path)
        if case_input.get("schema_version") != EXPECTED_CASE_SCHEMA:
            raise ValueError("unsupported case input artifact")
        if case_input.get("prompt_sha256") != record.get("prompt_sha256"):
            raise ValueError("case prompt hash mismatch")

        stage_dir = create_case_stage(run_dir, case_id, "generation")
        original_cwd = Path.cwd()
        started = time.monotonic()
        try:
            os.chdir(stage_dir)
            image, generation_meta = runtime.generate(
                str(case_input["medim_prompt"]),
                seed=int(case_input["seed"]),
                max_prompt_tokens=int(config["prompt"]["max_prompt_tokens"]),
            )
            image_path = stage_dir / "generated_cxr.png"
            dimensions = _save_generated_image(image, image_path)
            expected_size = int(config["medim"]["image_size"])
            if dimensions != [expected_size, expected_size]:
                raise ValueError("generated image dimensions do not match config")
            artifact = write_private_json(
                stage_dir / "generation.json",
                {
                    "schema_version": GENERATION_SCHEMA,
                    "run_id": run_id,
                    "case_id": case_id,
                    "producer": "frozen_official_medim_mask_image_adapter",
                    "prompt_sha256": str(case_input["prompt_sha256"]),
                    "seed": int(case_input["seed"]),
                    "sampling_recipe": runtime.sampling_recipe,
                    "sampling_steps": {
                        "requested": generation_meta["requested_sampling_steps"],
                        "effective": generation_meta["effective_sampling_steps"],
                    },
                    "checkpoint_load": runtime.checkpoint_load,
                    "output": {
                        "image_sha256": sha256_file(image_path),
                        "image_dimensions": dimensions,
                    },
                    "cost": generation_meta,
                    "case_elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            completed.append(
                {
                    "generation_sha256": sha256_file(artifact),
                    "elapsed_seconds": generation_meta["elapsed_seconds"],
                    "peak_vram_gib": generation_meta["peak_vram_gib"],
                }
            )
        except Exception as exc:
            write_private_json(
                stage_dir / "failure.json",
                {
                    "stage": "medim_generate",
                    "status": "failed",
                    "case_id": case_id,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        finally:
            os.chdir(original_cwd)

    return {
        "stage": "medim_generate",
        "status": "ok",
        "run_id": run_id,
        "task_index": task_index,
        "num_shards": num_shards,
        "case_count": len(completed),
        "artifact_hashes": [row["generation_sha256"] for row in completed],
        "inference_seconds": round(
            sum(float(row["elapsed_seconds"]) for row in completed),
            3,
        ),
        "peak_vram_gib": max(float(row["peak_vram_gib"]) for row in completed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.monotonic()
    phase = "setup"
    try:
        with Path(os.devnull).open("w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                phase = "model_or_case_inference"
                result = run_shard(
                    config_path=args.config,
                    run_id=args.run_id,
                    task_index=args.task_index,
                    num_shards=args.num_shards,
                )
        result["task_elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "stage": "medim_generate",
                    "status": "failed",
                    "task_index": args.task_index,
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
