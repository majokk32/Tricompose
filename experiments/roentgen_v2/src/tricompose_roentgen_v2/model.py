"""Frozen, local-only RoentGen-v2 runtime."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


OFFICIAL_REPO_COMMIT = "039e34b1a40a1ad58c42522e0332a93bb3930886"
OFFICIAL_MODEL_REVISION = "c88d2bf041a448f505fc13dc009d6991be1c9b0f"
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_INFERENCE_STEPS = 75
DEFAULT_IMAGE_SIZE = 512
DEFAULT_MAX_PROMPT_TOKENS = 77
SUPPORTED_INFERENCE_PRECISIONS = ("float16", "bfloat16")

EXPECTED_WEIGHT_SIZES = {
    "text_encoder/model.safetensors": 1_361_596_304,
    "text_encoder_and_tokenizer/model.safetensors": 1_361_596_304,
    "unet/diffusion_pytorch_model.safetensors": 3_463_726_504,
    "vae/diffusion_pytorch_model.safetensors": 334_643_268,
}

REQUIRED_CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_snapshot(model_dir: str | Path) -> dict[str, Any]:
    """Fail closed unless the audited official snapshot layout is complete."""

    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("RoentGen-v2 model path is not a directory")
    for relative in REQUIRED_CONFIG_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing required model component: {relative}")
    for relative, expected_size in EXPECTED_WEIGHT_SIZES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing required model component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected size for model component: {relative}")
    return {
        "model_revision": OFFICIAL_MODEL_REVISION,
        "model_index_sha256": _sha256(root / "model_index.json"),
        "weight_bytes": sum(EXPECTED_WEIGHT_SIZES.values()),
    }


@dataclass(frozen=True)
class BatchResult:
    images: Sequence[Any]
    prompt_token_counts: tuple[int, ...]
    elapsed_seconds: float
    peak_vram_gib: float


class FrozenRoentgenRuntime:
    """Load the official Diffusers pipeline without network access."""

    def __init__(
        self, model_dir: str | Path, *, precision: str = "bfloat16"
    ) -> None:
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import logging as diffusers_logging
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; use this adapter only through Slurm")
        if precision not in SUPPORTED_INFERENCE_PRECISIONS:
            raise ValueError("unsupported RoentGen-v2 inference precision")
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]

        self.audit = validate_model_snapshot(model_dir)
        self.precision = precision
        diffusers_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()
        self._torch = torch
        self.pipe = DiffusionPipeline.from_pretrained(
            str(Path(model_dir).resolve(strict=True)),
            torch_dtype=torch_dtype,
            local_files_only=True,
            use_safetensors=True,
        )
        self.pipe = self.pipe.to("cuda")
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        self.pipe.requires_safety_checker = False

    def _prompt_token_count(self, prompt: str) -> int:
        tokenizer = self.pipe.tokenizer
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)

    def generate_batch(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        *,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    ) -> BatchResult:
        if not prompts or len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must be non-empty and aligned")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("empty protected prompt")
        token_counts = tuple(self._prompt_token_count(prompt) for prompt in prompts)
        if any(count > max_prompt_tokens for count in token_counts):
            raise ValueError("protected prompt exceeds the tokenizer limit")

        torch = self._torch
        generators = [
            torch.Generator(device="cuda").manual_seed(int(seed)) for seed in seeds
        ]
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.pipe(
                prompt=list(prompts),
                generator=generators,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                num_images_per_prompt=1,
                output_type="pil",
            )
        elapsed = time.monotonic() - started
        images = tuple(output.images)
        if len(images) != len(prompts):
            raise RuntimeError("RoentGen-v2 returned an unexpected image count")
        return BatchResult(
            images=images,
            prompt_token_counts=token_counts,
            elapsed_seconds=elapsed,
            peak_vram_gib=torch.cuda.max_memory_allocated() / (1024**3),
        )
