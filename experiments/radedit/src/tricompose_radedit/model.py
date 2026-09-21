"""Audited, frozen and local-only RadEdit text-to-image runtime."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


RADEDIT_REPO = "microsoft/radedit"
RADEDIT_REVISION = "e8ebd31396ff8553c34084b98b5defb8ebea2817"
VAE_REPO = "stabilityai/sdxl-vae"
VAE_REVISION = "6f5909a7e596173e25d4e97b07fd19cdf9611c76"
TEXT_ENCODER_REPO = "microsoft/BiomedVLP-BioViL-T"
TEXT_ENCODER_REVISION = "692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23"

DEFAULT_INFERENCE_STEPS = 100
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_MAX_SEQUENCE_LENGTH = 128

EXPECTED_FILE_SIZES = {
    "radedit/unet/config.json": 1_935,
    "radedit/unet/diffusion_pytorch_model.safetensors": 1_757_849_552,
    "sdxl-vae/config.json": 607,
    "sdxl-vae/diffusion_pytorch_model.safetensors": 334_643_268,
    "biovil-t/config.json": 803,
    "biovil-t/configuration_cxrbert.py": 888,
    "biovil-t/model.safetensors": 440_909_232,
    "biovil-t/modeling_cxrbert.py": 5_876,
    "biovil-t/special_tokens_map.json": 112,
    "biovil-t/tokenizer_config.json": 741,
    "biovil-t/vocab.txt": 235_402,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_bundle(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("RadEdit model bundle is not a directory")
    for relative, expected_size in EXPECTED_FILE_SIZES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing required model component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected size for model component: {relative}")

    weight_paths = tuple(
        path for path in EXPECTED_FILE_SIZES if path.endswith(".safetensors")
    )
    return {
        "radedit_revision": RADEDIT_REVISION,
        "vae_revision": VAE_REVISION,
        "text_encoder_revision": TEXT_ENCODER_REVISION,
        "weight_bytes": sum(EXPECTED_FILE_SIZES[path] for path in weight_paths),
        "weight_sha256": {path: _sha256(root / path) for path in weight_paths},
        "config_sha256": {
            path: _sha256(root / path)
            for path in sorted(EXPECTED_FILE_SIZES)
            if not path.endswith(".safetensors")
        },
    }


@dataclass(frozen=True)
class BatchResult:
    images: Sequence[Any]
    prompt_token_counts: tuple[int, ...]
    elapsed_seconds: float
    peak_vram_gib: float


class FrozenRadEditRuntime:
    """Official text-to-image construction with every learned module frozen."""

    def __init__(self, model_dir: str | Path) -> None:
        import torch
        from diffusers import (
            AutoencoderKL,
            DDIMScheduler,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from diffusers.utils import logging as diffusers_logging
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run this adapter only through Slurm")
        root = Path(model_dir).resolve(strict=True)
        self.audit = validate_model_bundle(root)
        diffusers_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()
        self._torch = torch

        unet = UNet2DConditionModel.from_pretrained(
            str(root / "radedit" / "unet"),
            local_files_only=True,
            use_safetensors=True,
        )
        vae = AutoencoderKL.from_pretrained(
            str(root / "sdxl-vae"),
            local_files_only=True,
            use_safetensors=True,
        )
        text_encoder = AutoModel.from_pretrained(
            str(root / "biovil-t"),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(root / "biovil-t"),
            model_max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            trust_remote_code=True,
            local_files_only=True,
        )
        scheduler = DDIMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        self.pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        for component in self.pipe.components.values():
            if isinstance(component, torch.nn.Module):
                component.eval()
                component.requires_grad_(False)

    def _assert_frozen(self) -> None:
        torch = self._torch
        for name, component in self.pipe.components.items():
            if not isinstance(component, torch.nn.Module):
                continue
            if component.training:
                raise RuntimeError(f"RadEdit component entered training mode: {name}")
            if any(parameter.requires_grad for parameter in component.parameters()):
                raise RuntimeError(f"RadEdit component is not frozen: {name}")

    def _prompt_token_count(self, prompt: str) -> int:
        encoded = self.pipe.tokenizer(
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
        self, prompts: Sequence[str], seeds: Sequence[int]
    ) -> BatchResult:
        if not prompts or len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must be non-empty and aligned")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("empty protected prompt")
        self._assert_frozen()
        counts = tuple(self._prompt_token_count(prompt) for prompt in prompts)
        if any(count > DEFAULT_MAX_SEQUENCE_LENGTH for count in counts):
            raise ValueError("protected prompt exceeds BioViL-T tokenizer limit")

        torch = self._torch
        generators = [
            torch.Generator(device="cuda").manual_seed(int(seed)) for seed in seeds
        ]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            result = self.pipe(
                prompt=list(prompts),
                generator=generators,
                num_inference_steps=DEFAULT_INFERENCE_STEPS,
                guidance_scale=DEFAULT_GUIDANCE_SCALE,
                height=DEFAULT_HEIGHT,
                width=DEFAULT_WIDTH,
                output_type="pil",
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        images = tuple(result.images)
        if len(images) != len(prompts):
            raise RuntimeError("RadEdit returned an unexpected image count")
        return BatchResult(
            images=images,
            prompt_token_counts=counts,
            elapsed_seconds=elapsed,
            peak_vram_gib=torch.cuda.max_memory_allocated() / (1024**3),
        )
