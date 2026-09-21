"""Audited, frozen and local-only CheXGenBench PixArt-Sigma runtime."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


OFFICIAL_REPO_COMMIT = "cc7e91ebcee946836090acb4c399e4f3912280ac"
OFFICIAL_MODEL_REPO = "raman07/CheXGenBench-Models-Pixart-Sigma"
OFFICIAL_MODEL_REVISION = "ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727"
DEFAULT_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_MAX_SEQUENCE_LENGTH = 300

EXPECTED_MODEL_INDEX_SHA256 = (
    "e23af2e43699dd034cbe96413e43ec9fe9c4ffa89c08cf6f06df3b4cfab669d9"
)
EXPECTED_WEIGHT_SIZES = {
    "text_encoder/model-00001-of-00004.safetensors": 4_989_319_680,
    "text_encoder/model-00002-of-00004.safetensors": 4_999_830_656,
    "text_encoder/model-00003-of-00004.safetensors": 4_865_612_720,
    "text_encoder/model-00004-of-00004.safetensors": 4_194_506_688,
    "transformer/diffusion_pytorch_model.safetensors": 2_443_492_488,
    "vae/diffusion_pytorch_model.safetensors": 334_643_268,
}
REQUIRED_CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "vae/config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_snapshot(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("PixArt model path is not a directory")
    for relative in REQUIRED_CONFIG_FILES:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"missing required model component: {relative}")
    for relative, expected_size in EXPECTED_WEIGHT_SIZES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing required model component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected size for model component: {relative}")

    model_index = root / "model_index.json"
    if _sha256(model_index) != EXPECTED_MODEL_INDEX_SHA256:
        raise ValueError("unexpected model_index.json hash")
    payload = json.loads(model_index.read_text(encoding="utf-8"))
    if payload.get("_class_name") != "PixArtSigmaPipeline":
        raise ValueError("snapshot is not a PixArtSigmaPipeline")
    if payload.get("text_encoder") != ["transformers", "T5EncoderModel"]:
        raise ValueError("snapshot has an unexpected text encoder")
    weight_sha256 = {
        relative: _sha256(root / relative)
        for relative in sorted(EXPECTED_WEIGHT_SIZES)
    }
    return {
        "model_revision": OFFICIAL_MODEL_REVISION,
        "model_index_sha256": EXPECTED_MODEL_INDEX_SHA256,
        "weight_bytes": sum(EXPECTED_WEIGHT_SIZES.values()),
        "weight_sha256": weight_sha256,
    }


@dataclass(frozen=True)
class BatchResult:
    images: Sequence[Any]
    prompt_token_counts: tuple[int, ...]
    elapsed_seconds: float
    peak_vram_gib: float


class FrozenPixArtRuntime:
    def __init__(self, model_dir: str | Path) -> None:
        import torch
        from diffusers import PixArtSigmaPipeline
        from diffusers.utils import logging as diffusers_logging
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run this adapter only through Slurm")
        self.audit = validate_model_snapshot(model_dir)
        diffusers_logging.set_verbosity_error()
        transformers_logging.set_verbosity_error()
        self._torch = torch
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            str(Path(model_dir).resolve(strict=True)),
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
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
                raise RuntimeError(f"PixArt component entered training mode: {name}")
            if any(parameter.requires_grad for parameter in component.parameters()):
                raise RuntimeError(f"PixArt component is not frozen: {name}")

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

    def generate_batch(self, prompts: Sequence[str], seeds: Sequence[int]) -> BatchResult:
        if not prompts or len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must be non-empty and aligned")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("empty protected prompt")
        self._assert_frozen()
        counts = tuple(self._prompt_token_count(prompt) for prompt in prompts)
        if any(count > DEFAULT_MAX_SEQUENCE_LENGTH for count in counts):
            raise ValueError("protected prompt exceeds PixArt tokenizer limit")

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
                max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
                clean_caption=False,
                use_resolution_binning=True,
                output_type="pil",
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        images = tuple(result.images)
        if len(images) != len(prompts):
            raise RuntimeError("PixArt returned an unexpected image count")
        return BatchResult(
            images=images,
            prompt_token_counts=counts,
            elapsed_seconds=elapsed,
            peak_vram_gib=torch.cuda.max_memory_allocated() / (1024**3),
        )
