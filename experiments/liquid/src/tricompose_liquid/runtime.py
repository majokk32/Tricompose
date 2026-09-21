"""Load-once, batched runtime matching Liquid's released text-to-image path."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CFG = 7.0
DEFAULT_TOP_K = 4096
DEFAULT_TOP_P = 0.96
DEFAULT_TEMPERATURE = 0.99
IMAGE_TOKEN_STEPS = 1024
TEXT_VOCAB_BOUNDARY = 256_000
DEFAULT_IMAGE_SIZE = 512
EXPECTED_WEIGHT_SIZES = {
    "model-00001-of-00004.safetensors": 4_995_496_416,
    "model-00002-of-00004.safetensors": 4_982_953_168,
    "model-00003-of-00004.safetensors": 4_982_953_200,
    "model-00004-of-00004.safetensors": 2_164_320_216,
}
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPECTED_VQ_CONFIG_SIZE = 1_488
EXPECTED_VQ_CHECKPOINT_SIZE = 281_270_377


@dataclass(frozen=True)
class LiquidBatchResult:
    images: list[Any]
    elapsed_seconds: float
    peak_vram_gib: float
    input_token_counts: list[int]


def validate_liquid_deployment(
    model_dir: str | Path,
    vq_config: str | Path,
    vq_checkpoint: str | Path,
) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Liquid model root is not a directory")
    for name in REQUIRED_MODEL_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"missing Liquid model component: {name}")
    for name, expected_size in EXPECTED_WEIGHT_SIZES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing Liquid weight shard: {name}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected Liquid weight shard size: {name}")
    config = Path(vq_config).resolve(strict=True)
    checkpoint = Path(vq_checkpoint).resolve(strict=True)
    if not config.is_file() or config.stat().st_size != EXPECTED_VQ_CONFIG_SIZE:
        raise ValueError("Liquid VQ config is missing or has an unexpected size")
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != EXPECTED_VQ_CHECKPOINT_SIZE
    ):
        raise ValueError("Liquid VQ checkpoint is missing or has an unexpected size")
    return {
        "weight_bytes": sum(EXPECTED_WEIGHT_SIZES.values()),
        "weight_shards": len(EXPECTED_WEIGHT_SIZES),
        "vq_checkpoint_bytes": EXPECTED_VQ_CHECKPOINT_SIZE,
        "checkpoint_layout_verified": True,
    }


def _filter_logits(logits: Any, *, top_k: int, top_p: float) -> Any:
    """Match Liquid's released image-vocabulary top-k/top-p filtering."""

    import torch
    from torch.nn import functional as functional

    filtered = logits.clone()
    filtered[:, :TEXT_VOCAB_BOUNDARY] = -float("inf")
    if top_k > 0:
        top_k = min(max(top_k, 1), filtered.size(-1))
        threshold = torch.topk(filtered, top_k)[0][..., -1, None]
        filtered[filtered < threshold] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        filtered[remove] = -float("inf")
    return filtered


def _sample_rows(
    logits: Any,
    generators: list[Any],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Any:
    import torch
    from torch.nn import functional as functional

    scaled = logits[:, -1, :].float() / max(temperature, 1e-5)
    filtered = _filter_logits(scaled, top_k=top_k, top_p=top_p)
    probs = functional.softmax(filtered, dim=-1)
    if probs.shape[0] != len(generators):
        raise ValueError("Liquid generator count does not match batch size")
    samples = [
        torch.multinomial(probs[row], num_samples=1, generator=generator)
        for row, generator in enumerate(generators)
    ]
    return torch.stack(samples, dim=0)


class FrozenLiquidRuntime:
    """Frozen Liquid 7B plus its released Chameleon VQ decoder."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        evaluation_root: str | Path,
        vq_config: str | Path,
        vq_checkpoint: str | Path,
    ) -> None:
        self.model_dir = Path(model_dir).resolve(strict=True)
        self.evaluation_root = Path(evaluation_root).resolve(strict=True)
        self.vq_config = Path(vq_config).resolve(strict=True)
        self.vq_checkpoint = Path(vq_checkpoint).resolve(strict=True)
        validate_liquid_deployment(
            self.model_dir, self.vq_config, self.vq_checkpoint
        )
        evaluation_text = str(self.evaluation_root)
        if evaluation_text not in sys.path:
            sys.path.insert(0, evaluation_text)

        import torch
        from chameleon.inference.image_tokenizer import ImageTokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Liquid runtime requires a Slurm CUDA allocation")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            padding_side="left",
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to("cuda")
        self.model.eval().requires_grad_(False)
        self.original_vocab_size = len(self.tokenizer)
        self.image_tokenizer = ImageTokenizer(
            cfg_path=str(self.vq_config),
            ckpt_path=str(self.vq_checkpoint),
            device="cuda:0",
        )

    def generate_batch(self, prompts: list[str], seeds: list[int]) -> LiquidBatchResult:
        torch = self.torch
        if not prompts or len(prompts) != len(seeds):
            raise ValueError("Liquid prompts and seeds must be non-empty and aligned")
        if len(prompts) > 4:
            raise ValueError("Liquid batch size is capped at four")
        unconditional = ["<unconditional><boi>"] * len(prompts)
        tokenized = self.tokenizer(
            prompts + unconditional,
            return_tensors="pt",
            padding=True,
        ).to("cuda:0")
        input_token_counts = [
            int(value) for value in tokenized["attention_mask"][: len(prompts)].sum(dim=1)
        ]
        if tokenized["input_ids"].shape[1] + IMAGE_TOKEN_STEPS > 8192:
            raise ValueError("Liquid input plus image tokens exceeds context length")

        generators = []
        for seed in seeds:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(int(seed))
            generators.append(generator)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        input_ids = tokenized["input_ids"]
        current_length = input_ids.shape[1]
        model_kwargs = {
            "attention_mask": tokenized["attention_mask"],
            "use_cache": True,
            "cache_position": torch.arange(current_length, device=input_ids.device),
        }
        predicted: list[Any] = []
        with torch.inference_mode():
            for _ in range(IMAGE_TOKEN_STEPS):
                prepared = self.model.prepare_inputs_for_generation(
                    input_ids, **model_kwargs
                )
                outputs = self.model(
                    **prepared,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cond_logits, uncond_logits = torch.split(
                    outputs.logits[:, -1:, :], len(prompts), dim=0
                )
                guided = uncond_logits + (cond_logits - uncond_logits) * DEFAULT_CFG
                image_tokens = _sample_rows(
                    guided,
                    generators,
                    temperature=DEFAULT_TEMPERATURE,
                    top_k=DEFAULT_TOP_K,
                    top_p=DEFAULT_TOP_P,
                )
                predicted.append(image_tokens)
                next_token = torch.cat([image_tokens, image_tokens], dim=0)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.model.config.is_encoder_decoder,
                )

            image_ids = torch.cat(predicted, dim=1) - self.original_vocab_size
            image_ids = torch.clamp(image_ids, min=0, max=8191)
            images = [
                self.image_tokenizer.pil_from_img_toks(row).convert("RGB")
                for row in image_ids
            ]
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return LiquidBatchResult(
            images=images,
            elapsed_seconds=elapsed,
            peak_vram_gib=peak,
            input_token_counts=input_token_counts,
        )
