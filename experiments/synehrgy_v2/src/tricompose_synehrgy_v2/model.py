"""Local-only frozen runtimes for pinned official SynEHRgy-v2 checkpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tricompose.privacy import sha256_file


@dataclass(frozen=True)
class ModelSpec:
    """Immutable provenance and integrity policy for one model variant."""

    variant: str
    repo: str
    revision: str
    checkpoint: str
    architecture: str
    model_type: str
    context_length: int
    expected_file_sizes: Mapping[str, int]
    expected_sha256: Mapping[str, str]


_GPT2_CHECKPOINT = "checkpoint-18000"
_QWEN2_CHECKPOINT = "checkpoint-18000"

MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt2-10bins": ModelSpec(
        variant="gpt2-10bins",
        repo="hojjatkarami/SynEHRgy-gpt2-10bins",
        revision="ac6476b0376c26618386e4ca0d5d0c9d30be8c84",
        checkpoint=_GPT2_CHECKPOINT,
        architecture="GPT2LMHeadModel",
        model_type="gpt2",
        context_length=2048,
        expected_file_sizes={
            "config_main.yaml": 1_663,
            f"{_GPT2_CHECKPOINT}/config.json": 761,
            f"{_GPT2_CHECKPOINT}/generation_config.json": 148,
            f"{_GPT2_CHECKPOINT}/model.safetensors": 36_349_976,
            f"{_GPT2_CHECKPOINT}/special_tokens_map.json": 51,
            f"{_GPT2_CHECKPOINT}/tokenizer_config.json": 625,
            f"{_GPT2_CHECKPOINT}/vocab.json": 183_597,
            f"{_GPT2_CHECKPOINT}/ehr_tokenizer/special_tokens_map.json": 96,
            f"{_GPT2_CHECKPOINT}/ehr_tokenizer/tokenizer.json": 197_086,
            f"{_GPT2_CHECKPOINT}/ehr_tokenizer/tokenizer_config.json": 946,
        },
        expected_sha256={
            "config_main.yaml": (
                "0efb7e492b3245b49a7033c5b9d5ddc95425a76c2d598016ac57ad39fd5e13fb"
            ),
            f"{_GPT2_CHECKPOINT}/model.safetensors": (
                "73dcfa29a8f79b542cdb7453ed9805c7e4ae11722a9b66deb598dedacd486909"
            ),
            f"{_GPT2_CHECKPOINT}/vocab.json": (
                "13446162e2c57394bbd09ef3c28e4afffd68dc309e4fd08841e2e6986ccd874e"
            ),
        },
    ),
    "qwen2-40bins": ModelSpec(
        variant="qwen2-40bins",
        repo="hojjatkarami/SynEHRgy-qwen2-40bins",
        revision="41263cd34d287ed7a32a8122fa8a72a29e2e9e2b",
        checkpoint=_QWEN2_CHECKPOINT,
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        # The model config permits 4096 positions, but the author's
        # config_main.yaml fixes generation/training n_ctx to 2048.
        context_length=2048,
        expected_file_sizes={
            "config_main.yaml": 1_939,
            f"{_QWEN2_CHECKPOINT}/config.json": 797,
            f"{_QWEN2_CHECKPOINT}/generation_config.json": 170,
            f"{_QWEN2_CHECKPOINT}/model.safetensors": 42_637_064,
            f"{_QWEN2_CHECKPOINT}/special_tokens_map.json": 51,
            f"{_QWEN2_CHECKPOINT}/tokenizer_config.json": 625,
            f"{_QWEN2_CHECKPOINT}/vocab.json": 184_047,
            f"{_QWEN2_CHECKPOINT}/ehr_tokenizer/special_tokens_map.json": 96,
            f"{_QWEN2_CHECKPOINT}/ehr_tokenizer/tokenizer.json": 197_656,
            f"{_QWEN2_CHECKPOINT}/ehr_tokenizer/tokenizer_config.json": 946,
        },
        expected_sha256={
            "config_main.yaml": (
                "84720be1e01a997a65ce5127ba7b333bf6bc538998b498bd33d939cf04274525"
            ),
            f"{_QWEN2_CHECKPOINT}/config.json": (
                "309a98ce4d273d10ca7f6357099f5641e11c62ab9b0a023a944b65fc5e88ebd3"
            ),
            f"{_QWEN2_CHECKPOINT}/generation_config.json": (
                "b01a60f8ee0587c9c1a1f2d6c03a01f49a6acd81fd8163565a45de76495723bb"
            ),
            f"{_QWEN2_CHECKPOINT}/model.safetensors": (
                "448e1d2d8321ede77aa59089e76835509b1ce75e011ca21d452dd722c88cf0c6"
            ),
            f"{_QWEN2_CHECKPOINT}/special_tokens_map.json": (
                "0a04330605f956cd9b4495eebfc928d4fb859273da7025d95da0cbcc3570152b"
            ),
            f"{_QWEN2_CHECKPOINT}/tokenizer_config.json": (
                "c40e8ae612e37eb8902d3d8a2139cfa90e46fa365dc198846aca43365ea444c5"
            ),
            f"{_QWEN2_CHECKPOINT}/vocab.json": (
                "170ff0298ba8cacacb7176db586f07505dfff887961e08a3f06fed5df8c3da07"
            ),
            f"{_QWEN2_CHECKPOINT}/ehr_tokenizer/special_tokens_map.json": (
                "937ae3bcbf7d8fb4962a2bd1198a81b1d6feca96d087d8f698eb9ee1bdfbb256"
            ),
            f"{_QWEN2_CHECKPOINT}/ehr_tokenizer/tokenizer.json": (
                "78017eef7f509c311e8300e95e26a02fe640ffe258d58cebe728b7a10c7d7ac6"
            ),
            f"{_QWEN2_CHECKPOINT}/ehr_tokenizer/tokenizer_config.json": (
                "3f985fa77122660f6614ecf7f49c0f2f36304e84a2bfa2f0b3981aefdff78c39"
            ),
        },
    ),
}

DEFAULT_MODEL_VARIANT = "gpt2-10bins"

# Backwards-compatible aliases used by the first GPT-2 adapter.
MODEL_REPO = MODEL_SPECS[DEFAULT_MODEL_VARIANT].repo
MODEL_REVISION = MODEL_SPECS[DEFAULT_MODEL_VARIANT].revision
CHECKPOINT_NAME = MODEL_SPECS[DEFAULT_MODEL_VARIANT].checkpoint
MODEL_SHA256 = MODEL_SPECS[DEFAULT_MODEL_VARIANT].expected_sha256[
    f"{CHECKPOINT_NAME}/model.safetensors"
]
VOCAB_SHA256 = MODEL_SPECS[DEFAULT_MODEL_VARIANT].expected_sha256[
    f"{CHECKPOINT_NAME}/vocab.json"
]
CONFIG_MAIN_SHA256 = MODEL_SPECS[DEFAULT_MODEL_VARIANT].expected_sha256[
    "config_main.yaml"
]
EXPECTED_FILE_SIZES = MODEL_SPECS[DEFAULT_MODEL_VARIANT].expected_file_sizes


def get_model_spec(model_variant: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_variant]
    except KeyError as exc:
        raise ValueError(f"unsupported SynEHRgy model variant: {model_variant}") from exc


def _model_context_capacity(config: Mapping[str, Any]) -> int:
    for field in ("n_positions", "max_position_embeddings"):
        value = config.get(field)
        if value is not None:
            return int(value)
    raise ValueError("SynEHRgy model config has no context-capacity field")


def validate_model_snapshot(
    model_dir: str | Path,
    model_variant: str = DEFAULT_MODEL_VARIANT,
) -> dict[str, Any]:
    spec = get_model_spec(model_variant)
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("SynEHRgy model path is not a directory")
    for relative, expected_size in spec.expected_file_sizes.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing SynEHRgy component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected SynEHRgy component size: {relative}")

    checkpoint = root / spec.checkpoint
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))
    if config.get("architectures") != [spec.architecture]:
        raise ValueError("unexpected SynEHRgy architecture")
    if config.get("model_type") != spec.model_type:
        raise ValueError("unexpected SynEHRgy model type")
    if _model_context_capacity(config) < spec.context_length:
        raise ValueError("SynEHRgy model context is shorter than the official n_ctx")
    if config.get("vocab_size") != len(vocab):
        raise ValueError("SynEHRgy model and vocabulary sizes differ")
    inverse_vocab = {index: token for token, index in vocab.items()}
    if inverse_vocab.get(config.get("bos_token_id")) != "<s>":
        raise ValueError("unexpected SynEHRgy BOS token")
    if inverse_vocab.get(config.get("eos_token_id")) != "</s>":
        raise ValueError("unexpected SynEHRgy EOS token")

    hashes = {
        relative: sha256_file(root / relative)
        for relative in spec.expected_sha256
    }
    if hashes != dict(spec.expected_sha256):
        raise ValueError("SynEHRgy snapshot hash mismatch")
    return {
        "variant": spec.variant,
        "repo": spec.repo,
        "revision": spec.revision,
        "checkpoint": spec.checkpoint,
        "architecture": spec.architecture,
        "context_length": spec.context_length,
        "weight_bytes": spec.expected_file_sizes[
            f"{spec.checkpoint}/model.safetensors"
        ],
        "vocab_size": len(vocab),
        "sha256": hashes,
    }


class FrozenSynEHRgyRuntime:
    """Generate complete patient token streams from only the BOS token."""

    def __init__(
        self,
        model_dir: str | Path,
        model_variant: str = DEFAULT_MODEL_VARIANT,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run SynEHRgy only through Slurm")
        self.spec = get_model_spec(model_variant)
        self.root = Path(model_dir).resolve(strict=True)
        self.audit = validate_model_snapshot(self.root, model_variant)
        self.checkpoint = self.root / self.spec.checkpoint
        self._torch = torch
        config = json.loads((self.checkpoint / "config.json").read_text(encoding="utf-8"))
        self.vocab = json.loads((self.checkpoint / "vocab.json").read_text(encoding="utf-8"))
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.bos_token_id = int(config["bos_token_id"])
        self.eos_token_id = int(config["eos_token_id"])
        self.context_length = self.spec.context_length

        torch.cuda.reset_peak_memory_stats()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
        ).eval().to("cuda")
        self.model.requires_grad_(False)
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        if self.model.training:
            raise RuntimeError("SynEHRgy entered training mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("SynEHRgy parameters are not frozen")

    @property
    def peak_vram_gib(self) -> float:
        return self._torch.cuda.max_memory_allocated() / (1024**3)

    def generate(
        self,
        *,
        seed: int,
        temperature: float,
        top_k: int,
    ) -> tuple[list[str], float]:
        self._assert_frozen()
        torch = self._torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        input_ids = torch.tensor([[self.bos_token_id]], device="cuda")
        attention_mask = torch.ones_like(input_ids)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=1.0,
                repetition_penalty=1.0,
                max_new_tokens=self.context_length - 1,
                eos_token_id=self.eos_token_id,
                pad_token_id=int(self.model.config.pad_token_id),
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        token_ids = output[0].detach().cpu().tolist()
        if self.eos_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(self.eos_token_id) + 1]
        tokens = [self.id_to_token.get(index, "[UNK]") for index in token_ids]
        return tokens, elapsed
