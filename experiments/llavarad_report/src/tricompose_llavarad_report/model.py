"""Local-only, frozen LLaVA-Rad report-generation runtime."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from tricompose.privacy import sha256_file
from tricompose.reporting import GeneratedReport


MODEL_REPO = "microsoft/llava-rad"
MODEL_REVISION = "local_content_audited_llavarad_v1"
FIXED_PROMPT = "Given the chest X-ray image, describe the findings in the image:"
CONVERSATION_MODE = "v1"
MAX_NEW_TOKENS = 256
SEED = 42

EXPECTED_MODEL_FILE_SIZES = {
    "adapter_config.json": 521,
    "adapter_model.bin": 319_971_402,
    "biomedclipcxr_518.json": 546,
    "biomedclipcxr_518_checkpoint.pt": 2_363_054_683,
    "config.json": 1_202,
    "non_lora_trainables.bin": 39_864_496,
}

EXPECTED_BASE_FILE_SIZES = {
    "config.json": 615,
    "generation_config.json": 162,
    "pytorch_model-00001-of-00002.bin": 9_976_634_558,
    "pytorch_model-00002-of-00002.bin": 3_500_315_539,
    "pytorch_model.bin.index.json": 26_788,
    "special_tokens_map.json": 438,
    "tokenizer.model": 499_723,
    "tokenizer_config.json": 749,
}

REQUIRED_BIOMEDBERT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
)

REQUIRED_EXTERNAL_FILES = (
    "llava/constants.py",
    "llava/conversation.py",
    "llava/mm_utils.py",
    "llava/model/builder.py",
    "llava/utils.py",
)


def _resolve_directory(path: str | Path, label: str) -> Path:
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} is not a directory")
    return root


def _validate_sizes(root: Path, expected: dict[str, int], label: str) -> None:
    for relative, expected_size in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected {label} component size: {relative}")


def validate_model_snapshot(
    *,
    model_dir: str | Path,
    model_base: str | Path,
    biomedbert_dir: str | Path,
    external_root: str | Path,
) -> dict[str, Any]:
    adapter_root = _resolve_directory(model_dir, "LLaVA-Rad model snapshot")
    base_root = _resolve_directory(model_base, "Vicuna base snapshot")
    text_root = _resolve_directory(biomedbert_dir, "BiomedBERT snapshot")
    source_root = _resolve_directory(external_root, "LLaVA-Rad source checkout")

    _validate_sizes(adapter_root, EXPECTED_MODEL_FILE_SIZES, "LLaVA-Rad")
    _validate_sizes(base_root, EXPECTED_BASE_FILE_SIZES, "Vicuna")
    for relative in REQUIRED_BIOMEDBERT_FILES:
        if not (text_root / relative).is_file():
            raise FileNotFoundError(f"missing BiomedBERT component: {relative}")
    for relative in REQUIRED_EXTERNAL_FILES:
        if not (source_root / relative).is_file():
            raise FileNotFoundError(f"missing LLaVA-Rad source component: {relative}")

    adapter_config = json.loads(
        (adapter_root / "adapter_config.json").read_text(encoding="utf-8")
    )
    model_config = json.loads(
        (adapter_root / "config.json").read_text(encoding="utf-8")
    )
    base_config = json.loads((base_root / "config.json").read_text(encoding="utf-8"))
    base_index = json.loads(
        (base_root / "pytorch_model.bin.index.json").read_text(encoding="utf-8")
    )
    if adapter_config.get("peft_type") != "LORA" or not adapter_config.get(
        "inference_mode"
    ):
        raise ValueError("unexpected LLaVA-Rad adapter contract")
    if adapter_config.get("base_model_name_or_path") != "lmsys/vicuna-7b-v1.5":
        raise ValueError("unexpected LLaVA-Rad base model declaration")
    if model_config.get("model_type") != "llava":
        raise ValueError("unexpected LLaVA-Rad model type")
    if model_config.get("mm_vision_tower") != "biomedclip_cxr_518":
        raise ValueError("unexpected LLaVA-Rad vision tower")
    if base_config.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError("unexpected Vicuna architecture")
    expected_weight_files = sorted(
        name for name in EXPECTED_BASE_FILE_SIZES if name.endswith(".bin")
    )
    indexed_weight_files = sorted(set(base_index.get("weight_map", {}).values()))
    if indexed_weight_files != expected_weight_files:
        raise ValueError("Vicuna weight index is incomplete")

    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "adapter_weight_bytes": EXPECTED_MODEL_FILE_SIZES["adapter_model.bin"],
        "non_lora_weight_bytes": EXPECTED_MODEL_FILE_SIZES[
            "non_lora_trainables.bin"
        ],
        "vision_weight_bytes": EXPECTED_MODEL_FILE_SIZES[
            "biomedclipcxr_518_checkpoint.pt"
        ],
        "base_weight_bytes": sum(
            EXPECTED_BASE_FILE_SIZES[name] for name in expected_weight_files
        ),
        "small_file_sha256": {
            "adapter_config.json": sha256_file(adapter_root / "adapter_config.json"),
            "config.json": sha256_file(adapter_root / "config.json"),
            "vision_config.json": sha256_file(
                adapter_root / "biomedclipcxr_518.json"
            ),
            "base_config.json": sha256_file(base_root / "config.json"),
            "base_weight_index.json": sha256_file(
                base_root / "pytorch_model.bin.index.json"
            ),
            "biomedbert_config.json": sha256_file(text_root / "config.json"),
            "builder.py": sha256_file(source_root / "llava/model/builder.py"),
        },
    }


def _write_runtime_json(path: Path, payload: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


class FrozenLlavaRadRuntime:
    """Official deterministic CXR-findings path with local read-only weights."""

    def __init__(
        self,
        *,
        external_root: str | Path,
        model_dir: str | Path,
        model_base: str | Path,
        biomedbert_dir: str | Path,
        runtime_dir: str | Path,
    ) -> None:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run LLaVA-Rad only through Slurm")

        source_root = _resolve_directory(external_root, "LLaVA-Rad source checkout")
        adapter_root = _resolve_directory(model_dir, "LLaVA-Rad model snapshot")
        base_root = _resolve_directory(model_base, "Vicuna base snapshot")
        text_root = _resolve_directory(biomedbert_dir, "BiomedBERT snapshot")
        self.audit = validate_model_snapshot(
            model_dir=adapter_root,
            model_base=base_root,
            biomedbert_dir=text_root,
            external_root=source_root,
        )

        scratch_root = Path(runtime_dir).resolve(strict=False)
        scratch_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(scratch_root, 0o700)
        vision_config = adapter_root / "biomedclipcxr_518.json"
        vision_checkpoint = adapter_root / "biomedclipcxr_518_checkpoint.pt"
        local_vision_payload = json.loads(vision_config.read_text(encoding="utf-8"))
        local_vision_payload["text_cfg"]["hf_model_name"] = str(text_root)
        local_vision_payload["text_cfg"]["hf_tokenizer_name"] = str(text_root)
        local_vision_config = scratch_root / "biomedclipcxr_518.local.json"
        _write_runtime_json(local_vision_config, local_vision_payload)

        sys.path.insert(0, str(source_root))
        import llava.model.builder as llava_builder
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.utils import disable_torch_init

        if CONVERSATION_MODE not in conv_templates:
            raise ValueError("required LLaVA-Rad conversation template is unavailable")

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        disable_torch_init()

        original_auto_config = llava_builder.AutoConfig

        class LocalAutoConfig:
            @staticmethod
            def from_pretrained(
                location: str | Path, *positional: Any, **keywords: Any
            ) -> Any:
                config = original_auto_config.from_pretrained(
                    location, *positional, **keywords
                )
                if Path(location).resolve() == adapter_root:
                    config.mm_vision_tower_config = str(local_vision_config)
                    config.mm_vision_tower_checkpoint = str(vision_checkpoint)
                return config

        llava_builder.AutoConfig = LocalAutoConfig
        try:
            tokenizer, model, image_processor, _ = (
                llava_builder.load_pretrained_model(
                    str(adapter_root),
                    str(base_root),
                    "llavarad",
                    load_8bit=False,
                    load_4bit=False,
                    device="cuda",
                )
            )
        finally:
            llava_builder.AutoConfig = original_auto_config

        if image_processor is None:
            raise RuntimeError("LLaVA-Rad image processor was not initialized")
        model.eval()
        model.requires_grad_(False)

        question = DEFAULT_IMAGE_TOKEN + "\n" + FIXED_PROMPT
        conversation = conv_templates[CONVERSATION_MODE].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        self._stop_text = (
            conversation.sep
            if conversation.sep_style != SeparatorStyle.TWO
            else conversation.sep2
        )
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        if self._model.training:
            raise RuntimeError("LLaVA-Rad entered training mode")
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise RuntimeError("LLaVA-Rad parameters are not frozen")

    @property
    def peak_vram_gib(self) -> float:
        return self._torch.cuda.max_memory_allocated() / (1024**3)

    def generate(self, image_path: Path) -> GeneratedReport:
        from PIL import Image

        self._assert_frozen()
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB").copy()
        image_tensor = self._image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        self._torch.cuda.synchronize()
        started = time.monotonic()
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                self._input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )
        self._torch.cuda.synchronize()
        elapsed = time.monotonic() - started

        report = self._tokenizer.decode(
            output_ids[0, self._input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        if self._stop_text and report.endswith(self._stop_text):
            report = report[: -len(self._stop_text)].strip()
        if not report:
            raise ValueError("LLaVA-Rad returned an empty report")

        return GeneratedReport(
            canonical_text="FINDINGS:\n" + report,
            findings=report,
            impression=None,
            elapsed_seconds=elapsed,
            metadata={
                "strategy": "greedy",
                "temperature": 0,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "current_images": 1,
                "structured_ehr_used": False,
                "prior_used": False,
                "dtype": "float16",
                "prompt_template": "official_llavarad_findings_v1",
                "seed": SEED,
            },
        )
