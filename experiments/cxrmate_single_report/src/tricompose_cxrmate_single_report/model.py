"""Local-only, frozen CXRMate single-image runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tricompose.privacy import sha256_file
from tricompose.reporting import GeneratedReport


MODEL_REPO = "aehrc/cxrmate-single-tf"
MODEL_REVISION = "84dfcba8125c9b9296bfc1ee317749788b6b59b4"
MAX_LENGTH = 256
NUM_BEAMS = 4

EXPECTED_FILE_SIZES = {
    "config.json": 78_455,
    "generation_config.json": 90,
    "model.safetensors": 449_521_072,
    "modelling_single.py": 16_577,
    "preprocessor_config.json": 679,
    "special_tokens_map.json": 1_088,
    "tokenizer.json": 1_331_932,
    "tokenizer_config.json": 2_474,
}


def validate_model_snapshot(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("CXRMate single snapshot is not a directory")
    for relative, expected_size in EXPECTED_FILE_SIZES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing CXRMate component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected CXRMate component size: {relative}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["SingleCXREncoderDecoderModel"]:
        raise ValueError("unexpected CXRMate architecture")
    if config.get("auto_map", {}).get("AutoModel") != (
        "modelling_single.SingleCXREncoderDecoderModel"
    ):
        raise ValueError("unexpected CXRMate custom model mapping")
    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "weight_bytes": EXPECTED_FILE_SIZES["model.safetensors"],
        "small_file_sha256": {
            path: sha256_file(root / path)
            for path in (
                "config.json",
                "generation_config.json",
                "modelling_single.py",
                "preprocessor_config.json",
                "tokenizer_config.json",
            )
        },
    }


class FrozenCXRMateSingleRuntime:
    """Teacher-forced, single-image model with deterministic beam decoding."""

    def __init__(self, model_dir: str | Path) -> None:
        import torch
        import transformers
        from torchvision import transforms
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run CXRMate only through Slurm")
        root = Path(model_dir).resolve(strict=True)
        self.audit = validate_model_snapshot(root)
        self._torch = torch
        transformers_logging.set_verbosity_error()
        torch.cuda.reset_peak_memory_stats()
        self.model = transformers.AutoModel.from_pretrained(
            str(root),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
        ).eval().to("cuda")
        self.model.requires_grad_(False)
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            str(root), local_files_only=True
        )
        image_processor = transformers.AutoFeatureExtractor.from_pretrained(
            str(root), local_files_only=True
        )
        size = int(image_processor.size["shortest_edge"])
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.CenterCrop(size=[size, size]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean,
                    std=image_processor.image_std,
                ),
            ]
        )
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        if self.model.training:
            raise RuntimeError("CXRMate entered training mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("CXRMate parameters are not frozen")

    @property
    def peak_vram_gib(self) -> float:
        return self._torch.cuda.max_memory_allocated() / (1024**3)

    def generate(self, image_path: Path) -> GeneratedReport:
        from PIL import Image

        self._assert_frozen()
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB").copy()
        pixel_values = self.transform(image).unsqueeze(0).to("cuda")
        common_kwargs = {
            "pixel_values": pixel_values,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "use_cache": True,
            "num_beams": NUM_BEAMS,
            "special_token_ids": [self.tokenizer.sep_token_id],
            "max_length": MAX_LENGTH,
        }
        self._torch.cuda.synchronize()
        started = time.monotonic()
        with self._torch.inference_mode():
            outputs = self.model.generate(**common_kwargs)
        self._torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        findings, impression = self.model.split_and_decode_sections(
            outputs.sequences,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )
        findings_text = str(findings[0]).strip()
        impression_text = str(impression[0]).strip()
        if not findings_text and not impression_text:
            raise ValueError("CXRMate returned an empty report")
        sections = []
        if findings_text:
            sections.append("FINDINGS:\n" + findings_text)
        if impression_text:
            sections.append("IMPRESSION:\n" + impression_text)
        return GeneratedReport(
            canonical_text="\n\n".join(sections),
            findings=findings_text or None,
            impression=impression_text or None,
            elapsed_seconds=elapsed,
            metadata={
                "strategy": "beam_search",
                "num_beams": NUM_BEAMS,
                "max_length": MAX_LENGTH,
                "current_images": 1,
                "longitudinal": False,
                "reinforcement_learning": False,
                "dtype": "float32",
            },
        )

