"""Local-only, frozen MAIRA-2 report-generation runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tricompose.privacy import sha256_file
from tricompose.reporting import GeneratedReport


MODEL_REPO = "microsoft/maira-2"
MODEL_REVISION = "795a2b1cd4a310624b4e3d14b5a23e41fd273deb"
MAX_NEW_TOKENS = 300

EXPECTED_FILE_SIZES = {
    "added_tokens.json": 3_723,
    "chat_template.json": 611,
    "config.json": 2_401,
    "configuration_maira2.py": 1_061,
    "generation_config.json": 179,
    "model-00001-of-00006.safetensors": 4_955_289_768,
    "model-00002-of-00006.safetensors": 4_857_207_664,
    "model-00003-of-00006.safetensors": 4_857_207_704,
    "model-00004-of-00006.safetensors": 4_857_207_704,
    "model-00005-of-00006.safetensors": 4_857_207_704,
    "model-00006-of-00006.safetensors": 3_136_688_192,
    "model.safetensors.index.json": 49_835,
    "modeling_maira2.py": 4_747,
    "preprocessor_config.json": 560,
    "processing_maira2.py": 27_708,
    "processor_config.json": 405,
    "special_tokens_map.json": 552,
    "tokenizer.json": 3_656_334,
    "tokenizer.model": 499_723,
    "tokenizer_config.json": 37_032,
}


def validate_model_snapshot(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("MAIRA-2 snapshot is not a directory")
    for relative, expected_size in EXPECTED_FILE_SIZES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing MAIRA-2 component: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"unexpected MAIRA-2 component size: {relative}")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Maira2ForConditionalGeneration"]:
        raise ValueError("unexpected MAIRA-2 architecture")
    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_files = sorted(set(index.get("weight_map", {}).values()))
    expected_weights = sorted(
        path for path in EXPECTED_FILE_SIZES if path.endswith(".safetensors")
    )
    if weight_files != expected_weights:
        raise ValueError("MAIRA-2 weight index is incomplete")
    audited_small_files = (
        "config.json",
        "configuration_maira2.py",
        "model.safetensors.index.json",
        "modeling_maira2.py",
        "processing_maira2.py",
    )
    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "weight_bytes": sum(EXPECTED_FILE_SIZES[path] for path in expected_weights),
        "small_file_sha256": {
            path: sha256_file(root / path) for path in audited_small_files
        },
    }


class FrozenMaira2Runtime:
    """Official non-grounded, current-frontal-only MAIRA-2 pathway."""

    def __init__(self, model_dir: str | Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; run MAIRA-2 only through Slurm")
        root = Path(model_dir).resolve(strict=True)
        self.audit = validate_model_snapshot(root)
        self._torch = torch
        transformers_logging.set_verbosity_error()
        torch.cuda.reset_peak_memory_stats()
        self.processor = AutoProcessor.from_pretrained(
            str(root),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(root),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        ).eval().to("cuda")
        self.model.requires_grad_(False)
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        if self.model.training:
            raise RuntimeError("MAIRA-2 entered training mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("MAIRA-2 parameters are not frozen")

    @property
    def peak_vram_gib(self) -> float:
        return self._torch.cuda.max_memory_allocated() / (1024**3)

    def generate(self, image_path: Path) -> GeneratedReport:
        from PIL import Image

        self._assert_frozen()
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB").copy()
        inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication=None,
            technique=None,
            comparison=None,
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        ).to("cuda")
        prompt_length = int(inputs["input_ids"].shape[-1])
        self._torch.cuda.synchronize()
        started = time.monotonic()
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        self._torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        decoded = self.processor.decode(
            output[0][prompt_length:], skip_special_tokens=True
        ).lstrip()
        prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(
            decoded
        )
        if not isinstance(prediction, str) or not prediction.strip():
            raise ValueError("MAIRA-2 returned an invalid non-grounded report")
        findings = prediction.strip()
        return GeneratedReport(
            canonical_text="FINDINGS:\n" + findings,
            findings=findings,
            impression=None,
            elapsed_seconds=elapsed,
            metadata={
                "strategy": "greedy",
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "grounding": False,
                "current_frontal_only": True,
                "indication_used": False,
                "prior_used": False,
                "dtype": "float32",
                "prompt_token_count": prompt_length,
            },
        )

