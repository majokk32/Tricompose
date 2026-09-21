from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tricompose_llavarad_report.model import FIXED_PROMPT, validate_model_snapshot


class LlavaRadModelTests(unittest.TestCase):
    def test_prompt_matches_upstream_report_instruction(self) -> None:
        self.assertEqual(
            FIXED_PROMPT,
            "Given the chest X-ray image, describe the findings in the image:",
        )

    def test_snapshot_validation_checks_frozen_adapter_and_weight_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            base = root / "base"
            text = root / "text"
            source = root / "source"
            adapter.mkdir()
            base.mkdir()
            text.mkdir()
            source.mkdir()

            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "inference_mode": True,
                        "base_model_name_or_path": "lmsys/vicuna-7b-v1.5",
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llava",
                        "mm_vision_tower": "biomedclip_cxr_518",
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "adapter_model.bin",
                "biomedclipcxr_518.json",
                "biomedclipcxr_518_checkpoint.pt",
                "non_lora_trainables.bin",
            ):
                (adapter / name).write_bytes(b"x")

            (base / "config.json").write_text(
                json.dumps({"architectures": ["LlamaForCausalLM"]}),
                encoding="utf-8",
            )
            (base / "pytorch_model.bin.index.json").write_text(
                json.dumps({"weight_map": {"x": "pytorch_model.bin"}}),
                encoding="utf-8",
            )
            for name in (
                "generation_config.json",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ):
                (base / name).write_bytes(b"x")

            for name in ("config.json", "tokenizer_config.json", "vocab.txt"):
                (text / name).write_bytes(b"x")
            for relative in (
                "llava/constants.py",
                "llava/conversation.py",
                "llava/mm_utils.py",
                "llava/model/builder.py",
                "llava/utils.py",
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            model_sizes = {
                path.name: path.stat().st_size
                for path in adapter.iterdir()
                if path.is_file()
            }
            base_sizes = {
                path.name: path.stat().st_size
                for path in base.iterdir()
                if path.is_file()
            }
            with (
                patch(
                    "tricompose_llavarad_report.model.EXPECTED_MODEL_FILE_SIZES",
                    model_sizes,
                ),
                patch(
                    "tricompose_llavarad_report.model.EXPECTED_BASE_FILE_SIZES",
                    base_sizes,
                ),
            ):
                audit = validate_model_snapshot(
                    model_dir=adapter,
                    model_base=base,
                    biomedbert_dir=text,
                    external_root=source,
                )
            self.assertEqual(audit["adapter_weight_bytes"], 1)


if __name__ == "__main__":
    unittest.main()
