from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tricompose_chexgenbench_sana.model import validate_model_snapshot
from tricompose_chexgenbench_sana.prompting import build_sana_prompt


class SanaAdapterTests(unittest.TestCase):
    def test_prompt_reuses_clinical_serialization(self) -> None:
        result = build_sana_prompt(
            {
                "age_group": "older adult",
                "sex": "female",
                "positive_diagnoses": ["pleural effusion"],
                "positive_support_devices": [],
            }
        )
        self.assertIn("PA chest radiograph", result.text)
        self.assertIn("Pleural effusion is present", result.text)
        self.assertEqual(result.included_diagnoses, ("pleural effusion",))

    def test_snapshot_validation_checks_pipeline_and_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = (
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors.index.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer.model",
                "tokenizer/tokenizer_config.json",
                "transformer/config.json",
                "vae/config.json",
            )
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            model_index = root / "model_index.json"
            model_index.write_text(
                json.dumps(
                    {
                        "_class_name": "SanaPipeline",
                        "text_encoder": ["transformers", "Gemma2Model"],
                    }
                ),
                encoding="utf-8",
            )
            weight = root / "transformer" / "weights.safetensors"
            weight.write_bytes(b"abc")
            expected_hash = hashlib.sha256(model_index.read_bytes()).hexdigest()
            with (
                patch(
                    "tricompose_chexgenbench_sana.model.EXPECTED_WEIGHT_SIZES",
                    {"transformer/weights.safetensors": 3},
                ),
                patch(
                    "tricompose_chexgenbench_sana.model.EXPECTED_MODEL_INDEX_SHA256",
                    expected_hash,
                ),
            ):
                audit = validate_model_snapshot(root)
            self.assertEqual(audit["weight_bytes"], 3)


if __name__ == "__main__":
    unittest.main()
