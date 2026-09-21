from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tricompose_radedit.model import validate_model_bundle
from tricompose_radedit.prompting import build_radedit_prompt


class RadEditAdapterTests(unittest.TestCase):
    def test_prompt_uses_short_observation_style(self) -> None:
        result = build_radedit_prompt(
            {
                "age_group": "older adult",
                "sex": "female",
                "positive_diagnoses": ["pleural effusion", "pneumothorax"],
                "positive_support_devices": [],
            }
        )
        self.assertEqual(result.text, "Pneumothorax. Pleural effusion.")
        self.assertNotIn("older", result.text.lower())
        self.assertFalse(result.underconditioned)

    def test_empty_positive_set_does_not_claim_normal(self) -> None:
        result = build_radedit_prompt(
            {"positive_diagnoses": [], "positive_support_devices": []}
        )
        self.assertTrue(result.underconditioned)
        self.assertNotIn("normal", result.text.lower())
        self.assertNotIn("no acute", result.text.lower())

    def test_bundle_validation_checks_exact_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "radedit/unet/test.safetensors"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"abc")
            with patch(
                "tricompose_radedit.model.EXPECTED_FILE_SIZES",
                {relative: 3},
            ):
                audit = validate_model_bundle(root)
            self.assertEqual(audit["weight_bytes"], 3)


if __name__ == "__main__":
    unittest.main()

