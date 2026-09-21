"""Synthetic-only tests for the shared EHR-to-radiology text bridge."""

from __future__ import annotations

import unittest

from tricompose.ehr_prompt_cxr.prompting import (
    LIQUID_GENERATION_SUFFIX,
    build_model_prompt,
)
from tricompose_roentgen_v2.prompting import build_roentgen_prompt


SYNTHETIC_FACTS = {
    "age_group": "older adult",
    "sex": "female",
    "positive_diagnoses": ["pneumonia", "pleural effusion"],
    "positive_support_devices": [
        "endotracheal intubation",
        "mechanical ventilation",
    ],
    "measurement_bins": {"oxygen saturation": "low"},
}


class PromptingTest(unittest.TestCase):
    def test_models_share_exact_clinical_core(self) -> None:
        expected = build_roentgen_prompt(SYNTHETIC_FACTS).text
        unidisc = build_model_prompt("unidisc", SYNTHETIC_FACTS)
        liquid = build_model_prompt("liquid", SYNTHETIC_FACTS)
        self.assertEqual(unidisc.clinical_text, expected)
        self.assertEqual(liquid.clinical_text, expected)
        self.assertIn(expected, unidisc.text)
        self.assertTrue(unidisc.text.endswith(" <image>"))
        self.assertEqual(liquid.text, expected + LIQUID_GENERATION_SUFFIX)

    def test_non_radiographic_ehr_fields_are_omitted(self) -> None:
        for model_name in ("unidisc", "liquid"):
            prompt = build_model_prompt(model_name, SYNTHETIC_FACTS)
            self.assertNotIn("oxygen saturation", prompt.text.lower())
            self.assertNotIn("mechanical ventilation", prompt.text.lower())
            self.assertIn("endotracheal tube", prompt.text.lower())

    def test_unknown_model_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_model_prompt("unknown", SYNTHETIC_FACTS)


if __name__ == "__main__":
    unittest.main()
