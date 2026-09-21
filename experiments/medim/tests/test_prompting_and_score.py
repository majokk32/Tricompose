"""Synthetic-only unit tests; these tests never open MIMIC artifacts."""

from __future__ import annotations

import unittest

import numpy as np

from medim_ehr_prompt.consistency import _ehr_support
from medim_ehr_prompt.medim_adapter import (
    effective_image_sampling_steps,
    validate_checkpoint_compatibility,
)
from medim_ehr_prompt.official_image_metrics import _inception_score_from_probabilities
from medim_ehr_prompt.prompting import serialize_ehr_prompt


class PromptingTest(unittest.TestCase):
    def test_positive_only_prompt_and_observable_targets(self) -> None:
        row = {
            "AGE": 74,
            "GENDER": "F",
            "DX_PNEUMONIA": 1,
            "DX_PNEUMOTHORAX": 0,
            "DX_CHF": 0,
            "DX_PLEURAL_EFFUSION": 1,
            "DX_ATELECTASIS": None,
            "INTUBATED": 1,
            "VENTILATOR": 0,
            "PACEMAKER": 0,
            "WBC": 14.0,
            "BNP": None,
            "SPO2": 91,
            "RESP_RATE": 24,
        }
        facts, prompt = serialize_ehr_prompt(row)
        self.assertTrue(prompt.startswith("The image is a radiograph of the chest"))
        self.assertIn("Portable frontal AP chest radiograph", prompt)
        self.assertIn("Clinical history: an elderly adult female patient", prompt)
        self.assertIn("pneumonia", prompt)
        self.assertIn("pleural effusion", prompt)
        self.assertIn("endotracheal intubation", prompt)
        self.assertIn("white blood cell count is high", prompt)
        self.assertIn("oxygen saturation is low", prompt)
        self.assertIn("respiratory rate is elevated", prompt)
        self.assertNotIn("pneumothorax", prompt)
        self.assertNotIn("mechanical ventilation", prompt)
        self.assertEqual(
            set(facts["cxr_observable_targets"]),
            {"pneumonia", "pleural effusion"},
        )

    def test_no_positive_observable_fact_is_not_applicable(self) -> None:
        score = _ehr_support({}, {"pneumonia": 0.9}, operating_point=0.5)
        self.assertEqual(score["status"], "not_applicable")
        self.assertIsNone(score["ehr_support_score"])

    def test_support_uses_best_compatible_cxr_label(self) -> None:
        score = _ehr_support(
            {
                "pneumonia": ["pneumonia", "consolidation", "lung_opacity"],
                "pleural effusion": ["effusion"],
            },
            {
                "pneumonia": 0.2,
                "consolidation": 0.8,
                "lung_opacity": 0.6,
                "effusion": 0.4,
            },
            operating_point=0.5,
        )
        self.assertEqual(score["status"], "scored")
        self.assertAlmostEqual(score["ehr_support_score"], 0.6)
        self.assertAlmostEqual(score["support_rate"], 0.5)

    def test_only_tied_lm_head_may_be_missing(self) -> None:
        result = validate_checkpoint_compatibility(["lm_head.weight"], [])
        self.assertEqual(result["allowed_missing_key_count"], 1)
        with self.assertRaises(RuntimeError):
            validate_checkpoint_compatibility(["model.layers.0.mlp.up_proj.weight"], [])
        with self.assertRaises(RuntimeError):
            validate_checkpoint_compatibility([], ["unexpected.weight"])

    def test_official_inception_score_formula(self) -> None:
        alternating_classes = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        mean, std = _inception_score_from_probabilities(
            alternating_classes,
            splits=2,
        )
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, 0.0)

        uniform = np.full((4, 2), 0.5)
        mean, std = _inception_score_from_probabilities(uniform, splits=2)
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(std, 0.0)

    def test_official_sampling_clamps_to_masked_image_tokens(self) -> None:
        self.assertEqual(effective_image_sampling_steps(100, 1024), 100)
        self.assertEqual(effective_image_sampling_steps(1280, 1024), 1024)
        with self.assertRaises(ValueError):
            effective_image_sampling_steps(0, 1024)


if __name__ == "__main__":
    unittest.main()
