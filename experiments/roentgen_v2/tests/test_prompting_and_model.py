"""Synthetic-only tests; no protected or MIMIC artifact is opened."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tricompose_roentgen_v2.model import (
    EXPECTED_WEIGHT_SIZES,
    REQUIRED_CONFIG_FILES,
    validate_model_snapshot,
)
from tricompose_roentgen_v2.prompting import build_roentgen_prompt


class PromptingTest(unittest.TestCase):
    def test_positive_observable_facts_only(self) -> None:
        result = build_roentgen_prompt(
            {
                "age_group": "elderly adult",
                "sex": "female",
                "positive_diagnoses": ["pneumonia", "pleural effusion"],
                "positive_support_devices": [
                    "endotracheal intubation",
                    "mechanical ventilation",
                ],
                "measurement_bins": {
                    "white blood cell count": "high",
                    "oxygen saturation": "low",
                },
            }
        )
        self.assertTrue(result.text.startswith("Elderly adult female patient."))
        self.assertIn("PA chest radiograph", result.text)
        self.assertNotIn("Portable", result.text)
        self.assertNotIn(" AP ", result.text)
        self.assertIn("Findings:", result.text)
        self.assertIn("pneumonia", result.text)
        self.assertIn("is present", result.text)
        self.assertIn("Pleural effusion", result.text)
        self.assertIn("endotracheal tube", result.text)
        self.assertNotIn("concerning for", result.text)
        self.assertNotIn("white blood cell", result.text)
        self.assertNotIn("oxygen saturation", result.text)
        self.assertNotIn("mechanical ventilation", result.text)
        self.assertEqual(result.omitted_devices, ("mechanical ventilation",))

    def test_missing_positive_fact_does_not_invent_normality(self) -> None:
        result = build_roentgen_prompt(
            {
                "age_group": "adult",
                "sex": "unspecified-sex",
                "positive_diagnoses": [],
                "positive_support_devices": [],
            }
        )
        self.assertIn("clinical evaluation", result.text)
        self.assertNotIn("Findings:", result.text)
        self.assertNotIn("normal", result.text.lower())
        self.assertNotIn("no acute", result.text.lower())

    def test_chf_is_serialized_as_explicit_observable_findings(self) -> None:
        result = build_roentgen_prompt(
            {
                "age_group": "older adult",
                "sex": "male",
                "positive_diagnoses": ["congestive heart failure"],
                "positive_support_devices": [],
            }
        )
        self.assertIn("Cardiomegaly", result.text)
        self.assertIn("pulmonary vascular congestion", result.text)
        self.assertIn("interstitial edema", result.text)
        self.assertIn("congestive heart failure", result.text)

    def test_unknown_facts_are_not_serialized(self) -> None:
        result = build_roentgen_prompt(
            {
                "positive_diagnoses": ["elevated troponin"],
                "positive_support_devices": ["unknown device"],
            }
        )
        self.assertNotIn("troponin", result.text)
        self.assertNotIn("unknown device", result.text)


class SnapshotValidationTest(unittest.TestCase):
    def test_missing_snapshot_component_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                validate_model_snapshot(temp_dir)

    def test_audited_layout_and_sizes_are_accepted(self) -> None:
        # Sparse files avoid allocating 6.5 GB while exercising exact size checks.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in REQUIRED_CONFIG_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            for relative, size in EXPECTED_WEIGHT_SIZES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    handle.truncate(size)
            audit = validate_model_snapshot(root)
            self.assertEqual(audit["weight_bytes"], sum(EXPECTED_WEIGHT_SIZES.values()))


if __name__ == "__main__":
    unittest.main()
