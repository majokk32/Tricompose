from __future__ import annotations

import unittest

from medim_ehr_prompt.official_paired_t2i import (
    OFFICIAL_CHEST_PREFIX,
    build_official_style_ehr_prompt,
    strip_single_official_prefix,
)


class OfficialPairedT2ITest(unittest.TestCase):
    def test_strip_single_official_prefix(self) -> None:
        prompt = OFFICIAL_CHEST_PREFIX + " This is a synthetic continuation."
        self.assertEqual(
            strip_single_official_prefix(prompt),
            "This is a synthetic continuation.",
        )

    def test_strip_single_official_prefix_rejects_missing_prefix(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "lacks the expected official prefix"
        ):
            strip_single_official_prefix("This is a synthetic continuation.")

    def test_strip_single_official_prefix_rejects_duplicate_prefix(self) -> None:
        prompt = f"{OFFICIAL_CHEST_PREFIX} {OFFICIAL_CHEST_PREFIX} continuation"
        with self.assertRaisesRegex(ValueError, "more than one prefix"):
            strip_single_official_prefix(prompt)

    def test_official_style_ehr_prompt_omits_labs_and_vitals(self) -> None:
        prompt = build_official_style_ehr_prompt(
            {
                "age_group": "elderly adult",
                "sex": "female",
                "positive_diagnoses": ["pneumonia", "pleural effusion"],
                "positive_support_devices": ["endotracheal intubation"],
                "measurement_bins": {
                    "white blood cell count": "high",
                    "oxygen saturation": "low",
                },
            }
        )
        self.assertTrue(prompt.startswith("Portable frontal AP chest radiograph"))
        self.assertIn("Clinical indication: evaluation for pneumonia", prompt)
        self.assertIn("Support devices: endotracheal intubation", prompt)
        self.assertNotIn("white blood cell count", prompt)
        self.assertNotIn("oxygen saturation", prompt)


if __name__ == "__main__":
    unittest.main()
