from __future__ import annotations

import unittest

from tricompose_roentgen_v2.evaluation import (
    _ehr_support,
    age_distance_to_group,
    age_group_match,
    pseudo_label_auroc,
    robust_quality_flags,
)


class EvaluationTest(unittest.TestCase):
    def test_age_group_semantics(self) -> None:
        self.assertTrue(age_group_match("young adult", 29.9))
        self.assertFalse(age_group_match("young adult", 30.0))
        self.assertTrue(age_group_match("elderly adult", 84.0))
        self.assertIsNone(age_group_match("adult", 50.0))
        self.assertEqual(age_distance_to_group("older adult", 46.0), 4.0)
        self.assertEqual(age_distance_to_group("older adult", 60.0), 0.0)

    def test_ehr_absence_is_not_negative(self) -> None:
        empty = _ehr_support({}, {"edema": 0.9}, operating_point=0.5)
        self.assertEqual(empty["status"], "not_applicable")
        scored = _ehr_support(
            {"heart failure": ["edema", "cardiomegaly"]},
            {"edema": 0.2, "cardiomegaly": 0.8},
            operating_point=0.5,
        )
        self.assertEqual(scored["mean_support"], 0.8)
        self.assertEqual(scored["support_rate"], 1.0)

    def test_pseudo_label_auroc_requires_both_classes(self) -> None:
        self.assertEqual(
            pseudo_label_auroc(
                [0.1, 0.2, 0.8, 0.9],
                [0.2, 0.3, 0.7, 0.8],
                operating_point=0.5,
            ),
            1.0,
        )
        self.assertIsNone(
            pseudo_label_auroc(
                [0.8, 0.9], [0.1, 0.2], operating_point=0.5
            )
        )

    def test_quality_gate_uses_real_reference(self) -> None:
        real = [
            {
                "intensity_std": 0.20 + index * 0.001,
                "clipped_fraction": 0.10 + index * 0.001,
                "entropy_bits": 6.0 + index * 0.01,
                "gradient_p95": 0.30 + index * 0.001,
                "laplacian_variance": 0.02 + index * 0.0001,
                "high_frequency_ratio": 0.10 + index * 0.001,
            }
            for index in range(10)
        ]
        generated = [dict(row) for row in real]
        generated[1]["gradient_p95"] = 2.0
        generated[1]["high_frequency_ratio"] = 0.9
        flags = robust_quality_flags(
            real,
            generated,
            z_threshold=3.5,
            extreme_z_threshold=6.0,
        )
        self.assertFalse(flags[0]["flagged"])
        self.assertTrue(flags[1]["flagged"])


if __name__ == "__main__":
    unittest.main()
