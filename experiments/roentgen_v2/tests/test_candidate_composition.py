"""Synthetic-only candidate composition tests; no protected data is opened."""

from __future__ import annotations

import unittest

from tricompose_roentgen_v2.candidate_selector import (
    choose_candidate,
    hard_quality_failure,
    positive_ehr_support,
)
from tricompose_roentgen_v2.candidate_sweep import build_candidate_plan


class CandidatePlanTest(unittest.TestCase):
    def test_cfg_by_seed_grid_is_deterministic(self) -> None:
        plans = build_candidate_plan(100, [3.0, 4.0], 4)
        self.assertEqual(len(plans), 8)
        self.assertEqual(len({plan.candidate_id for plan in plans}), 8)
        self.assertEqual([plan.seed for plan in plans[:4]], [100, 101, 102, 103])
        self.assertEqual([plan.seed for plan in plans[4:]], [100, 101, 102, 103])
        self.assertEqual(plans[0].candidate_id, "roentgen_v2_cfg3p0_seed00")
        self.assertEqual(plans[4].candidate_id, "roentgen_v2_cfg4p0_seed00")

    def test_duplicate_guidance_scales_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_candidate_plan(100, [3.0, 3.0], 4)


class CandidateSelectorTest(unittest.TestCase):
    @staticmethod
    def _row(candidate_id: str, support: float, hard: bool) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "ehr_support": support,
            "hard_quality_failure": hard,
        }

    def test_selector_prefers_non_hard_candidate(self) -> None:
        selected = choose_candidate(
            [
                self._row("high_but_hard", 0.95, True),
                self._row("lower_but_valid", 0.60, False),
            ]
        )
        self.assertEqual(selected["candidate_id"], "lower_but_valid")

    def test_selector_maximizes_positive_ehr_support(self) -> None:
        selected = choose_candidate(
            [
                self._row("lower", 0.40, False),
                self._row("higher", 0.70, False),
            ]
        )
        self.assertEqual(selected["candidate_id"], "higher")

    def test_positive_support_does_not_create_negative_targets(self) -> None:
        support = positive_ehr_support(
            {"heart failure": ["edema", "cardiomegaly"]},
            {"edema": 0.2, "cardiomegaly": 0.8, "pneumothorax": 0.9},
            operating_point=0.5,
        )
        self.assertEqual(support["fact_count"], 1)
        self.assertEqual(support["mean_support"], 0.8)
        self.assertEqual(support["support_rate"], 1.0)

    def test_hard_quality_gate(self) -> None:
        good = {
            "intensity_std": 0.2,
            "clipped_fraction": 0.1,
            "entropy_bits": 6.0,
        }
        bad = dict(good, entropy_bits=2.0)
        self.assertFalse(hard_quality_failure(good))
        self.assertTrue(hard_quality_failure(bad))


if __name__ == "__main__":
    unittest.main()
