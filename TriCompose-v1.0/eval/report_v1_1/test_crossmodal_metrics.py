"""Dependency-free tests for unknown-safe cross-modal metrics."""

from __future__ import annotations

import unittest

from contracts import CHEXPERT_FINDINGS
from crossmodal_metrics import (
    pairwise_state_disagreement,
    relation,
    score_state_pair,
    summarize_state_pairs,
)


def states(**overrides: str) -> dict[str, str]:
    payload = {finding: "unknown" for finding in CHEXPERT_FINDINGS}
    payload.update(overrides)
    return payload


class CrossModalMetricTests(unittest.TestCase):
    def test_unknown_is_not_negative_or_contradiction(self) -> None:
        self.assertEqual(relation("positive", "unknown"), "unknown")
        summary = summarize_state_pairs(
            [{"left": states(edema="positive"), "right": states()}],
            reference_key="left",
            candidate_key="right",
        )
        self.assertEqual(summary["known_reference_fact_count"], 1)
        self.assertEqual(summary["comparable_explicit_fact_count"], 0)
        self.assertEqual(summary["explicit_contradiction_count"], 0)
        self.assertEqual(summary["known_fact_coverage"], 0.0)

    def test_explicit_opposites_are_contradictions(self) -> None:
        summary = summarize_state_pairs(
            [
                {
                    "left": states(edema="positive", pneumothorax="negative"),
                    "right": states(edema="negative", pneumothorax="positive"),
                }
            ],
            reference_key="left",
            candidate_key="right",
        )
        self.assertEqual(summary["explicit_contradiction_count"], 2)
        self.assertEqual(summary["explicit_contradiction_rate_over_known_facts"], 1.0)

    def test_support_and_unverifiable_are_separate(self) -> None:
        summary = summarize_state_pairs(
            [
                {
                    "left": states(edema="positive"),
                    "right": states(edema="positive", fracture="positive"),
                }
            ],
            reference_key="left",
            candidate_key="right",
        )
        self.assertEqual(summary["support_count"], 1)
        self.assertEqual(summary["explicit_contradiction_count"], 0)
        self.assertEqual(summary["candidate_positive_reference_unknown_count"], 1)

    def test_candidate_edge_score_penalizes_only_explicit_opposites(self) -> None:
        score = score_state_pair(
            states(edema="positive", pneumothorax="negative", fracture="positive"),
            states(edema="positive", pneumothorax="positive"),
        )
        self.assertEqual(score["known_reference_fact_count"], 3)
        self.assertEqual(score["comparable_explicit_fact_count"], 2)
        self.assertAlmostEqual(score["support_recall"], 1 / 3, places=7)
        self.assertAlmostEqual(score["contradiction_rate"], 1 / 3, places=7)
        self.assertEqual(score["edge_score"], 0.0)

    def test_disagreement_ignores_unknown_pairs(self) -> None:
        disagreement = pairwise_state_disagreement(
            [
                states(edema="positive", fracture="unknown"),
                states(edema="negative", fracture="positive"),
                states(edema="positive", fracture="unknown"),
            ]
        )
        self.assertEqual(disagreement["candidate_pair_count"], 3)
        self.assertEqual(disagreement["comparable_fact_pair_count"], 3)
        self.assertEqual(disagreement["disagreement_count"], 2)
        self.assertAlmostEqual(disagreement["disagreement_rate"], 2 / 3, places=7)


if __name__ == "__main__":
    unittest.main()
