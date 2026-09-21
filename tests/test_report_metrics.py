"""Unit tests for dependency-free report metrics."""

from __future__ import annotations

import unittest

from tricompose.scoring.report_metrics import (
    bleu_n,
    meteor,
    rouge_l,
    score_direction,
    tokenize,
)


class ReportMetricTests(unittest.TestCase):
    def test_tokenization_is_stable(self) -> None:
        self.assertEqual(
            tokenize("Left-base opacity; NO effusion."),
            ["left", "base", "opacity", "no", "effusion"],
        )

    def test_identical_reports(self) -> None:
        tokens = ["no", "acute", "cardiopulmonary", "process"]
        self.assertAlmostEqual(bleu_n(tokens, tokens, 1), 1.0)
        self.assertAlmostEqual(bleu_n(tokens, tokens, 2), 1.0)
        self.assertAlmostEqual(bleu_n(tokens, tokens, 3), 1.0)
        self.assertAlmostEqual(rouge_l(tokens, tokens), 1.0)
        self.assertGreater(meteor(tokens, tokens), 0.99)

    def test_disjoint_reports(self) -> None:
        left = ["left", "effusion"]
        right = ["right", "pneumothorax"]
        scores = score_direction(left, right)
        self.assertAlmostEqual(scores["rouge_l"], 0.0)
        self.assertAlmostEqual(scores["meteor"], 0.0)
        self.assertLess(scores["bleu_1"], 1e-8)


if __name__ == "__main__":
    unittest.main()
