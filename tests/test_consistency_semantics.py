from __future__ import annotations

import unittest

from tricompose.scoring.consistency_semantics import summarize_fact_judgments


class ConsistencySemanticsTests(unittest.TestCase):
    def test_unknown_is_not_counted_as_contradiction(self) -> None:
        summary = summarize_fact_judgments(
            [
                {"relation": "support", "severity": "none"},
                {"relation": "unknown", "severity": "none"},
            ]
        )
        self.assertEqual(summary["unknown_count"], 1)
        self.assertEqual(summary["contradiction_count"], 0)
        self.assertEqual(summary["support_rate_among_comparable"], 1.0)

    def test_strong_contradiction_fails_hard_gate(self) -> None:
        summary = summarize_fact_judgments(
            [
                {"relation": "support", "severity": "none"},
                {"relation": "contradiction", "severity": "strong"},
            ]
        )
        self.assertFalse(summary["hard_gate_pass"])
        self.assertEqual(summary["contradiction_rate_among_comparable"], 0.5)


if __name__ == "__main__":
    unittest.main()

