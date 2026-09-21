from __future__ import annotations

import unittest

from tricompose_promptehr.evaluation import evaluate_cases


def _case(diag: list[int], prod: list[int], med: list[int], novel: int) -> dict:
    return {
        "event_indices": [{"diag": diag, "prod": prod, "med": med}],
        "validation": {"strict_valid": True},
        "prompt_comparison": {
            "diag": {
                "input_count": len(diag),
                "output_count": len(diag),
                "overlap_count": len(diag) - novel,
                "novel_output_count": novel,
            },
            "prod": {
                "input_count": len(prod),
                "output_count": len(prod),
                "overlap_count": len(prod),
                "novel_output_count": 0,
            },
            "med": {
                "input_count": len(med),
                "output_count": len(med),
                "overlap_count": len(med),
                "novel_output_count": 0,
            },
        },
    }


class EvaluationTests(unittest.TestCase):
    def test_diversity_and_prompt_novelty(self) -> None:
        result = evaluate_cases(
            [_case([1, 2], [3], [4], novel=1), _case([1, 5], [3], [4], novel=1)]
        )
        self.assertEqual(result["strict_valid_rate"], 1.0)
        self.assertEqual(result["unique_sequence_rate"], 1.0)
        self.assertGreater(result["novel_output_fraction"], 0.0)
        self.assertFalse(result["fidelity_claimed"])


if __name__ == "__main__":
    unittest.main()

