from __future__ import annotations

import unittest

from tricompose_synehrgy_v2.evaluation import evaluate_synthetic_cases
from tricompose_synehrgy_v2.structure import parse_token_sequence


class StructureTests(unittest.TestCase):
    def test_valid_single_visit_sequence(self) -> None:
        parsed = parse_token_sequence(
            [
                "<s>",
                "<v>",
                "<covars>",
                "Gender_1",
                "</covars>",
                "<problems>",
                "I50",
                "</problems>",
                "<labs>",
                "WBC",
                "bin_3",
                "</labs>",
                "</v>",
                "</s>",
            ]
        )
        self.assertTrue(parsed.validation["strict_valid"])
        self.assertEqual(parsed.validation["visit_count"], 1)
        self.assertEqual(parsed.structure["visits"][0]["problems"], ["I50"])

    def test_missing_eos_is_invalid(self) -> None:
        parsed = parse_token_sequence(["<s>", "<v>", "</v>"])
        self.assertFalse(parsed.validation["strict_valid"])
        self.assertFalse(parsed.validation["ended_with_eos"])

    def test_mismatched_section_is_invalid(self) -> None:
        parsed = parse_token_sequence(
            ["<s>", "<v>", "<labs>", "WBC", "</charts>", "</v>", "</s>"]
        )
        self.assertFalse(parsed.validation["strict_valid"])
        self.assertIn("section_close_mismatch", parsed.validation["parser_errors"])

    def test_reference_free_evaluation_reports_diversity(self) -> None:
        tokens = ["<s>", "<v>", "<problems>", "I50", "</problems>", "</v>", "</s>"]
        parsed = parse_token_sequence(tokens)
        result = evaluate_synthetic_cases(
            [
                {
                    "tokens": tokens,
                    "structure": parsed.structure,
                    "validation": parsed.validation,
                },
                {
                    "tokens": tokens + ["extra"],
                    "structure": parsed.structure,
                    "validation": parsed.validation,
                },
            ]
        )
        self.assertFalse(result["fidelity_claimed"])
        self.assertEqual(result["unique_sequence_rate"], 1.0)
        self.assertEqual(
            result["problem_ngram_diversity"]["1"]["unique_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
