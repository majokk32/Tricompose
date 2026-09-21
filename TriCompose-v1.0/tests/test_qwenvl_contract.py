from __future__ import annotations

import unittest

from tools.score_tricompose_v1_qwenvl import (
    JUDGE_PROMPT,
    PROMPT_VERSION,
    _parse_response,
)


class QwenVLContractTests(unittest.TestCase):
    def test_prompt_example_has_exact_fixed_width_vectors(self) -> None:
        self.assertIn("fixed_width", PROMPT_VERSION)
        self.assertIn('"image":"??????????????"', JUDGE_PROMPT)
        self.assertIn('"report":"??????????????"', JUDGE_PROMPT)

    def test_fixed_width_response_parses(self) -> None:
        payload = _parse_response(
            '{"score":0.75,"image":"+?????????????",'
            '"report":"+?????????????","strong":[]}'
        )
        self.assertEqual(payload["qwen_match_score"], 0.75)
        self.assertEqual(payload["image_finding_states"]["atelectasis"], "positive")
        self.assertEqual(payload["report_finding_states"]["atelectasis"], "positive")

    def test_short_vectors_preserve_scalar_but_not_finding_evidence(self) -> None:
        payload = _parse_response(
            '{"score":0.5,"image":"?","report":"?","strong":[0]}'
        )
        self.assertEqual(payload["qwen_match_score"], 0.5)
        self.assertEqual(
            payload["finding_contract_status"], "unavailable_invalid_width"
        )
        self.assertEqual(
            payload["finding_comparison"]["strong_contradiction_count"], 0
        )
        self.assertTrue(
            all(
                state == "unknown"
                for state in payload["image_finding_states"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
