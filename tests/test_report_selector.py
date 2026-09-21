from __future__ import annotations

import unittest

from tricompose.agent.report_selector import (
    _extract_peer_scores,
    choose_action,
)


class ReportSelectorTests(unittest.TestCase):
    def test_selects_when_score_and_margin_pass(self) -> None:
        decision = choose_action(
            {"unidisc": 0.61, "llavarad": 0.82},
            minimum_score=0.55,
            minimum_margin=0.10,
            suggested_next_verifier="independent_verifier",
        )

        self.assertEqual(decision["action"], "select")
        self.assertEqual(decision["selected_candidate_id"], "llavarad")
        self.assertEqual(decision["status"], "provisional")
        self.assertAlmostEqual(decision["margin"], 0.21)

    def test_requests_verification_when_margin_is_small(self) -> None:
        decision = choose_action(
            {"unidisc": 0.80, "llavarad": 0.84},
            minimum_score=0.55,
            minimum_margin=0.10,
            suggested_next_verifier="independent_verifier",
        )

        self.assertEqual(decision["action"], "verify_more")
        self.assertEqual(
            decision["reason_code"],
            "candidate_margin_below_threshold",
        )
        self.assertNotIn("selected_candidate_id", decision)

    def test_requests_verification_when_top_score_is_low(self) -> None:
        decision = choose_action(
            {"unidisc": 0.20, "llavarad": 0.40},
            minimum_score=0.55,
            minimum_margin=0.10,
            suggested_next_verifier="independent_verifier",
        )

        self.assertEqual(decision["action"], "verify_more")
        self.assertEqual(
            decision["reason_code"],
            "top_score_below_threshold",
        )

    def test_tie_is_explicit_and_does_not_select(self) -> None:
        decision = choose_action(
            {"unidisc": 0.0, "llavarad": 0.0},
            minimum_score=0.55,
            minimum_margin=0.10,
            suggested_next_verifier="independent_verifier",
        )

        self.assertEqual(decision["action"], "verify_more")
        self.assertEqual(
            decision["tied_top_candidate_ids"],
            ["llavarad", "unidisc"],
        )
        self.assertNotIn("selected_candidate_id", decision)

    def test_peer_scores_require_matching_candidate_hashes(self) -> None:
        bundle = {
            "schema_version": "tricompose.score_bundle.v1",
            "comparison_role": "peer",
            "candidates": {
                "unidisc": {"sha256": "a" * 64},
                "llavarad": {"sha256": "b" * 64},
            },
            "records": [
                {
                    "metric": "report_pair_bleu_1",
                    "scope": "report_pair",
                    "candidate_ids": ["unidisc", "llavarad"],
                    "value": 0.25,
                }
            ],
        }

        scores = _extract_peer_scores(
            bundle,
            candidate_ids={"unidisc", "llavarad"},
            candidate_hashes={
                "unidisc": "a" * 64,
                "llavarad": "b" * 64,
            },
        )

        self.assertEqual(scores, {"report_pair_bleu_1": 0.25})


if __name__ == "__main__":
    unittest.main()
