from __future__ import annotations

import unittest

from tricompose.scored_weekly_report import build_score_section


class ScoredWeeklyReportTests(unittest.TestCase):
    def test_candidate_scores_are_kept_separate(self) -> None:
        report_bundle = {
            "schema_version": "tricompose.score_bundle.v1",
            "run_id": "smoke",
            "comparison_role": "peer",
            "directional": {
                "unidisc_to_llavarad": {
                    "bleu_1": 0.1,
                    "bleu_2": 0.2,
                    "bleu_3": 0.3,
                    "rouge_l": 0.4,
                    "meteor": 0.5,
                },
                "llavarad_to_unidisc": {
                    "bleu_1": 0.6,
                    "bleu_2": 0.7,
                    "bleu_3": 0.8,
                    "rouge_l": 0.9,
                    "meteor": 1.0,
                },
            },
        }
        qwen_bundle = {
            "schema_version": "tricompose.score_bundle.v1",
            "run_id": "smoke",
            "records": [
                {
                    "metric": "qwen25vl_cxr_report_match",
                    "candidate_ids": ["unidisc"],
                    "value": 0.25,
                },
                {
                    "metric": "qwen25vl_cxr_report_match",
                    "candidate_ids": ["llavarad"],
                    "value": 0.75,
                },
            ],
        }
        agent_bundle = {
            "schema_version": "tricompose.agent_decision.v1",
            "run_id": "smoke",
            "decision": {
                "action": "select",
                "status": "provisional",
                "reason_code": "qwen_score_and_margin_passed",
                "selected_candidate_id": "llavarad",
            },
        }

        section = build_score_section(
            report_bundle,
            qwen_bundle,
            agent_bundle,
        )

        self.assertIn(
            "| A — UniDisc | 0.10000000 | 0.20000000 | "
            "0.30000000 | 0.40000000 | 0.50000000 | 0.25000000 |",
            section,
        )
        self.assertIn(
            "| B — LLaVA-Rad | 0.60000000 | 0.70000000 | "
            "0.80000000 | 0.90000000 | 1.00000000 | 0.75000000 |",
            section,
        )
        self.assertIn("B — LLaVA-Rad", section)
        self.assertIn("`select`", section)


if __name__ == "__main__":
    unittest.main()
