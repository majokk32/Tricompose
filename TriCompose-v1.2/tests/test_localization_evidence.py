"""Pure, invented-state tests for the conservative evidence audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from audit_localization_evidence import triad_pattern  # noqa: E402


class LocalizationEvidenceTests(unittest.TestCase):
    def test_direct_asymmetry_is_not_generic_pair_mismatch(self) -> None:
        ehr = {"edema": "positive"}
        image = {"edema": "positive"}
        report = {"edema": "negative"}
        row = triad_pattern(ehr, image, report)
        self.assertEqual(row["pattern"], "report_candidate_signal")
        self.assertEqual(row["three_way_comparable_fact_count"], 1)
        self.assertEqual(triad_pattern(ehr, report, image)["pattern"],
                         "cxr_candidate_signal")

    def test_unknown_and_uncertain_force_abstention(self) -> None:
        self.assertEqual(triad_pattern({"edema": "unknown"},
                                       {"edema": "positive"},
                                       {"edema": "negative"})["pattern"],
                         "abstain_no_direct_ehr_fact")
        self.assertEqual(triad_pattern({"edema": "positive"},
                                       {"edema": "uncertain"},
                                       {"edema": "negative"})["pattern"],
                         "abstain_no_three_way_comparable_fact")

    def test_two_modalities_opposing_ehr_is_not_localized(self) -> None:
        row = triad_pattern({"edema": "positive"},
                            {"edema": "negative"},
                            {"edema": "negative"})
        self.assertEqual(row["pattern"], "abstain_both_modalities_oppose_ehr")
        self.assertEqual(row["both_opposed"], 1)


if __name__ == "__main__":
    unittest.main()
