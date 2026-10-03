"""Synthetic metadata tests; no real source data or model inference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "real_validation"
sys.path.insert(0, str(ROOT))

from xrv_real_findings import MAPPING, summarize  # noqa: E402


class XRVRealFindingTests(unittest.TestCase):
    def test_unknown_and_uncertain_never_become_negative(self) -> None:
        records = []
        for split in ("val", "test"):
            for score, state in ((0.9, "positive"), (0.1, "negative"),
                                 (0.8, "unknown"), (0.2, "uncertain")):
                records.append({"split": split,
                                "scores": {name: score for name in MAPPING},
                                "reference_states": {name: state for name in MAPPING}})
        summary = summarize(records, min_per_class=1)
        for split in ("val", "test"):
            row = summary[split]["findings"]["Edema"]
            self.assertEqual((row["positive"], row["negative"]), (1, 1))
            self.assertEqual(row["excluded_unknown_or_uncertain"], 2)
            self.assertEqual(row["auroc"], 1.0)

    def test_insufficient_support_disables_metric(self) -> None:
        records = [{"split": split,
                    "scores": {name: 0.3 for name in MAPPING},
                    "reference_states": {name: "positive" for name in MAPPING}}
                   for split in ("val", "test")]
        summary = summarize(records, min_per_class=1)
        self.assertIsNone(summary["val"]["findings"]["Pneumonia"]["auroc"])
        self.assertEqual(summary["val"]["eligible_findings"], 0)


if __name__ == "__main__":
    unittest.main()
