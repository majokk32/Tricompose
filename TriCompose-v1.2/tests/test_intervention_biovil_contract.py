"""Synthetic-only tests for blinded scorer artifact binding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from score_intervention_biovil import validate_artifacts  # noqa: E402


class InterventionBioViLContractTests(unittest.TestCase):
    def test_resolves_only_requested_artifacts(self) -> None:
        resolver = [{"displayed_cxr_candidate_id": "cxr_a",
                     "displayed_cxr_sha256": "a" * 64,
                     "displayed_report_candidate_id": "report_b",
                     "displayed_report_sha256": "b" * 64}]
        images = {"cxr_a": {"artifact": {"sha256": "a" * 64}},
                  "cxr_unused": {"artifact": {"sha256": "c" * 64}}}
        reports = {"report_b": {"artifact": {"sha256": "b" * 64}}}
        self.assertEqual(validate_artifacts(resolver, images, reports),
                         (["cxr_a"], ["report_b"]))

    def test_rejects_missing_or_changed_artifact(self) -> None:
        resolver = [{"displayed_cxr_candidate_id": "cxr_a",
                     "displayed_cxr_sha256": "a" * 64,
                     "displayed_report_candidate_id": "report_b",
                     "displayed_report_sha256": "b" * 64}]
        images = {"cxr_a": {"artifact": {"sha256": "a" * 64}}}
        with self.assertRaisesRegex(ValueError, "absent"):
            validate_artifacts(resolver, images, {})
        reports = {"report_b": {"artifact": {"sha256": "c" * 64}}}
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_artifacts(resolver, images, reports)


if __name__ == "__main__":
    unittest.main()
