"""Dependency-free tests for edge-specific EHR metrics."""

from __future__ import annotations

import unittest

from contracts import CHEXPERT_FINDINGS
from ehr_edge_metrics import (
    probability_metrics,
    score_direct_ehr_report,
    score_weak_priors,
    weak_clinical_priors,
)


def states(**overrides: str) -> dict[str, str]:
    payload = {finding: "unknown" for finding in CHEXPERT_FINDINGS}
    payload.update(overrides)
    return payload


def facts(*, chf: str = "unknown") -> dict:
    return {
        "direct_facts": {
            "congestive_heart_failure": {
                "state": chf,
                "source_fields": [] if chf == "unknown" else ["diagnoses[0]"],
            }
        }
    }


def canonical(*medications: str) -> dict:
    return {
        "timeline": [{"visit_index": 0}],
        "medications": [
            {
                "visit_index": 0,
                "code": "",
                "description": medication,
                "token": medication,
            }
            for medication in medications
        ],
    }


class EHREdgeMetricTests(unittest.TestCase):
    def test_missing_report_mention_is_not_a_contradiction(self) -> None:
        result = score_direct_ehr_report(
            states(pneumonia="positive"), states()
        )
        self.assertEqual(result["contradiction_count"], 0)
        self.assertEqual(result["comparable_explicit_fact_count"], 0)

    def test_no_finding_contradicts_direct_positive_pathology(self) -> None:
        result = score_direct_ehr_report(
            states(pneumonia="positive"), states(no_finding="positive")
        )
        self.assertEqual(result["contradiction_count"], 1)
        self.assertEqual(
            result["no_finding_cross_contradiction_findings"], ["pneumonia"]
        )

    def test_chf_denial_is_weak_not_hard(self) -> None:
        rules = weak_clinical_priors(facts(chf="positive"), canonical())
        result = score_weak_priors(
            rules, states(cardiomegaly="negative", edema="negative")
        )
        self.assertEqual(result["weak_incompatibility_count"], 1)
        self.assertEqual(result["weak_support_count"], 0)

    def test_loop_diuretic_adds_support_only_rule(self) -> None:
        rules = weak_clinical_priors(facts(), canonical("Furosemide 40 mg"))
        self.assertEqual([row["rule_id"] for row in rules], ["loop_diuretic_to_edema_weak_v1"])
        result = score_weak_priors(rules, states(edema="positive"))
        self.assertEqual(result["weak_support_count"], 1)

    def test_probability_metrics_reject_weak_priors_and_reports_single_class(self) -> None:
        result = probability_metrics(
            [
                {
                    "ehr_finding_states": states(pneumonia="positive"),
                    "cxr_finding_probabilities": {
                        finding: (0.8 if finding == "pneumonia" else None)
                        for finding in CHEXPERT_FINDINGS
                    },
                }
            ]
        )
        pneumonia = result["per_disease"]["pneumonia"]
        self.assertEqual(
            pneumonia["status"],
            "diagnostic_calibration_only_single_reference_class",
        )
        self.assertIsNone(pneumonia["auroc"])
        self.assertAlmostEqual(pneumonia["brier"], 0.04)


if __name__ == "__main__":
    unittest.main()
