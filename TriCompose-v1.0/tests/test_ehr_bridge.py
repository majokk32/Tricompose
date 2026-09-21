from __future__ import annotations

import unittest

from tricompose_v1.ehr_bridge import (
    CANONICAL_SCHEMA,
    canonicalize_promptehr_case,
    canonicalize_synehrgy_case,
)
from tricompose_v1.facts import extract_ehr_facts


class EHRBridgeTests(unittest.TestCase):
    def test_synehrgy_builds_canonical_events_and_evidence_grounded_facts(self) -> None:
        source = {
            "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
            "case_id": "case_000",
            "structure": {
                "top_level_tokens": ["<1d-3d>"],
                "visits": [
                    {
                        "covariates": ["Gender_0.0"],
                        "problems": ["J189_Pneumonia, unspecified organism"],
                        "labs": ["WBC", "bin_20"],
                        "charts": [],
                        "other_tokens": [],
                    },
                    {
                        "covariates": [],
                        "problems": [
                            "J90_Pleural effusion, not elsewhere classified"
                        ],
                        "labs": ["pH", "bin_11"],
                        "charts": ["Cardiac pacemaker assessment"],
                        "other_tokens": [],
                    },
                ],
            },
            "validation": {"strict_valid": True},
        }
        result = canonicalize_synehrgy_case(
            source,
            source_model_id="synehrgy_qwen2_40bins",
        )
        self.assertEqual(result["schema_version"], CANONICAL_SCHEMA)
        self.assertEqual(result["validation"]["visit_count"], 2)
        self.assertEqual(result["timeline"][1]["relative_time_tokens"], ["<1d-3d>"])
        self.assertEqual(result["diagnoses"][1]["code"], "J90")
        facts = extract_ehr_facts(result)
        self.assertEqual(facts["facts"]["pleural_effusion"]["state"], "positive")
        self.assertEqual(
            facts["facts"]["pleural_effusion"]["source_fields"],
            ["diagnoses[1]"],
        )
        self.assertEqual(facts["facts"]["cardiac_pacemaker"]["state"], "positive")
        self.assertEqual(facts["facts"]["pneumonia"]["state"], "unknown")

    def test_invalid_synehrgy_candidate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            canonicalize_synehrgy_case(
                {
                    "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
                    "case_id": "case_001",
                    "structure": {"visits": []},
                    "validation": {"strict_valid": False},
                },
                source_model_id="synehrgy_qwen2_40bins",
            )

    def test_promptehr_can_be_normalized_but_remains_ineligible(self) -> None:
        result = canonicalize_promptehr_case(
            {
                "schema": "tricompose.promptehr.synthetic_ehr.v1",
                "case_id": "case_002",
                "event_codes": [
                    {
                        "diag": ["I50_Congestive heart failure"],
                        "prod": [],
                        "med": [],
                    }
                ],
                "validation": {"strict_valid": True},
            }
        )
        self.assertFalse(result["source"]["cold_start"])
        self.assertEqual(result["source"]["mimic_schema_family"], "MIMIC-III")
        self.assertFalse(result["validation"]["strict_v1_eligible"])
        facts = extract_ehr_facts(result)
        self.assertFalse(facts["summary"]["underconditioned"])
        self.assertEqual(
            facts["facts"]["congestive_heart_failure"]["state"], "positive"
        )
        self.assertEqual(facts["summary"]["clinical_indication_fact_count"], 1)


if __name__ == "__main__":
    unittest.main()
