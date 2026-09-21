from __future__ import annotations

import unittest

from tricompose_v1.scoring import (
    CHEXPERT_FINDINGS,
    compare_finding_states,
    deterministic_report_quality,
    ehr_prompt_intent_states,
    extract_report_finding_states,
    validate_qwen_states,
    xrv_ehr_support,
)


def facts(**states: str):
    fact_ids = (
        "congestive_heart_failure",
        "chest_pain",
        "cardiomegaly",
        "pleural_effusion",
        "pulmonary_edema",
        "pneumonia",
        "pneumothorax",
        "atelectasis",
        "consolidation",
        "lung_opacity",
        "endotracheal_tube",
        "central_venous_catheter",
        "enteric_tube",
        "cardiac_pacemaker",
    )
    categories = {
        "congestive_heart_failure": "clinical_indication",
        "chest_pain": "clinical_indication",
        "endotracheal_tube": "explicit_device",
        "central_venous_catheter": "explicit_device",
        "enteric_tube": "explicit_device",
        "cardiac_pacemaker": "explicit_device",
    }
    rows = {}
    for fact_id in fact_ids:
        state = states.get(fact_id, "unknown")
        rows[fact_id] = {
            "category": categories.get(fact_id, "direct_radiographic_finding"),
            "state": state,
            "evidence": [] if state == "unknown" else [f"diagnosis:{fact_id}"],
            "source_fields": [] if state == "unknown" else ["diagnoses[0]"],
        }
    return {
        "schema_version": "tricompose-facts-v1",
        "case_id": "case_000",
        "extractor_version": "radiology_facts_v2_legacy_bridge",
        "fact_scope": "latest_complete_synthetic_visit_only",
        "missing_is_unknown": True,
        "patient_context": {
            "age_group": "adult",
            "sex": "unspecified-sex",
            "source_fields": ["demographics.age_group", "demographics.sex"],
            "mapping_status": "test",
        },
        "facts": rows,
        "summary": {"radiology_relevant_fact_count": len(states), "underconditioned": False},
    }


class ScoringTests(unittest.TestCase):
    def test_chf_intent_matches_exact_legacy_prompt_semantics(self):
        states = ehr_prompt_intent_states(
            facts(congestive_heart_failure="positive")
        )
        self.assertEqual(states["cardiomegaly"], "positive")
        self.assertEqual(states["edema"], "positive")
        self.assertEqual(states["pleural_effusion"], "unknown")

    def test_negative_ehr_fact_not_rendered_does_not_become_cxr_target(self):
        states = ehr_prompt_intent_states(facts(pneumothorax="negative"))
        self.assertEqual(states["pneumothorax"], "unknown")

    def test_unknown_observation_is_not_contradiction(self):
        expected = {label: "unknown" for label in CHEXPERT_FINDINGS}
        observed = dict(expected)
        expected["pneumothorax"] = "positive"
        result = compare_finding_states(expected, observed)
        self.assertIsNone(result["score"])
        self.assertEqual(result["strong_contradiction_count"], 0)

    def test_explicit_opposite_state_is_contradiction(self):
        expected = {label: "unknown" for label in CHEXPERT_FINDINGS}
        observed = dict(expected)
        expected["pneumothorax"] = "positive"
        observed["pneumothorax"] = "negative"
        result = compare_finding_states(expected, observed)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["strong_contradiction_count"], 1)

    def test_lexical_report_parser_preserves_unmentioned_unknown(self):
        states = extract_report_finding_states(
            "No pleural effusion or pneumothorax. Mild cardiomegaly."
        )
        self.assertEqual(states["pleural_effusion"], "negative")
        self.assertEqual(states["pneumothorax"], "negative")
        self.assertEqual(states["cardiomegaly"], "positive")
        self.assertEqual(states["pneumonia"], "unknown")

    def test_xrv_support_ignores_unavailable_device(self):
        expected = {label: "unknown" for label in CHEXPERT_FINDINGS}
        expected["cardiomegaly"] = "positive"
        expected["support_devices"] = "positive"
        result = xrv_ehr_support(expected, {"cardiomegaly": 0.8})
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["unavailable_findings"], ["support_devices"])

    def test_report_quality_flags_temporal_language_without_hard_reject(self):
        result = deterministic_report_quality(
            "Compared with the prior examination, mild cardiomegaly remains unchanged."
        )
        self.assertTrue(result["hard_gate_pass"])
        self.assertTrue(result["flags"]["unsupported_temporal_language"])
        self.assertLess(result["score"], 1.0)

    def test_qwen_state_vector_requires_exact_order_length(self):
        values = validate_qwen_states(["?"] * len(CHEXPERT_FINDINGS))
        self.assertEqual(values, ("unknown",) * len(CHEXPERT_FINDINGS))
        with self.assertRaises(ValueError):
            validate_qwen_states(["?"])


if __name__ == "__main__":
    unittest.main()
