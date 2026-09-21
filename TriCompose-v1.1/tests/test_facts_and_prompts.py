from __future__ import annotations

import unittest

from tricompose_v1.ehr_bridge import canonicalize_synehrgy_case
from tricompose_v11.facts import CONTEXT_PATTERNS, extract_v11_facts
from tricompose_v11.prompts import (
    ACTIVE_PROMPT_MODELS_V11,
    render_v11_prompt,
    validate_v11_prompt,
)


def _canonical(*visits: list[str]) -> dict:
    return canonicalize_synehrgy_case(
        {
            "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
            "case_id": "case_000",
            "structure": {
                "top_level_tokens": [],
                "visits": [
                    {
                        "covariates": [],
                        "problems": problems,
                        "labs": [],
                        "charts": [],
                        "other_tokens": [],
                    }
                    for problems in visits
                ],
            },
            "validation": {"strict_valid": True},
        },
        source_model_id="synehrgy_gpt2_10bins",
    )


class V11FactsAndPromptsTests(unittest.TestCase):
    def test_context_is_grounded_but_not_promoted_to_image_finding(self) -> None:
        facts = extract_v11_facts(
            _canonical(
                ["J44_Chronic obstructive pulmonary disease"],
                ["I10_Essential hypertension"],
            )
        )
        for context_id in (
            "chronic_obstructive_pulmonary_disease",
            "hypertensive_disease",
        ):
            context = facts["clinical_contexts"][context_id]
            self.assertEqual(context["status"], "documented")
            self.assertTrue(context["evidence"])
            self.assertTrue(
                all(field.startswith("diagnoses[") for field in context["source_fields"])
            )
        rendered = render_v11_prompt(facts, "roentgen_v2")
        self.assertEqual(rendered["included_direct_fact_ids"], [])
        self.assertIn(
            "chronic obstructive pulmonary disease", rendered["text"].lower()
        )
        self.assertIn("radiographic findings are unspecified", rendered["text"].lower())
        self.assertTrue(rendered["context_is_not_a_radiographic_assertion"])

    def test_same_direct_finding_with_different_context_does_not_collapse(self) -> None:
        copd = extract_v11_facts(
            _canonical(["I50_Congestive heart failure", "J44_Chronic airway obstruction"])
        )
        asthma = extract_v11_facts(
            _canonical(["I50_Congestive heart failure", "J45_Asthma"])
        )
        for model_id in ACTIVE_PROMPT_MODELS_V11:
            first = render_v11_prompt(copd, model_id)
            second = render_v11_prompt(asthma, model_id)
            self.assertNotEqual(first["clinical_intent_sha256"], second["clinical_intent_sha256"])
            self.assertNotEqual(first["prompt_sha256"], second["prompt_sha256"])
            self.assertEqual(first["included_direct_fact_ids"], ["congestive_heart_failure"])
            self.assertEqual(second["included_direct_fact_ids"], ["congestive_heart_failure"])

    def test_all_models_share_intent_but_use_model_specific_surfaces(self) -> None:
        facts = extract_v11_facts(
            _canonical(["J18_Pneumonia", "I48_Atrial fibrillation"])
        )
        rendered = [
            render_v11_prompt(facts, model_id)
            for model_id in ACTIVE_PROMPT_MODELS_V11
        ]
        self.assertEqual(len({row["clinical_intent_sha256"] for row in rendered}), 1)
        self.assertEqual(len({row["text"] for row in rendered}), 2)
        for row in rendered:
            validate_v11_prompt(row, facts)
            self.assertNotIn("case_000", row["text"])

    def test_unknown_and_negative_facts_are_not_rendered(self) -> None:
        facts = extract_v11_facts(_canonical(["Z_No pleural effusion"]))
        self.assertEqual(facts["direct_facts"]["pleural_effusion"]["state"], "negative")
        rendered = render_v11_prompt(facts, "chexgenbench_sana")
        self.assertEqual(rendered["included_direct_fact_ids"], [])
        self.assertNotIn("pleural effusion", rendered["text"].lower())
        self.assertNotIn("pneumonia", rendered["text"].lower())

    def test_neutral_case_remains_honestly_underconditioned(self) -> None:
        facts = extract_v11_facts(_canonical([]))
        self.assertTrue(facts["summary"]["underconditioned"])
        for model_id in ACTIVE_PROMPT_MODELS_V11:
            rendered = render_v11_prompt(facts, model_id)
            self.assertTrue(rendered["underconditioned"])
            self.assertEqual(rendered["included_direct_fact_ids"], [])
            self.assertEqual(rendered["included_context_ids"], [])
            self.assertNotIn("normal", rendered["text"].lower())

    def test_chest_pain_alone_is_not_promoted_to_an_image_finding(self) -> None:
        facts = extract_v11_facts(_canonical(["R07.9_Chest pain"]))
        rendered = render_v11_prompt(facts, "roentgen_v2")
        self.assertTrue(facts["summary"]["underconditioned"])
        self.assertEqual(rendered["included_direct_fact_ids"], [])
        self.assertNotIn("chest pain", rendered["text"].lower())

    def test_no_untraceable_attribute_or_random_nonce_is_added(self) -> None:
        facts = extract_v11_facts(
            _canonical(["J90_Left small pleural effusion", "I10_Hypertension"])
        )
        banned = ("left", "right", "small", "large", "mild", "severe", "portable")
        for model_id in ACTIVE_PROMPT_MODELS_V11:
            first = render_v11_prompt(facts, model_id)
            second = render_v11_prompt(facts, model_id)
            self.assertEqual(first, second)
            self.assertTrue(all(word not in first["text"].lower() for word in banned))
            for context_id in first["included_context_ids"]:
                self.assertIn(context_id, CONTEXT_PATTERNS)
                self.assertTrue(facts["clinical_contexts"][context_id]["source_fields"])

    def test_roentgen_context_budget_preserves_full_intent_lineage(self) -> None:
        facts = extract_v11_facts(
            _canonical(
                [
                    "J44_Chronic obstructive pulmonary disease",
                    "J45_Asthma",
                    "I25_Coronary artery disease",
                    "I48_Atrial fibrillation",
                    "I10_Hypertension",
                    "N18_Chronic kidney disease",
                ]
            )
        )
        rendered = render_v11_prompt(facts, "roentgen_v2")
        self.assertGreater(len(rendered["available_context_ids"]), 4)
        self.assertEqual(len(rendered["included_context_ids"]), 4)
        self.assertEqual(
            rendered["available_context_ids"],
            rendered["included_context_ids"] + rendered["omitted_context_ids"],
        )


if __name__ == "__main__":
    unittest.main()
