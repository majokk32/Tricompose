from __future__ import annotations

import hashlib
import unittest

from tricompose_v1.ehr_bridge import canonicalize_synehrgy_case
from tricompose_v1.facts import FACT_PATTERNS, extract_ehr_facts
from tricompose_v1.prompts import (
    ACTIVE_PROMPT_MODELS,
    render_prompt,
    validate_rendered_prompt,
)


def _canonical(problem: str) -> dict:
    return canonicalize_synehrgy_case(
        {
            "schema": "tricompose.synehrgy_v2.synthetic_ehr.v1",
            "case_id": "case_000",
            "structure": {
                "top_level_tokens": [],
                "visits": [
                    {
                        "covariates": [],
                        "problems": [problem] if problem else [],
                        "labs": [],
                        "charts": [],
                        "other_tokens": [],
                    }
                ],
            },
            "validation": {"strict_valid": True},
        },
        source_model_id="synehrgy_qwen2_40bins",
    )


class PromptTests(unittest.TestCase):
    def test_unknown_does_not_become_negative_and_all_facts_are_traceable(self) -> None:
        facts = extract_ehr_facts(_canonical("J90_Pleural effusion"))
        for model_id in ACTIVE_PROMPT_MODELS:
            rendered = render_prompt(facts, model_id)
            self.assertEqual(rendered["included_fact_ids"], ["pleural_effusion"])
            self.assertNotIn("no pneumonia", rendered["text"].lower())
            for fact_id in rendered["included_fact_ids"]:
                self.assertIn(fact_id, FACT_PATTERNS)
                self.assertNotEqual(facts["facts"][fact_id]["state"], "unknown")
                self.assertTrue(facts["facts"][fact_id]["source_fields"])

    def test_no_unsupported_view_laterality_severity_or_location_is_added(self) -> None:
        facts = extract_ehr_facts(_canonical("J90_Left small pleural effusion"))
        banned = (
            "left",
            "right",
            "bilateral",
            "small",
            "large",
            "mild",
            "severe",
            "frontal",
            "portable",
            "ap view",
        )
        for model_id in ACTIVE_PROMPT_MODELS:
            text = render_prompt(facts, model_id)["text"].lower()
            self.assertTrue(all(term not in text for term in banned))

    def test_deterministic_output_and_no_duplicate_prefix(self) -> None:
        facts = extract_ehr_facts(_canonical("I517_Cardiomegaly"))
        for model_id in ACTIVE_PROMPT_MODELS:
            first = render_prompt(facts, model_id)
            second = render_prompt(facts, model_id)
            self.assertEqual(first, second)
            tampered = dict(first)
            tampered["text"] = first["text"] + " " + first["text"]
            tampered["prompt_sha256"] = hashlib.sha256(
                tampered["text"].encode("utf-8")
            ).hexdigest()
            with self.assertRaises(ValueError):
                validate_rendered_prompt(tampered, facts)

    def test_fact_absence_uses_neutral_prompt_and_is_marked(self) -> None:
        facts = extract_ehr_facts(_canonical(""))
        self.assertTrue(facts["summary"]["underconditioned"])
        for model_id in ACTIVE_PROMPT_MODELS:
            rendered = render_prompt(facts, model_id)
            self.assertEqual(rendered["included_fact_ids"], [])
            self.assertTrue(rendered["underconditioned"])
            self.assertNotIn("normal", rendered["text"].lower())

    def test_explicit_negation_is_preserved_but_legacy_prompt_is_positive_only(self) -> None:
        explicit = extract_ehr_facts(
            _canonical("Z_No pleural effusion")
        )
        unrelated_not = extract_ehr_facts(
            _canonical("J90_Pleural effusion, not elsewhere classified")
        )
        self.assertEqual(
            explicit["facts"]["pleural_effusion"]["state"], "negative"
        )
        self.assertEqual(
            unrelated_not["facts"]["pleural_effusion"]["state"], "positive"
        )
        rendered = render_prompt(explicit, "roentgen_v2")
        self.assertEqual(rendered["included_fact_ids"], [])
        self.assertNotIn("pleural effusion", rendered["text"].lower())

    def test_all_three_models_reproduce_the_previous_shared_prompt(self) -> None:
        facts = extract_ehr_facts(_canonical("I50_Congestive heart failure"))
        expected = (
            "Adult patient. PA chest radiograph. Findings: Cardiomegaly with "
            "pulmonary vascular congestion and interstitial edema compatible "
            "with congestive heart failure is present."
        )
        renderings = [render_prompt(facts, model_id) for model_id in ACTIVE_PROMPT_MODELS]
        self.assertEqual({row["text"] for row in renderings}, {expected})
        self.assertTrue(
            all(row["legacy_experiment_reproduction"] for row in renderings)
        )

    def test_heart_failure_uses_a_traceable_disease_to_cxr_prior(self) -> None:
        facts = extract_ehr_facts(_canonical("I50_Congestive heart failure"))
        rendered = render_prompt(facts, "roentgen_v2")
        self.assertEqual(
            rendered["included_fact_ids"], ["congestive_heart_failure"]
        )
        self.assertEqual(
            rendered["derived_rule_ids"],
            ["congestive_heart_failure_to_cxr_prior_v1"],
        )
        self.assertIn("cardiomegaly", rendered["text"].lower())

    def test_nonlegacy_indication_does_not_become_an_image_condition(self) -> None:
        facts = extract_ehr_facts(_canonical("R07_Chest pain"))
        self.assertEqual(facts["facts"]["chest_pain"]["state"], "positive")
        rendered = render_prompt(facts, "chexgenbench_sana")
        self.assertEqual(rendered["included_fact_ids"], [])
        self.assertTrue(rendered["underconditioned"])
        self.assertNotIn("chest pain", rendered["text"].lower())


if __name__ == "__main__":
    unittest.main()
