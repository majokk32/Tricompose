from __future__ import annotations

import json
import unittest
from pathlib import Path

from tricompose_v1.planner import build_exhaustive_plan
from tricompose_v1.registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]


class RegistryAndPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ModelRegistry.from_path(
            ROOT / "configs" / "model_registry_v1.json"
        )

    def test_active_registry_is_frozen_and_scientifically_scoped(self) -> None:
        summary = self.registry.summary()
        self.assertEqual(
            summary["active_counts"],
            {"ehr": 2, "cxr": 3, "report": 4, "verifier": 5},
        )
        self.assertTrue(summary["all_active_models_frozen"])
        self.assertFalse(summary["active_cxr_models_require_previous_cxr"])
        self.assertEqual(summary["active_direct_structured_ehr_to_cxr_count"], 0)

    def test_incompatible_models_remain_registered_but_disabled(self) -> None:
        disabled = {model.model_id for model in self.registry.disabled()}
        self.assertTrue(
            {
                "promptehr_mimic3_seeded",
                "halo",
                "ehrxdiff_longitudinal",
                "ddl_cxr_longitudinal",
                "unidisc_t2i",
                "liquid_t2i",
                "radedit_t2i",
                "medim_t2i",
                "cxrmate_ed",
                "unidisc_cxr_report",
            }.issubset(disabled)
        )

    def test_default_plan_expands_all_active_generators(self) -> None:
        plan = build_exhaustive_plan(self.registry)
        self.assertTrue(plan["fully_synthetic"])
        self.assertFalse(plan["uses_real_patient_input"])
        self.assertEqual(
            plan["counts"],
            {
                "ehr_candidates": 2,
                "cxr_candidates": 6,
                "report_candidates": 24,
                "generator_calls": 32,
                "total_actions": 118,
            },
        )
        actions = plan["actions"]
        report_actions = [
            action for action in actions if action["stage"] == "generate_report"
        ]
        self.assertEqual(len(report_actions), 24)
        self.assertTrue(
            all(len(action["parent_candidate_ids"]) == 2 for action in report_actions)
        )

    def test_default_interface_selects_no_model_and_uses_cold_start_pool(self) -> None:
        plan = build_exhaustive_plan(self.registry)
        self.assertFalse(plan["ehr_selection"]["user_selects_model"])
        self.assertFalse(plan["ehr_selection"]["use_prompt"])
        self.assertEqual(
            plan["ehr_selection"]["routing_mode"],
            "unconditional_cold_start_candidate_pool",
        )
        self.assertTrue(plan["ehr_selection"]["strict_cold_start"])
        self.assertEqual(plan["ehr_selection"]["prompt_contract"], "none")
        self.assertEqual(plan["counts"]["ehr_candidates"], 2)

    def test_prompt_boolean_routes_internally_to_promptehr(self) -> None:
        plan = build_exhaustive_plan(
            self.registry,
            use_ehr_prompt=True,
        )
        self.assertTrue(plan["fully_synthetic"])
        self.assertFalse(plan["ehr_selection"]["user_selects_model"])
        self.assertTrue(plan["ehr_selection"]["use_prompt"])
        self.assertEqual(
            plan["ehr_selection"]["routing_mode"],
            "prompt_conditioned_promptehr",
        )
        self.assertEqual(
            plan["ehr_selection"]["model_ids"],
            ["promptehr_mimic3_seeded"],
        )
        self.assertFalse(plan["ehr_selection"]["strict_cold_start"])
        self.assertEqual(
            plan["ehr_selection"]["prompt_contract"],
            "partial_structured_synthetic_ehr_seed",
        )
        self.assertEqual(
            plan["ehr_selection"]["exploratory_incompatible_model_ids"],
            ["promptehr_mimic3_seeded"],
        )
        self.assertEqual(plan["counts"]["ehr_candidates"], 1)
        self.assertEqual(plan["counts"]["cxr_candidates"], 3)
        self.assertEqual(plan["counts"]["report_candidates"], 12)

    def test_all_json_configs_and_schemas_parse(self) -> None:
        for path in sorted((ROOT / "configs").glob("*.json")) + sorted(
            (ROOT / "schemas").glob("*.json")
        ):
            with self.subTest(path=path.name):
                self.assertIsInstance(json.loads(path.read_text()), dict)


if __name__ == "__main__":
    unittest.main()
