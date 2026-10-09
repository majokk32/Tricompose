"""Invented action metadata only, no report/image/model/API inspection."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import summarize_fresh_action_effects as effects
from test_bounded_regeneration import examples
from test_full_pool_report_control import reseal


def image_example():
    base, alt = examples()
    return base, alt, effects.images.compare(base, alt)


def report_example():
    base, _ = examples()
    alt = deepcopy(base)
    alt.update(report_model_id="llavarad", report_candidate_id="fixture_report_switch",
        report_sha256="b" * 64, triple_candidate_id="fixture_triple_switch")
    alt["receipt"].update(report_candidate_id=alt["report_candidate_id"], report_sha256=alt["report_sha256"])
    alt["structure"].update(report_model_id="llavarad", report_candidate_id=alt["report_candidate_id"],
        report_sha256=alt["report_sha256"], normalized_report_sha256=alt["report_sha256"],
        impression_required_by_model_contract=False)
    reseal(alt)
    return base, alt, effects.reports.compare(base, alt)


class FreshActionEffectsTests(unittest.TestCase):
    def test_seed_action_replays_frozen_gate(self):
        base, alt, comparison = image_example()
        original = deepcopy((base, alt, comparison))
        row = effects.action_effect(base, alt, effects.ACTIONS[1], comparison, 4)
        self.assertTrue(row["exploratory_gate_pass"])
        self.assertEqual(row["delta_ehr_cxr_supported_facts"], 1)
        self.assertIsNone(row["clinical_accuracy"])
        self.assertEqual((base, alt, comparison), original)

    def test_report_switch_holds_image_fixed(self):
        base, alt, comparison = report_example()
        row = effects.action_effect(base, alt, effects.ACTIONS[0], comparison, 2)
        self.assertEqual(row["delta_ehr_cxr_supported_facts"], 0)
        self.assertEqual(row["baseline_cxr_sha256"], row["alternative_cxr_sha256"])

    def test_changed_image_cannot_be_called_report_only(self):
        base, alt, comparison = image_example()
        with self.assertRaises(ValueError): effects.action_effect(base, alt, effects.ACTIONS[0], comparison, 2)

    def test_same_generator_cannot_be_called_generator_switch(self):
        base, alt, comparison = image_example()
        with self.assertRaises(ValueError): effects.action_effect(base, alt, effects.ACTIONS[2], comparison, 4)

    def test_image_switch_keeps_common_report_expert(self):
        base, alt, comparison = image_example()
        alt["report_model_id"] = "llavarad"
        with self.assertRaises(ValueError): effects.action_effect(base, alt, effects.ACTIONS[1], comparison, 4)

    def test_changed_ehr_or_fact_hash_rejected(self):
        for field in ("ehr_sha256", "ehr_facts_sha256"):
            base, alt, comparison = image_example()
            alt[field] = "c" * 64
            with self.assertRaises(ValueError): effects.action_effect(base, alt, effects.ACTIONS[1], comparison, 4)

    def test_changed_gate_cannot_force_acceptance(self):
        base, alt, comparison = report_example()
        comparison["exploratory_gate_pass"] = not comparison["exploratory_gate_pass"]
        with self.assertRaises(ValueError): effects.action_effect(base, alt, effects.ACTIONS[0], comparison, 2)

    def test_positive_and_negative_support_not_combined_into_new_score(self):
        base, alt, comparison = report_example()
        row = effects.action_effect(base, alt, effects.ACTIONS[0], comparison, 2)
        for role in ("baseline", "alternative"):
            for edge in effects.EDGES:
                prefix = role + "_" + edge + "_"
                self.assertEqual(row[prefix + "supported_facts"],
                    row[prefix + "supported_positive"] + row[prefix + "supported_negative"])

    def test_missing_ehr_denominator_stays_null(self):
        base, alt, _ = image_example()
        for row in (base, alt):
            for fact in row["receipt"]["fact_states"]: fact["ehr"] = "unknown"
            reseal(row)
        row = effects.action_effect(base, alt, effects.ACTIONS[1], effects.images.compare(base, alt), 4)
        self.assertIsNone(row["alternative_ehr_cxr_support_over_known"])
        self.assertIn("NA", effects.table_csv([row]))

    def test_invalid_charges_rejected(self):
        base, alt, comparison = image_example()
        for charge in (True, -1, 0, 1.5):
            with self.assertRaises(ValueError): effects.action_effect(base, alt, effects.ACTIONS[1], comparison, charge)

    def test_missing_or_duplicate_action_rows_rejected(self):
        base, alt, comparison = image_example()
        row = effects.action_effect(base, alt, effects.ACTIONS[1], comparison, 4)
        for rows in ([row] * 6, [row] * 5):
            with self.assertRaises(ValueError): effects.aggregate(rows)

    def test_aggregate_counts_two_anchors_not_six_patients(self):
        base, alt, comparison = image_example()
        prototype = effects.action_effect(base, alt, effects.ACTIONS[1], comparison, 4)
        rows = []
        for case in ("fixture_anchor_a", "fixture_anchor_b"):
            for action in effects.ACTIONS:
                row = deepcopy(prototype)
                row.update(case_id=case, action=action,
                    charged_new_worker_attempts=2 if action == effects.ACTIONS[0] else 4)
                rows.append(row)
        summary = effects.aggregate(rows)
        self.assertEqual((summary["fixed_ehr_cases"], summary["observed_pairs"]), (2, 6))
        self.assertEqual(summary["charged_new_worker_attempts_in_two_source_jobs"], 20)
        self.assertFalse(summary["action_effects_are_randomized_causal_estimates"])
        self.assertFalse(summary["historical_acquisition_or_planner_cost_refunded"])

    def test_original_numeric_receipt_cannot_be_forged(self):
        base, alt, comparison = image_example()
        alt["raw_edge_readouts"]["ehr_cxr"]["supported_facts"] += 1
        with self.assertRaises(ValueError): effects.action_effect(base, alt, effects.ACTIONS[1], comparison, 4)

    def test_cpu_guard_before_metadata_reads(self):
        with patch.object(effects, "cpu_guard", side_effect=RuntimeError("blocked")), \
                patch.object(effects, "load_sources") as load:
            with self.assertRaises(RuntimeError): effects.build(SimpleNamespace())
            load.assert_not_called()


if __name__ == "__main__": unittest.main()
