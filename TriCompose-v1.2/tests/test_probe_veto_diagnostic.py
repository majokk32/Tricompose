"""Authored numeric fixtures; no clinical outputs or model execution."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import diagnose_fresh_cxr_probe as diagnostic
from test_fresh_cxr_postflight import package


def example(image_state="negative"):
    p = package(image_state=image_state)
    return p["case"]["reference"], p["step"]["proposal"]


def analyze(before, after, left=-.1, right=.7):
    return diagnostic.decompose(before, after, diagnostic.image_gate.compare(before, after), left, right)


class ProbeVetoDiagnosticTests(unittest.TestCase):
    def test_higher_cosine_is_not_ehr_gain_or_clinical_acceptance(self):
        before, after = example()
        result = analyze(before, after)
        self.assertTrue(result["higher_cosine_but_vetoed"])
        self.assertEqual(result["failed_checks"], ["fixed_ehr_image_gain"])
        self.assertEqual(result["actual_quality_failures"], [])
        self.assertIsNone(result["clinical_fault_location"])
        self.assertFalse(result["selection_changed"])
        self.assertFalse(result["thresholds_changed"])

    def test_small_repetition_increase_named_separately_from_format(self):
        before, after = example()
        after["structure"]["repeated_4gram_ratio"] = .00833333
        result = analyze(before, after)
        self.assertTrue(result["quality_checks"]["section_and_nonempty_pass"])
        self.assertTrue(result["quality_checks"]["risk_flags_not_worse"])
        self.assertEqual(result["actual_quality_failures"], ["repeated_4gram_ratio_not_worse"])
        self.assertIn("fixed_ehr_image_gain", result["failed_checks"])

    def test_existing_temporal_risk_not_invented_as_new_worsening(self):
        before, after = example()
        for r in (before, after): r["structure"]["unsupported_temporal_comparison_language"] = True
        result = analyze(before, after)
        self.assertTrue(result["quality_checks"]["risk_flags_not_worse"])

    def test_new_temporal_risk_named(self):
        before, after = example()
        before["structure"]["unsupported_temporal_comparison_language"] = False
        after["structure"]["unsupported_temporal_comparison_language"] = True
        result = analyze(before, after)
        self.assertIn("risk_flags_not_worse", result["actual_quality_failures"])

    def test_actual_section_failure_not_hidden(self):
        before, after = example()
        after["structure"].update(impression_complete=False, section_contract_pass=False)
        result = analyze(before, after)
        self.assertFalse(result["quality_checks"]["section_and_nonempty_pass"])

    def test_label_improvement_with_no_regression_passes_but_no_truth_claim(self):
        before, after = example("positive")
        result = analyze(before, after)
        self.assertEqual(result["failed_checks"], [])
        self.assertFalse(result["higher_cosine_but_vetoed"])
        self.assertIsNone(result["clinical_accuracy"])

    def test_unknown_is_lost_comparability_not_resolved_fact(self):
        before, after = example("unknown")
        result = analyze(before, after)
        self.assertEqual(result["edges"]["ehr_cxr"]["lost_comparable_count"], 1)
        self.assertIn("previous_fact_sets_preserved", result["failed_checks"])
        self.assertEqual(result["edges"]["ehr_cxr"]["opposition_silenced_by_lost_comparison_count"], 1)
        # The frozen raw gain predicate counts removed opposition; the separate
        # preservation veto blocks silence-by-unknown. Do not rewrite that gate.
        self.assertTrue(result["gate_checks"]["fixed_ehr_image_gain"])

    def test_unavailable_secondary_endpoint_stays_none(self):
        before, after = example()
        result = analyze(before, after, right=None)
        self.assertIsNone(result["cosine_delta"])
        self.assertFalse(result["higher_cosine_but_vetoed"])

    def test_tampered_transition_rejected(self):
        before, after = example()
        transition = diagnostic.image_gate.compare(before, after)
        transition["exploratory_gate_pass"] = True
        with self.assertRaisesRegex(ValueError, "must_replay"):
            diagnostic.decompose(before, after, transition, -.1, .7)

    def test_mutation_not_applied_to_input_rows(self):
        before, after = example(); initial = copy.deepcopy((before,after))
        analyze(before, after)
        self.assertEqual((before,after), initial)

    def test_no_classifier_metadata_missing_pin_cannot_be_invented(self):
        before, _ = example()
        with self.assertRaisesRegex(ValueError, "one_authenticated"):
            diagnostic.label_readout(before, {})

    def test_classifier_records_schema_and_legacy_score_key(self):
        before, _ = example()
        facts = before["receipt"]["fact_states"]
        labels = {"score_semantics": "xrv_op_norm_0_1", "probability_semantics": False,
            "records": [{"cxr_candidate_id":before["cxr_candidate_id"], "image_sha256":before["cxr_sha256"],
                "finding_states":{f["finding"]:f["xrv"] for f in facts},
                "finding_probabilities":{f["finding"]:.2 for f in facts}}],
            "thresholds":{f["finding"]:{"enabled":True,"negative_max":.4,"positive_min":.6} for f in facts}}
        pins = {"/invented/scored/cxr_finding_labels.json":before["receipt"]["xrv_labels_sha256"]}
        with patch.object(diagnostic.endpoint.previous.postflight, "MetadataReader") as reader:
            reader.return_value.json.return_value = labels
            reader.return_value.pins = {}
            result, _ = diagnostic.label_readout(before, pins)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["op_norm_score"], .2)
        self.assertNotIn("finding",result[0])

    def test_cpu_guard_precedes_metadata_or_writes(self):
        with patch.object(diagnostic.endpoint.previous.gate, "cpu_guard", side_effect=RuntimeError("invented_login")), \
             patch.object(diagnostic.endpoint.previous.postflight, "MetadataReader") as reader, \
             patch.object(diagnostic, "new_atomic_run") as output:
            with self.assertRaises(RuntimeError): diagnostic.diagnose(object())
            reader.assert_not_called(); output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
