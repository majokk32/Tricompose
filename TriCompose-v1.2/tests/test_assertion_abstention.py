"""Invented metadata fixtures; no protected inputs, inference, or clinical gold."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from tricompose_v12 import assertion_abstention as m
import audit_cached_opacity_abstention as worker


def proposed(state="positive"):
    return {"status": "complete", "state": state, "failure_reason": None,
            "selected_segment_ids": [0], "assertions": [{"segment_id": 0, "state": state}]}


def missing():
    return {"status": "failed_unavailable", "state": None, "failure_reason": "object_response_required"}


def decision(state="positive", other="positive", source=True):
    return m.gate(proposed(state), proposed(other), source_contract_checked=source)


class EligibilityTests(unittest.TestCase):
    def test_exact_positive_agreement_is_soft_only(self):
        d = decision()
        self.assertEqual(d["decision"], "correlated_agreement_soft_only")
        self.assertEqual(d["soft_retained_state"], "positive")
        self.assertTrue(d["soft_comparable"])
        self.assertFalse(d["hard_action_eligible"])

    def test_exact_negative_preserved(self):
        d = decision("negative", "negative")
        self.assertEqual(d["raw_state"], d["soft_retained_state"])

    def test_unknown_is_not_negative_or_comparable(self):
        d = decision("unknown", "unknown")
        self.assertEqual(d["decision"], "not_comparable_unknown")
        self.assertEqual(d["raw_state"], "unknown")
        self.assertIsNone(d["soft_retained_state"])

    def test_uncertain_retained_without_signed_comparison(self):
        d = decision("uncertain", "positive")
        self.assertEqual(d["decision"], "not_comparable_uncertain")
        self.assertEqual(d["raw_state"], "uncertain")

    def test_unavailable_is_null_not_unknown(self):
        d = m.gate(missing(), proposed(), source_contract_checked=True)
        self.assertEqual(d["decision"], "extractor_unavailable")
        self.assertIsNone(d["raw_state"])

    def test_incomplete_source_prevents_agreement(self):
        self.assertEqual(decision(source=False)["decision"], "source_contract_unavailable")

    def test_no_boolean_coercion_of_source_check(self):
        for value in (1, 0, None, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError): decision(source=value)

    def test_polarity_disagreement_veto_only(self):
        d = decision("positive", "negative")
        self.assertEqual(d["decision"], "abstain_cross_format_disagreement")
        self.assertEqual(d["raw_state"], "positive")
        self.assertIsNone(d["soft_retained_state"])

    def test_uncertain_or_unknown_crosscheck_not_agreement(self):
        for state in ("uncertain", "unknown"):
            with self.subTest(state=state):
                self.assertEqual(decision(other=state)["decision"], "abstain_cross_format_disagreement")

    def test_crosscheck_failure_does_not_become_unknown(self):
        d = m.gate(proposed(), missing(), source_contract_checked=True)
        self.assertEqual(d["decision"], "abstain_cross_format_unavailable")
        self.assertIsNone(d["crosscheck_state"])

    def test_mixed_segment_unknown_abstains(self):
        p = proposed()
        p["selected_segment_ids"].append(1)
        p["assertions"].append({"segment_id": 1, "state": "unknown"})
        d = m.gate(p, proposed(), source_contract_checked=True)
        self.assertEqual(d["decision"], "abstain_mixed_segment_states")

    def test_opposing_segment_states_not_soft_determinate(self):
        p = proposed()
        p["selected_segment_ids"].append(1)
        p["assertions"].append({"segment_id": 1, "state": "negative"})
        self.assertFalse(m.gate(p, proposed(), source_contract_checked=True)["soft_comparable"])

    def test_missing_assertion_inventory_refused(self):
        p = proposed(); del p["assertions"]
        with self.assertRaises(ValueError): m.gate(p, proposed(), source_contract_checked=True)

    def test_empty_determinate_locator_refused(self):
        p = proposed(); p.update(selected_segment_ids=[], assertions=[])
        with self.assertRaises(ValueError): m.gate(p, proposed(), source_contract_checked=True)

    def test_duplicate_or_boolean_ids_refused(self):
        for ids in ([0, 0], [True], [0.0], [-1], [1, 0]):
            p = proposed(); p["selected_segment_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                m.gate(p, proposed(), source_contract_checked=True)

    def test_nonexhaustive_or_extra_assertions_refused(self):
        for assertions in ([], [{"segment_id": 1, "state": "positive"}],
                [{"segment_id": 0, "state": "positive", "confidence": 1}]):
            p = proposed(); p["assertions"] = assertions
            with self.subTest(assertions=assertions), self.assertRaises(ValueError):
                m.gate(p, proposed(), source_contract_checked=True)

    def test_failed_prediction_cannot_retain_placeholder_unknown(self):
        p = missing(); p["state"] = "unknown"
        with self.assertRaises(ValueError): m.gate(p, proposed(), source_contract_checked=True)

    def test_complete_prediction_cannot_have_null_state(self):
        p = proposed(); p["state"] = None
        with self.assertRaises(ValueError): m.gate(p, proposed(), source_contract_checked=True)

    def test_invalid_status_refused(self):
        p = proposed(); p["status"] = "accepted"
        with self.assertRaises(ValueError): m.gate(p, proposed(), source_contract_checked=True)

    def test_complete_failure_reason_refused(self):
        p = proposed(); p["failure_reason"] = "bad"
        with self.assertRaises(ValueError): m.gate(p, proposed(), source_contract_checked=True)

    def test_failed_requires_reason(self):
        p = missing(); p["failure_reason"] = None
        with self.assertRaises(ValueError): m.gate(p, proposed(), source_contract_checked=True)

    def test_inputs_unchanged_and_output_deterministic(self):
        p, c = proposed(), proposed(); before = copy.deepcopy((p, c))
        first = m.gate(p, c, source_contract_checked=True)
        self.assertEqual((p, c), before)
        self.assertEqual(first, m.gate(p, c, source_contract_checked=True))

    def test_reference_metadata_cannot_override_gate(self):
        p = proposed(); p.update(expected_state="negative", family="arbitrary", clinical_confidence=1)
        self.assertEqual(m.gate(p, proposed(), source_contract_checked=True), decision())

    def test_no_scope_or_independence_promotion(self):
        d = decision()
        for key in ("semantic_scope_verified", "independent_clinical_qualification", "cross_format_evidence_independent",
                    "clinical_fault_localization", "regeneration_authorized", "selection_changed", "candidate_dropped", "state_corrected"):
            self.assertIs(d[key], False)
        self.assertIsNone(d["clinical_score"])

    def test_full_four_state_inventory_never_enables_hard_action(self):
        rows = [decision(a, b) for a in m.STATES for b in m.STATES]
        self.assertEqual(m.summarize(rows)["rows_retained"], 16)
        self.assertEqual(m.summarize(rows)["hard_action_eligible"], 0)

    def test_empty_coverage_is_null_not_zero_error_success(self):
        self.assertIsNone(m.summarize([])["soft_coverage_all_rows"])
        self.assertIsNone(m.authored_readout([])["conditional_authored_match"])

    def test_same_wrong_agreement_remains_visible(self):
        row = {**decision(), "expected_state": "uncertain"}
        r = m.authored_readout([row])
        self.assertEqual(r["soft_incorrect"], 1)
        self.assertEqual(r["conditional_authored_match"], 0)
        self.assertEqual(r["hard_action_eligible"], 0)

    def test_withheld_errors_and_lost_correct_both_counted(self):
        rows = [{**decision(other="negative"), "expected_state": s} for s in ("positive", "uncertain")]
        r = m.authored_readout(rows)
        self.assertEqual(r["incorrect_determinate_withheld"], 1)
        self.assertEqual(r["correct_determinate_withheld"], 1)
        self.assertIsNone(r["conditional_authored_match"])

    def test_correct_unknown_not_credited_as_retained_assertion(self):
        r = m.authored_readout([{**decision("unknown", "unknown"), "expected_state": "unknown"}])
        self.assertEqual(r["raw_four_state_matches"], 1)
        self.assertEqual(r["soft_correct"], 0)
        self.assertFalse(r["abstention_credited_as_match"])
        self.assertIsNone(r["overall_gated_accuracy"])

    def test_retained_label_flip_rejected_by_readout(self):
        row = {**decision(), "expected_state": "positive", "soft_retained_state": "negative"}
        with self.assertRaises(ValueError): m.authored_readout([row])

    def test_policy_discloses_known_development_and_no_scope(self):
        self.assertTrue(m.POLICY["declared_after_known_v3_development_results"])
        self.assertFalse(m.POLICY["heldout_test"])
        self.assertEqual(m.POLICY["scope_qualification"], "no_frozen_opacity_scope_head")

    def test_csv_does_not_contain_source_or_response_bodies(self):
        row = {"item_id": "invented_0000", "report_sha256": "a" * 64, **decision()}
        output = worker.csv_text([row])
        for key in ("quote", "response", "expected_state", "family", "source_span_references"):
            self.assertNotIn(key, output.splitlines()[0])

    def test_short_authored_join_refused(self):
        with self.assertRaises(ValueError): worker.join_authored([], [])

    def test_wrapper_requires_actual_slurm_before_loading_inputs(self):
        with patch.object(worker.original.legacy.OP, "guard", side_effect=RuntimeError("slurm_required")), \
                patch.object(worker, "load_fixed") as loader:
            with self.assertRaises(RuntimeError): worker.execute("unused", "invented")
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
