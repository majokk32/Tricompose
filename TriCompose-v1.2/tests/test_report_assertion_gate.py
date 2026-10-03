"""Wholly invented fixtures; no trained parser, patient input or GPU."""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"benchmarks"))
from tricompose_v12 import report_assertions as a
from repair_cached_report_evidence import scope_check
import gate_report_assertion_predictions as g
import score_report_assertions_context as c
import compare_report_assertion_sources as comparison


def flags(**updates):
    return {**dict.fromkeys(a.CONTEXT_FLAGS, False), **updates}


def context_record(text, state, context_flags, finding="cardiomegaly"):
    start = text.lower().index(finding)
    mention = {"span": a.checked_span(text, start, start+len(finding)), "flags": context_flags, "modifiers": []}
    return {"state": state, "mentions": [mention]}


class SourceContractTests(unittest.TestCase):
    def test_unicode_offset_and_hash(self):
        text = "虚构\nCardiomegaly is present."
        start = text.index("Cardiomegaly")
        span = a.checked_span(text, start, len(text))
        a.validate_span(text, span)
        self.assertEqual(span["offset_unit"], "unicode_codepoint")

    def test_span_tampering_refused(self):
        text = "Cardiomegaly is present."
        span = a.checked_span(text, 0, len(text))
        for key, value in (("char_start", 1), ("quote", "invented_wrong"), ("quote_sha256", "wrong"), ("offset_unit", "bytes")):
            changed = {**span, key: value}
            with self.assertRaises(ValueError):
                a.validate_span(text, changed)

    def test_invalid_source_and_offsets_refused(self):
        for start, end in ((True, 3), (0, 0), (-1, 3), (0, 999)):
            with self.assertRaises(ValueError):
                a.checked_span("fiction", start, end)
        for text in ("", None, "x"*8193):
            with self.assertRaises(ValueError):
                a.gate_assertion(text, "cardiomegaly", "unknown", scope_check)

    def test_no_implicit_clinical_validation(self):
        row = a.gate_assertion("Cardiomegaly is present.", "cardiomegaly", "positive", scope_check)
        self.assertFalse(row["independent_clinical_validation"])
        self.assertFalse(row["regeneration_authorized"])
        self.assertEqual(row["evidence_dependency"], "same_report_text")


class ScopeGateTests(unittest.TestCase):
    def gate(self, text, state):
        return a.gate_assertion(text, "cardiomegaly", state, scope_check)

    def test_checked_positive_commit(self):
        row = self.gate("Cardiomegaly is present.", "positive")
        self.assertEqual((row["state"], row["decision"]), ("positive", "scope_commit"))
        self.assertTrue(row["scope_verified"])

    def test_checked_negative_commit(self):
        row = self.gate("Cardiomegaly is absent.", "negative")
        self.assertEqual((row["state"], row["decision"]), ("negative", "scope_commit"))

    def test_wrong_polarity_abstains_never_flips(self):
        row = self.gate("Cardiomegaly is absent.", "positive")
        self.assertEqual((row["state"], row["decision"]), ("unknown", "abstain"))
        self.assertEqual(row["proposed_state"], "positive")
        self.assertEqual(row["review_action"], "verify_more")

    def test_unknown_never_promoted_even_explicit_source(self):
        row = self.gate("Cardiomegaly is present.", "unknown")
        self.assertEqual((row["state"], row["decision"]), ("unknown", "no_model_assertion"))
        self.assertFalse(row["scope_verified"])
        self.assertEqual(row["evidence"], [])

    def test_uncovered_scope_is_not_a_pass(self):
        row = self.gate("Cardiomegaly.", "positive")
        self.assertEqual(row["decision"], "abstain")
        self.assertFalse(row["scope_check"]["veto"])

    def test_uncovered_synonym_is_not_a_pass(self):
        row = self.gate("The heart is not enlarged.", "negative")
        self.assertEqual(row["decision"], "abstain")
        self.assertEqual(row["scope_check"]["literal_mentions_checked"], 0)

    def test_unknown_mention_scope_blocks_other_checked_mention(self):
        row = self.gate("Cardiomegaly is present. Cardiomegaly.", "positive")
        self.assertEqual(row["decision"], "abstain")

    def test_conflicting_assertions_abstain(self):
        row = self.gate("Cardiomegaly is present. Cardiomegaly is absent.", "positive")
        self.assertEqual(row["decision"], "abstain")

    def test_uncertainty_cannot_be_committed_as_positive(self):
        text = "Cardiomegaly may be present."
        self.assertEqual(self.gate(text, "positive")["decision"], "abstain")
        self.assertEqual(self.gate(text, "uncertain")["state"], "uncertain")

    def test_qualified_absence_not_global_negative(self):
        text = "No large cardiomegaly."
        self.assertEqual(self.gate(text, "negative")["decision"], "abstain")
        self.assertEqual(self.gate(text, "uncertain")["state"], "uncertain")

    def test_historical_context_not_current_positive(self):
        self.assertEqual(self.gate("History of cardiomegaly is present.", "positive")["decision"], "abstain")

    def test_bad_scope_contract_refused(self):
        for row in ({"literal_mentions_checked": True, "scope_reasons": ["explicit_presence_copula"], "rule_suggested_states": ["positive"]},
                    {"literal_mentions_checked": 1, "scope_reasons": [], "rule_suggested_states": ["positive"]},
                    {"literal_mentions_checked": 1, "scope_reasons": ["explicit_presence_copula"], "rule_suggested_states": ["made_up"]}):
            with self.assertRaises(ValueError):
                a.gate_assertion("Cardiomegaly is present.", "cardiomegaly", "positive", lambda *_: row)

    def test_deterministic_and_no_model_or_reference_dependency(self):
        text = "Cardiomegaly is present."
        self.assertEqual(self.gate(text, "positive"), self.gate(text, "positive"))
        source = inspect.getsource(a.gate_assertion)
        for phrase in ("expected_state", "load_references", "torch", "chexbert"):
            self.assertNotIn(phrase, source)


class ConTextAdapterTests(unittest.TestCase):
    def test_current_flag_mapping(self):
        self.assertEqual(a.context_state(flags()), "positive")
        self.assertEqual(a.context_state(flags(is_negated=True)), "negative")
        self.assertEqual(a.context_state(flags(is_negated=True, is_uncertain=True)), "uncertain")

    def test_history_hypothetical_and_family_not_current(self):
        for name in ("is_historical", "is_hypothetical", "is_family"):
            self.assertEqual(a.context_state(flags(**{name: True})), "unknown")

    def test_non_boolean_or_missing_flags_refused(self):
        for row in ({}, flags(is_negated=1), {**flags(), "made_up": False}):
            with self.assertRaises(ValueError):
                a.context_state(row)

    def test_missing_mentions_not_negative(self):
        self.assertEqual(a.aggregate_context_mentions([]), "unknown")

    def test_opposed_mentions_uncertain(self):
        mentions = [{"flags": flags()}, {"flags": flags(is_negated=True)}]
        self.assertEqual(a.aggregate_context_mentions(mentions), "uncertain")

    def test_agreement_still_not_independent_clinical_votes(self):
        text = "Cardiomegaly is present."
        row = a.gate_assertion(text, "cardiomegaly", "positive", scope_check)
        result = a.crosscheck_context(row, context_record(text, "positive", flags()), text)
        self.assertEqual(result["decision"], "scope_commit")
        self.assertFalse(result["parsers_are_independent_clinical_votes"])

    def test_disagreement_vetoes_not_flips(self):
        text = "Cardiomegaly is present."
        row = a.gate_assertion(text, "cardiomegaly", "positive", scope_check)
        result = a.crosscheck_context(row, context_record(text, "negative", flags(is_negated=True)), text)
        self.assertEqual((result["state"], result["decision"]), ("unknown", "abstain"))

    def test_context_cannot_promote_abstained_proposal(self):
        text = "Cardiomegaly."
        row = a.gate_assertion(text, "cardiomegaly", "positive", scope_check)
        result = a.crosscheck_context(row, context_record(text, "positive", flags()), text)
        self.assertEqual(result["decision"], "abstain")

    def test_context_binding_and_aggregation_refused(self):
        text = "Cardiomegaly is present."
        row = a.gate_assertion(text, "cardiomegaly", "positive", scope_check)
        other = context_record(text, "negative", flags())
        with self.assertRaises(ValueError):
            a.crosscheck_context(row, other, text)
        with self.assertRaises(ValueError):
            a.crosscheck_context(row, context_record(text, "positive", flags()), text+" Changed.")

    def test_parser_slurm_guard_before_imports_inputs_or_versions(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(c, "load_inputs") as inputs, patch.object(c.importlib.metadata, "version") as versions:
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                c.run(SimpleNamespace())
            inputs.assert_not_called(); versions.assert_not_called()

    def test_official_runner_no_custom_trigger_or_gold_reader(self):
        source = inspect.getsource(c.run)
        for phrase in ("load_references", "authored_cases", "ConTextRule(", "context.add("):
            self.assertNotIn(phrase, source)
        calls = [node.func for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "spacy" and node.attr == "load" for node in calls))
        self.assertIn("official_rules_modified", source)

    def test_wrong_parser_versions_refused_before_inputs(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "invented"}), patch.object(c.importlib.metadata, "version", return_value="wrong"), patch.object(c, "load_inputs") as inputs:
            with self.assertRaisesRegex(ValueError, "versions differ"):
                c.run(SimpleNamespace())
            inputs.assert_not_called()

    def test_empty_document_entities_unknown_not_negative(self):
        text = "Invented acquisition note."
        doc = SimpleNamespace(text=text, ents=[])
        result = c.serialize_document(doc, text)
        self.assertEqual(set(result), set(a.FINDINGS))
        self.assertTrue(all(row["state"] == "unknown" and row["mentions"] == [] for row in result.values()))

    def test_unmodified_mentions_are_explicitly_unverified(self):
        text = "Cardiomegaly is present."
        entity = SimpleNamespace(label_="cardiomegaly", start_char=0, end_char=12,
            _=SimpleNamespace(**flags(), modifiers=[]))
        result = c.serialize_document(SimpleNamespace(text=text, ents=[entity]), text)
        row = result["cardiomegaly"]
        self.assertEqual(row["state"], "positive")
        self.assertTrue(row["mentions"][0]["unmodified_mention_is_not_verified_positive"])
        self.assertFalse(row["mentions"][0]["independent_clinical_validation"])

    def test_document_or_unknown_entity_changed_refused(self):
        with self.assertRaises(ValueError):
            c.serialize_document(SimpleNamespace(text="different", ents=[]), "invented")
        with self.assertRaises(ValueError):
            c.serialize_document(SimpleNamespace(text="invented", ents=[SimpleNamespace(label_="made_up")]), "invented")

    def test_modifier_cue_and_scope_preserve_character_binding(self):
        text = "Cardiomegaly is absent."
        entity = SimpleNamespace(label_="cardiomegaly", start_char=0, end_char=12,
            _=SimpleNamespace(**flags(is_negated=True), modifiers=[SimpleNamespace(
                modifier_span=(2, 3), scope_span=(0, 2), category="NEGATED_EXISTENCE", direction="BACKWARD")]))
        class Document:
            def __init__(self):
                self.text, self.ents = text, [entity]
            def __getitem__(self, span):
                offsets = {(2,3): (16,22), (0,2): (0,15)}
                start, end = offsets[span.start, span.stop]
                return SimpleNamespace(start_char=start, end_char=end)
        result = c.serialize_document(Document(), text)
        modifier = result["cardiomegaly"]["mentions"][0]["modifiers"][0]
        a.validate_span(text, modifier["cue"])
        a.validate_span(text, modifier["scope"])
        self.assertEqual(modifier["cue"]["quote"], "absent")


class CoverageTests(unittest.TestCase):
    def row(self, expected, raw, decision, gated):
        return {"expected": expected, "raw_state": raw, "decision": decision, "gated_state": gated}

    def test_abstention_cannot_inflate_overall_denominator(self):
        rows = [self.row("positive", "positive", "scope_commit", "positive"), self.row("negative", "positive", "abstain", "unknown"), self.row("unknown", "unknown", "no_model_assertion", "unknown")]
        result = g.selective_counts(rows)
        self.assertEqual(result["checks"], 3)
        self.assertEqual(result["scope_commits"], 1)
        self.assertAlmostEqual(result["commit_coverage"], 1/3)
        self.assertEqual(result["noncommitted_checks"], 2)
        self.assertEqual(result["conditional_authored_match"], 1.0)
        self.assertEqual(result["raw_correct_not_committed"], 1)

    def test_no_coverage_is_null_not_perfect(self):
        result = g.selective_counts([self.row("negative", "positive", "abstain", "unknown")])
        self.assertIsNone(result["conditional_authored_match"])
        self.assertEqual(result["commit_coverage"], 0)

    def test_empty_checks_null(self):
        result = g.selective_counts([])
        self.assertIsNone(result["conditional_authored_match"])
        self.assertIsNone(result["commit_coverage"])

    def test_counts_preserve_input_and_json_types(self):
        rows = [self.row("uncertain", "positive", "scope_commit", "positive")]
        before = copy.deepcopy(rows)
        result = g.selective_counts(rows)
        self.assertEqual(result["incorrect_commits"], 1)
        self.assertEqual(result["unsafe_unknown_or_uncertain_commits"], 1)
        self.assertEqual(rows, before)
        self.assertEqual(result, json.loads(json.dumps(result)))

    def test_blind_gate_does_not_read_authored_key(self):
        self.assertNotIn("load_references", inspect.getsource(g.gate))

    def test_nonoverwrite_before_gate(self):
        argv = ["gate", "--mode", "gate", "--bank-run", "invented", "--output-root", "invented", "--run-id", "invented"]
        with patch.object(sys, "argv", argv), patch.object(g, "new_atomic_run", side_effect=FileExistsError), patch.object(g, "gate") as gate, patch("builtins.print"):
            self.assertEqual(g.main(), 1)
            gate.assert_not_called()


class LiteralReadoutTests(unittest.TestCase):
    def read(self, text):
        return comparison.literal_readout(text, "cardiomegaly", scope_check)

    def test_source_readout_does_not_need_or_overwrite_model_state(self):
        row = self.read("Cardiomegaly is absent.")
        self.assertEqual(row["state"], "negative")
        self.assertTrue(row["scope_covered"])
        self.assertNotIn("proposed_state", row)
        self.assertFalse(row["independent_clinical_validation"])

    def test_source_positive_and_uncertain(self):
        self.assertEqual(self.read("Cardiomegaly is present.")["state"], "positive")
        self.assertEqual(self.read("Cardiomegaly may be present.")["state"], "uncertain")

    def test_missing_synonym_and_unchecked_scope_unknown(self):
        for text in ("No findings supplied.", "The heart is enlarged.", "Cardiomegaly."):
            row = self.read(text)
            self.assertEqual(row["state"], "unknown")
            self.assertFalse(row["scope_covered"])

    def test_qualified_absence_does_not_create_global_negative(self):
        self.assertEqual(self.read("No large cardiomegaly.")["state"], "uncertain")

    def test_opposed_assertions_unresolved_not_preferred_section(self):
        row = self.read("Cardiomegaly is present. Cardiomegaly is absent.")
        self.assertEqual(row["state"], "unknown")
        self.assertFalse(row["scope_covered"])

    def test_historical_not_current_assertion(self):
        self.assertEqual(self.read("History of cardiomegaly is present.")["state"], "unknown")

    def test_one_checked_mention_cannot_hide_an_unchecked_one(self):
        self.assertEqual(self.read("Cardiomegaly is present. Cardiomegaly.")["state"], "unknown")

    def test_source_comparison_does_not_read_key(self):
        source = inspect.getsource(comparison.compare)
        self.assertNotIn("load_references", source)
        self.assertNotIn("authored_cases", source)
        self.assertIn('"report_content_error_established": False', source)
        self.assertIn('"image_error_established": False', source)

    def test_deterministic_without_model_or_threshold(self):
        self.assertEqual(self.read("Cardiomegaly is absent."), self.read("Cardiomegaly is absent."))
        source = inspect.getsource(comparison.literal_readout)
        self.assertNotIn("threshold", source)
        self.assertNotIn("expected", source)


if __name__ == "__main__":
    unittest.main()
