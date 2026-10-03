"""Invented text only: no checkpoint, real data, GPU or external API."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import repair_cached_report_evidence as repair


def response(finding="pleural_effusion", polarity="positive", quotes=()):
    payload = {name: {state: [] for state in repair.POLARITIES} for name in repair.FINDINGS}
    payload[finding][polarity] = list(quotes)
    return json.dumps(payload)


def guarded(report, quote=None, finding="pleural_effusion", polarity="positive"):
    quote = report if quote is None else quote
    aligned = repair.decode_aligned(response(finding, polarity, [quote]), report)
    return repair.guard_record(aligned, report)["findings"][finding]


class WhitespaceAlignmentTests(unittest.TestCase):
    def test_exact_unicode_offsets(self):
        source = "🫁 中文. No pleural effusion."
        span = repair.align_quote(source, "No pleural effusion.")
        self.assertEqual(source[span["char_start"]:span["char_end"]], span["quote"])
        self.assertEqual(span["char_start"], 6)
        self.assertEqual(span["alignment_mode"], "exact")

    def test_whitespace_restores_original_quote_and_offsets(self):
        source = "Intro. No\t\npleural\u00a0\u00a0effusion. End."
        span = repair.align_quote(source, "No pleural effusion.")
        self.assertEqual(span["quote"], "No\t\npleural\u00a0\u00a0effusion.")
        self.assertEqual(source[span["char_start"]:span["char_end"]], span["quote"])
        self.assertEqual(span["alignment_mode"], "whitespace_equivalent")
        self.assertEqual(span["returned_quote"], "No pleural effusion.")

    def test_fold_is_whitespace_only_and_deterministic(self):
        source = " X\n\tY\u00a0Z  "
        first = repair.fold_with_offsets(source)
        self.assertEqual(first, repair.fold_with_offsets(source))
        self.assertEqual(first[0], " X Y Z ")
        for char, (start, end) in zip(*first):
            self.assertEqual(repair.fold_with_offsets(source[start:end])[0], char)

    def test_case_punctuation_units_negation_paraphrase_not_changed(self):
        source = "No pleural effusion. Tube tip is 2 cm."
        for quote in ("no pleural effusion.", "No pleural effusion!", "Pleural effusion is absent.", "Tube tip is 2 mm.", "Pleural effusion."):
            with self.subTest(quote=quote), self.assertRaises(repair.EvidenceContractError):
                repair.align_quote(source, quote)

    def test_crop_allowed_but_not_semantically_validated(self):
        span = repair.align_quote("No pleural effusion.", "pleural effusion")
        self.assertEqual(span["char_start"], 3)

    def test_exact_and_folded_repeated_locations_are_ambiguous(self):
        for text in ("No pleural effusion. No pleural effusion.", "No pleural effusion. No\tpleural effusion."):
            with self.assertRaises(repair.EvidenceContractError):
                repair.align_quote(text, "No pleural effusion.")

    def test_overlapping_locations_count(self):
        self.assertEqual(repair.occurrences("aaaa", "aaa"), [0, 1])

    def test_quote_bounds_and_types(self):
        for quote in (None, 1, "", "No", " No pleural effusion.", "x"*257):
            with self.assertRaises(repair.EvidenceContractError):
                repair.align_quote("No pleural effusion.", quote)

    def test_bad_json_and_duplicate_keys_stay_wholly_unknown(self):
        for value in ("bad", '{"cardiomegaly":{},"cardiomegaly":{}}', "[]"):
            row = repair.decode_aligned(value, "No pleural effusion.")
            self.assertEqual(row["contract_status"], "failed_unavailable")
            self.assertTrue(all(item["state"] == "unknown" for item in row["findings"].values()))

    def test_partial_inventory_is_rejected(self):
        payload = json.loads(response())
        del payload["cardiomegaly"]
        self.assertEqual(repair.decode_aligned(json.dumps(payload), "Invented report.")["contract_failure_reason"], "finding_inventory_mismatch")

    def test_normalized_duplicate_across_polarities_rejected(self):
        payload = json.loads(response(quotes=["No pleural effusion."]))
        payload["pleural_effusion"]["negative"] = ["No\tpleural effusion."]
        row = repair.decode_aligned(json.dumps(payload), "No\tpleural effusion.")
        self.assertEqual(row["contract_failure_reason"], "duplicate_or_conflicting_quote")

    def test_token_limit_is_not_repaired(self):
        row = repair.decode_aligned(response(quotes=["No pleural effusion."]), "No pleural effusion.", token_limit_reached=True)
        self.assertEqual(row["contract_failure_reason"], "token_limit_reached")

    def test_wrong_polarity_inventory_and_excess_quotes_rejected(self):
        payload = json.loads(response())
        payload["pleural_effusion"]["absent"] = []
        self.assertEqual(repair.decode_aligned(json.dumps(payload), "Invented report.")["contract_failure_reason"], "polarity_inventory_mismatch")
        payload = json.loads(response(quotes=["No pleural effusion."]*3))
        self.assertEqual(repair.decode_aligned(json.dumps(payload), "No pleural effusion.")["contract_failure_reason"], "invalid_quote_list")

    def test_absent_stays_unknown_and_explicit_uncertainty_stays_uncertain(self):
        row = repair.decode_aligned(response(), "Invented report.")
        self.assertEqual(row["findings"]["pleural_effusion"]["state"], "unknown")
        row = guarded("Possible pleural effusion.", polarity="uncertain")
        self.assertEqual(row["state"], "uncertain")


class ScopeVetoTests(unittest.TestCase):
    def test_positive_negated_quote_abstains_never_flips(self):
        row = guarded("No pleural effusion.")
        self.assertEqual(row["state"], "unknown")
        self.assertEqual(row["evidence"]["negative"], [])
        self.assertEqual(row["vetoed_evidence"][0]["scope_guard"]["veto_reason"], "positive_quote_explicitly_negated")

    def test_cropped_quote_checks_negation_in_original_source(self):
        row = guarded("No pleural effusion.", quote="pleural effusion")
        self.assertEqual(row["state"], "unknown")

    def test_post_negation(self):
        row = guarded("Pleural effusion is not seen.")
        self.assertEqual(row["state"], "unknown")

    def test_negative_with_presence_abstains(self):
        row = guarded("Pleural effusion is present.", polarity="negative")
        self.assertEqual(row["state"], "unknown")
        self.assertEqual(row["evidence"]["positive"], [])

    def test_shared_negated_disease_list(self):
        text = "No pleural effusion or pneumothorax."
        for finding, quote in (("pleural_effusion", "pleural effusion"), ("pneumothorax", "pneumothorax")):
            row = guarded(text, quote, finding)
            self.assertEqual(row["state"], "unknown")
            self.assertEqual(row["vetoed_evidence"][0]["scope_guard"]["rule_suggested_states"], ["negative"])

    def test_other_disease_negation_cannot_cross_sentence_or_contrast(self):
        for text in ("No pneumothorax. Pleural effusion is present.", "No pneumothorax, but pleural effusion is present.", "No pneumothorax; pleural effusion is present."):
            quote = "Pleural effusion is present." if ". Pleural" in text else "pleural effusion is present."
            self.assertEqual(guarded(text, quote)["state"], "positive")

    def test_negation_cannot_cross_independent_copula(self):
        text = "No pneumothorax, and pleural effusion is present."
        self.assertEqual(guarded(text, "pleural effusion is present.")["state"], "positive")

    def test_original_line_boundary_not_erased_by_alignment_folding(self):
        text = "No pneumothorax\nPleural effusion is present."
        self.assertEqual(guarded(text, "Pleural effusion is present.")["state"], "positive")
        text = "No large pleural effusion\nPneumothorax is present."
        self.assertEqual(guarded(text, "Pneumothorax is present.", "pneumothorax")["state"], "positive")

    def test_wrap_line_does_not_erase_shared_negation(self):
        for text in ("No\npleural effusion is present.", "No pneumothorax or\npleural effusion is present."):
            self.assertEqual(guarded(text, "pleural effusion is present.")["state"], "unknown")

    def test_new_line_does_not_skip_own_negation(self):
        text = "No pneumothorax\nNo pleural effusion is present."
        self.assertEqual(guarded(text, "pleural effusion is present.")["state"], "unknown")
        text = "No pneumothorax is seen and pleural effusion is present."
        self.assertEqual(guarded(text, "pleural effusion is present.")["state"], "positive")

    def test_qualified_absence_not_global_negative(self):
        for word in ("large", "significant", "left"):
            text = f"No {word} pleural effusion."
            self.assertEqual(guarded(text, polarity="negative")["state"], "unknown")
            self.assertEqual(guarded(text, polarity="uncertain")["state"], "uncertain")

    def test_double_negative_and_uncertainty_not_absence(self):
        for text in ("Cannot exclude pleural effusion.", "Pleural effusion is not excluded.", "Possible pleural effusion."):
            for polarity in ("positive", "negative"):
                self.assertEqual(guarded(text, polarity=polarity)["state"], "unknown")

    def test_pseudo_negation_not_flipped_or_claimed_checked(self):
        for text in ("No change in pleural effusion.", "No increase in pleural effusion."):
            row = guarded(text)
            self.assertEqual(row["state"], "positive")
            self.assertEqual(row["evidence"]["positive"][0]["scope_guard"]["rule_suggested_states"], [])

    def test_history_or_indication_is_not_current_evidence(self):
        for text in ("History of pleural effusion.", "Evaluate for pleural effusion."):
            self.assertEqual(guarded(text)["state"], "unknown")

    def test_nonpleural_effusion_is_not_pleural(self):
        self.assertEqual(guarded("Pericardial effusion is present.")["state"], "unknown")

    def test_uncovered_synonyms_explicitly_unchecked(self):
        row = guarded("Heart silhouette is enlarged.", finding="cardiomegaly")
        self.assertEqual(row["state"], "positive")
        self.assertEqual(row["evidence"]["positive"][0]["scope_guard"]["scope_reasons"], ["literal_finding_not_covered"])
        self.assertFalse(row["semantic_correctness_independently_verified"])

    def test_no_cross_finding_inference(self):
        row = guarded("No pneumothorax.", finding="pleural_effusion", polarity="negative")
        self.assertEqual(row["evidence"]["negative"][0]["scope_guard"]["literal_mentions_checked"], 0)
        self.assertFalse(row["semantic_correctness_independently_verified"])

    def test_original_record_and_source_not_modified(self):
        text = "No\tpleural effusion."
        aligned = repair.decode_aligned(response(quotes=["No pleural effusion."]), text)
        before = copy.deepcopy(aligned)
        first = repair.guard_record(aligned, text)
        self.assertEqual(aligned, before)
        self.assertEqual(first, repair.guard_record(aligned, text))
        self.assertEqual(text, "No\tpleural effusion.")

    def test_veto_does_not_resolve_conflict_into_opposite_state(self):
        text = "No pleural effusion. Pleural effusion is not seen."
        payload = json.loads(response(quotes=["No pleural effusion."]))
        payload["pleural_effusion"]["negative"] = ["Pleural effusion is not seen."]
        aligned = repair.decode_aligned(json.dumps(payload), text)
        row = repair.guard_record(aligned, text)["findings"]["pleural_effusion"]
        self.assertEqual(row["state_before_guard"], "uncertain")
        self.assertEqual(row["state"], "unknown")
        self.assertTrue(row["unresolved_veto_prevents_determinate_state"])
        self.assertEqual(len(row["evidence"]["negative"]), 1)

    def test_opposition_veto_does_not_manufacture_negative_winner(self):
        text = "No pleural effusion. Pleural effusion is present."
        payload = json.loads(response(quotes=["No pleural effusion."]))
        payload["pleural_effusion"]["negative"] = ["pleural effusion"]
        # Unique source fragments avoid ambiguous locations.
        payload["pleural_effusion"]["negative"] = ["Pleural effusion is present."]
        row = repair.guard_record(repair.decode_aligned(json.dumps(payload), text), text)["findings"]["pleural_effusion"]
        self.assertEqual(row["state"], "unknown")
        self.assertEqual(len(row["vetoed_evidence"]), 2)

    def test_forged_offsets_are_rejected(self):
        text = "No pleural effusion."
        span = repair.align_quote(text, text)
        span["char_start"] = 1
        with self.assertRaises(ValueError):
            repair.scope_check(text, span, "pleural_effusion", "positive")

    def test_failed_records_retain_denominator_and_unknown(self):
        rows = [repair.decode_aligned("bad", "Invented report."), repair.decode_aligned(response(), "Invented report.")]
        summary = repair.summarize_stage(rows)
        self.assertEqual(summary["distinct_texts"], 2)
        self.assertEqual(summary["finding_record_denominator"], 8)
        self.assertEqual(summary["states_not_unknown"], 0)
        self.assertEqual(summary["failed_unavailable"], 1)


class PipelineSeparationTests(unittest.TestCase):
    def test_key_reader_runs_after_blinded_guard_pass(self):
        from types import SimpleNamespace
        text = "No pleural effusion."
        raw = response(quotes=[text])
        strict = repair.decode_evidence(raw, text)
        visited = []
        def guard(aligned, source):
            visited.append("guard")
            return repair.guard_record_original(aligned, source)
        def analyzer(args):
            self.assertEqual(visited, ["guard"])
            raise RuntimeError("invented evaluator stop")
        repair.guard_record_original = repair.guard_record
        try:
            with patch.object(repair, "load_blinded_cache", return_value=({"h": text}, [], {"h": strict}, {"h": {"response": raw, "token_limit_reached": False}}, {}, {})), patch.object(repair, "guard_record", side_effect=guard), patch.object(repair, "analyze_strict", side_effect=analyzer):
                with self.assertRaisesRegex(RuntimeError, "invented evaluator stop"):
                    repair.run(SimpleNamespace())
        finally:
            del repair.guard_record_original

    def test_nonoverwrite_refusal_precedes_any_cache_read(self):
        args = ["repair", "--bank-run", "b", "--review-run", "r", "--context-run", "c", "--output-root", "o", "--run-id", "i"]
        with patch.object(sys, "argv", args), patch.object(repair, "new_atomic_run", side_effect=FileExistsError), patch.object(repair, "run") as run, patch("builtins.print"):
            self.assertEqual(repair.main(), 1)
            run.assert_not_called()

    def test_format_controls_keep_unavailable_and_changed_denominators(self):
        key = [
            {"case_id": "invented", "item_id": "old", "intervention_type": "unchanged"},
            {"case_id": "invented", "item_id": "ws", "intervention_type": "whitespace_only"},
        ]
        resolver = [{"item_id": "old", "report_sha256": "a"}, {"item_id": "ws", "report_sha256": "b"}]
        empty = repair.decode_aligned(response(), "Invented report.")
        failed = repair.decode_aligned("bad", "Invented report.")
        result = repair.format_controls(key, resolver, {"a": empty, "b": failed})
        self.assertEqual(result["case_linked_controls"], 1)
        self.assertEqual(result["unavailable"], 1)
        self.assertEqual(result["comparable"], 0)
        self.assertEqual(result["state_vectors_changed"], 0)
        present = repair.decode_aligned(response(quotes=["Pleural effusion is present."]), "Pleural effusion is present.")
        result = repair.format_controls(key, resolver, {"a": empty, "b": present})
        self.assertEqual(result["comparable"], 1)
        self.assertEqual(result["state_vectors_changed"], 1)


if __name__ == "__main__":
    unittest.main()
