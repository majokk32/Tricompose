"""Invented fixtures only; no model loading, real data or inference."""
import argparse
import copy
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import diagnose_chexbert_polarity as diagnostic
from prepare_expanded_polarity import prepare_records


def fixture():
    cases = [{"case_id": "invented_a", "ehr_sha256": "a"*64},
             {"case_id": "invented_b", "ehr_sha256": "b"*64}]
    texts = {"invented_a": "Findings: Cardiomegaly is present. No pneumothorax.",
             "invented_b": "Findings: No pleural effusion."}
    records, _ = prepare_records(cases, texts)
    key, by_id = [], {}
    for index, row in enumerate(records):
        item = f"fixture_{index}"
        key.append({"case_id": row["case"]["case_id"], "item_id": item, "finding": row["finding"],
                    "intervention_type": row["kind"], "edit": row["edit"]})
        by_id[item] = row["text"]
    return key, by_id


def invented_predictions(pairs):
    full, sentences = {}, {}
    for pair in pairs:
        for mode, fingerprint, state in (
            (full, pair["original_report_sha256"], pair["expected_source_text_state"]),
            (full, pair["edited_report_sha256"], pair["expected_edited_text_state"]),
            (sentences, pair["original_sentence_sha256"], pair["expected_source_text_state"]),
            (sentences, pair["edited_sentence_sha256"], pair["expected_edited_text_state"]),
        ):
            mode.setdefault(fingerprint, {})[pair["finding"]] = {"state": state, "top_two_raw_logit_margin": 0.1}
    return full, sentences


class CheXbertPolarityDiagnosisTests(unittest.TestCase):
    def test_normalization_matches_documented_csv_contract(self):
        self.assertEqual(diagnostic.normalize_official_csv("  No\n\tpneumothorax.  "), "No pneumothorax.")
        self.assertEqual(diagnostic.normalize_wrapper("  No\n\tpneumothorax.  "), "No \tpneumothorax.")

    def test_bindings_reconstruct_sources_and_do_not_expose_text(self):
        key, texts = fixture()
        before = copy.deepcopy((key, texts))
        sentences, pairs = diagnostic.sentence_bindings(key, texts)
        self.assertEqual((key, texts), before)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(len(value) == 64 for pair in pairs for name, value in pair.items() if name.endswith("sha256")))
        self.assertTrue(all("source_statement" not in pair and "replacement_statement" not in pair for pair in pairs))
        self.assertTrue(sentences)

    def test_changed_unrelated_bytes_are_refused(self):
        key, texts = fixture()
        edited = next(row for row in key if row["intervention_type"] == "minimal_polarity_flip")
        texts[edited["item_id"]] += " Unrelated invented text."
        with self.assertRaisesRegex(ValueError, "unrelated bytes"):
            diagnostic.sentence_bindings(key, texts)

    def test_invalid_span_and_unsupported_polarity_are_refused(self):
        key, texts = fixture()
        edited = next(row for row in key if row["intervention_type"] == "minimal_polarity_flip")
        edited["edit"]["span_start"] = -1
        with self.assertRaisesRegex(ValueError, "span"):
            diagnostic.sentence_bindings(key, texts)
        key, texts = fixture()
        edited = next(row for row in key if row["intervention_type"] == "minimal_polarity_flip")
        edited["edit"]["edited_text_assertion"] = "unknown"
        with self.assertRaisesRegex(ValueError, "polarity"):
            diagnostic.sentence_bindings(key, texts)

    def test_duplicate_control_and_intervention_are_refused(self):
        key, texts = fixture()
        control = next(row for row in key if row["intervention_type"] == "unchanged")
        with self.assertRaisesRegex(ValueError, "duplicate unchanged"):
            diagnostic.sentence_bindings(key+[control], texts)
        edited = next(row for row in key if row["intervention_type"] == "minimal_polarity_flip")
        with self.assertRaisesRegex(ValueError, "duplicate case/finding"):
            diagnostic.sentence_bindings(key+[edited], texts)

    def test_raw_logits_use_frozen_order_and_four_states(self):
        logits = [[0., 3., 1., 2.] for _ in range(13)] + [[2., 1.]]
        heads = diagnostic.decode_heads(logits)
        self.assertEqual(list(heads), list(diagnostic.CHEXBERT_ORDER))
        self.assertEqual(heads["pleural_effusion"]["state"], "positive")
        self.assertEqual(heads["pleural_effusion"]["top_two_raw_logit_margin"], 1.)
        self.assertEqual(heads["no_finding"]["state"], "unknown")

    def test_invalid_head_width_or_nonfinite_logits_are_refused(self):
        logits = [[0., 1., 2., 3.] for _ in range(13)] + [[0., 1.]]
        with self.assertRaisesRegex(ValueError, "inventory"):
            diagnostic.decode_heads(logits[:-1])
        logits[0][0] = math.nan
        with self.assertRaisesRegex(ValueError, "invalid diagnostic logits"):
            diagnostic.decode_heads(logits)

    def test_context_sensitivity_is_not_clinical_fault_localization(self):
        _, pairs = diagnostic.sentence_bindings(*fixture())
        full, sentences = invented_predictions(pairs)
        pair = next(row for row in pairs if row["finding"] == "pneumothorax")
        full[pair["edited_report_sha256"]]["pneumothorax"]["state"] = "negative"
        summary, details = diagnostic.diagnose_pairs(pairs, full, sentences)
        self.assertEqual(summary["pneumothorax"]["case_linked_counts"]["full_edited_mismatch_sentence_match"], 1)
        self.assertTrue(all(row["clinical_mismatch_verified"] is False for row in details))

    def test_unknown_sentence_is_not_success_or_negative(self):
        _, pairs = diagnostic.sentence_bindings(*fixture())
        full, sentences = invented_predictions(pairs)
        pair = next(row for row in pairs if row["finding"] == "pneumothorax")
        full[pair["edited_report_sha256"]]["pneumothorax"]["state"] = "negative"
        sentences[pair["edited_sentence_sha256"]]["pneumothorax"]["state"] = "unknown"
        summary, details = diagnostic.diagnose_pairs(pairs, full, sentences)
        counts = summary["pneumothorax"]["case_linked_counts"]
        self.assertEqual(counts["sentence_both_text_states_extracted"], 0)
        self.assertEqual(counts["edited_mismatch_both_contexts"], 1)
        self.assertEqual(next(row for row in details if row["finding"] == "pneumothorax")["sentence_edited_state"], "unknown")

    def test_full_success_is_not_replaced_by_sentence_failure(self):
        _, pairs = diagnostic.sentence_bindings(*fixture())
        full, sentences = invented_predictions(pairs)
        pair = next(row for row in pairs if row["finding"] == "pneumothorax")
        sentences[pair["edited_sentence_sha256"]]["pneumothorax"]["state"] = "unknown"
        summary, _ = diagnostic.diagnose_pairs(pairs, full, sentences)
        counts = summary["pneumothorax"]["case_linked_counts"]
        self.assertEqual(counts["full_both_text_states_extracted"], 1)
        self.assertEqual(counts["sentence_both_text_states_extracted"], 0)

    def test_duplicate_text_pairs_have_separate_denominators(self):
        _, pairs = diagnostic.sentence_bindings(*fixture())
        full, sentences = invented_predictions(pairs)
        pair = next(row for row in pairs if row["finding"] == "pneumothorax")
        summary, _ = diagnostic.diagnose_pairs(pairs+[dict(pair, case_id="invented_duplicate")], full, sentences)
        self.assertEqual(summary["pneumothorax"]["case_linked_pairs"], 2)
        self.assertEqual(summary["pneumothorax"]["unique_text_pairs"], 1)

    def test_login_guard_precedes_model_import_and_bank_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(diagnostic, "load_expanded_bank") as loader:
            with self.assertRaisesRegex(RuntimeError, "approved Slurm"):
                diagnostic.run(argparse.Namespace())
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
