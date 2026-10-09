"""Invented string/offset probes; no patients or model calls."""
import hashlib
import unittest

from tricompose_v12.cxrgraph_gold_adapter import normalize_manual_document
from tricompose_v12.entity_gold_contract import normalize_graph
from tricompose_v12.entity_gold_character_contract import (
    aggregate_character_comparisons, compare_character_annotation, token_bounds,
)


def probe(short_head=False, state='OBS-DP', empty=False):
    words = ['left-sided', 'lung']
    native = ['left', '-', 'sided', 'lung']
    sha = hashlib.sha256(' '.join(words).encode()).hexdigest()
    raw = {'doc_key': 'authored_only', 'sentences': [words],
        'ner': [[[0, 0, 'Observation-Present'], [1, 1, 'Anatomy']]],
        'relations': [[[0, 0, 1, 1, 'located_at']]], 'entity_attributes': [[]]}
    entities = {'1': {'start_ix': 0, 'end_ix': 0 if short_head else 2,
        'label': state, 'tokens': 'left' if short_head else 'left - sided',
        'relations': [['located_at', '2']]},
        '2': {'start_ix': 3, 'end_ix': 3, 'label': 'ANAT-DP',
              'tokens': 'lung', 'relations': []}}
    if empty:
        raw['ner'], raw['relations'], entities = [[]], [[]], {}
    gold = normalize_manual_document(raw, report_id='report_0000',
        source_report_sha256=sha, origin='authored_fixture')
    prediction = normalize_graph({'text': ' '.join(native), 'entities': entities},
        native_schema='radgraph_v1', report_id='report_0000',
        source_report_sha256=sha, origin='authored_fixture')
    return gold, prediction, words, native


def compare(**kwargs):
    gold, prediction, words, native = probe(**kwargs)
    return compare_character_annotation(gold, prediction,
        source_words=words, prediction_words=native)


class CharacterGoldTests(unittest.TestCase):
    def test_split_punctuation_has_exact_source_offsets(self):
        original, native = token_bounds(['left-sided', 'lung'], ['left', '-', 'sided', 'lung'])
        self.assertEqual(original, [[0, 10], [11, 15]])
        self.assertEqual(native, [[0, 4], [4, 5], [5, 10], [11, 15]])

    def test_identical_tokens(self):
        self.assertEqual(*token_bounds(['a', 'b'], ['a', 'b']))

    def test_modified_characters_rejected(self):
        with self.assertRaises(ValueError):
            token_bounds(['left'], ['right'])

    def test_changed_case_rejected(self):
        with self.assertRaises(ValueError):
            token_bounds(['LEFT'], ['left'])

    def test_deleted_punctuation_rejected(self):
        with self.assertRaises(ValueError):
            token_bounds(['a-b'], ['a', 'b'])

    def test_partial_source_consumption_rejected(self):
        with self.assertRaises(ValueError):
            token_bounds(['a-b'], ['a'])

    def test_merge_across_original_whitespace_rejected(self):
        with self.assertRaises(ValueError):
            token_bounds(['a', 'b'], ['ab'])

    def test_model_tokens_cannot_contain_whitespace(self):
        with self.assertRaises(ValueError):
            token_bounds(['a', 'b'], ['a b'])

    def test_lossless_split_scores_exact_entities_and_relations(self):
        row = compare()
        self.assertEqual(row['common_comparison']['entity_metrics_in_gold_label_scope']['f1'], 1)
        self.assertEqual(row['common_comparison']['relation_metrics_in_gold_label_scope']['f1'], 1)
        self.assertEqual(row['gold_character_entities'], row['prediction_character_entities'])
        self.assertFalse(row['token_boundaries_identical'])

    def test_partial_entity_is_not_snapped_to_gold_boundary(self):
        row = compare(short_head=True)['common_comparison']
        self.assertEqual(row['entity_metrics_in_gold_label_scope']['tp'], 1)
        self.assertEqual(row['entity_metrics_in_gold_label_scope']['fp'], 1)
        self.assertEqual(row['entity_metrics_in_gold_label_scope']['fn'], 1)
        self.assertEqual(row['relation_metrics_in_gold_label_scope']['tp'], 0)
        self.assertEqual(row['polarity']['matched_observation_spans'], 0)

    def test_polarity_conflict_at_exact_span_remains_error(self):
        row = compare(state='OBS-DA')['common_comparison']
        self.assertEqual(row['polarity']['matched_observation_spans'], 1)
        self.assertEqual(row['polarity']['confusion_on_matched_observation_spans']['positive']['negative'], 1)

    def test_wrong_native_inventory_rejected(self):
        gold, prediction, words, _ = probe()
        with self.assertRaises(ValueError):
            compare_character_annotation(gold, prediction,
                source_words=words, prediction_words=['other'])

    def test_deterministic_and_text_free(self):
        self.assertEqual(compare(), compare())
        row = compare()
        self.assertNotIn('text', row)
        self.assertNotIn('tokens', row)
        self.assertEqual(row['offset_unit'], 'exact_source_character_end_exclusive')
        self.assertIsNone(row['clinical_score'])
        self.assertFalse(row['full_official_cxrgraph_metric'])

    def test_empty_entity_predictions_are_not_perfect(self):
        row = compare(empty=True)['common_comparison']
        self.assertIsNone(row['entity_metrics_in_gold_label_scope']['f1'])
        self.assertIsNone(row['relation_metrics_in_gold_label_scope']['f1'])

    def test_unavailable_reports_preserve_denominator_and_domain(self):
        cohort = [{'report_id': 'report_0000', 'source_domain': 'mimic'},
                  {'report_id': 'report_0001', 'source_domain': 'chexpert'}]
        result = aggregate_character_comparisons([compare()], cohort)
        self.assertEqual(result['all']['comparison_availability'], 0.5)
        self.assertIsNone(result['chexpert']['entity_micro'])
        self.assertEqual(result['chexpert']['unavailable_report_ids'], ['report_0001'])

    def test_duplicate_comparison_rejected(self):
        cohort = [{'report_id': 'report_0000', 'source_domain': 'mimic'},
                  {'report_id': 'report_0001', 'source_domain': 'chexpert'}]
        row = compare()
        with self.assertRaises(ValueError):
            aggregate_character_comparisons([row, row], cohort)


if __name__ == '__main__':
    unittest.main()
