"""Wholly invented documents only; no acquired clinical annotation payloads."""
from copy import deepcopy
import json
import unittest

from tricompose_v12.cxrgraph_gold_adapter import (
    ATTRIBUTES, CORE_LABELS, LABELS, POLICY, SOURCE,
    compare_to_native_prediction, normalize_manual_document, validate_adapter,
)
from tricompose_v12.entity_gold_contract import digest, normalize_graph

SOURCE_HASH = digest(['authored_raw_source'])


def document():
    return {'doc_key': 'authored_private_source_key',
        'sentences': [['authored_finding', '.'], ['authored_anatomy', 'authored_modifier', '.']],
        'ner': [[[0, 0, 'Observation-Present']], [[2, 2, 'Anatomy'], [3, 3, 'Observation-Present']]],
        'relations': [[[0, 0, 2, 2, 'located_at']], [[3, 3, 2, 2, 'located_at']]],
        'entity_attributes': [[[0, 0, 'Abnormal', 'Essential', 'Unchanged']],
                              [[3, 3, 'NA', 'Removable', 'Negative']]]}


def normalized(doc=None, **extra):
    return normalize_manual_document(document() if doc is None else doc,
        report_id='report_0000', source_report_sha256=SOURCE_HASH, origin='authored_fixture', **extra)


def reseal(value):
    value['adapter_sha256'] = digest({k: v for k, v in value.items() if k != 'adapter_sha256'})
    return value


class CXRGraphGoldAdapterTests(unittest.TestCase):
    def test_manual_spans_are_global_inclusive_then_end_exclusive(self):
        a = normalized()
        self.assertEqual(a['sentence_bounds'], [[0, 2], [2, 5]])
        self.assertEqual(a['native_entities'][1], [2, 3, 'Anatomy'])
        self.assertEqual(a['common_label_record']['entities'][1], [2, 3, 'Anatomy::definitely present'])
        validate_adapter(a)

    def test_sentence_local_offsets_fail_not_silently_shifted(self):
        d = document()
        d['ner'][1][0][0:2] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'document_global_sentence_bound_span_required'):
            normalized(d)

    def test_cross_sentence_relation_preserved_not_truncated(self):
        a = normalized()
        self.assertEqual(a['native_coverage']['cross_sentence_relations_preserved'], 1)
        self.assertEqual(len(a['common_label_record']['relations']), 2)

    def test_assigned_attributes_and_unknown_slots_preserved(self):
        a = normalized()
        self.assertEqual(a['native_coverage']['assigned_attributes_by_dimension'],
            {'normality': 1, 'action': 2, 'change': 2})
        self.assertEqual(a['native_coverage']['unassigned_attribute_slots_not_imputed'], 4)
        self.assertTrue(all(v is None for v in a['native_extra_dimension_scores'].values()))

    def test_positive_change_never_changes_negative_finding_polarity(self):
        d = document()
        d['ner'][0][0][2] = 'Observation-Absent'
        d['entity_attributes'][0][0][4] = 'Positive'
        a = normalized(d)
        self.assertEqual(a['common_label_record']['entities'][0][2], 'Observation::definitely absent')
        self.assertIn([0, 1, 'Observation-Absent', 'change', 'Positive'], a['native_attributes'])

    def test_normality_does_not_change_polarity_or_impute_disease(self):
        d = document()
        d['entity_attributes'][0][0][2] = 'Normal'
        a = normalized(d)
        self.assertEqual(a['common_label_record']['entities'][0][2], 'Observation::definitely present')
        self.assertFalse(a['policy']['normality_defaults_imputed'])

    def test_absent_rows_and_all_na_rows_have_identical_unknown_metadata(self):
        a, b = document(), document()
        a['entity_attributes'] = [[], []]
        b['entity_attributes'] = [[[0, 0, 'NA', 'NA', 'NA']], []]
        self.assertEqual(normalized(a), normalized(b))

    def test_unassigned_normality_not_assumed_abnormal(self):
        d = document()
        d['entity_attributes'] = [[], []]
        a = normalized(d)
        self.assertEqual(a['native_attributes'], [])
        self.assertEqual(a['native_coverage']['unassigned_attribute_slots_not_imputed'], 9)

    def test_location_attribute_chain_not_flattened(self):
        d = document()
        d['ner'][1][1][2] = 'Location-Attribute'
        d['relations'] = [[[0, 0, 3, 3, 'located_at']], [[3, 3, 2, 2, 'located_at']]]
        d['entity_attributes'] = [[], []]
        a = normalized(d)
        self.assertEqual(len(a['native_relations']), 2)
        self.assertEqual(a['common_label_record']['relations'], [])
        self.assertEqual(a['native_coverage']['location_attribute_entities_unmapped'], 1)
        self.assertEqual(a['native_coverage']['relations_touching_location_attribute_unmapped'], 2)
        self.assertIsNone(a['native_extra_dimension_scores']['location_attribute'])

    def test_part_of_preserved_not_rewritten_as_modify(self):
        d = document()
        d['relations'][1][0][4] = 'part_of'
        a = normalized(d)
        self.assertEqual(a['native_coverage']['part_of_relations_unmapped'], 1)
        self.assertEqual(len(a['common_label_record']['relations']), 1)
        self.assertIn('part_of', [r[0] for r in a['native_relations']])

    def test_all_core_entity_names_map_only_to_exact_native_labels(self):
        for label in CORE_LABELS:
            d = document()
            d['ner'][0][0][2] = label
            normalized(d)
        self.assertEqual(len(LABELS), 5)
        self.assertEqual(SOURCE['revision'], '4b0edaf75d18128cbccfaf90ff92549984600056')

    def test_model_prediction_keys_not_treated_as_manual_gold(self):
        for key in ('pred_ner', 'pred_rel', 'pred_attr'):
            d = document()
            d[key] = []
            with self.assertRaisesRegex(ValueError, 'exact_manual_document_schema_required'):
                normalized(d)

    def test_missing_required_field_rejected(self):
        for key in document():
            d = document()
            del d[key]
            with self.assertRaises(ValueError):
                normalized(d)

    def test_empty_or_whitespace_tokens_not_retokenized(self):
        for value in ('', 'two words', ' tab\tword', '\n'):
            d = document()
            d['sentences'][0][0] = value
            with self.assertRaises(ValueError):
                normalized(d)

    def test_missing_or_empty_sentence_not_invented(self):
        for sentences in ([], [[]], ['authored_finding']):
            d = document()
            d['sentences'] = sentences
            with self.assertRaises(ValueError):
                normalized(d)

    def test_total_tokens_or_characters_rejected_before_document_join(self):
        for sentences in ((['a'] * 15000, ['b'] * 15000),
                          (['x' * 10000] * 6, ['y' * 10000] * 6)):
            d = document()
            d['sentences'] = list(sentences)
            with self.assertRaisesRegex(ValueError, 'bounded_manual_document_required'):
                normalized(d)

    def test_sentence_annotation_inventory_must_match(self):
        for key in ('ner', 'relations', 'entity_attributes'):
            d = document()
            d[key] = d[key][:1]
            with self.assertRaises(ValueError):
                normalized(d)

    def test_boolean_bad_or_cross_sentence_entity_offsets_rejected(self):
        for span in ((True, 0), (-1, 0), (0, 8), (0, 2), (1, 0)):
            d = document()
            d['ner'][0][0][:2] = span
            with self.assertRaises(ValueError):
                normalized(d)

    def test_unknown_entity_and_classifier_null_class_fail_closed(self):
        for label in ('X', 'OBS-DP', 'authored_unknown'):
            d = document()
            d['ner'][0][0][2] = label
            with self.assertRaises(ValueError):
                normalized(d)

    def test_inverse_classifier_relation_tags_not_silently_reversed(self):
        for label in ('of_part', 'inverse_modify', 'X', 'authored_unknown'):
            d = document()
            d['relations'][0][0][4] = label
            with self.assertRaises(ValueError):
                normalized(d)

    def test_relation_must_bind_both_entities(self):
        for head_tail in ((1, 1, 2, 2), (0, 0, 4, 4), (0, 0, True, 2)):
            d = document()
            d['relations'][0][0][:4] = head_tail
            with self.assertRaises(ValueError):
                normalized(d)

    def test_relation_subject_must_match_containing_sentence(self):
        d = document()
        d['relations'][0][0][:4] = [3, 3, 2, 2]
        with self.assertRaises(ValueError):
            normalized(d)

    def test_duplicate_or_conflicting_spans_rejected(self):
        d = document()
        d['ner'][0].append([0, 0, 'Observation-Absent'])
        with self.assertRaises(ValueError):
            normalized(d)

    def test_duplicate_relations_rejected(self):
        d = document()
        d['relations'][0].append(d['relations'][0][0])
        with self.assertRaises(ValueError):
            normalized(d)

    def test_attributes_bind_to_known_unique_entity(self):
        for change in ('unknown_span', 'duplicate'):
            d = document()
            if change == 'unknown_span':
                d['entity_attributes'][0][0][:2] = [1, 1]
            else:
                d['entity_attributes'][0].append(d['entity_attributes'][0][0])
            with self.assertRaises(ValueError):
                normalized(d)

    def test_attribute_domains_not_interchangeable(self):
        for index, value in ((2, 'Positive'), (3, 'Normal'), (4, 'Essential'), (4, 'X'), (2, None)):
            d = document()
            d['entity_attributes'][0][0][index] = value
            with self.assertRaises(ValueError):
                normalized(d)
        self.assertEqual(len(ATTRIBUTES), 3)

    def test_entity_and_relation_order_does_not_change_result(self):
        d = document()
        d['ner'][1].reverse()
        self.assertEqual(normalized(d), normalized())

    def test_source_key_and_report_words_never_returned(self):
        text = json.dumps(normalized())
        for value in ('authored_private_source_key', 'authored_finding', 'authored_anatomy', 'authored_modifier', 'doc_key', 'sentences'):
            self.assertNotIn(value, text)

    def test_private_values_never_echoed_by_errors(self):
        secret = 'authored_secret_text'
        d = document()
        d['ner'][0][0][2] = secret
        with self.assertRaises(ValueError) as cm:
            normalized(d)
        self.assertNotIn(secret, str(cm.exception))

    def test_joint_annotation_not_misrepresented_as_two_reader_votes(self):
        a = normalized()
        self.assertIn('not_two_independent_readers', a['annotation_process'])
        self.assertEqual(a['common_label_record']['reader_id'], 'reader_01')

    def test_human_provenance_or_performance_not_verified_by_schema(self):
        a = normalized()
        self.assertEqual(a['policy'], POLICY)
        self.assertTrue(all(value is False for value in a['policy'].values()))
        self.assertIsNone(a['clinical_score'])

    def test_frozen_prediction_origin_cannot_claim_manual_gold(self):
        with self.assertRaisesRegex(ValueError, 'manual_gold_origin_required'):
            normalize_manual_document(document(), report_id='report_0000',
                source_report_sha256=SOURCE_HASH, origin='declared_frozen_prediction')

    def test_changed_adapter_rejected(self):
        a = normalized()
        a['native_coverage']['part_of_relations_unmapped'] = 10
        with self.assertRaises(ValueError):
            validate_adapter(a)

    def test_resealed_invented_projection_or_scope_scores_rejected(self):
        for key in ('common_label_record', 'native_extra_dimension_scores'):
            a = normalized()
            if key == 'common_label_record':
                a[key]['entities'][0][2] = 'Observation::definitely absent'
                a[key]['record_sha256'] = digest({k: v for k, v in a[key].items() if k != 'record_sha256'})
            else:
                a[key]['change'] = 1.
            reseal(a)
            with self.assertRaises(ValueError):
                validate_adapter(a)

    def test_resealed_native_entity_attribute_or_sentence_tampering_rejected(self):
        for key in ('native_entities', 'native_attributes', 'sentence_bounds'):
            a = normalized()
            if key == 'native_entities':
                a[key][0][0] = True
            elif key == 'native_attributes':
                a[key][0][2] = 'Location-Attribute'
            else:
                a[key][1][0] = 3
            reseal(a)
            with self.assertRaises(ValueError):
                validate_adapter(a)

    def test_native_prediction_core_comparison_keeps_extra_attributes_null(self):
        d = document()
        a = normalized(d)
        g = {'text': ' '.join(w for s in d['sentences'] for w in s), 'entities': {
            '1': {'tokens': 'authored_finding', 'start_ix': 0, 'end_ix': 0,
                  'label': 'Observation::definitely present', 'relations': [['located_at', '2']]},
            '2': {'tokens': 'authored_anatomy', 'start_ix': 2, 'end_ix': 2,
                  'label': 'Anatomy::definitely present', 'relations': []},
            '3': {'tokens': 'authored_modifier', 'start_ix': 3, 'end_ix': 3,
                  'label': 'Observation::definitely present', 'relations': [['located_at', '2']]}}}
        p = normalize_graph(g, native_schema='radgraph_xl', report_id='report_0000',
            source_report_sha256=SOURCE_HASH, origin='authored_fixture')
        r = compare_to_native_prediction(a, p)
        self.assertEqual(r['common_comparison']['entity_metrics_in_gold_label_scope']['f1'], 1.)
        self.assertEqual(r['common_comparison']['relation_metrics_in_gold_label_scope']['f1'], 1.)
        self.assertIsNone(r['native_extra_dimension_scores']['change'])
        self.assertIsNone(r['clinical_score'])


if __name__ == '__main__':
    unittest.main()
