"""All report words and graphs below are wholly investigator-authored fixtures."""
from copy import deepcopy
import json
import unittest

from tricompose_v12.entity_gold_contract import (
    POLICY, V1_LABELS, aggregate_reader, compare_annotations, count_metrics, digest,
    normalize_graph, validate_record,
)
from tricompose_v12.radgraph_reference_contract_v2 import NATIVE_LABELS

TEXT = 'authored_finding authored_anatomy authored_other .'
SOURCE = digest(['invented_source_only'])


def graph(specs=(), *, text=TEXT):
    tokens = text.split()
    return {'text': text, 'entities': {str(i + 1): {'start_ix': start, 'end_ix': end,
        'label': label, 'tokens': ' '.join(tokens[start:end + 1]), 'relations': relations}
        for i, (start, end, label, relations) in enumerate(specs)}}


def normalized(source, *, gold=False, schema='radgraph_v1', reader='reader_01', **extra):
    return normalize_graph(source, native_schema=schema, report_id='report_0000',
        source_report_sha256=SOURCE, origin='authored_fixture', reader_id=reader if gold else None,
        **extra)


def score(gold, prediction, **kwargs):
    return compare_annotations(normalized(gold, gold=True), normalized(prediction, **kwargs))


def reseal(record):
    record['record_sha256'] = digest({k: v for k, v in record.items() if k != 'record_sha256'})
    return record


class EntityGoldContractTests(unittest.TestCase):
    def test_perfect_entity_and_exact_typed_relation(self):
        g = graph([(0, 0, 'OBS-DP', [['located_at', '2']]), (1, 1, 'ANAT-DP', [])])
        r = score(g, g)
        self.assertEqual(r['entity_metrics_in_gold_label_scope']['tp'], 2)
        self.assertEqual(r['entity_metrics_in_gold_label_scope']['f1'], 1.)
        self.assertEqual(r['relation_metrics_in_gold_label_scope']['tp'], 1)
        self.assertEqual(r['polarity']['conditional_accuracy'], 1.)

    def test_polarity_mismatch_is_one_fp_and_one_fn_not_span_miss(self):
        r = score(graph([(0, 0, 'OBS-DP', [])]), graph([(0, 0, 'OBS-DA', [])]))
        self.assertEqual(r['entity_metrics_in_gold_label_scope'], count_metrics(0, 1, 1))
        self.assertEqual(r['polarity']['matched_observation_spans'], 1)
        self.assertEqual(r['polarity']['conditional_accuracy'], 0.)
        self.assertEqual(r['polarity']['confusion_on_matched_observation_spans']['positive']['negative'], 1)

    def test_uncertain_is_not_negative(self):
        r = score(graph([(0, 0, 'OBS-U', [])]), graph([(0, 0, 'OBS-DA', [])]))
        matrix = r['polarity']['confusion_on_matched_observation_spans']
        self.assertEqual(matrix['uncertain']['negative'], 1)
        self.assertEqual(matrix['negative']['negative'], 0)

    def test_missing_entities_are_not_negative_predictions(self):
        r = score(graph([(0, 0, 'OBS-DA', [])]), graph())
        self.assertEqual(r['entity_metrics_in_gold_label_scope'], count_metrics(0, 0, 1))
        self.assertEqual(r['polarity']['gold_span_coverage'], 0.)
        self.assertIsNone(r['polarity']['conditional_accuracy'])
        self.assertEqual(sum(sum(row.values()) for row in r['polarity']['confusion_on_matched_observation_spans'].values()), 0)

    def test_missing_graph_not_fabricated_as_valid_empty(self):
        with self.assertRaises(ValueError):
            normalized(None)

    def test_empty_entity_sets_have_undefined_f1_not_perfect(self):
        r = score(graph(), graph())
        self.assertEqual(r['entity_metrics_in_gold_label_scope'], count_metrics(0, 0, 0))
        self.assertIsNone(r['polarity']['gold_span_coverage'])

    def test_extra_entity_counted_without_gold_negative_invention(self):
        r = score(graph(), graph([(0, 0, 'OBS-DP', [])]))
        self.assertEqual(r['entity_metrics_in_gold_label_scope'], count_metrics(0, 1, 0))
        self.assertIsNone(r['entity_metrics_in_gold_label_scope']['recall'])

    def test_relation_endpoint_polarity_is_part_of_strict_match(self):
        g = graph([(0, 0, 'OBS-DP', [['located_at', '2']]), (1, 1, 'ANAT-DP', [])])
        p = graph([(0, 0, 'OBS-DA', [['located_at', '2']]), (1, 1, 'ANAT-DP', [])])
        self.assertEqual(score(g, p)['relation_metrics_in_gold_label_scope'], count_metrics(0, 1, 1))

    def test_relation_direction_is_not_ignored(self):
        g = graph([(0, 0, 'OBS-DP', [['modify', '2']]), (2, 2, 'OBS-DP', [])])
        p = graph([(0, 0, 'OBS-DP', []), (2, 2, 'OBS-DP', [['modify', '1']])])
        self.assertEqual(score(g, p)['relation_metrics_in_gold_label_scope'], count_metrics(0, 1, 1))

    def test_relation_type_is_not_ignored(self):
        g = graph([(0, 0, 'OBS-DP', [['modify', '2']]), (2, 2, 'OBS-DP', [])])
        p = graph([(0, 0, 'OBS-DP', [['suggestive_of', '2']]), (2, 2, 'OBS-DP', [])])
        r = score(g, p)
        self.assertEqual(r['relation_metrics_by_type']['modify'], count_metrics(0, 0, 1))
        self.assertEqual(r['relation_metrics_by_type']['suggestive_of'], count_metrics(0, 1, 0))

    def test_span_boundaries_must_match_exactly(self):
        r = score(graph([(0, 0, 'OBS-DP', [])]), graph([(0, 1, 'OBS-DP', [])]))
        self.assertEqual(r['entity_metrics_in_gold_label_scope'], count_metrics(0, 1, 1))
        self.assertEqual(r['polarity']['matched_observation_spans'], 0)

    def test_exact_v1_to_xl_four_label_conversion(self):
        for native, label in V1_LABELS.items():
            g = graph([(0, 0, native, [])])
            p = graph([(0, 0, label, [])])
            self.assertEqual(score(g, p, schema='radgraph_xl')['entity_metrics_in_gold_label_scope']['tp'], 1)

    def test_extended_xl_labels_remain_visible_outside_v1_scope(self):
        p = graph([(0, 0, 'Observation::measurement::definitely present', [['located_at', '2']]),
                   (1, 1, 'Anatomy::definitely present', [])])
        g = graph([(0, 0, 'OBS-DP', [['located_at', '2']]), (1, 1, 'ANAT-DP', [])])
        r = score(g, p, schema='radgraph_xl')
        self.assertEqual(r['entity_metrics_in_gold_label_scope'], count_metrics(1, 0, 1))
        self.assertEqual(r['out_of_gold_label_scope']['prediction_entities'], 1)
        self.assertEqual(r['out_of_gold_label_scope']['prediction_relations'], 1)
        self.assertEqual(r['polarity']['gold_span_coverage'], 0.)

    def test_all_eleven_xl_labels_preserved_without_disease_mapping(self):
        for label in NATIVE_LABELS:
            r = normalized(graph([(0, 0, label, [])]), schema='radgraph_xl')
            self.assertEqual(r['entities'][0][2], label)
            self.assertEqual(len(r['supported_labels']), 11)

    def test_anatomy_on_observation_span_not_a_polarity_prediction(self):
        r = score(graph([(0, 0, 'OBS-DP', [])]), graph([(0, 0, 'ANAT-DP', [])]))
        self.assertEqual(r['polarity']['matched_observation_spans'], 0)
        self.assertIsNone(r['polarity']['conditional_accuracy'])

    def test_readers_stay_separate_without_automatic_consensus(self):
        p = normalized(graph([(0, 0, 'OBS-DP', [])]))
        a = normalized(graph([(0, 0, 'OBS-DP', [])]), gold=True, reader='reader_01')
        b = normalized(graph([(0, 0, 'OBS-DA', [])]), gold=True, reader='reader_02')
        ra, rb = compare_annotations(a, p), compare_annotations(b, p)
        self.assertEqual(ra['reader_id'], 'reader_01')
        self.assertEqual(rb['reader_id'], 'reader_02')
        self.assertEqual(ra['polarity']['conditional_accuracy'], 1.)
        self.assertEqual(rb['polarity']['conditional_accuracy'], 0.)

    def test_model_prediction_cannot_be_gold_by_origin(self):
        a = normalized(graph([(0, 0, 'OBS-DP', [])]))
        with self.assertRaises(ValueError):
            compare_annotations(a, a)

    def test_explicit_declared_human_origin_requires_reader(self):
        with self.assertRaises(ValueError):
            normalize_graph(graph(), native_schema='radgraph_v1', report_id='report_0000',
                source_report_sha256=SOURCE, origin='declared_human_annotation')

    def test_declared_prediction_cannot_claim_reader_role(self):
        with self.assertRaises(ValueError):
            normalize_graph(graph(), native_schema='radgraph_v1', report_id='report_0000',
                source_report_sha256=SOURCE, origin='declared_frozen_prediction', reader_id='reader_01')

    def test_clinical_qualification_not_inferred_from_perfect_fixture(self):
        g = graph([(0, 0, 'OBS-DP', [])])
        r = score(g, g)
        self.assertEqual(r['policy'], POLICY)
        self.assertTrue(all(value is False for value in r['policy'].values()))
        self.assertIsNone(r['clinical_score'])
        self.assertIsNone(r['confirmed_faulty_modality'])

    def test_exact_token_sequence_identity_required(self):
        a = normalized(graph([(0, 0, 'OBS-DP', [])]), gold=True)
        b = normalized(graph([(0, 0, 'OBS-DP', [])], text='different words .'))
        with self.assertRaisesRegex(ValueError, 'same_declared_report_and_exact_token_sequence_required'):
            compare_annotations(a, b)

    def test_source_hash_identity_required(self):
        g = graph([(0, 0, 'OBS-DP', [])])
        a, b = normalized(g, gold=True), normalized(g)
        b['source_report_sha256'] = digest(['other_invented_source'])
        reseal(b)
        with self.assertRaises(ValueError):
            compare_annotations(a, b)

    def test_report_id_identity_required(self):
        a, b = normalized(graph(), gold=True), normalized(graph())
        b['report_id'] = 'report_0001'
        reseal(b)
        with self.assertRaises(ValueError):
            compare_annotations(a, b)

    def test_entity_tokens_must_bind_to_span(self):
        g = graph([(0, 0, 'OBS-DP', [])])
        g['entities']['1']['tokens'] = 'authored_unbound_words'
        with self.assertRaisesRegex(ValueError, 'native_entity_text_span_mismatch'):
            normalized(g)

    def test_boolean_and_invalid_offsets_rejected(self):
        for start, end in ((True, 0), (-1, 0), (0, 4), (2, 1)):
            g = graph([(0, 0, 'OBS-DP', [])])
            g['entities']['1'].update(start_ix=start, end_ix=end)
            with self.assertRaises(ValueError):
                normalized(g)

    def test_unknown_entity_and_relation_labels_fail_closed(self):
        for g in (graph([(0, 0, 'invented_unsupported', [])]),
                  graph([(0, 0, 'OBS-DP', [['invented_relation', '1']])])):
            with self.assertRaises(ValueError):
                normalized(g)

    def test_duplicate_spans_and_relations_rejected_not_deduplicated(self):
        for g in (graph([(0, 0, 'OBS-DP', []), (0, 0, 'OBS-DA', [])]),
                  graph([(0, 0, 'OBS-DP', [['modify', '2'], ['modify', '2']]), (2, 2, 'OBS-DP', [])])):
            with self.assertRaises(ValueError):
                normalized(g)

    def test_missing_relation_destination_rejected(self):
        with self.assertRaises(ValueError):
            normalized(graph([(0, 0, 'OBS-DP', [['located_at', '9']])]))

    def test_invalid_patient_like_or_reader_id_is_not_echoed(self):
        secret = 'invented_patient_identifier'
        for field, value in (('report_id', secret), ('reader_id', secret)):
            kwargs = dict(native_schema='radgraph_v1', report_id='report_0000',
                source_report_sha256=SOURCE, origin='authored_fixture', reader_id=None)
            kwargs[field] = value
            with self.assertRaises(ValueError) as cm:
                normalize_graph(graph(), **kwargs)
            self.assertNotIn(secret, str(cm.exception))

    def test_outputs_contain_no_report_words_source_keys_or_entity_tokens(self):
        g = graph([(0, 0, 'OBS-DP', [])])
        g['unused_private_field'] = 'invented_private_path'
        a, b = normalized(g, gold=True), normalized(g)
        text = json.dumps([a, b, compare_annotations(a, b)])
        for word in TEXT.split():
            if word != '.':
                self.assertNotIn(word, text)
        self.assertNotIn('invented_private_path', text)
        self.assertNotIn('"tokens"', text)

    def test_entity_id_or_dictionary_order_does_not_change_receipt(self):
        g = graph([(0, 0, 'OBS-DP', [['located_at', '2']]), (1, 1, 'ANAT-DP', [])])
        reordered = deepcopy(g)
        reordered['entities'] = {'7': reordered['entities']['2'], '8': reordered['entities']['1']}
        reordered['entities']['8']['relations'] = [['located_at', '7']]
        self.assertEqual(normalized(g), normalized(reordered))

    def test_changed_sealed_records_rejected(self):
        r = normalized(graph([(0, 0, 'OBS-DP', [])]))
        r['entities'][0][2] = V1_LABELS['OBS-DA']
        with self.assertRaises(ValueError):
            validate_record(r)

    def test_extra_record_text_rejected_even_if_resealed(self):
        r = normalized(graph())
        r['raw_report'] = 'invented_private_words'
        reseal(r)
        with self.assertRaises(ValueError):
            validate_record(r)

    def test_invalid_resealed_endpoint_or_label_scope_rejected(self):
        original = normalized(graph([(0, 0, 'OBS-DP', [['located_at', '2']]), (1, 1, 'ANAT-DP', [])]))
        for key in ('relations', 'supported_labels', 'token_count'):
            r = deepcopy(original)
            if key == 'relations':
                r[key][0][4] = 3
            elif key == 'supported_labels':
                r[key] = [V1_LABELS['OBS-DP']]
            else:
                r[key] = True
            reseal(r)
            with self.assertRaises(ValueError):
                validate_record(r)

    def test_count_metrics_denominators_and_no_boolean_counts(self):
        self.assertEqual(count_metrics(2, 1, 3)['f1'], 4 / 8)
        self.assertIsNone(count_metrics(0, 0, 0)['f1'])
        for counts in ((True, 0, 0), (-1, 0, 0), (1., 0, 0)):
            with self.assertRaises(ValueError):
                count_metrics(*counts)

    def test_no_cxrgraph_schema_conversion_claimed(self):
        with self.assertRaisesRegex(ValueError, 'unsupported_annotation_schema'):
            normalized(graph(), schema='cxrgraph')


class ReaderAggregateTests(unittest.TestCase):
    def item(self, index, *, correct=False, reader='reader_01'):
        g = normalized(graph([(0, 0, 'OBS-DP', [])]), gold=True, reader=reader)
        p = normalized(graph([(0, 0, 'OBS-DP' if correct else 'OBS-DA', [])]))
        for r in (g, p):
            r['report_id'] = f'report_{index:04d}'
            reseal(r)
        return compare_annotations(g, p)

    def aggregate(self, records, ids=None, reader='reader_01'):
        return aggregate_reader(records, reader_id=reader,
            attempted_report_ids=ids or ['report_0000', 'report_0001'],
            supported_labels=sorted(V1_LABELS.values()))

    def test_micro_counts_and_polarity_on_identical_attempted_inventory(self):
        r = self.aggregate([self.item(0, correct=True), self.item(1)])
        self.assertEqual(r['entity_micro'], count_metrics(1, 1, 1))
        self.assertEqual(r['polarity']['conditional_accuracy'], .5)
        self.assertEqual(r['comparison_availability'], 1.)

    def test_missing_attempted_comparison_not_scored_as_zero_or_omitted(self):
        r = self.aggregate([self.item(0, correct=True)])
        self.assertEqual(r['eligible_reports'], 1)
        self.assertEqual(r['unavailable_report_ids'], ['report_0001'])
        self.assertEqual(r['comparison_availability'], .5)
        self.assertEqual(r['entity_micro'], count_metrics(1, 0, 0))

    def test_no_eligible_comparison_leaves_all_metrics_null(self):
        r = self.aggregate([])
        self.assertIsNone(r['entity_micro'])
        self.assertIsNone(r['relation_micro'])
        self.assertEqual(r['comparison_availability'], 0.)

    def test_reordering_comparisons_does_not_change_aggregate(self):
        records = [self.item(0, correct=True), self.item(1)]
        self.assertEqual(self.aggregate(records), self.aggregate(list(reversed(records))))

    def test_duplicate_report_comparisons_rejected(self):
        with self.assertRaises(ValueError):
            self.aggregate([self.item(0), self.item(0)])

    def test_readers_cannot_be_pooled_as_independent_votes(self):
        with self.assertRaises(ValueError):
            self.aggregate([self.item(0), self.item(1, reader='reader_02')])

    def test_outside_attempted_report_rejected(self):
        with self.assertRaises(ValueError):
            self.aggregate([self.item(2)])

    def test_unordered_duplicate_or_nonopaque_attempted_ids_rejected(self):
        for ids in (['report_0001', 'report_0000'], ['report_0000', 'report_0000'], ['invented_patient_key']):
            with self.assertRaises(ValueError):
                self.aggregate([], ids=ids)

    def test_entity_partitions_and_f1_must_reconcile(self):
        for change in ('false_metric', 'wrong_partition'):
            r = self.item(0)
            metric = r['entity_metrics_in_gold_label_scope']
            if change == 'false_metric':
                metric['f1'] = 1.
            else:
                r['entity_metrics_in_gold_label_scope'] = count_metrics(1, 1, 1)
            r['comparison_sha256'] = digest({k: v for k, v in r.items() if k != 'comparison_sha256'})
            with self.assertRaises(ValueError):
                self.aggregate([r])

    def test_confusion_matrix_coverage_denominators_must_reconcile(self):
        r = self.item(0)
        r['polarity']['conditional_accuracy'] = 1.
        r['comparison_sha256'] = digest({k: v for k, v in r.items() if k != 'comparison_sha256'})
        with self.assertRaises(ValueError):
            self.aggregate([r])

    def test_changed_sealed_comparison_rejected(self):
        r = self.item(0)
        r['reader_id'] = 'reader_02'
        with self.assertRaises(ValueError):
            self.aggregate([r], reader='reader_02')

    def test_authored_fixtures_cannot_be_mixed_into_real_gold_cohort(self):
        a, b = self.item(0), self.item(1)
        b['gold_origin'] = 'declared_human_annotation'
        b['prediction_origin'] = 'declared_frozen_prediction'
        b['comparison_sha256'] = digest({k: v for k, v in b.items() if k != 'comparison_sha256'})
        with self.assertRaises(ValueError):
            self.aggregate([a, b])


if __name__ == '__main__':
    unittest.main()
