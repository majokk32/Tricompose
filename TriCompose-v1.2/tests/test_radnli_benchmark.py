"""All sentence pairs are invented here, not drawn from any medical dataset."""
from copy import deepcopy
import hashlib
import json
import unittest

from tricompose_v12.radnli_benchmark import (
    LABELS, digest, evaluate, reference_inventory, request_inventory,
    constant_neutral_predictions, split_overlap,
)


def documents(labels=LABELS):
    return [{'pair_id': f'fictional-original-id-{i}',
             'sentence1': f'Invented premise {i} about object alpha.',
             'sentence2': f'Invented hypothesis {i} about object beta.', 'gold_label': label}
            for i, label in enumerate(labels)]


def predictions(requests, labels):
    rows = constant_neutral_predictions(requests)
    for row, label in zip(rows, labels):
        row['prediction_label'] = label
    return rows


class RadNLIBenchmarkTests(unittest.TestCase):
    def test_opaque_metadata_does_not_export_original_id_or_text(self):
        docs = documents()
        inputs, requests = request_inventory(docs, split='dev')
        refs = reference_inventory(docs, split='dev')
        output = json.dumps([requests, refs])
        for doc in docs:
            for key in ('pair_id', 'sentence1', 'sentence2'):
                self.assertNotIn(doc[key], output)
        self.assertEqual(set(inputs), {f'pair_{i:04d}' for i in range(3)})

    def test_model_requests_ignore_gold_and_original_ids(self):
        docs = documents()
        before = request_inventory(docs, split='dev')
        for doc in docs:
            doc['gold_label'] = 'unsupported-secret-gold'
            doc['pair_id'] = 'changed-internal-source-id'
        self.assertEqual(before, request_inventory(docs, split='dev'))

    def test_exact_unicode_whitespace_and_case_are_not_normalized(self):
        docs = documents(('neutral',))
        docs[0]['sentence1'] = '  Fictional Alpha\nβeta.  '
        inputs, rows = request_inventory(docs, split='dev')
        self.assertEqual(inputs['pair_0000'][0], docs[0]['sentence1'])
        self.assertEqual(rows[0]['premise_sha256'], hashlib.sha256(docs[0]['sentence1'].encode()).hexdigest())
        self.assertNotEqual(rows[0]['premise_sha256'], hashlib.sha256(docs[0]['sentence1'].strip().encode()).hexdigest())

    def test_direction_changes_pair_hash_but_not_unordered_group(self):
        forward = documents(('entailment',))[0]
        reverse = {**forward, 'pair_id': 'fictional-reverse',
            'sentence1': forward['sentence2'], 'sentence2': forward['sentence1'], 'gold_label': 'neutral'}
        _, rows = request_inventory([forward, reverse], split='dev')
        self.assertNotEqual(rows[0]['source_pair_sha256'], rows[1]['source_pair_sha256'])
        self.assertEqual(rows[0]['unordered_pair_sha256'], rows[1]['unordered_pair_sha256'])

    def test_missing_text_remains_failed_slot_not_dummy_pair(self):
        docs = documents()
        docs[1]['sentence2'] = ''
        inputs, rows = request_inventory(docs, split='dev')
        self.assertEqual(len(rows), 3)
        self.assertNotIn('pair_0001', inputs)
        self.assertEqual(rows[1]['status'], 'failed_unavailable')
        preds = constant_neutral_predictions(rows)
        self.assertIsNone(preds[1]['prediction_label'])

    def test_source_labels_invalid_does_not_suppress_valid_model_input(self):
        docs = documents()
        docs[1]['gold_label'] = 'unknown'
        _, requests = request_inventory(docs, split='dev')
        refs = reference_inventory(docs, split='dev')
        preds = constant_neutral_predictions(requests)
        self.assertEqual(preds[1]['status'], 'complete')
        self.assertEqual(refs[1]['status'], 'failed_unavailable')
        self.assertIsNone(refs[1]['gold_label'])
        result = evaluate(refs, preds)
        self.assertEqual(result['attempted_pairs'], 3)
        self.assertEqual(result['reference_unavailable'], 1)

    def test_duplicate_internal_source_ids_retained_as_reference_failure(self):
        docs = documents()
        docs[1]['pair_id'] = docs[0]['pair_id']
        refs = reference_inventory(docs, split='dev')
        self.assertEqual(len(refs), 3)
        self.assertEqual(refs[1]['status'], 'failed_unavailable')
        self.assertEqual(refs[0]['status'], 'complete')

    def test_extra_source_field_is_reference_schema_failure(self):
        doc = documents(('neutral',))[0]
        doc['unsupported_gold_extension'] = 'authored-only'
        ref = reference_inventory([doc], split='dev')[0]
        self.assertEqual(ref['status'], 'failed_unavailable')
        self.assertIsNone(ref['gold_label'])

    def test_three_class_perfect_predictions(self):
        docs = documents()
        _, requests = request_inventory(docs, split='dev')
        result = evaluate(reference_inventory(docs, split='dev'), predictions(requests, LABELS))
        self.assertEqual(result['failure_aware_accuracy'], 1)
        self.assertEqual(result['macro_f1_reference_supported_classes'], 1)
        self.assertEqual(result['reference_available'], 3)
        self.assertTrue(all(r['reference_support'] == 1 for r in result['per_label'].values()))

    def test_constant_neutral_baseline_is_not_a_majority_fitted_rule(self):
        docs = documents(('contradiction',) * 3)
        _, requests = request_inventory(docs, split='dev')
        preds = constant_neutral_predictions(requests)
        self.assertTrue(all(p['prediction_label'] == 'neutral' for p in preds))
        self.assertEqual(evaluate(reference_inventory(docs, split='dev'), preds)['failure_aware_accuracy'], 0)

    def test_failed_prediction_stays_in_f1_and_accuracy_denominators(self):
        docs = documents(('entailment', 'entailment'))
        _, requests = request_inventory(docs, split='dev')
        preds = predictions(requests, ('entailment', 'entailment'))
        preds[1].update(status='failed_unavailable', prediction_label=None, failure_type='RuntimeError')
        result = evaluate(reference_inventory(docs, split='dev'), preds)
        self.assertEqual(result['failure_aware_accuracy'], .5)
        self.assertEqual(result['completed_only_accuracy'], 1)
        self.assertEqual(result['per_label']['entailment']['fn'], 1)
        self.assertEqual(result['per_label']['entailment']['f1'], 2 / 3)

    def test_empty_class_has_null_recall_and_not_macro_f1_member(self):
        docs = documents(('neutral', 'neutral'))
        _, requests = request_inventory(docs, split='dev')
        result = evaluate(reference_inventory(docs, split='dev'), constant_neutral_predictions(requests))
        self.assertIsNone(result['per_label']['contradiction']['recall'])
        self.assertIsNone(result['per_label']['contradiction']['f1'])
        self.assertEqual(result['macro_f1_included_labels'], ['neutral'])

    def test_failure_aware_confusion_retains_all_attempted_pairs(self):
        docs = documents()
        docs[0]['gold_label'] = 'invalid'
        _, requests = request_inventory(docs, split='dev')
        preds = constant_neutral_predictions(requests)
        preds[1].update(status='failed_unavailable', prediction_label=None, failure_type='RuntimeError')
        result = evaluate(reference_inventory(docs, split='dev'), preds)
        self.assertEqual(sum(sum(row.values()) for row in result['confusion_matrix_all_attempted'].values()), 3)
        self.assertEqual(result['confusion_matrix_all_attempted']['reference_unavailable']['neutral'], 1)
        self.assertEqual(result['confusion_matrix_all_attempted']['neutral']['prediction_unavailable'], 1)

    def test_wrong_directional_hash_or_wrong_split_rejected(self):
        docs = documents()
        _, requests = request_inventory(docs, split='dev')
        refs = reference_inventory(docs, split='dev')
        for field, value in (('source_pair_sha256', 'a' * 64), ('source_split', 'test')):
            changed = constant_neutral_predictions(requests)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                evaluate(refs, changed)

    def test_missing_or_duplicate_prediction_rows_rejected(self):
        docs = documents()
        _, requests = request_inventory(docs, split='dev')
        refs = reference_inventory(docs, split='dev')
        preds = constant_neutral_predictions(requests)
        for changed in (preds[:1], [preds[0], preds[0], preds[2]]):
            with self.assertRaises(ValueError):
                evaluate(refs, changed)

    def test_failed_predictions_cannot_be_encoded_as_neutral(self):
        docs = documents()
        _, requests = request_inventory(docs, split='dev')
        preds = constant_neutral_predictions(requests)
        preds[0].update(status='failed_unavailable', failure_type='RuntimeError')
        with self.assertRaises(ValueError):
            evaluate(reference_inventory(docs, split='dev'), preds)

    def test_invalid_classifier_label_is_rejected_not_recoded(self):
        docs = documents()
        _, requests = request_inventory(docs, split='dev')
        preds = predictions(requests, ('negative', 'neutral', 'contradiction'))
        with self.assertRaises(ValueError):
            evaluate(reference_inventory(docs, split='dev'), preds)

    def test_paired_directions_need_separate_answers(self):
        forward = documents(('entailment',))[0]
        reverse = {**forward, 'pair_id': 'fictional-reverse',
            'sentence1': forward['sentence2'], 'sentence2': forward['sentence1'], 'gold_label': 'neutral'}
        docs = [forward, reverse]
        _, requests = request_inventory(docs, split='dev')
        result = evaluate(reference_inventory(docs, split='dev'), predictions(requests, ('entailment', 'entailment')))
        self.assertEqual(result['distinct_unordered_sentence_pairs'], 1)
        self.assertEqual(result['bidirectional_sentence_pair_groups'], 1)
        self.assertEqual(result['fully_annotated_bidirectional_groups'], 1)
        self.assertEqual(result['bidirectional_groups_all_directions_correct'], 0)

    def test_conflicting_gold_labels_for_same_exact_input_are_not_hidden(self):
        first = documents(('entailment',))[0]
        docs = [first, {**first, 'pair_id': 'different-fictional-id', 'gold_label': 'contradiction'}]
        _, requests = request_inventory(docs, split='dev')
        result = evaluate(reference_inventory(docs, split='dev'), constant_neutral_predictions(requests))
        self.assertEqual(result['ordered_input_groups_with_conflicting_gold_labels'], 1)
        self.assertEqual(result['attempted_pairs'], 2)

    def test_reverse_dev_test_overlap_is_detected_without_dropping_rows(self):
        dev = documents(('entailment',))
        first = dev[0]
        test = [{**first, 'sentence1': first['sentence2'], 'sentence2': first['sentence1']}]
        result = split_overlap(reference_inventory(dev, split='dev'), reference_inventory(test, split='test'))
        self.assertEqual(result['ordered_pair_hash_overlap'], 0)
        self.assertEqual(result['unordered_pair_hash_overlap'], 1)
        self.assertEqual(result['sentence_hash_overlap'], 2)
        self.assertFalse(result['patient_disjointness_verified'])
        self.assertFalse(result['test_examples_automatically_dropped'])

    def test_unordered_group_hash_tampering_is_rejected(self):
        docs = documents()
        _, requests = request_inventory(docs, split='dev')
        refs = reference_inventory(docs, split='dev')
        refs[0]['unordered_pair_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            evaluate(refs, constant_neutral_predictions(requests))

    def test_repeated_eval_is_deterministic_without_input_mutation(self):
        docs = documents()
        snapshot = deepcopy(docs)
        _, requests = request_inventory(docs, split='dev')
        refs = reference_inventory(docs, split='dev')
        preds = constant_neutral_predictions(requests)
        before = deepcopy((refs, preds))
        self.assertEqual(evaluate(refs, preds), evaluate(refs, preds))
        self.assertEqual((refs, preds), before)
        self.assertEqual(docs, snapshot)

    def test_no_clinical_or_modality_localization_claim_is_unlocked(self):
        docs = documents()
        _, requests = request_inventory(docs, split='dev')
        result = evaluate(reference_inventory(docs, split='dev'), predictions(requests, LABELS))
        for flag in ('clinical_qualified', 'primary_clinical_metric_eligible', 'image_truth_verified',
            'ehr_truth_verified', 'selection_changed', 'regeneration_authorized', 'new_training',
            'neutral_is_clinical_negative', 'entailment_is_symmetric', 'sentence_pairs_are_independent_patients'):
            self.assertIs(result[flag], False)

    def test_bounded_inventory_and_split_required(self):
        for docs, split in (([], 'dev'), (documents(), 'train'), (documents(), None), (documents() * 342, 'dev')):
            with self.assertRaises(ValueError):
                request_inventory(docs, split=split)


if __name__ == '__main__':
    unittest.main()
