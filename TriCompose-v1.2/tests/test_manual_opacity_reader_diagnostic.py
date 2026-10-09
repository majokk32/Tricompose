"""Invented hash-only reference/prediction fixtures; no patient source reads."""
import copy
import hashlib
import unittest

from tricompose_v12.manual_opacity_reader_diagnostic import evaluate, HEADS, STATES


def fixture(truths, predicted):
    references, predictions = [], []
    for index, (truth, pred) in enumerate(zip(truths, predicted)):
        h = hashlib.sha256(f'completely-invented-fixture-{index}'.encode()).hexdigest()
        references.append({'report_id': f'report_{index:04d}', 'source_index': index, 'source_sha256': h,
            'status': 'complete' if truth is not None else 'failed_unavailable',
            'failure_type': None if truth is not None else 'ValueError',
            'projection': None if truth is None else {'source_sha256': h,
                'manual_literal_state': truth, 'current_lung_opacity_reference': None,
                'clinical_qualified': False}})
        states = {head: 'unknown' for head in HEADS}
        states['lung_opacity'] = pred
        predictions.append({'report_id': f'report_{index:04d}', 'source_sha256': h,
            'status': 'complete' if pred is not None else 'failed_unavailable',
            'failure_type': None if pred is not None else 'RuntimeError',
            'finding_states': states if pred is not None else None})
    return references, predictions


class ManualOpacityReaderDiagnosticTests(unittest.TestCase):
    def test_all_slots_remain_and_zero_support_is_null(self):
        refs, preds = fixture(['positive', 'negative', 'unknown', None], ['positive', 'positive', 'negative', 'positive'])
        result, details = evaluate(refs, preds)
        self.assertEqual(result['attempted_reports'], 4)
        self.assertEqual(result['reference_available'], 3)
        self.assertEqual(result['reference_unavailable'], 1)
        self.assertEqual(len(details), 4)
        self.assertEqual(sum(sum(row.values()) for row in result['confusion_matrix_all_attempted'].values()), 4)
        self.assertIsNone(result['per_reference_state']['uncertain']['match_fraction_all_reference_supported'])

    def test_unknown_is_not_negative_or_hallucination(self):
        result, details = evaluate(*fixture(['unknown', 'unknown'], ['positive', 'negative']))
        self.assertEqual(result['determinate_predictions_on_literal_unknown'], 2)
        self.assertEqual(result['known_literal_polarity_flips'], 0)
        self.assertEqual(result['known_literal_reference_support'], 0)
        self.assertFalse(result['literal_unknown_determinate_is_clinical_hallucination'])
        self.assertTrue(all(not row['clinical_error_adjudicated'] for row in details))

    def test_missing_prediction_does_not_improve_all_supported_fraction(self):
        result, _ = evaluate(*fixture(['positive', 'positive'], ['positive', None]))
        self.assertEqual(result['known_literal_reference_support'], 2)
        self.assertEqual(result['known_literal_prediction_available'], 1)
        self.assertEqual(result['known_literal_match_fraction_all_supported'], .5)
        self.assertEqual(result['known_literal_match_fraction_prediction_available'], 1)
        self.assertEqual(result['per_reference_state']['positive']['prediction_unavailable'], 1)

    def test_polarity_flip_denominator_is_explicit_known_not_unknown(self):
        result, _ = evaluate(*fixture(['positive', 'negative', 'uncertain', 'unknown'],
                                      ['negative', 'positive', 'positive', 'positive']))
        self.assertEqual(result['known_literal_polarity_flips'], 2)
        self.assertEqual(result['explicit_literal_reference_support'], 2)
        self.assertEqual(result['known_literal_polarity_flip_fraction_prediction_available'], 1)

    def test_reference_unavailable_does_not_prevent_model_prediction(self):
        result, details = evaluate(*fixture([None], ['positive']))
        self.assertEqual(result['prediction_status_counts'], {'complete': 1})
        self.assertEqual(result['confusion_matrix_all_attempted']['reference_unavailable']['positive'], 1)
        self.assertEqual(result['jointly_available'], 0)
        self.assertEqual(details[0]['assessment'], 'reference_unavailable')

    def test_wrong_source_hash_rejected(self):
        refs, preds = fixture(['positive'], ['positive'])
        preds[0]['source_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            evaluate(refs, preds)

    def test_duplicate_or_missing_or_reordered_reference_rejected(self):
        refs, preds = fixture(['positive', 'negative'], ['positive', 'negative'])
        for altered_refs, altered_preds in ((refs, preds[:1]), (refs, [preds[0], preds[0]]), (refs[::-1], preds)):
            with self.assertRaises(ValueError):
                evaluate(altered_refs, altered_preds)

    def test_failed_states_cannot_be_unknown(self):
        refs, preds = fixture([None], [None])
        altered = copy.deepcopy(refs)
        altered[0]['projection'] = {'manual_literal_state': 'unknown'}
        with self.assertRaises(ValueError):
            evaluate(altered, preds)
        altered = copy.deepcopy(preds)
        altered[0]['finding_states'] = {head: 'unknown' for head in HEADS}
        with self.assertRaises(ValueError):
            evaluate(refs, altered)

    def test_global_clinical_reference_promotion_rejected(self):
        refs, preds = fixture(['positive'], ['positive'])
        for field, value in (('clinical_qualified', True), ('current_lung_opacity_reference', 'positive')):
            altered = copy.deepcopy(refs)
            altered[0]['projection'][field] = value
            with self.assertRaises(ValueError):
                evaluate(altered, preds)

    def test_bad_class_width_or_binary_no_finding_rejected(self):
        refs, preds = fixture(['positive'], ['positive'])
        for field, value in (('lung_opacity', 'absent'), ('no_finding', 'negative')):
            altered = copy.deepcopy(preds)
            altered[0]['finding_states'][field] = value
            with self.assertRaises(ValueError):
                evaluate(refs, altered)
        del preds[0]['finding_states']['edema']
        with self.assertRaises(ValueError):
            evaluate(refs, preds)

    def test_determinism_no_mutation_and_no_qualification(self):
        inputs = fixture(list(STATES), list(STATES))
        snapshot = copy.deepcopy(inputs)
        first = evaluate(*inputs)
        self.assertEqual(first, evaluate(*inputs))
        self.assertEqual(snapshot, inputs)
        for flag in ('clinical_qualified', 'primary_metric_eligible', 'selection_changed',
                     'regeneration_authorized', 'untouched_final_test', 'independent_image_truth'):
            self.assertIs(first[0][flag], False)
        self.assertNotIn('accuracy', first[0])
        self.assertFalse(first[0]['other_thirteen_heads_have_reference'])


if __name__ == '__main__':
    unittest.main()
