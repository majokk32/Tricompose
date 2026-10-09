from copy import deepcopy
import unittest

from tricompose_v12.radeval_error_types import diagnose
from tricompose_v12.radeval_expert import CATEGORIES, inventory
from tricompose_v12.radeval_image_benchmark import METRIC, METRICS, join
from test_radeval_expert import row


class ErrorTypeTests(unittest.TestCase):
    def setUp(self):
        rows = [row(i) for i in range(4)]
        for i, source in enumerate(rows):
            source['images_path'] = f'/invented/p{90000000+i}/s{90000010+i}/image.png'
        self.plan = inventory(rows)
        images = [{'item_id': r['item_id'], 'status': 'complete',
                   'value': .9 - r['candidate_slot'] * .2} for r in self.plan['records']]
        reference = [{'item_id': r['item_id'], 'status': 'complete',
                      'scores': {m: .5 for m in METRICS[1:]}} for r in self.plan['records']]
        self.records = join(self.plan, images, reference)

    def result(self, category='false_prediction', metric=METRIC, records=None):
        return next(r for r in diagnose(self.records if records is None else records,
                                       resamples=20)['results']
                    if r['category'] == category and r['metric'] == metric)

    def test_all_seven_types_and_four_metrics(self):
        result = diagnose(self.records, resamples=20)
        self.assertEqual(len(result['results']), 28)
        self.assertEqual(result['categories'], list(CATEGORIES))

    def test_direction_and_perfect_within_anchor_ranking(self):
        result = self.result()
        self.assertAlmostEqual(result['pooled_correlation']['spearman'], 1)
        self.assertEqual(result['pairwise_accuracy'], 1)
        self.assertLess(result['means']['metric_minus_random_errors'], 0)

    def test_zero_error_type_is_not_perfect_accuracy(self):
        result = self.result('incorrect_location')
        self.assertEqual(result['strict_candidate_pairs'], 0)
        self.assertIsNone(result['pairwise_accuracy'])
        self.assertIsNone(result['pooled_correlation']['spearman'])
        self.assertFalse(result['has_error_discriminating_pairs'])

    def test_constant_scores_count_ties_as_half(self):
        result = self.result(metric=METRICS[1])
        self.assertEqual(result['pairwise_accuracy'], .5)
        self.assertEqual(result['means']['metric_minus_random_errors'], 0)

    def test_missing_score_preserves_rows_and_invalidates_entire_choice_anchor(self):
        records = deepcopy(self.records)
        records[0].update(image_score_status='unavailable_source_image', paired_cohort_available=False)
        records[0]['scores'] = {m: None for m in METRICS}
        result = self.result(records=records)
        self.assertEqual(result['all_attempted_pairs'], 12)
        self.assertEqual(result['score_and_category_available_pairs'], 11)
        self.assertEqual(result['attempted_anchors'], 4)
        self.assertEqual(result['complete_anchors'], 3)

    def test_missing_category_is_not_zero_and_does_not_erase_other_category(self):
        records = deepcopy(self.records)
        expert = records[0]['expert_outcomes']
        expert['clinically_significant_false_prediction'] = None
        expert['clinically_significant_total'] = None
        expert['all_errors_total'] = None
        self.assertEqual(self.result(records=records)['complete_anchors'], 3)
        self.assertEqual(self.result(category='omission', records=records)['complete_anchors'], 4)

    def test_same_patient_group_resampling(self):
        result = self.result()
        self.assertEqual(result['error_delta_cluster_ci']['source_groups'], 4)

    def test_repeated_patient_is_one_cluster_not_two_anchors(self):
        records = deepcopy(self.records)
        for r in records:
            if r['source_group_id'] == 'group_0001':
                r['source_group_id'] = 'group_0000'
        result = self.result(records=records)
        self.assertEqual(result['complete_anchors'], 4)
        self.assertEqual(result['error_delta_cluster_ci']['source_groups'], 3)

    def test_deterministic_even_when_input_order_changes(self):
        self.assertEqual(diagnose(self.records, resamples=20),
                         diagnose(self.records[::-1], resamples=20))

    def test_raw_text_field_refused(self):
        records = deepcopy(self.records)
        records[0]['report'] = 'invented fixture text'
        with self.assertRaises(ValueError):
            diagnose(records, resamples=20)

    def test_inconsistent_common_cohort_refused(self):
        records = deepcopy(self.records)
        records[0]['scores'][METRICS[1]] = None
        with self.assertRaises(ValueError):
            diagnose(records, resamples=20)

    def test_duplicate_pair_refused(self):
        records = deepcopy(self.records)
        records[1]['item_id'] = records[0]['item_id']
        with self.assertRaises(ValueError):
            diagnose(records, resamples=20)

    def test_incomplete_anchor_refused(self):
        with self.assertRaises(ValueError):
            diagnose(self.records[:-1], resamples=20)

    def test_bad_count_or_nan_refused(self):
        for value in (-1, float('nan'), True):
            records = deepcopy(self.records)
            records[0]['expert_outcomes']['clinically_significant_false_prediction'] = value
            with self.assertRaises(ValueError):
                diagnose(records, resamples=20)

    def test_no_clinical_promotion_or_new_calls(self):
        result = diagnose(self.records, resamples=20)
        for key in ('clinical_qualified', 'selection_changed', 'weight_or_threshold_fitting',
                    'raw_patient_input_read', 'regeneration_authorized'):
            self.assertFalse(result[key])
        self.assertEqual(result['new_model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
