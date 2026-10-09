import unittest

from tricompose_v12.radeval_expert import inventory, METRICS as RG
from tricompose_v12.radeval_image_benchmark import join, evaluate, METRIC
from test_radeval_expert import row


class ImageBenchmarkTests(unittest.TestCase):
    def setUp(self):
        rows = [row(i) for i in range(4)]
        for i,r in enumerate(rows):
            r['images_path'] = f'/fixture/p{90000000+i}/s{90000010+i}/a.jpg'
        self.plan = inventory(rows)
        self.images = [{'item_id': p['item_id'], 'status': 'complete', 'value': .5 - p['candidate_slot']*.2}
                       for p in self.plan['records']]
        self.reference = [{'item_id': p['item_id'], 'status': 'complete', 'scores': {m: .5 for m in RG}}
                          for p in self.plan['records']]

    def test_signed_cosine_allowed(self):
        records = join(self.plan,self.images,self.reference)
        self.assertLess(records[2]['scores'][METRIC],0)

    def test_missing_image_not_zero(self):
        self.images[0].update(status='unavailable_source_image',value=None)
        records = join(self.plan,self.images,self.reference)
        self.assertTrue(all(v is None for v in records[0]['scores'].values()))

    def test_all_reference_baselines_use_same_available_cohort(self):
        self.images[0].update(status='unavailable_source_image',value=None)
        result = evaluate(join(self.plan,self.images,self.reference),resamples=20)
        self.assertTrue(all(r['clinically_significant_total']['paired_rows'] == 11 for r in result['correlations'].values()))

    def test_primary_direction(self):
        result = evaluate(join(self.plan,self.images,self.reference),resamples=20)
        self.assertAlmostEqual(result['correlations'][METRIC]['clinically_significant_total']['spearman'],1)

    def test_choice_lower_error(self):
        result = evaluate(join(self.plan,self.images,self.reference),resamples=20)
        item = next(r for r in result['selection_diagnostic'] if r['metric'] == METRIC)
        self.assertEqual(item['means']['selected_expected_errors'],1)
        self.assertEqual(item['means']['metric_minus_random_errors'],-1)

    def test_tie_expected_choice(self):
        for r in self.images:
            r['value'] = -.1
        result = evaluate(join(self.plan,self.images,self.reference),resamples=20)
        item = next(r for r in result['selection_diagnostic'] if r['metric'] == METRIC)
        self.assertEqual(item['means']['metric_minus_random_errors'],0)
        self.assertEqual(item['pairwise_accuracy'],.5)

    def test_failed_text_invalidates_full_anchor_for_all_comparators(self):
        self.images[0].update(status='failed_image_or_text',value=None)
        result = evaluate(join(self.plan,self.images,self.reference),resamples=20)
        self.assertTrue(all(r['complete_anchors'] == 3 for r in result['selection_diagnostic']))

    def test_nan_rejected(self):
        self.images[0]['value'] = float('nan')
        with self.assertRaises(ValueError):
            join(self.plan,self.images,self.reference)

    def test_join_swap_rejected(self):
        with self.assertRaises(ValueError):
            join(self.plan,self.images[::-1],self.reference)

    def test_unknown_metric_availability_rejected(self):
        self.images[0]['status'] = 'unknown'
        with self.assertRaises(ValueError):
            join(self.plan,self.images,self.reference)

    def test_unavailable_numeric_rejected(self):
        self.images[0]['status'] = 'unavailable_source_image'
        with self.assertRaises(ValueError):
            join(self.plan,self.images,self.reference)

    def test_no_promotion(self):
        result = evaluate(join(self.plan,self.images,self.reference),resamples=20)
        self.assertFalse(result['clinical_qualified'])
        self.assertFalse(result['selection_changed'])
        self.assertFalse(result['independent_image_radiologist_adjudication'])

    def test_interval_patient_clusters(self):
        result = evaluate(join(self.plan,self.images,self.reference),resamples=30)
        first = next(r for r in result['selection_diagnostic'] if r['metric'] == METRIC)
        self.assertEqual(first['error_delta_cluster_ci']['source_groups'],4)
        self.assertEqual(first['error_delta_cluster_ci']['interval95'],[-1,-1])


if __name__ == '__main__':
    unittest.main()
