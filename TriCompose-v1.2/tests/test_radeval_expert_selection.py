import unittest

from tricompose_v12.radeval_expert import inventory, METRICS
from tricompose_v12.radeval_expert_selection import replay, summarize
from test_radeval_expert import row


class WithinAnchorTests(unittest.TestCase):
    def setUp(self):
        self.plan = inventory([row(i) for i in range(3)])
        self.scores = [{'item_id': p['item_id'], 'status': 'complete',
            'scores': {m: 1/p['candidate_slot'] for m in METRICS}} for p in self.plan['records']]

    def test_three_fixed_slots(self):
        result = replay(self.plan, self.scores)
        self.assertEqual(result['anchor_inventory'], 3)
        self.assertEqual(len(result['records']), 18)

    def test_lower_error_winner(self):
        item = replay(self.plan, self.scores)['records'][0]
        self.assertEqual(item['metric_top_expected_errors'], 1)
        self.assertEqual(item['uniform_random_expected_errors'], 2)
        self.assertEqual(item['metric_minus_random_errors'], -1)
        self.assertEqual(item['metric_oracle_hit_probability'], 1)

    def test_score_tie_uniform_not_first_slot(self):
        for score in self.scores:
            score['scores'] = {m: .5 for m in METRICS}
        item = replay(self.plan, self.scores)['records'][0]
        self.assertEqual(item['top_tie_size'], 3)
        self.assertEqual(item['metric_top_expected_errors'], 2)
        self.assertEqual(item['metric_minus_random_errors'], 0)
        self.assertAlmostEqual(item['metric_oracle_hit_probability'], 1/3)
        self.assertEqual(item['expected_metric_pairwise_correct'], 1.5)

    def test_expert_tie_oracle_not_single_arbitrary_id(self):
        for pair in self.plan['records']:
            pair['errors']['clinically_significant'] = [0]*7
        item = replay(self.plan, self.scores)['records'][0]
        self.assertEqual(item['random_oracle_hit_probability'], 1)
        self.assertEqual(item['metric_oracle_hit_probability'], 1)
        self.assertEqual(item['strict_expert_pair_comparisons'], 0)

    def test_missing_expert_invalidates_whole_anchor(self):
        self.plan['records'][0]['errors']['clinically_significant'][0] = None
        records = replay(self.plan, self.scores)['records']
        self.assertEqual(sum(r['status'] == 'incomplete_anchor' for r in records), 6)

    def test_failed_score_does_not_drop_only_bad_candidate(self):
        self.scores[0]['status'] = 'unavailable_graph'
        self.scores[0]['scores'] = {m: None for m in METRICS}
        summary = summarize(replay(self.plan, self.scores), resamples=20)
        self.assertTrue(all(s['eligible_anchors'] == 2 for s in summary['summaries']))

    def test_summary_does_not_promote_gate(self):
        result = summarize(replay(self.plan, self.scores), resamples=20)
        self.assertFalse(result['clinical_qualified'])
        self.assertFalse(result['selection_changed'])
        self.assertEqual(result['summaries'][0]['expected_pairwise_accuracy'], 1)

    def test_cluster_ci_insufficient_one_patient(self):
        result = summarize(replay(self.plan, self.scores), resamples=20)
        self.assertIsNone(result['summaries'][0]['metric_minus_random_error_delta_ci95'])

    def test_missing_candidate_slot_rejected(self):
        self.plan['records'].pop()
        self.scores.pop()
        with self.assertRaises(ValueError):
            replay(self.plan, self.scores)

    def test_deterministic(self):
        first = summarize(replay(self.plan, self.scores), resamples=30, seed=0)
        self.assertEqual(first, summarize(replay(self.plan, self.scores), resamples=30, seed=0))


if __name__ == '__main__':
    unittest.main()
