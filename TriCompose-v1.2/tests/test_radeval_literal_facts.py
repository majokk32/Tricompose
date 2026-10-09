"""Invented schemas/graphs only; not copied patient reports or entity gold."""
import copy
import json
import unittest

from test_radeval_expert import inventory, row
from test_radgraph_literal_evidence import graph, POS, NEG, ANAT
from tricompose_v12.fact_comparison_contract import from_native_report, compare_native_reports
from tricompose_v12.radgraph_literal_evidence import extract
from tricompose_v12.radeval_literal_facts import feature_row, join, evaluate, binary_diagnostic, FEATURES


class LiteralDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.plan = inventory([row(i) for i in range(3)])
        self.pred = []
        for p in self.plan['records']:
            def adapter(g, sha):
                return from_native_report(g, case_id='case_000', report_sha256=sha,
                    expected_native_graph_sha256=extract(g)['native_graph_sha256'])
            a = adapter(graph([('authored_finding', POS if p['candidate_slot'] == 1 else NEG, [])]), p['hypothesis_sha256'])
            b = adapter(graph([('authored_finding', POS, [])]), p['reference_sha256'])
            self.pred.append({'item_id': p['item_id'], 'status': 'complete', 'comparison': compare_native_reports(a, b)})

    def records(self):
        return join(self.plan, self.pred)

    def result(self):
        return evaluate(self.records(), resamples=20, seed=0)

    def test_complete_inventory(self):
        self.assertEqual(len(self.records()), 9)
        self.assertEqual(self.result()['complete_pairs'], 9)

    def test_native_opposition_count(self):
        self.assertEqual(self.records()[0]['features']['polarity_opposition_proposals'], 0)
        self.assertEqual(self.records()[1]['features']['polarity_opposition_proposals'], 1)

    def test_no_raw_text_in_output(self):
        exported = json.dumps(self.records()) + json.dumps(self.result())
        for forbidden in ('authored_finding', 'Invented reference', 'Invented candidate', '/invented/'):
            self.assertNotIn(forbidden, exported)

    def test_all_features_targets_retained(self):
        self.assertEqual(len(self.result()['results']), 6 * 17)

    def test_missing_graph_is_null(self):
        self.pred[0].update(status='unavailable_graph', comparison=None)
        self.assertTrue(all(v is None for v in self.records()[0]['features'].values()))
        self.assertEqual(self.result()['complete_pairs'], 8)

    def test_missing_expert_count_not_zero(self):
        self.plan['records'][0]['errors']['clinically_significant'][0] = None
        result = self.result()
        selected = next(x for x in result['results'] if x['feature'] == FEATURES[0] and x['target'] == 'clinically_significant_total')
        self.assertEqual(selected['correlation']['paired_rows'], 8)

    def test_unavailable_with_comparison_rejected(self):
        self.pred[0]['status'] = 'unavailable_graph'
        with self.assertRaises(ValueError):
            self.records()

    def test_reordered_join_rejected(self):
        self.pred.reverse()
        with self.assertRaises(ValueError):
            self.records()

    def test_wrong_direction_rejected(self):
        c = self.pred[0]['comparison']
        c['left_source_artifact_sha256'], c['right_source_artifact_sha256'] = c['right_source_artifact_sha256'], c['left_source_artifact_sha256']
        with self.assertRaises(ValueError):
            self.records()

    def test_duplicate_rejected(self):
        self.plan['records'][1]['item_id'] = self.plan['records'][0]['item_id']
        self.pred[1]['item_id'] = self.pred[0]['item_id']
        with self.assertRaises(ValueError):
            self.records()

    def test_slot_duplicate_rejected(self):
        self.plan['records'][0]['candidate_slot'] = 2
        with self.assertRaises(ValueError):
            self.records()

    def test_empty_cannot_be_complete(self):
        self.plan['records'][0]['input_nonempty'] = False
        with self.assertRaises(ValueError):
            self.records()

    def test_complete_all_zero_not_clinical_success(self):
        result = self.result()
        self.assertIsNone(result['clinical_score'])
        self.assertTrue(all(v is False for v in result['policy'].values()))

    def test_count_bool_rejected(self):
        self.pred[0]['comparison']['presence']['counts']['native_state_agreement'] = True
        with self.assertRaises(ValueError):
            self.records()

    def test_native_count_tamper_rejected(self):
        self.pred[0]['comparison']['presence']['counts']['native_state_agreement'] += 1
        with self.assertRaises(ValueError):
            self.records()

    def test_severity_promoted_rejected(self):
        self.pred[0]['comparison']['severity']['comparison'] = 'different'
        with self.assertRaises(ValueError):
            self.records()

    def test_scope_promoted_rejected(self):
        self.pred[0]['comparison']['presence']['current_scope_verified'] = True
        with self.assertRaises(ValueError):
            self.records()

    def test_policy_promoted_rejected(self):
        self.pred[0]['comparison']['policy']['clinical_qualified'] = True
        with self.assertRaises(ValueError):
            self.records()

    def test_exact_count_alarm_and_auc(self):
        d = binary_diagnostic([(2, 1), (1, 0), (0, 1), (0, 0)])
        self.assertEqual([d[k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 1, 1, 1])
        self.assertEqual(d['count_auroc'], .625)
        self.assertEqual(d['precision'], .5)

    def test_no_positive_class_auc_null(self):
        self.assertIsNone(binary_diagnostic([(0, 0), (1, 0)])['count_auroc'])

    def test_no_alarm_precision_null(self):
        self.assertIsNone(binary_diagnostic([(0, 1), (0, 0)])['precision'])

    def test_no_available_is_null_not_perfect(self):
        for p in self.pred:
            p.update(status='unavailable_graph', comparison=None)
        result = self.result()
        self.assertEqual(result['complete_pairs'], 0)
        self.assertTrue(all(r['correlation']['spearman'] is None for r in result['results']))

    def test_constant_signal_not_evidence(self):
        chosen = next(r for r in self.result()['results'] if r['feature'] == 'different_anatomy_context_concepts')
        self.assertIsNone(chosen['correlation']['spearman'])

    def test_determinism_and_input_preservation(self):
        before = copy.deepcopy((self.plan, self.pred))
        self.assertEqual(self.result(), self.result())
        self.assertEqual((self.plan, self.pred), before)

    def test_untouched_and_entity_gold_not_claimed(self):
        result = self.result()
        self.assertFalse(result['independently_held_out_or_blinded'])
        self.assertTrue(result['post_hoc_development_diagnostic'])
        self.assertTrue(all(r['binary_diagnostic']['target_is_any_expert_category_error_not_entity_gold'] for r in result['results']))

    def test_within_anchor_min_count_direction(self):
        r = next(r for r in self.result()['results'] if r['feature'] == FEATURES[0] and r['target'] == 'clinically_significant_total')
        self.assertEqual(r['within_anchor_diagnostic']['means']['selected_expected_errors'], 1)
        self.assertEqual(r['within_anchor_diagnostic']['means']['metric_minus_random_errors'], -1)

    def test_constant_features_uniform_not_first(self):
        r = next(r for r in self.result()['results'] if r['feature'] == 'different_anatomy_context_concepts' and r['target'] == 'clinically_significant_total')
        self.assertEqual(r['within_anchor_diagnostic']['means']['metric_minus_random_errors'], 0)
        self.assertEqual(r['within_anchor_diagnostic']['expected_pairwise_accuracy'], .5)

    def test_one_missing_candidate_keeps_anchor_incomplete(self):
        self.pred[0].update(status='unavailable_graph', comparison=None)
        r = next(r for r in self.result()['results'] if r['target'] == 'clinically_significant_total')
        self.assertEqual(r['within_anchor_diagnostic']['attempted_anchors'], 3)
        self.assertEqual(r['within_anchor_diagnostic']['complete_anchors'], 2)
        self.assertEqual(len(r['within_anchor_diagnostic']['anchor_rows']), 3)

    def test_location_differences_not_false_polarity(self):
        p = self.plan['records'][0]
        def adapter(g, sha):
            return from_native_report(g, case_id='case_000', report_sha256=sha,
                expected_native_graph_sha256=extract(g)['native_graph_sha256'])
        a = adapter(graph([('authored_finding', POS, [['located_at', '2']]), ('left', ANAT, [])]), p['hypothesis_sha256'])
        b = adapter(graph([('authored_finding', NEG, [['located_at', '2']]), ('right', ANAT, [])]), p['reference_sha256'])
        f = feature_row(compare_native_reports(a, b))
        self.assertEqual(f['polarity_opposition_proposals'], 0)
        self.assertEqual(f['different_anatomy_context_concepts'], 1)
