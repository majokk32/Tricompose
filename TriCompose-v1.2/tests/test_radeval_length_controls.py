"""Wholly invented metadata/count fixtures, no source report or model calls."""
import copy
import hashlib
import json
import unittest

from test_radeval_expert import row, inventory
from tricompose_v12.radeval_expert import outcomes
from tricompose_v12.radeval_literal_facts import FEATURES
from tricompose_v12.radeval_length_controls import build_table, evaluate, token_length, CONTROLS, SELECTORS


def fixtures():
    source = [row(i) for i in range(4)]
    for i, r in enumerate(source):
        r['images_path'] = f'/invented/p{90000000+i}/s{90000100+i}/image.png'
    plan = inventory(source)
    tokens, graphs = {}, {}
    for g in plan['graphs']:
        sha = g['text_sha256']
        tokens[sha] = {'text_sha256': sha, 'status': 'complete', 'failure_type': None,
            'native_token_count': 50, 'token_ids_sha256': hashlib.sha256(sha.encode()).hexdigest(),
            'truncated': False, 'adapter_added_prefix': False}
        graphs[g['graph_id']] = {**g, 'status': 'complete', 'failure_reason': None, 'metadata': {'entity_count': 10}}
    literals = []
    for p in plan['records']:
        slot = p['candidate_slot']
        tokens[p['hypothesis_sha256']]['native_token_count'] = 20 + 10 * slot
        graphs[p['hypothesis_graph_id']]['metadata']['entity_count'] = 20 - slot
        literals.append({**{k: p[k] for k in ('item_id', 'source_id', 'source_group_id', 'section_id',
            'section_name', 'candidate_slot', 'reference_sha256', 'hypothesis_sha256')}, 'status': 'complete',
            'features': {feature: slot - 1 for feature in FEATURES}, 'expert_outcomes': outcomes(p['errors'])})
    return plan, literals, tokens, list(graphs.values())


class LengthControlTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixtures()

    def table(self):
        return build_table(*self.inputs)

    def result(self):
        return evaluate(self.table(), resamples=20, seed=0)

    def test_all_attempted_and_nine_selectors(self):
        r = self.result()
        self.assertEqual(r['all_attempted_pairs'], 12)
        self.assertEqual(r['common_available_pairs'], 12)
        self.assertEqual(r['selectors'], list(SELECTORS))
        self.assertEqual(len(r['results']), 9 * 17)
        self.assertEqual(len(r['paired_literal_control_comparisons']), 6 * 3 * 17)

    def test_costs_are_fixed_metadata(self):
        costs = self.table()[0]['costs']
        self.assertEqual([costs[c] for c in CONTROLS], [30, 20, 19])

    def test_shortest_same_as_literal_fixture(self):
        r = next(r for r in self.result()['paired_literal_control_comparisons']
            if r['feature'] == FEATURES[0] and r['control'] == CONTROLS[0] and r['target'] == 'clinically_significant_total')
        self.assertEqual(r['literal_minus_control_mean_errors'], 0)
        self.assertEqual(r['paired_cluster_ci']['interval95'], [0, 0])

    def test_reference_distance_and_entity_controls(self):
        for control in CONTROLS[1:]:
            r = next(r for r in self.result()['paired_literal_control_comparisons']
                if r['feature'] == FEATURES[0] and r['control'] == control and r['target'] == 'clinically_significant_total')
            self.assertEqual(r['literal_minus_control_mean_errors'], -2)
            self.assertEqual(r['paired_cluster_ci']['interval95'], [-2, -2])

    def test_same_mask_for_every_selector(self):
        p = self.inputs[0]['records'][0]
        self.inputs[2][p['hypothesis_sha256']].update(status='failed_embedding', native_token_count=None, token_ids_sha256=None)
        t = self.table()
        self.assertFalse(t[0]['common_available'])
        self.assertTrue(all(v is None for v in t[0]['costs'].values()))
        r = self.result()
        self.assertEqual(r['common_available_pairs'], 11)
        self.assertTrue(all(x['complete_anchors'] == 3 for x in r['results']))

    def test_successful_tokenization_survives_embedding_failure(self):
        p = self.inputs[0]['records'][0]
        self.inputs[2][p['hypothesis_sha256']]['status'] = 'failed_embedding'
        self.assertTrue(self.table()[0]['common_available'])

    def test_missing_literal_is_not_zero(self):
        self.inputs[1][0].update(status='unavailable_graph', features=dict.fromkeys(FEATURES))
        self.assertTrue(all(v is None for v in self.table()[0]['costs'].values()))

    def test_no_metadata_available_not_perfect(self):
        for receipt in self.inputs[2].values():
            receipt.update(status='failed_embedding', native_token_count=None, token_ids_sha256=None)
        r = self.result()
        self.assertEqual(r['common_available_pairs'], 0)
        self.assertTrue(all(x['means']['selected_expected_errors'] is None for x in r['results']))

    def test_missing_annotation_preserved(self):
        p = self.inputs[0]['records'][0]
        p['errors']['clinically_significant'][0] = None
        self.inputs[1][0]['expert_outcomes'] = outcomes(p['errors'])
        r = self.result()
        self.assertTrue(all(x['complete_anchors'] == 3 for x in r['results'] if x['target'] == 'clinically_significant_total'))

    def test_zero_entities_valid_but_not_clinical_quality(self):
        p = self.inputs[0]['records'][0]
        next(g for g in self.inputs[3] if g['graph_id'] == p['hypothesis_graph_id'])['metadata']['entity_count'] = 0
        self.assertEqual(self.table()[0]['costs'][CONTROLS[2]], 0)
        self.assertFalse(self.result()['policy']['entity_count_is_clinical_quality'])

    def test_bool_token_count_rejected(self):
        next(iter(self.inputs[2].values()))['native_token_count'] = True
        with self.assertRaises(ValueError):
            self.table()

    def test_negative_entity_count_rejected(self):
        self.inputs[3][0]['metadata']['entity_count'] = -1
        with self.assertRaises(ValueError):
            self.table()

    def test_truncated_receipt_rejected(self):
        next(iter(self.inputs[2].values()))['truncated'] = True
        with self.assertRaises(ValueError):
            self.table()

    def test_prefix_changed_rejected(self):
        next(iter(self.inputs[2].values()))['adapter_added_prefix'] = True
        with self.assertRaises(ValueError):
            self.table()

    def test_extra_report_text_rejected(self):
        next(iter(self.inputs[2].values()))['report_text'] = 'authored extra fixture'
        with self.assertRaises(ValueError):
            self.table()

    def test_wrong_sha_rejected(self):
        next(iter(self.inputs[2].values()))['text_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            self.table()

    def test_missing_inventory_rejected(self):
        self.inputs[3].pop()
        with self.assertRaises(ValueError):
            self.table()

    def test_reordered_literals_rejected(self):
        self.inputs[1].reverse()
        with self.assertRaises(ValueError):
            self.table()

    def test_wrong_graph_sha_rejected(self):
        self.inputs[3][0]['text_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            self.table()

    def test_annotation_tampering_rejected(self):
        self.inputs[1][0]['expert_outcomes']['clinically_significant_total'] += 1
        with self.assertRaises(ValueError):
            self.table()

    def test_ties_not_first_slot(self):
        for receipt in self.inputs[2].values():
            receipt['native_token_count'] = 50
        r = next(r for r in self.result()['results'] if r['selector'] == CONTROLS[0] and r['target'] == 'clinically_significant_total')
        self.assertEqual(r['means']['metric_minus_random_errors'], 0)
        self.assertEqual(r['expected_pairwise_accuracy'], .5)

    def test_inputs_preserved_and_deterministic(self):
        before = copy.deepcopy(self.inputs)
        self.assertEqual(self.result(), self.result())
        self.assertEqual(self.inputs, before)

    def test_no_patient_content_in_table_or_results(self):
        text = json.dumps(self.table()) + json.dumps(self.result())
        for forbidden in ('Invented reference', 'Invented candidate', '/invented/', '90000000'):
            self.assertNotIn(forbidden, text)

    def test_no_clinical_promotion(self):
        r = self.result()
        self.assertTrue(all(v is False for v in r['policy'].values()))
        self.assertFalse(r['actual_bank_selection_performed'])
        self.assertTrue(r['previous_best_feature_known_before_controls'])
        self.assertIsNone(r['clinical_score'])

    def test_overcapacity_tokenization_is_still_full_metadata(self):
        receipt = next(iter(self.inputs[2].values()))
        receipt.update(status='over_capacity', native_token_count=700)
        self.assertEqual(token_length(receipt, receipt['text_sha256']), 700)
