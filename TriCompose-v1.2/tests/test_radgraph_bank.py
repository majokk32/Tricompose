import copy
import unittest

from tricompose_v12.radgraph_bank import CXR_MODELS, REPORT_MODELS, inventory, summarize
from tricompose_v12.radgraph_reference_contract import METRICS


def fixture():
    rows = []
    for c, cxr in enumerate(CXR_MODELS):
        for r, report in enumerate(REPORT_MODELS):
            index = c * 4 + r
            rows.append({'case_id': 'case_000', 'triple_candidate_id': f'triple_{index:03d}',
                'cxr_candidate_id': f'cxr_{c:03d}', 'cxr_model_id': cxr, 'cxr_seed': '0',
                'report_candidate_id': f'report_{index:03d}', 'report_model_id': report,
                'report_path': f'/invented/report_{index:03d}.txt',
                'report_sha256': f'{index + 1:064x}', 'cxr_sha256': f'{c + 200:064x}',
                'ehr_sha256': 'a' * 64, 'ehr_facts_sha256': 'b' * 64})
    return rows


def receipts(plan):
    graphs = [{'graph_id': r['graph_id'], 'report_sha256': r['report_sha256'],
               'status': 'complete', 'failure_reason': None,
               'metadata': {'entity_count': 4, 'relation_count': 2}} for r in plan['graphs']]
    pairs = [{**pair, 'status': 'complete', 'scores': dict.fromkeys(METRICS, .5)}
             for pair in plan['pairs']]
    return graphs, pairs


class RadGraphBankTests(unittest.TestCase):
    def test_complete_grid_and_pairs(self):
        plan = inventory(fixture(), expected_cases=1)
        self.assertEqual((plan['candidate_slots'], plan['image_slots'],
                          plan['same_image_pairs'], plan['unique_report_graphs']), (12, 3, 18, 12))

    def test_order_is_deterministic(self):
        rows = fixture()
        self.assertEqual(inventory(rows, expected_cases=1), inventory(rows[::-1], expected_cases=1))

    def test_scores_do_not_affect_plan(self):
        rows = fixture()
        base = inventory(rows, expected_cases=1)
        for i, row in enumerate(rows):
            row['biovil_raw_cosine'] = i
            row['historical_static_selected'] = False
        self.assertEqual(base, inventory(rows, expected_cases=1))

    def test_deduplication_preserves_all_candidates(self):
        rows = fixture()
        rows[0]['report_sha256'] = rows[1]['report_sha256']
        plan = inventory(rows, expected_cases=1)
        self.assertEqual(plan['unique_report_graphs'], 11)
        self.assertEqual(plan['candidate_slots'], 12)
        self.assertEqual(sum(p['identical_report_bytes'] for p in plan['pairs']), 1)

    def test_missing_or_duplicate_slots_rejected(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            inventory(rows[:-1], expected_cases=1)
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):
            inventory(rows, expected_cases=1)

    def test_ehr_or_image_lineage_change_rejected(self):
        for key in ('ehr_sha256', 'cxr_sha256'):
            rows = fixture()
            rows[0][key] = 'c' * 64
            with self.assertRaises(ValueError):
                inventory(rows, expected_cases=1)

    def test_valid_peer_means_are_diagnostic_only(self):
        plan = inventory(fixture(), expected_cases=1)
        graphs, pairs = receipts(plan)
        table = summarize(plan, graphs, pairs)
        self.assertEqual(len(table), 12)
        self.assertTrue(all(row['radgraph_available_peers'] == 3 for row in table))
        self.assertEqual(table[0]['radgraph_peer_mean_entity_f1'], .5)
        self.assertFalse(table[0]['radgraph_agreement_clinical_qualified'])

    def test_empty_graph_not_zero_or_unknown(self):
        plan = inventory(fixture(), expected_cases=1)
        graphs, pairs = receipts(plan)
        graph_id = graphs[0]['graph_id']
        graphs[0].update(status='empty_input', failure_reason='empty_input', metadata=None)
        for pair in pairs:
            if graph_id in (pair['left_graph_id'], pair['right_graph_id']):
                pair.update(status='unavailable_graph', scores=dict.fromkeys(METRICS, None))
        table = summarize(plan, graphs, pairs)
        self.assertEqual(table[0]['radgraph_available_peers'], 0)
        self.assertIsNone(table[0]['radgraph_peer_mean_entity_f1'])
        self.assertTrue(any(row['radgraph_available_peers'] == 2 for row in table))

    def test_failed_graph_cannot_supply_score(self):
        plan = inventory(fixture(), expected_cases=1)
        graphs, pairs = receipts(plan)
        graphs[0].update(status='failed_unavailable', failure_reason='model_failed', metadata=None)
        with self.assertRaises(ValueError):
            summarize(plan, graphs, pairs)

    def test_valid_zero_retained(self):
        plan = inventory(fixture(), expected_cases=1)
        graphs, pairs = receipts(plan)
        for pair in pairs:
            pair['scores'] = dict.fromkeys(METRICS, 0.)
        self.assertEqual(summarize(plan, graphs, pairs)[0]['radgraph_peer_mean_entity_f1'], 0.)

    def test_invented_metric_or_nan_rejected(self):
        plan = inventory(fixture(), expected_cases=1)
        for mutation in ('nan', 'extra'):
            graphs, pairs = receipts(plan)
            pairs[0]['scores'][METRICS[0]] = float('nan')
            if mutation == 'extra':
                pairs[0]['scores']['clinical_truth'] = 1.
            with self.assertRaises(ValueError):
                summarize(plan, graphs, pairs)


if __name__ == '__main__':
    unittest.main()
