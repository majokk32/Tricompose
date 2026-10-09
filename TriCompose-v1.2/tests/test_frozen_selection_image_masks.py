"""Invented metadata fixtures only; no protected/body/pixel IO."""
from copy import deepcopy
import unittest

from tricompose_v12.frozen_selection_image_masks import attach, summarize, METHODS, POLICIES, PREFIX, relation


def fixtures():
    rows, trials = [], []
    for i, report in enumerate(('positive', 'unknown')):
        row = {'case_id': f'case_{i:03}', 'triple_candidate_id': f'candidate_{i}',
            'ehr_sha256': str(i) * 64, 'cxr_sha256': 'a' * 64, 'report_sha256': str(i+2) * 64,
            'opacity_cached_ehr_state': 'unknown', 'opacity_cached_report_state': report,
            'opacity_exact_state_0_5': 'positive'}
        for flag in ('clinical_primary_eligible', 'reference_metric_transferred', 'synthetic_domain_transport_validated',
                     'selector_used', 'regeneration_authorized'):
            row[PREFIX + flag] = 'False'
        for p in POLICIES:
            state = None if p == POLICIES[-1] else 'positive'
            row.update({PREFIX + p + '_state': state or '',
                PREFIX + p + '_status': 'accepted_determinate' if state else 'abstain_template_sensitive',
                PREFIX + p + '_report_relation': relation(state, report)})
        rows.append(row)
        for method in METHODS:
            seeds = [(a, b) for a in range(5) for b in range(5)] if method == METHODS[-1] else (
                [(a, None) for a in range(5)] if method == 'random' else [(None, None)])
            for a, b in seeds:
                trials.append({'case_id': row['case_id'], 'method': method, 'model_call_budget': 4,
                    'acquisition_seed': a, 'final_choice_seed': b, 'head_definition': 'exact_opacity_0_5',
                    'frozen_trial_sha256': 'b' * 64, 'selected_candidate_id': row['triple_candidate_id'],
                    'selected_ehr_sha256': row['ehr_sha256'], 'selected_cxr_sha256': row['cxr_sha256'],
                    'selected_report_sha256': row['report_sha256'], 'simulated_calls': 4, 'selected': 1,
                    'image_state': 'positive', 'report_state': report, 'clinical_accuracy': None,
                    'ehr_cxr_clinical_score': None, 'ehr_report_clinical_score': None})
    return trials, rows, [r['case_id'] for r in rows]


class FrozenSelectionMaskTests(unittest.TestCase):
    def run_fixture(self, trials=None, rows=None):
        t, r, c = fixtures()
        return attach(t if trials is None else trials, r if rows is None else rows, cases=c, caps=(4,))

    def test_all_seeds_case_means_and_missing_denominators(self):
        t, r, cases = fixtures()
        original = deepcopy((t, r))
        out, means = attach(t, r, cases=cases, caps=(4,))
        self.assertEqual((len(out), len(means)), (66, 40))
        self.assertEqual((t, r), original)
        table, paired, losses = summarize(means, cases=cases, caps=(4,))
        self.assertEqual((len(table), len(paired), len(losses)), (20, 20, 15))
        exact = next(x for x in table if x['method'] == 'fixed' and x['mask'] == POLICIES[0])
        stable = next(x for x in table if x['method'] == 'fixed' and x['mask'] == POLICIES[-1])
        self.assertEqual(exact['mean_comparable'], .5)
        self.assertEqual(exact['conditional_proxy_agreement'], 1)
        self.assertEqual(stable['mean_not_comparable'], 1)
        self.assertIsNone(stable['conditional_proxy_agreement'])
        self.assertTrue(all(x['clinical_errors_repaired'] is None for x in losses))
        self.assertTrue(all(x['clinical_accuracy'] is None for x in table))
        self.assertEqual((out, means), attach(t, r, cases=cases, caps=(4,)))

    def test_average_replicates_before_ehrs(self):
        t, r, cases = fixtures()
        for x in t:
            if x['case_id'] == cases[0] and x['method'] in ('random', METHODS[-1]):
                x['simulated_calls'] = x['acquisition_seed']
        _, means = attach(t, r, cases=cases, caps=(4,))
        table, _, _ = summarize(means, cases=cases, caps=(4,))
        self.assertTrue(all(x['mean_simulated_calls'] == 3 for x in table if x['method'] in ('random', METHODS[-1])))

    def test_missing_choice_not_negative_or_dropped(self):
        t, _, _ = fixtures()
        t[0].update(selected_candidate_id=None, selected=0, selected_cxr_sha256=None,
                    selected_report_sha256=None, image_state='unknown', report_state='unknown')
        out, _ = self.run_fixture(trials=t)
        self.assertEqual(out[0]['masks'][POLICIES[0]]['not_comparable'], 1)
        self.assertEqual(out[0]['masks'][POLICIES[0]]['selected'], 0)
        self.assertEqual(len(out), 66)

    def test_opposition_is_not_confirmed_clinical_error(self):
        t, r, _ = fixtures()
        r[0]['opacity_cached_report_state'] = 'negative'
        for p in POLICIES:
            r[0][PREFIX + p + '_report_relation'] = relation(r[0][PREFIX + p + '_state'] or None, 'negative')
        for x in t:
            if x['case_id'] == 'case_000':
                x['report_state'] = 'negative'
        out, _ = self.run_fixture(trials=t, rows=r)
        self.assertEqual(out[0]['masks'][POLICIES[0]]['proxy_opposition'], 1)

    def test_duplicate_or_missing_seed_rejected(self):
        t, _, _ = fixtures()
        for altered in (t[:-1], [*t, t[0]]):
            with self.assertRaises(ValueError):
                self.run_fixture(trials=altered)

    def test_cannot_use_other_head_replicas(self):
        t, _, _ = fixtures()
        t[0]['head_definition'] = 'same_call_max_0_5_secondary'
        with self.assertRaises(ValueError):
            self.run_fixture(trials=t)

    def test_lineage_and_calls_cannot_change(self):
        for field, value in (('selected_candidate_id', 'absent'), ('selected_ehr_sha256', 'x'),
                             ('selected_report_sha256', 'x'), ('selected_cxr_sha256', 'x'),
                             ('simulated_calls', 5), ('simulated_calls', float('nan')),
                             ('simulated_calls', True), ('image_state', 'negative')):
            with self.subTest(field=field, value=value):
                t, _, _ = fixtures()
                t[0][field] = value
                with self.assertRaises(ValueError):
                    self.run_fixture(trials=t)

    def test_no_repair_or_known_ehr_by_metadata_promotion(self):
        for field, value in (('image_mask_regeneration_authorized', 'True'),
                             ('opacity_cached_ehr_state', 'positive'),
                             ('image_mask_xrv_exact_0_5_report_relation', 'proxy_opposition')):
            _, r, _ = fixtures()
            r[0][field] = value
            with self.assertRaises(ValueError):
                self.run_fixture(rows=r)

    def test_full_grid_and_equal_cost_contrast(self):
        _, means = self.run_fixture()
        cases = ['case_000', 'case_001']
        with self.assertRaises(ValueError):
            summarize(means[:-1], cases=cases, caps=(4,))
        for m in means:
            if m['method'] == METHODS[-1]:
                m['simulated_calls'] = 2
        with self.assertRaises(ValueError):
            summarize(means, cases=cases, caps=(4,))


if __name__ == '__main__':
    unittest.main()
