"""Authored, text-free headroom fixtures; not clinical qualification."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import diagnose_guard_headroom_v1 as m
from test_probe_repair_v1 import observation


class GuardHeadroomTests(unittest.TestCase):
    def test_shortest_paths_charge_real_registered_transitions(self):
        graph = {'a': [('b', 'report_probe', 2), ('c', 'image_probe', 4)],
            'b': [('c', 'report_probe', 2)], 'c': []}
        costs, paths = m.shortest_paths(graph, 'a', 6)
        self.assertEqual(costs, {'a': 4, 'b': 6})
        self.assertEqual(paths['b'][0]['action'], 'report_probe')
        self.assertEqual(m.shortest_paths(graph, 'a', 8)[0]['c'], 8)

    def test_invalid_charge_cannot_create_free_oracle_path(self):
        with self.assertRaises(ValueError): m.shortest_paths({'a': [('b', 'report_probe', 0)], 'b': []}, 'a', 8)
        with self.assertRaises(ValueError): m.shortest_paths({'a': []}, 'a', True)

    def test_unchanged_choice_not_strictly_dominating(self):
        a = observation()
        self.assertFalse(m.dominates(a, a))

    def test_positive_supported_gain_dominates_without_scalar(self):
        a = observation(); z = observation(report='cxrmate', number=1, text='positive')
        self.assertTrue(m.dominates(z, a))
        self.assertFalse(m.dominates(a, z))
        edges = m.graph({a['candidate_id']: a, z['candidate_id']: z})
        self.assertEqual(edges[a['candidate_id']][0][1:], ('report_probe', 2))

    def test_negative_only_support_not_target_headroom(self):
        a = observation(ehr='unknown', image='negative', text='unknown')
        z = observation(ehr='unknown', image='negative', text='negative', report='cxrmate', number=1)
        self.assertFalse(m.dominates(z, a))
        self.assertFalse(m.graph({a['candidate_id']: a, z['candidate_id']: z})[a['candidate_id']])

    def test_quality_na_or_regression_not_dominating(self):
        a = observation(); z = observation(report='cxrmate', number=1, text='positive')
        for q in (None, .1):
            value = deepcopy(z); value['quality']['report_structure_quality_score_0_1'] = q
            self.assertFalse(m.dominates(value, a))

    def test_unknown_cannot_silence_opposition(self):
        a = observation(); z = observation(report='cxrmate', number=1, text='unknown')
        self.assertFalse(m.dominates(z, a))

    def test_cross_ehr_comparison_rejected(self):
        a = observation(); z = deepcopy(a); z['lineage']['ehr_sha256'] = '9' * 64
        with self.assertRaises(ValueError): m.dominates(z, a)

    def test_cpu_guard_before_cache_read(self):
        with patch.object(m.b, 'cpu_guard', side_effect=RuntimeError('authored')), patch.object(m.b, 'load_primary') as read:
            with self.assertRaises(RuntimeError): m.run(None)
            read.assert_not_called()


if __name__ == '__main__': unittest.main()
