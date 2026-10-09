"""Invented numeric fixtures only; never source reports or clinical records."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'tools/check_radevalx_encoding_sensitivity.py'
spec = importlib.util.spec_from_file_location('test_encoding_sensitivity', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixture():
    definitions = {name: {'orientation': direction, 'provenance': 'published_cached'}
                   for name, direction in worker.prior.adapter.METRICS.values()}
    predictions = {'schema_version': worker.alignment.VERSION+'-predictions', 'benchmark': 'radevalx-1.0.0',
        'metric_definitions': definitions, 'records': []}
    references = {'schema_version': worker.alignment.VERSION+'-references', 'benchmark': 'radevalx-1.0.0',
        'reference_policy': 'two_reader_consensus', 'records': []}
    for index, count in enumerate((0, 1, 4, 5)):
        ids = {'item_id': f'pair_{index:04d}', 'source_group_id': f'group_{index:04d}'}
        predictions['records'].append({**ids, 'scores': {name: {'status': 'complete',
            'value': count if definition['orientation'] == 'lower_is_better' else -count}
            for name, definition in definitions.items()}})
        references['records'].append({**ids, 'errors': {
            'clinically_significant': [count]+[None]*7, 'clinically_insignificant': [None]*8}})
    return predictions, references


class SensitivityTests(unittest.TestCase):
    def test_unresolved_and_hypothesis_separated_and_no_mutation(self):
        p, r = fixture()
        before = copy.deepcopy((p, r))
        result = worker.calculate(p, r)
        self.assertEqual((p, r), before)
        self.assertEqual(result['error_count_cells'], 64)
        self.assertEqual(result['unresolved_count_cells'], 60)
        for metric in p['metric_definitions']:
            unresolved = result['scenarios']['blank_is_unresolved']['all100']['metrics'][metric]
            hypothetical = result['scenarios']['hypothetical_blank_is_zero']['all100']['metrics'][metric]
            self.assertIsNone(unresolved['clinically_significant_total']['spearman'])
            self.assertEqual(unresolved['clinically_significant_total']['paired_rows'], 0)
            self.assertAlmostEqual(hypothetical['clinically_significant_total']['spearman'], 1)
            self.assertEqual(hypothetical['clinically_significant_total']['paired_rows'], 4)
        self.assertEqual(result['hypothetical_noisy_subset_pairs'], 2)

    def test_deterministic_calculation(self):
        self.assertEqual(worker.calculate(*fixture()), worker.calculate(*fixture()))

    def test_missing_scores_stay_missing(self):
        p, r = fixture()
        p['records'][0]['scores']['published_radcliq'] = {'status': 'failed_unavailable', 'value': None}
        result = worker.calculate(p, r)
        self.assertEqual(result['scenarios']['hypothetical_blank_is_zero']['all100']['metrics']
            ['published_radcliq']['all_errors_total']['paired_rows'], 3)

    def test_full_known_zero_and_constant_is_not_perfect(self):
        p, r = fixture()
        for row in r['records']:
            row['errors'] = {level: [0]*8 for level in row['errors']}
        result = worker.calculate(p, r)
        self.assertEqual(result['unresolved_count_cells'], 0)
        self.assertEqual(result['hypothetical_noisy_subset_pairs'], 0)
        self.assertIsNone(result['scenarios']['hypothetical_blank_is_zero']['all100']['metrics']
            ['published_radcliq']['all_errors_total']['spearman'])
        self.assertEqual(result['matching_published_values'], 0)

    def test_invalid_counts_or_duplicate_identity_fail(self):
        for value in (-1, True, 1.5, float('nan')):
            p, r = fixture()
            r['records'][0]['errors']['clinically_significant'][0] = value
            with self.assertRaises(ValueError):
                worker.calculate(p, r)
        p, r = fixture()
        p['records'][1]['item_id'] = p['records'][0]['item_id']
        r['records'][1]['item_id'] = r['records'][0]['item_id']
        with self.assertRaises(ValueError):
            worker.calculate(p, r)

    def test_interpretation_never_promotes_or_changes_gate(self):
        policy = worker.calculate(*fixture())['policy']
        for flag in ('encoding_verified', 'clinical_qualified', 'local_implementation_qualified',
                     'primary_metric_eligible', 'selection_changed', 'regeneration_authorized'):
            self.assertIs(policy[flag], False)

    def test_guard_and_assumption_required_before_any_derived_read(self):
        for allowed, failure in ((True, RuntimeError('no_slurm')), (False, None)):
            with patch.object(worker.base.guard, 'guard', side_effect=failure), \
                    patch.object(worker.base.guard, 'metadata') as read:
                with self.assertRaises((RuntimeError, ValueError)):
                    worker.evaluate(Path('invented'), 'a'*64, 'invented', allowed)
                read.assert_not_called()

    def test_markdown_is_explicitly_hypothetical(self):
        rendered = worker.markdown(worker.calculate(*fixture()))
        self.assertIn('official blank encoding remains unverified', rendered)
        self.assertIn('not proof', rendered)


if __name__ == '__main__':
    unittest.main()
