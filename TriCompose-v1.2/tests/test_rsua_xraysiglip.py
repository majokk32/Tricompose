"""Invented vectors and metadata mocks only, no public/patient pixel access."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rsua_siglip_fixture', ROOT / 'real_validation/rsua_xraysiglip.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def row(case_id, positive=.8, negative=.1):
    return {'case_id': case_id, 'artifact_sha256': 'a' * 64,
        'pairs': w.probe.decode_scores([positive, negative] * 24, [1.0, -1.0] * 24)}


def biovil(case_id, positive=.8, negative=.1):
    return {'case_id': case_id, 'score_pairs': {family: {
        'positive_cosine': positive, 'negative_cosine': negative} for family in w.baseline.FAMILIES}}


class RSUASiglipTests(unittest.TestCase):
    def test_same_catalog_all_heads_no_new_prompts(self):
        self.assertEqual(len(w.probe.probes()), 24)
        pneumonia = [p for p in w.probe.probes() if p['finding'] == 'pneumonia']
        self.assertEqual([p['family'] for p in pneumonia], list(w.baseline.FAMILIES))
        self.assertEqual([(p['positive_text'], p['negative_text']) for p in pneumonia],
            [(p['positive_text'], p['negative_text']) for p in w.baseline.catalog()])

    def test_perfect_class_separation_and_both_polarities(self):
        result = w.analyze([row('a'), row('b', .1, .8)], {'a': 'positive', 'b': 'negative'})
        self.assertEqual(result['stable_coverage'], 1)
        self.assertEqual(result['stable_proxy_agreement'], 1)
        self.assertTrue(all(m['auroc'] == 1 and m['balanced_polarity_win_rate'] == 1
            for m in result['full_cohort_metrics'].values()))

    def test_inverted_scores_not_flipped_to_make_good_auc(self):
        result = w.analyze([row('a', .1, .8), row('b')], {'a': 'positive', 'b': 'negative'})
        self.assertEqual(result['full_cohort_metrics'][w.baseline.MEAN]['auroc'], 0)
        self.assertEqual(result['stable_proxy_agreement'], 0)

    def test_high_auc_not_reinterpreted_as_negation_success(self):
        result = w.analyze([row('a', .8, .1), row('b', .4, .1)], {'a': 'positive', 'b': 'negative'})
        metric = result['full_cohort_metrics'][w.baseline.MEAN]
        self.assertEqual(metric['auroc'], 1)
        self.assertEqual(metric['negative_reference_win_rate'], 0)
        self.assertEqual(metric['balanced_polarity_win_rate'], .5)
        self.assertEqual(result['stable_proxy_agreement'], .5)

    def test_ties_unknown_not_negative_or_wins(self):
        result = w.analyze([row('a', .1, .1), row('b', .1, .1)], {'a': 'positive', 'b': 'negative'})
        self.assertEqual(result['stable_coverage'], 0)
        self.assertIsNone(result['stable_proxy_agreement'])
        metric = result['full_cohort_metrics'][w.baseline.MEAN]
        self.assertEqual((metric['auroc'], metric['ties'], metric['balanced_polarity_win_rate']), (.5, 2, 0))

    def test_template_sensitive_retains_all_templates_and_fixed_mean(self):
        a, b = row('a', .1, .3), row('b', .1, .3)
        head = next(p for p in a['pairs'] if p['finding'] == 'pneumonia')
        head['positive_cosine'] = .9
        result = w.analyze([a, b], {'a': 'positive', 'b': 'negative'})
        self.assertEqual(result['stable_text_direction_counts']['template_sensitive'], 1)
        self.assertEqual(result['stable_coverage'], .5)
        value = w.probe.summarize_pairs(a['pairs'])['pneumonia']['mean_cosine_margin']
        self.assertAlmostEqual(value, (.6 - .2 - .2) / 3)
        self.assertEqual(set(result['full_cohort_metrics']), {*w.baseline.FAMILIES, w.baseline.MEAN})

    def test_unavailable_preserved_in_full_denominator_no_complete_case_score(self):
        a, b = row('a'), row('b')
        b['pairs'] = None
        result = w.analyze([a, b], {'a': 'positive', 'b': 'negative'})
        self.assertEqual((result['source_cases'], result['available'], result['unavailable']), (2, 1, 1))
        self.assertIsNone(result['full_cohort_metrics'])
        self.assertIsNone(result['stable_coverage'])

    def test_all_unavailable_is_not_perfect_abstention(self):
        records = [row('a'), row('b')]
        for record in records:
            record['pairs'] = None
        result = w.analyze(records, {'a': 'positive', 'b': 'negative'})
        self.assertEqual(result['available'], 0)
        self.assertIsNone(result['full_cohort_metrics'])

    def test_unknown_uncertain_reference_cannot_be_negative(self):
        for state in ('unknown', 'uncertain', None):
            with self.assertRaises(ValueError):
                w.analyze([row('a')], {'a': state})

    def test_reference_missing_duplicate_case_and_extra_case_refused(self):
        for records, refs in (([row('a'), row('a')], {'a': 'positive', 'b': 'negative'}),
                ([row('a')], {'b': 'positive'}), ([row('a')], {'a': 'positive', 'b': 'negative'})):
            with self.assertRaises(ValueError):
                w.analyze(records, refs)

    def test_nonfinite_missing_and_out_of_range_probe_refused(self):
        for kind in ('missing', 'nan', 'bounds'):
            a = row('a')
            if kind == 'missing':
                a['pairs'].pop()
            else:
                a['pairs'][0]['positive_cosine'] = float('nan') if kind == 'nan' else 1.1
            with self.assertRaises(ValueError):
                w.analyze([a], {'a': 'positive'})

    def test_single_class_auc_unavailable(self):
        result = w.analyze([row('a')], {'a': 'positive'})
        self.assertIsNone(result['full_cohort_metrics'][w.baseline.MEAN]['auroc'])
        self.assertIsNone(result['full_cohort_metrics'][w.baseline.MEAN]['balanced_polarity_win_rate'])

    def test_inputs_not_mutated_and_results_deterministic(self):
        records = [row('a'), row('b', .1, .8)]
        before = copy.deepcopy(records)
        references = {'a': 'positive', 'b': 'negative'}
        self.assertEqual(w.analyze(records, references), w.analyze(copy.deepcopy(records), references))
        self.assertEqual(records, before)

    def test_repeatability_same_two_ids_no_best_of_two(self):
        primary = [row('case_0000'), row('case_0001')]
        repeats = copy.deepcopy(primary)
        repeats[0]['pairs'][0]['positive_cosine'] += .01
        result = w.repeatability(primary, repeats)
        self.assertAlmostEqual(result[0]['max_abs_endpoint_score_delta'], .01)
        self.assertEqual(result[1]['max_abs_endpoint_score_delta'], 0)
        self.assertTrue(all(r['primary_score_replaced'] is False for r in result))
        self.assertEqual(primary[0]['pairs'][0]['positive_cosine'], .8)

    def test_repeatability_wrong_image_wrong_id_or_order_refused(self):
        primary = [row('case_0000'), row('case_0001')]
        for kind in ('image', 'case', 'order'):
            repeats = copy.deepcopy(primary)
            if kind == 'image':
                repeats[0]['artifact_sha256'] = 'b' * 64
            elif kind == 'case':
                repeats[0]['case_id'] = 'case_0002'
            else:
                repeats.reverse()
            with self.assertRaises(ValueError):
                w.repeatability(primary, repeats)

    def test_repeat_failure_retains_null_delta(self):
        primary = [row('case_0000'), row('case_0001')]
        repeats = copy.deepcopy(primary)
        repeats[0]['pairs'] = None
        result = w.repeatability(primary, repeats)
        self.assertIsNone(result[0]['max_abs_endpoint_score_delta'])

    def test_cross_scorer_operational_disagreement_not_truth(self):
        results = w.compare_directions([row('a'), row('b', .1, .8)],
            [biovil('a', .1, .8), biovil('b')], {'a': .1, 'b': .9})
        for counts in results.values():
            self.assertEqual(counts['opposed_direction_unqualified'], 2)

    def test_cross_scorer_ties_and_unavailable_not_negative(self):
        records = [row('a'), row('b', .1, .1)]
        results = w.compare_directions(records, [biovil('a', .1, .1), biovil('b')], {'a': .1, 'b': .9})
        self.assertEqual(results['biovil_mean_margin']['not_comparable'], 2)
        records[0]['pairs'] = None
        self.assertTrue(all(c['not_comparable'] == 2 for c in w.compare_directions(
            records, [biovil('a'), biovil('b')], {'a': .1, 'b': .9}).values()))

    def test_cross_scorer_invalid_probabilities_and_inventory_refused(self):
        for score in (True, float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                w.compare_directions([row('a')], [biovil('a')], {'a': score})
        with self.assertRaises(ValueError):
            w.compare_directions([row('a')], [biovil('b')], {'a': .1})

    def test_approval_guard_precedes_torch_paths_pixels_and_runtime(self):
        for env, allow in (({}, True), ({'SLURM_JOB_ID': '123'}, False), ({'SLURM_JOB_ID': 'bad'}, True)):
            with patch.dict(w.probe.os.environ, env, clear=True), patch.object(w, 'load_plan') as plan_loader:
                with self.assertRaises(RuntimeError):
                    w.run(SimpleNamespace(allow_rsua_siglip_probe=allow))
                plan_loader.assert_not_called()

    def test_forged_slurm_without_actual_cgroup_refused(self):
        with patch.dict(w.probe.os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(w.probe.Path, 'read_text', return_value='0::/not-slurm'):
            with self.assertRaises(RuntimeError):
                w.probe.require_gpu_approval(True)

    def test_preparation_failure_cleans_only_new_temporary(self):
        with patch.object(w.probe.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', return_value=(Path('/invented/new_tmp'), Path('/invented/new_run'))), \
                patch.object(w, 'prepare_payload', side_effect=ValueError('injected')), \
                patch.object(w, 'discard_atomic_run') as discard:
            with self.assertRaises(ValueError):
                w.prepare(Path('/invented'), 'new_run')
            discard.assert_called_once_with(Path('/invented/new_tmp'))

    def test_existing_plan_never_overwritten_or_cleaned(self):
        with patch.object(w.probe.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', side_effect=FileExistsError('existing')), \
                patch.object(w, 'prepare_payload') as prepare, patch.object(w, 'discard_atomic_run') as discard:
            with self.assertRaises(FileExistsError):
                w.prepare(Path('/invented'), 'existing')
            prepare.assert_not_called()
            discard.assert_not_called()

    def test_references_parsed_after_prediction_fsync_and_no_probability_claims(self):
        source = (ROOT / 'real_validation/rsua_xraysiglip.py').read_text()
        self.assertLess(source.index('os.fsync(handle.fileno())', source.index('predictions = write_private_json')),
            source.index('cohort = json.loads(COHORT.read_text())'))
        self.assertIn("'probability_semantics': False", source)
        self.assertIn("'thresholds_fitted': False", source)
        self.assertIn("'other_seven_heads_have_reference_labels': False", source)


if __name__ == '__main__':
    unittest.main()
