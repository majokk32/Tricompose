"""Invented numeric/metadata fixtures only; no images, model or patient data."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frozen_scorecard_fixture',
    ROOT / 'tools/summarize_frozen_evaluators.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def margins(sign):
    pair = {'positive_cosine': .8 if sign > 0 else .1,
        'negative_cosine': .1 if sign > 0 else .8}
    return w.family_margins({f: pair.copy() for f in w.FAMILIES})


def fixture():
    cohort, xs, bs, ss = [], [], [], []
    for i in range(50):
        cid, state = f'case_{i:04d}', 'positive' if i < 25 else 'negative'
        original, png = f'{i+1:064x}', f'{i+101:064x}'
        pair = {'positive_cosine': .8 if i < 25 else .1,
            'negative_cosine': .1 if i < 25 else .8}
        cohort.append({'case_id': cid, 'reference_state': state})
        xs.append({'case_id': cid, 'reference_state': state, 'image_sha256': original,
            'pneumonia_score': .8 if i < 25 else .2})
        bs.append({'case_id': cid, 'source_image_sha256': original, 'png_sha256': png,
            'score_pairs': {f: pair.copy() for f in w.FAMILIES}})
        ss.append({'case_id': cid, 'original_bmp_sha256': original, 'artifact_sha256': png,
            'clinical_state': None, 'calibrated_probability': None,
            'independent_clinical_validation': False, 'failure_reason': None,
            'pairs': [{'finding': 'pneumonia', 'family': f, **pair} for f in w.FAMILIES]})
    return {'cohort': {'records': cohort, 'patient_grouping_verified': False,
            'independent_image_disease_adjudication': False},
        ('xrv', 'scores.json'): {'records': xs},
        ('biovil', 'scores.json'): {'records': bs},
        ('siglip', 'predictions.json'): {'records': ss}}


class FrozenScorecardTests(unittest.TestCase):
    def test_exact_fifty_same_image_alignment(self):
        values = fixture()
        labels, xs, vlm = w.align_inputs(values)
        self.assertEqual((len(xs), sum(labels)), (50, 25))
        self.assertEqual(set(vlm), {'biovil', 'siglip'})
        self.assertEqual(vlm['biovil'], vlm['siglip'])

    def test_unknown_uncertain_reference_not_negative(self):
        for state in ('unknown', 'uncertain', None):
            values = fixture()
            values['cohort']['records'][0]['reference_state'] = state
            with self.assertRaises(ValueError):
                w.align_inputs(values)

    def test_missing_duplicate_reordered_cases_rejected(self):
        for mode in ('missing', 'duplicate', 'reordered'):
            values = fixture()
            rows = values['siglip', 'predictions.json']['records']
            if mode == 'missing':
                rows.pop()
            elif mode == 'duplicate':
                rows[1]['case_id'] = rows[0]['case_id']
            else:
                rows.reverse()
            with self.assertRaises(ValueError):
                w.align_inputs(values)

    def test_original_or_png_image_mismatch_rejected(self):
        for model, field in (('xrv', 'image_sha256'), ('biovil', 'png_sha256'),
                ('biovil', 'source_image_sha256'), ('siglip', 'original_bmp_sha256'),
                ('siglip', 'artifact_sha256')):
            values = fixture()
            values[model, w.PARENTS[model][2]]['records'][0][field] = 'f' * 64
            with self.assertRaises(ValueError):
                w.align_inputs(values)

    def test_failed_readout_not_dropped_or_zero_filled(self):
        values = fixture()
        values['siglip', 'predictions.json']['records'][0]['pairs'] = None
        with self.assertRaises(ValueError):
            w.align_inputs(values)

    def test_clinical_state_or_probability_cannot_be_promoted(self):
        for field, value in (('clinical_state', 'positive'), ('calibrated_probability', .9),
                ('independent_clinical_validation', True)):
            values = fixture()
            values['siglip', 'predictions.json']['records'][0][field] = value
            with self.assertRaises(ValueError):
                w.align_inputs(values)

    def test_template_missing_or_duplicated_refused(self):
        for mode in ('missing', 'duplicate'):
            values = fixture()
            pairs = values['siglip', 'predictions.json']['records'][0]['pairs']
            if mode == 'missing':
                pairs.pop()
            else:
                pairs[1]['family'] = pairs[0]['family']
            with self.assertRaises(ValueError):
                w.align_inputs(values)

    def test_nonfinite_boolean_and_out_of_range_values_refused(self):
        for value in (True, float('nan'), float('inf'), -.1, 1.1):
            values = fixture()
            values['xrv', 'scores.json']['records'][0]['pneumonia_score'] = value
            with self.assertRaises(ValueError):
                w.align_inputs(values)
        with self.assertRaises(ValueError):
            w.family_margins({f: {'positive_cosine': 1.1, 'negative_cosine': .1} for f in w.FAMILIES})

    def test_all_templates_used_fixed_arithmetic_mean(self):
        values = {'shows_no': {'positive_cosine': .8, 'negative_cosine': .2},
            'evidence_no': {'positive_cosine': .1, 'negative_cosine': .4},
            'present_absent': {'positive_cosine': .2, 'negative_cosine': .5}}
        result = w.family_margins(values)
        self.assertAlmostEqual(result[w.MEAN], 0)
        self.assertEqual(set(result), {*w.FAMILIES, w.MEAN})

    def test_high_ranking_does_not_mean_polarity_accuracy(self):
        value = w.decision_metrics([1, 0], [.8, .2], threshold=0., equality_is_positive=False)
        self.assertEqual(value['auroc'], 1)
        self.assertEqual(value['balanced_win_rate'], .5)
        self.assertEqual(value['negative_wins'], 0)
        self.assertIsNone(value['clinical_accuracy'])

    def test_ties_not_negative_and_xrv_boundary_is_separate(self):
        vlm = w.decision_metrics([1, 0], [0., 0.], threshold=0., equality_is_positive=False)
        xrv = w.decision_metrics([1, 0], [0., 0.], threshold=0., equality_is_positive=True)
        self.assertEqual((vlm['positive_wins'], vlm['negative_wins'], vlm['boundary_ties']), (0, 0, 2))
        self.assertEqual((xrv['positive_wins'], xrv['negative_wins']), (1, 0))
        self.assertEqual(vlm['auroc'], .5)
        self.assertEqual(vlm['average_precision'], .5)

    def test_template_stability_full_and_conditional_denominators(self):
        stable = margins(-1)
        sensitive = margins(1)
        sensitive['evidence_no'] *= -1
        sensitive[w.MEAN] = sum(sensitive[f] for f in w.FAMILIES) / 3
        value = w.template_stability([0, 1], [stable, sensitive])
        self.assertEqual(value['stable_cases'], 1)
        self.assertEqual(value['stable_coverage'], .5)
        self.assertEqual(value['conditional_proxy_win_rate'], 1)
        self.assertIsNone(value['clinical_accuracy'])

    def test_all_tied_templates_not_perfect_abstention(self):
        row = {f: 0. for f in (*w.FAMILIES, w.MEAN)}
        value = w.template_stability([1, 0], [row, row])
        self.assertEqual(value['stable_cases'], 0)
        self.assertIsNone(value['conditional_proxy_win_rate'])

    def test_forged_template_mean_refused(self):
        row = margins(1)
        row[w.MEAN] = -.9
        with self.assertRaises(ValueError):
            w.template_stability([1], [row])

    def test_percentile_interpolation_and_degenerate_interval(self):
        self.assertEqual(w.percentile([0., 1.], .25), .25)
        self.assertEqual(w.percentile([.5, .5], .025), .5)
        self.assertEqual(w.percentile([0., 1.], 1.), 1.)

    def test_invalid_percentile_input_refused(self):
        for values, q in (([], .5), ([0., 1.], -1), ([0., float('nan')], .5), ([0., 1.], True)):
            with self.assertRaises(ValueError):
                w.percentile(values, q)

    def test_paired_bootstrap_equal_scores_have_zero_difference_every_resample(self):
        labels = [1, 1, 0, 0]
        scores = {m: [.9, .1, .8, .2] for m in w.RANK_MODELS}
        value = w.paired_bootstrap(labels, scores, replicates=30, seed=0)
        for metrics in value['paired_differences'].values():
            for result in metrics.values():
                self.assertEqual((result['estimate'], result['lower_95'], result['upper_95']), (0, 0, 0))
        self.assertEqual(value['class_counts_preserved'], {'positive': 2, 'negative': 2})
        self.assertTrue(value['paired_indices_shared_across_models'])

    def test_bootstrap_deterministic_without_input_mutation(self):
        labels, scores = [1, 1, 0, 0], {m: [.9, .1, .8, .2] for m in w.RANK_MODELS}
        before = copy.deepcopy(scores)
        first = w.paired_bootstrap(labels, scores, replicates=30, seed=0)
        self.assertEqual(first, w.paired_bootstrap(labels, scores, replicates=30, seed=0))
        self.assertEqual(scores, before)

    def test_bootstrap_direction_not_flipped_or_best_template_selected(self):
        scores = {'xrv': [.1, .9], 'biovil': [.8, .2], 'siglip': [.7, .3]}
        result = w.paired_bootstrap([1, 0], scores, replicates=10, seed=0)
        self.assertEqual(result['models']['xrv']['auroc']['estimate'], 0)
        self.assertEqual(result['paired_differences']['siglip_minus_xrv']['auroc']['estimate'], 1)
        self.assertEqual(set(result['models']), set(w.RANK_MODELS))
        self.assertIsNone(result['p_values'])
        self.assertFalse(result['clinical_significance_established'])

    def test_missing_extra_model_or_shared_case_inventory_refused(self):
        good = {m: [.8, .2] for m in w.RANK_MODELS}
        for mode in ('missing', 'extra', 'length'):
            scores = copy.deepcopy(good)
            if mode == 'missing':
                scores.pop('xrv')
            elif mode == 'extra':
                scores['best_template'] = [.8, .2]
            else:
                scores['xrv'].pop()
            with self.assertRaises(ValueError):
                w.paired_bootstrap([1, 0], scores, replicates=10)

    def test_invalid_reference_and_bootstrap_settings_refused(self):
        scores = {m: [.8, .2] for m in w.RANK_MODELS}
        for labels in ([1, 1], [True, False], [1, None], []):
            with self.assertRaises(ValueError):
                w.paired_bootstrap(labels, scores, replicates=10)
        for rep, seed in ((1, 0), (True, 0), (10, True)):
            with self.assertRaises(ValueError):
                w.paired_bootstrap([1, 0], scores, replicates=rep, seed=seed)

    def test_usage_document_cannot_authorize_policy_or_clinical_scores(self):
        usage = w.score_usage_contract()
        self.assertTrue(usage['documentation_only_not_deployment_policy'])
        self.assertFalse(usage['new_action_or_repair_authorization'])
        self.assertFalse(usage['old_lexicographic_selection_changed'])
        self.assertIsNone(usage['universal_weighted_score'])
        self.assertIsNone(usage['clinical_fault_localization_accuracy'])
        for model in usage['evaluators'].values():
            self.assertFalse(model['qualified_calibrated_probability'])
            self.assertFalse(model['clinical_truth_or_fault_authority'])

    def test_slurm_guard_precedes_input_reads(self):
        with patch.dict(w.os.environ, {}, clear=True), patch.object(w, 'load_inputs') as load:
            with self.assertRaises(RuntimeError):
                w.execute(Path('/invented'), 'invented_run')
            load.assert_not_called()

    def test_forged_slurm_and_gpu_allocation_refused(self):
        with patch.dict(w.os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(w.Path, 'read_text', return_value='/not/slurm'):
            with self.assertRaises(RuntimeError):
                w.require_cpu_slurm()
        with patch.dict(w.os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_GPUS': '0'}, clear=True), \
                patch.object(w.Path, 'read_text', return_value='/job_123/step_batch'):
            with self.assertRaises(RuntimeError):
                w.require_cpu_slurm()

    def test_source_change_aborts_and_cleans_only_new_temporary(self):
        row = {'model': 'xrv', 'profile': 'invented'}
        src = {'invented': Path('/invented/source')}
        with patch.object(w, 'require_cpu_slurm'), \
                patch.object(w, 'load_inputs', return_value=({}, src)), \
                patch.object(w, 'sha256_file', side_effect=['a' * 64, 'b' * 64]), \
                patch.object(w, 'analyze', return_value=([row], {}, {})), \
                patch.object(w, 'new_atomic_run', return_value=(Path('/invented/tmp'), Path('/invented/run'))), \
                patch.object(w, 'write_private_text'), patch.object(w, 'write_private_json'), \
                patch.object(w, 'render', return_value='invented'), \
                patch.object(w, 'discard_atomic_run') as cleanup, patch.object(w, 'commit_atomic_run') as commit:
            with self.assertRaises(ValueError):
                w.execute(Path('/invented'), 'invented_run')
            cleanup.assert_called_once_with(Path('/invented/tmp'))
            commit.assert_not_called()

    def test_existing_run_is_not_overwritten(self):
        with patch.object(w, 'require_cpu_slurm'), \
                patch.object(w, 'load_inputs', return_value=({}, {})), \
                patch.object(w, 'analyze', return_value=([], {}, {})), \
                patch.object(w, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(w, 'write_private_text') as write:
            with self.assertRaises(FileExistsError):
                w.execute(Path('/invented'), 'existing')
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
