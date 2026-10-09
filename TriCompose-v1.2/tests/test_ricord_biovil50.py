"""Invented cosine vectors, opaque metadata and mocked boundaries; no model/data."""
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1] / 'real_validation'
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location('ricord_biovil_fixture', CODE / 'ricord_biovil50.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(cid, positive, negative):
    return {'case_id': cid, 'score_pairs': {f: {'positive_cosine': positive, 'negative_cosine': negative}
                                         for f in m.reuse.FAMILIES}}


def reference(state, score):
    return {'reference_state': state, 'direct_lung_opacity_score': score}


def binding_fixture():
    images = [{'case_id': f'case_{i:03d}', 'sha256': f'{i:064x}'} for i in range(50)]
    records = [{'case_id': r['case_id'], 'display_sha256': r['sha256'],
                'reference_state': 'positive' if i < 25 else 'negative',
                'direct_lung_opacity_score': .8 if i < 25 else .2} for i, r in enumerate(images)]
    return images, {'records': records}


class RICORDBioViLTests(unittest.TestCase):
    def test_all_existing_opacity_templates_and_fixed_order(self):
        prompts = m.catalog()
        self.assertEqual(len(prompts), 3)
        self.assertEqual([p['family'] for p in prompts], list(m.reuse.FAMILIES))
        self.assertTrue(all(p['finding'] == 'Lung Opacity' for p in prompts))
        self.assertEqual(prompts[0]['positive_text'], 'The chest X-ray shows lung opacity.')
        self.assertEqual(prompts[0]['negative_text'], 'The chest X-ray shows no lung opacity.')

    def test_perfect_ranking_and_positive_negative_wins(self):
        result = m.summarize([row('a', .7, .1), row('b', .1, .7)],
                             {'a': reference('positive', .8), 'b': reference('negative', .2)})
        main = result['templates'][m.reuse.MEAN]
        self.assertEqual(main['auroc'], 1)
        self.assertEqual(main['balanced_polarity_win_rate'], 1)
        self.assertEqual(result['coverage'], 1)

    def test_high_auroc_with_all_positive_preference_not_negative_understanding(self):
        result = m.summarize([row('a', .8, .1), row('b', .3, .1)],
                             {'a': reference('positive', .8), 'b': reference('negative', .2)})
        main = result['templates'][m.reuse.MEAN]
        self.assertEqual(main['auroc'], 1)
        self.assertEqual(main['negative_reference_win_rate'], 0)
        self.assertEqual(main['balanced_polarity_win_rate'], .5)

    def test_inverse_direction_not_flipped_after_reference_seen(self):
        result = m.summarize([row('a', .1, .8), row('b', .8, .1)],
                             {'a': reference('positive', .8), 'b': reference('negative', .2)})
        self.assertEqual(result['templates'][m.reuse.MEAN]['auroc'], 0)

    def test_exact_zero_ties_are_unknown_not_negative_wins(self):
        result = m.summarize([row('a', .2, .2), row('b', .2, .2)],
                             {'a': reference('positive', .8), 'b': reference('negative', .2)})
        self.assertEqual(result['templates'][m.reuse.MEAN]['ties'], 2)
        self.assertEqual(result['templates'][m.reuse.MEAN]['negative_polarity_wins'], 0)
        self.assertEqual(result['same_display_xrv_exact_0_5_comparison'], {'biovil_tie_unknown': 2})

    def test_no_best_template_substitution(self):
        value = row('a', .2, .1)
        value['score_pairs']['shows_no']['negative_cosine'] = .9
        mean = m.reuse.margins(value)[m.reuse.MEAN]
        self.assertAlmostEqual(mean, (-.7 + .1 + .1)/3)
        self.assertLess(mean, 0)

    def test_partial_failure_keeps_all_planned_class_denominators(self):
        result = m.summarize([row('a', .8, .1)],
                             {'a': reference('positive', .8), 'b': reference('positive', .8),
                              'c': reference('negative', .2)})
        main = result['templates'][m.reuse.MEAN]
        self.assertEqual((result['planned_cases'], result['failed_cases']), (3, 2))
        self.assertTrue(result['metrics_conditional_on_scored_subset'])
        self.assertEqual(main['positive_reference_recovery_all_planned'], .5)
        self.assertEqual(main['negative_reference_recovery_all_planned'], 0)
        self.assertIsNone(main['auroc'])

    def test_all_failures_metric_unavailable_not_perfect_unknown(self):
        result = m.summarize([], {'a': reference('positive', .8), 'b': reference('negative', .2)})
        self.assertIsNone(result['templates'])
        self.assertEqual((result['coverage'], result['failed_cases']), (0, 2))

    def test_unknown_reference_rejected_not_negative(self):
        for state in ('uncertain', 'unknown', None):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, 'unknown_reference_not_negative'):
                m.summarize([], {'a': reference(state, .5)})

    def test_duplicate_or_outside_scored_inventory_rejected(self):
        for records in ([row('a', .8, .1)] * 2, [row('b', .8, .1)]):
            with self.assertRaisesRegex(ValueError, 'unique_scored_subset_required'):
                m.summarize(records, {'a': reference('positive', .8)})

    def test_empty_plan_reference_denominator_refused(self):
        with self.assertRaisesRegex(ValueError, 'unique_scored_subset_required'):
            m.summarize([], {})

    def test_nonfinite_or_invalid_cosine_rejected(self):
        for value in (math.nan, math.inf, True, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.summarize([row('a', value, .1)], {'a': reference('positive', .8)})

    def test_cross_scorer_opposition_is_not_truth(self):
        result = m.summarize([row('a', .8, .1), row('b', .1, .8)],
                             {'a': reference('positive', .2), 'b': reference('negative', .8)})
        self.assertEqual(result['same_display_xrv_exact_0_5_comparison'],
                         {'xrv_negative_biovil_positive': 1, 'xrv_positive_biovil_negative': 1})
        self.assertFalse(result['cross_model_agreement_is_clinical_truth'])

    def test_reference_join_binds_all50_same_display_and_exact_head(self):
        images, scores = binding_fixture()
        self.assertEqual(len(m.reference_join(scores, images)), 50)

    def test_reference_join_different_display_refused(self):
        images, scores = binding_fixture()
        scores['records'][0]['display_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'same_display_exact_xrv_required'):
            m.reference_join(scores, images)

    def test_reference_join_no_duplicate_or_missing_records(self):
        for kind in ('missing', 'duplicate'):
            images, scores = binding_fixture()
            if kind == 'missing':
                scores['records'].pop()
            else:
                scores['records'][-1] = scores['records'][0]
            with self.assertRaisesRegex(ValueError, 'complete_reference_lineage_required'):
                m.reference_join(scores, images)

    def test_reference_join_no_unknown_to_negative_or_rebalance(self):
        images, scores = binding_fixture()
        scores['records'][0]['reference_state'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'balanced_positive_negative_reference_required'):
            m.reference_join(scores, images)

    def test_reference_join_invalid_xrv_refused(self):
        for value in (True, math.nan, 2):
            images, scores = binding_fixture()
            scores['records'][0]['direct_lung_opacity_score'] = value
            with self.assertRaisesRegex(ValueError, 'same_display_exact_xrv_required'):
                m.reference_join(scores, images)

    def test_approval_guard_precedes_input_and_factory(self):
        for env, approval in (({}, True), ({'SLURM_JOB_ID': '123'}, False),
                              ({'SLURM_JOB_ID': 'fixture'}, True)):
            with patch.dict(os.environ, env, clear=True), patch.object(m, 'require_inside') as resolve, \
                    patch.object(m.reuse, '_load_runtime') as factory:
                with self.assertRaisesRegex(RuntimeError, 'explicit_slurm_approval_required'):
                    m.run(SimpleNamespace(allow_ricord_biovil_pixels=approval))
                resolve.assert_not_called(); factory.assert_not_called()

    def test_forged_slurm_env_without_job_cgroup_refused(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '0'}, clear=True), \
                patch.object(m.Path, 'read_text', return_value='0::/elsewhere'):
            with self.assertRaisesRegex(RuntimeError, 'actual_slurm_cgroup_required'):
                m.require_slurm(True)

    def test_cpu_prepare_guard_does_not_need_cuda(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(m.Path, 'read_text', return_value='0::/slurm/job_123/step_0'):
            m.require_slurm()
            with self.assertRaisesRegex(RuntimeError, 'allocated_cuda_environment_required'):
                m.require_slurm(True)

    def test_source_metadata_prepare_never_reads_score_rows(self):
        artifacts = {f'displays/case_{i:03d}.png': f'{i:064x}' for i in range(50)}
        artifacts.update({'scores.json': 'scores', 'summary.json': 'summary'})
        manifest = {'schema_version': 'tricompose-ricord-xrv50-diagnostic-v1-manifest',
                    'artifacts': artifacts, 'sources': {}, 'dependency_sources': {}}
        summary = {'scored_cases': 50, 'failed_cases': 0, 'frozen': True,
                   'patient_grouping_verified': True, 'primary_metric_eligible': False}
        def read(path):
            self.assertNotEqual(Path(path).name, 'scores.json')
            return manifest if Path(path).name == 'manifest.json' else summary
        def digest(path):
            name = str(Path(path).relative_to(m.SOURCE))
            return m.SOURCE_SHA if name == 'manifest.json' else artifacts[name]
        with patch.object(m, 'require_inside', side_effect=lambda p, *args, **kwargs: Path(p)), \
                patch.object(m, 'read_json', side_effect=read), patch.object(m, 'sha256_file', side_effect=digest):
            _, _, images = m.source_receipt()
        self.assertEqual(len(images), 50)

    def test_preprocessing_discloses_official_remap_and_unverified_display(self):
        self.assertEqual(m.PREPROCESSING['official_loader_remap'], 'image_minmax_to_uint8')
        self.assertFalse(m.PREPROCESSING['display_transform_clinically_verified'])
        self.assertEqual((m.PREPROCESSING['resize'], m.PREPROCESSING['center_crop']), (512, 448))

    def test_result_readout_retains_all_profiles_and_nonclinical_limits(self):
        result = m.summarize([row('a', .8, .1), row('b', .1, .8)],
                             {'a': reference('positive', .8), 'b': reference('negative', .2)})
        text = m.result_markdown(result)
        for name in (*m.reuse.FAMILIES, m.reuse.MEAN):
            self.assertIn(name, text)
        self.assertIn('not a held-out clinical evaluator', text)
        self.assertIn('not negatives or successful repair', text)


if __name__ == '__main__':
    unittest.main()
