"""Invented score vectors, mocked image API and runtime metadata only."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

code = Path(__file__).resolve().parents[1] / 'real_validation'
sys.path.insert(0, str(code))
spec = importlib.util.spec_from_file_location('rsua_biovil_fixture', code / 'rsua_biovil.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(case_id, positive, negative):
    return {'case_id': case_id, 'score_pairs': {family: {'positive_cosine': positive, 'negative_cosine': negative}
                                             for family in module.FAMILIES}}


class RSUABioViLTests(unittest.TestCase):
    def test_three_existing_pneumonia_templates_no_new_clinical_fact(self):
        probes = module.catalog()
        self.assertEqual([p['family'] for p in probes], list(module.FAMILIES))
        self.assertEqual(len(probes), 3)
        self.assertTrue(all(p['finding'] == 'Pneumonia' for p in probes))
        self.assertEqual(probes[0]['positive_text'], 'The chest X-ray shows pneumonia.')
        self.assertEqual(probes[0]['negative_text'], 'The chest X-ray shows no pneumonia.')

    def test_perfect_margin_classifier_and_class_specific_wins(self):
        records = [row('case_0000', .8, .2), row('case_0001', .1, .6)]
        result = module.summarize(records, {'case_0000': 'positive', 'case_0001': 'negative'})
        for metrics in result.values():
            self.assertEqual(metrics['auroc'], 1)
            self.assertEqual(metrics['balanced_polarity_win_rate'], 1)

    def test_inverse_scores_not_renamed_or_flipped(self):
        records = [row('case_0000', .1, .6), row('case_0001', .8, .2)]
        result = module.summarize(records, {'case_0000': 'positive', 'case_0001': 'negative'})[module.MEAN]
        self.assertEqual(result['auroc'], 0)
        self.assertEqual(result['balanced_polarity_win_rate'], 0)

    def test_ties_are_unknown_not_negative_or_wins(self):
        records = [row('case_0000', .3, .3), row('case_0001', .3, .3)]
        refs = {'case_0000': 'positive', 'case_0001': 'negative'}
        result = module.summarize(records, refs)[module.MEAN]
        self.assertEqual((result['auroc'], result['ties'], result['balanced_polarity_win_rate']), (.5, 2, 0))
        agreement = module.scorer_disagreement(records, refs, {'case_0000': .7, 'case_0001': .2})
        for profile in agreement.values():
            self.assertEqual(profile['counts']['biovil_tie_unknown'], 2)
            self.assertIsNone(profile['agreement_rate_on_non_ties'])

    def test_high_auc_does_not_imply_negative_polarity_sensitivity(self):
        records = [row('case_0000', .8, .1), row('case_0001', .4, .1)]
        result = module.summarize(records, {'case_0000': 'positive', 'case_0001': 'negative'})[module.MEAN]
        self.assertEqual(result['auroc'], 1)
        self.assertEqual(result['negative_reference_win_rate'], 0)
        self.assertEqual(result['balanced_polarity_win_rate'], .5)

    def test_fixed_mean_not_best_template(self):
        value = row('case_0000', .3, .1)
        value['score_pairs']['shows_no']['negative_cosine'] = .9
        margins = module.margins(value)
        self.assertAlmostEqual(margins[module.MEAN], (-.6 + .2 + .2) / 3)
        self.assertLess(margins[module.MEAN], 0)

    def test_missing_template_and_nonfinite_or_out_of_range_cosine_rejected(self):
        for bad in (float('nan'), float('inf'), 2, True):
            with self.assertRaisesRegex(ValueError, 'invalid_cosine'):
                module.margins(row('case_0000', bad, .1))
        value = row('case_0000', .3, .1)
        value['score_pairs'].pop('shows_no')
        with self.assertRaisesRegex(ValueError, 'invalid_probe_inventory'):
            module.margins(value)

    def test_unknown_reference_not_coerced_negative(self):
        with self.assertRaisesRegex(ValueError, 'unknown_is_not_negative'):
            module.summarize([row('case_0000', .3, .1)], {'case_0000': 'unknown'})

    def test_single_class_auc_and_balanced_wins_unavailable(self):
        result = module.summarize([row('case_0000', .3, .1)], {'case_0000': 'positive'})[module.MEAN]
        self.assertIsNone(result['auroc'])
        self.assertIsNone(result['balanced_polarity_win_rate'])

    def test_reference_lineage_duplicate_ids_and_missing_ids_rejected(self):
        for records, refs in (([row('case_0000', .3, .1)] * 2, {'case_0000': 'positive'}),
                              ([row('case_0000', .3, .1)], {'case_0001': 'positive'})):
            with self.assertRaisesRegex(ValueError, 'reference_lineage_mismatch'):
                module.summarize(records, refs)

    def test_cross_scorer_disagreement_not_majority_clinical_reference(self):
        records = [row('case_0000', .7, .1), row('case_0001', .1, .7)]
        result = module.scorer_disagreement(records, {'case_0000': 'positive', 'case_0001': 'negative'},
                                            {'case_0000': .1, 'case_0001': .9})
        for profile in result.values():
            self.assertEqual(profile['counts']['xrv_negative_biovil_positive'], 1)
            self.assertEqual(profile['counts']['xrv_positive_biovil_negative'], 1)
            self.assertEqual(profile['agreement_rate_on_non_ties'], 0)
            self.assertIn('not_independent', profile['interpretation'])

    def test_approval_guard_before_paths_factories_and_torch_import(self):
        for env, approval in (({}, True), ({'SLURM_JOB_ID': '123'}, False), ({'SLURM_JOB_ID': 'fixture'}, True)):
            with patch.dict(module.os.environ, env, clear=True), patch.object(module, 'require_inside') as resolve, \
                    patch.object(module, '_load_runtime') as runtime:
                with self.assertRaisesRegex(RuntimeError, 'approved_rsua_biovil_slurm_required'):
                    module.run(SimpleNamespace(allow_rsua_pixels=approval))
                resolve.assert_not_called()
                runtime.assert_not_called()

    def test_forged_slurm_env_without_cgroup_refused(self):
        with patch.dict(module.os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(module.Path, 'read_text', return_value='0::/not-a-slurm-job'):
            with self.assertRaisesRegex(RuntimeError, 'slurm_cgroup_required'):
                module.require_gpu_approval(SimpleNamespace(allow_rsua_pixels=True))

    def test_lossless_png_checked_with_mock_pixels_no_patient_inputs(self):
        image_api = MagicMock()
        grayscale = image_api.open.return_value.__enter__.return_value
        grayscale.convert.return_value = grayscale
        grayscale.size, grayscale.mode = (256, 256), 'L'
        grayscale.getextrema.return_value = (1, 254)
        grayscale.tobytes.return_value = b'invented-fixture-bytes'
        with patch.object(module.os, 'chmod') as chmod:
            module.lossless_png(Path('fixture.bmp'), Path('fixture.png'), image_api=image_api)
        grayscale.save.assert_called_once_with(Path('fixture.png'), format='PNG')
        chmod.assert_called_once_with(Path('fixture.png'), 0o660)

    def test_constant_image_rejected_not_replaced(self):
        image_api = MagicMock()
        grayscale = image_api.open.return_value.__enter__.return_value
        grayscale.convert.return_value = grayscale
        grayscale.size = (256, 256)
        grayscale.getextrema.return_value = (0, 0)
        with self.assertRaisesRegex(ValueError, 'no_substitution'):
            module.lossless_png(Path('fixture.bmp'), Path('fixture.png'), image_api=image_api)
        grayscale.save.assert_not_called()

    def test_format_conversion_changes_refused(self):
        image_api = MagicMock()
        grayscale = image_api.open.return_value.__enter__.return_value
        grayscale.convert.return_value = grayscale
        grayscale.size, grayscale.mode = (256, 256), 'L'
        grayscale.getextrema.return_value = (1, 254)
        grayscale.tobytes.side_effect = [b'fixture-before', b'fixture-changed']
        with patch.object(module.os, 'chmod'), self.assertRaisesRegex(ValueError, 'lossless_png_pixels_changed'):
            module.lossless_png(Path('fixture.bmp'), Path('fixture.png'), image_api=image_api)


if __name__ == '__main__':
    unittest.main()
