"""Invented references/scores only; never load a model or real patient input."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import biovil_fact_polarity as method


def fixture(states=('positive', 'negative', 'uncertain', 'unknown')):
    rows, refs = [], {}
    for split in ('val', 'test'):
        for index, state in enumerate(states):
            case_id = f'{split}_{index}'
            refs[case_id] = dict.fromkeys(method.FINDINGS, state)
            positive, negative = (0.8, 0.2) if state == 'positive' else (0.2, 0.8)
            rows.append({'case_id': case_id, 'split': split, 'image_sha256': '1'*64,
                         'score_pairs': {p['probe_id']: {'positive_cosine': positive,
                                                        'negative_cosine': negative} for p in method.probes()}})
    return rows, refs


class FactPolarityTests(unittest.TestCase):
    def test_catalog_is_fixed_balanced_and_deterministic(self):
        self.assertEqual(method.probes(), method.probes())
        self.assertEqual(len(method.probes()), 24)
        self.assertEqual(len({p['probe_id'] for p in method.probes()}), 24)
        for p in method.probes():
            self.assertIn(p['finding'].lower(), p['positive_text'].lower())
            self.assertIn(p['finding'].lower(), p['negative_text'].lower())
            for banned in ('severe', 'left', 'right', 'portable', 'prior', 'interval'):
                self.assertNotIn(banned, p['positive_text'].lower())

    def test_known_polarity_direction_and_margin_auc(self):
        rows, refs = fixture()
        result = method.summarize(rows, refs, min_per_class=1)
        metrics = result['test']['findings']['Edema']['templates']['mean_all_three_predeclared']
        self.assertEqual(metrics['reference_polarity_win_rate'], 1)
        self.assertEqual(metrics['balanced_polarity_win_rate'], 1)
        self.assertEqual(metrics['auroc'], 1)

    def test_unknown_and_uncertain_remain_excluded_with_denominators(self):
        rows, refs = fixture()
        finding = method.summarize(rows, refs, min_per_class=1)['val']['findings']['Pneumonia']
        self.assertEqual(finding['full_case_denominator'], 4)
        self.assertEqual(finding['known_reference_checks'], 2)
        self.assertEqual(finding['excluded_unknown'], 1)
        self.assertEqual(finding['excluded_uncertain'], 1)
        self.assertEqual(finding['known_reference_coverage'], 0.5)

    def test_insufficient_support_does_not_fabricate_auc(self):
        rows, refs = fixture()
        finding = method.summarize(rows, refs)['val']['findings']['Pneumothorax']
        self.assertIsNone(finding['templates']['shows_no']['auroc'])
        self.assertEqual(finding['templates']['shows_no']['reference_polarity_win_rate'], 1)

    def test_unknown_only_refs_do_not_become_fake_negatives(self):
        rows, refs = fixture(('unknown', 'uncertain'))
        finding = method.summarize(rows, refs, min_per_class=1)['val']['findings']['Edema']
        for metrics in finding['templates'].values():
            self.assertEqual(metrics['known_reference_checks'], 0)
            self.assertIsNone(metrics['reference_polarity_win_rate'])
            self.assertIsNone(metrics['auroc'])

    def test_ties_are_not_wins_or_above_chance_auc(self):
        rows, refs = fixture(('positive', 'negative'))
        for row in rows:
            for pair in row['score_pairs'].values():
                pair['positive_cosine'] = pair['negative_cosine'] = 0.5
        metrics = method.summarize(rows, refs, min_per_class=1)['test']['findings']['Cardiomegaly']['templates']['shows_no']
        self.assertEqual(metrics['reference_polarity_win_rate'], 0)
        self.assertEqual(metrics['tie_rate'], 1)
        self.assertEqual(metrics['auroc'], 0.5)

    def test_wrong_polarity_yields_zero_not_flipped_metric(self):
        rows, refs = fixture(('positive', 'negative'))
        for row in rows:
            for pair in row['score_pairs'].values():
                pair['positive_cosine'], pair['negative_cosine'] = pair['negative_cosine'], pair['positive_cosine']
        metrics = method.summarize(rows, refs, min_per_class=1)['test']['findings']['Consolidation']['templates']['shows_no']
        self.assertEqual(metrics['reference_polarity_win_rate'], 0)
        self.assertEqual(metrics['auroc'], 0)

    def test_all_templates_retained_including_predeclared_mean(self):
        rows, refs = fixture()
        findings = method.summarize(rows, refs, min_per_class=1)['val']['findings']
        for finding in findings.values():
            self.assertEqual(set(finding['templates']), {*method.FAMILIES, 'mean_all_three_predeclared'})

    def test_invalid_cosines_are_rejected(self):
        for value in (float('nan'), float('inf'), 2.0, True, '0.1'):
            rows, refs = fixture()
            rows[0]['score_pairs'][method.probes()[0]['probe_id']]['positive_cosine'] = value
            with self.assertRaisesRegex(ValueError, 'invalid_cosine_value'):
                method.summarize(rows, refs)

    def test_duplicate_missing_or_private_fields_fail_closed(self):
        rows, refs = fixture()
        for changed in ([rows[0], *rows], rows[:-1], [{**rows[0], 'report_path':'invented'}, *rows[1:]]):
            with self.assertRaises(ValueError):
                method.summarize(changed, refs)

    def test_bad_state_and_incomplete_probe_inventory_fail(self):
        rows, refs = fixture()
        refs[rows[0]['case_id']]['Edema'] = 'not_a_state'
        with self.assertRaisesRegex(ValueError, 'invalid_reference_state'):
            method.summarize(rows, refs)
        rows, refs = fixture()
        rows[0]['score_pairs'].pop(method.probes()[0]['probe_id'])
        with self.assertRaisesRegex(ValueError, 'invalid_score_inventory'):
            method.summarize(rows, refs)

    def test_score_rows_never_require_or_export_reference_fields(self):
        rows, refs = fixture()
        self.assertTrue(all(set(row) == {'case_id','split','image_sha256','score_pairs'} for row in rows))
        method.summarize(rows, refs)

    def test_slurm_approval_guard_precedes_real_paths(self):
        for env, allowed in (({}, False), ({}, True), ({'SLURM_JOB_ID':'123'}, False)):
            with patch.dict(os.environ, env, clear=True), patch.object(method, 'read_json') as source:
                with self.assertRaisesRegex(RuntimeError, 'explicit_image_polarity_slurm_required'):
                    method.run(SimpleNamespace(allow_real_image_polarity=allowed))
                source.assert_not_called()

    def test_fake_slurm_env_cannot_bypass_cgroup(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID':'999999'}, clear=True), patch.object(Path, 'read_text', return_value='/unrelated/job'):
            with self.assertRaisesRegex(RuntimeError, 'slurm_cgroup_required'):
                method.require_approved_slurm(SimpleNamespace(allow_real_image_polarity=True))

    def test_cached_cohort_binding_requires_exact_images_indices_splits(self):
        row = {'case_id':'case_0000','split':'val','source_row_index':0,'image_sha256':'1'*64}
        previous = {'schema_version':'tricompose-real-biovil-pair-check-v1',
                    'source_manifest_sha256':method.MANIFEST_SHA,'records':[row],
                    'producer':{'frozen':True}}
        reference = {'schema_version':'tricompose-real-xrv-weak-finding-check-v1',
                     'source_manifest_sha256':method.MANIFEST_SHA,'previous_scores_sha256':method.PREVIOUS_SHA,
                     'records':[{**row,'reference_states':dict.fromkeys(method.FINDINGS,'unknown')}]}
        method.validate_cached_cohort(previous, reference)
        for key, value in (('split','test'), ('image_sha256','2'*64), ('source_row_index',1)):
            changed = copy.deepcopy(reference);changed['records'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'fixed_cohort_membership_changed'):
                method.validate_cached_cohort(previous, changed)

    def test_balanced_win_exposes_always_positive_text_preference(self):
        rows, refs = fixture(('positive', 'negative'))
        for row in rows:
            for pair in row['score_pairs'].values():
                pair.update(positive_cosine=0.8, negative_cosine=0.2)
        m = method.summarize(rows, refs, min_per_class=1)['test']['findings']['Edema']['templates']['shows_no']
        self.assertEqual(m['positive_reference_win_rate'], 1)
        self.assertEqual(m['negative_reference_win_rate'], 0)
        self.assertEqual(m['balanced_polarity_win_rate'], 0.5)

    def test_auc_can_be_good_while_pairwise_polarity_fails(self):
        rows, refs = fixture(('positive', 'negative'))
        for row in rows:
            positive = refs[row['case_id']]['Edema'] == 'positive'
            for pair in row['score_pairs'].values():
                pair.update(positive_cosine=0.9 if positive else 0.5, negative_cosine=0.1)
        m = method.summarize(rows, refs, min_per_class=1)['test']['findings']['Edema']['templates']['shows_no']
        self.assertEqual(m['auroc'], 1)
        self.assertEqual(m['balanced_polarity_win_rate'], 0.5)

    def test_empty_inventory_and_absent_split_preserve_unavailable(self):
        result = method.summarize([], {})
        self.assertEqual(result['val']['cases'], 0)
        self.assertIsNone(result['test']['findings']['Edema']['templates']['shows_no']['auroc'])
        rows, refs = fixture(('positive',))
        result = method.summarize(rows, refs, min_per_class=1)
        self.assertIsNone(result['test']['findings']['Edema']['templates']['shows_no']['balanced_polarity_win_rate'])

    def test_summary_does_not_export_reference_rows_or_private_identifiers(self):
        rows, refs = fixture()
        rows[0]['case_id'] = 'invented_unique_private_key'
        refs[rows[0]['case_id']] = refs.pop('val_0')
        import json
        payload = json.dumps(method.summarize(rows, refs))
        self.assertNotIn('invented_unique_private_key', payload)
        self.assertNotIn('reference_states', payload)


if __name__ == '__main__':
    unittest.main()
