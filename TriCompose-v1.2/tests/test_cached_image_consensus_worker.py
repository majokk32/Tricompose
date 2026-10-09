"""Invented cached scores only; no image, source record or model access."""
import copy
import hashlib
import unittest
from unittest.mock import patch

import benchmark_cached_image_consensus as worker
from tricompose_v12.image_reader_consensus import analyze


def fixture():
    xrv = {'records': [], 'outcomes': []}
    biovil = {'records': [], 'outcomes': []}
    manifest = {'artifacts': {}}
    for i in range(50):
        cid = f'case_{i:03d}'
        digest = hashlib.sha256(f'invented-display-{i}'.encode()).hexdigest()
        positive = i < 25
        xrv['records'].append({'case_id': cid, 'display_sha256': digest,
            'reference_state': 'positive' if positive else 'negative',
            'direct_lung_opacity_score': .8 if positive else .2,
            'legacy_max_proxy_score': .99})
        biovil['records'].append({'case_id': cid, 'source_image_sha256': digest,
            'score_pairs': {f: {'positive_cosine': .3 if positive else .1,
                                'negative_cosine': .1 if positive else .3}
                            for f in worker.FAMILIES}})
        for scores in (xrv, biovil):
            scores['outcomes'].append({'case_id': cid, 'status': 'scored'})
        manifest['artifacts']['displays/' + cid + '.png'] = digest
    return xrv, biovil, manifest


class CachedImageJoinTests(unittest.TestCase):
    def test_complete_inventory_and_fixed_class_balance(self):
        rows = worker.cached_join(*fixture())
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(r['reference_state'] == 'positive' for r in rows), 25)
        self.assertEqual(sum(r['reference_state'] == 'negative' for r in rows), 25)
        self.assertEqual(len({r['image_sha256'] for r in rows}), 50)

    def test_exact_head_not_legacy_proxy(self):
        rows = worker.cached_join(*fixture())
        self.assertEqual(rows[-1]['xrv_score'], .2)

    def test_family_order_fixed_and_all_templates_retained(self):
        x, b, m = fixture()
        values = (.05, .15, -.25)
        b['records'][0]['score_pairs'] = {
            f: {'positive_cosine': v, 'negative_cosine': 0.}
            for f, v in reversed(tuple(zip(worker.FAMILIES, values)))}
        self.assertEqual(worker.cached_join(x, b, m)[0]['margins'], list(values))

    def test_input_not_mutated(self):
        f = fixture()
        original = copy.deepcopy(f)
        worker.cached_join(*f)
        self.assertEqual(f, original)

    def test_missing_case_not_dropped(self):
        x, b, m = fixture()
        b['records'].pop()
        with self.assertRaises(ValueError):
            worker.cached_join(x, b, m)

    def test_duplicate_case_rejected(self):
        x, b, m = fixture()
        x['records'][-1] = copy.deepcopy(x['records'][0])
        with self.assertRaises(ValueError):
            worker.cached_join(x, b, m)

    def test_bound_image_hashes_must_match(self):
        for changed in ('biovil', 'manifest'):
            x, b, m = fixture()
            if changed == 'biovil':
                b['records'][0]['source_image_sha256'] = 'a' * 64
            else:
                m['artifacts']['displays/case_000.png'] = 'a' * 64
            with self.assertRaises(ValueError):
                worker.cached_join(x, b, m)

    def test_failed_original_attempt_not_a_scored_case(self):
        x, b, m = fixture()
        b['outcomes'][0]['status'] = 'failed'
        with self.assertRaises(ValueError):
            worker.cached_join(x, b, m)

    def test_reference_cannot_be_supplied_by_reader_vote(self):
        x, b, m = fixture()
        x['records'][0]['reference_state'] = 'negative'
        with self.assertRaises(ValueError):
            worker.cached_join(x, b, m)

    def test_cannot_choose_favorable_template(self):
        x, b, m = fixture()
        b['records'][0]['score_pairs'].pop(worker.FAMILIES[0])
        with self.assertRaises(ValueError):
            worker.cached_join(x, b, m)

    def test_nonfinite_cosine_rejected(self):
        x, b, m = fixture()
        b['records'][0]['score_pairs'][worker.FAMILIES[0]]['positive_cosine'] = float('nan')
        with self.assertRaises(ValueError):
            worker.cached_join(x, b, m)


class CachedImageTableTests(unittest.TestCase):
    def test_all_counts_and_paired_metric_rows_kept(self):
        result = analyze(worker.cached_join(*fixture()), repetitions=100)
        rows, pairs = worker.tables(result)
        self.assertEqual((len(rows), len(pairs)), (4, 21))
        self.assertEqual(rows[0]['error_risk_denominator'], 50)
        self.assertEqual(rows[0]['proposal_coverage'], 1.)
        self.assertTrue(all(r['shared_paired_draws'] for r in pairs))
        self.assertTrue(all(r['clinical_qualified'] is False for r in rows))
        self.assertTrue(all(r['official_adjudication_reproduced'] is False for r in rows))

    def test_null_risk_not_zero(self):
        rows = worker.cached_join(*fixture())
        for r in rows:
            r['margins'] = [1., -1., 0.]
        table, _ = worker.tables(analyze(rows, repetitions=100))
        self.assertIsNone(table[1]['accepted_error_risk'])
        self.assertIsNone(table[1]['risk_interval_lower'])
        self.assertEqual(table[1]['risk_zero_denominator_draws'], 100)

    def test_original_fraction_not_reweighted(self):
        rows = worker.cached_join(*fixture())
        rows[-1]['xrv_score'] = .8
        table, pairs = worker.tables(analyze(rows, repetitions=100))
        self.assertEqual(table[0]['error_risk_numerator'], 1)
        self.assertEqual(table[0]['error_risk_denominator'], 50)
        self.assertEqual(table[0]['accepted_error_risk'], 1 / 50)
        self.assertEqual(table[2]['proposal_coverage'], 49 / 50)
        contrast = next(r for r in pairs if r['left'] == 'xrv_exact_0_5'
                        and r['metric'] == 'accepted_error_risk')
        self.assertEqual(contrast['right_minus_left'], -1 / 50)

    def test_actual_cgroup_required_before_source_access(self):
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', {'SLURM_JOB_ID': '12784259'}), \
             patch.object(worker.Path, 'read_text', return_value='/login_node'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_other_job_not_current_authority(self):
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', {'SLURM_JOB_ID': '111'}), \
             patch.object(worker.Path, 'read_text', return_value='/slurm/job_111/step_batch'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()

    def test_gpu_allocation_not_current_authority(self):
        env = {'SLURM_JOB_ID': '12784259', 'SLURM_JOB_GPUS': '0'}
        with patch.object(worker.sys, 'argv', ['worker']), \
             patch.object(worker.os, 'environ', env), \
             patch.object(worker.Path, 'read_text', return_value='/slurm/job_12784259/step_batch'), \
             patch.object(worker, 'execute') as execute, patch('builtins.print'):
            self.assertEqual(worker.main(), 1)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
