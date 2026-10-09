"""No-network/no-model tests for the fixed authored MedCPT diagnostic."""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest
from urllib.parse import urlsplit

from tricompose_v12 import medcpt_authored_probes as probes
from run_medcpt_authored_smoke import CONFIG, WORKSPACE, asset_inventory
from prepare_ratescore_assets import require_slurm


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.source = probes.probes()
        self.plan, self.texts = probes.inventory(self.source)

    def test_fixed_inventory(self):
        self.assertEqual(self.plan['pair_count'], 41)
        self.assertEqual(sum(p['input_nonempty'] for p in self.plan['pairs']), 40)
        self.assertEqual(self.plan['distinct_nonempty_texts'], 38)

    def test_six_complete_groups(self):
        for name, *_ in probes.BASES:
            kinds = {p['variant'] for p in self.plan['pairs'] if p['group'] == name}
            self.assertEqual(kinds, {'identity', 'paraphrase', 'negation', 'history',
                                     'hypothetical', 'different_finding'})

    def test_independent_repeat(self):
        self.assertEqual((self.plan, self.texts), probes.inventory(probes.probes()))
        self.source[0]['candidate'] = 'changed authored fixture'
        self.assertNotEqual(self.plan, probes.inventory(self.source)[0])

    def test_hashes_are_exact_utf8(self):
        for key, value in self.texts.items():
            self.assertEqual(key, hashlib.sha256(value.encode('utf-8')).hexdigest())

    def test_plan_has_no_raw_text(self):
        for row in self.plan['pairs']:
            self.assertNotIn('reference', row)
            self.assertNotIn('candidate', row)

    def test_no_clinical_or_repair_promotion(self):
        policy = self.plan['policy']
        self.assertIsNone(policy['clinical_score'])
        for key, value in policy.items():
            if key != 'clinical_score':
                self.assertIs(value, False)

    def test_schema_rejects_extra_fields(self):
        self.source[0]['patient_id'] = 'not a real identifier'
        with self.assertRaises(ValueError):
            probes.inventory(self.source)

    def test_nontext_rejected(self):
        self.source[0]['candidate'] = 3
        with self.assertRaises(ValueError):
            probes.inventory(self.source)

    def test_size_cap(self):
        self.source[0]['candidate'] = 'x' * 1025
        with self.assertRaises(ValueError):
            probes.inventory(self.source)

    def test_empty_has_null_not_zero(self):
        vectors = {key: [1.0, 0.0] for key in self.texts}
        row = probes.score_pairs(self.plan, vectors)[-1]
        self.assertEqual(row['status'], 'empty_input_not_comparable')
        self.assertIsNone(row['query_query_cosine'])

    def test_missing_not_dropped(self):
        rows = probes.score_pairs(self.plan, {})
        self.assertEqual(len(rows), 41)
        self.assertEqual(sum(r['status'] == 'unavailable_embedding' for r in rows), 40)
        self.assertTrue(all(r['query_query_cosine'] is None for r in rows))

    def test_invalid_not_promoted(self):
        rows = probes.score_pairs(self.plan, {key: [0.0, 0.0] for key in self.texts})
        self.assertEqual(sum(r['status'] == 'invalid_embedding' for r in rows), 40)
        self.assertTrue(all(r['clinical_score'] is None and not r['clinical_qualified']
                            and not r['regeneration_authorized'] for r in rows))

    def test_scoring_repeat(self):
        vectors = {key: [1.0, 2.0] for key in self.texts}
        rows = probes.score_pairs(self.plan, vectors)
        self.assertEqual(rows, probes.score_pairs(copy.deepcopy(self.plan), copy.deepcopy(vectors)))
        self.assertTrue(all(math.isclose(r['query_query_cosine'], 1.0)
                            for r in rows if r['status'] == 'complete'))

    def test_summary_retains_empty(self):
        summary = probes.diagnostic_summary(probes.score_pairs(self.plan, {}))
        self.assertEqual(summary['all_attempted_pairs'], 41)
        self.assertEqual(summary['status_counts']['empty_input_not_comparable'], 1)
        self.assertEqual(summary['cosine_by_variant'], {})
        for values in summary['authored_ordering_diagnostics_not_clinical_accuracy'].values():
            self.assertEqual(values['unavailable'], 6)

    def test_ordering_is_not_accuracy(self):
        rows = probes.score_pairs(self.plan, {key: [1.0, 0.0] for key in self.texts})
        summary = probes.diagnostic_summary(rows)
        for values in summary['authored_ordering_diagnostics_not_clinical_accuracy'].values():
            self.assertEqual(values['tie'], 6)
        self.assertFalse(summary['policy']['clinical_qualified'])


class CosineTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(probes.cosine([1, 0], [1, 0]), 1)
        self.assertEqual(probes.cosine([1, 0], [0, 1]), 0)
        self.assertEqual(probes.cosine([1, 0], [-1, 0]), -1)

    def test_shape_and_type_rejections(self):
        for left, right in (([], []), ([1], [1, 2]), ([True], [1]),
                            (['1'], [1]), ([1] * 4097, [1] * 4097), (None, [1])):
            with self.subTest(left_type=type(left).__name__):
                with self.assertRaises(ValueError):
                    probes.cosine(left, right)

    def test_nonfinite_and_zero_rejections(self):
        for vector in ([float('nan')], [float('inf')], [0]):
            with self.assertRaises(ValueError):
                probes.cosine(vector, [1])

    def test_scale_invariance(self):
        self.assertAlmostEqual(probes.cosine([1, 2, 3], [4, 5, 6]),
                               probes.cosine([10, 20, 30], [8, 10, 12]))


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())

    def test_fixed_inventory_and_cap(self):
        entries = asset_inventory(self.config)
        self.assertEqual(len(entries), 11)
        self.assertLess(sum(e['bytes'] for e in entries), 500000000)
        self.assertEqual(sum(e['bytes'] for e in entries if e['path'].endswith('.safetensors')),
                         437951328)

    def test_only_public_urls(self):
        for entry in asset_inventory(self.config):
            self.assertTrue(entry['url'].startswith(('https://huggingface.co/ncbi/',
                                                    'https://raw.githubusercontent.com/ncbi/')))
            self.assertNotIn('?', entry['url'])
            parsed = urlsplit(entry['url'])
            self.assertIsNone(parsed.username)
            self.assertIsNone(parsed.password)
            self.assertEqual(parsed.query, '')

    def test_wrong_checkpoint_rejected(self):
        for key, value in (('repository', 'unapproved/model'), ('revision', '0' * 40),
                           ('license', 'unconfirmed')):
            cfg = copy.deepcopy(self.config)
            cfg[key] = value
            with self.assertRaises(ValueError):
                asset_inventory(cfg)

    def test_wrong_source_rejected(self):
        self.config['source_documentation']['revision'] = '0' * 40
        with self.assertRaises(ValueError):
            asset_inventory(self.config)

    def test_traversal_rejected(self):
        self.config['files'][0]['path'] = '../escape'
        with self.assertRaises(ValueError):
            asset_inventory(self.config)

    def test_extra_weights_rejected(self):
        self.config['files'].append({'path': 'pytorch_model.bin', 'bytes': 1, 'sha256': '0' * 64})
        with self.assertRaises(ValueError):
            asset_inventory(self.config)

    def test_duplicate_rejected(self):
        self.config['files'][-1] = copy.deepcopy(self.config['files'][0])
        with self.assertRaises(ValueError):
            asset_inventory(self.config)

    def test_cap_cannot_expand(self):
        self.config['maximum_download_bytes'] = 1000000000
        with self.assertRaises(ValueError):
            asset_inventory(self.config)

    def test_size_and_hash_rejected(self):
        for key, value in (('bytes', True), ('bytes', 0), ('sha256', 'z' * 64)):
            cfg = copy.deepcopy(self.config)
            cfg['files'][0][key] = value
            with self.assertRaises(ValueError):
                asset_inventory(cfg)

    def test_no_scope_expansion(self):
        for key in ('train', 'clinical_qualified', 'replace_biolord_in_official_ratescore',
                    'selection_changed', 'regeneration_authorized'):
            cfg = copy.deepcopy(self.config)
            cfg[key] = True
            with self.assertRaises(ValueError):
                asset_inventory(cfg)

    def test_slurm_cgroup_not_environment_alone(self):
        require_slurm('12:cpu:/slurm/uid_1/job_12714150/step_batch/task_0\n', '12714150')
        for cgroup, job in (('12:cpu:/\n', '12714150'),
                            ('12:cpu:/slurm/job_127141500/step_batch\n', '12714150'),
                            ('12:cpu:/slurm/job_12714150/step_batch\n', '')):
            with self.assertRaises(ValueError):
                require_slurm(cgroup, job)

    def test_existing_cpu_entry(self):
        script = (WORKSPACE / 'TriCompose-v1.2/slurm/62_medcpt_existing_cpu.sh').read_text()
        self.assertNotIn('#SBATCH', script)
        self.assertNotIn('sbatch ', script)
        self.assertIn("'/job_12714150/'", script)
        self.assertIn('CUDA_VISIBLE_DEVICES', script)
        self.assertIn('HF_HUB_DISABLE_IMPLICIT_TOKEN=1', script)
        self.assertIn('NETRC=/dev/null', script)
        self.assertIn('--approved-download-and-authored-cpu-smoke', script)


if __name__ == '__main__':
    unittest.main()
