"""Wholly invented fixtures: no actual author/patient input or neural calls."""
import copy
import contextlib
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_radeval_expert import row
from tricompose_v12.radeval_expert import inventory
from tricompose_v12.radeval_medcpt_benchmark import join, evaluate, choices, METRIC, BIOVIL, RADGRAPH, FULL_METRICS, IMAGE_METRICS
from score_radeval_medcpt import recover_texts, prediction_rows, encode, execute, WORKSPACE


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(i) for i in range(4)]
        for i, value in enumerate(self.rows):
            value['images_path'] = f'/invented/p{90000000+i}/s{90000010+i}/fixture.jpg'
        self.plan = inventory(self.rows)
        self.med = [{'item_id': p['item_id'], 'status': 'complete', 'value': 0.8 - p['candidate_slot'] * .2}
                    for p in self.plan['records']]
        self.rg = [{'item_id': p['item_id'], 'status': 'complete', 'scores': {m: .5 for m in RADGRAPH}}
                   for p in self.plan['records']]
        self.bio = [{'item_id': p['item_id'], 'status': 'complete', 'value': -.1}
                    for p in self.plan['records']]

    def joined(self):
        return join(self.plan, self.med, self.rg, self.bio)

    def result(self):
        return evaluate(self.joined(), resamples=20, seed=0)

    def test_all_attempted_retained(self):
        self.assertEqual(len(self.joined()), 12)
        self.med[0].update(status='failed_embedding', value=None)
        self.assertEqual(len(self.joined()), 12)

    def test_same_full_cohort_mask(self):
        self.med[0].update(status='failed_embedding', value=None)
        result = self.result()['cohorts']['full_text_common']
        self.assertEqual(result['common_score_available_pairs'], 11)
        self.assertTrue(all(m['outcomes']['clinically_significant_total']['paired_rows'] == 11
                            for m in result['metrics'].values()))

    def test_same_image_cohort_mask(self):
        self.bio[0].update(status='unavailable_source_image', value=None)
        result = self.result()['cohorts']
        self.assertEqual(result['full_text_common']['common_score_available_pairs'], 12)
        self.assertEqual(result['cached_image_common']['common_score_available_pairs'], 11)
        self.assertTrue(all(m['outcomes']['clinically_significant_total']['paired_rows'] == 11
                            for m in result['cached_image_common']['metrics'].values()))

    def test_metric_inventory_not_tuned(self):
        result = self.result()['cohorts']
        self.assertEqual(set(result['full_text_common']['metrics']), set(FULL_METRICS))
        self.assertEqual(set(result['cached_image_common']['metrics']), set(IMAGE_METRICS))

    def test_known_correlation_and_error_delta(self):
        result = self.result()['cohorts']['full_text_common']['metrics'][METRIC]
        self.assertAlmostEqual(result['outcomes']['clinically_significant_total']['spearman'], 1)
        selection = result['selection_diagnostic']['clinically_significant_total']
        self.assertEqual(selection['means']['selected_expected_errors'], 1)
        self.assertEqual(selection['means']['metric_minus_random_errors'], -1)
        self.assertEqual(selection['error_delta_cluster_ci']['interval95'], [-1, -1])

    def test_negative_cosine_valid(self):
        self.med[0]['value'] = -.7
        self.assertEqual(self.joined()[0]['scores'][METRIC], -.7)

    def test_nan_rejected(self):
        self.med[0]['value'] = float('nan')
        with self.assertRaises(ValueError):
            self.joined()

    def test_bool_is_not_score(self):
        self.med[0]['value'] = True
        with self.assertRaises(ValueError):
            self.joined()

    def test_out_of_range_rejected(self):
        self.med[0]['value'] = 1.2
        with self.assertRaises(ValueError):
            self.joined()

    def test_failed_numeric_rejected(self):
        self.med[0]['status'] = 'over_capacity'
        with self.assertRaises(ValueError):
            self.joined()

    def test_unknown_status_rejected(self):
        self.med[0]['status'] = 'normal'
        with self.assertRaises(ValueError):
            self.joined()

    def test_join_reordering_rejected(self):
        self.med.reverse()
        with self.assertRaises(ValueError):
            self.joined()

    def test_duplicate_pair_rejected(self):
        for source in (self.plan['records'], self.med, self.rg, self.bio):
            source[1]['item_id'] = source[0]['item_id']
        with self.assertRaises(ValueError):
            self.joined()

    def test_missing_annotation_remains_missing(self):
        self.plan['records'][0]['errors']['clinically_significant'][0] = None
        records = self.joined()
        self.assertIsNone(records[0]['expert_outcomes']['clinically_significant_total'])
        result = self.result()['cohorts']['full_text_common']['metrics'][METRIC]
        self.assertEqual(result['outcomes']['clinically_significant_total']['paired_rows'], 11)
        self.assertEqual(result['selection_diagnostic']['clinically_significant_total']['complete_anchors'], 3)

    def test_one_bad_candidate_invalidates_anchor_for_all_metrics(self):
        self.med[0].update(status='failed_embedding', value=None)
        result = self.result()['cohorts']['full_text_common']['metrics']
        for metric in result.values():
            for target in ('clinically_significant_total', 'all_errors_total'):
                item = metric['selection_diagnostic'][target]
                self.assertEqual(item['attempted_anchors'], 4)
                self.assertEqual(item['complete_anchors'], 3)
                self.assertEqual(len(item['anchor_records']), 4)

    def test_ties_uniform_expected(self):
        for r in self.med:
            r['value'] = .4
        item = self.result()['cohorts']['full_text_common']['metrics'][METRIC]['selection_diagnostic']['clinically_significant_total']
        self.assertEqual(item['means']['metric_minus_random_errors'], 0)
        self.assertEqual(item['expected_pairwise_accuracy'], .5)

    def test_cluster_count_not_candidate_count(self):
        item = self.result()['cohorts']['full_text_common']['metrics'][METRIC]['outcomes']['clinically_significant_total']
        self.assertEqual(item['cluster_bootstrap']['source_groups'], 4)

    def test_no_image_available_preserves_full_cohort(self):
        for r in self.bio:
            r.update(status='unavailable_source_image', value=None)
        result = self.result()['cohorts']
        self.assertEqual(result['full_text_common']['common_score_available_pairs'], 12)
        item = result['cached_image_common']['metrics'][METRIC]['outcomes']['clinically_significant_total']
        self.assertEqual(item['status'], 'insufficient_pairs')
        self.assertIsNone(item['spearman'])

    def test_all_failed_retained_not_zero(self):
        for r in self.med:
            r.update(status='over_capacity', value=None)
        result = self.result()['cohorts']['full_text_common']['metrics'][METRIC]
        self.assertIsNone(result['outcomes']['clinically_significant_total']['spearman'])
        self.assertIsNone(result['selection_diagnostic']['clinically_significant_total']['means']['selected_expected_errors'])

    def test_policy_not_promoted(self):
        result = self.result()
        self.assertTrue(all(v is False for v in result['policy'].values()))
        self.assertFalse(result['untouched_final_test'])

    def test_anchor_wrong_slot_rejected(self):
        records = self.joined()
        records[0]['candidate_slot'] = 2
        with self.assertRaises(ValueError):
            choices(records, METRIC, 'clinically_significant_total', 'full_common_available')

    def test_missing_native_graph_masks_comparison(self):
        self.rg[0].update(status='unavailable_graph', scores={m: None for m in RADGRAPH})
        self.assertFalse(self.joined()[0]['full_common_available'])
        self.assertEqual(self.result()['cohorts']['full_text_common']['common_score_available_pairs'], 11)

    def test_invalid_graph_score_rejected(self):
        self.rg[0]['scores'][RADGRAPH[0]] = -.1
        with self.assertRaises(ValueError):
            self.joined()

    def test_cached_image_failure_not_zero(self):
        self.bio[0].update(status='failed_image_or_text', value=None)
        records = self.joined()
        self.assertIsNone(records[0]['scores'][BIOVIL])
        self.assertFalse(records[0]['image_common_available'])

    def test_no_input_mutation(self):
        before = copy.deepcopy((self.plan, self.med, self.rg, self.bio))
        self.result()
        self.assertEqual(before, (self.plan, self.med, self.rg, self.bio))


class WorkerInputTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(0), row(1)]
        self.plan = inventory(self.rows)
        self.texts = recover_texts(self.rows, self.plan)
        self.receipts = {k: {'status': 'complete'} for k in self.texts}
        self.vectors = {k: [1.0, 2.0] for k in self.texts}

    def test_exact_text_recovery(self):
        self.assertEqual(len(self.texts), len(self.plan['graphs']))
        for key, text in self.texts.items():
            self.assertEqual(key, hashlib.sha256(text.encode()).hexdigest())

    def test_recovery_rejects_modified_text(self):
        self.rows[0]['prediction1'] = 'Different invented fixture.'
        with self.assertRaises(ValueError):
            recover_texts(self.rows, self.plan)

    def test_recovery_rejects_changed_row_count(self):
        with self.assertRaises(ValueError):
            recover_texts(self.rows[:-1], self.plan)

    def test_native_cosine_from_exact_vector_links(self):
        rows = prediction_rows(self.plan, self.vectors, self.receipts)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(abs(r['value'] - 1) < 1e-12 for r in rows))

    def test_over_capacity_not_prefix_score(self):
        key = self.plan['records'][0]['hypothesis_sha256']
        self.receipts[key]['status'] = 'over_capacity'
        row0 = prediction_rows(self.plan, self.vectors, self.receipts)[0]
        self.assertEqual(row0['status'], 'over_capacity')
        self.assertIsNone(row0['value'])

    def test_failed_embedding_retained(self):
        key = self.plan['records'][0]['hypothesis_sha256']
        self.receipts[key]['status'] = 'failed_embedding'
        self.vectors.pop(key)
        rows = prediction_rows(self.plan, self.vectors, self.receipts)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['status'], 'failed_embedding')

    def test_empty_not_encoded(self):
        self.plan['records'][0]['input_nonempty'] = False
        self.vectors.pop(self.plan['records'][0]['hypothesis_sha256'])
        rows = prediction_rows(self.plan, self.vectors, self.receipts)
        self.assertEqual(rows[0]['status'], 'empty_input')
        self.assertIsNone(rows[0]['value'])

    def test_missing_vector_not_silent_drop(self):
        self.vectors.pop(self.plan['records'][0]['hypothesis_sha256'])
        rows = prediction_rows(self.plan, self.vectors, self.receipts)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['status'], 'failed_embedding')
        self.assertIsNone(rows[0]['value'])

    def test_approval_guard_before_any_io(self):
        with patch('pathlib.Path.read_text', side_effect=AssertionError('unexpected IO')):
            with self.assertRaises(ValueError):
                execute(False, io.StringIO())

    def test_entry_cpu_offline_and_separate_scope(self):
        text = (WORKSPACE / 'TriCompose-v1.2/slurm/63_radeval_medcpt_existing_cpu.sh').read_text()
        self.assertNotIn('#SBATCH', text)
        self.assertIn("'/job_12714150/'", text)
        self.assertIn('HF_HUB_OFFLINE=1', text)
        self.assertIn('CUDA_VISIBLE_DEVICES', text)
        self.assertIn('NETRC=/dev/null', text)
        self.assertIn('--approved-expert-report-cpu-execution', text)


class FakeRow:
    def __init__(self, values):
        self.values = values

    def bool(self):
        return [bool(v) for v in self.values]

    def __getitem__(self, mask):
        return FakeRow([v for v, use in zip(self.values, mask) if use])

    def tolist(self):
        return self.values


class FakeArray:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def __getitem__(self, index):
        return FakeRow(self.values[index])


class FakeVectors:
    def __init__(self, count):
        self.values = [[1.0] * 768 for _ in range(count)]
        self.shape = (count, 768)

    def __getitem__(self, index):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def fake_tokenizer(value, **kwargs):
    if isinstance(value, str):
        return {'input_ids': list(range(len(value)))}
    cap = max(len(v) for v in value)
    return {'input_ids': FakeArray([list(range(len(v))) + [0] * (cap - len(v)) for v in value]),
            'attention_mask': FakeArray([[1] * len(v) + [0] * (cap - len(v)) for v in value])}


class EncodingFixtureTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
            isfinite=lambda v: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)))
        self.model_calls = 0

    def model(self, **encoded):
        self.model_calls += 1
        return SimpleNamespace(last_hidden_state=FakeVectors(encoded['input_ids'].shape[0]))

    def test_fulltext_capacity_not_truncation(self):
        texts = {'fixture_short': 'authored', 'fixture_oversize': 'x' * 513, 'fixture_empty': ''}
        vectors, receipts, calls, replay = encode(texts, fake_tokenizer, self.model, self.torch, io.StringIO())
        self.assertEqual(set(vectors), {'fixture_short'})
        self.assertEqual(receipts['fixture_oversize']['status'], 'over_capacity')
        self.assertEqual(receipts['fixture_empty']['status'], 'empty_input')
        self.assertEqual(receipts['fixture_short']['status'], 'complete')
        self.assertTrue(all(r['truncated'] is False for r in receipts.values()))
        self.assertEqual(calls, self.model_calls)
        self.assertEqual(calls, 2)
        self.assertTrue(replay['exact_match'])

    def test_replay_only_fixed_first_eight(self):
        texts = {f'fixture_{i:03d}': f'authored_{i:03d}' for i in range(9)}
        vectors, receipts, calls, replay = encode(texts, fake_tokenizer, self.model, self.torch, io.StringIO())
        self.assertEqual(len(vectors), 9)
        self.assertEqual(calls, 3)
        self.assertEqual(replay['attempted_texts'], 8)
        self.assertEqual(replay['scope'], 'first_8_hash_sorted_eligible_texts_only')
        self.assertTrue(replay['exact_match'])

    def test_failed_batch_preserves_every_input(self):
        def broken(**encoded):
            raise RuntimeError('invented fixture failure')
        texts = {'fixture_short': 'authored'}
        vectors, receipts, calls, replay = encode(texts, fake_tokenizer, broken, self.torch, io.StringIO())
        self.assertEqual(vectors, {})
        self.assertEqual(receipts['fixture_short']['status'], 'failed_embedding')
        self.assertEqual(receipts['fixture_short']['failure_type'], 'RuntimeError')
        self.assertEqual(calls, 1)
        self.assertIsNone(replay['exact_match'])

    def test_exact_capacity_is_allowed(self):
        texts = {'fixture_at_capacity': 'x' * 512}
        vectors, receipts, calls, replay = encode(texts, fake_tokenizer, self.model, self.torch, io.StringIO())
        self.assertEqual(receipts['fixture_at_capacity']['native_token_count'], 512)
        self.assertEqual(receipts['fixture_at_capacity']['status'], 'complete')
        self.assertEqual(len(vectors), 1)


if __name__ == '__main__':
    unittest.main()
