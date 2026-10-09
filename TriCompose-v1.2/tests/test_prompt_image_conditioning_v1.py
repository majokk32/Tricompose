"""Invented lineage/numeric/token fixtures only; no bodies, weights or inference."""
import copy
import csv
import hashlib
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import score_prompt_image_conditioning_v1 as m


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(cases=2):
    rows = []
    for case in range(cases):
        for model in m.MODELS:
            for expert in sorted(m.EXPERTS):
                image_id = f'image_{case}_{model}'
                row = {'case_id': f'case_fixture_{case}',
                    'triple_candidate_id': f'triple_{case}_{model}_{expert}',
                    'cxr_candidate_id': image_id, 'cxr_model_id': model, 'cxr_seed': '0',
                    'report_candidate_id': f'report_{case}_{model}_{expert}', 'report_model_id': expert,
                    'ehr_sha256': digest(f'ehr_{case}'), 'ehr_facts_sha256': digest(f'facts_{case}'),
                    'prompt_sha256': digest(f'prompt_{case}_{model}'), 'cxr_sha256': digest(image_id),
                    'report_sha256': digest(f'report_{case}_{model}_{expert}'),
                    'prompt_path': str(m.PROTECTED_ROOT / f'fixture/{image_id}.txt'),
                    'cxr_path': str(m.PROTECTED_ROOT / f'fixture/{image_id}.png'),
                    'historical_static_selected': 'False', 'old_raw_score': '0.31415'}
                rows.append(row)
    return rows


def retrieval_record(image, own='tokens_fixture_a', scores=None):
    scores = scores or {own: .8, 'tokens_fixture_b': .1}
    return {**image, 'prompt_group_id': own, **m.retrieval(scores, own)}


class ConditioningTests(unittest.TestCase):
    def test_protocol_is_diagnostic_not_action_authority(self):
        p = m.protocol()
        self.assertTrue(p['post_hoc_development'])
        for k in ('clinical_qualified', 'regeneration_authorized', 'selection_changed',
                  'training_allowed', 'external_api_allowed', 'native_structured_ehr_to_cxr',
                  'different_prompt_is_clinical_negative', 'untouched_final_test'):
            self.assertIs(p[k], False)

    def test_metadata_projection_reads_no_referenced_artifact(self):
        rows = fixture(); before = copy.deepcopy(rows)
        with patch.object(Path, 'read_text', side_effect=AssertionError('no body read')):
            images = m.images_from_index(rows, full=False)
        self.assertEqual(len(images), 6)
        self.assertEqual(rows, before)
        self.assertEqual(set(images[0]), set(m.IMAGE_FIELDS))
        self.assertNotIn('report_path', images[0])

    def test_complete_80_3_4_inventory(self):
        rows = fixture(80)
        images = m.images_from_index(rows)
        self.assertEqual((len(rows), len(images)), (960, 240))

    def test_incomplete_grid_refused(self):
        with self.assertRaisesRegex(ValueError, 'complete_fixed80'):
            m.images_from_index(fixture())

    def test_duplicate_triple_refused(self):
        rows = fixture(); rows.append(rows[0])
        with self.assertRaisesRegex(ValueError, 'unique_triple'):
            m.images_from_index(rows, full=False)

    def test_duplicate_expert_slot_refused(self):
        rows = fixture(); row = copy.deepcopy(rows[0])
        row.update(triple_candidate_id='new_triple', report_candidate_id='new_report')
        with self.assertRaisesRegex(ValueError, 'duplicate_generator_expert'):
            m.images_from_index(rows + [row], full=False)

    def test_fixed_ehr_cannot_change_for_expert(self):
        rows = fixture(); rows[1]['ehr_sha256'] = digest('changed_ehr')
        with self.assertRaisesRegex(ValueError, 'fixed_ehr'):
            m.images_from_index(rows, full=False)

    def test_one_image_cannot_have_two_final_prompts(self):
        rows = fixture(); rows[1]['prompt_sha256'] = digest('changed_prompt')
        with self.assertRaisesRegex(ValueError, 'one_image_one_final_prompt'):
            m.images_from_index(rows, full=False)

    def test_unprotected_prompt_path_refused(self):
        rows = fixture(); rows[0]['prompt_path'] = str(m.WORKSPACE / 'public.txt')
        with self.assertRaisesRegex(ValueError, 'protected'):
            m.images_from_index(rows, full=False)

    def test_missing_sha_refused(self):
        rows = fixture(); rows[0]['prompt_sha256'] = ''
        with self.assertRaisesRegex(ValueError, 'sha256_lineage'):
            m.images_from_index(rows, full=False)

    def test_inventory_does_not_count_reports_as_images(self):
        inv = m.inventory(m.images_from_index(fixture(), full=False))
        self.assertEqual((inv['image_slots'], inv['cases']), (6, 2))
        self.assertEqual(inv['by_generator'][m.MODELS[0]]['slots'], 2)

    def test_token_hash_uses_actual_unpadded_ids(self):
        self.assertEqual(m.token_sequence_hash([1, 2]), m.token_sequence_hash([1, 2]))
        self.assertNotEqual(m.token_sequence_hash([1, 2]), m.token_sequence_hash([1, 2, 0]))
        for ids in ([], [True], [-1], ['2']):
            with self.assertRaises(ValueError):
                m.token_sequence_hash(ids)

    def test_critic_token_collision_merges_byte_distinct_prompts(self):
        images = m.images_from_index(fixture(), full=False)
        receipts = {i['prompt_sha256']: {'status': 'encoded',
                    'token_sequence_sha256': digest('same_tokens')} for i in images}
        groups = m.make_groups(images, receipts)
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(len(g['prompt_sha256s']) == 2 for g in groups))
        self.assertTrue(all(len(g['case_ids']) == 2 for g in groups))

    def test_missing_text_record_refused_not_discarded(self):
        images = m.images_from_index(fixture(), full=False)
        with self.assertRaisesRegex(ValueError, 'every_bound_prompt'):
            m.make_groups(images, {})

    def test_failed_text_keeps_explicit_group(self):
        images = m.images_from_index(fixture(), full=False)
        receipts = {i['prompt_sha256']: {'status': 'failed_without_replacement',
                    'token_sequence_sha256': None} for i in images}
        groups = m.make_groups(images, receipts)
        self.assertEqual(len(groups), 6)
        self.assertTrue(all(g['status'] == 'failed_without_replacement' for g in groups))

    def test_matched_prompt_ranks_first(self):
        row = m.retrieval({'own': .8, 'other': .1}, 'own')
        self.assertEqual((row['best_rank'], row['worst_rank']), (1, 1))
        self.assertEqual(row['expected_recall_at_1'], 1)
        self.assertAlmostEqual(row['matched_minus_other_mean'], .7)

    def test_worse_match_is_not_positive_by_definition(self):
        row = m.retrieval({'own': .1, 'other': .8}, 'own')
        self.assertEqual(row['expected_recall_at_1'], 0)
        self.assertEqual(row['best_rank'], 2)
        self.assertAlmostEqual(row['matched_minus_other_mean'], -.7)

    def test_all_ties_cannot_be_called_perfect_retrieval(self):
        row = m.retrieval({str(i): .2 for i in range(10)}, '0')
        self.assertEqual((row['best_rank'], row['worst_rank']), (1, 10))
        self.assertAlmostEqual(row['expected_recall_at_1'], .1)
        self.assertAlmostEqual(row['expected_recall_at_5'], .5)
        self.assertAlmostEqual(row['expected_reciprocal_rank'], sum(1/i for i in range(1, 11))/10)

    def test_rank5_boundary_ties_fractional_not_optimistic(self):
        scores = {'own': .5, 'tie': .5, 'tie2': .5, **{str(i): .8 for i in range(4)}}
        row = m.retrieval(scores, 'own')
        self.assertEqual((row['best_rank'], row['worst_rank']), (5, 7))
        self.assertAlmostEqual(row['expected_recall_at_5'], 1/3)

    def test_missing_prompt_cosine_keeps_no_rank(self):
        row = m.retrieval({'own': .5, 'other': None}, 'own')
        self.assertEqual(row['matched_cosine'], .5)
        self.assertIsNone(row['best_rank'])
        self.assertIsNone(row['matched_minus_other_mean'])
        self.assertEqual(row['available_prompt_groups'], 1)

    def test_single_group_is_non_discriminating(self):
        row = m.retrieval({'own': .5}, 'own')
        self.assertEqual(row['status'], 'single_prompt_group_not_discriminating')
        self.assertIsNone(row['expected_recall_at_1'])

    def test_invalid_cosines_refused(self):
        for value in (True, float('nan'), float('inf'), 2, '0.4'):
            with self.assertRaisesRegex(ValueError, 'finite_cosines'):
                m.retrieval({'own': value, 'other': .1}, 'own')

    def test_group_balancing_and_duplicate_images(self):
        base = m.images_from_index(fixture(), full=False)[0]
        a = retrieval_record(base, 'group_a')
        b = retrieval_record({**base, 'cxr_sha256': digest('image_b')}, 'group_b',
                             {'group_b': .1, 'group_a': .8})
        _, reduced = m.reduce_groups([a]*20 + [b])
        result = reduced[base['cxr_model_id']]
        self.assertEqual(result['planned_prompt_groups'], 2)
        self.assertAlmostEqual(result['expected_recall_at_1'], .5)
        self.assertAlmostEqual(result['uniform_random_group_recall_at_1'], .5)

    def test_changed_duplicate_score_refused(self):
        image = m.images_from_index(fixture(), full=False)[0]
        a = retrieval_record(image); b = copy.deepcopy(a); b['matched_cosine'] = .7
        with self.assertRaisesRegex(ValueError, 'same_image_and_encoder_group'):
            m.reduce_groups([a, b])

    def test_missing_group_coverage_explicit_not_zero(self):
        image = m.images_from_index(fixture(), full=False)[0]
        a = retrieval_record(image, 'group_a')
        b = retrieval_record({**image, 'cxr_sha256': digest('failed_image')}, 'group_b',
                             {'group_b': None, 'group_a': None})
        _, summary = m.reduce_groups([a, b])
        result = summary[image['cxr_model_id']]
        self.assertEqual(result['available_prompt_groups'], 1)
        self.assertTrue(result['conditional_on_available_groups'])

    def test_partial_image_failure_within_group_keeps_coverage(self):
        image = m.images_from_index(fixture(), full=False)[0]
        a = retrieval_record(image, 'group_a')
        failed = retrieval_record({**image, 'cxr_sha256': digest('failed_image')}, 'group_a',
                                  {'group_a': None, 'group_b': None})
        b = retrieval_record({**image, 'cxr_sha256': digest('image_b')}, 'group_b',
                             {'group_b': .1, 'group_a': .8})
        groups, summary = m.reduce_groups([a, failed, b])
        result = summary[image['cxr_model_id']]
        self.assertEqual(result['available_prompt_groups'], 2)
        self.assertFalse(result['conditional_on_available_groups'])
        self.assertTrue(result['conditional_on_available_images'])
        self.assertEqual(result['planned_unique_image_group_pairs'], 3)
        self.assertEqual(result['complete_unique_image_group_pairs'], 2)

    def test_every_old_score_choice_and_column_preserved(self):
        rows = fixture()
        images = m.images_from_index(rows, full=False)
        records = [retrieval_record(image) for image in images]
        output = m.candidate_overlay(rows, records)
        self.assertEqual(len(output), len(rows))
        for old, new in zip(rows, output):
            self.assertEqual({k: new[k] for k in old}, old)
            self.assertIsNone(new['conditioning_diagnostic_clinical_accuracy'])
            self.assertIsNone(new['conditioning_diagnostic_faulty_modality'])
            self.assertIs(new['conditioning_diagnostic_selector_used'], False)

    def test_four_reports_share_same_parent_conditioning_score(self):
        rows = fixture(); images = m.images_from_index(rows, full=False)
        output = m.candidate_overlay(rows, [retrieval_record(i) for i in images])
        first = [r for r in output if r['cxr_candidate_id'] == images[0]['cxr_candidate_id']]
        self.assertEqual(len(first), 4)
        self.assertEqual(len({r['conditioning_diagnostic_matched_cosine'] for r in first}), 1)

    def test_overlay_lineage_mismatch_refused(self):
        rows = fixture(); images = m.images_from_index(rows, full=False)
        records = [retrieval_record(i) for i in images]; records[0]['prompt_sha256'] = digest('wrong')
        with self.assertRaisesRegex(ValueError, 'exact_fixed_ehr'):
            m.candidate_overlay(rows, records)

    def test_existing_overlay_not_overwritten(self):
        rows = fixture(); images = m.images_from_index(rows, full=False)
        rows[0]['conditioning_diagnostic_status'] = 'historical'
        with self.assertRaisesRegex(ValueError, 'cannot_be_overwritten'):
            m.candidate_overlay(rows, [retrieval_record(i) for i in images])

    def test_csv_retains_null_as_empty_not_zero(self):
        csv_rows = list(csv.DictReader(io.StringIO(m.csv_text([{'id': 'fixture', 'score': None}]))))
        self.assertEqual(csv_rows[0]['score'], '')

    def test_evaluate_checks_approval_before_any_plan_or_body_read(self):
        args = SimpleNamespace(allow_synthetic_prompt_image=False)
        with patch.object(m.reuse, 'guard', side_effect=RuntimeError('approval_required')), \
                patch.object(m.reuse, 'metadata', side_effect=AssertionError('read too early')):
            with self.assertRaisesRegex(RuntimeError, 'approval_required'):
                m.evaluate(args)

    def test_prepare_requires_actual_existing_cpu_allocation(self):
        with patch.object(m.reuse, 'guard', side_effect=RuntimeError('slurm_required')), \
                patch.object(m, 'source_inputs', side_effect=AssertionError('read too early')):
            with self.assertRaisesRegex(RuntimeError, 'slurm_required'):
                m.prepare(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
