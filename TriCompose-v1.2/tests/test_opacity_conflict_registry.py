"""Invented CSV states/hashes/cosines only; no reports, images or model calls."""
import copy
import csv
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('opacity_registry_fixture', ROOT / 'tools/build_opacity_conflict_registry.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    originals, images, outcomes = [], [], []
    models = ('chexgenbench_pixart', 'chexgenbench_sana', 'roentgen_v2')
    experts = ('chexagent2', 'cxrmate_single', 'llavarad', 'maira2')
    for i, model in enumerate(models):
        image = {'case_id': 'case_fixture', 'cxr_candidate_id': f'image_fixture_{i}',
                 'cxr_model_id': model, 'cxr_sha256': f'{i+10:064x}',
                 'ehr_sha256': 'a' * 64, 'ehr_facts_sha256': 'b' * 64}
        images.append(image)
        image_positive, text_positive = i != 1, i == 0
        positive, negative = (.7, .1) if text_positive else (.1, .7)
        outcomes.append({k: image[k] for k in ('case_id', 'cxr_candidate_id', 'cxr_sha256', 'cxr_model_id')} |
            {'status': 'scored', 'image_calls': 1, 'score_pairs': {
                f: {'positive_cosine': positive, 'negative_cosine': negative}
                for f in m.evidence.frozen.reuse.FAMILIES}})
        for j, expert in enumerate(experts):
            report_state = ('negative', 'unknown', 'positive', 'positive')[j]
            row = {k: image[k] for k in ('case_id', 'cxr_candidate_id', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')}
            row.update(triple_candidate_id=f'triple_fixture_{i}_{j}',
                report_candidate_id=f'report_fixture_{i}_{j}', report_model_id=expert,
                report_sha256=f'{(21,22,23,23)[j]:064x}', opacity_cached_ehr_state='unknown',
                opacity_cached_report_state=report_state, opacity_exact_score='.8' if image_positive else '.2',
                opacity_exact_state_0_5='positive' if image_positive else 'negative',
                opacity_clinical_accuracy='', opacity_confirmed_faulty_modality='', opacity_selector_used='False')
            for k in range(66 - len(row)):
                row[f'invented_original_{k}'] = f'fixture_{k}'
            originals.append(row)
    rows = list(csv.DictReader(io.StringIO(m.evidence.opacity.csv_text(m.evidence.overlay(originals, images, outcomes)))))
    return rows, images, outcomes


class OpacityRegistryTests(unittest.TestCase):
    def test_all_rows_images_cases_preserved(self):
        registry = m.build_registry(*fixture())
        self.assertEqual((len(registry['candidates']), len(registry['images']), len(registry['cases'])), (12, 3, 1))
        self.assertTrue(all(len(r['candidate_ids']) == 4 for r in registry['images']))

    def test_deterministic_and_no_input_mutation(self):
        args = fixture(); old = copy.deepcopy(args)
        self.assertEqual(m.build_registry(*args), m.build_registry(*args))
        self.assertEqual(args, old)

    def test_original_row_order_and_digest(self):
        args = fixture(); registry = m.build_registry(*args)
        for index, (source, row) in enumerate(zip(args[0], registry['candidates'])):
            self.assertEqual(row['source_row_index'], index)
            self.assertEqual(row['source_row_sha256'], m.row_digest(source))
            self.assertEqual(row['triple_candidate_id'], source['triple_candidate_id'])

    def test_row_digest_named_order_invariant(self):
        self.assertEqual(m.row_digest({'a': '1', 'b': '2'}), m.row_digest({'b': '2', 'a': '1'}))
        self.assertNotEqual(m.row_digest({'a': '1'}), m.row_digest({'a': '2'}))

    def test_dedup_by_report_content_not_candidate(self):
        registry = m.build_registry(*fixture())
        self.assertEqual(len(registry['reports']), 3)
        self.assertEqual(sorted(r['global_slot_multiplicity'] for r in registry['reports']), [3, 3, 6])
        self.assertEqual(len({r['report_group_id'] for r in registry['reports']}), 3)

    def test_same_text_multiple_image_contexts_not_collapsed(self):
        registry = m.build_registry(*fixture())
        negative = next(r for r in registry['reports'] if r['cached_report_opacity_state'] == 'negative')
        self.assertEqual(len(negative['image_ids']), 3)
        self.assertTrue(negative['cross_image_reuse'])
        self.assertTrue(negative['cross_pattern_reuse'])
        self.assertEqual(len(negative['pattern_slot_counts']), 3)

    def test_text_requests_one_per_hash_and_all_contexts_retained(self):
        registry = m.build_registry(*fixture())
        requests = registry['requests']['report_assertion_requests']
        self.assertEqual(len(requests), 2)
        self.assertEqual(len({r['report_sha256'] for r in requests}), 2)
        positive = next(r for r in requests if len(r['reattach_all_candidate_ids']) == 6)
        self.assertEqual(len(positive['trigger_candidate_ids']), 2)

    def test_image_requests_once_per_image_not_per_report_slot(self):
        registry = m.build_registry(*fixture())
        requests = registry['requests']['image_evidence_requests']
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(requests[0]['candidate_ids']), 4)

    def test_unknown_is_missing_not_negative_agreement(self):
        registry = m.build_registry(*fixture())
        missing = [r for r in registry['candidates'] if r['report_opacity_state'] == 'unknown']
        self.assertEqual(len(missing), 3)
        self.assertEqual([r['diagnostic_lane'] for r in missing],
            ['missing_report_assertion', 'missing_report_assertion', 'verify_image_evidence'])

    def test_uncertain_is_kept_and_not_comparable(self):
        rows, images, outcomes = fixture()
        for row in rows:
            if row['opacity_cached_report_state'] == 'unknown':
                row['opacity_cached_report_state'] = 'uncertain'
        registry = m.build_registry(rows, images, outcomes)
        self.assertEqual(sum(r['report_opacity_state'] == 'uncertain' for r in registry['candidates']), 3)

    def test_agreement_not_gold_or_clinical_repair(self):
        registry = m.build_registry(*fixture())
        self.assertTrue(any(r['diagnostic_lane'] == 'proxy_agreement_control' for r in registry['candidates']))
        self.assertTrue(all(r['clinical_fault_label'] is None and r['clinical_accuracy'] is None
                            and r['regeneration_authorized'] is False and r['selector_used'] is False
                            for r in registry['candidates']))

    def test_requests_unmaterialized_no_predictions_or_model_execution(self):
        registry = m.build_registry(*fixture())
        for request in [*registry['requests']['report_assertion_requests'], *registry['requests']['image_evidence_requests']]:
            self.assertEqual(request['status'], 'metadata_only_unmaterialized')
            self.assertEqual(request['model_calls'], 0)
            self.assertFalse(request['cached_predictions_exposed_to_verifier'])
            self.assertFalse(request['regeneration_authorized'])
            self.assertFalse(any(k in request for k in ('text', 'path', 'proposed_state', 'scores', 'report_model_id')))

    def test_template_mixed_flag_not_new_gate(self):
        rows, images, outcomes = fixture()
        outcomes[0]['score_pairs']['shows_no'] = {'positive_cosine': .1, 'negative_cosine': .9}
        old = [{k: v for k, v in r.items() if not k.startswith(m.evidence.PREFIX)} for r in rows]
        changed = list(csv.DictReader(io.StringIO(m.evidence.opacity.csv_text(m.evidence.overlay(old, images, outcomes)))))
        registry = m.build_registry(changed, images, outcomes)
        self.assertTrue(registry['images'][0]['template_sign_variation'])
        self.assertTrue(all(r['regeneration_authorized'] is False for r in registry['candidates']))

    def test_failed_image_retains_all_four_slots(self):
        rows, images, outcomes = fixture()
        outcomes[0].update(status='failed_without_replacement', score_pairs=None)
        old = [{k: v for k, v in r.items() if not k.startswith(m.evidence.PREFIX)} for r in rows]
        changed = list(csv.DictReader(io.StringIO(m.evidence.opacity.csv_text(m.evidence.overlay(old, images, outcomes)))))
        registry = m.build_registry(changed, images, outcomes)
        self.assertEqual(len(registry['candidates']), 12)
        self.assertEqual(registry['images'][0]['scoring_status'], 'failed_without_replacement')
        self.assertEqual(sum(r['diagnostic_lane'] == 'missing_image_evidence' for r in registry['candidates']), 4)

    def test_changed_saved_margin_refused(self):
        rows, images, outcomes = fixture(); rows[0][m.evidence.PREFIX + 'mean_margin'] = '.9'
        with self.assertRaisesRegex(ValueError, 'all_saved_evidence_cells'):
            m.build_registry(rows, images, outcomes)

    def test_changed_saved_pattern_refused(self):
        rows, images, outcomes = fixture(); rows[0][m.evidence.PREFIX + 'joint_proxy_pattern'] = 'three_proxy_sources_agree'
        with self.assertRaisesRegex(ValueError, 'all_saved_evidence_cells'):
            m.build_registry(rows, images, outcomes)

    def test_inconsistent_shared_report_assertions_refused(self):
        rows, images, outcomes = fixture(); rows[2]['opacity_cached_report_state'] = 'negative'
        with self.assertRaisesRegex(ValueError, 'shared_report_artifact_state_changed'):
            m.build_registry(rows, images, outcomes)

    def test_fixed_ehr_changed_refused(self):
        rows, images, outcomes = fixture(); rows[0]['ehr_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'same_fixed_ehr_image_lineage'):
            m.build_registry(rows, images, outcomes)

    def test_failed_image_outcome_cannot_smuggle_scores(self):
        rows, images, outcomes = fixture(); outcomes[0]['status'] = 'failed_without_replacement'
        with self.assertRaisesRegex(ValueError, 'explicit_scored_or_failed'):
            m.build_registry(rows, images, outcomes)

    def test_missing_or_duplicate_outcomes_refused(self):
        rows, images, outcomes = fixture()
        for bad in (outcomes[:2], outcomes + outcomes[:1]):
            with self.assertRaisesRegex(ValueError, 'all_unique_image_outcomes'):
                m.build_registry(rows, images, bad)

    def test_duplicate_candidate_ids_refused(self):
        rows, images, outcomes = fixture(); rows[1]['triple_candidate_id'] = rows[0]['triple_candidate_id']
        with self.assertRaisesRegex(ValueError, 'unique_nonempty_inventory'):
            m.build_registry(rows, images, outcomes)

    def test_inconsistent_same_image_xrv_score_refused(self):
        rows, images, outcomes = fixture(); rows[0]['opacity_exact_score'] = '.7'
        with self.assertRaisesRegex(ValueError, 'same_image_shared_proposals'):
            m.build_registry(rows, images, outcomes)

    def test_image_model_taken_from_metadata_not_id_spelling(self):
        rows, images, outcomes = fixture()
        misleading = 'image_has_unrelated_model_words'
        original_id = images[0]['cxr_candidate_id']
        images[0]['cxr_candidate_id'] = outcomes[0]['cxr_candidate_id'] = misleading
        for row in rows:
            if row['cxr_candidate_id'] == original_id:
                row['cxr_candidate_id'] = misleading
        registry = m.build_registry(rows, images, outcomes)
        self.assertEqual(registry['images'][0]['cxr_model_id'], 'chexgenbench_pixart')
        self.assertTrue(all(r['cxr_model_id'] == 'chexgenbench_pixart'
            for r in registry['candidates'] if r['cxr_candidate_id'] == misleading))

    def test_non_csv_types_refused(self):
        rows, images, outcomes = fixture(); rows[0]['opacity_exact_score'] = .8
        with self.assertRaisesRegex(ValueError, 'nonempty_csv_named_cells'):
            m.build_registry(rows, images, outcomes)

    def test_full_bank_cannot_be_shrunk_to_favorable_fixture(self):
        with self.assertRaisesRegex(ValueError, 'complete_80_3_4_grid'):
            m.build_registry(*fixture(), full=True)

    def test_summary_full_denominators_and_dedup_distinct_units(self):
        summary = m.summarize(m.build_registry(*fixture()))
        self.assertEqual((summary['candidate_slots'], summary['image_slots'], summary['fixed_ehr_cases']), (12, 3, 1))
        self.assertEqual(summary['unique_report_hashes'], 3)
        self.assertEqual(sum(v['candidate_slots'] for v in summary['dependency_patterns'].values()), 12)
        self.assertEqual(summary['image_slots_by_cxr_model'], {'chexgenbench_pixart': 1, 'chexgenbench_sana': 1, 'roentgen_v2': 1})
        self.assertTrue(all(v['candidate_slots'] == 4 for v in summary['by_cxr_model'].values()))
        self.assertTrue(all(v['candidate_slots'] == 3 for v in summary['by_report_model'].values()))

    def test_shared_report_groups_can_overlap_pattern_counts(self):
        summary = m.summarize(m.build_registry(*fixture()))
        self.assertGreater(sum(v['unique_report_hashes'] for v in summary['dependency_patterns'].values()),
                           summary['unique_report_hashes'])
        self.assertEqual(summary['report_groups_reused_across_patterns'], 3)

    def test_complete_empty_conflict_lane_retained_without_fake_success(self):
        rows, images, outcomes = fixture()
        for row in rows:
            row['report_sha256'] = 'e' * 64; row['opacity_cached_report_state'] = 'unknown'
        old = [{k: v for k, v in r.items() if not k.startswith(m.evidence.PREFIX)} for r in rows]
        changed = list(csv.DictReader(io.StringIO(m.evidence.opacity.csv_text(m.evidence.overlay(old, images, outcomes)))))
        summary = m.summarize(m.build_registry(changed, images, outcomes))
        self.assertEqual(summary['pending_unique_text_assertion_requests'], 0)
        self.assertEqual(summary['resolved_verification_requests'], 0)
        self.assertIsNone(summary['clinical_repair_success'])

    def test_unknown_ehr_never_enriched(self):
        registry = m.build_registry(*fixture())
        self.assertTrue(all(r['ehr_opacity_state'] == 'unknown' for r in registry['candidates']))
        self.assertTrue(all(r['ehr_opacity_state'] == 'unknown' for r in registry['cases']))
        self.assertFalse(m.summarize(registry)['ehr_opacity_edges_available'])

    def test_guard_before_any_source_or_output_read(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(m, 'source_inputs') as loader, \
                patch.object(m, 'new_atomic_run') as writer:
            with self.assertRaisesRegex(RuntimeError, 'actual_slurm_allocation_required'):
                m.run(SimpleNamespace())
            loader.assert_not_called(); writer.assert_not_called()

    def test_existing_output_refused_before_any_write(self):
        args = fixture(); registry = m.build_registry(*args)
        with patch.object(m.evidence.opacity, 'guard'), patch.object(m, 'source_inputs', return_value=(*args, {})), \
                patch.object(m, 'build_registry', return_value=registry), \
                patch.object(m, 'new_atomic_run', side_effect=FileExistsError('existing')), \
                patch.object(m, 'write_private_text') as writer:
            with self.assertRaises(FileExistsError):
                m.run(SimpleNamespace(output_root='fixture', run_id='fixture'))
            writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
