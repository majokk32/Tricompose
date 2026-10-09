"""Invented cached vectors/hashes only; no artifact bodies, pixels or models."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tricompose_v12 import scorer_reliability as module
spec = importlib.util.spec_from_file_location('reliability_cli_fixture', ROOT/'benchmarks/build_scorer_reliability_sidecar.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture(experts=('positive', 'negative'), ehr='unknown', image='positive'):
    rows, bank = [], {'fixture_case': {}}
    hashes = {'ehr_sha256': module.digest('ehr'), 'ehr_facts_sha256': module.digest('facts'),
              'cxr_sha256': module.digest('image')}
    for index, state in enumerate(experts):
        cid, report, model = f'fixture_triple_{index}', f'fixture_report_{index}', f'expert_{index}'
        lineage = {**hashes, 'report_sha256': module.digest(report), 'cxr_candidate_id': 'fixture_image',
                   'report_candidate_id': report, 'report_model_id': model}
        original = {'case_id': 'fixture_case', 'triple_candidate_id': cid, 'lineage': lineage}
        facts = []
        for finding in module.FINDINGS:
            states = {'ehr': 'unknown', 'xrv': 'unknown', 'chexbert': 'unknown'}
            if finding == 'pneumonia':
                states = {'ehr': ehr, 'xrv': image, 'chexbert': state}
            facts.append({'finding': finding, 'case_id': 'fixture_case', 'triple_candidate_id': cid,
                'artifact_hashes': {**hashes, 'report_sha256': lineage['report_sha256']}, 'states': states,
                'weak_context_promoted': False, 'clinical_truth_verified': False,
                'source_categories': ['diagnoses'] if states['ehr'] != 'unknown' else [],
                'evidence_id': module.digest((cid, finding))})
        bank['fixture_case'][index] = {'score_record': original, 'facts': facts}
        row = {'case_id': 'fixture_case', 'triple_candidate_id': cid, **lineage,
               'profile': module.PROFILE, 'source_primary_prefix': '[0, 0, 0, -1]',
               'biovil_raw_cosine': '0.2', 'endpoint_unavailable_reason': '',
               'clinical_accuracy': '', 'source_selected': 'false'}
        for edge, (left, right) in module.EDGES.items():
            value = module._edge([f['states'] for f in facts], left, right)
            row.update({f'{edge}_{key}': '' if number is None else str(number) for key, number in value.items()})
        rows.append(row)
    return rows, bank


class ReliabilityTests(unittest.TestCase):
    def test_original_cells_order_candidates_and_states_preserved(self):
        rows, bank = fixture()
        before = copy.deepcopy((rows, bank))
        annotated, facts, groups, summary = module.build_overlay(rows, bank)
        self.assertEqual((rows, bank), before)
        for old, new in zip(rows, annotated):
            self.assertTrue(all(new[key] == value for key, value in old.items()))
        self.assertTrue(summary['original_cells_preserved'])
        self.assertTrue(summary['original_row_order_preserved'])
        self.assertEqual(summary['candidate_rows'], 2)
        self.assertEqual(summary['fact_rows'], 28)

    def test_unknown_ehr_not_negative_perfect_consistency_or_new_fact(self):
        rows, bank = fixture()
        annotated, facts, _, summary = module.build_overlay(rows, bank)
        self.assertEqual(summary['fixed_ehr_cases_without_explicit_cached_facts'], 1)
        for row in annotated:
            self.assertEqual(row['reliability_ehr_cxr_evidence_status'], 'no_explicit_reference_facts')
            self.assertIsNone(row['reliability_ehr_cxr_clinical_score'])
        self.assertTrue(all(row['states']['ehr'] == 'unknown' for row in facts))

    def test_uncertain_stays_noncomparable(self):
        rows, bank = fixture(experts=('uncertain',), ehr='positive')
        _, facts, _, _ = module.build_overlay(rows, bank)
        finding = next(row for row in facts if row['finding'] == 'pneumonia')
        self.assertEqual(finding['states']['chexbert'], 'uncertain')
        self.assertEqual(finding['relations']['ehr_report'], 'not_comparable')

    def test_explicit_opposition_is_unverified_not_confirmed_fault(self):
        rows, bank = fixture(experts=('negative',), ehr='positive')
        annotated, facts, _, _ = module.build_overlay(rows, bank)
        self.assertEqual(annotated[0]['reliability_ehr_report_evidence_status'], 'proxy_opposition_unverified')
        self.assertIsNone(annotated[0]['reliability_confirmed_faulty_modality'])
        self.assertFalse(annotated[0]['reliability_automatic_regeneration_authorized'])
        self.assertTrue(all(row['clinical_conflict_verified'] is None for row in facts))

    def test_report_experts_disagreement_counted_per_shared_image_finding(self):
        rows, bank = fixture()
        annotated, facts, groups, summary = module.build_overlay(rows, bank)
        self.assertEqual(summary['image_slots'], 1)
        self.assertEqual(summary['image_finding_dependency_groups'], 14)
        self.assertEqual(summary['same_image_report_expert_proxy_disagreement_groups'], 1)
        self.assertTrue(all(row['reliability_report_expert_disagreement_findings'] == 1 for row in annotated))
        self.assertTrue(all(group['independent_votes'] is False for group in groups))

    def test_unknown_report_not_opposite_negative(self):
        rows, bank = fixture(experts=('positive', 'unknown'))
        _, _, _, summary = module.build_overlay(rows, bank)
        self.assertEqual(summary['same_image_report_expert_proxy_disagreement_groups'], 0)

    def test_same_report_hash_not_counted_as_two_independent_artifacts(self):
        rows, bank = fixture(experts=('positive', 'positive'))
        report_hash = rows[0]['report_sha256']
        rows[1]['report_sha256'] = report_hash
        candidate = bank['fixture_case'][1]
        candidate['score_record']['lineage']['report_sha256'] = report_hash
        for fact in candidate['facts']:
            fact['artifact_hashes']['report_sha256'] = report_hash
        _, _, groups, summary = module.build_overlay(rows, bank)
        self.assertEqual(summary['unique_report_artifacts'], 1)
        self.assertTrue(all(group['unique_report_artifacts'] == 1 for group in groups))

    def test_candidate_image_scorer_disagreement_not_invented_from_real_benchmark(self):
        rows, bank = fixture()
        annotated, facts, _, summary = module.build_overlay(rows, bank)
        self.assertEqual(summary['candidate_same_fact_image_scorer_disagreements_measured'], 0)
        self.assertTrue(all(row['reliability_image_scorer_disagreement'] is None for row in annotated))
        self.assertTrue(all(row['candidate_same_fact_image_scorer_disagreement'] is None for row in facts))
        self.assertFalse(any(row['benchmark_metric_transferred_to_candidate'] for row in facts))

    def test_biovil_cosine_does_not_change_raw_ranking_or_clinical_status(self):
        rows, bank = fixture()
        rows[0]['biovil_raw_cosine'], rows[1]['biovil_raw_cosine'] = '-0.5', '0.99'
        annotated, _, _, _ = module.build_overlay(rows, bank)
        self.assertEqual(annotated[0]['biovil_raw_cosine'], '-0.5')
        self.assertTrue(all(row['source_primary_prefix'] == '[0, 0, 0, -1]' for row in annotated))
        self.assertTrue(all(row['reliability_clinical_verdict'] == 'unverified' for row in annotated))

    def test_missing_endpoint_keeps_na_and_reason_not_zero(self):
        rows, bank = fixture(experts=('positive',))
        rows[0]['biovil_raw_cosine'] = ''
        rows[0]['endpoint_unavailable_reason'] = 'fixture_unavailable'
        annotated, _, _, _ = module.build_overlay(rows, bank)
        self.assertEqual(annotated[0]['biovil_raw_cosine'], '')
        self.assertEqual(annotated[0]['reliability_biovil_role'], 'unavailable_with_source_reason')

    def test_missing_endpoint_reason_rejected(self):
        rows, bank = fixture(experts=('positive',))
        rows[0]['biovil_raw_cosine'] = ''
        with self.assertRaisesRegex(ValueError, 'missing_endpoint_reason_required'):
            module.build_overlay(rows, bank)

    def test_nonfinite_invalid_cosine_and_false_edge_counts_refused(self):
        for field, value in (('biovil_raw_cosine', 'NaN'), ('biovil_raw_cosine', '2'),
                             ('ehr_cxr_supported_facts', '1')):
            rows, bank = fixture()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                module.build_overlay(rows, bank)

    def test_mixed_profile_and_already_annotated_source_rejected(self):
        for field, value in (('profile', 'fresh_eight_heads'), ('reliability_clinical_verdict', 'confirmed')):
            rows, bank = fixture()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                module.build_overlay(rows, bank)

    def test_missing_duplicate_and_foreign_candidates_rejected(self):
        rows, bank = fixture()
        for invalid in (rows[:1], [rows[0], rows[0]], [{**rows[0], 'triple_candidate_id': 'foreign'}, rows[1]]):
            with self.assertRaisesRegex(ValueError, 'exact_source_candidate_inventory_required'):
                module.build_overlay(invalid, bank)

    def test_case_or_artifact_hash_changes_rejected(self):
        for field in ('case_id', 'ehr_sha256', 'report_sha256'):
            rows, bank = fixture()
            rows[0][field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'lineage_or_hash_changed'):
                module.build_overlay(rows, bank)

    def test_fixed_ehr_state_change_and_unsupported_ehr_fact_rejected(self):
        rows, bank = fixture(experts=('positive', 'positive'), ehr='positive')
        bank['fixture_case'][1]['facts'][10]['states']['ehr'] = 'negative'
        with self.assertRaisesRegex(ValueError, 'shared_artifact_or_fixed_ehr_changed'):
            module.build_overlay(rows, bank)
        rows, bank = fixture(experts=('positive',), ehr='positive')
        bank['fixture_case'][0]['facts'][10]['source_categories'] = []
        with self.assertRaisesRegex(ValueError, 'lacks_cached_source_category'):
            module.build_overlay(rows, bank)

    def test_statuses_distinguish_no_comparison_missing_and_supported(self):
        def edge(known, comparable, opposed=0):
            return {'known_reference_facts': known, 'comparable_facts': comparable,
                    'proxy_opposition_facts': opposed, 'missing_comparisons': known-comparable}
        self.assertEqual(module.status(edge(0, 0)), 'no_explicit_reference_facts')
        self.assertEqual(module.status(edge(1, 0)), 'no_comparable_proxy_facts')
        self.assertEqual(module.status(edge(2, 1)), 'proxy_support_with_missing_evidence')
        self.assertEqual(module.status(edge(1, 1)), 'proxy_support_only_unverified')
        self.assertEqual(module.status(edge(1, 1, 1)), 'proxy_opposition_unverified')

    def test_cpu_allocation_guard_precedes_cache_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, 'require_inside') as resolve:
            with self.assertRaisesRegex(RuntimeError, 'existing_cpu_slurm_required'):
                cli.run(SimpleNamespace())
            resolve.assert_not_called()

    def test_benchmark_aggregate_refuses_patient_records_before_other_fields(self):
        with self.assertRaisesRegex(ValueError, 'aggregate_only'):
            cli.reliability_profile({'records': []}, {}, {})

    def test_csv_serialization_preserves_null_and_false_without_zero_imputation(self):
        text = cli.csv_text([{'metric': None, 'verdict': False, 'original_prefix': '[0, 1]'}])
        self.assertIn(',false,', text)
        self.assertNotIn('0.0', text)
        self.assertIn('[0, 1]', text)


if __name__ == '__main__':
    unittest.main()
