"""Invented metadata only; no clinical bodies, pixels, models or real records."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from test_scorer_reliability import fixture
from tricompose_v12 import scorer_reliability as source
from tricompose_v12 import reliability_preview as module

spec = importlib.util.spec_from_file_location('reliability_preview_cli_fixture', ROOT/'benchmarks/preview_reliability_evidence.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def sidecar(**kwargs):
    rows, bank = fixture(**kwargs)
    rows, facts, groups, _ = source.build_overlay(rows, bank)
    return rows, facts, groups


def preview(**kwargs):
    return module.preview_reliability(*sidecar(**kwargs))


class ReliabilityPreviewTests(unittest.TestCase):
    def test_original_inputs_order_hashes_and_selection_are_preserved(self):
        inputs = sidecar()
        before = copy.deepcopy(inputs)
        decisions, requests, summary = module.preview_reliability(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual([d['triple_candidate_id'] for d in decisions], [r['triple_candidate_id'] for r in inputs[0]])
        for row, decision in zip(inputs[0], decisions):
            self.assertEqual(decision['artifact_hashes'], {key: row[key] for key in source.HASH_FIELDS})
        self.assertFalse(summary['selection_changed'])
        self.assertEqual(summary['new_model_calls'], 0)
        self.assertEqual(summary['cases_dropped_or_rejected'], 0)

    def test_unknown_ehr_retained_and_observability_request_deduplicated(self):
        decisions, requests, summary = preview(experts=('positive', 'positive', 'positive', 'positive'))
        observability = [r for r in requests if r['request_kind'] == 'assess_ehr_radiographic_observability']
        self.assertEqual(len(observability), 1)
        self.assertEqual(len(observability[0]['consumer_candidate_ids']), 4)
        self.assertEqual(summary['fixed_ehr_cases_without_explicit_facts'], 1)
        self.assertTrue(all(d['fixed_ehr_retained'] and not d['case_rejected'] for d in decisions))
        self.assertTrue(all(d['explicit_ehr_reference_facts'] == 0 for d in decisions))
        self.assertTrue(all(d['clinical_selection_score'] is None for d in decisions))

    def test_all_agreement_does_not_accept_or_execute(self):
        decisions, requests, summary = preview(ehr='positive', experts=('positive',))
        self.assertEqual(decisions[0]['preview_action'], 'retain_unresolved_preview')
        self.assertEqual(requests, [])
        self.assertEqual(decisions[0]['proxy_pattern_counts']['all_three_proxies_agree_unverified'], 1)
        self.assertFalse(decisions[0]['clinical_acceptance'])
        self.assertFalse(decisions[0]['model_execution_allowed'])
        self.assertFalse(summary['primary_metric_eligible'])

    def test_report_proxy_opposition_requests_report_relation_not_regeneration(self):
        decisions, requests, _ = preview(ehr='positive', experts=('negative',))
        kinds = {r['request_kind'] for r in requests}
        self.assertEqual(kinds, {'verify_report_assertion', 'verify_image_report_relation'})
        self.assertEqual(decisions[0]['proxy_pattern_counts']['report_differs_from_ehr_and_image_proxies'], 1)
        self.assertIsNone(decisions[0]['confirmed_faulty_modality'])
        self.assertFalse(decisions[0]['regeneration_authorized'])

    def test_image_proxy_opposition_does_not_use_report_as_independent_truth(self):
        decisions, requests, _ = preview(ehr='positive', image='negative', experts=('positive',))
        self.assertEqual({r['request_kind'] for r in requests}, {'verify_image_finding', 'verify_image_report_relation'})
        self.assertIn('cxr_conditioned_report_is_not_independent_image_truth', decisions[0]['reason_codes'])
        self.assertIsNone(decisions[0]['confirmed_faulty_modality'])

    def test_both_downstream_oppose_ehr_requests_scope_check_not_ehr_replacement(self):
        decisions, requests, _ = preview(ehr='positive', image='negative', experts=('negative',))
        self.assertIn('verify_conditioning_and_ehr_fact_scope', {r['request_kind'] for r in requests})
        self.assertEqual(decisions[0]['proxy_pattern_counts']['both_downstream_proxies_oppose_ehr'], 1)
        self.assertTrue(decisions[0]['fixed_ehr_retained'])
        self.assertFalse(decisions[0]['case_rejected'])

    def test_uncertain_report_is_missing_not_opposite_negative(self):
        decisions, requests, _ = preview(ehr='positive', experts=('uncertain',))
        self.assertEqual({r['request_kind'] for r in requests}, {'verify_report_assertion'})
        self.assertIn('report_evidence_missing_for_explicit_ehr_fact', decisions[0]['reason_codes'])
        self.assertNotIn('ehr_report_proxy_opposition_unverified', decisions[0]['reason_codes'])

    def test_unknown_image_is_missing_not_opposite_negative(self):
        decisions, requests, _ = preview(ehr='positive', image='unknown', experts=('positive',))
        self.assertEqual({r['request_kind'] for r in requests}, {'verify_image_finding'})
        self.assertIn('image_evidence_missing_for_explicit_ehr_fact', decisions[0]['reason_codes'])

    def test_same_image_request_shared_by_four_reports(self):
        _, requests, _ = preview(ehr='positive', image='negative', experts=('positive',)*4)
        image = [r for r in requests if r['request_kind'] == 'verify_image_finding']
        self.assertEqual(len(image), 1)
        self.assertEqual(len(image[0]['consumer_candidate_ids']), 4)
        self.assertEqual(set(image[0]['dependency_hashes']), {'cxr_sha256'})

    def test_report_disagreement_is_not_majority_vote(self):
        decisions, requests, _ = preview(experts=('positive', 'positive', 'positive', 'negative'))
        self.assertTrue(all('correlated_same_image_reports_disagree' in d['reason_codes'] for d in decisions))
        self.assertTrue(any(r['request_kind'] == 'verify_image_finding' for r in requests))
        self.assertTrue(all(d['confirmed_faulty_modality'] is None for d in decisions))

    def test_duplicate_report_artifacts_deduplicated_per_dependency(self):
        rows, bank = fixture(ehr='positive', experts=('negative', 'negative'))
        report_hash = rows[0]['report_sha256']
        rows[1]['report_sha256'] = report_hash
        bank['fixture_case'][1]['score_record']['lineage']['report_sha256'] = report_hash
        for fact in bank['fixture_case'][1]['facts']:
            fact['artifact_hashes']['report_sha256'] = report_hash
        annotated, facts, groups, _ = source.build_overlay(rows, bank)
        _, requests, _ = module.preview_reliability(annotated, facts, groups)
        self.assertEqual(CounterKinds(requests), {'verify_report_assertion': 1, 'verify_image_report_relation': 1})
        self.assertTrue(all(len(r['consumer_candidate_ids']) == 2 for r in requests))

    def test_biovil_score_does_not_change_requests_or_decisions(self):
        rows, facts, groups = sidecar()
        first = module.preview_reliability(rows, facts, groups)
        rows[0]['biovil_raw_cosine'] = '0.99'
        rows[1]['biovil_raw_cosine'] = '-0.99'
        self.assertEqual(first, module.preview_reliability(rows, facts, groups))

    def test_budget_missing_not_zero_or_fake_affordable(self):
        decisions, requests, summary = preview()
        self.assertEqual(summary['budget']['verification_affordability'], 'additional_budget_not_configured')
        self.assertIsNone(summary['budget']['remaining_additional_gpu_seconds'])
        self.assertIsNone(summary['estimated_additional_model_calls'])
        self.assertIsNone(summary['actual_compute_savings'])
        self.assertTrue(all(r['estimated_gpu_seconds'] is None and r['estimated_model_calls'] is None for r in requests))
        self.assertTrue(all(d['model_execution_allowed'] is False for d in decisions))

    def test_budget_exhausted_stops_preview_but_does_not_reject_case(self):
        decisions, requests, summary = module.preview_reliability(*sidecar(), budget=module.Budget(0, 0))
        self.assertTrue(all(d['preview_action'] == 'stop_unresolved_preview' for d in decisions))
        self.assertTrue(all(not d['case_rejected'] for d in decisions))
        self.assertTrue(all(r['blocked_by_declared_budget'] for r in requests))
        self.assertEqual(summary['cases_dropped_or_rejected'], 0)

    def test_failed_calls_count_and_missing_runtime_stays_missing(self):
        history = ({'call_id': 'failed_fixture', 'status': 'failed', 'gpu_seconds': None},)
        _, _, summary = module.preview_reliability(*sidecar(), budget=module.Budget(2, 30), history=history)
        self.assertEqual(summary['budget']['attempted_additional_calls'], 1)
        self.assertEqual(summary['budget']['remaining_additional_calls'], 1)
        self.assertIsNone(summary['budget']['remaining_additional_gpu_seconds'])
        self.assertEqual(summary['budget']['verification_affordability'], 'unknown_observed_runtime')

    def test_history_without_budget_refused(self):
        with self.assertRaisesRegex(ValueError, 'explicit_budget_required'):
            module.preview_reliability(*sidecar(), history=({'fake': True},))

    def test_requests_have_trace_consumers_and_never_execute(self):
        decisions, requests, summary = preview(ehr='positive', experts=('negative', 'positive'))
        ids = {r['request_id'] for r in requests}
        for request in requests:
            self.assertEqual(request['execution_status'], 'not_executed')
            self.assertFalse(request['model_execution_allowed'])
            self.assertFalse(request['clinical_truth_established'])
        self.assertEqual({rid for d in decisions for rid in d['request_ids']}, ids)
        self.assertEqual(summary['deduplicated_logical_evidence_requests'], len(requests))
        self.assertFalse(summary['requests_are_model_calls'])

    def test_csv_false_and_na_serialization(self):
        decisions, _, _ = preview()
        rendered = cli.render_csv(decisions)
        self.assertIn(',false,false,false,false', rendered)
        self.assertNotIn('None', rendered)

    def test_deterministic_preview_and_request_ids(self):
        first = preview(ehr='positive')
        self.assertEqual(first, preview(ehr='positive'))

    def test_incomplete_duplicate_and_foreign_inventory_refused(self):
        rows, facts, groups = sidecar()
        for invalid in (facts[:-1], facts[:-1]+[facts[0]]):
            with self.assertRaises(ValueError):
                module.preview_reliability(rows, invalid, groups)
        with self.assertRaises(ValueError):
            module.preview_reliability([rows[0], rows[0]], facts, groups)

    def test_fixed_ehr_or_shared_artifact_change_refused(self):
        rows, facts, groups = sidecar()
        rows[1]['ehr_sha256'] = source.digest('changed')
        with self.assertRaises(ValueError):
            module.preview_reliability(rows, facts, groups)

    def test_forged_clinical_eligibility_profile_or_measured_disagreement_refused(self):
        for key, value in (('profile', 'fresh_eight_heads'), ('reliability_primary_metric_eligible', True),
                           ('reliability_confirmed_faulty_modality', 'cxr'), ('reliability_image_scorer_disagreement', 0)):
            rows, facts, groups = sidecar()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                module.preview_reliability(rows, facts, groups)

    def test_false_is_not_numeric_zero_authorization(self):
        rows, facts, groups = sidecar()
        rows[0]['reliability_automatic_regeneration_authorized'] = 0
        with self.assertRaises(ValueError):
            module.preview_reliability(rows, facts, groups)

    def test_forged_fact_relation_provenance_or_clinical_truth_refused(self):
        for key, value in (('relations', {}), ('cached_source_categories', []), ('clinical_truth_available', True)):
            rows, facts, groups = sidecar(ehr='positive')
            finding = next(f for f in facts if f['finding'] == 'pneumonia')
            finding[key] = value
            with self.assertRaises(ValueError):
                module.preview_reliability(rows, facts, groups)

    def test_forged_group_independence_counts_or_missing_groups_refused(self):
        for key, value in (('independent_votes', True), ('unique_report_artifacts', 77),
                           ('explicit_positive_negative_disagreement', True)):
            rows, facts, groups = sidecar()
            groups[0][key] = value
            with self.assertRaises(ValueError):
                module.preview_reliability(rows, facts, groups)
        rows, facts, groups = sidecar()
        with self.assertRaises(ValueError):
            module.preview_reliability(rows, facts, groups[:-1])

    def test_false_raw_edge_counts_or_availability_refused(self):
        for key, value in (('ehr_cxr_supported_facts', '9'), ('ehr_cxr_support_over_known', 'NaN'),
                           ('reliability_ehr_report_evidence_status', 'perfect_agreement')):
            rows, facts, groups = sidecar()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                module.preview_reliability(rows, facts, groups)

    def test_cpu_guard_precedes_metadata_access(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli, 'require_inside') as resolve:
            with self.assertRaisesRegex(RuntimeError, 'existing_cpu_slurm_required'):
                cli.load_sidecar('unused')
            resolve.assert_not_called()

    def test_partial_cli_budget_refused_before_output_creation(self):
        args = SimpleNamespace(reliability_run='unused', max_additional_calls=1,
            max_additional_gpu_seconds=None)
        with patch.object(cli, 'load_sidecar', return_value=(*sidecar(), {})), patch.object(cli, 'new_atomic_run') as writer:
            with self.assertRaisesRegex(ValueError, 'both_budget_limits'):
                cli.run(args)
            writer.assert_not_called()


def CounterKinds(requests):
    from collections import Counter
    return dict(Counter(r['request_kind'] for r in requests))


if __name__ == '__main__':
    unittest.main()
