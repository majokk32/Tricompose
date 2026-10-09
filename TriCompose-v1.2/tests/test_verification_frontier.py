"""Invented metadata only; no patient bodies, model weights, or image decoding."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verification_frontier_fixture',
    ROOT / 'tools/build_verification_frontier.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def fixture(reports=4, unchecked=False, changed=True, opposition=False,
        current='uncertain', retained=None):
    rows, facts = [], []
    for i in range(reports + int(unchecked)):
        checked = not (unchecked and i == reports)
        row = {'case_id': 'invented_case', 'triple_candidate_id': 'triple' + str(i),
            'cxr_candidate_id': 'image0' if checked else 'same_hash_unchecked',
            'report_candidate_id': 'report' + str(i), 'ehr_sha256': 'e' * 64,
            'ehr_facts_sha256': 'f' * 64, 'cxr_sha256': 'a' * 64, 'report_sha256': 'b' * 64,
            'old_clinical_score': '', 'old_proxy_score': '0.21',
            'liveimage_status': 'checked_clinically_unqualified' if checked else 'not_checked'}
        rows.append(row)
        for head in w.CHEXPERT_FINDINGS:
            supported = head in w.guard.HEADS
            status = 'not_checked' if not checked else 'checked_clinically_unqualified' if supported else 'outside_image_verifier_scope'
            fresh = (current if head == 'pneumonia' else 'negative') if checked and supported else None
            old = ('positive' if changed else fresh) if head == 'pneumonia' and fresh is not None else fresh
            xrv = 'positive' if opposition and head == 'pneumonia' else 'negative'
            exact = checked and supported
            kept = retained if exact else None
            facts.append({'case_id': row['case_id'], 'triple_candidate_id': row['triple_candidate_id'],
                'cxr_candidate_id': row['cxr_candidate_id'], 'report_candidate_id': row['report_candidate_id'],
                'artifact_hashes': {k: row[k] for k in w.HASH_FIELDS}, 'finding': head,
                'source_evidence_id': w.digest(['invented', row['triple_candidate_id'], head]),
                'states': {'ehr': 'unknown', 'xrv': xrv, 'chexbert': 'unknown'},
                'cached_source_categories': [],
                'liveimage_status': status,
                'liveimage_guard_status': 'basic_pass_not_clinical' if checked else None,
                'liveimage_guard_receipt_sha256': 'c' * 64 if checked else None,
                'liveimage_run_manifest_sha256': 'd' * 64 if checked else None,
                'liveimage_response_sha256': '1' * 64 if checked else None,
                'liveimage_baseline_state': old, 'liveimage_qwen_state': fresh,
                'liveimage_readout_repeatability': status if fresh is None else 'changed_state_unqualified' if old != fresh else 'same_state_unqualified',
                'liveimage_readout_changed': None if fresh is None else old != fresh,
                'liveimage_xrv_relation': status if fresh is None else w.cached.relation(xrv, fresh),
                'liveimage_exact_report_comparison_available': exact,
                'liveimage_retained_report_relation': status if fresh is None else 'report_assertion_not_retained' if kept is None else w.cached.relation(fresh, kept),
                'liveimage_independent_clinical_validation': False,
                'liveimage_primary_metric_eligible': False, 'liveimage_regeneration_authorized': False,
                'reportgate_status': 'not_checked' if not checked else 'outside_verifier_scope' if not supported else 'scope_commit' if kept else 'abstain',
                'reportgate_retained_state': kept, 'reportgate_independent_clinical_validation': False,
                'reportgate_primary_metric_eligible': False, 'reportgate_regeneration_authorized': False})
    unique = []
    for f in facts[:14]:
        unique.append({'cxr_candidate_id': f['cxr_candidate_id'], 'cxr_sha256': f['artifact_hashes']['cxr_sha256'],
            'finding': f['finding'], 'raw_xrv_state': f['states']['xrv'],
            **{k: f[k] for k in (*w.LIVE_SHARED, 'liveimage_independent_clinical_validation',
                'liveimage_primary_metric_eligible', 'liveimage_regeneration_authorized')}})
    requests = [request(rows[:reports], 'verify_image_finding', 'pneumonia', 'original_image'),
        request(rows[:reports], 'verify_image_report_relation', 'pneumonia', 'original_pair'),
        request(rows[:reports], 'verify_report_assertion', 'pneumonia', 'original_report'),
        request(rows, 'assess_ehr_radiographic_observability', None, 'original_observability')]
    if unchecked:
        requests.append(request(rows[-1:], 'verify_image_finding', 'pneumonia', 'unchecked_request'))
    return rows, facts, unique, requests


def request(rows, kind, head, name):
    row = rows[0]
    return {'request_id': name, 'request_kind': kind, 'case_id': row['case_id'], 'finding': head,
        'dependency_hashes': {k: row[k] for k in w.tables.REQUEST_DEPENDENCIES[kind]},
        'consumer_candidate_ids': [r['triple_candidate_id'] for r in rows],
        'source_evidence_ids': ['invented_evidence'], 'reason_codes': ['unchanged_reason'],
        'execution_status': 'not_executed', 'model_execution_allowed': False,
        'clinical_truth_established': False, 'estimated_model_calls': None,
        'estimated_gpu_seconds': None, 'blocked_by_declared_budget': False,
        'reportcheck_clinically_resolved': False, 'reportgate_clinically_resolved': False,
        'imageverify_clinically_resolved': False, 'imageverify_regeneration_authorized': False,
        'reportcheck_new_model_calls': 0, 'reportgate_new_model_calls': 0, 'imageverify_new_model_calls': 0}


def refresh_unique(data):
    for u in data[2]:
        f = next(f for f in data[1] if f['cxr_candidate_id'] == u['cxr_candidate_id'] and f['finding'] == u['finding'])
        for k in w.LIVE_SHARED:
            u[k] = f[k]
        u['raw_xrv_state'] = f['states']['xrv']


class VerificationFrontierTests(unittest.TestCase):
    def test_lossless_cells_and_execution_history(self):
        data = fixture()
        before = copy.deepcopy(data)
        actions, original, _, _ = w.build(*data)
        self.assertEqual(data, before)
        for old, new in zip(data[0], actions):
            self.assertEqual({k: new[k] for k in old}, old)
        for old, new in zip(data[3], original):
            self.assertEqual({k: new[k] for k in old}, old)
            self.assertEqual(new['execution_status'], 'not_executed')
            self.assertFalse(new['frontier_clinically_resolved'])

    def test_changed_readout_gets_independent_not_regeneration_request(self):
        actions, original, extra, summary = w.build(*fixture())
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]['frontier_priority_tier'], 0)
        self.assertEqual(extra[0]['request_kind'], 'obtain_independent_image_finding_evidence')
        self.assertEqual(extra[0]['current_readout'], 'uncertain')
        self.assertIsNone(extra[0]['confirmed_faulty_modality'])
        self.assertFalse(extra[0]['same_model_retry_is_independent_evidence'])
        self.assertTrue(all(r['frontier_priority_tier'] == 0 for r in actions))
        self.assertEqual(summary['supplemental_logical_requests'], 1)

    def test_four_reports_are_one_image_finding_request(self):
        _, _, extra, summary = w.build(*fixture(reports=4))
        self.assertEqual(len(extra), 1)
        self.assertEqual(len(extra[0]['consumer_candidate_ids']), 4)
        self.assertEqual(summary['supplemental_candidate_request_links'], 4)
        self.assertEqual(summary['supplemental_unique_image_slots'], 1)

    def test_no_hash_only_spread_to_unchecked_image(self):
        actions, original, extra, _ = w.build(*fixture(unchecked=True))
        self.assertEqual(actions[-1]['frontier_priority_tier'], 3)
        self.assertEqual(actions[-1]['frontier_supplemental_logical_request_count'], 0)
        self.assertNotIn(actions[-1]['triple_candidate_id'], extra[0]['consumer_candidate_ids'])
        self.assertEqual(original[-1]['frontier_unchecked_consumer_count'], 1)

    def test_unchecked_candidate_needs_initial_check_even_without_old_request(self):
        data = fixture(unchecked=True, changed=False, current='negative', retained='negative')
        data[3].clear()
        actions, _, _, summary = w.build(*data)
        self.assertEqual(actions[-1]['frontier_priority_tier'], 3)
        self.assertTrue(actions[-1]['frontier_initial_image_check_required'])
        self.assertTrue(actions[-1]['frontier_initial_report_check_required'])
        self.assertTrue(actions[-1]['frontier_next_requirement_has_no_request_id'])
        self.assertEqual(actions[-1]['frontier_next_request_ids'], '[]')
        self.assertFalse(actions[-1]['frontier_model_execution_allowed'])
        self.assertEqual(summary['candidate_slots_requiring_initial_image_check'], 1)

    def test_available_candidate_without_requests_is_not_clinically_qualified(self):
        data = fixture(changed=False, current='negative', retained='negative')
        data[3].clear()
        actions, _, _, _ = w.build(*data)
        self.assertEqual(actions[0]['frontier_priority_tier'], 4)
        self.assertFalse(actions[0]['frontier_initial_image_check_required'])
        self.assertEqual(actions[0]['frontier_next_requirements'], '["independent_clinical_qualification_missing"]')

    def test_explicit_scorer_disagreement_gets_tier_one(self):
        _, _, extra, _ = w.build(*fixture(changed=False, current='negative', opposition=True))
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]['frontier_priority_tier'], 1)
        self.assertEqual(extra[0]['reason_codes'], ['explicit_current_image_scorer_opposition_unqualified'])

    def test_uncertain_unknown_cannot_be_explicit_opposition(self):
        for state in ('unknown', 'uncertain'):
            _, original, extra, _ = w.build(*fixture(changed=False, current=state, opposition=True))
            self.assertEqual(extra, [])
            self.assertEqual(original[0]['frontier_priority_tier'], 4)

    def test_unstable_and_opposition_is_one_request_not_two(self):
        _, _, extra, _ = w.build(*fixture(changed=True, current='negative', opposition=True))
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]['frontier_priority_tier'], 0)
        self.assertEqual(len(extra[0]['reason_codes']), 2)

    def test_stable_agreement_is_not_acceptance(self):
        actions, original, extra, _ = w.build(*fixture(changed=False, current='negative', retained='negative'))
        self.assertEqual(extra, [])
        self.assertEqual(original[0]['frontier_priority_tier'], 4)
        self.assertEqual(original[1]['frontier_priority_tier'], 4)
        self.assertTrue(all(not r['frontier_clinical_selection_eligible'] for r in actions))

    def test_missing_report_assertion_is_scope_request_not_negative(self):
        _, original, _, _ = w.build(*fixture(changed=False, current='negative'))
        self.assertEqual(original[1]['frontier_next_requirements'], ['report_scope_not_retained'])
        self.assertEqual(original[1]['frontier_consumer_relation_counts'], {'report_assertion_not_retained': 4})

    def test_exact_report_slot_needed_even_if_retained_state_present(self):
        data = fixture(changed=False, current='negative', retained='negative')
        for f in data[1]:
            if f['liveimage_status'] == 'checked_clinically_unqualified':
                f['liveimage_exact_report_comparison_available'] = False
                f['liveimage_retained_report_relation'] = 'no_exact_report_check'
        _, original, _, _ = w.build(*data)
        self.assertEqual(original[1]['frontier_priority_tier'], 3)

    def test_no_direct_ehr_reference_not_a_generation_failure(self):
        actions, _, _, summary = w.build(*fixture())
        self.assertEqual(summary['cases_without_direct_radiographic_reference'], 1)
        self.assertTrue(all(r['frontier_ehr_observability'] == 'no_direct_radiographic_reference' for r in actions))
        self.assertFalse(summary['fixed_ehr_changed'])

    def test_explicit_ehr_fact_requires_existing_evidence(self):
        data = fixture()
        for f in data[1]:
            if f['finding'] == 'pneumonia':
                f['states']['ehr'] = 'positive'
        with self.assertRaises(ValueError):
            w.build(*data)
        for f in data[1]:
            if f['finding'] == 'pneumonia':
                f['cached_source_categories'] = ['explicit_diagnosis']
        _, _, _, summary = w.build(*data)
        self.assertEqual(summary['cases_with_direct_radiographic_reference'], 1)

    def test_deterministic_supplement_ids_and_consumer_order(self):
        data = fixture()
        first = w.build(*data)
        second = w.build(*copy.deepcopy(data))
        self.assertEqual(first, second)
        shuffled = copy.deepcopy(data)
        shuffled[0].reverse()
        shuffled[1].reverse()
        shuffled[2].reverse()
        for r in shuffled[3]:
            r['consumer_candidate_ids'].reverse()
        self.assertEqual(first[2], w.build(*shuffled)[2])

    def test_missing_complete_inventory_refused(self):
        for mode in ('missing', 'duplicate', 'unsupported'):
            data = fixture()
            if mode == 'missing':
                data[1].pop()
            elif mode == 'duplicate':
                data[1][-1] = copy.deepcopy(data[1][0])
            else:
                data[1][0]['finding'] = 'invented_unsupported_head'
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_changed_lineage_or_report_hash_refused(self):
        for key in ('case_id', 'report_candidate_id', 'cxr_candidate_id'):
            data = fixture()
            data[1][0][key] = 'invented_other'
            with self.assertRaises(ValueError):
                w.build(*data)
        for key in w.HASH_FIELDS:
            data = fixture()
            data[1][0]['artifact_hashes'][key] = '9' * 64
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_invalid_sha256_refused(self):
        data = fixture()
        data[0][0]['ehr_sha256'] = 'not_a_hash'
        with self.assertRaises(ValueError):
            w.build(*data)

    def test_request_exact_dependencies_and_case_required(self):
        for change in ('hash', 'case', 'keys', 'consumer'):
            data = fixture()
            r = data[3][0]
            if change == 'hash':
                r['dependency_hashes']['cxr_sha256'] = '9' * 64
            elif change == 'case':
                r['case_id'] = 'invented_other'
            elif change == 'keys':
                r['dependency_hashes']['ehr_sha256'] = 'e' * 64
            else:
                r['consumer_candidate_ids'].append('missing_consumer')
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_duplicate_request_or_consumer_refused(self):
        for mode in ('request', 'consumer'):
            data = fixture()
            if mode == 'request':
                data[3].append(copy.deepcopy(data[3][0]))
            else:
                data[3][0]['consumer_candidate_ids'].append(data[3][0]['consumer_candidate_ids'][0])
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_same_image_cannot_have_report_dependent_readout(self):
        data = fixture()
        data[1][14]['states']['xrv'] = 'positive'
        with self.assertRaises(ValueError):
            w.build(*data)

    def test_false_repeatability_and_polarity_relation_refused(self):
        for field, value in (('liveimage_readout_changed', False),
                ('liveimage_xrv_relation', 'explicit_opposition_unqualified'),
                ('liveimage_retained_report_relation', 'explicit_agreement_unqualified')):
            data = fixture()
            next(f for f in data[1] if f['finding'] == 'pneumonia')[field] = value
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_unavailable_readout_is_null_not_unknown(self):
        data = fixture()
        for f in data[1]:
            if f['finding'] in w.guard.HEADS:
                f.update(liveimage_status='image_verifier_unavailable', liveimage_qwen_state=None,
                    liveimage_baseline_state=None, liveimage_readout_changed=None,
                    liveimage_readout_repeatability='image_verifier_unavailable',
                    liveimage_xrv_relation='image_verifier_unavailable',
                    liveimage_retained_report_relation='image_verifier_unavailable')
        refresh_unique(data)
        _, original, extra, _ = w.build(*data)
        self.assertEqual(extra, [])
        self.assertEqual(original[0]['frontier_next_requirements'], ['verifier_result_unavailable'])

    def test_unsupported_head_cannot_receive_state(self):
        data = fixture()
        next(f for f in data[1] if f['finding'] not in w.guard.HEADS)['liveimage_qwen_state'] = 'unknown'
        with self.assertRaises(ValueError):
            w.build(*data)

    def test_unique_sidecar_cannot_omit_or_add_unchecked_slot(self):
        for mode in ('missing', 'unchecked', 'state', 'hash'):
            data = fixture(unchecked=True)
            if mode == 'missing':
                data[2].pop()
            elif mode == 'unchecked':
                data[2][0]['cxr_candidate_id'] = 'same_hash_unchecked'
            elif mode == 'state':
                data[2][0]['liveimage_qwen_state'] = 'positive'
            else:
                data[2][0]['cxr_sha256'] = '9' * 64
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_fixed_ehr_anchor_cannot_be_replaced(self):
        data = fixture()
        data[0][1]['ehr_sha256'] = '9' * 64
        with self.assertRaises(ValueError):
            w.build(*data)

    def test_old_requests_cannot_be_marked_executed_or_authorized(self):
        for key, value in (('execution_status', 'completed'), ('model_execution_allowed', True),
                ('clinical_truth_established', True), ('imageverify_new_model_calls', 1),
                ('reportgate_clinically_resolved', True), ('imageverify_regeneration_authorized', True)):
            data = fixture()
            data[3][0][key] = value
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_input_clinical_flag_cannot_be_promoted(self):
        for key in ('liveimage_primary_metric_eligible', 'reportgate_regeneration_authorized'):
            data = fixture()
            data[1][0][key] = True
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_all_outputs_leave_cost_clinical_truth_and_execution_unavailable(self):
        actions, original, extra, summary = w.build(*fixture())
        self.assertEqual(summary['new_model_calls'], 0)
        self.assertEqual(summary['clinically_resolved_requests'], 0)
        self.assertIsNone(summary['clinical_accuracy'])
        for r in actions:
            self.assertIsNone(r['frontier_clinical_selection_score'])
            self.assertIsNone(r['frontier_estimated_gpu_seconds'])
            self.assertIsNone(r['frontier_declared_budget'])
            self.assertFalse(r['frontier_regeneration_authorized'])
        for r in extra:
            self.assertIsNone(r['estimated_model_calls'])
            self.assertFalse(r['model_execution_allowed'])
        for r in original:
            self.assertFalse(r['frontier_model_execution_allowed'])

    def test_repeat_frontier_annotation_refused(self):
        data = fixture()
        data[0][0]['frontier_priority_tier'] = 0
        with self.assertRaises(ValueError):
            w.build(*data)
        data = fixture()
        data[3][0]['frontier_priority_tier'] = 0
        with self.assertRaises(ValueError):
            w.build(*data)

    def test_slurm_required_before_read_or_directory(self):
        with patch.object(w.tables, 'require_cpu_slurm', side_effect=RuntimeError), \
                patch.object(w, 'new_atomic_run') as new:
            with self.assertRaises(RuntimeError):
                w.execute('/invented', 'test')
            new.assert_not_called()

    def test_existing_run_not_loaded_or_overwritten(self):
        with patch.object(w.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(w, 'load_inputs') as load:
            with self.assertRaises(FileExistsError):
                w.execute('/invented', 'test')
            load.assert_not_called()

    def test_failure_cleans_only_new_atomic_temporary(self):
        with patch.object(w.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', return_value=(Path('/invented/tmp'), Path('/invented/new'))), \
                patch.object(w, 'load_inputs', side_effect=ValueError), \
                patch.object(w, 'discard_atomic_run') as discard:
            with self.assertRaises(ValueError):
                w.execute('/invented', 'test')
            discard.assert_called_once_with(Path('/invented/tmp'))


if __name__ == '__main__':
    unittest.main()
