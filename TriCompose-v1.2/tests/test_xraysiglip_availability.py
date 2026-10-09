"""Wholly invented metadata; no protected cases, pixels, weights or model calls."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

w = load('siglip_availability_fixture', ROOT / 'tools/attach_xraysiglip_availability.py')
vf = load('frontier_invented_support', ROOT / 'tests/test_verification_frontier.py')


def fixture(unchecked=True, template_sensitive=False, failed=False):
    rows, facts, unique, original_requests = vf.fixture(reports=4, unchecked=unchecked,
        changed=False, opposition=True, current='negative')
    image_sha, observation, control_sha, control_id = 'a' * 64, '2' * 64, '3' * 64, '4' * 64
    def receipt(sha, oid, uniform):
        basic = w.guard.metadata_guard(sha, width=64, height=64, mode='RGB', format_name='PNG',
            frames=1, extrema=((0, 0),) * 3 if uniform else ((0, 255),) * 3)
        return w.guard.receipt(sha, basic['status'], basic['reason'],
            evidence=basic['native_image_metadata'], pixel_sha256=oid)
    original_guard, control_guard = receipt(image_sha, observation, False), receipt(control_sha, control_id, True)
    for f in facts:
        if f['cxr_candidate_id'] == 'image0':
            f['liveimage_guard_receipt_sha256'] = original_guard['receipt_sha256']
    vf.refresh_unique((rows, facts, unique, original_requests))
    actions, _, supplements, _ = w.frontier.build(rows, facts, unique, original_requests)
    plan = {'inputs': [
        {'observation_id': observation, 'sha256': image_sha, 'width': 64, 'height': 64},
        {'observation_id': control_id, 'sha256': control_sha, 'width': 64, 'height': 64}],
        'logical_slots': [
            {'slot_id': 'invented_original', 'arm': 'original', 'cxr_candidate_id': 'image0',
                'cxr_sha256': image_sha, 'observation_id': observation, 'width': 64, 'height': 64},
            {'slot_id': 'invented_control', 'arm': 'uniform_black', 'cxr_candidate_id': 'image0',
                'cxr_sha256': image_sha, 'observation_id': control_id, 'width': 64, 'height': 64}],
        'max_scoring_attempts': 1}
    plan['request_refs'] = w.probe.bind_requests(plan, supplements)
    pairs = w.probe.decode_scores([0.1, 0.2] * 24, [-1.0, 1.0] * 24)
    if template_sensitive:
        pairs[-6]['positive_cosine'] = 0.3  # pneumonia, one of its three pairs
    records = [
        {'observation_id': observation, 'artifact_sha256': image_sha, 'guard': original_guard,
            'callback_invoked': True, 'model_forward_attempted': True,
            'pairs': None if failed else pairs, 'contract_status': 'failed_unavailable' if failed else 'complete',
            'failure_reason': 'RuntimeError' if failed else None, 'elapsed_seconds': 0.25,
            'independent_clinical_validation': False},
        {'observation_id': control_id, 'artifact_sha256': control_sha, 'guard': control_guard,
            'callback_invoked': False, 'model_forward_attempted': False, 'pairs': None,
            'contract_status': 'blocked_before_callback', 'failure_reason': control_guard['reason'],
            'elapsed_seconds': None, 'independent_clinical_validation': False}]
    predictions = {'schema_version': w.probe.SCHEMA, 'frozen': True, 'image_only': True,
        'probe_catalog_sha256': w.guard.digest(w.probe.probes()), 'records': records}
    comparisons = w.probe.posthoc(records, plan, supplements)
    summary = {'schema_version': w.probe.SCHEMA, 'unique_inputs': 2, 'scoring_attempts': 1,
        'actual_model_forward_attempts': 1, 'blocked_before_callback': 1, 'model_load_attempts': 1,
        'complete_model_responses': 0 if failed else 1, 'failed_unavailable_responses': int(failed),
        'model_retries': 0, 'logical_requests_compared': len(supplements), 'clinical_requests_resolved': 0,
        'clinical_accuracy': None, 'posthoc_comparisons_computed_after_prediction_fsync': True,
        'shares_vision_encoder_with_chexagent2': True, 'training_overlap_unverified': True,
        'distinct_from_qwen_and_xrv_not_independent_clinical_truth': True,
        'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
        'fixed_ehr_changed': False, 'calibration_or_policy_fitting': False,
        'raw_patient_inputs_opened': False, 'ehr_report_or_real_target_bodies_opened': False}
    journal = [{'event': 'reserved', 'scoring_attempt': 1, 'observation_id': observation},
        {'event': 'failed_unavailable' if failed else 'completed', 'scoring_attempt': 1,
            'observation_id': observation, 'model_forward_attempted': True,
            'failure_reason': 'RuntimeError' if failed else None, 'elapsed_seconds': 0.25}]
    return [actions, facts, unique, supplements, plan, predictions, comparisons, summary, journal]


class SiglipAvailabilityTests(unittest.TestCase):
    def test_lossless_inputs_cells_and_order(self):
        data = fixture()
        old = copy.deepcopy(data)
        outputs = w.build(*data)
        self.assertEqual(data, old)
        for before, after in zip(data[:4], outputs[:4]):
            self.assertEqual(len(before), len(after))
            for a, b in zip(before, after):
                self.assertEqual(a, {k: b[k] for k in a})

    def test_hash_only_match_does_not_extend_scope(self):
        rows, facts, _, _, summary = w.build(*fixture())
        self.assertEqual(rows[-1]['siglip_status'], 'not_checked')
        self.assertIsNone(rows[-1]['siglip_supported_readout_count'])
        self.assertTrue(all(f['siglip_readout'] is None for f in facts[-14:]))
        self.assertEqual(summary['checked_candidate_slots'], 4)
        self.assertEqual(summary['unchecked_candidate_slots'], 1)

    def test_all_fourteen_findings_kept_eight_have_numeric_readouts(self):
        rows, facts, unique, _, summary = w.build(*fixture())
        self.assertEqual(len(facts), 70)
        self.assertEqual(len(unique), 14)
        self.assertEqual(summary['unique_supported_image_findings'], 8)
        self.assertEqual(rows[0]['siglip_supported_readout_count'], 8)
        self.assertEqual(rows[0]['siglip_outside_scope_count'], 6)
        self.assertEqual({f['finding'] for f in facts[:14]}, set(w.CHEXPERT_FINDINGS))

    def test_three_template_raw_margins_retained_not_probabilities(self):
        _, facts, _, _, _ = w.build(*fixture())
        value = next(f['siglip_readout'] for f in facts if f['finding'] == 'pneumonia')
        self.assertEqual(len(value['cosine_margins']), 3)
        self.assertAlmostEqual(value['mean_cosine_margin'], -0.1)
        self.assertIsNone(value['clinical_state'])
        self.assertIsNone(value['calibrated_probability'])

    def test_template_sensitive_not_converted_to_winner_or_negative(self):
        _, _, _, requests, _ = w.build(*fixture(template_sensitive=True))
        self.assertEqual(requests[0]['siglip_status'], 'checked_template_sensitive')
        self.assertEqual(requests[0]['siglip_vs_raw_xrv'], 'not_comparable')
        self.assertEqual(requests[0]['siglip_vs_current_qwen'], 'not_comparable')
        self.assertIsNone(requests[0]['siglip_clinical_state'])

    def test_unknown_uncertain_missing_are_not_negative(self):
        absent = w.probe.summarize_pairs(w.probe.decode_scores([0.1, 0.2] * 24, [0.0] * 48))['pneumonia']
        for state in ('unknown', 'uncertain'):
            self.assertEqual(w.direction(state, absent), 'not_comparable')
        self.assertEqual(w.direction('positive', None), 'not_comparable')
        self.assertEqual(w.direction('negative', absent), 'same_direction_unqualified')
        self.assertEqual(w.direction('positive', absent), 'opposed_direction_unqualified')

    def test_missing_scope_does_not_get_numeric_zero(self):
        unsupported = w.extension(fixture()[5]['records'][0], 'support_devices')
        unchecked = w.extension(None, 'pneumonia')
        self.assertEqual(unsupported['siglip_status'], 'outside_siglip_probe_scope')
        self.assertIsNone(unsupported['siglip_readout'])
        self.assertIsNone(unchecked['siglip_readout'])
        self.assertIsNone(unchecked['siglip_run_manifest_sha256'])

    def test_one_image_eight_heads_not_four_independent_votes(self):
        _, _, _, _, summary = w.build(*fixture())
        self.assertEqual(summary['checked_image_slots'], 1)
        self.assertEqual(summary['unique_supported_image_findings'], 8)
        self.assertEqual(summary['supplemental_logical_requests'], 1)
        self.assertEqual(summary['supplemental_candidate_finding_links'], 4)
        self.assertEqual(summary['historical_siglip_actual_forward_attempts'], 1)
        self.assertEqual(summary['new_model_calls'], 0)

    def test_failed_attempt_keeps_cost_but_no_readout(self):
        rows, facts, _, requests, summary = w.build(*fixture(failed=True))
        self.assertEqual(rows[0]['siglip_status'], 'readout_unavailable')
        self.assertEqual(rows[0]['siglip_supported_readout_count'], 0)
        self.assertEqual(rows[0]['siglip_unavailable_count'], 8)
        self.assertIsNone(requests[0]['siglip_readout'])
        self.assertEqual(summary['historical_siglip_scoring_attempts'], 1)
        self.assertEqual(summary['historical_siglip_actual_forward_attempts'], 1)
        self.assertTrue(all(f['siglip_readout'] is None for f in facts))

    def test_controls_never_propagate(self):
        data = fixture()
        rows, _, unique, _, _ = w.build(*data)
        self.assertFalse(any(r.get('observation_id') == '4' * 64 for r in rows + unique))
        data[5]['records'][1]['pairs'] = data[5]['records'][0]['pairs']
        with self.assertRaises(ValueError):
            w.build(*data)

    def test_no_clinical_flags_scores_actions_or_history_changed(self):
        data = fixture()
        rows, facts, _, requests, summary = w.build(*data)
        for row in rows:
            self.assertFalse(row['siglip_clinical_selection_eligible'])
            self.assertFalse(row['siglip_primary_metric_eligible'])
            self.assertFalse(row['siglip_regeneration_authorized'])
            self.assertFalse(row['siglip_report_factuality_verified'])
            self.assertIsNone(row['siglip_clinical_selection_score'])
        for request in requests:
            self.assertEqual(request['execution_status'], 'not_executed')
            self.assertFalse(request['siglip_independent_evidence_requirement_satisfied'])
            self.assertFalse(request['siglip_clinically_resolved'])
            self.assertIsNone(request['siglip_confirmed_faulty_modality'])
        self.assertEqual(summary['clinical_requests_resolved'], 0)
        self.assertFalse(summary['selection_changed'])
        self.assertTrue(all(f['siglip_shares_encoder_with_chexagent2'] for f in facts))

    def test_repeated_attachment_refused(self):
        data = fixture()
        for index in (0, 1, 2, 3):
            modified = copy.deepcopy(data)
            modified[index][0]['siglip_status'] = 'not_checked'
            with self.assertRaises(ValueError):
                w.build(*modified)

    def test_determinism(self):
        data = fixture(template_sensitive=True)
        a, b = w.build(*data), w.build(*copy.deepcopy(data))
        self.assertEqual(a, b)
        self.assertEqual(w.tables.csv_text(a[0]), w.tables.csv_text(b[0]))
        self.assertEqual(w.tables.jsonl_text(a[1]), w.tables.jsonl_text(b[1]))

    def test_exact_case_guard_image_and_proxy_tampering_refused(self):
        for kind in ('case', 'hash', 'guard', 'proxy', 'consumer', 'unrelated_head_guard'):
            data = fixture()
            if kind == 'case':
                data[3][0]['case_id'] = 'wrong_case'
            elif kind == 'hash':
                data[3][0]['dependency_hashes']['cxr_sha256'] = 'f' * 64
            elif kind == 'guard':
                data[5]['records'][0]['guard']['receipt_sha256'] = 'f' * 64
            elif kind == 'proxy':
                data[3][0]['current_readout'] = 'positive'
            elif kind == 'consumer':
                data[3][0]['consumer_candidate_ids'].pop()
            else:
                data[1][0]['liveimage_guard_receipt_sha256'] = 'f' * 64
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_posthoc_comparison_and_request_reference_tampering_refused(self):
        for index, key, value in ((6, 'vs_current_qwen', 'opposed_direction_unqualified'),
                (4, 'case_id', 'wrong_case')):
            data = fixture()
            if index == 4:
                data[4]['request_refs'][0][key] = value
            else:
                data[6][0][key] = value
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_count_budget_journal_and_authorization_tampering_refused(self):
        for kind in ('calls', 'attempts', 'budget', 'events', 'order', 'authorization', 'independence', 'loads', 'missing_load'):
            data = fixture()
            if kind == 'calls':
                data[7]['actual_model_forward_attempts'] = 0
            elif kind == 'attempts':
                data[7]['scoring_attempts'] = True
            elif kind == 'budget':
                data[4]['max_scoring_attempts'] = 0
            elif kind == 'events':
                data[8].pop()
            elif kind == 'order':
                data[8].reverse()
            elif kind == 'authorization':
                data[7]['regeneration_authorized'] = True
            elif kind == 'independence':
                data[7]['distinct_from_qwen_and_xrv_not_independent_clinical_truth'] = False
            elif kind == 'missing_load':
                data[7]['model_load_attempts'] = 0
            else:
                data[7]['model_load_attempts'] = 2
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_complete_probe_inventory_and_finite_scores_required(self):
        for kind in ('missing', 'nonfinite', 'bounds', 'catalog'):
            data = fixture()
            if kind == 'missing':
                data[5]['records'][0]['pairs'].pop()
            elif kind == 'nonfinite':
                data[5]['records'][0]['pairs'][0]['positive_cosine'] = float('nan')
            elif kind == 'bounds':
                data[5]['records'][0]['pairs'][0]['negative_cosine'] = 1.2
            else:
                data[5]['probe_catalog_sha256'] = 'f' * 64
            with self.assertRaises(ValueError):
                w.build(*data)

    def test_failed_metadata_load_cleans_only_new_atomic_temporary(self):
        with patch.object(w.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', return_value=(Path('/invented/new_tmp'), Path('/invented/new_run'))), \
                patch.object(w, 'load_inputs', side_effect=ValueError('injected_failure')), \
                patch.object(w, 'discard_atomic_run') as discard:
            with self.assertRaises(ValueError):
                w.execute(Path('/invented'), 'new_run')
            discard.assert_called_once_with(Path('/invented/new_tmp'))

    def test_existing_run_refusal_not_cleaned_or_overwritten(self):
        with patch.object(w.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', side_effect=FileExistsError('existing')), \
                patch.object(w, 'load_inputs') as load_inputs, patch.object(w, 'discard_atomic_run') as discard:
            with self.assertRaises(FileExistsError):
                w.execute(Path('/invented'), 'existing')
            load_inputs.assert_not_called()
            discard.assert_not_called()

    def test_no_body_or_recursive_source_reader_in_metadata_loader(self):
        source = (ROOT / 'tools/attach_xraysiglip_availability.py').read_text()
        for forbidden in ('from_pretrained', 'import torch', 'Image.open(', 'read_report_text(', 'manifest[\'source_paths\']'):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
