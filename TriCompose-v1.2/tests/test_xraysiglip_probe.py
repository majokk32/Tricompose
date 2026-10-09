"""Invented metadata and callbacks only; no model/pixel/weight access."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('xraysiglip_probe_fixture', ROOT / 'tools/probe_xraysiglip_findings.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def model_metadata():
    return ({'architectures': ['SiglipModel'], 'model_type': 'siglip', 'torch_dtype': 'float32',
        'vision_config': {'image_size': 512}, 'text_config': {'max_position_embeddings': 256}},
        {'processor_class': 'SiglipProcessor', 'size': {'height': 512, 'width': 512},
            'do_normalize': True, 'do_rescale': True, 'do_resize': True, 'resample': 3,
            'image_mean': [0.5] * 3, 'image_std': [0.5] * 3, 'rescale_factor': 1 / 255},
        {'model_max_length': 64, 'tokenizer_class': 'SiglipTokenizer'})


def scores(positive=0.2, negative=0.1):
    return w.decode_scores([v for _ in range(24) for v in (positive, negative)], [1.0, 0.5] * 24)


def fixture():
    inp = [{'observation_id': 'a' * 64, 'sha256': 'c' * 64, 'width': 64, 'height': 64,
                'path': '/invented/original.png', 'image_file_stats': [12, 34]},
        {'observation_id': 'b' * 64, 'sha256': 'd' * 64, 'width': 64, 'height': 64,
                'path': '/invented/control.png', 'image_file_stats': [12, 35]}]
    slots = [{'slot_id': 'original_slot', 'arm': 'original', 'cxr_candidate_id': 'invented_image',
            'cxr_sha256': 'c' * 64, 'observation_id': 'a' * 64, 'width': 64, 'height': 64},
        {'slot_id': 'control_slot', 'arm': 'uniform_black', 'cxr_candidate_id': 'invented_image',
            'cxr_sha256': 'c' * 64, 'observation_id': 'b' * 64, 'width': 64, 'height': 64}]
    requests = [{'request_id': 'invented_request', 'cxr_candidate_id': 'invented_image',
        'case_id': 'invented_case', 'finding': 'pneumonia',
        'request_kind': 'obtain_independent_image_finding_evidence', 'execution_status': 'not_executed',
        'source_availability_manifest_sha256': '0120146adc043b482f51ef515d70ad3dd1b78132c3b989b22e7bd918b3c68dcb',
        'dependency_hashes': {'cxr_sha256': 'c' * 64},
        'consumer_candidate_ids': ['invented_triple0', 'invented_triple1', 'invented_triple2', 'invented_triple3'],
        'model_execution_allowed': False, 'regeneration_authorized': False, 'clinical_truth_established': False,
        'clinically_resolved': False, 'primary_metric_eligible': False,
        'current_readout': 'positive', 'raw_xrv_state': 'negative'}]
    plan = {'inputs': inp, 'logical_slots': slots, 'max_scoring_attempts': 1}
    plan['request_refs'] = w.bind_requests(plan, requests)
    return plan, requests


def guard_receipt(item, uniform=False):
    value = w.guard.metadata_guard(item['sha256'], width=64, height=64, mode='RGB', format_name='PNG',
        frames=1, extrema=((0, 0),) * 3 if uniform else ((0, 255),) * 3)
    return w.guard.receipt(item['sha256'], value['status'], value['reason'],
        evidence=value['native_image_metadata'], pixel_sha256=item['observation_id'])


def fake_guard(plan):
    by_path = {r['path']: r for r in plan['inputs']}
    def invoke(path, sha, callback, *, Image):
        item = by_path[path]
        uniform = path.endswith('control.png')
        current = guard_receipt(item, uniform)
        value = None if uniform else callback(SimpleNamespace(width=64, height=64, token=item['observation_id']))
        return {'guard': current, 'callback_invoked': not uniform, 'callback_result': value}
    return invoke


def successful_callback(_):
    return {'pairs': scores(), 'failure_reason': None, 'model_forward_attempted': True}


def invoke_fixture(plan, callback=successful_callback, **kwargs):
    with patch.object(w.guard, 'guarded_invoke', side_effect=fake_guard(plan)), \
            patch.object(w.guard, 'normalized_pixel_sha', side_effect=lambda im: im.token):
        return w.invoke(plan, callback, Image=None, **kwargs)


class XraySiglipProbeTests(unittest.TestCase):
    def test_catalog_has_three_polarity_pairs_per_head(self):
        catalog = w.probes()
        self.assertEqual(len(catalog), 24)
        self.assertEqual({p['finding'] for p in catalog}, set(w.guard.HEADS))
        self.assertEqual(len({p['probe_id'] for p in catalog}), 24)
        self.assertEqual({p['family'] for p in catalog}, set(w.FAMILIES))
        self.assertTrue(all(p['positive_text'] != p['negative_text'] for p in catalog))

    def test_catalog_deterministic_and_no_ehr_view_or_severity_addition(self):
        self.assertEqual(w.probes(), w.probes())
        text = ' '.join(p[k] for p in w.probes() for k in ('positive_text', 'negative_text')).lower()
        for unsupported in ('left', 'right', 'portable', 'frontal', 'severe', 'pacemaker', 'prior', 'diagnosis', 'ehr'):
            self.assertNotIn(unsupported, text)

    def test_native_model_metadata_512_not_repository_name_384(self):
        w.validate_model_metadata(*model_metadata())
        for kind, key, value in ((0, 'architectures', ['SiglipForImageClassification']),
                (0, 'torch_dtype', 'bfloat16'), (1, 'size', {'height': 384, 'width': 384}),
                (2, 'model_max_length', 256), (1, 'do_normalize', False)):
            data = model_metadata()
            data[kind][key] = value
            with self.assertRaises(ValueError):
                w.validate_model_metadata(*data)

    def test_raw_score_inventory_and_bounded_finite_values(self):
        self.assertEqual(len(scores()), 24)
        for value in (float('nan'), float('inf'), 1.1, True):
            c = [0.1] * 48
            c[0] = value
            with self.assertRaises(ValueError):
                w.decode_scores(c, [1.0] * 48)
        with self.assertRaises(ValueError):
            w.decode_scores([0.1] * 47, [1.0] * 48)

    def test_finite_logits_not_calibrated_probabilities(self):
        pairs = w.decode_scores([0.2, 0.1] * 24, [100.0, -100.0] * 24)
        finding = w.summarize_pairs(pairs)['pneumonia']
        self.assertEqual(finding['text_preference'], 'present_prompt_higher')
        self.assertIsNone(finding['clinical_state'])
        self.assertIsNone(finding['calibrated_probability'])

    def test_three_template_signs_all_used_without_best_template_selection(self):
        for p, n, preference in ((0.2, 0.1, 'present_prompt_higher'),
                (0.1, 0.2, 'absent_prompt_higher'), (0.1, 0.1, 'tied_templates')):
            result = w.summarize_pairs(scores(p, n))
            self.assertEqual(result['pneumonia']['text_preference'], preference)
            self.assertEqual(len(result['pneumonia']['cosine_margins']), 3)
        pairs = scores()
        pairs[0]['positive_cosine'] = 0.0
        result = w.summarize_pairs(pairs)['atelectasis']
        self.assertEqual(result['text_preference'], 'template_sensitive')
        self.assertLess(result['min_cosine_margin'], 0)
        self.assertGreater(result['max_cosine_margin'], 0)

    def test_missing_duplicate_reordered_or_private_pair_fields_refused(self):
        for kind in ('missing', 'duplicate', 'reorder', 'private'):
            pairs = scores()
            if kind == 'missing':
                pairs.pop()
            elif kind == 'duplicate':
                pairs[-1] = copy.deepcopy(pairs[0])
            elif kind == 'reorder':
                pairs.reverse()
            else:
                pairs[0]['raw_report'] = 'invented text forbidden by the metadata contract'
            with self.assertRaises(ValueError):
                w.summarize_pairs(pairs)

    def test_preparation_refs_exclude_proxy_states_and_preserve_source_request(self):
        plan, requests = fixture()
        before = copy.deepcopy(requests)
        refs = w.bind_requests(plan, requests)
        self.assertEqual(requests, before)
        self.assertFalse(any(k in refs[0] for k in ('current_readout', 'raw_xrv_state', 'baseline_readout', 'reason_codes')))

    def test_four_consumers_do_not_expand_image_inventory(self):
        plan, requests = fixture()
        self.assertEqual(len(w.bind_requests(plan, requests)), 1)
        self.assertEqual(len(plan['request_refs'][0]['consumer_candidate_ids']), 4)
        records, attempts = invoke_fixture(plan)
        self.assertEqual(attempts, 1)
        self.assertEqual(len(records), 2)

    def test_control_never_enters_request_inventory(self):
        plan, requests = fixture()
        plan['logical_slots'][1]['arm'] = 'original'
        with self.assertRaises(ValueError):
            w.bind_requests(plan, requests)

    def test_request_hash_candidate_or_authorization_tampering_refused(self):
        for kind in ('hash', 'candidate', 'permission', 'duplicate', 'missing_image'):
            plan, requests = fixture()
            if kind == 'hash':
                requests[0]['dependency_hashes']['cxr_sha256'] = 'e' * 64
            elif kind == 'candidate':
                requests[0]['cxr_candidate_id'] = 'missing'
            elif kind == 'permission':
                requests[0]['regeneration_authorized'] = True
            elif kind == 'duplicate':
                requests.append(copy.deepcopy(requests[0]))
            else:
                requests.clear()
            with self.assertRaises(ValueError):
                w.bind_requests(plan, requests)

    def test_scoring_callback_only_sees_checked_pixels_not_ids_states_or_reports(self):
        plan, _ = fixture()
        seen = []
        def callback(image):
            seen.append(set(vars(image)))
            return successful_callback(image)
        invoke_fixture(plan, callback)
        self.assertEqual(seen, [{'width', 'height', 'token'}])

    def test_uniform_control_has_null_scores_and_zero_callbacks(self):
        plan, _ = fixture()
        records, _ = invoke_fixture(plan)
        blocked = records[1]
        self.assertEqual(blocked['contract_status'], 'blocked_before_callback')
        self.assertIsNone(blocked['pairs'])
        self.assertFalse(blocked['callback_invoked'])
        self.assertFalse(blocked['model_forward_attempted'])

    def test_reservation_precedes_scoring_and_completion(self):
        plan, _ = fixture()
        order = []
        def callback(image):
            order.append('infer')
            return successful_callback(image)
        invoke_fixture(plan, callback, reserve=lambda *_: order.append('reserve'),
            complete=lambda *_: order.append('complete'))
        self.assertEqual(order, ['reserve', 'infer', 'complete'])

    def test_failed_callback_consumes_attempt_and_does_not_become_negative(self):
        plan, _ = fixture()
        records, attempts = invoke_fixture(plan, lambda _: {'pairs': None,
            'failure_reason': 'model_unavailable', 'model_forward_attempted': False})
        self.assertEqual(attempts, 1)
        self.assertEqual(records[0]['contract_status'], 'failed_unavailable')
        self.assertIsNone(records[0]['pairs'])
        self.assertFalse(records[0]['model_forward_attempted'])

    def test_failure_after_forward_is_charged_and_unavailable(self):
        plan, _ = fixture()
        records, attempts = invoke_fixture(plan, lambda _: {'pairs': None,
            'failure_reason': 'OutOfMemoryError', 'model_forward_attempted': True})
        self.assertEqual(attempts, 1)
        self.assertTrue(records[0]['model_forward_attempted'])
        self.assertIsNone(records[0]['pairs'])

    def test_zero_budget_blocks_callback_before_execution(self):
        plan, _ = fixture()
        plan['max_scoring_attempts'] = 0
        seen = []
        with self.assertRaises(ValueError):
            invoke_fixture(plan, lambda im: seen.append(im))
        self.assertEqual(seen, [])

    def test_boolean_budget_and_unsorted_inputs_refused(self):
        for mode in ('boolean', 'order'):
            plan, _ = fixture()
            if mode == 'boolean':
                plan['max_scoring_attempts'] = True
            else:
                plan['inputs'].reverse()
            with self.assertRaises(ValueError):
                invoke_fixture(plan)

    def test_invalid_pixel_identity_blocks_before_reservation(self):
        plan, _ = fixture()
        seen = []
        with patch.object(w.guard, 'guarded_invoke', side_effect=fake_guard(plan)), \
                patch.object(w.guard, 'normalized_pixel_sha', return_value='e' * 64):
            with self.assertRaises(ValueError):
                w.invoke(plan, successful_callback, Image=None, reserve=lambda *_: seen.append(True))
        self.assertEqual(seen, [])

    def test_complete_scores_cannot_claim_no_forward_or_contain_private_callback_fields(self):
        for mode in ('forward', 'private'):
            plan, _ = fixture()
            def callback(_):
                result = successful_callback(None)
                if mode == 'forward':
                    result['model_forward_attempted'] = False
                else:
                    result['raw_report'] = 'invented forbidden field'
                return result
            with self.assertRaises(ValueError):
                invoke_fixture(plan, callback)

    def test_posthoc_comparison_is_not_clinical_resolution(self):
        plan, requests = fixture()
        records, _ = invoke_fixture(plan)
        result = w.posthoc(records, plan, requests)[0]
        self.assertEqual(result['vs_current_qwen'], 'same_direction_unqualified')
        self.assertEqual(result['vs_raw_xrv'], 'opposed_direction_unqualified')
        self.assertFalse(result['clinical_request_resolved'])
        self.assertFalse(result['primary_metric_eligible'])
        self.assertFalse(result['regeneration_authorized'])
        self.assertIsNone(result['confirmed_faulty_modality'])

    def test_unknown_uncertain_and_missing_scores_are_not_comparable(self):
        for state in ('uncertain', 'unknown'):
            plan, requests = fixture()
            requests[0]['current_readout'] = state
            records, _ = invoke_fixture(plan)
            self.assertEqual(w.posthoc(records, plan, requests)[0]['vs_current_qwen'], 'not_comparable')
        plan, requests = fixture()
        records, _ = invoke_fixture(plan, lambda _: {'pairs': None,
            'failure_reason': 'unavailable', 'model_forward_attempted': False})
        result = w.posthoc(records, plan, requests)[0]
        self.assertIsNone(result['siglip_readout'])
        self.assertEqual(result['vs_raw_xrv'], 'not_comparable')

    def test_template_sensitivity_does_not_resolve_proxy_opposition(self):
        plan, requests = fixture()
        pairs = scores()
        next(p for p in pairs if p['finding'] == 'pneumonia')['positive_cosine'] = 0.0
        records, _ = invoke_fixture(plan, lambda _: {'pairs': pairs,
            'failure_reason': None, 'model_forward_attempted': True})
        result = w.posthoc(records, plan, requests)[0]
        self.assertEqual(result['siglip_readout']['text_preference'], 'template_sensitive')
        self.assertEqual(result['vs_current_qwen'], 'not_comparable')

    def test_posthoc_exact_candidate_hash_and_consumer_ids_required(self):
        for mode in ('hash', 'consumer', 'candidate', 'duplicate'):
            plan, requests = fixture()
            records, _ = invoke_fixture(plan)
            if mode == 'hash':
                requests[0]['dependency_hashes']['cxr_sha256'] = 'e' * 64
            elif mode == 'consumer':
                requests[0]['consumer_candidate_ids'] = ['unchecked_same_text']
            elif mode == 'candidate':
                requests[0]['cxr_candidate_id'] = 'unchecked_same_hash'
            else:
                requests.append(copy.deepcopy(requests[0]))
            with self.assertRaises(ValueError):
                w.posthoc(records, plan, requests)

    def test_posthoc_guard_or_weight_truth_promotion_refused(self):
        for mode in ('guard', 'truth', 'artifact'):
            plan, requests = fixture()
            records, _ = invoke_fixture(plan)
            if mode == 'guard':
                records[0]['guard']['clinical_acceptance'] = True
            elif mode == 'truth':
                records[0]['independent_clinical_validation'] = True
            else:
                records[0]['artifact_sha256'] = 'e' * 64
            with self.assertRaises(ValueError):
                w.posthoc(records, plan, requests)

    def test_slurm_guard_before_any_preparation_read_or_directory(self):
        with patch.object(w.tables, 'require_cpu_slurm', side_effect=RuntimeError), \
                patch.object(w, 'new_atomic_run') as new:
            with self.assertRaises(RuntimeError):
                w.execute_prepare('/invented', 'test')
            new.assert_not_called()

    def test_existing_preparation_run_refused_without_reads(self):
        with patch.object(w.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', side_effect=FileExistsError), patch.object(w, 'prepare') as prepare:
            with self.assertRaises(FileExistsError):
                w.execute_prepare('/invented', 'test')
            prepare.assert_not_called()

    def test_preparation_failure_discards_only_new_metadata_temporary(self):
        with patch.object(w.tables, 'require_cpu_slurm'), \
                patch.object(w, 'new_atomic_run', return_value=(Path('/invented/tmp'), Path('/invented/new'))), \
                patch.object(w, 'prepare', side_effect=ValueError), patch.object(w, 'discard_atomic_run') as discard:
            with self.assertRaises(ValueError):
                w.execute_prepare('/invented', 'test')
            discard.assert_called_once_with(Path('/invented/tmp'))

    def test_gpu_probe_requires_explicit_approval_before_model_import(self):
        with patch.object(w.tables, 'require_cpu_slurm'), patch.object(w, 'load_plan') as load:
            with self.assertRaises(RuntimeError):
                w.execute_run('/invented', '/invented', 'test', approved=False)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
