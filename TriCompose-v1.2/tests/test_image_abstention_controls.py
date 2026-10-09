"""Invented buffers, hashes and mock calls; no models or patient inputs."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('image_control_fixture',
    ROOT/'tools/verify_image_abstention_controls.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Buffer:
    mode = 'RGB'
    def __init__(self, value, size=(64, 64)):
        self.size = size
        self.width, self.height = size
        self.value = value
    def tobytes(self):
        return bytes([self.value])*(self.width*self.height*3)


def fixture(control_state='unknown', fresh_state='positive', failed_arm=None):
    images = [{'cxr_candidate_id': 'invented_image_0', 'cxr_sha256': 'a'*64},
        {'cxr_candidate_id': 'invented_image_1', 'cxr_sha256': 'b'*64}]
    slots = m.logical_slots(images)
    frames = [Buffer(100), Buffer(0), Buffer(255), Buffer(101), Buffer(0), Buffer(255)]
    unique, realized = m.deduplicate_frames(slots, frames)
    by_observation = {}
    for slot in realized:
        by_observation.setdefault(slot['observation_id'], slot['arm'])
    records = []
    for key in sorted(unique):
        arm = by_observation[key]
        complete = arm != failed_arm
        states = dict.fromkeys(m.cached.image_interface.FINDINGS,
            fresh_state if arm == 'original' else control_state) if complete else None
        records.append({'observation_id': key, 'contract_status': 'complete' if complete else 'failed_unavailable',
            'states': states, 'token_limit_reached': False, 'independent_clinical_validation': False})
    baseline = {'schema_version': m.cached.SCHEMA, 'frozen': True, 'image_only': True,
        'model_received_ehr_reports_scores_or_candidate_ids': False,
        'records': [{'cxr_candidate_id': image['cxr_candidate_id'], 'cxr_sha256': image['cxr_sha256'],
            'contract_status': 'complete', 'states': dict.fromkeys(m.cached.image_interface.FINDINGS, 'positive')}
            for image in images]}
    return records, realized, baseline


class ImageAbstentionControlTests(unittest.TestCase):
    def test_fixed_three_arms_per_source_no_outcome_selection(self):
        inputs = [{'cxr_candidate_id': str(i), 'cxr_sha256': 'a'*64} for i in range(6)]
        before = copy.deepcopy(inputs)
        slots = m.logical_slots(inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(len(slots), 18)
        for i in range(6):
            self.assertEqual([s['arm'] for s in slots[i*3:i*3+3]], list(m.ARMS))

    def test_duplicate_logical_source_refused(self):
        image = {'cxr_candidate_id': 'invented', 'cxr_sha256': 'a'*64}
        with self.assertRaises(ValueError):
            m.logical_slots([image, image])

    def test_pixel_hash_requires_mode_dimensions_and_complete_buffer(self):
        frame = Buffer(0)
        self.assertNotEqual(m.pixel_sha(frame), m.pixel_sha(Buffer(255)))
        self.assertNotEqual(m.pixel_sha(frame), m.pixel_sha(Buffer(0, (128, 64))))
        for key, value in (('mode', 'L'), ('size', (63, 64)), ('size', (64, 4097))):
            bad = Buffer(0)
            setattr(bad, key, value)
            with self.assertRaises(ValueError):
                m.pixel_sha(bad)
        bad = Buffer(0)
        bad.tobytes = lambda: b'bad'
        with self.assertRaises(ValueError):
            m.pixel_sha(bad)

    def test_identical_uniform_controls_deduplicated_not_two_independent_calls(self):
        records, slots, _ = fixture()
        self.assertEqual(len(records), 4)
        self.assertEqual(len(slots), 6)
        for arm in ('uniform_black', 'uniform_white'):
            self.assertEqual(len({s['observation_id'] for s in slots if s['arm'] == arm}), 1)

    def test_deduplication_includes_dimensions(self):
        slots = [{'slot_id': 'a'}, {'slot_id': 'b'}]
        unique, _ = m.deduplicate_frames(slots, [Buffer(0), Buffer(0, (128, 64))])
        self.assertEqual(len(unique), 2)

    def test_incorrect_frame_inventory_refused(self):
        with self.assertRaises(ValueError):
            m.deduplicate_frames([{'slot_id': 'a'}], [])

    def test_unknown_control_is_successful_no_information_abstention(self):
        summary, _ = m.analyze(*fixture())
        for arm in ('uniform_black', 'uniform_white'):
            value = summary['arms'][arm]
            self.assertEqual(value['logical_slots'], 2)
            self.assertEqual(value['unique_inputs'], 1)
            self.assertEqual(value['state_counts']['unknown'], 8)
            self.assertEqual(value['unknown_fraction_on_complete'], 1)
            self.assertEqual(value['explicit_assertion_fraction_on_complete'], 0)
        self.assertEqual(summary['actual_model_calls'], 4)

    def test_negative_control_assertions_are_unsupported_not_safe_abstention(self):
        for state in ('positive', 'negative'):
            summary, _ = m.analyze(*fixture(control_state=state))
            for arm in ('uniform_black', 'uniform_white'):
                self.assertEqual(summary['arms'][arm]['explicit_assertion_fraction_on_complete'], 1)
                self.assertEqual(summary['arms'][arm]['unknown_fraction_on_complete'], 0)

    def test_uncertain_counted_separately_not_negative_or_unknown(self):
        summary, _ = m.analyze(*fixture(control_state='uncertain'))
        value = summary['arms']['uniform_black']
        self.assertEqual(value['state_counts']['uncertain'], 8)
        self.assertEqual(value['state_counts']['unknown'], 0)
        self.assertEqual(value['explicit_assertion_fraction_on_complete'], 0)

    def test_failed_response_not_successful_unknown_and_bounds_keep_missingness(self):
        summary, _ = m.analyze(*fixture(failed_arm='uniform_black'))
        value = summary['arms']['uniform_black']
        self.assertEqual(value['unavailable_unique_responses'], 1)
        self.assertEqual(value['all_finding_slots'], 8)
        self.assertEqual(value['observed_finding_slots'], 0)
        self.assertIsNone(value['explicit_assertion_fraction_on_complete'])
        self.assertIsNone(value['unknown_fraction_on_complete'])
        self.assertEqual(value['explicit_assertion_all_slot_lower_bound'], 0)
        self.assertEqual(value['explicit_assertion_all_slot_upper_bound'], 1)

    def test_original_repeatability_not_clinical_accuracy(self):
        summary, rows = m.analyze(*fixture())
        self.assertEqual(summary['original_readout_match_fraction'], 1)
        self.assertEqual(summary['original_readout_comparable_finding_slots'], 16)
        self.assertTrue(all(r['clinical_accuracy'] is None for r in rows))
        self.assertIsNone(summary['independent_clinical_accuracy'])

    def test_original_disagreement_does_not_change_cached_reference(self):
        parts = fixture(fresh_state='negative')
        before = copy.deepcopy(parts)
        summary, _ = m.analyze(*parts)
        self.assertEqual(parts, before)
        self.assertEqual(summary['original_readout_matching_states'], 0)
        self.assertEqual(summary['original_readout_match_fraction'], 0)

    def test_unavailable_original_retained_in_denominator(self):
        summary, _ = m.analyze(*fixture(failed_arm='original'))
        self.assertEqual(summary['original_readout_unavailable_finding_slots'], 16)
        self.assertEqual(summary['original_readout_comparable_finding_slots'], 0)
        self.assertIsNone(summary['original_readout_match_fraction'])

    def test_missing_duplicate_foreign_predictions_refused(self):
        for mode in ('missing', 'duplicate', 'foreign'):
            records, slots, baseline = fixture()
            if mode == 'missing': records.pop()
            elif mode == 'duplicate': records.append(copy.deepcopy(records[0]))
            else: records[0]['observation_id'] = 'foreign'
            with self.assertRaises(ValueError):
                m.analyze(records, slots, baseline)

    def test_invalid_state_key_or_token_cap_refused(self):
        for mode in ('missing', 'extra', 'state', 'cap'):
            records, slots, baseline = fixture()
            if mode == 'missing': records[0]['states'].pop('edema')
            elif mode == 'extra': records[0]['states']['extra'] = 'unknown'
            elif mode == 'state': records[0]['states']['edema'] = 'absent'
            else: records[0]['token_limit_reached'] = True
            with self.assertRaises(ValueError):
                m.analyze(records, slots, baseline)

    def test_failed_prediction_cannot_carry_unknown_vector(self):
        records, slots, baseline = fixture(failed_arm='uniform_black')
        rec = next(r for r in records if r['contract_status'] != 'complete')
        rec['states'] = dict.fromkeys(m.cached.image_interface.FINDINGS, 'unknown')
        with self.assertRaises(ValueError):
            m.analyze(records, slots, baseline)

    def test_exact_baseline_image_hash_and_inventory_required(self):
        for mode in ('hash', 'missing', 'duplicate'):
            parts = fixture()
            baseline = parts[2]['records']
            if mode == 'hash': baseline[0]['cxr_sha256'] = 'f'*64
            elif mode == 'missing': baseline.pop()
            else: baseline.append(copy.deepcopy(baseline[0]))
            with self.assertRaises(ValueError):
                m.analyze(*parts)

    def test_baseline_must_keep_same_blind_frozen_contract(self):
        for key, value in (('image_only', False), ('frozen', False),
                ('model_received_ehr_reports_scores_or_candidate_ids', True)):
            parts = fixture()
            parts[2][key] = value
            with self.assertRaises(ValueError):
                m.analyze(*parts)

    def test_clinical_truth_and_regeneration_remain_unavailable(self):
        summary, _ = m.analyze(*fixture(control_state='positive'))
        for field in ('primary_metric_eligible', 'selection_changed', 'regeneration_authorized'):
            self.assertFalse(summary[field])
        self.assertEqual(summary['clinically_resolved_requests'], 0)
        self.assertTrue(summary['verifier_controls_never_used_for_generation'])
        records, slots, baseline = fixture()
        records[0]['independent_clinical_validation'] = True
        with self.assertRaises(ValueError):
            m.analyze(records, slots, baseline)

    def test_same_unchanged_model_prompt_with_image_only_no_arm_answer_metadata(self):
        sentinel = object()
        messages = m.cached.image_interface.request_messages('image', image=sentinel)
        content = messages[0]['content']
        self.assertIs(content[0]['image'], sentinel)
        prompt = m.cached.image_interface.IMAGE_PROMPT.format(findings=', '.join(m.cached.image_interface.FINDINGS))
        self.assertEqual(content[1]['text'], prompt)
        for word in ('uniform_black', 'uniform_white', 'invented_image', 'expected_answer', 'report_sha256'):
            self.assertNotIn(word, content[1]['text'])

    def test_prepare_hashes_baseline_without_parsing_states_or_opening_pixels(self):
        original = {k: None for k in ('image_slots', 'fixed_ehr_cases', 'model_path', 'model_file_sha256',
            'model_file_stats', 'finding_order', 'prompt_version', 'image_prompt_sha256', 'min_pixels',
            'max_pixels', 'max_new_tokens', 'model_retries', 'seed', 'do_sample', 'min_vram_gib')}
        original['image_inputs'] = [{'cxr_candidate_id': str(i), 'cxr_sha256': 'a'*64} for i in range(6)]
        manifest = {'new_model_calls': 6, 'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'artifacts': {'predictions.json': {'sha256': 'a'*64}}}
        def sha(path):
            if path == m.SOURCE_PLAN/'manifest.json': return m.SOURCE_PLAN_SHA
            if path == m.BASELINE/'manifest.json': return m.BASELINE_SHA
            return 'a'*64
        with patch.object(m, 'sha256_file', side_effect=sha), \
                patch.object(m.cached, 'load_plan', return_value=(original, {})), \
                patch.object(m.cached, 'bounded_json', return_value=manifest) as read, \
                patch.object(m.Path, 'is_file', return_value=True), \
                patch.object(m.Path, 'stat', return_value=SimpleNamespace(st_size=128)):
            plan, _ = m.prepare()
            read.assert_called_once_with(m.BASELINE/'manifest.json')
            self.assertFalse(plan['cached_prediction_states_parsed_in_prepare'])
            self.assertFalse(plan['image_bytes_or_pixels_opened_in_prepare'])
            self.assertEqual(plan['logical_slot_count'], 18)

    def test_approval_guard_precedes_plan_pixels_and_torch_import(self):
        with patch.object(m.cached, 'require_slurm', side_effect=RuntimeError), \
                patch.object(m, 'load_plan') as load, patch.object(m, 'build_frames') as pixels:
            with self.assertRaises(RuntimeError):
                m.run(Path('/invented'), Path('/invented'), approved=False)
            load.assert_not_called()
            pixels.assert_not_called()

    def test_execute_guard_and_existing_run_reject_before_loading_inputs(self):
        args = SimpleNamespace(mode='prepare', allow_image_controls=False, output_root=Path('/invented'), run_id='invented')
        with patch.object(m.cached, 'require_slurm', side_effect=RuntimeError), \
                patch.object(m, 'new_atomic_run') as new:
            with self.assertRaises(RuntimeError): m.execute(args)
            new.assert_not_called()
        with patch.object(m.cached, 'require_slurm'), patch.object(m, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(m, 'prepare') as prepare:
            with self.assertRaises(FileExistsError): m.execute(args)
            prepare.assert_not_called()

    def test_failure_only_discards_created_temp(self):
        args = SimpleNamespace(mode='prepare', allow_image_controls=False, output_root=Path('/invented'), run_id='invented')
        with patch.object(m.cached, 'require_slurm'), \
                patch.object(m, 'new_atomic_run', return_value=(Path('/invented/temp'), Path('/invented/target'))), \
                patch.object(m, 'prepare', side_effect=ValueError), patch.object(m, 'discard_atomic_run') as discard:
            with self.assertRaises(ValueError): m.execute(args)
            discard.assert_called_once_with(Path('/invented/temp'))

    def test_deterministic_analysis_and_all_inputs_unchanged(self):
        parts = fixture()
        before = copy.deepcopy(parts)
        self.assertEqual(m.analyze(*parts), m.analyze(*parts))
        self.assertEqual(parts, before)


if __name__ == '__main__':
    unittest.main()
