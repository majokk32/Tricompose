"""Invented buffers, mock PNG decoder and metadata; no patient inputs/models."""
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
m = load('mechanical_guard_fixture', ROOT/'tools/image_validity_guard.py')
w = load('mechanical_guard_worker_fixture', ROOT/'tools/evaluate_image_validity_guard.py')
PAYLOAD = b'\x89PNG\r\n\x1a\nwholly_invented_buffer_not_a_patient_image'
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeImage:
    def __init__(self, mode='RGB', extrema=((0, 100), (0, 100), (0, 100)), size=(64, 64), frames=1):
        self.mode, self.format, self.n_frames = mode, 'PNG', frames
        self.width, self.height = size
        self.size, self.extrema = size, extrema
        self.load_calls = 0
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def load(self): self.load_calls += 1
    def getextrema(self): return self.extrema
    def convert(self, mode):
        return FakeImage(mode, self.extrema if self.mode == 'RGB' else (self.extrema,)*3, self.size)
    def copy(self): return self
    def tobytes(self):
        uniform = all(lo == hi for lo, hi in self.extrema)
        return b'\x00'*(self.width*self.height*3) if uniform else b'\x00\x01'*(self.width*self.height*3//2)


class FakeDecoder:
    class DecompressionBombError(Exception): pass
    class DecompressionBombWarning(Warning): pass
    def __init__(self, image=None, error=None): self.image, self.error = image or FakeImage(), error
    def open(self, source):
        if self.error: raise self.error
        return self.image


def inspected(uniform=False):
    image = FakeImage(extrema=((0, 0),)*3 if uniform else ((0, 100),)*3)
    return m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(image))


def cached(guard, state='negative', failed=False):
    return {'observation_id': guard['normalized_pixel_sha256'],
        'contract_status': 'failed_unavailable' if failed else 'complete',
        'states': None if failed else dict.fromkeys(m.HEADS, state), 'token_limit_reached': False}


class ImageValidityGuardTests(unittest.TestCase):
    def test_nonuniform_pass_is_not_clinical_acceptance(self):
        result, image = inspected()
        self.assertEqual(result['status'], 'basic_pass_not_clinical')
        self.assertIsNotNone(image)
        self.assertTrue(result['basic_comparison_permitted'])
        for field in ('clinical_acceptance', 'primary_metric_eligible', 'regeneration_authorized',
                'model_execution_authorized_by_guard'):
            self.assertFalse(result[field])
        self.assertIsNone(result['clinical_accuracy'])

    def test_uniform_gray_black_white_and_colored_frames_blocked(self):
        for color in ((0, 0, 0), (255, 255, 255), (128, 128, 128), (255, 0, 17)):
            image = FakeImage(extrema=tuple((v, v) for v in color))
            result, _ = m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(image))
            self.assertEqual(result['status'], 'artifact_invalid')
            self.assertEqual(result['reason'], 'spatially_uniform')
            self.assertFalse(result['basic_comparison_permitted'])

    def test_single_varying_channel_suffices_for_mechanical_pass_not_anatomy(self):
        result, _ = m.inspect_bytes(PAYLOAD, SHA,
            FakeDecoder(FakeImage(extrema=((1, 1), (2, 3), (4, 4)))))
        self.assertEqual(result['status'], 'basic_pass_not_clinical')

    def test_native_grayscale_supported_without_false_color_range(self):
        for extrema, status in (((8, 8), 'artifact_invalid'), ((8, 9), 'basic_pass_not_clinical')):
            result, image = m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(FakeImage(mode='L', extrema=extrema)))
            self.assertEqual(result['status'], status)
            self.assertEqual(image.mode, 'RGB')

    def test_unsupported_16bit_alpha_and_palette_not_converted_to_blank(self):
        for mode in ('I;16', 'I', 'F', 'RGBA', 'P', '1'):
            image = FakeImage(mode=mode)
            result, rgb = m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(image))
            self.assertEqual(result['status'], 'verification_unavailable')
            self.assertEqual(result['reason'], 'unsupported_native_mode')
            self.assertIsNone(rgb)
            self.assertEqual(image.load_calls, 0)

    def test_dimensions_bound_before_decoding(self):
        for size in ((63, 64), (64, 4097), (4097, 64), (0, 64)):
            image = FakeImage(size=size)
            result, _ = m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(image))
            self.assertEqual(result['reason'], 'bounded_dimensions_required')
            self.assertEqual(image.load_calls, 0)

    def test_animation_or_multiple_frames_blocked_before_decode(self):
        image = FakeImage(frames=2)
        result, _ = m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(image))
        self.assertEqual(result['reason'], 'unsupported_frame_count')
        self.assertEqual(image.load_calls, 0)

    def test_corrupt_hash_bound_png_has_decode_failure_not_negative_labels(self):
        result, image = m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(error=OSError('invented decode error')))
        self.assertEqual(result['status'], 'artifact_invalid')
        self.assertEqual(result['reason'], 'png_decode_failed')
        self.assertIsNone(image)
        self.assertNotIn('invented decode error', str(result))

    def test_decompression_bomb_is_resource_unavailable_not_clinically_invalid(self):
        result, _ = m.inspect_bytes(PAYLOAD, SHA, FakeDecoder(error=FakeDecoder.DecompressionBombError()))
        self.assertEqual(result['status'], 'verification_unavailable')
        self.assertEqual(result['reason'], 'bounded_decode_required')

    def test_wrong_hash_refused_before_decoder(self):
        decoder = FakeDecoder()
        decoder.open = Mock()
        result, _ = m.inspect_bytes(PAYLOAD, 'f'*64, decoder)
        self.assertEqual(result['reason'], 'source_hash_changed')
        decoder.open.assert_not_called()

    def test_oversized_buffer_refused_before_decoder(self):
        decoder = FakeDecoder()
        decoder.open = Mock()
        with patch.object(m, 'MAX_BYTES', 3):
            result, _ = m.inspect_bytes(PAYLOAD, SHA, decoder)
        self.assertEqual(result['reason'], 'bounded_file_bytes_required')
        decoder.open.assert_not_called()

    def test_non_png_marked_unsupported_not_decode_success(self):
        payload = b'invented_other_format'
        result, _ = m.inspect_bytes(payload, hashlib.sha256(payload).hexdigest(), FakeDecoder())
        self.assertEqual(result['reason'], 'unsupported_format')
        self.assertFalse(result['basic_comparison_permitted'])

    def test_missing_unreadable_and_oversized_files_unavailable(self):
        with patch.object(m, 'require_inside', return_value=Path('/invented')), \
                patch.object(m.Path, 'is_file', return_value=False):
            result, _ = m.inspect_png('/invented', SHA, Image=FakeDecoder())
            self.assertEqual(result['reason'], 'source_file_unavailable')
        with patch.object(m, 'require_inside', return_value=Path('/invented')), \
                patch.object(m.Path, 'is_file', return_value=True), \
                patch.object(m.Path, 'stat', return_value=SimpleNamespace(st_size=len(PAYLOAD))), \
                patch.object(m.Path, 'read_bytes', side_effect=PermissionError):
            result, _ = m.inspect_png('/invented', SHA, Image=FakeDecoder())
            self.assertEqual(result['reason'], 'source_file_unreadable')
        with patch.object(m, 'require_inside', return_value=Path('/invented')), \
                patch.object(m.Path, 'is_file', return_value=True), \
                patch.object(m.Path, 'stat', return_value=SimpleNamespace(st_size=m.MAX_BYTES+1)), \
                patch.object(m.Path, 'read_bytes') as read:
            result, _ = m.inspect_png('/invented', SHA, Image=FakeDecoder())
            self.assertEqual(result['reason'], 'bounded_file_bytes_required')
            read.assert_not_called()

    def test_path_outside_protected_boundary_never_opened(self):
        with patch.object(m, 'require_inside', side_effect=ValueError), patch.object(m.Path, 'read_bytes') as read:
            with self.assertRaises(ValueError): m.inspect_png('/invented/outside', SHA)
            read.assert_not_called()

    def test_cache_view_preserves_raw_unknown_uncertain_positive_and_negative(self):
        current, _ = inspected()
        for state in m.STATES:
            raw = cached(current, state)
            before = copy.deepcopy(raw)
            view = m.cached_view(current, raw)
            self.assertEqual(raw, before)
            self.assertEqual(view['raw_verifier_result'], before)
            self.assertEqual(view['guarded_states'], before['states'])

    def test_invalid_view_withholds_entire_vector_null_not_all_unknown(self):
        current, _ = inspected(uniform=True)
        raw = cached(current, 'negative')
        before = copy.deepcopy(raw)
        view = m.cached_view(current, raw)
        self.assertEqual(view['raw_verifier_result'], before)
        self.assertIsNone(view['guarded_states'])
        self.assertFalse(view['image_evidence_available'])
        self.assertEqual(raw, before)

    def test_failed_cached_contract_not_successful_unknown(self):
        current, _ = inspected()
        view = m.cached_view(current, cached(current, failed=True))
        self.assertIsNone(view['guarded_states'])
        self.assertFalse(view['image_evidence_available'])

    def test_changed_receipt_truth_or_decision_refused(self):
        current, _ = inspected()
        for key, value in (('clinical_acceptance', True), ('regeneration_authorized', True),
                ('basic_comparison_permitted', False), ('status', 'artifact_invalid')):
            bad = {**current, key: value}
            with self.assertRaises(ValueError): m.cached_view(bad, cached(current))

    def test_cached_pixel_identity_and_complete_head_contract_required(self):
        current, _ = inspected()
        for mode in ('pixel', 'missing', 'extra', 'state', 'cap'):
            raw = cached(current)
            if mode == 'pixel': raw['observation_id'] = 'f'*64
            elif mode == 'missing': raw['states'].pop('edema')
            elif mode == 'extra': raw['states']['device'] = 'unknown'
            elif mode == 'state': raw['states']['edema'] = 'absent'
            else: raw['token_limit_reached'] = True
            with self.assertRaises(ValueError): m.cached_view(current, raw)

    def test_guarded_hook_never_calls_callback_on_invalid_or_unavailable(self):
        for current, image in (inspected(uniform=True),
                (m.receipt(SHA, 'verification_unavailable', 'source_file_unavailable'), None)):
            cb = Mock()
            with patch.dict(m.os.environ, {'SLURM_JOB_ID': '1234'}), \
                    patch.object(m.Path, 'read_text', return_value='/job_1234/'), \
                    patch.object(m, 'inspect_png', return_value=(current, image)):
                result = m.guarded_invoke('/invented', SHA, cb)
            cb.assert_not_called()
            self.assertFalse(result['callback_invoked'])
            self.assertIsNone(result['callback_result'])

    def test_guarded_hook_passes_exact_checked_image_and_preserves_callback_result(self):
        current, image = inspected()
        value = {'invented': 'result'}
        cb = Mock(return_value=value)
        with patch.dict(m.os.environ, {'SLURM_JOB_ID': '1234'}), \
                patch.object(m.Path, 'read_text', return_value='/job_1234/'), \
                patch.object(m, 'inspect_png', return_value=(current, image)):
            result = m.guarded_invoke('/invented', SHA, cb)
        cb.assert_called_once_with(image)
        self.assertIs(result['callback_result'], value)
        self.assertIsNone(result['actual_model_calls'])
        self.assertFalse(result['model_execution_authorized_by_guard'])

    def test_callback_guard_requires_actual_slurm_before_input_inspection(self):
        with patch.dict(m.os.environ, {'SLURM_JOB_ID': ''}), patch.object(m, 'inspect_png') as inspect:
            with self.assertRaises(RuntimeError): m.guarded_invoke('/invented', SHA, Mock())
            inspect.assert_not_called()

    def test_callback_guard_refuses_changed_in_memory_pixels(self):
        current, image = inspected()
        image.tobytes = lambda: b'\xff'*(64*64*3)
        cb = Mock()
        with patch.dict(m.os.environ, {'SLURM_JOB_ID': '1234'}), \
                patch.object(m.Path, 'read_text', return_value='/job_1234/'), \
                patch.object(m, 'inspect_png', return_value=(current, image)):
            with self.assertRaises(ValueError): m.guarded_invoke('/invented', SHA, cb)
        cb.assert_not_called()

    def test_metadata_alone_does_not_authorize_callback(self):
        value = m.metadata_guard(SHA, width=64, height=64, mode='RGB', format_name='PNG', frames=1,
            extrema=((0, 1),)*3)
        with self.assertRaises(ValueError): m.validate_receipt(value)

    def test_native_extrema_must_be_exact_int_and_well_formed(self):
        for extrema in (((0.0, 1),)*3, ((0, 256),)*3, ((2, 1),)*3, ((0, 1),), ((True, 1),)*3):
            with self.assertRaises(ValueError):
                m.metadata_guard(SHA, width=64, height=64, mode='RGB', format_name='PNG', frames=1, extrema=extrema)

    def test_candidate_join_is_lossless_exact_identity_not_hash_only(self):
        current, _ = inspected()
        rows = [{'triple_candidate_id': 'a', 'case_id': 'invented_case', 'cxr_candidate_id': 'image0',
            'cxr_sha256': SHA, 'ehr_sha256': 'b'*64, 'ehr_facts_sha256': 'c'*64, 'old_score': '0.21'},
            {'triple_candidate_id': 'b', 'case_id': 'invented_case', 'cxr_candidate_id': 'unchecked_same_hash',
            'cxr_sha256': SHA, 'ehr_sha256': 'b'*64, 'ehr_facts_sha256': 'c'*64, 'old_score': ''}]
        original = {'image_inputs': [{k: rows[0][k] for k in ('case_id', 'cxr_candidate_id',
            'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')}]}
        slots = [{'arm': 'original', 'cxr_candidate_id': 'image0', 'observation_id': 'obs'}]
        before = copy.deepcopy(rows)
        annotated = w.attach_candidates(rows, original, slots, {'obs': current})
        self.assertEqual(rows, before)
        for old, new in zip(rows, annotated):self.assertEqual({k:new[k] for k in old}, old)
        self.assertEqual(annotated[1]['imageguard_status'], 'not_checked')
        self.assertIsNone(annotated[1]['imageguard_basic_comparison_permitted'])

    def test_candidate_join_cannot_change_image_hash_or_fixed_ehr(self):
        current, _ = inspected()
        row = {'case_id': 'case', 'cxr_candidate_id': 'img', 'cxr_sha256': SHA,
            'ehr_sha256': 'b'*64, 'ehr_facts_sha256': 'c'*64}
        original = {'image_inputs': [copy.deepcopy(row)]}
        slots = [{'arm': 'original', 'cxr_candidate_id': 'img', 'observation_id': 'obs'}]
        for key in ('cxr_sha256', 'ehr_sha256', 'case_id'):
            bad = {**row, key: 'changed'}
            with self.assertRaises(ValueError): w.attach_candidates([bad], original, slots, {'obs':current})

    def test_repeat_candidate_attachment_refused(self):
        with self.assertRaises(ValueError):
            w.attach_candidates([{'imageguard_status': 'old', 'cxr_candidate_id':'x'}], {'image_inputs':[]}, [], {})

    def test_worker_actual_slurm_and_overwrite_guards_precede_loading(self):
        with patch.object(w.cached, 'require_slurm', side_effect=RuntimeError), patch.object(w, 'new_atomic_run') as new:
            with self.assertRaises(RuntimeError): w.execute('/invented', 'invented')
            new.assert_not_called()
        with patch.object(w.cached, 'require_slurm'), patch.object(w, 'new_atomic_run', side_effect=FileExistsError), \
                patch.object(w, 'load_inputs') as load:
            with self.assertRaises(FileExistsError): w.execute('/invented', 'invented')
            load.assert_not_called()

    def test_worker_failure_discards_only_created_temp(self):
        with patch.object(w.cached, 'require_slurm'), \
                patch.object(w, 'new_atomic_run', return_value=(Path('/invented/temp'),Path('/invented/target'))), \
                patch.object(w, 'load_inputs', side_effect=ValueError), patch.object(w, 'discard_atomic_run') as discard:
            with self.assertRaises(ValueError): w.execute('/invented', 'invented')
            discard.assert_called_once_with(Path('/invented/temp'))

    def test_deterministic_receipts_and_no_near_uniform_threshold(self):
        one, _ = inspected()
        two, _ = inspected()
        self.assertEqual(one, two)
        value = m.metadata_guard(SHA, width=64, height=64, mode='RGB', format_name='PNG', frames=1,
            extrema=((0, 1), (0, 0), (0, 0)))
        self.assertEqual(value['status'], 'basic_pass_not_clinical')


if __name__ == '__main__':
    unittest.main()
