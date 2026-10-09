"""Fail-closed mechanical PNG guard, not anatomy/clinical validation."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'))
from contracts import PROTECTED_ROOT, require_inside

SCHEMA = 'tricompose-mechanical-image-validity-guard-v1'
RULE_VERSION = 'bounded_native_L_RGB_exact_spatial_uniformity_v1'
MAX_BYTES = 64*1024*1024
MIN_SIDE, MAX_SIDE = 64, 4096
HEADS = ('atelectasis', 'cardiomegaly', 'consolidation', 'edema',
    'lung_opacity', 'pleural_effusion', 'pneumonia', 'pneumothorax')
STATES = frozenset(('positive', 'negative', 'uncertain', 'unknown'))
HASH = re.compile(r'[0-9a-f]{64}\Z')


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def receipt(artifact_sha256, status, reason, *, evidence=None, pixel_sha256=None):
    if not isinstance(artifact_sha256, str) or not HASH.fullmatch(artifact_sha256):
        raise ValueError('encoded_artifact_sha256_required')
    if status not in ('artifact_invalid', 'verification_unavailable', 'basic_pass_not_clinical'):
        raise ValueError('known_guard_status_required')
    if pixel_sha256 is not None and not HASH.fullmatch(pixel_sha256):
        raise ValueError('normalized_pixel_sha256_required')
    payload = {'schema_version': SCHEMA, 'rule_version': RULE_VERSION,
        'artifact_sha256': artifact_sha256, 'status': status, 'reason': reason,
        'native_image_metadata': evidence, 'normalized_pixel_sha256': pixel_sha256,
        'basic_comparison_permitted': status == 'basic_pass_not_clinical',
        'clinical_accuracy': None, 'clinical_acceptance': False,
        'primary_metric_eligible': False, 'regeneration_authorized': False,
        'model_execution_authorized_by_guard': False}
    return {**payload, 'receipt_sha256': digest(payload)}


def metadata_guard(artifact_sha256, *, width, height, mode, format_name, frames, extrema=None):
    evidence = {'width': width, 'height': height, 'mode': mode, 'format': format_name,
        'frames': frames, 'channel_extrema': None}
    if (type(width) is not int or type(height) is not int
            or not MIN_SIDE <= width <= MAX_SIDE or not MIN_SIDE <= height <= MAX_SIDE):
        return receipt(artifact_sha256, 'verification_unavailable', 'bounded_dimensions_required', evidence=evidence)
    if format_name != 'PNG':
        return receipt(artifact_sha256, 'verification_unavailable', 'unsupported_format', evidence=evidence)
    if type(frames) is not int or frames != 1:
        return receipt(artifact_sha256, 'verification_unavailable', 'unsupported_frame_count', evidence=evidence)
    if mode not in ('L', 'RGB'):
        return receipt(artifact_sha256, 'verification_unavailable', 'unsupported_native_mode', evidence=evidence)
    # Bounds can be checked before decoding; no extrema yet is NOT a pass.
    if extrema is None:
        return receipt(artifact_sha256, 'verification_unavailable', 'pixels_not_decoded', evidence=evidence)
    bands = [extrema] if mode == 'L' else extrema
    if (not isinstance(bands, (list, tuple)) or len(bands) != (1 if mode == 'L' else 3)
            or any(not isinstance(band, (list, tuple)) or len(band) != 2
                or any(type(v) is not int for v in band) or not 0 <= band[0] <= band[1] <= 255 for band in bands)):
        raise ValueError('exact_native_uint8_channel_extrema_required')
    evidence['channel_extrema'] = [list(band) for band in bands]
    uniform = all(lo == hi for lo, hi in bands)
    return receipt(artifact_sha256, 'artifact_invalid' if uniform else 'basic_pass_not_clinical',
        'spatially_uniform' if uniform else 'decoded_nonuniform_no_clinical_claim', evidence=evidence)


def normalized_pixel_sha(image):
    if image.mode != 'RGB':
        raise ValueError('normalized_RGB_required')
    pixels = image.tobytes()
    if len(pixels) != image.width*image.height*3:
        raise ValueError('complete_RGB_buffer_required')
    h = hashlib.sha256(json.dumps(['RGB', image.width, image.height], separators=(',', ':')).encode()+b'\0')
    h.update(pixels)
    return h.hexdigest()


def inspect_bytes(payload, artifact_sha256, Image):
    """Bounded buffer decoder, injectable for wholly invented unit tests."""
    if len(payload) > MAX_BYTES:
        return receipt(artifact_sha256, 'verification_unavailable', 'bounded_file_bytes_required'), None
    if hashlib.sha256(payload).hexdigest() != artifact_sha256:
        return receipt(artifact_sha256, 'verification_unavailable', 'source_hash_changed'), None
    if not payload.startswith(b'\x89PNG\r\n\x1a\n'):
        return receipt(artifact_sha256, 'verification_unavailable', 'unsupported_format'), None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                kwargs = {'width': image.width, 'height': image.height, 'mode': image.mode,
                    'format_name': image.format, 'frames': getattr(image, 'n_frames', 1)}
                preflight = metadata_guard(artifact_sha256, **kwargs)
                if preflight['reason'] != 'pixels_not_decoded':
                    return preflight, None
                image.load()
                guard = metadata_guard(artifact_sha256, **kwargs, extrema=image.getextrema())
                rgb = image.convert('RGB').copy()
                guard = receipt(artifact_sha256, guard['status'], guard['reason'],
                    evidence=guard['native_image_metadata'], pixel_sha256=normalized_pixel_sha(rgb))
                return guard, rgb
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return receipt(artifact_sha256, 'verification_unavailable', 'bounded_decode_required'), None
    except (OSError, SyntaxError, ValueError):
        return receipt(artifact_sha256, 'artifact_invalid', 'png_decode_failed'), None


def inspect_png(path, expected_sha256, *, Image=None):
    source = require_inside(path, PROTECTED_ROOT, must_exist=False)
    try:
        if not source.is_file():
            return receipt(expected_sha256, 'verification_unavailable', 'source_file_unavailable'), None
        if source.stat().st_size > MAX_BYTES:
            return receipt(expected_sha256, 'verification_unavailable', 'bounded_file_bytes_required'), None
        payload = source.read_bytes()
    except OSError:
        return receipt(expected_sha256, 'verification_unavailable', 'source_file_unreadable'), None
    if Image is None:
        from PIL import Image
    return inspect_bytes(payload, expected_sha256, Image)


def validate_receipt(guard):
    expected = receipt(guard['artifact_sha256'], guard['status'], guard['reason'],
        evidence=guard['native_image_metadata'], pixel_sha256=guard['normalized_pixel_sha256'])
    if guard != expected:
        raise ValueError('unchanged_mechanical_guard_receipt_required')
    if guard['status'] == 'basic_pass_not_clinical' and guard['normalized_pixel_sha256'] is None:
        raise ValueError('normalized_image_binding_required')
    if guard['status'] in ('basic_pass_not_clinical', 'artifact_invalid') and guard['reason'] != 'png_decode_failed':
        native = guard['native_image_metadata']
        if native is None:
            raise ValueError('native_pixels_required_for_uniformity_decision')
        rebuilt = metadata_guard(guard['artifact_sha256'], width=native['width'], height=native['height'],
            mode=native['mode'], format_name=native['format'], frames=native['frames'],
            extrema=native['channel_extrema'][0] if native['mode'] == 'L' else native['channel_extrema'])
        if rebuilt['status'] != guard['status'] or rebuilt['reason'] != guard['reason']:
            raise ValueError('guard_decision_must_match_native_evidence')
    return guard


def cached_view(guard, raw):
    validate_receipt(guard)
    if raw['observation_id'] != guard['normalized_pixel_sha256']:
        raise ValueError('exact_guard_and_cached_input_required')
    complete = raw['contract_status'] == 'complete'
    if complete:
        if (not isinstance(raw['states'], dict) or set(raw['states']) != set(HEADS)
                or not set(raw['states'].values()) <= STATES or raw['token_limit_reached'] is not False):
            raise ValueError('complete_named_four_state_contract_required')
    elif raw['contract_status'] != 'failed_unavailable' or raw['states'] is not None:
        raise ValueError('unavailable_cached_states_must_be_null')
    allowed = guard['basic_comparison_permitted'] and complete
    return {'guard': guard, 'raw_verifier_result': raw,
        'guarded_states': raw['states'] if allowed else None,
        'image_evidence_available': allowed, 'clinical_accuracy': None,
        'primary_metric_eligible': False, 'regeneration_authorized': False}


def guarded_invoke(path, expected_sha256, callback, *, Image=None):
    """Optional hook; callback owns its separate Slurm/approval/privacy ledger."""
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('actual_slurm_allocation_required_before_callback')
    guard, image = inspect_png(path, expected_sha256, Image=Image)
    validate_receipt(guard)
    called = guard['basic_comparison_permitted']
    if called and (image is None or normalized_pixel_sha(image) != guard['normalized_pixel_sha256']):
        raise ValueError('callback_requires_exact_checked_in_memory_image')
    value = callback(image) if called else None
    return {'guard': guard, 'callback_invoked': called, 'callback_result': value,
        'actual_model_calls': None, 'clinical_accuracy': None, 'regeneration_authorized': False,
        'model_execution_authorized_by_guard': False}
