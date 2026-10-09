"""Frozen DICOM presentation policy; callers supply invented or approved pixels.

This module never opens a patient file, calls a model or displays an image.
No fitted percentiles, histogram equalization, per-image contrast search or
anatomical edits. Missing/ambiguous presentation metadata fails closed.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
HEADER_DEPENDENCIES = WORKSPACE / '.tmp/ricord_header_dependencies_12666569_001'
PIXEL_DEPENDENCIES = WORKSPACE / '.tmp/ricord_pixel_dependencies_12666569_001'
POLICY = {
    'version': 'ricord-dicom-presentation-v1',
    'decoder': 'pylibjpeg_for_jpeg_lossless_sv1_native_for_explicit_vr_little_endian',
    'modality': 'pydicom_apply_modality_lut',
    'voi': 'first_voi_lut_preferred_else_first_window_pair',
    'output_scaling': 'metadata_defined_range_not_observed_image_minmax',
    'polarity': 'presentation_lut_shape_once_else_photometric_once',
    'ambiguous_presentation': 'reject_without_substitution',
    'padding': 'explicit_padding_mask_to_black_no_crop',
    'quantization': 'round_to_uint8_0_255',
    'post_png': 'unchanged_FrozenXRVRuntime_PIL_L_normalize255_centercrop_resize224',
    'percentile_fitting': False, 'enhancement': False,
}


def dependencies():
    sys.path[:0] = [str(HEADER_DEPENDENCIES), str(PIXEL_DEPENDENCIES)]
    import numpy as np
    import pydicom
    import importlib.metadata as metadata
    if pydicom.__version__ != '3.0.2' or not Path(pydicom.__file__).resolve().is_relative_to(HEADER_DEPENDENCIES):
        raise ValueError('pydicom_dependency_pin_mismatch')
    for package, version in (('pylibjpeg', '2.1.0'), ('pylibjpeg-libjpeg', '2.4.0')):
        if metadata.version(package) != version:
            raise ValueError('jpeg_dependency_version_mismatch')
        distribution = metadata.distribution(package)
        if not Path(distribution.locate_file('')).resolve().is_relative_to(PIXEL_DEPENDENCIES):
            raise ValueError('jpeg_dependency_location_mismatch')
    from pydicom.pixels import get_decoder
    from pydicom.uid import JPEGLosslessSV1
    if 'pylibjpeg' not in get_decoder(JPEGLosslessSV1).available_plugins:
        raise ValueError('jpeg_lossless_decoder_unavailable')
    return np, pydicom


def first_number(ds, field):
    element = ds.get(field)
    if element is None:
        raise ValueError('required_display_number_missing')
    if isinstance(element, (tuple, list)) or hasattr(element, '__len__') and not isinstance(element, (str, bytes)):
        if len(element) == 0:
            raise ValueError('empty_display_number')
        element = element[0]
    value = float(element)
    if not math.isfinite(value):
        raise ValueError('nonfinite_display_number')
    return value


def presentation_contract(ds):
    photo = ds.get('PhotometricInterpretation')
    if photo not in ('MONOCHROME1', 'MONOCHROME2') or ds.get('Modality') not in ('CR', 'DX'):
        raise ValueError('unsupported_display_modality')
    if int(ds.get('SamplesPerPixel', 0)) != 1 or int(ds.get('NumberOfFrames', 1)) != 1:
        raise ValueError('single_frame_monochrome_required')
    if int(ds.get('Rows', 0)) <= 0 or int(ds.get('Columns', 0)) <= 0:
        raise ValueError('invalid_display_dimensions')
    syntax = str(ds.file_meta.get('TransferSyntaxUID', ''))
    if syntax not in ('1.2.840.10008.1.2.4.70', '1.2.840.10008.1.2.1'):
        raise ValueError('unapproved_transfer_syntax')
    if ds.get('PresentationLUTSequence'):
        raise ValueError('presentation_lut_sequence_not_in_frozen_policy')
    expected_shape = 'INVERSE' if photo == 'MONOCHROME1' else 'IDENTITY'
    shape = ds.get('PresentationLUTShape')
    if shape is not None and str(shape).strip().upper() != expected_shape:
        raise ValueError('photometric_presentation_shape_conflict')
    if ('RescaleSlope' in ds) != ('RescaleIntercept' in ds):
        raise ValueError('incomplete_modality_rescale')
    if 'RescaleSlope' in ds:
        if first_number(ds, 'RescaleSlope') <= 0:
            raise ValueError('nonpositive_modality_slope')
        first_number(ds, 'RescaleIntercept')
    bits, allocated, representation = int(ds.get('BitsStored', 0)), int(ds.get('BitsAllocated', 0)), int(ds.get('PixelRepresentation', -1))
    if allocated not in (8, 16) or not 1 <= bits <= allocated or representation not in (0, 1):
        raise ValueError('invalid_display_pixel_storage')
    if ds.get('ModalityLUTSequence'):
        item = ds.ModalityLUTSequence[0]
        if int(item.LUTDescriptor[2]) not in (8, 16):
            raise ValueError('unsupported_modality_lut_depth')
        if 'RescaleSlope' in ds:
            raise ValueError('ambiguous_modality_transform')
    if ds.get('VOILUTSequence'):
        item = ds.VOILUTSequence[0]
        if len(item.LUTDescriptor) != 3 or not 8 <= int(item.LUTDescriptor[2]) <= 16:
            raise ValueError('invalid_voi_lut_descriptor')
        if 'LUTData' not in item:
            raise ValueError('missing_voi_lut_data')
        voi = 'first_voi_lut'
    else:
        if 'WindowCenter' not in ds or 'WindowWidth' not in ds:
            raise ValueError('missing_voi_no_minmax_fallback')
        if ds['WindowCenter'].VM != ds['WindowWidth'].VM:
            raise ValueError('unpaired_window_multiplicity')
        first_number(ds, 'WindowCenter')
        width = first_number(ds, 'WindowWidth')
        function = str(ds.get('VOILUTFunction', 'LINEAR')).upper()
        if function not in ('LINEAR', 'LINEAR_EXACT', 'SIGMOID') or width <= 0 or function == 'LINEAR' and width < 1:
            raise ValueError('invalid_window_function_or_width')
        voi = 'first_window_pair'
    return {'voi_transform': voi, 'polarity_source': 'presentation_shape' if shape is not None else 'photometric',
            'invert_once': expected_shape == 'INVERSE', 'pixel_padding_present': 'PixelPaddingValue' in ds,
            'decoder': 'pylibjpeg' if syntax == '1.2.840.10008.1.2.4.70' else 'native',
            'metadata_contract_passed': True}


def window_output_range(ds):
    """Match apply_windowing's metadata-defined output range, never image extrema."""
    if ds.get('ModalityLUTSequence'):
        low, high = 0.0, float(2 ** int(ds.ModalityLUTSequence[0].LUTDescriptor[2]) - 1)
    elif int(ds.PixelRepresentation) == 0:
        low, high = 0.0, float(2 ** int(ds.BitsStored) - 1)
    else:
        low = float(-(2 ** (int(ds.BitsStored) - 1)))
        high = float(2 ** (int(ds.BitsStored) - 1) - 1)
    if 'RescaleSlope' in ds:
        slope, intercept = first_number(ds, 'RescaleSlope'), first_number(ds, 'RescaleIntercept')
        low, high = low * slope + intercept, high * slope + intercept
    return low, high


def render_array(raw, ds):
    np, pd = dependencies()
    contract = presentation_contract(ds)
    if raw.shape != (int(ds.Rows), int(ds.Columns)) or raw.dtype.kind not in ('i', 'u'):
        raise ValueError('decoded_array_shape_or_type_mismatch')
    if not np.isfinite(raw).all():
        raise ValueError('decoded_nonfinite_pixels')
    stored = int(ds.BitsStored)
    low_raw, high_raw = ((0, 2 ** stored - 1) if int(ds.PixelRepresentation) == 0 else
                         (-(2 ** (stored - 1)), 2 ** (stored - 1) - 1))
    if int(raw.min()) < low_raw or int(raw.max()) > high_raw:
        raise ValueError('decoded_pixel_storage_range_mismatch')
    padding = np.zeros(raw.shape, dtype=bool)
    if 'PixelPaddingValue' in ds:
        pad = int(ds.PixelPaddingValue)
        end = int(ds.get('PixelPaddingRangeLimit', pad))
        padding = (raw >= min(pad, end)) & (raw <= max(pad, end))
    if padding.all():
        raise ValueError('all_padding_image')
    modality = pd.pixels.apply_modality_lut(raw, ds)
    if contract['voi_transform'] == 'first_voi_lut':
        # Official VOI LUT processing expects integral modality-domain inputs.
        # Integer-valued rescale results can be safely cast, nonintegral ones
        # are not rounded into a different clinical image.
        if modality.dtype.kind == 'f':
            if not np.isfinite(modality).all() or not np.array_equal(modality, np.rint(modality)):
                raise ValueError('nonintegral_voi_lut_input')
            modality = modality.astype(np.int64)
        voi = pd.pixels.apply_voi_lut(modality, ds, index=0, prefer_lut=True)
        low, high = 0.0, float(2 ** int(ds.VOILUTSequence[0].LUTDescriptor[2]) - 1)
    else:
        with np.errstate(over='ignore'):
            voi = pd.pixels.apply_windowing(modality, ds, index=0)
        low, high = window_output_range(ds)
    if not math.isfinite(high - low) or high <= low or not np.isfinite(voi).all():
        raise ValueError('invalid_display_output_range')
    display = np.clip((voi.astype(np.float64) - low) / (high - low), 0, 1)
    # Presentation shape already accounts for MONOCHROME1. Apply ONCE, not
    # shape plus a second photometric inversion. Do not use intensity sign.
    if contract['invert_once']:
        display = 1.0 - display
    display[padding] = 0.0
    quantized = np.rint(display * 255).astype(np.uint8)
    if int(quantized[~padding].max()) == int(quantized[~padding].min()):
        raise ValueError('constant_display_image')
    return quantized, {**contract, 'display_policy_version': POLICY['version'],
                       'observed_minmax_used_for_scaling': False, 'output_bits': 8,
                       'padding_pixels': int(padding.sum()), 'display_transform_clinically_verified': False}
