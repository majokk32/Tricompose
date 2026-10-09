"""Pure image-path metadata normalization, not patient/image inspection.

Only exact source suffixes may propose a local match. No basename-only image
retrieval, silent extension change, view substitution, or clinical selection.
Raw source/path strings are transient worker keys, never public metadata.
"""
from pathlib import PurePosixPath
import re

VERSION = 'tricompose-radeval-image-linkage-v1'


def author_path(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        return {'status': 'invalid_path_metadata', 'family': 'unknown', 'relative_suffix': None}
    text = value.strip()
    if any(c in text for c in ('\x00', '\n', '\r', '\\')):
        return {'status': 'unsupported_path_encoding', 'family': 'unknown', 'relative_suffix': None}
    parts = PurePosixPath(text).parts
    if '..' in parts or any(p.startswith('~') for p in parts):
        return {'status': 'unsafe_path_metadata', 'family': 'unknown', 'relative_suffix': None}
    suffix = PurePosixPath(text).suffix.lower()
    if suffix not in ('.jpg', '.jpeg', '.png'):
        return {'status': 'unsupported_image_extension', 'family': 'unknown', 'relative_suffix': None}
    # Explicit MIMIC patient, study and image components, not a random image ID.
    for i in range(len(parts) - 3):
        prefix, subject, study, filename = parts[i:i+4]
        if (re.fullmatch(r'p[0-9]{2}', prefix) and re.fullmatch(r'p[0-9]{8}', subject)
                and re.fullmatch(r's[0-9]{8}', study) and i+4 == len(parts)
                and re.fullmatch(r'[A-Za-z0-9_-]+\.(?:jpg|jpeg|png)', filename, re.I)
                and subject[1:3] == prefix[1:]):
            return {'status': 'recognized_exact_suffix', 'family': 'mimic',
                    'relative_suffix': '/'.join((prefix, subject, study, filename))}
    for i in range(len(parts) - 2):
        subject, study, filename = parts[i:i+3]
        if (re.fullmatch(r'patient[0-9]+', subject) and re.fullmatch(r'study[0-9]+', study)
                and i+3 == len(parts)):
            return {'status': 'recognized_other_dataset_requires_separate_access', 'family': 'chexpert',
                    'relative_suffix': '/'.join((subject, study, filename))}
    return {'status': 'unrecognized_source_layout', 'family': 'unknown', 'relative_suffix': None}


def public_record(source_id, parsed, local_status, *, candidate_count=0):
    if not re.fullmatch(r'source_[0-9]{4}', source_id):
        raise ValueError('opaque_source_id_required')
    return {'source_id': source_id, 'source_family': parsed['family'],
            'author_path_status': parsed['status'], 'local_image_status': local_status,
            'candidate_file_count': candidate_count, 'pixel_values_decoded': False,
            'image_bytes_read': False, 'source_path_exported': False,
            'image_report_alignment_clinically_verified': False}
