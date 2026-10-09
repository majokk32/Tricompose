#!/usr/bin/env python3
"""Freeze a 25+25 RICORD plan, then acquire <=50 DICOMs/1 GiB; no inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request
import warnings

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
REAL = WORKSPACE / 'TriCompose-v1.2/real_validation'
spec = importlib.util.spec_from_file_location('_ricord_group_audit', REAL / 'audit_ricord_reader_groups.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
annotation = audit.acquire
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       private_directory, require_inside, sha256_file,
                       write_private_json)

BASE = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/'
COLLECTION = 'MIDRC-RICORD-1C'
MAX_IMAGES, QUOTA, MAX_BYTES = 50, 25, 1024 ** 3
MAX_OBJECT = 32 * 1024 ** 2
PLAN_SCHEMA = 'tricompose-ricord-image-plan-v1'
SCHEMA = 'tricompose-ricord-image-acquisition-v1'
PROTOCOL = WORKSPACE / 'docs/ricord_image_acquisition_protocol.md'
TESTS = WORKSPACE / 'TriCompose-v1.2/tests/test_ricord_images.py'
DEPENDENCIES = WORKSPACE / '.tmp/ricord_header_dependencies_12666569_001'
DEFAULT_ROOT = annotation.PROTECTED / 'tricompose_v1_2/reference_datasets/ricord_1c'
UID_PATTERN = re.compile(r'[0-9]+(?:\.[0-9]+)+\Z')


class BoundError(ValueError):
    pass


class PublicClient:
    """All endpoint parameters are structured internally; never log a URL."""
    def __init__(self, initial_bytes=0):
        self.opener = annotation.public_opener()
        self.received = initial_bytes
        self.image_requests = 0

    @staticmethod
    def url(endpoint, params):
        allowed = {'getSeries': {'Collection'}, 'getSOPInstanceUIDs': {'SeriesInstanceUID'},
                   'getSingleImage': {'SeriesInstanceUID', 'SOPInstanceUID'}}
        if endpoint not in allowed or set(params) != allowed[endpoint]:
            raise ValueError('endpoint_parameters_rejected')
        for key, value in params.items():
            if key == 'Collection':
                if value != COLLECTION:
                    raise ValueError('collection_rejected')
            elif not isinstance(value, str) or len(value) > 64 or not UID_PATTERN.fullmatch(value):
                raise ValueError('source_key_format_rejected')
        return BASE + endpoint + '?' + urlencode(params)

    def get(self, endpoint, params, limit, destination=None, expected=None):
        url = self.url(endpoint, params)
        if destination is not None:
            if endpoint != 'getSingleImage' or self.image_requests >= MAX_IMAGES:
                raise BoundError('image_request_limit')
            self.image_requests += 1
        elif endpoint == 'getSingleImage':
            raise ValueError('image_destination_required')
        remaining = MAX_BYTES - self.received
        if remaining <= 0 or expected is not None and expected > min(limit, remaining):
            raise BoundError('byte_preflight_limit')
        count, chunks, started = 0, [], time.monotonic()
        request = Request(url, headers={'User-Agent': 'TriCompose-bounded-public-acquisition/1.0'})
        with self.opener.open(request, timeout=25) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError('response_rejected')
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise ValueError('content_encoding_rejected')
            length = response.headers.get('Content-Length')
            if length is not None and (int(length) > min(limit, remaining)
                    or expected is not None and int(length) != expected):
                raise BoundError('content_length_rejected')
            handle = None
            try:
                if destination is not None:
                    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
                    os.chmod(destination, 0o660)
                    handle = os.fdopen(fd, 'wb')
                while True:
                    # Read at most remaining budget, including a failed body.
                    room = min(limit - count, MAX_BYTES - self.received)
                    if room <= 0:
                        if expected == count or length is not None and int(length) == count:
                            break
                        raise BoundError('body_limit_reached_without_verified_end')
                    chunk = response.read(min(65536, room))
                    if not chunk:
                        break
                    count += len(chunk)
                    self.received += len(chunk)
                    if handle is not None:
                        handle.write(chunk)
                    else:
                        chunks.append(chunk)
                    if time.monotonic() - started > 180:
                        raise BoundError('response_time_limit')
            finally:
                if handle is not None:
                    handle.close()
        if count == 0 or length is not None and int(length) != count or expected is not None and count != expected:
            raise ValueError('response_size_mismatch')
        return count if destination is not None else b''.join(chunks)

    def metadata(self, endpoint, params, limit=1500000):
        return json.loads(self.get(endpoint, params, limit), object_pairs_hook=annotation.reject_duplicate_keys)


def source_pins():
    paths = [Path(__file__), TESTS, PROTOCOL, REAL / 'audit_ricord_reader_groups.py',
             REAL / 'acquire_ricord_annotations.py',
             WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
    return {str(p.relative_to(WORKSPACE)): sha256_file(p) for p in paths}


def verified_manifest(root, schema):
    root = require_inside(root, annotation.PROTECTED, must_exist=True)
    manifest = json.loads((root / 'manifest.json').read_bytes(), object_pairs_hook=annotation.reject_duplicate_keys)
    if manifest.get('schema_version') != schema + '-manifest':
        raise ValueError('manifest_schema_rejected')
    for name, digest in manifest.get('artifacts', {}).items():
        if sha256_file(require_inside(root / name, root, must_exist=True)) != digest:
            raise ValueError('artifact_hash_mismatch')
    for name, digest in manifest.get('sources', {}).items():
        if sha256_file(require_inside(WORKSPACE / name, WORKSPACE, must_exist=True)) != digest:
            raise ValueError('consumed_source_changed')
    return root, manifest


def derived_labels(payload, removed):
    """Internal keyed result only; aggregate consistency checked against sealed audit."""
    result = audit.audit_groups(payload, removed)
    complete = result['comprehensive_group_indices']
    definitions = {label['id']: (i, annotation.label_name(label.get('name')), label.get('scope'))
                   for i, group in enumerate(payload['labelGroups']) for label in group['labels']}
    values = defaultdict(set)
    for dataset in payload['datasets']:
        for row in dataset['annotations']:
            group, name, scope = definitions[row['labelId']]
            if group in complete and name in annotation.CLASSIFICATIONS and scope == 'STUDY' and row.get('data') is None:
                values[(row.get('StudyInstanceUID'), group)].add(name)
    labels, studies = {}, []
    for dataset in payload['datasets']:
        studies.extend(dataset['studies'])
    for index, study in enumerate(studies):
        key = study['StudyInstanceUID']
        if key in removed:
            continue
        observations = [values[(key, group)] for group in complete]
        if not all(len(v) == 1 for v in observations):
            continue
        names = {next(iter(v)) for v in observations}
        if len(names) != 1:
            continue
        name = next(iter(names))
        state = ('positive' if name in ('typical_appearance', 'indeterminate_appearance')
                 else 'negative' if name == 'negative_for_pneumonia' else None)
        if state:
            labels[key] = (index, state)
    if Counter(state for _, state in labels.values()) != Counter({
            'positive': result['derived_opacity_positive_studies'],
            'negative': result['derived_opacity_negative_studies']}):
        raise ValueError('derived_reference_count_mismatch')
    return studies, labels, result


def positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('positive_integer_required')
    return value


def select_cases(studies, labels, rows, removed, quota=QUOTA):
    if not isinstance(quota, int) or isinstance(quota, bool) or quota < 1 or 2 * quota > MAX_IMAGES:
        raise ValueError('quota_rejected')
    by_study, series_keys = defaultdict(list), set()
    for index, row in enumerate(rows):
        if row.get('Collection') != COLLECTION:
            raise ValueError('mixed_collection_inventory')
        if row.get('SeriesInstanceUID') in series_keys:
            raise ValueError('duplicate_series_inventory')
        series_keys.add(row.get('SeriesInstanceUID'))
        positive_int(row.get('ImageCount'))
        positive_int(row.get('FileSize'))
        by_study[row.get('StudyInstanceUID')].append((index, row))
    count, excluded, selected, patients = Counter(), Counter(), [], set()
    eligible = Counter()
    for index, study in enumerate(studies):
        key = study['StudyInstanceUID']
        if key in removed or key not in labels:
            continue
        reference_index, state = labels[key]
        if index != reference_index:
            raise ValueError('annotation_offset_mismatch')
        members = by_study.get(key, [])
        if sum(row['ImageCount'] for _, row in members) != 1 or len(members) != 1:
            excluded['not_single_image_study'] += 1
            continue
        row_index, row = members[0]
        if row.get('Modality') not in ('CR', 'DX'):
            excluded['unsupported_modality'] += 1
            continue
        if not row.get('PatientID') or not isinstance(row['PatientID'], str):
            excluded['missing_patient_membership'] += 1
            continue
        if row.get('LicenseURI', '').rstrip('/') not in (
                'http://creativecommons.org/licenses/by-nc/4.0',
                'https://creativecommons.org/licenses/by-nc/4.0'):
            raise ValueError('license_contract_rejected')
        if row['FileSize'] > MAX_OBJECT:
            excluded['object_exceeds_bound'] += 1
            continue
        eligible[state] += 1
        if count[state] >= quota:
            continue
        if row['PatientID'] in patients:
            excluded['selected_patient_already_used'] += 1
            continue
        patients.add(row['PatientID'])
        selected.append({'case_id': f'case_{len(selected):03d}',
                         'annotation_study_index': index, 'series_inventory_index': row_index,
                         'lung_opacity_reference': state, 'expected_dicom_bytes': row['FileSize']})
        count[state] += 1
    if count != Counter({'positive': quota, 'negative': quota}):
        raise ValueError('patient_unique_balanced_quota_not_met')
    if sum(case['expected_dicom_bytes'] for case in selected) > MAX_BYTES:
        raise BoundError('published_total_exceeds_cap')
    return selected, {'selected_class_counts': dict(count), 'eligible_single_image_study_counts': dict(eligible),
                      'excluded_counts': dict(excluded), 'selected_unique_patients': len(patients),
                      'single_image_studies_in_inventory': sum(sum(r['ImageCount'] for _, r in v) == 1
                                                               for v in by_study.values()),
                      'inventory_series_count': len(rows), 'inventory_study_count': len(by_study)}


def prepare(*, acquisition_root, audit_root, output_root, run_id, approved):
    if approved is not True:
        raise ValueError('explicit_small_acquisition_approval_required')
    source, _ = verified_manifest(acquisition_root, annotation.SCHEMA)
    audit_root, audit_manifest = verified_manifest(audit_root, audit.SCHEMA)
    prior = json.loads((audit_root / 'audit.json').read_bytes())
    if prior['input_manifest_sha256'] != sha256_file(source / 'manifest.json'):
        raise ValueError('annotation_audit_lineage_mismatch')
    temporary, target = new_atomic_run(output_root, run_id)
    started = time.monotonic()
    try:
        catalog = annotation.public_get(annotation.public_opener(), annotation.CATALOG, 1500000)
        removed = annotation.catalog_contract(catalog)
        payload = json.loads((source / 'annotations.json').read_bytes(), object_pairs_hook=annotation.reject_duplicate_keys)
        studies, labels, current = derived_labels(payload, removed)
        if current != prior['inventory']:
            raise ValueError('sealed_group_audit_no_longer_matches')
        client = PublicClient(initial_bytes=len(catalog))
        rows = client.metadata('getSeries', {'Collection': COLLECTION})
        if not isinstance(rows, list):
            raise ValueError('series_inventory_schema')
        selected, counts = select_cases(studies, labels, rows, removed)
        sop_metadata = []
        for case in selected:
            row = rows[case['series_inventory_index']]
            sops = client.metadata('getSOPInstanceUIDs', {'SeriesInstanceUID': row['SeriesInstanceUID']}, 65536)
            if not isinstance(sops, list) or len(sops) != 1 or not isinstance(sops[0], dict):
                raise ValueError('single_sop_binding_required')
            client.url('getSingleImage', {'SeriesInstanceUID': row['SeriesInstanceUID'],
                                         'SOPInstanceUID': sops[0].get('SOPInstanceUID')})
            case['sop_inventory_index'] = len(sop_metadata)
            sop_metadata.append(sops)
        expected = sum(case['expected_dicom_bytes'] for case in selected)
        if expected + client.received > MAX_BYTES:
            raise BoundError('combined_preflight_exceeds_cap')
        write_private_json(temporary / 'raw_series_inventory.json', {'rows': rows})
        write_private_json(temporary / 'raw_sop_inventory.json', {'responses': sop_metadata})
        plan = {'schema_version': PLAN_SCHEMA, 'run_id': run_id,
                'source_acquisition_root': str(source.relative_to(WORKSPACE)),
                'source_acquisition_manifest_sha256': sha256_file(source / 'manifest.json'),
                'source_reader_audit_root': str(audit_root.relative_to(WORKSPACE)),
                'source_reader_audit_manifest_sha256': sha256_file(audit_root / 'manifest.json'),
                'catalog_sha256': hashlib.sha256(catalog).hexdigest(),
                'annotation_sha256': sha256_file(source / 'annotations.json'),
                'reference_kind': current['derived_reference_kind'], 'finding': 'lung_opacity',
                'selection_rule': 'first_25_per_class_in_annotation_export_order_unique_patients_single_image_studies',
                'selection_order_bias_disclosed': True, 'unanimity_selection_bias_disclosed': True,
                'metadata_response_bytes': client.received, 'expected_dicom_bytes': expected,
                'limits': {'max_images': MAX_IMAGES, 'max_combined_response_bytes': MAX_BYTES,
                           'max_object_bytes': MAX_OBJECT},
                'counts': counts, 'cases': selected, 'images_downloaded': 0, 'model_calls': 0,
                'official_adjudication_reproduced': False, 'clinical_pneumonia_reference': False,
                'checkpoint_training_overlap_verified': False, 'primary_metric_eligible': False,
                'elapsed_seconds': round(time.monotonic() - started, 6)}
        write_private_json(temporary / 'plan.json', plan)
        write_private_json(temporary / 'manifest.json', {
            'schema_version': PLAN_SCHEMA + '-manifest', 'run_id': run_id,
            'sources': source_pins(), 'artifacts': {name: sha256_file(temporary / name)
                for name in ('plan.json', 'raw_series_inventory.json', 'raw_sop_inventory.json')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, plan


def load_pydicom():
    sys.path.insert(0, str(DEPENDENCIES))
    import pydicom
    if pydicom.__version__ != '3.0.2' or not Path(pydicom.__file__).resolve().is_relative_to(DEPENDENCIES):
        raise ValueError('header_dependency_pin_rejected')
    return pydicom


HEADER_TAGS = ('StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID',
               'Modality', 'Rows', 'Columns', 'NumberOfFrames', 'SamplesPerPixel',
               'PhotometricInterpretation', 'BitsAllocated', 'BitsStored', 'HighBit',
               'PixelRepresentation', 'ViewPosition', 'WindowCenter', 'WindowWidth',
               'RescaleSlope', 'RescaleIntercept', 'VOILUTSequence', 'ModalityLUTSequence',
               'PresentationLUTShape')


def check_header(path, series, sop, pydicom):
    with path.open('rb') as handle:
        prefix = handle.read(132)
    if len(prefix) != 132 or prefix[128:] != b'DICM':
        raise ValueError('dicom_part10_required')
    # Do not allow pydicom warnings/logs to leak malformed metadata values.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=list(HEADER_TAGS), force=False)
        return header_contract(dataset, series, sop)


def header_contract(ds, series, sop):
    for key, expected in (('StudyInstanceUID', series['StudyInstanceUID']),
                          ('SeriesInstanceUID', series['SeriesInstanceUID']),
                          ('SOPInstanceUID', sop), ('PatientID', series['PatientID'])):
        if str(ds.get(key, '')) != expected:
            raise ValueError('dicom_inventory_membership_mismatch')
    if ds.get('Modality') not in ('DX', 'CR') or ds.get('Modality') != series['Modality']:
        raise ValueError('dicom_modality_mismatch')
    rows, columns = int(ds.get('Rows', 0)), int(ds.get('Columns', 0))
    if not (1 <= rows <= 16384 and 1 <= columns <= 16384):
        raise ValueError('dicom_dimensions_rejected')
    if int(ds.get('NumberOfFrames', 1)) != 1 or int(ds.get('SamplesPerPixel', 0)) != 1:
        raise ValueError('single_frame_grayscale_required')
    photo = ds.get('PhotometricInterpretation', '')
    if photo not in ('MONOCHROME1', 'MONOCHROME2'):
        raise ValueError('monochrome_required')
    allocated, stored = int(ds.get('BitsAllocated', 0)), int(ds.get('BitsStored', 0))
    if allocated not in (8, 16) or not 1 <= stored <= allocated or int(ds.get('HighBit', -1)) != stored - 1:
        raise ValueError('pixel_storage_rejected')
    if int(ds.get('PixelRepresentation', -1)) not in (0, 1):
        raise ValueError('pixel_representation_rejected')
    view = ds.get('ViewPosition', '')
    if view not in ('', 'AP', 'PA'):
        raise ValueError('unsupported_view')
    syntax = str(ds.file_meta.get('TransferSyntaxUID', ''))
    if not UID_PATTERN.fullmatch(syntax):
        raise ValueError('transfer_syntax_required')
    # Persist only fixed image/storage metadata, no clinical text/identifiers.
    return {'rows': rows, 'columns': columns, 'frames': 1, 'modality': str(ds.get('Modality')),
            'photometric_interpretation': photo, 'bits_allocated': allocated, 'bits_stored': stored,
            'pixel_representation': int(ds.get('PixelRepresentation')), 'view': view or 'unknown',
            'transfer_syntax_uid': syntax,
            'window_pair_present': 'WindowCenter' in ds and 'WindowWidth' in ds,
            'rescale_pair_present': 'RescaleSlope' in ds and 'RescaleIntercept' in ds,
            'voi_lut_present': 'VOILUTSequence' in ds,
            'modality_lut_present': 'ModalityLUTSequence' in ds,
            'presentation_lut_shape_present': 'PresentationLUTShape' in ds,
            'membership_verified': True, 'pixels_decoded': False, 'display_transform_verified': False}


def acquire(*, plan_root, output_root, run_id, approved):
    if approved is not True:
        raise ValueError('explicit_small_acquisition_approval_required')
    plan_root, _ = verified_manifest(plan_root, PLAN_SCHEMA)
    plan = json.loads((plan_root / 'plan.json').read_bytes())
    source, _ = verified_manifest(WORKSPACE / plan['source_acquisition_root'], annotation.SCHEMA)
    reader_root, _ = verified_manifest(WORKSPACE / plan['source_reader_audit_root'], audit.SCHEMA)
    if sha256_file(source / 'manifest.json') != plan['source_acquisition_manifest_sha256'] or \
            sha256_file(reader_root / 'manifest.json') != plan['source_reader_audit_manifest_sha256']:
        raise ValueError('input_manifest_changed')
    rows = json.loads((plan_root / 'raw_series_inventory.json').read_bytes())['rows']
    sops = json.loads((plan_root / 'raw_sop_inventory.json').read_bytes())['responses']
    cases = plan['cases']
    if len(cases) != MAX_IMAGES or plan['limits'] != {'max_images': MAX_IMAGES,
            'max_combined_response_bytes': MAX_BYTES, 'max_object_bytes': MAX_OBJECT}:
        raise ValueError('fixed_plan_bounds_rejected')
    if sum(c['expected_dicom_bytes'] for c in cases) + plan['metadata_response_bytes'] > MAX_BYTES:
        raise BoundError('plan_exceeds_cap')
    pydicom = load_pydicom()
    # This logger is limited to this process; no model imports/other tasks.
    logging.getLogger('pydicom').disabled = True
    temporary, target = new_atomic_run(output_root, run_id)
    started, outcomes = time.monotonic(), []
    client = PublicClient(initial_bytes=plan['metadata_response_bytes'])
    try:
        for case in cases:
            case_dir = temporary / case['case_id']
            private_directory(case_dir)
            image = case_dir / 'image.dcm'
            outcome = {'case_id': case['case_id'], 'lung_opacity_reference': case['lung_opacity_reference']}
            try:
                series = rows[case['series_inventory_index']]
                sop = sops[case['sop_inventory_index']][0]['SOPInstanceUID']
                size = client.get('getSingleImage', {'SeriesInstanceUID': series['SeriesInstanceUID'],
                    'SOPInstanceUID': sop}, MAX_OBJECT, destination=image, expected=case['expected_dicom_bytes'])
                header = check_header(image, series, sop, pydicom)
                outcome.update(status='acquired_header_bound', bytes=size, sha256=sha256_file(image),
                               artifact=f"{case['case_id']}/image.dcm", header=header)
            except Exception as error:
                # No error messages/URL/tracebacks/source IDs; never replace.
                outcome.update(status='failed_without_replacement', error_type=type(error).__name__,
                               partial_bytes=image.stat().st_size if image.exists() else 0)
                if image.exists():
                    outcome.update(artifact=f"{case['case_id']}/image.dcm", sha256=sha256_file(image))
            outcomes.append(outcome)
            print(json.dumps({'status': 'bounded_acquisition_progress', 'attempted': len(outcomes),
                'acquired': sum(r['status'] == 'acquired_header_bound' for r in outcomes),
                'response_bytes': client.received, 'model_calls': 0}), flush=True)
            if client.received >= MAX_BYTES:
                break
        counts = Counter(o['status'] for o in outcomes)
        report = {'schema_version': SCHEMA, 'run_id': run_id,
                  'status': 'acquisition_complete' if counts['acquired_header_bound'] == MAX_IMAGES else 'acquisition_incomplete_no_replacement',
                  'plan_root': str(plan_root.relative_to(WORKSPACE)),
                  'plan_manifest_sha256': sha256_file(plan_root / 'manifest.json'),
                  'planned_images': len(cases), 'attempted_images': len(outcomes),
                  'acquired_images': counts['acquired_header_bound'],
                  'failed_images': counts['failed_without_replacement'],
                  'not_attempted_images': len(cases) - len(outcomes),
                  'acquired_class_counts': dict(Counter(o['lung_opacity_reference'] for o in outcomes if o['status'] == 'acquired_header_bound')),
                  'image_response_bytes': client.received - plan['metadata_response_bytes'],
                  'combined_response_bytes': client.received, 'byte_cap': MAX_BYTES,
                  'patient_grouping_verified_via_public_inventory': True,
                  'image_label_binding_verified': counts['acquired_header_bound'] == MAX_IMAGES,
                  'reference_kind': plan['reference_kind'], 'clinical_pneumonia_reference': False,
                  'primary_metric_eligible': False, 'display_transform_verified': False,
                  'published_image_checksum_verified': False,
                  'checkpoint_training_overlap_verified': False,
                  'pixels_decoded': 0, 'model_calls': 0, 'slurm_submissions': 0,
                  'pydicom_version': pydicom.__version__, 'cases': outcomes,
                  'elapsed_seconds': round(time.monotonic() - started, 6)}
        write_private_json(temporary / 'acquisition.json', report)
        dependencies = {str(p.relative_to(WORKSPACE)): sha256_file(p)
                        for p in sorted((DEPENDENCIES / 'pydicom').rglob('*.py'))}
        artifacts = {'acquisition.json': sha256_file(temporary / 'acquisition.json')}
        artifacts.update({o['artifact']: o['sha256'] for o in outcomes if 'artifact' in o})
        write_private_json(temporary / 'manifest.json', {
            'schema_version': SCHEMA + '-manifest', 'run_id': run_id,
            'input_plan_manifest_sha256': report['plan_manifest_sha256'],
            'sources': source_pins(), 'header_dependency_sources': dependencies, 'artifacts': artifacts})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'acquire'))
    parser.add_argument('--acquisition-root')
    parser.add_argument('--audit-root')
    parser.add_argument('--plan-root')
    parser.add_argument('--output-root', default=str(DEFAULT_ROOT))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--approve-max50-max1gib', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.action == 'prepare':
            target, result = prepare(acquisition_root=args.acquisition_root, audit_root=args.audit_root,
                output_root=args.output_root, run_id=args.run_id, approved=args.approve_max50_max1gib)
            safe = {'status': 'plan_frozen_before_images', 'counts': result['counts'],
                    'expected_dicom_bytes': result['expected_dicom_bytes'], 'images_downloaded': 0}
        else:
            target, result = acquire(plan_root=args.plan_root, output_root=args.output_root,
                                    run_id=args.run_id, approved=args.approve_max50_max1gib)
            safe = {k: result[k] for k in ('status', 'planned_images', 'acquired_images', 'failed_images',
                    'not_attempted_images', 'acquired_class_counts', 'combined_response_bytes', 'elapsed_seconds')}
    except Exception as error:
        print(json.dumps({'status': 'acquisition_failed_closed', 'error_type': type(error).__name__, 'model_calls': 0}))
        return 2
    safe.update(output_root=str(target.relative_to(WORKSPACE)), model_calls=0,
                manifest_sha256=sha256_file(target / 'manifest.json'))
    print(json.dumps(safe), flush=True)
    return 0 if args.action == 'prepare' or result['status'] == 'acquisition_complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
