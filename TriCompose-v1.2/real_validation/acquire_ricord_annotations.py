#!/usr/bin/env python3
"""Acquire only the user-approved official RICORD-1C annotation JSON.

No image download, pixels, model imports, inference, credentials or submission.
Only allowlisted schema/label names and aggregate counts leave protected data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
PROTECTED = WORKSPACE / 'artifacts/protected'
sys.path.insert(0, str(WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       sha256_file, write_private_json)

CATALOG = 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281'
ANNOTATIONS = ('https://wiki.cancerimagingarchive.net/download/attachments/70230281/'
               '1c_mdai_rsna_project_MwBeK3Nr_annotations_labelgroup_all_2021-01-08-164102.json?api=v2')
EXPECTED_BYTES = 2449004  # Metadata-only HEAD observed before approval; not a checksum.
SCHEMA = 'tricompose-ricord-annotation-acquisition-v1'
HEADERS = {'User-Agent': 'TriCompose-public-annotation-acquisition/1.0'}
PROTOCOL = WORKSPACE / 'docs/ricord_annotation_readiness_protocol.md'
TESTS = WORKSPACE / 'TriCompose-v1.2/tests/test_ricord_annotations.py'
SAFE_FIELDS = frozenset(('datasets', 'labelGroups', 'labels', 'studies', 'series',
    'images', 'annotations', 'id', 'name', 'type', 'scope', 'labelId', 'data',
    'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'PatientID',
    'patientId', 'height', 'width', 'metadata', 'project'))
CLASSIFICATIONS = ('typical_appearance', 'indeterminate_appearance',
                   'atypical_appearance', 'negative_for_pneumonia')
GRADES = ('mild', 'moderate', 'severe')
ALIASES = {
    'typical': 'typical_appearance', 'typical appearance': 'typical_appearance',
    'indeterminate': 'indeterminate_appearance',
    'indeterminate appearance': 'indeterminate_appearance',
    'atypical': 'atypical_appearance', 'atypical appearance': 'atypical_appearance',
    'negative for pneumonia': 'negative_for_pneumonia',
    'mild': 'mild', 'mild (1-2 zones)': 'mild',
    'moderate': 'moderate', 'moderate (3-4 zones)': 'moderate',
    'severe': 'severe', 'severe (>4 zones)': 'severe',
}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect_not_authorized')


def public_opener():
    return build_opener(ProxyHandler({}), NoRedirect())


def public_get(opener, url, limit):
    if url not in (CATALOG, ANNOTATIONS):
        raise ValueError('nonallowlisted_url')
    started = time.monotonic()
    with opener.open(Request(url, headers=HEADERS), timeout=15) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError('public_response_rejected')
        chunks, received = [], 0
        while True:
            chunk = response.read(min(65536, limit + 1 - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            if received > limit or time.monotonic() - started > 90:
                raise ValueError('public_download_bound_exceeded')
        length = response.headers.get('Content-Length')
        if length is not None and int(length) != received:
            raise ValueError('content_length_mismatch')
    return b''.join(chunks)


class CatalogParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs, self.text = set(), []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.hrefs.add(dict(attrs).get('href', ''))

    def handle_data(self, data):
        self.text.append(data)


def catalog_contract(raw):
    parser = CatalogParser()
    parser.feed(raw.decode('utf-8', errors='strict'))
    text = ' '.join(parser.text)
    expected_path = ANNOTATIONS.removeprefix('https://wiki.cancerimagingarchive.net')
    if (ANNOTATIONS not in parser.hrefs and expected_path not in parser.hrefs
            or 'MIDRC-RICORD-1C' not in text
            or not any('creativecommons.org/licenses/by-nc/4.0' in href for href in parser.hrefs)):
        raise ValueError('official_catalog_contract_rejected')
    # Public catalog has two withdrawn study keys. Never emit/save these keys.
    removed = set(re.findall(r'1\.2\.826\.0\.1\.3680043\.10\.474\.\d+', text))
    if len(removed) != 2 or 'subsequently removed' not in text:
        raise ValueError('withdrawal_notice_requires_review')
    return removed


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def dictionaries(payload):
    pending, visited = [(payload, 0)], 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if depth > 40 or visited > 250000:
            raise ValueError('json_structure_bound_exceeded')
        if isinstance(node, dict):
            yield node
            pending.extend((value, depth + 1) for value in node.values()
                           if isinstance(value, (dict, list)))
        elif isinstance(node, list):
            pending.extend((value, depth + 1) for value in node
                           if isinstance(value, (dict, list)))


def label_name(name):
    if not isinstance(name, str):
        return None
    return ALIASES.get(' '.join(name.strip().casefold().split()))


def inventory(payload, removed):
    if not isinstance(payload, dict):
        raise ValueError('annotation_root_must_be_object')
    nodes = list(dictionaries(payload))
    definitions = []
    for node in nodes:
        groups = node.get('labelGroups')
        if isinstance(groups, list):
            for group in groups:
                if isinstance(group, dict) and isinstance(group.get('labels'), list):
                    definitions.extend(group['labels'])
    label_map, unknown_definition_count = {}, 0
    for definition in definitions:
        if not isinstance(definition, dict) or not isinstance(definition.get('id'), str):
            raise ValueError('label_definition_schema_rejected')
        key = definition['id']
        if key in label_map:
            raise ValueError('duplicate_label_definition')
        name = label_name(definition.get('name'))
        unknown_definition_count += name is None
        label_map[key] = name

    studies = [study for node in nodes if isinstance(node.get('studies'), list)
               for study in node['studies']]
    study_ids, active, withdrawn = set(), [], 0
    for study in studies:
        if not isinstance(study, dict) or not isinstance(study.get('StudyInstanceUID'), str):
            raise ValueError('study_schema_requires_review')
        key = study['StudyInstanceUID']
        if key in study_ids:
            raise ValueError('duplicate_study_key')
        study_ids.add(key)
        if key in removed:
            withdrawn += 1
        else:
            active.append(study)

    annotation_rows = [node for node in nodes if 'labelId' in node]
    annotations = Counter()
    by_study = {}
    unbound_annotations, unknown_label_annotations = 0, 0
    for row in annotation_rows:
        key = row.get('StudyInstanceUID')
        if key in removed:
            continue
        name = label_map.get(row.get('labelId'))
        if name is None:
            unknown_label_annotations += 1
            continue
        annotations[name] += 1
        if isinstance(key, str) and key in study_ids:
            by_study.setdefault(key, set()).add(name)
        else:
            unbound_annotations += 1

    classification = Counter()
    grades = Counter()
    for study in active:
        labels = by_study.get(study['StudyInstanceUID'], set())
        classes = labels.intersection(CLASSIFICATIONS)
        classification[next(iter(classes)) if len(classes) == 1 else
                       'conflicting' if len(classes) > 1 else 'unknown'] += 1
        levels = labels.intersection(GRADES)
        grades[next(iter(levels)) if len(levels) == 1 else
               'conflicting' if len(levels) > 1 else 'unknown'] += 1

    active_nodes = [node for study in active for node in dictionaries(study)]
    schema_fields = sorted({key for node in nodes for key in node if key in SAFE_FIELDS})
    images = sum(len(node['images']) for node in active_nodes if isinstance(node.get('images'), list))
    series = sum(len(node['series']) for node in active_nodes if isinstance(node.get('series'), list))
    scopes = Counter()
    for row in definitions:
        scope = row.get('scope')
        scopes[scope if scope in ('STUDY', 'SERIES', 'IMAGE', 'INSTANCE', 'GLOBAL', 'LOCAL')
               else 'unknown_or_other'] += 1
    return {
        'safe_schema_fields_observed': schema_fields,
        'label_definition_count': len(definitions),
        'recognized_label_names': sorted({name for name in label_map.values() if name is not None}),
        'unrecognized_label_definition_count': unknown_definition_count,
        'label_scope_counts': dict(sorted(scopes.items())),
        'studies_in_annotation_export': len(studies),
        'official_withdrawal_notice_count': len(removed),
        'withdrawn_studies_excluded': withdrawn,
        'active_study_count': len(active),
        'active_series_metadata_count': series,
        'active_image_metadata_count': images,
        'recognized_annotation_counts': dict(sorted(annotations.items())),
        'unrecognized_label_annotation_count': unknown_label_annotations,
        'recognized_annotations_without_study_binding': unbound_annotations,
        'study_classification_counts': dict(sorted(classification.items())),
        'study_airspace_grade_counts': dict(sorted(grades.items())),
        'patient_id_field_observed': any('PatientID' in node or 'patientId' in node for node in nodes),
        'patient_grouping_verified': False,
        'image_label_binding_verified': False,
        'dicom_preprocessing_verified': False,
        'checkpoint_training_overlap_verified': False,
        'pneumonia_diagnosis_reference_validated': False,
        'primary_metric_eligible': False,
    }


def run(*, output_root, run_id, allow_annotation_download):
    if allow_annotation_download is not True:
        raise RuntimeError('explicit_annotation_download_approval_required')
    temporary, target = new_atomic_run(output_root, run_id)
    started = time.monotonic()
    try:
        opener = public_opener()
        catalog = public_get(opener, CATALOG, 1500000)
        removed = catalog_contract(catalog)
        raw = public_get(opener, ANNOTATIONS, EXPECTED_BYTES)
        if len(raw) != EXPECTED_BYTES:
            raise ValueError('annotation_size_changed_requires_review')
        # Approved raw annotation stays protected; never print its body.
        descriptor = os.open(temporary / 'annotations.json',
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary / 'annotations.json', 0o660)
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
        result = inventory(payload, removed)
        summary = {
            'schema_version': SCHEMA,
            'status': 'annotations_acquired_image_binding_pending',
            'dataset': 'MIDRC-RICORD-1C', 'dataset_release': '2021-01-15',
            'official_catalog': CATALOG, 'data_doi': '10.7937/91ah-v663',
            'license': 'CC BY-NC 4.0',
            'source_catalog_sha256': hashlib.sha256(catalog).hexdigest(),
            'source_annotation_url_sha256': hashlib.sha256(ANNOTATIONS.encode()).hexdigest(),
            'annotation_bytes': len(raw), 'annotation_sha256': hashlib.sha256(raw).hexdigest(),
            'published_checksum_verified': False,
            'integrity_scope': 'observed_size_and_local_sha256_not_published_checksum',
            'inventory': result, 'credentials_read': False, 'images_downloaded': 0,
            'pixel_values_decoded': False, 'model_calls': 0, 'slurm_submissions': 0,
            'thresholds_changed': False, 'selection_changed': False,
            'benchmark_executed': False, 'primary_metric_eligible': False,
            'elapsed_seconds': round(time.monotonic() - started, 6),
        }
        write_private_json(temporary / 'acquisition.json', summary)
        write_private_json(temporary / 'manifest.json', {
            'schema_version': SCHEMA + '-manifest', 'run_id': run_id,
            'sources': {str(path.relative_to(WORKSPACE)): sha256_file(path)
                        for path in (Path(__file__), TESTS, PROTOCOL,
                                     WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py')},
            'artifacts': {name: sha256_file(temporary / name)
                          for name in ('annotations.json', 'acquisition.json')},
        })
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', default=str(PROTECTED / 'tricompose_v1_2/reference_datasets/ricord_1c'))
    parser.add_argument('--allow-annotation-download', action='store_true')
    args = parser.parse_args(argv)
    try:
        target, summary = run(output_root=args.output_root, run_id=args.run_id,
                              allow_annotation_download=args.allow_annotation_download)
    except Exception as error:
        print(json.dumps({'status': 'acquisition_failed_closed',
                          'error_type': type(error).__name__, 'model_calls': 0}))
        return 2
    print(json.dumps({'status': summary['status'], 'annotation_bytes': summary['annotation_bytes'],
        'inventory': summary['inventory'], 'model_calls': 0, 'images_downloaded': 0,
        'elapsed_seconds': summary['elapsed_seconds'],
        'manifest_sha256': sha256_file(target / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
