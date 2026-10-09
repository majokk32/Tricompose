"""Actual CPU allocation, source-path metadata and stat only, no image bytes.

Raw path resolver stays protected and separate from sanitized manifests. All
source keys retained; unavailable/other-license images are not substituted.
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path

import smoke_radgraph_xl as installed
from tricompose_v12.radeval_image_linkage import author_path, public_record, VERSION

WORKSPACE = installed.WORKSPACE
PROJECT = WORKSPACE.parent
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'report_metric_sources/radeval_expert_source_12714150_001'
CONTRACT = BASE / 'report_metric_sources/radeval_expert_contract_12714150_003/inventory.json'
OUT = BASE / 'radeval_image_linkage_runs/local_stat_12714150_001'
ROOTS = (
    PROJECT / 'datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org/files',
    PROJECT / 'datasets/vlm_radiology_report_generation/mimic-cxr-jpg-2.0.0.physionet.org/files',
    PROJECT / 'datasets/mimic-cxr-jpg/2.1.0/files',
    PROJECT / 'datasets/mimic-cxr-jpg/2.0.0/files',
    PROJECT / 'datasets/mimic-cxr-jpg/files',
    PROJECT / 'local_data/mimic-cxr-jpg/files',
    PROJECT / 'EHRXDiff_baseline/local_data/mimic-cxr-jpg/files',
)


def execute():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_existing_cpu_allocation_required')
    if installed.sha256(SOURCE / 'manifest.json') != 'be99b29497d8bf75d52cdd7ca1cc353814bbc18a9ef5c9c1675f03b5752e9ff4':
        raise ValueError('pinned_author_source_required')
    if installed.sha256(CONTRACT) != '29c076bc24fd4c628703cbe1a1ea5049cf090660ff06a2c46e09e59e50b49365':
        raise ValueError('accepted_author_pair_contract_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    source = SOURCE / manifest['file']
    if installed.sha256(source) != manifest['sha256']:
        raise ValueError('pinned_author_csv_required')
    # Data are consumed only internally to read image-path metadata. No EHR or
    # reference/prediction text is interpreted or exported in this stage.
    with source.open(newline='', encoding='utf-8-sig') as stream:
        image_keys = [row['images_path'].strip() for row in csv.DictReader(stream)]
    source_ids = {}
    records, resolver = [], {}
    for key in image_keys:
        if key in source_ids:
            continue
        source_id = f'source_{len(source_ids):04d}'
        source_ids[key] = source_id
        parsed = author_path(key)
        candidates = set()
        status = 'source_layout_not_resolved'
        if parsed['family'] == 'chexpert':
            status = 'separate_dataset_access_not_confirmed'
        elif parsed['family'] == 'mimic':
            for root in ROOTS:
                candidate = root / parsed['relative_suffix']
                try:
                    path = candidate.resolve(strict=True)
                    if path.is_relative_to(PROJECT.resolve()) and path.is_file():
                        candidates.add(path)
                except (FileNotFoundError, PermissionError, OSError):
                    continue
            status = ('exact_suffix_file_available' if len(candidates) == 1 else
                      'ambiguous_multiple_local_files' if candidates else 'exact_suffix_file_not_found')
            if len(candidates) == 1:
                path = next(iter(candidates))
                stat = path.stat()
                resolver[source_id] = {'local_path': str(path), 'size': stat.st_size,
                    'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino,
                    'image_bytes_hashed': False, 'identity_basis': 'exact_author_patient_study_image_suffix'}
        records.append(public_record(source_id, parsed, status, candidate_count=len(candidates)))
    plan = json.loads(CONTRACT.read_text())
    by_source = {r['source_id']: r for r in records}
    if set(by_source) != {p['source_id'] for p in plan['records']}:
        raise ValueError('all_source_inventory_join_required')
    summary = {'schema_version': VERSION, 'status': 'metadata_preflight_complete',
        'source_keys_attempted': len(records), 'report_pairs_retained': len(plan['records']),
        'source_family_counts': dict(Counter(r['source_family'] for r in records)),
        'author_layout_status_counts': dict(Counter(r['author_path_status'] for r in records)),
        'source_local_status_counts': dict(Counter(r['local_image_status'] for r in records)),
        'report_pair_local_status_counts': dict(Counter(by_source[p['source_id']]['local_image_status'] for p in plan['records'])),
        'generic_root_availability': [{'root': str(root.relative_to(PROJECT)), 'directory_exists': root.is_dir()} for root in ROOTS],
        'image_pixels_decoded': False, 'image_bytes_read_or_hashed': False,
        'model_calls': 0, 'new_slurm_submissions': 0, 'downloads': 0,
        'real_reports_interpreted': False, 'clinical_qualified': False,
        'source_path_or_patient_key_in_public_metadata': False,
        'source_coverage_based_on_stat_only': True, 'selection_changed': False}
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    installed.write_json(OUT / 'source_records.json', records)
    installed.write_json(OUT / 'internal_resolver.json', resolver)
    installed.write_json(OUT / 'summary.json', summary)
    pins = [{'path': str(p.relative_to(WORKSPACE)), 'sha256': installed.sha256(p)} for p in (
        Path(__file__).resolve(), WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_linkage.py',
        SOURCE / 'manifest.json', source, CONTRACT)]
    installed.write_json(OUT / 'manifest.json', {'schema_version': VERSION + '-receipt',
        'pins': pins, 'outputs': [{'path': p.name, 'sha256': installed.sha256(p)} for p in sorted(OUT.glob('*.json'))],
        'raw_paths_in_separate_internal_resolver_only': True, 'image_bytes_read': False,
        'source_input_computation_authorized_by_this_receipt': False})
    print(json.dumps({'status': 'protected_image_linkage_metadata_complete',
                     'manifest_sha256': installed.sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
