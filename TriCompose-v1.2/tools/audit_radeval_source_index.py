"""Bounded exact metadata linkage; no EHR interpretation or image bytes.

Only the cxr_path column of the existing linkage CSV is retained internally.
No basename-only matches: full patient/study/image suffix must be preserved.
Permission errors are separate from missing paths. All results are protected.
"""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path

import smoke_radgraph_xl as installed
from audit_radeval_image_linkage import SOURCE, ROOTS, BASE, PROJECT, CONTRACT
from tricompose_v12.radeval_image_linkage import author_path

INDEX = PROJECT / 'datasets/three_modalities/v2_labs_vitals/manifest.csv'
OUT = BASE / 'radeval_image_linkage_runs/exact_index_12714150_001'


def bounded_stat(path):
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(PROJECT.resolve()) or not resolved.is_file():
            return 'outside_project_or_not_file', None
        return 'file_metadata_available', resolved
    except PermissionError:
        return 'permission_denied', None
    except FileNotFoundError:
        return 'file_not_found', None
    except OSError:
        return 'other_filesystem_error', None


def execute():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_existing_cpu_allocation_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    source = SOURCE / manifest['file']
    if installed.sha256(source) != manifest['sha256']:
        raise ValueError('fixed_author_csv_required')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        source_keys = list(dict.fromkeys(row['images_path'].strip() for row in csv.DictReader(stream)))
    targets = {parsed['relative_suffix']: f'source_{i:04d}' for i, key in enumerate(source_keys)
               if (parsed := author_path(key))['family'] == 'mimic'}
    candidates = defaultdict(set)
    stat_counts = Counter()
    for suffix, source_id in targets.items():
        for root in ROOTS:
            status, resolved = bounded_stat(root / suffix)
            stat_counts[status] += 1
            if resolved is not None:
                candidates[source_id].add(resolved)
    index_status, checked, exact_rows = 'index_file_unavailable', 0, 0
    index_sha = None
    if INDEX.is_file():
        index_sha = installed.sha256(INDEX)
        with INDEX.open(newline='', encoding='utf-8') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if header.count('cxr_path') != 1:
                raise ValueError('unambiguous_linkage_path_column_required')
            path_column = header.index('cxr_path')
            index_status = 'path_column_metadata_join_complete'
            for fields in reader:
                if checked >= 1000000:
                    raise ValueError('bounded_linkage_limit_exceeded')
                checked += 1
                if len(fields) != len(header):
                    raise ValueError('linkage_csv_structure_invalid')
                # Do not interpret/store diagnoses, labs, EHR or report fields.
                text = fields[path_column]
                parsed = author_path(text)
                if parsed['family'] != 'mimic' or parsed['relative_suffix'] not in targets:
                    continue
                exact_rows += 1
                candidate = Path(text)
                if not candidate.is_absolute():
                    candidate = INDEX.parent / candidate
                status, resolved = bounded_stat(candidate)
                stat_counts['index_' + status] += 1
                if resolved is not None:
                    candidates[targets[parsed['relative_suffix']]].add(resolved)
        if installed.sha256(INDEX) != index_sha:
            raise ValueError('linkage_changed_during_metadata_audit')
    records, resolver = [], {}
    for source_id in targets.values():
        paths = candidates[source_id]
        status = ('exact_source_file_metadata_available' if len(paths) == 1 else
                  'ambiguous_exact_source_files' if paths else 'no_exact_local_file_metadata')
        records.append({'source_id': source_id, 'status': status, 'local_candidates': len(paths)})
        if len(paths) == 1:
            path = next(iter(paths))
            stat = path.stat()
            resolver[source_id] = {'local_path': str(path), 'bytes': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino, 'identity_basis': 'exact_subject_study_dicom_suffix'}
    summary = {'status': 'exact_metadata_index_audit_complete', 'mimic_source_keys': len(targets),
        'filesystem_status_counts': dict(stat_counts), 'linkage_index_status': index_status,
        'linkage_rows_examined_for_path_metadata': checked, 'exact_author_suffix_index_rows': exact_rows,
        'source_status_counts': dict(Counter(r['status'] for r in records)),
        'image_bytes_read': False, 'clinical_ehr_fields_interpreted': False,
        'model_calls': 0, 'downloads': 0, 'source_images_copied': False,
        'basename_only_retrieval': False, 'new_inference_authorized': False,
        'old_selection_changed': False}
    os.umask(0o007)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    installed.write_json(OUT / 'records.json', records)
    installed.write_json(OUT / 'internal_resolver.json', resolver)
    installed.write_json(OUT / 'summary.json', summary)
    pins = [{'path': str(p.relative_to(installed.WORKSPACE)), 'sha256': installed.sha256(p)} for p in (
        Path(__file__).resolve(), SOURCE / 'manifest.json', source, CONTRACT,
        installed.WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_linkage.py')]
    installed.write_json(OUT / 'manifest.json', {'pins': pins,
        'external_linkage_index': str(INDEX), 'external_linkage_index_sha256': index_sha,
        'outputs': [{'path': p.name, 'sha256': installed.sha256(p)} for p in sorted(OUT.glob('*.json'))],
        'image_bytes_read': False, 'new_inference_authorized': False})
    print(json.dumps({'status': 'protected_exact_source_index_audit_complete',
                     'manifest_sha256': installed.sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
