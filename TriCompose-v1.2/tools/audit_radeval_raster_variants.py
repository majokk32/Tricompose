"""Explicit source-raster metadata diagnostic; no pixels or substitutions.

Check whether author PNG names correspond to the original MIMIC JPEG under
the SAME complete subject/study/DICOM suffix. A match is only a metadata lead,
never a claim that author's raster/crop and original JPEG pixels are equal.
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path, PurePosixPath
import re

import smoke_radgraph_xl as installed
from audit_radeval_image_linkage import SOURCE, ROOTS, BASE, PROJECT, CONTRACT
from tricompose_v12.radeval_image_linkage import author_path

OUT = BASE / 'radeval_image_linkage_runs/raster_stat_12714150_001'


def execute():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_existing_cpu_allocation_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    path = SOURCE / manifest['file']
    if installed.sha256(path) != manifest['sha256']:
        raise ValueError('fixed_author_source_required')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        keys = list(dict.fromkeys(row['images_path'].strip() for row in csv.DictReader(stream)))
    layout, extension_counts, records, resolver = Counter(), Counter(), [], {}
    for index, key in enumerate(keys):
        parsed = author_path(key)
        if parsed['family'] != 'mimic':
            continue
        suffix = PurePosixPath(parsed['relative_suffix'])
        extension_counts[suffix.suffix.lower()] += 1
        standard_id = bool(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{8}){4}', suffix.stem, re.I))
        layout['standard_mimic_dicom_filename'] += standard_id
        available = set()
        if suffix.suffix.lower() == '.png' and standard_id:
            for root in ROOTS:
                candidate = root / str(suffix.with_suffix('.jpg'))
                try:
                    resolved = candidate.resolve(strict=True)
                    if resolved.is_relative_to(PROJECT.resolve()) and resolved.is_file():
                        available.add(resolved)
                except (OSError, ValueError):
                    pass
        status = ('original_jpeg_metadata_lead' if len(available) == 1 else
                  'ambiguous_original_candidates' if available else 'no_author_png_to_original_jpeg_lead')
        record = {'source_id': f'source_{index:04d}', 'status': status,
                  'author_extension': suffix.suffix.lower(), 'standard_image_key_format': standard_id,
                  'candidate_count': len(available), 'raster_pixel_equivalence_verified': False,
                  'authorized_for_model_inference_by_this_receipt': False}
        records.append(record)
        if len(available) == 1:
            resolved = next(iter(available))
            stat = resolved.stat()
            resolver[record['source_id']] = {'original_jpeg_path': str(resolved), 'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino,
                'identity_basis': 'explicit_png_to_jpg_same_patient_study_standard_dicom_id',
                'author_raster_byte_equivalence': False}
    summary = {'status': 'source_raster_metadata_audit_complete',
        'mimic_sources': len(records), 'author_extension_counts': dict(extension_counts),
        'filename_format_counts': dict(layout), 'raster_mapping_status_counts':
        dict(Counter(r['status'] for r in records)), 'image_bytes_read': False,
        'model_calls': 0, 'pixel_equivalence_claimed': False, 'downloads': 0,
        'selection_changed': False, 'source_raster_substitution_performed': False}
    os.umask(0o007)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    installed.write_json(OUT / 'summary.json', summary)
    installed.write_json(OUT / 'records.json', records)
    installed.write_json(OUT / 'internal_resolver.json', resolver)
    pins = [{'path': str(p.relative_to(installed.WORKSPACE)), 'sha256': installed.sha256(p)}
        for p in (Path(__file__).resolve(), SOURCE / 'manifest.json', path, CONTRACT,
                  installed.WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_linkage.py',
                  installed.WORKSPACE / 'TriCompose-v1.2/tools/audit_radeval_image_linkage.py')]
    installed.write_json(OUT / 'manifest.json', {'pins': pins,
        'outputs': [{'path': p.name, 'sha256': installed.sha256(p)} for p in sorted(OUT.glob('*.json'))],
        'source_image_computation_authorized': False})
    print(json.dumps({'status': 'protected_raster_metadata_preflight_complete',
                     'manifest_sha256': installed.sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
