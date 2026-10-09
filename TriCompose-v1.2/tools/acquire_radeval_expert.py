"""Approved small author-released benchmark; raw CSV stays protected.

This worker is only permitted inside the existing CPU Slurm allocation. Its
public output is a receipt, never source rows, reports, annotations, or IDs.
"""
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import urllib.request

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
ROOT = WORKSPACE / 'artifacts/protected/tricompose_v1_2/report_metric_sources'
RUN_ID = 'radeval_expert_source_12714150_001'
REPOSITORY = 'IAMJB/RadEvalExpertDataset'
REVISION = 'b4bd9d6ee75fcd155b6de466f1b7b7bd774408be'
FILENAME = 'reader_study_final_with_annotationsv3.csv'
SIZE = 691815
BLOB = 'c8862ccd726129a6e2a986b93e07f41b30dcee85'
EXPECTED_FIELDS = {'annotator', 'type', 'ground_truth', 'images_path',
                   'prediction1', 'prediction2', 'prediction3',
                   'annotation1', 'annotation2', 'annotation3'}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def write_json(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, sort_keys=True, indent=2)
        stream.write('\n')
    path.chmod(0o660)


def directory(path, exist_ok=False):
    path.mkdir(mode=0o2770, exist_ok=exist_ok)
    path.chmod(0o2770)
    require(path.stat().st_gid in (96293, 65534), 'project_group_mount_required')


def acquire():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(),
            'approved_existing_cpu_slurm_required')
    os.umask(0o007)
    directory(ROOT, exist_ok=True)
    run = ROOT / RUN_ID
    directory(run)
    url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{FILENAME}'
    # Public, credential-free GET; no patient text is sent in the request.
    request = urllib.request.Request(url, headers={'User-Agent': 'TriCompose-public-asset-receipt/1'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(SIZE + 1)
    blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    require(len(data) == SIZE and blob == BLOB, 'pinned_author_csv_mismatch')
    source = run / FILENAME
    with source.open('xb') as stream:
        stream.write(data)
    source.chmod(0o660)
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    require(set(reader.fieldnames or []) == EXPECTED_FIELDS, 'author_schema_changed')
    rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            'malformed_csv_row')
    # Only declared structure, availability, and aggregate grammar indicators.
    # Never export annotation lines or free-form values.
    formats = Counter()
    section_counts = Counter()
    for row in rows:
        kind = row['type'].strip().lower()
        section_counts[kind if kind in ('findings', 'impression') else 'unrecognized_section'] += 1
        for slot in range(1, 4):
            value = row[f'annotation{slot}']
            formats['annotation_cells'] += 1
            if not value.strip():
                formats['blank_cells'] += 1
                continue
            formats['nonblank_cells'] += 1
            for line in value.splitlines():
                if not line.strip():
                    continue
                formats['nonblank_lines'] += 1
                if re.fullmatch(r'\s*(?:clinically\s+)?(?:insignificant|significant)(?:\s+errors)?\s*:\s*',
                                line, re.I):
                    formats['recognized_severity_header_lines'] += 1
                elif re.match(r'\s*[1-7][.)]\s', line):
                    formats['numbered_category_lines'] += 1
                    if re.search(r'[:=]\s*[0-9]+\s*[.;]?\s*$', line):
                        formats['category_lines_with_explicit_terminal_integer'] += 1
                    if re.fullmatch(r'\s*[1-7][.)]\s*[0-9]+\s*', line):
                        formats['integer_only_category_lines'] += 1
                else:
                    formats['unrecognized_lines'] += 1
    audit = {'schema_version': 'radeval-expert-structure-audit-v1', 'csv_fields': reader.fieldnames,
        'rows': len(rows), 'section_counts': dict(section_counts),
        'annotation_grammar_counts': dict(formats),
        'unique_internal_source_groups': len({r['images_path'] for r in rows}),
        'unique_internal_readers': len({r['annotator'] for r in rows}),
        'blank_references': sum(not r['ground_truth'].strip() for r in rows),
        'blank_predictions': sum(not r[f'prediction{s}'].strip() for r in rows for s in range(1, 4)),
        'raw_text_or_source_ids_exported': False, 'blank_annotation_means_zero': False}
    write_json(run / 'structure_audit.json', audit)
    manifest = {'schema_version': 'radeval-expert-pinned-source-v1',
        'repository': REPOSITORY, 'revision': REVISION, 'file': FILENAME,
        'bytes': len(data), 'git_blob': blob, 'sha256': hashlib.sha256(data).hexdigest(),
        'source_url': url, 'card_license': 'mit', 'mimic_dua_still_applies': True,
        'user_confirmed_authorization': True, 'job_id': 12714150,
        'structure_audit_sha256': hashlib.sha256((run / 'structure_audit.json').read_bytes()).hexdigest(),
        'acquisition_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_payload_protected': True, 'raw_payload_publicly_printed': False}
    write_json(run / 'manifest.json', manifest)
    print(json.dumps({'status': 'protected_pinned_expert_source_verified',
                      'manifest_sha256': hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    try:
        acquire()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
