"""Export repeated annotation SCHEMA labels only, never patient statements.

An allowed category prefix and >=100 exact occurrences are required. This is
format discovery before inference, not clinical annotation or metric tuning.
"""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
SOURCE = ROOT / 'artifacts/protected/tricompose_v1_2/report_metric_sources/radeval_expert_source_12714150_001'
OUT = ROOT / 'artifacts/protected/tricompose_v1_2/report_metric_sources/radeval_expert_label_schema_12714150_001'


def execute():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_existing_allocation_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    source = SOURCE / manifest['file']
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('pinned_source_required')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    labels = [Counter() for _ in range(7)]
    path_forms = Counter()
    for row in rows:
        path_forms['chexpert_patient_folder'] += bool(re.search(r'(?:^|[/\\])patient[0-9]+(?=[/\\]|$)', row['images_path']))
        path_forms['mimic_patient_folder'] += bool(re.search(r'(?:^|[/\\])p[0-9]{6,}(?=[/\\]|$)', row['images_path']))
        for slot in range(1, 4):
            for line in row[f'annotation{slot}'].splitlines():
                match = re.fullmatch(r'\s*([1-7])[.)]\s+(.+?)\s*:\s*([0-9]+)\s*[.;]?\s*', line)
                if match:
                    normalized = ' '.join(re.sub(r'[^a-z ]', ' ', match[2].lower()).split())
                    labels[int(match[1]) - 1][normalized] += 1
    allowed_prefixes = ('false prediction', 'omission of finding', 'incorrect location',
        'incorrect severity', 'mention comparison', 'omission change', 'inarticulate report')
    schemas = []
    for index, counts in enumerate(labels):
        approved = {key: value for key, value in counts.items() if value >= 100 and
                    key.startswith(allowed_prefixes[index]) and len(key) <= 160 and
                    re.fullmatch(r'[a-z ]+', key)}
        schemas.append({'category_index': index + 1, 'repeated_schema_descriptions': approved,
                        'other_description_count': sum(counts.values()) - sum(approved.values())})
    payload = {'status': 'repeated_schema_labels_only', 'categories': schemas,
               'source_group_field_family_counts': dict(path_forms),
               'raw_annotations_or_reports_exported': False}
    os.umask(0o007)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    with (OUT / 'schema_labels.json').open('x') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write('\n')
    (OUT / 'schema_labels.json').chmod(0o660)
    print(json.dumps({'status': 'protected_repeated_schema_audit_complete', 'audit_sha256':
        hashlib.sha256((OUT / 'schema_labels.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
