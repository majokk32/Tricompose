"""Resolve fixed comparison/change schema descriptors before model calls."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from audit_radeval_expert_labels import SOURCE, ROOT


def execute():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_existing_allocation_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    source = SOURCE / manifest['file']
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('pinned_source_required')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    categories = {5: Counter(), 6: Counter()}
    for row in rows:
        for slot in range(1, 4):
            for line in row[f'annotation{slot}'].splitlines():
                match = re.fullmatch(r'\s*([56])[.)]\s+(.+?)\s*:\s*([0-9]+)\s*[.;]?\s*', line)
                if match:
                    label = ' '.join(re.sub(r'[^a-z ]', ' ', match[2].lower()).split())
                    categories[int(match[1])][label] += 1
    schemas = {}
    for index, counts in categories.items():
        prefix = 'mention ' if index == 5 else 'omission '
        safe = {k: v for k, v in counts.items() if v >= 100 and k.startswith(prefix)
                and len(k) <= 160 and re.fullmatch(r'[a-z ]+', k)}
        schemas[str(index)] = {'repeated_schema_descriptions': safe,
                              'unexported_occurrences': sum(counts.values()) - sum(safe.values())}
    out = ROOT / 'artifacts/protected/tricompose_v1_2/report_metric_sources/radeval_expert_label_schema_12714150_002'
    os.umask(0o007)
    out.mkdir(mode=0o2770)
    out.chmod(0o2770)
    with (out / 'schema_labels.json').open('x') as stream:
        json.dump({'status': 'fixed_repeated_category_schema_only', 'categories': schemas,
                   'raw_clinical_statements_exported': False}, stream, sort_keys=True, indent=2)
        stream.write('\n')
    (out / 'schema_labels.json').chmod(0o660)
    print(json.dumps({'status': 'schema_labels_verified', 'audit_sha256':
        hashlib.sha256((out / 'schema_labels.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
