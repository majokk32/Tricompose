"""Internal format-only preflight; no clinical text or source IDs exported."""
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
SOURCE = ROOT / 'artifacts/protected/tricompose_v1_2/report_metric_sources/radeval_expert_source_12714150_001'
OUT = ROOT / 'artifacts/protected/tricompose_v1_2/report_metric_sources/radeval_expert_schema_12714150_001'
SECTIONS = {'findings', 'impression', 'both', 'report', 'full', 'all',
            'findings and impression', 'findings+impression', 'findings/impression',
            'findings & impression', 'findings, impression'}


def audit():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_allocation_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    data = (SOURCE / manifest['file']).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise ValueError('pinned_source_required')
    with (SOURCE / manifest['file']).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    headers, sections, exceptional, path_forms = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        section = row['type'].strip().lower()
        sections[section if section in SECTIONS else 'unknown'] += 1
        try:
            paths = ast.literal_eval(row['images_path'])
            if isinstance(paths, list) and paths and all(isinstance(p, str) for p in paths):
                path_forms['literal_list_of_strings'] += 1
                path_forms['list_size_' + str(len(paths))] += 1
            else:
                path_forms['unsupported_literal_type'] += 1
        except (ValueError, SyntaxError):
            path_forms['not_python_literal'] += 1
        for slot in range(1, 4):
            for line in row[f'annotation{slot}'].splitlines():
                if not line.strip():
                    continue
                if re.match(r'\s*[1-7][.)]\s', line):
                    if not re.search(r'[:=]\s*[0-9]+\s*[.;]?\s*$', line):
                        exceptional['category_without_strict_terminal_integer'] += 1
                        exceptional['has_colon_integer_anywhere'] += bool(re.search(r':\s*\d+', line))
                        exceptional['ends_in_period'] += line.rstrip().endswith('.')
                        exceptional['ends_in_parenthesis'] += line.rstrip().endswith(')')
                        exceptional['contains_nan_or_na'] += bool(re.search(r'\b(?:nan|n/a|none)\b', line, re.I))
                        exceptional['contains_negative_integer'] += bool(re.search(r':\s*-\d+', line))
                    continue
                normalized = re.sub(r'[^a-z ]', '', line.lower())
                normalized = ' '.join(normalized.split())
                allowed = {'significant', 'insignificant', 'significant errors', 'insignificant errors',
                    'clinically significant', 'clinically insignificant',
                    'clinically significant errors', 'clinically insignificant errors'}
                headers[normalized if normalized in allowed else 'unknown_header'] += 1
    payload = {'status': 'structure_only', 'sections': dict(sections),
        'severity_header_normalizations': dict(headers), 'exceptional_numeric_grammar': dict(exceptional),
        'source_group_field_forms': dict(path_forms), 'source_manifest_sha256':
        hashlib.sha256((SOURCE / 'manifest.json').read_bytes()).hexdigest(),
        'raw_source_values_exported': False}
    os.umask(0o007)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    with (OUT / 'schema_audit.json').open('x') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write('\n')
    (OUT / 'schema_audit.json').chmod(0o660)
    print(json.dumps({'status': 'private_schema_audit_complete', 'audit_sha256':
        hashlib.sha256((OUT / 'schema_audit.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
