#!/usr/bin/env python3
"""Bounded open IU-Xray benchmark acquisition, not inference or text inspection.

Only known release URLs. Store untouched source bytes protected; inspect CSV
headers only. Never print source rows, report text, original IDs or exceptions.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, new_atomic_run, commit_atomic_run,
                       discard_atomic_run, write_private_json, sha256_file)
from score_cached_opacity_candidates import guard

SCHEMA = 'tricompose-open-radevalx-source-v1'
REMOTE = 'https://physionet.org/files/rad-eval-x/1.0.0/'
FILES = ('RadEval_clinically_significant_errors.csv',
         'RadEval-clinically_insignificant_errors.csv', 'metrcis_scores_m2tr.csv')
CAP = 256 * 1024


def retrieve(name):
    if name not in (*FILES, 'SHA256SUMS.txt'):
        raise ValueError('fixed_release_filename_required')
    url = REMOTE + name
    request = urllib.request.Request(url, headers={'User-Agent': 'TriCompose-benchmark-metadata/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError('exact_open_release_response_required')
        length = response.headers.get('Content-Length')
        if length is not None and int(length) > CAP:
            raise ValueError('bounded_open_benchmark_required')
        payload = response.read(CAP + 1)
    if not payload or len(payload) > CAP:
        raise ValueError('bounded_nonempty_release_file_required')
    return payload


def released_digests(payload):
    result = {}
    for line in payload.decode('ascii').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (\*?)([^/\\]+)', line)
        if match is None or match[3] in result:
            raise ValueError('exact_release_checksum_schema_required')
        result[match[3]] = match[1]
    if not set(FILES) <= set(result) or set(result) - set(FILES) - {'LICENSE.txt'}:
        raise ValueError('fixed_release_checksum_inventory_required')
    return result


def header_only(payload):
    first = payload.split(b'\n', 1)[0]
    if not first or len(first) > 8192:
        raise ValueError('bounded_csv_header_required')
    fields = next(csv.reader(io.StringIO(first.decode('utf-8-sig'))))
    if not 1 <= len(fields) <= 48 or len(fields) != len(set(fields)) or any(
            not re.fullmatch(r'[A-Za-z0-9 _().:/-]{0,96}', field) for field in fields):
        raise ValueError('bounded_column_name_schema_required')
    return fields


def private_bytes(path, payload):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(path, flags, 0o660), 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o660)


def acquire(run_id, allowed):
    guard()
    if not allowed:
        raise ValueError('explicit_open_benchmark_acquisition_required')
    temporary, target = new_atomic_run(PROTECTED_ROOT / 'tricompose_v1_2/report_metric_sources', run_id)
    try:
        checksum_bytes = retrieve('SHA256SUMS.txt')
        expected = released_digests(checksum_bytes)
        private_bytes(temporary / 'SHA256SUMS.txt', checksum_bytes)
        schemas, hashes, sizes = {}, {}, {}
        for name in FILES:
            payload = retrieve(name)
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected[name]:
                raise ValueError('released_benchmark_checksum_mismatch')
            schemas[name] = header_only(payload)
            private_bytes(temporary / name, payload)
            hashes[name], sizes[name] = actual, len(payload)
        summary = {'schema_version': SCHEMA, 'status': 'open_source_bytes_and_header_schemas_only',
            'benchmark': 'radevalx-1.0.0', 'release_url': REMOTE,
            'license': 'CC-BY-NC-SA-4.0', 'source_domain': 'IU-Xray_not_MIMIC',
            'source_sha256': hashes, 'source_bytes': sizes, 'csv_headers': schemas,
            'checksum_source': 'same_official_release_not_independent_authenticity_attestation',
            'source_rows_or_report_bodies_inspected': False, 'model_calls': 0,
            'credentials_used': False, 'clinical_qualified': False,
            'selection_changed': False, 'regeneration_authorized': False}
        write_private_json(temporary / 'summary.json', summary)
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'program_sha256': sha256_file(Path(__file__)),
            'sources': {str(WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py'):
                sha256_file(WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py'),
                str(ROOT / 'tools/score_cached_opacity_candidates.py'): sha256_file(ROOT / 'tools/score_cached_opacity_candidates.py')},
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, sum(sizes.values())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run-id', required=True)
    cli.add_argument('--allow-open-benchmark-acquisition', action='store_true')
    args = cli.parse_args()
    os.umask(0o007)
    started = time.monotonic()
    try:
        target, size = acquire(args.run_id, args.allow_open_benchmark_acquisition)
    except Exception as error:
        print(json.dumps({'status': 'open_benchmark_acquisition_failed_closed',
                          'error_type': type(error).__name__, 'source_text_exposed': False}))
        return 2
    print(json.dumps({'status': 'open_benchmark_acquisition_complete', 'source_bytes': size,
        'elapsed_seconds': round(time.monotonic()-started, 3), 'model_calls': 0,
        'manifest_sha256': sha256_file(target / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
