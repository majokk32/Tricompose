#!/usr/bin/env python3
"""Same bounded acquisition, supporting the release's single-space checksums.

V1 failed before CSV acquisition. Preserve it; this changes checksum document
syntax only, not checksum values, known filenames, clinical data or thresholds.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import time

PATH = Path(__file__).with_name('acquire_radevalx_benchmark.py')
spec = importlib.util.spec_from_file_location('immutable_radevalx_source_v1', PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SCHEMA = 'tricompose-open-radevalx-source-v2'


def released_digests(payload):
    result = {}
    for line in payload.decode('ascii').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})[ \t]+(\*?)([^/\\]+)', line)
        if match is None or match[3] not in (*base.FILES, 'LICENSE.txt') or match[3] in result:
            raise ValueError('exact_known_release_checksum_schema_required')
        result[match[3]] = match[1]
    if not set(base.FILES) <= set(result):
        raise ValueError('fixed_release_checksum_inventory_required')
    return result


def acquire(run_id, allowed):
    base.guard()
    if not allowed:
        raise ValueError('explicit_open_benchmark_acquisition_required')
    temporary, target = base.new_atomic_run(base.PROTECTED_ROOT / 'tricompose_v1_2/report_metric_sources', run_id)
    try:
        checksum_bytes = base.retrieve('SHA256SUMS.txt')
        expected = released_digests(checksum_bytes)
        base.private_bytes(temporary / 'SHA256SUMS.txt', checksum_bytes)
        schemas, hashes, sizes = {}, {}, {}
        for name in base.FILES:
            payload = base.retrieve(name)
            actual = base.hashlib.sha256(payload).hexdigest()
            if actual != expected[name]:
                raise ValueError('released_benchmark_checksum_mismatch')
            schemas[name] = base.header_only(payload)
            base.private_bytes(temporary / name, payload)
            hashes[name], sizes[name] = actual, len(payload)
        base.write_private_json(temporary / 'summary.json', {'schema_version': SCHEMA,
            'status': 'open_source_bytes_and_header_schemas_only', 'benchmark': 'radevalx-1.0.0',
            'release_url': base.REMOTE, 'license': 'CC-BY-NC-SA-4.0', 'source_domain': 'IU-Xray_not_MIMIC',
            'source_sha256': hashes, 'source_bytes': sizes, 'csv_headers': schemas,
            'prior_v1_attempt': 'checksum_spacing_refused_before_csv_acquisition',
            'checksum_source': 'same_official_release_not_independent_authenticity_attestation',
            'source_rows_or_report_bodies_inspected': False, 'model_calls': 0, 'credentials_used': False,
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        paths = [Path(__file__), PATH, base.ROOT / 'tools/score_cached_opacity_candidates.py',
                 base.WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
        base.write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'sources': {str(p.resolve()): base.sha256_file(p) for p in paths},
            'artifacts': {p.name: base.sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        base.commit_atomic_run(temporary, target)
    except BaseException:
        base.discard_atomic_run(temporary)
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
        'manifest_sha256': base.sha256_file(target / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
