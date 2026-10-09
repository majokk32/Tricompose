"""Protected CXRGraph manual-test contract intake inside an approved Slurm job.

Clinical JSON is consumed internally only. Public stdout never contains source
identifiers, reports, native annotations or exception payloads. No model calls.
"""
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json
from tricompose_v12.cxrgraph_gold_adapter import normalize_manual_document, validate_adapter

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
INTAKE = BASE / 'reference_datasets/cxrgraph_source_12766754_001'
INTAKE_SHA = '6b50154bf091e03bd2c6bb9a2d592dce31bb06a9156e3de6ca853626d36eece7'
ROOT = BASE / 'cxrgraph_gold_contracts'
SCHEMA = 'cxrgraph-manual-test-contract-v1'


def read_documents(payload):
    require(isinstance(payload, bytes) and 0 < len(payload) <= 32_000_000,
            'bounded_manual_file_required')
    text = payload.decode('utf-8')
    try:
        value = json.loads(text)
        documents = value if isinstance(value, list) else [value]
        storage = 'json_array' if isinstance(value, list) else 'single_json_document'
    except json.JSONDecodeError:
        documents = [json.loads(line) for line in text.splitlines() if line.strip()]
        storage = 'json_lines'
    require(1 <= len(documents) <= 1024 and all(isinstance(d, dict) for d in documents),
            'bounded_document_inventory_required')
    require(all(isinstance(d.get('doc_key'), str) for d in documents),
            'internal_source_key_required')
    require(len({d['doc_key'] for d in documents}) == len(documents),
            'unique_internal_source_keys_required')
    return documents, storage


def split_membership(test, mimic, chexpert):
    keys = [{d['doc_key'] for d in docs} for docs in (test, mimic, chexpert)]
    require(not keys[1] & keys[2] and keys[1] | keys[2] == keys[0],
            'exact_disjoint_manual_test_partition_required')
    combined = {d['doc_key']: d for d in mimic + chexpert}
    require(all(combined[d['doc_key']] == d for d in test),
            'identical_split_documents_required')
    return {i: ('mimic' if d['doc_key'] in keys[1] else 'chexpert')
            for i, d in enumerate(test)}


def normalize_documents(documents, groups):
    receipts, adapters = [], []
    # This order is fixed by the released test file, not by scores or easy cases.
    for index, document in enumerate(documents):
        report_id = f'report_{index:04d}'
        row = {'report_id': report_id, 'source_index': index,
               'source_domain': groups[index], 'status': 'schema_unavailable',
               'failure_type': None, 'source_report_sha256': None,
               'adapter_sha256': None, 'native_coverage': None}
        try:
            sentences = document.get('sentences')
            require(isinstance(sentences, list) and all(isinstance(s, list) for s in sentences),
                    'released_sentence_inventory_required')
            words = [w for sentence in sentences for w in sentence]
            require(all(isinstance(w, str) for w in words), 'released_token_strings_required')
            report_hash = hashlib.sha256(' '.join(words).encode('utf-8')).hexdigest()
            adapter = normalize_manual_document(document, report_id=report_id,
                source_report_sha256=report_hash, origin='declared_human_annotation')
            validate_adapter(adapter)
            row.update(status='validated', source_report_sha256=report_hash,
                       adapter_sha256=adapter['adapter_sha256'],
                       native_coverage=adapter['native_coverage'])
            adapters.append(adapter)
        except Exception as error:
            # An exception message can carry source contents; only its class is retained.
            row['failure_type'] = type(error).__name__
        receipts.append(row)
    return receipts, adapters


def execute(run):
    started = time.monotonic()
    require(sha256(INTAKE / 'intake_manifest.json') == INTAKE_SHA,
            'pinned_upload_intake_required')
    intake = json.loads((INTAKE / 'intake_manifest.json').read_text())
    files = {row['path']: row for row in intake['files']}
    source_names = ('test.json', 'test_mimic.json', 'test_chexpert.json')
    source_paths = [INTAKE / 'source/manual_data' / name for name in source_names]
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in (
        Path(__file__).resolve(), INTAKE / 'intake_manifest.json', *source_paths,
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/cxrgraph_gold_adapter.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_contract.py',
        BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json')}
    write_json(run / 'frozen_plan.json', {'schema_version': SCHEMA + '-plan',
        'pins': pins, 'scope': 'all_released_manual_test_documents_in_file_order',
        'user_provided_authorized_source': True, 'model_calls': 0,
        'training_overlap_status': 'unresolved_not_independent_qualification'})
    loaded, storage = [], {}
    for name, path in zip(source_names, source_paths):
        row = files['source/manual_data/' + name]
        require(path.resolve().is_relative_to(INTAKE) and not path.is_symlink()
                and row['embedded_sha256_verified'] and sha256(path) == row['sha256'],
                'pinned_manual_source_required')
        docs, fmt = read_documents(path.read_bytes())
        loaded.append(docs)
        storage[name] = fmt
    test, mimic, chexpert = loaded
    require((len(test), len(mimic), len(chexpert)) == (100, 50, 50),
            'published_manual_test_inventory_required')
    groups = split_membership(test, mimic, chexpert)
    receipts, adapters = normalize_documents(test, groups)
    write_json(run / 'cohort.json', receipts)
    write_json(run / 'gold_adapters.json', adapters)
    status_counts = dict(Counter(r['status'] for r in receipts))
    summary = {'schema_version': SCHEMA, 'status': 'manual_test_contract_complete',
        'attempted_reports': len(test), 'status_counts': status_counts,
        'domain_counts': dict(Counter(groups.values())), 'storage_formats': storage,
        'test_partition_disjoint_and_exact': True,
        'all_source_reports_retained_as_attempted': True,
        'source_order_fixed': True, 'source_identifiers_exported': False,
        'report_text_exported': False,
        'input_representation': 'released_tokens_joined_with_single_spaces',
        'annotation_process': 'joint_manual_not_two_independent_readers',
        'native_schema_validation_on_actual_manual_data': status_counts.get('validated', 0) == len(test),
        'checkpoint_training_overlap': None, 'independent_heldout_qualified': False,
        'model_calls': 0, 'extraction_accuracy': None,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    require(all(sha256(WORKSPACE / path) == value for path, value in pins.items()),
            'input_or_consumed_code_changed')
    write_json(run / 'summary.json', summary)
    artifacts = {p.name: sha256(p) for p in sorted(run.glob('*.json'))}
    write_json(run / 'manifest.json', {'schema_version': SCHEMA + '-receipt',
        'pins': pins, 'artifacts': artifacts, 'model_calls': 0})
    return summary


def main():
    run = None
    try:
        require(len(sys.argv) == 1, 'fixed_scope_worker_takes_no_arguments')
        job = os.environ.get('SLURM_JOB_ID', '')
        require_slurm(Path('/proc/self/cgroup').read_text(), job)
        require(job == '12766754', 'approved_existing_cpu_allocation_required')
        os.umask(0o007)
        private_dir(ROOT)
        run = ROOT / ('manual_test_' + job + '_001')
        private_dir(run, fresh=True)
        with (run / 'worker.log').open('x') as log:
            (run / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = execute(run)
        print(json.dumps({'status': summary['status'],
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'manifest_sha256': sha256(run / 'manifest.json')}))
        return 0
    except Exception as error:
        if run is not None and run.is_dir() and not (run / 'manifest.json').exists():
            write_json(run / 'failure.json', {'status': 'failed', 'failure_type': type(error).__name__})
        print(json.dumps({'status': 'protected_manual_test_contract_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
