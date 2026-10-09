#!/usr/bin/env python
"""Fixed real-manual CPU benchmark, Python 3.6, NOT the authored-only CLI.

Reuse frozen generic parser components; retain the old authored worker/guard.
Patient text stays inside approved Slurm and never enters outputs or stdout.
No gold or prior model prediction enters a parser request. No new rules.
"""
from __future__ import print_function

from collections import Counter
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from unittest.mock import patch

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
BASE = WORKSPACE/'artifacts/protected/tricompose_v1_2'
ENV = WORKSPACE/'runtime/venvs/chexpert-negbio-py36-12682821-v1'
LEGACY = WORKSPACE/'TriCompose-v1.2/interfaces/chexpert_negbio_legacy_worker_v1.py'
CONTRACT = BASE/'cxrgraph_gold_contracts/manual_test_12766754_001'
CONTRACT_SHA = '19f61d54c4cfda59212564e615f14518f9e3ce5cab3c71a59b9e1da9cdd56c15'
SOURCE = BASE/'reference_datasets/cxrgraph_source_12766754_001/source/manual_data/test.json'
CACHE = BASE/'cxrgraph_extraction_runs/manual_xl_12766754_002'
CACHE_SHA = 'a56743b0a3a940184faebbc7a39b0947d227d7f9ed3ffadc00592e10bed7898c'
PARENT = BASE/'manual_literal_reader_runs/manual100_12766754_001'
PARENT_SHA = '030ac27c56217db264bb09c93cba6a3ec2ae7b5322c17323a918fd89ded01a5f'
BANK = BASE/'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
OUT = BASE/'manual_negbio_runs/manual100_12766754_001'
SCHEMA = 'fixed-manual100-native-chexpert-negbio-v1'


def require(condition, code):
    if not condition:
        raise ValueError(code)


def load_legacy():
    spec = importlib.util.spec_from_file_location('_frozen_generic_negbio_components', str(LEGACY))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def private_directory(path, fresh=False):
    require(str(path).startswith(str(BASE) + '/') and not path.is_symlink(), 'protected_run_boundary_required')
    if path.exists():
        require(not fresh, 'refuse_existing_manual_negbio_run')
    else:
        path.mkdir(mode=0o2770)
        os.chmod(str(path), 0o2770)
    stat = path.stat()
    require(path.is_dir() and stat.st_gid in (96293, 65534) and stat.st_mode & 0o7777 == 0o2770,
            'private_project_directory_required')


def input_items(documents, cohort):
    require(len(documents) == len(cohort) == 100, 'all_hundred_fixed_manual_reports_required')
    result = []
    for index, pair in enumerate(zip(documents, cohort)):
        document, entry = pair
        require(entry['report_id'] == 'report_' + str(index).zfill(4)
                and entry['source_index'] == index and entry['status'] == 'validated',
                'unchanged_release_order_required')
        sentences = document['sentences']
        require(isinstance(sentences, list) and all(isinstance(s, list) for s in sentences),
                'released_source_sentences_required')
        words = [word for sentence in sentences for word in sentence]
        require(words and all(isinstance(w, str) and w and not any(c.isspace() for c in w) for w in words),
                'literal_source_words_required')
        text = ' '.join(words)
        require(len(text) <= 32768 and hashlib.sha256(text.encode('utf-8')).hexdigest() == entry['source_report_sha256'],
                'exact_same_manual_source_text_required')
        result.append({'item_id': entry['report_id'], 'report_sha256': entry['source_report_sha256'], 'text': text})
    return result


def execute(legacy, public_stdout):
    started = time.monotonic()
    deployment = legacy.metadata(legacy.DEPLOYMENT/'manifest.json', legacy.DEPLOYMENT_SHA)
    legacy.verify_pins(deployment['sources'])
    runtime = legacy.metadata(legacy.DEPLOYMENT/'runtime.json', deployment['artifacts']['runtime.json'])
    require(legacy.sha(CONTRACT/'manifest.json') == CONTRACT_SHA
            and legacy.sha(CACHE/'manifest.json') == CACHE_SHA
            and legacy.sha(PARENT/'manifest.json') == PARENT_SHA
            and legacy.sha(BANK) == BANK_SHA, 'unchanged_manual_cohort_and_previous_runs_required')
    contract = json.loads((CONTRACT/'manifest.json').read_text())
    cache = json.loads((CACHE/'manifest.json').read_text())
    require(legacy.sha(CONTRACT/'cohort.json') == contract['artifacts']['cohort.json']
            and legacy.sha(SOURCE) == cache['pins'][str(SOURCE.relative_to(WORKSPACE))],
            'same_fixed_manual_sources_required')
    paths = [Path(__file__).resolve(), LEGACY,
        WORKSPACE/'TriCompose-v1.2/slurm/81_manual_negbio_existing_cpu.sh',
        WORKSPACE/'TriCompose-v1.2/src/tricompose_v12/manual_three_reader_agreement.py',
        WORKSPACE/'TriCompose-v1.2/tests/test_manual_three_reader_agreement.py',
        WORKSPACE/'TriCompose-v1.2/tests/test_negbio_manual100_worker.py',
        WORKSPACE/'TriCompose-v1.2/tools/score_manual_three_readers.py',
        CONTRACT/'manifest.json', CONTRACT/'cohort.json', CACHE/'manifest.json',
        PARENT/'manifest.json', SOURCE, BANK, legacy.DEPLOYMENT/'manifest.json', legacy.DEPLOYMENT/'runtime.json']
    pins = dict(deployment['sources'])
    pins.update({str(p): legacy.sha(p) for p in paths})
    legacy.private_json(OUT/'frozen_plan.json', {'schema_version': SCHEMA + '-plan', 'pins': pins,
        'attempted_reports': 100, 'parser_passes_maximum': 200, 'replay_all': True,
        'native_rules_or_weights_changed': False, 'new_case_selection': False,
        'real_manual_source_in_approved_cpu_slurm': True, 'authored_only': False,
        'parser_receives_annotation_or_baseline': False,
        'native_labels_and_mention_conflict_view_separate': True,
        'source_cleaning': 'unchanged_official_loader', 'failed_is_unavailable_not_unknown': True,
        'new_training_or_download': False, 'external_api': False, 'new_slurm_submission': False,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
    private_directory(OUT/'tmp', fresh=True)
    private_directory(OUT/'cache', fresh=True)
    os.environ.update({'TMPDIR': str(OUT/'tmp'), 'XDG_CACHE_HOME': str(OUT/'cache'),
        'HF_HOME': str(OUT/'cache/huggingface'), 'TORCH_HOME': str(OUT/'cache/torch'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    cohort = json.loads((CONTRACT/'cohort.json').read_text())
    source_bytes = SOURCE.read_bytes()
    try:
        value = json.loads(source_bytes.decode('utf-8'))
        documents = value if isinstance(value, list) else [value]
    except ValueError:
        documents = [json.loads(line) for line in source_bytes.decode('utf-8').splitlines() if line.strip()]
    items = input_items(documents, cohort)
    del documents, source_bytes
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')), \
         patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled')):
        components = legacy.initialize(runtime, OUT/'tmp')
        initialized = time.monotonic()
        predictions, replays = [], []
        for index, item in enumerate(items):
            primary = legacy.parse_item(item, components)
            replay = legacy.parse_item(item, components)
            predictions.append(primary)
            replays.append({'item_id': item['item_id'], 'report_sha256': item['report_sha256'],
                'same_full_evidence': primary == replay,
                'both_complete': primary['output']['status'] == replay['output']['status'] == 'complete',
                'replay_output': replay['output']})
            if (index + 1) % 20 == 0:
                print(json.dumps({'status': 'protected_manual_negbio_progress',
                    'runtime_seconds': round(time.monotonic() - initialized, 3)}), file=public_stdout, flush=True)
    parsed = time.monotonic()
    legacy.private_json(OUT/'predictions.json', {'schema_version': SCHEMA, 'records': predictions})
    legacy.private_json(OUT/'replays.json', {'schema_version': SCHEMA, 'records': replays})
    sealed = {name: legacy.sha(OUT/name) for name in ('predictions.json', 'replays.json')}
    legacy.private_json(OUT/'prediction_freeze_receipt.json', {'sha256': sealed,
        'reference_projection_or_other_reader_states_decoded': False,
        'bundled_annotation_json_read_but_not_used_by_parser': True,
        'patient_text_or_source_identifiers_written': False})
    legacy.verify_pins(pins)
    summary = {'schema_version': SCHEMA + '-run', 'status': 'complete',
        'attempted_reports': 100, 'parser_passes_including_replay': 200,
        'primary_status_counts': dict(Counter(r['output']['status'] for r in predictions)),
        'primary_failure_reasons': dict(Counter(r['output']['failure_reason'] for r in predictions
                                              if r['output']['status'] != 'complete')),
        'full_evidence_replay_differences': sum(not r['same_full_evidence'] for r in replays),
        'replays_both_complete': sum(r['both_complete'] for r in replays),
        'initialization_seconds': initialized - started, 'parsing_seconds': parsed - initialized,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'official_classifier_called_unchanged': True, 'network_disabled': True,
        'patient_text_or_keys_written': False, 'new_slurm_submissions': 0, 'gpu_calls': 0,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False}
    legacy.private_json(OUT/'summary.json', summary)
    for directory in (OUT/'tmp', OUT/'cache'):
        for path in [directory] + list(directory.rglob('*')):
            require(not path.is_symlink() and legacy.inside(path, OUT)
                    and path.stat().st_gid in (96293, 65534), 'protected_runtime_boundary_required')
            os.chmod(str(path), 0o2770 if path.is_dir() else 0o660)
    legacy.private_json(OUT/'manifest.json', {'schema_version': SCHEMA + '-receipt', 'pins': pins,
        'artifacts': {p.name: legacy.sha(p) for p in sorted(OUT.glob('*.json'))},
        'reference_states_used_by_parser': False, 'clinical_qualified': False})
    return summary


def main():
    created = False
    try:
        require(len(sys.argv) == 1, 'fixed_manual_negbio_scope_takes_no_arguments')
        require(os.environ.get('SLURM_JOB_ID') == '12766754'
                and '/job_12766754/' in Path('/proc/self/cgroup').read_text()
                and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),
                'approved_existing_cpu_slurm_required')
        require(sys.version_info[:3] == (3, 6, 7) and Path(sys.prefix).resolve() == ENV.resolve()
                and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'exact_frozen_legacy_python_required')
        os.umask(0o007)
        private_directory(OUT.parent)
        private_directory(OUT, fresh=True)
        created = True
        legacy = load_legacy()
        with (OUT/'worker.log').open('x') as log:
            os.chmod(str(OUT/'worker.log'), 0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = execute(legacy, sys.__stdout__)
        print(json.dumps({'status': 'protected_manual_negbio_complete',
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'manifest_sha256': legacy.sha(OUT/'manifest.json')}))
        return 0
    except Exception as error:
        if created:
            with (OUT/'failure.json').open('x') as stream:
                json.dump({'status': 'failed', 'failure_type': type(error).__name__}, stream)
            os.chmod(str(OUT/'failure.json'), 0o660)
        print(json.dumps({'status': 'protected_manual_negbio_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
