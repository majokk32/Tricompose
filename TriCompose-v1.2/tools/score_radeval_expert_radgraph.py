"""Approved private expert-reference benchmark on frozen offline CPU RadGraph."""
import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import socket
import sys
import time
from unittest.mock import patch

import smoke_radgraph_xl as installed
from tricompose_v12.radeval_expert import inventory, evaluate, outcomes, POLICY
from tricompose_v12.radgraph_reference_contract import METRICS
from tricompose_v12.radgraph_reference_contract_v2 import graph_metadata

WORKSPACE = installed.WORKSPACE
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'report_metric_sources/radeval_expert_source_12714150_001'
SOURCE_MANIFEST_SHA = 'be99b29497d8bf75d52cdd7ca1cc353814bbc18a9ef5c9c1675f03b5752e9ff4'
CONTRACT = BASE / 'report_metric_sources/radeval_expert_contract_12714150_003/inventory.json'
CONTRACT_SHA = '29c076bc24fd4c628703cbe1a1ea5049cf090660ff06a2c46e09e59e50b49365'
ROOT = BASE / 'radeval_expert_radgraph_runs'


def require(condition, code):
    if not condition:
        raise ValueError(code)


def private_dir(path, exist_ok=False):
    path.mkdir(mode=0o2770, exist_ok=exist_ok)
    path.chmod(0o2770)
    require(path.stat().st_gid in (96293, 65534), 'project_group_mount_required')


def private_csv(path, rows):
    require(bool(rows), 'nonempty_opaque_inventory_required')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o660)


def source_inventory():
    require(installed.sha256(SOURCE / 'manifest.json') == SOURCE_MANIFEST_SHA,
            'approved_pinned_source_manifest_required')
    source = json.loads((SOURCE / 'manifest.json').read_text())
    require(source['user_confirmed_authorization'] is True and source['raw_payload_protected'] is True,
            'authorized_protected_source_required')
    path = SOURCE / source['file']
    require(path.resolve().is_relative_to(SOURCE) and not path.is_symlink()
            and installed.sha256(path) == source['sha256'], 'pinned_source_payload_required')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    plan = inventory(rows)
    require(installed.sha256(CONTRACT) == CONTRACT_SHA and
            plan == json.loads(CONTRACT.read_text()), 'accepted_format_preflight_required')
    texts = {}
    for row in rows:
        for key in ('ground_truth', 'prediction1', 'prediction2', 'prediction3'):
            text = row[key]
            digest = hashlib.sha256(text.encode()).hexdigest()
            require(texts.setdefault(digest, text) == text, 'text_digest_collision')
    require(len(texts) == len(plan['graphs']) == 762 and len(plan['records']) == 624,
            'frozen_all_attempted_author_inventory_required')
    return plan, texts, source, path


def execute(run, public_stdout):
    started = time.monotonic()
    plan, texts, source, source_path = source_inventory()
    model_assets, tokenizer_assets = installed.verify_assets()
    pin_paths = [Path(__file__).resolve(), Path(installed.__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/report_metric_alignment.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_radeval_expert.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/59_radeval_expert_existing_cpu.sh',
        WORKSPACE / 'docs/radeval_expert_radgraph_protocol.md',
        SOURCE / 'manifest.json', source_path, CONTRACT,
        installed.ASSETS / 'asset_manifest.json', installed.TOKENIZER_ROOT / 'asset_manifest.json',
        BASE / 'radgraph_bank_runs/native_pool960_12714150_001/candidate_score_table.csv',
        BASE / 'deliverables/first_version_12714150_001/candidate_index.csv']
    pins = [{'path': str(p.relative_to(WORKSPACE)), 'sha256': installed.sha256(p)} for p in pin_paths]
    installed.write_json(run / 'frozen_plan.json', {'inventory': plan, 'pins': pins,
        'bootstrap_resamples': 1000, 'seed': 0,
        'planned_maximum_native_forward_passes': 762,
        'outcome_or_score_based_source_selection': False})
    os.environ.update({'HF_HOME': str(installed.HUB_CACHE.parent),
        'HF_HUB_CACHE': str(installed.HUB_CACHE), 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '',
        'XDG_CACHE_HOME': str(WORKSPACE / '.cache/radgraph_v12/xdg'),
        'TMPDIR': str(WORKSPACE / '.tmp/radgraph_deploy_12714150_001'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    graph_rows, native = [], {}
    native_calls = 0
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        import radgraph
        from radgraph import RadGraph
        from radgraph.rewards import compute_reward
        require(importlib.metadata.version('radgraph') == '0.1.18', 'official_package_version_required')
        package = Path(radgraph.__file__).parent
        matched = 0
        for path in (WORKSPACE / 'RadGraph/radgraph').rglob('*.py'):
            counterpart = package / path.relative_to(WORKSPACE / 'RadGraph/radgraph')
            require(counterpart.is_file() and installed.sha256(path) == installed.sha256(counterpart),
                    'official_installed_source_mismatch')
            matched += 1
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        load_started = time.monotonic()
        model = RadGraph(model_type='radgraph-xl', cuda=-1,
            model_cache_dir=str(installed.ASSETS / 'model_cache'),
            tokenizer_cache_dir=str(installed.HUB_CACHE))
        model.eval()
        model.requires_grad_(False)
        require(not any(p.requires_grad for p in model.parameters()), 'frozen_parameters_required')
        load_seconds = time.monotonic() - load_started
        infer_started = time.monotonic()
        for request in plan['graphs']:
            record = {**request, 'status': 'failed_unavailable', 'failure_reason': None, 'metadata': None}
            text = texts[request['text_sha256']]
            try:
                if not text.strip():
                    record.update(status='empty_input', failure_reason='empty_input')
                else:
                    native_calls += 1
                    with torch.inference_mode():
                        output = model([text])
                    require(set(output) == {'0'}, 'complete_native_graph_required')
                    metadata = graph_metadata(output['0'])
                    native[request['graph_id']] = output['0']
                    record.update(status='complete', metadata=metadata)
            except Exception as error:
                # No free-form exception: it can contain a real report.
                record['failure_reason'] = 'unavailable_' + type(error).__name__
            graph_rows.append(record)
            if len(graph_rows) % 100 == 0:
                print(json.dumps({'status': 'protected_expert_metric_progress',
                    'runtime_seconds': round(time.monotonic() - infer_started, 3)}),
                    file=public_stdout, flush=True)
        infer_seconds = time.monotonic() - infer_started
        by_id = {g['graph_id']: g for g in graph_rows}
        scores = []
        for pair in plan['records']:
            if not pair['input_nonempty']:
                status, values = 'empty_input', (None, None, None)
            elif all(by_id[pair[k]]['status'] == 'complete' for k in ('reference_graph_id', 'hypothesis_graph_id')):
                status = 'complete'
                values = compute_reward(native[pair['hypothesis_graph_id']], native[pair['reference_graph_id']], 'all')
            else:
                status, values = 'unavailable_graph', (None, None, None)
            scores.append({'item_id': pair['item_id'], 'status': status,
                           'scores': dict(zip(METRICS, values))})
    statistics_started = time.monotonic()
    evaluation = evaluate(plan, scores, resamples=1000, seed=0)
    statistics_seconds = time.monotonic() - statistics_started
    table = []
    for pair, score in zip(plan['records'], scores):
        table.append({**{k: pair[k] for k in ('item_id', 'source_group_id', 'source_id', 'section_id',
            'section_name', 'candidate_slot', 'reference_sha256', 'hypothesis_sha256', 'released_reader_cells',
            'complete_reader_cells')}, 'model_status': score['status'], **score['scores'], **outcomes(pair['errors'])})
    private_csv(run / 'score_table.csv', table)
    installed.write_json(run / 'scores.json', scores)
    installed.write_json(run / 'evaluation.json', evaluation)
    installed.write_json(run / 'graph_receipts.json', graph_rows)
    installed.write_json(run / 'native_graphs.json', {'scope': 'authorized_real_reference_benchmark_only', 'graphs': native})
    summary = {'schema_version': 'radeval-expert-radgraph-run-v1', 'status': 'complete',
        'all_attempted_pairs': len(scores), 'source_rows': plan['source_rows'],
        'source_keys': plan['source_keys'], 'source_groups': plan['source_groups'],
        'group_key_strategy_counts': plan['group_key_strategy_counts'],
        'expert_readers_in_release': plan['reader_count'],
        'released_reader_cell_widths': dict(Counter(p['released_reader_cells'] for p in plan['records'])),
        'primary_reference_available': sum(outcomes(p['errors'])['clinically_significant_total'] is not None for p in plan['records']),
        'all_error_reference_available': sum(outcomes(p['errors'])['all_errors_total'] is not None for p in plan['records']),
        'reader_cell_status_counts': plan['reader_cell_status_counts'],
        'annotation_issue_counts': plan['annotation_issue_counts'],
        'graph_status_counts': dict(Counter(g['status'] for g in graph_rows)),
        'pair_status_counts': dict(Counter(s['status'] for s in scores)),
        'zero_entity_graphs': sum(g['status'] == 'complete' and g['metadata']['entity_count'] == 0 for g in graph_rows),
        'native_forward_passes': native_calls, 'installed_source_files_verified': matched,
        'frozen_parameters': True, 'network_disabled_during_scoring': True,
        'authorized_real_reference_inputs_read_internally': True,
        'source_ehr_or_images_read': False, 'public_report_or_source_id_export': False,
        'historical_scores_and_choices_preserved': True, 'policy': dict(POLICY),
        'device': 'cpu', 'job_id': 12714150, 'new_slurm_submissions': 0,
        'initialization_seconds': load_seconds, 'inference_seconds': infer_seconds,
        'statistics_seconds': statistics_seconds, 'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    installed.write_json(run / 'summary.json', summary)
    require(all(installed.sha256(WORKSPACE / p['path']) == p['sha256'] for p in pins),
            'immutable_source_or_prior_scores_changed')
    artifacts = [{'path': p.name, 'sha256': installed.sha256(p)}
                 for p in sorted(run.iterdir()) if p.is_file() and p.name != 'worker.log']
    installed.write_json(run / 'manifest.json', {'schema_version': 'radeval-expert-radgraph-receipt-v1',
        'policy': dict(POLICY), 'pins': pins, 'artifacts': artifacts,
        'official_model_archive_sha256': model_assets['archive_sha256'],
        'tokenizer_revision': tokenizer_assets['revision'],
        'dataset_revision': source['revision'], 'dataset_sha256': source['sha256']})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    run = None
    try:
        args = parser.parse_args()
        require('/job_12714150/' in Path('/proc/self/cgroup').read_text(), 'approved_existing_cpu_slurm_required')
        require(re.fullmatch(r'expert_radgraph_12714150_[0-9]{3}', args.run_id), 'opaque_run_id_required')
        os.umask(0o007)
        private_dir(ROOT, exist_ok=True)
        run = ROOT / args.run_id
        private_dir(run)
        public_stdout = sys.stdout
        with (run / 'worker.log').open('x') as log:
            (run / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = execute(run, public_stdout)
        print(json.dumps({'status': 'protected_expert_benchmark_complete',
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'manifest_sha256': installed.sha256(run / 'manifest.json')}))
    except Exception as error:
        if run is not None:
            installed.write_json(run / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                                       'clinical_qualified': False})
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
