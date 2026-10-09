"""Frozen offline CPU RadGraph on the sealed, fully synthetic report bank.

No source/target MIMIC, image/EHR/prompt reads, new generation, or selection.
Model-returned text stays protected. Public progress is sanitized status only.
"""
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
import time
import traceback
from unittest.mock import patch

import smoke_radgraph_xl as installed
from tricompose_v12.radgraph_bank import inventory, summarize, POLICY
from tricompose_v12.radgraph_reference_contract import METRICS
from tricompose_v12.radgraph_reference_contract_v2 import graph_metadata

WORKSPACE = installed.WORKSPACE
PROTECTED = WORKSPACE / 'artifacts/protected'
DELIVERY = PROTECTED / 'tricompose_v1_2/deliverables/first_version_12714150_001'
DELIVERY_SHA = 'a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc'
INDEX_SHA = 'f8f7c7536b6db9af75cd3f58a0275ef2f28ec0dbca4885017eee4d6728138980'
OUTPUT_ROOT = PROTECTED / 'tricompose_v1_2/radgraph_bank_runs'


def require(condition, code):
    if not condition:
        raise ValueError(code)


def private_dir(path, *, exist_ok=False):
    path.mkdir(mode=0o2770, exist_ok=exist_ok)
    path.chmod(0o2770)
    require(path.stat().st_gid in (96293, 65534), 'project_mount_group_required')


def private_csv(path, rows):
    require(bool(rows), 'nonempty_csv_inventory_required')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o660)


def source_inventory():
    source = DELIVERY / 'manifest.json'
    require(installed.sha256(source) == DELIVERY_SHA, 'sealed_synthetic_delivery_required')
    delivery = json.loads(source.read_text())
    require(delivery['schema_version'] == 'tricompose-first-version-protected-delivery-v1'
            and delivery['original_selection_changed'] is False and
            delivery['clinical_qualified'] is False, 'historical_synthetic_development_scope_required')
    index = DELIVERY / 'candidate_index.csv'
    require(delivery['artifacts']['candidate_index.csv']['sha256'] == INDEX_SHA
            and installed.sha256(index) == INDEX_SHA, 'sealed_candidate_index_required')
    with index.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    plan = inventory(rows)
    require(plan['unique_report_graphs'] == 428 and plan['same_image_pairs'] == 1440,
            'historical_bank_inventory_changed')
    sources = {}
    for row in plan['candidates']:
        path = Path(row['report_path'])
        require(not path.is_symlink() and path.resolve().is_relative_to(PROTECTED)
                and path.is_file() and path.stat().st_size <= 400000,
                'bounded_protected_synthetic_report_required')
        require(sources.setdefault(str(path), row['report_sha256']) == row['report_sha256'],
                'report_source_hash_conflict')
    for path, digest in sources.items():
        require(installed.sha256(Path(path)) == digest, 'synthetic_report_hash_mismatch')
    return rows, plan, sources


def execute(run):
    started = time.monotonic()
    rows, plan, sources = source_inventory()
    model_assets, tokenizer_assets = installed.verify_assets()
    pin_paths = [Path(__file__).resolve(), Path(installed.__file__).resolve(),
                 WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_bank.py',
                 WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
                 WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
                 DELIVERY / 'manifest.json', DELIVERY / 'candidate_index.csv',
                 installed.ASSETS / 'asset_manifest.json',
                 installed.TOKENIZER_ROOT / 'asset_manifest.json']
    pins = [{'path': str(path.relative_to(WORKSPACE)), 'sha256': installed.sha256(path)}
            for path in pin_paths]
    installed.write_json(run / 'frozen_plan.json', {**plan, 'pins': pins,
        'source_report_files': len(sources), 'selection_scores_used_for_inventory': False,
        'planned_maximum_native_forward_passes': 428})
    os.environ.update({'HF_HOME': str(installed.HUB_CACHE.parent),
        'HF_HUB_CACHE': str(installed.HUB_CACHE), 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '',
        'XDG_CACHE_HOME': str(WORKSPACE / '.cache/radgraph_v12/xdg'),
        'TMPDIR': str(WORKSPACE / '.tmp/radgraph_deploy_12714150_001'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    graph_rows, annotations = [], {}
    native_calls = 0
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        import radgraph
        from radgraph import RadGraph
        from radgraph.rewards import compute_reward
        require(importlib.metadata.version('radgraph') == '0.1.18', 'official_package_version_required')
        package = Path(radgraph.__file__).parent
        official_source = WORKSPACE / 'RadGraph/radgraph'
        for path in official_source.rglob('*.py'):
            counterpart = package / path.relative_to(official_source)
            require(counterpart.is_file() and installed.sha256(path) == installed.sha256(counterpart),
                    'installed_official_source_mismatch')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        before_load = time.monotonic()
        model = RadGraph(model_type='radgraph-xl', cuda=-1,
                         model_cache_dir=str(installed.ASSETS / 'model_cache'),
                         tokenizer_cache_dir=str(installed.HUB_CACHE))
        model.eval()
        model.requires_grad_(False)
        require(not any(p.requires_grad for p in model.parameters()), 'frozen_parameters_required')
        load_seconds = time.monotonic() - before_load
        infer_started = time.monotonic()
        # Only decode the pinned synthetic report artifacts internally here.
        # Never open the real anchor, image, EHR, facts, prompt, or a target.
        for request in plan['graphs']:
            graph_id = request['graph_id']
            record = {'graph_id': graph_id, 'report_sha256': request['report_sha256'],
                      'candidate_slots': request['candidate_slots'],
                      'status': 'failed_unavailable', 'failure_reason': None, 'metadata': None}
            try:
                data = Path(request['source_path']).read_bytes()
                require(hashlib.sha256(data).hexdigest() == request['report_sha256'],
                        'report_changed_before_inference')
                text = data.decode('utf-8')
                if not text.strip():
                    record.update(status='empty_input', failure_reason='empty_input')
                elif len(text) > 100000:
                    record['failure_reason'] = 'input_limit_exceeded_without_truncation'
                else:
                    native_calls += 1
                    with torch.inference_mode():
                        output = model([text])
                    require(set(output) == {'0'}, 'complete_native_graph_return_required')
                    metadata = graph_metadata(output['0'])
                    annotations[graph_id] = output['0']
                    record.update(status='complete', metadata=metadata)
            except Exception as error:
                # Exception details may contain text: keep only the class here.
                record['failure_reason'] = 'unavailable_' + type(error).__name__
            graph_rows.append(record)
            if len(graph_rows) % 50 == 0:
                print(json.dumps({'status': 'protected_synthetic_graph_progress',
                                  'runtime_seconds': round(time.monotonic() - infer_started, 3)}),
                      file=sys_stdout, flush=True)
        infer_seconds = time.monotonic() - infer_started
        pairs = []
        graph_lookup = {row['graph_id']: row for row in graph_rows}
        for pair in plan['pairs']:
            available = all(graph_lookup[pair[k]]['status'] == 'complete'
                            for k in ('left_graph_id', 'right_graph_id'))
            if available:
                values = compute_reward(annotations[pair['left_graph_id']],
                                        annotations[pair['right_graph_id']], 'all')
            else:
                values = (None, None, None)
            pairs.append({**pair, 'status': 'complete' if available else 'unavailable_graph',
                          'scores': dict(zip(METRICS, values))})
    overlay = summarize(plan, graph_rows, pairs)
    joined = {row['triple_candidate_id']: row for row in overlay}
    table = []
    for old in rows:
        addition = {key: value for key, value in joined[old['triple_candidate_id']].items()
                    if key.startswith('radgraph_')}
        table.append({**old, **addition})
    private_csv(run / 'candidate_score_table.csv', table)
    private_csv(run / 'report_graph_index.csv', [{**{key: row[key] for key in
        ('graph_id', 'report_sha256', 'candidate_slots', 'status', 'failure_reason')},
        'entity_count': (row['metadata'] or {}).get('entity_count'),
        'relation_count': (row['metadata'] or {}).get('relation_count')}
        for row in graph_rows])
    private_csv(run / 'same_image_agreement.csv', [{**{k: v for k, v in pair.items() if k != 'scores'},
        **pair['scores']} for pair in pairs])
    installed.write_json(run / 'graph_receipts.json', graph_rows)
    installed.write_json(run / 'native_graphs.json', {'scope': 'synthetic_only', 'graphs': annotations})
    installed.write_json(run / 'same_image_agreement.json', pairs)
    summary = {'schema_version': 'tricompose-radgraph-bank-run-v1', 'status': 'complete',
        'case_count': plan['case_count'], 'candidate_slots': len(table),
        'unique_reports_attempted': len(graph_rows), 'native_forward_passes': native_calls,
        'graph_status_counts': dict(Counter(row['status'] for row in graph_rows)),
        'candidate_slot_status_counts': dict(Counter(row['radgraph_status'] for row in overlay)),
        'same_image_pairs': len(pairs),
        'pair_status_counts': dict(Counter(row['status'] for row in pairs)),
        'identical_report_byte_pairs': sum(pair['identical_report_bytes'] for pair in pairs),
        'nonmeasurement_positive_entities': sum((row['metadata'] or {}).get(
            'nonmeasurement_observation_state_counts', {}).get('positive', 0) for row in graph_rows),
        'policy': dict(POLICY), 'frozen_parameters': True, 'network_disabled': True,
        'real_mimic_inputs_read': False, 'source_scores_preserved': True,
        'historical_static_selected_slots': sum(row['historical_static_selected'] == 'True' for row in rows),
        'initialization_seconds': load_seconds, 'inference_seconds': infer_seconds,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'device': 'cpu', 'job_id': 12714150, 'new_slurm_submissions': 0}
    installed.write_json(run / 'summary.json', summary)
    require(all(installed.sha256(WORKSPACE / pin['path']) == pin['sha256'] for pin in pins)
            and all(installed.sha256(Path(path)) == digest for path, digest in sources.items()),
            'immutable_sources_changed')
    artifacts = [{'path': path.name, 'sha256': installed.sha256(path)}
                 for path in sorted(run.iterdir()) if path.is_file() and path.name != 'worker.log']
    installed.write_json(run / 'manifest.json', {'schema_version': 'tricompose-radgraph-bank-receipt-v1',
        'policy': dict(POLICY), 'pins': pins, 'artifacts': artifacts,
        'source_report_hashes': sources, 'official_model_archive_sha256': model_assets['archive_sha256'],
        'tokenizer_revision': tokenizer_assets['revision']})
    return summary


if __name__ == '__main__':
    import sys
    sys_stdout = sys.stdout
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = None
    try:
        require('/job_12714150/' in Path('/proc/self/cgroup').read_text(),
                'approved_existing_cpu_slurm_allocation_required')
        require(re.fullmatch(r'native_pool960_12714150_[0-9]{3}', args.run_id),
                'opaque_run_id_required')
        os.umask(0o007)
        private_dir(OUTPUT_ROOT, exist_ok=True)
        run = OUTPUT_ROOT / args.run_id
        private_dir(run)
        with (run / 'worker.log').open('x') as log:
            (run / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    summary = execute(run)
                except Exception:
                    traceback.print_exc()
                    raise
        print(json.dumps({'status': summary['status'],
                          'runtime_seconds': round(summary['runtime_seconds'], 3),
                          'peak_rss_gib': round(summary['peak_rss_gib'], 3),
                          'manifest_sha256': installed.sha256(run / 'manifest.json')}))
    except Exception as error:
        if run is not None:
            installed.write_json(run / 'failure.json', {'status': 'failed',
                'error_type': type(error).__name__, 'clinical_qualified': False})
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
