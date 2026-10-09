"""Offline frozen XL extraction versus the fixed protected CXRGraph manual test.

Development diagnostic only: checkpoint training membership is unresolved.
Not official full CXRGraph scoring, report factuality, or image/EHR truth.
"""
from collections import Counter
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from unittest.mock import patch

import smoke_radgraph_xl as installed
from prepare_ratescore_assets import private_dir, require, require_slurm, sha256, write_json
from prepare_cxrgraph_manual_cohort import INTAKE, read_documents
from tricompose_v12.cxrgraph_gold_adapter import validate_adapter
from tricompose_v12.entity_gold_contract import normalize_graph
from tricompose_v12.entity_gold_character_contract import (
    compare_character_annotation, aggregate_character_comparisons,
)

WORKSPACE = installed.WORKSPACE
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
CONTRACT = BASE / 'cxrgraph_gold_contracts/manual_test_12766754_001'
CONTRACT_SHA = '19f61d54c4cfda59212564e615f14518f9e3ce5cab3c71a59b9e1da9cdd56c15'
ROOT = BASE / 'cxrgraph_extraction_runs'


def aggregates(comparisons, cohort):
    return aggregate_character_comparisons(comparisons, cohort)


def load_contract():
    require(sha256(CONTRACT / 'manifest.json') == CONTRACT_SHA,
            'accepted_fixed_manual_contract_required')
    manifest = json.loads((CONTRACT / 'manifest.json').read_text())
    for name, digest in manifest['artifacts'].items():
        require(name in ('frozen_plan.json', 'cohort.json', 'gold_adapters.json', 'summary.json')
                and sha256(CONTRACT / name) == digest, 'manual_contract_integrity_required')
    for path, digest in manifest['pins'].items():
        require(sha256(WORKSPACE / path) == digest, 'manual_contract_source_binding_required')
    cohort = json.loads((CONTRACT / 'cohort.json').read_text())
    adapters = json.loads((CONTRACT / 'gold_adapters.json').read_text())
    source = INTAKE / 'source/manual_data/test.json'
    documents, _ = read_documents(source.read_bytes())
    require(len(cohort) == len(adapters) == len(documents) == 100
            and all(r['status'] == 'validated' and r['source_index'] == i
                    and r['report_id'] == f'report_{i:04d}' for i, r in enumerate(cohort)),
            'all_hundred_fixed_valid_manual_reports_required')
    by_id = {}
    for adapter in adapters:
        validate_adapter(adapter)
        record = adapter['common_label_record']
        require(record['report_id'] not in by_id, 'unique_manual_adapter_required')
        by_id[record['report_id']] = adapter
    require(set(by_id) == {r['report_id'] for r in cohort}, 'complete_gold_inventory_required')
    return cohort, documents, by_id, source


def execute(run, public_stdout):
    started = time.monotonic()
    cohort, documents, gold, source = load_contract()
    model_assets, tokenizer_assets = installed.verify_assets()
    pin_paths = [Path(__file__).resolve(), Path(installed.__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_cxrgraph_manual_cohort.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/cxrgraph_gold_adapter.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_entity_gold_character_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_character_contract.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/78_cxrgraph_character_existing_cpu.sh',
        CONTRACT / 'manifest.json', source,
        installed.ASSETS / 'asset_manifest.json', installed.TOKENIZER_ROOT / 'asset_manifest.json',
        BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in pin_paths}
    write_json(run / 'frozen_plan.json', {
        'schema_version': 'cxrgraph-manual-xl-character-extraction-plan-v1', 'pins': pins,
        'attempted_report_ids': [r['report_id'] for r in cohort],
        'maximum_native_forward_passes': 100, 'device': 'cpu', 'model_threads': 2,
        'selection_by_gold_or_score': False,
        'alignment': 'lossless_exact_source_character_boundaries_no_snapping',
        'training_overlap_status': 'unresolved_development_diagnostic_only',
        'input_representation': 'released_native_tokens_joined_with_single_spaces',
        'metric_scope': 'four_common_entity_labels_three_direct_relation_types',
        'extra_native_attributes_scored': False, 'source_images_or_ehr_consumed': False,
        'model_archive_sha256': model_assets['archive_sha256'],
        'tokenizer_revision': tokenizer_assets['revision'], 'historical_bank_mutated': False})
    private_dir(run / 'tmp', fresh=True)
    private_dir(run / 'cache', fresh=True)
    os.environ.update({'HF_HOME': str(installed.HUB_CACHE.parent),
        'HF_HUB_CACHE': str(installed.HUB_CACHE), 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '',
        'XDG_CACHE_HOME': str(run / 'cache'),
        'TMPDIR': str(run / 'tmp'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    receipts, comparisons, predictions, native_graphs = [], [], [], {}
    calls = 0
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        import radgraph
        from radgraph import RadGraph
        require(importlib.metadata.version('radgraph') == '0.1.18', 'pinned_radgraph_package_required')
        package = Path(radgraph.__file__).parent
        matched = 0
        for path in (WORKSPACE / 'RadGraph/radgraph').rglob('*.py'):
            counterpart = package / path.relative_to(WORKSPACE / 'RadGraph/radgraph')
            require(counterpart.is_file() and sha256(path) == sha256(counterpart),
                    'installed_official_source_mismatch')
            matched += 1
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
        before_inference = time.monotonic()
        for entry, document in zip(cohort, documents):
            report_id = entry['report_id']
            words = [w for sentence in document['sentences'] for w in sentence]
            text = ' '.join(words)
            row = {'report_id': report_id, 'source_domain': entry['source_domain'],
                   'status': 'prediction_unavailable', 'failure_type': None,
                   'prediction_record_sha256': None, 'comparison_sha256': None}
            try:
                calls += 1
                with torch.inference_mode():
                    output = model([text])
                require(isinstance(output, dict) and set(output) == {'0'},
                        'complete_single_native_graph_required')
                graph = output['0']
                prediction = normalize_graph(graph, native_schema='radgraph_xl',
                    report_id=report_id, source_report_sha256=entry['source_report_sha256'],
                    origin='declared_frozen_prediction')
                comparison = compare_character_annotation(gold[report_id], prediction,
                    source_words=words, prediction_words=graph['text'].split())
                row.update(status='complete',
                    prediction_record_sha256=prediction['record_sha256'],
                    comparison_sha256=comparison['comparison_sha256'])
                predictions.append(prediction)
                comparisons.append(comparison)
                native_graphs[report_id] = graph
            except Exception as error:
                row['failure_type'] = type(error).__name__
            receipts.append(row)
            if len(receipts) % 20 == 0:
                print(json.dumps({'status': 'protected_manual_extraction_progress',
                    'runtime_seconds': round(time.monotonic() - before_inference, 3)}),
                    file=public_stdout, flush=True)
        inference_seconds = time.monotonic() - before_inference
    evaluation = aggregates(comparisons, cohort)
    write_json(run / 'prediction_receipts.json', receipts)
    write_json(run / 'prediction_records.json', predictions)
    write_json(run / 'comparisons.json', comparisons)
    write_json(run / 'evaluation.json', evaluation)
    write_json(run / 'native_graphs.json', {
        'scope': 'authorized_manual_test_internal_predictions_do_not_display', 'graphs': native_graphs})
    summary = {'schema_version': 'cxrgraph-manual-xl-character-extraction-run-v1',
        'status': 'development_character_extraction_diagnostic_complete',
        'attempted_reports': 100, 'prediction_status_counts': dict(Counter(r['status'] for r in receipts)),
        'native_forward_passes': calls, 'domain_counts': dict(Counter(r['source_domain'] for r in receipts)),
        'frozen_parameters': True, 'network_disabled': True, 'device': 'cpu',
        'installed_source_files_verified': matched, 'checkpoint_training_overlap': None,
        'independent_heldout_qualified': False, 'official_full_cxrgraph_metric': False,
        'clinical_score': None, 'image_or_ehr_truth_validated': False,
        'new_slurm_submissions': 0, 'historical_scores_and_choices_preserved': True,
        'initialization_seconds': load_seconds, 'inference_seconds': inference_seconds,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    require(all(sha256(WORKSPACE / path) == value for path, value in pins.items()),
            'input_or_consumed_worker_changed')
    write_json(run / 'summary.json', summary)
    write_json(run / 'manifest.json', {'schema_version': 'cxrgraph-manual-xl-character-extraction-receipt-v1',
        'pins': pins, 'artifacts': {p.name: sha256(p) for p in sorted(run.glob('*.json'))}})
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
        target = ROOT / ('manual_xl_' + job + '_002')
        private_dir(target, fresh=True)
        run = target
        with (run / 'worker.log').open('x') as log:
            (run / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = execute(run, sys.__stdout__)
        print(json.dumps({'status': summary['status'],
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'manifest_sha256': sha256(run / 'manifest.json')}))
        return 0
    except Exception as error:
        if run is not None:
            write_json(run / 'failure.json', {'status': 'failed', 'failure_type': type(error).__name__})
        print(json.dumps({'status': 'protected_manual_extraction_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
