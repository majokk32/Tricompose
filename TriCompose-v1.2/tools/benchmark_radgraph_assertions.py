"""Frozen CPU XL on the existing authored assertion challenge, never patients.

All 56 texts and 80 designated checks stay fixed. Predictions are fsynced before
opening authored labels or baseline predictions. No fitting or winner changes.
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

import report_assertion_challenge as challenge
import smoke_radgraph_xl as installed
from prepare_ratescore_assets import private_dir, require, require_slurm, sha256, write_json
from tricompose_v12.radgraph_assertion_readout import (
    PATTERNS, VERSION, readout, assertion_reliability,
)

WORKSPACE = installed.WORKSPACE
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
BANK = BASE / 'benchmarks/authored_assertions_20261002_001'
BANK_SHA = '5f76ca3d7f710f4fdf722f87f27afc6548c839f1c4d77e56a6502ec4e3583757'
ROOT = BASE / 'radgraph_assertion_runs'
BASELINES = {
    'chexbert': (BASE / 'verification_runs/authored_assertions_chexbert_12580901',
        '2fde45f8cef8993e7c6f932448c761557a61c4af8c258109e951ba0d36b348c1'),
    'qwen_span_v2': (BASE / 'verification_runs/authored_span_v2_12646689',
        '2d104e65a350bdca9970d8e4b080e0b5a9c39b93a3572a4f3a30897e94fb82eb'),
}
MANUAL = BASE / 'cxrgraph_extraction_runs/manual_xl_12766754_002'
MANUAL_SHA = 'a56743b0a3a940184faebbc7a39b0947d227d7f9ed3ffadc00592e10bed7898c'


def compatible(records):
    return [{**r, 'finding_states': dict.fromkeys(challenge.FINDINGS, 'unknown')}
            if r['status'] == 'failed_unavailable' and r['finding_states'] is None else r
            for r in records]


def compare_baseline(references, records, baseline_records):
    left = {r['item_id']: r for r in records}
    right = {r['item_id']: r for r in baseline_records}
    require(set(left) == set(right), 'same_input_baseline_inventory_required')
    result = Counter({'both_correct': 0, 'radgraph_only_correct': 0,
                      'baseline_only_correct': 0, 'both_wrong': 0})
    for ref in references:
        for finding in ref['evaluation_findings']:
            a, b = left[ref['item_id']], right[ref['item_id']]
            x = a['status'] == 'complete' and a['finding_states'][finding] == ref['expected_states'][finding]
            y = b['status'] == 'complete' and b['finding_states'][finding] == ref['expected_states'][finding]
            key = 'both_correct' if x and y else 'radgraph_only_correct' if x else 'baseline_only_correct' if y else 'both_wrong'
            result[key] += 1
    require(sum(result.values()) == 80, 'all_designated_checks_required')
    return dict(result)


def execute(run, public_stdout):
    started = time.monotonic()
    require(sha256(BANK / 'manifest.json') == BANK_SHA, 'unchanged_fixed_authored_challenge_required')
    require(tuple(PATTERNS) == challenge.FINDINGS, 'fixed_four_head_inventory_required')
    resolver, texts, source = challenge.load_inputs(BANK)
    assets, tokenizer = installed.verify_assets()
    paths = [Path(__file__).resolve(), Path(installed.__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_assertion_readout.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_character_contract.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_radgraph_assertion_readout.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/79_radgraph_assertions_existing_cpu.sh',
        Path(challenge.__file__).resolve(), BANK / 'manifest.json', BANK / 'resolver.jsonl',
        installed.ASSETS / 'asset_manifest.json', installed.TOKENIZER_ROOT / 'asset_manifest.json',
        BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json']
    paths.extend(BANK / r['report_path'] for r in resolver)
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    write_json(run / 'frozen_plan.json', {'schema_version': VERSION + '-plan', 'pins': pins,
        'fixed_reports': 56, 'designated_checks': 80, 'secondary_checks': 224,
        'max_model_calls': 56, 'seed': 0, 'model_threads': 2,
        'author_expected_states_not_expert_gold': True, 'independent_heldout': False,
        'model_receives_reference_or_baseline': False, 'case_or_policy_changes': False,
        'assertion_readout': 'native_states_covering_literal_heads_no_semantic_scope_correction',
        'thresholds_fitted': False, 'clinical_qualified': False})
    private_dir(run / 'tmp', fresh=True)
    private_dir(run / 'cache', fresh=True)
    os.environ.update({'HF_HOME': str(installed.HUB_CACHE.parent),
        'HF_HUB_CACHE': str(installed.HUB_CACHE), 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(run / 'tmp'),
        'XDG_CACHE_HOME': str(run / 'cache'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    records, native = [], {}
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        import radgraph
        from radgraph import RadGraph
        require(importlib.metadata.version('radgraph') == '0.1.18', 'official_package_version_required')
        installed_root = Path(radgraph.__file__).parent
        for path in (WORKSPACE / 'RadGraph/radgraph').rglob('*.py'):
            other = installed_root / path.relative_to(WORKSPACE / 'RadGraph/radgraph')
            require(other.is_file() and sha256(path) == sha256(other), 'installed_official_code_mismatch')
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
        for request in resolver:
            text = texts[request['report_sha256']]
            row = {'item_id': request['item_id'], 'report_sha256': request['report_sha256'],
                'status': 'failed_unavailable', 'finding_states': None,
                'source_span_references': None, 'failure_type': None}
            try:
                with torch.inference_mode():
                    output = model([text])
                require(isinstance(output, dict) and set(output) == {'0'}, 'complete_single_native_graph_required')
                graph = output['0']
                value = readout(text, graph)
                require(value['source_artifact_sha256'] == request['report_sha256'], 'source_readout_binding_required')
                row.update(status='complete', finding_states=value['finding_states'],
                    source_span_references=value['source_span_references'])
                native[request['item_id']] = graph
            except Exception as error:
                row['failure_type'] = type(error).__name__
            records.append(row)
            if len(records) % 20 == 0:
                print(json.dumps({'status': 'protected_authored_assertion_progress',
                    'runtime_seconds': round(time.monotonic() - before_inference, 3)}),
                    file=public_stdout, flush=True)
        inference_seconds = time.monotonic() - before_inference
    prediction_path = run / 'predictions.json'
    write_json(prediction_path, {'schema_version': VERSION + '-predictions', 'source': source,
        'records': records, 'producer': {'frozen': True, 'model_type': 'radgraph-xl', 'device': 'cpu'},
        'model_received_reference_states': False, 'scorer_read_reference_key': False,
        'clinical_qualified': False})
    sealed_prediction_sha = sha256(prediction_path)
    # Only after predictions are closed and fsynced do labels/baselines enter analysis.
    references, reference_path = challenge.load_references(BANK, resolver)
    metrics, details = challenge.evaluate_predictions(references, compatible(records))
    baseline_metrics, paired = {}, {}
    for name, (root, expected) in BASELINES.items():
        require(sha256(root / 'manifest.json') == expected, 'pinned_baseline_manifest_required')
        manifest = json.loads((root / 'manifest.json').read_text())
        path = root / 'predictions.json'
        require(sha256(path) == manifest['artifacts']['predictions.json']['sha256'],
                'pinned_baseline_predictions_required')
        prior = json.loads(path.read_text())['records']
        baseline_metrics[name], _ = challenge.evaluate_predictions(references, compatible(prior))
        paired[name] = compare_baseline(references, records, prior)
    require(sha256(MANUAL / 'manifest.json') == MANUAL_SHA, 'completed_manual_diagnostic_required')
    manual_manifest = json.loads((MANUAL / 'manifest.json').read_text())
    require(sha256(MANUAL / 'evaluation.json') == manual_manifest['artifacts']['evaluation.json'],
            'pinned_manual_metrics_required')
    reliability = assertion_reliability(json.loads((MANUAL / 'evaluation.json').read_text()))
    write_json(run / 'manual_assertion_reliability.json', reliability)
    write_json(run / 'evaluation.json', {'radgraph': metrics, 'baselines': baseline_metrics,
                                       'paired_designated_checks': paired})
    write_json(run / 'details.json', {'records': details})
    write_json(run / 'native_graphs.json', {'scope': 'authored_text_native_annotations_not_patients', 'graphs': native})
    require(sha256(prediction_path) == sealed_prediction_sha
            and all(sha256(WORKSPACE / path) == value for path, value in pins.items()),
            'frozen_predictions_or_sources_changed')
    summary = {'schema_version': VERSION + '-run', 'status': 'authored_assertion_diagnostic_complete',
        'authored_reports': 56, 'native_forward_passes': 56,
        'status_counts': dict(Counter(r['status'] for r in records)),
        'designated_checks': 80, 'full_vectors_secondary': 224,
        'predictions_fsynced_before_reference_or_baseline_reads': True,
        'prediction_sha256': sealed_prediction_sha,
        'patient_data_used_in_new_inference': False, 'clinical_qualified': False,
        'primary_metric_eligible': False, 'thresholds_fitted': False,
        'selection_changed': False, 'regeneration_authorized': False,
        'frozen_parameters': True, 'network_disabled': True, 'device': 'cpu',
        'model_archive_sha256': assets['archive_sha256'], 'tokenizer_revision': tokenizer['revision'],
        'initialization_seconds': load_seconds, 'inference_seconds': inference_seconds,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    write_json(run / 'summary.json', summary)
    analysis_paths = [reference_path, MANUAL / 'manifest.json', MANUAL / 'evaluation.json']
    analysis_paths.extend(root / name for root, _ in BASELINES.values()
                          for name in ('manifest.json', 'predictions.json'))
    write_json(run / 'manifest.json', {'schema_version': VERSION + '-receipt', 'pins': pins,
        'analysis_input_pins': {str(p.relative_to(WORKSPACE)): sha256(p) for p in analysis_paths},
        'artifacts': {p.name: sha256(p) for p in sorted(run.glob('*.json'))}})
    return summary


def main():
    run = None
    try:
        require(len(sys.argv) == 1, 'fixed_scope_worker_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_cpu_allocation_required')
        os.umask(0o007)
        private_dir(ROOT)
        target = ROOT / 'authored56_12766754_001'
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
        print(json.dumps({'status': 'protected_assertion_diagnostic_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
