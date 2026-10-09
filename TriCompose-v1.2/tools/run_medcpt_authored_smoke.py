"""Approved public MedCPT deployment and tiny wholly authored CPU smoke only."""
from __future__ import annotations

import argparse
import contextlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import urllib.request
from unittest.mock import patch

from prepare_ratescore_assets import (
    WORKSPACE, private_dir, relative_asset, require, require_slurm, sha256, write_json, fetch,
)
from tricompose_v12.medcpt_authored_probes import inventory, probes, score_pairs, diagnostic_summary

CONFIG = WORKSPACE / 'TriCompose-v1.2/configs/medcpt_query_assets_v1.json'
ASSETS = WORKSPACE / 'runtime/models/medcpt-query-v12-12714150-001'
SOURCE = WORKSPACE / 'MedCPT'
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
OUT = BASE / 'medcpt_authored_runs/query_12714150_001'
VERSIONS = {'torch': '2.6.0+cpu', 'transformers': '4.44.2', 'numpy': '1.26.4',
            'safetensors': '0.8.0', 'tokenizers': '0.19.1', 'huggingface_hub': '0.36.2'}
OLD = {
    'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json':
        'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c',
    'radgraph_literal_evidence_runs/literal_pool960_12714150_001/manifest.json':
        'da1bfb5c82f1ce51d000843ee11471c6edddaafc20a5ab075227a4c675eb57c9',
}


def asset_inventory(config):
    require(config['schema_version'] == 'tricompose-medcpt-query-assets-v1' and
            config['repository'] == 'ncbi/MedCPT-Query-Encoder' and
            config['revision'] == 'd83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc' and
            config['license'] == 'public-domain', 'exact_approved_medcpt_checkpoint_required')
    require(all(config.get(key) is False for key in ('train', 'clinical_qualified',
            'replace_biolord_in_official_ratescore', 'selection_changed', 'regeneration_authorized')),
            'unqualified_authored_only_scope_required')
    source = config['source_documentation']
    require(source['repository'] == 'ncbi/MedCPT' and
            source['revision'] == '11e129be74102c98d16a11c310b0b5ce74c1db5e',
            'exact_official_source_snapshot_required')
    entries = []
    for component, prefix in ((config, 'model'), (source, 'source')):
        for item in component['files']:
            relative = relative_asset(item['path']).as_posix()
            require(type(item['bytes']) is int and 0 < item['bytes'] <= 450000000
                    and len(item['sha256']) == 64 and all(c in '0123456789abcdef' for c in item['sha256']),
                    'bounded_pinned_asset_required')
            url = ('https://huggingface.co/' + component['repository'] + '/resolve/'
                   if prefix == 'model' else 'https://raw.githubusercontent.com/' + component['repository'] + '/')
            entries.append({**item, 'relative_path': prefix + '/' + relative,
                            'url': url + component['revision'] + '/' + relative})
    require(len(entries) == len({e['relative_path'] for e in entries}) == 11 and
            sum(e['bytes'] for e in entries) <= config['maximum_download_bytes'] == 500000000,
            'approved_small_single_model_inventory_required')
    require({e['path'] for e in entries if e['relative_path'].startswith('model/')} ==
            {'LICENSE', 'README.md', 'added_tokens.json', 'config.json', 'special_tokens_map.json',
             'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'model.safetensors'} and
            {e['path'] for e in entries if e['relative_path'].startswith('source/')} == {'LICENSE', 'README.md'},
            'no_other_weights_or_datasets_allowed')
    return entries


def verify(pins):
    require(all(sha256(WORKSPACE / p) == expected for p, expected in pins.items()),
            'pinned_source_or_historical_manifest_changed')


def execute(public_stdout):
    started = time.monotonic()
    entries = asset_inventory(json.loads(CONFIG.read_text()))
    pins = {str(path.relative_to(WORKSPACE)): sha256(path) for path in (
        CONFIG, Path(__file__).resolve(), WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/medcpt_authored_probes.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_medcpt_authored_smoke.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/62_medcpt_existing_cpu.sh',
        WORKSPACE / 'docs/medcpt_authored_smoke_protocol.md')}
    pins.update({str((BASE / p).relative_to(WORKSPACE)): value for p, value in OLD.items()})
    verify(pins)
    require({n: version(n) for n in VERSIONS} == VERSIONS, 'compatible_existing_frozen_runtime_required')
    plan, texts = inventory(probes())
    require(plan['pair_count'] == 41 and sum(p['input_nonempty'] for p in plan['pairs']) == 40,
            'fixed_small_authored_cohort_required')
    write_json(OUT / 'frozen_plan.json', {'pins': pins, 'public_assets': entries,
        'authored_inventory': plan, 'batch_size': 8, 'max_input_tokens': 64,
        'encoding': 'native_last_hidden_state_CLS', 'diagnostic_similarity': 'query_query_cosine',
        'training': False, 'clinical_inputs_read': False, 'clinical_qualified': False})
    private_dir(WORKSPACE / 'runtime/models', allow_existing_readonly=True)
    private_dir(ASSETS, fresh=True)
    private_dir(ASSETS / 'model', fresh=True)
    private_dir(ASSETS / 'source', fresh=True)
    download_started = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for entry in entries:
        fetch(entry, ASSETS / entry['relative_path'], opener)
    download_seconds = time.monotonic() - download_started
    # Atomic mkdir refuses any pre-existing source; hard links never replace files.
    private_dir(SOURCE, fresh=True)
    for name in ('LICENSE', 'README.md'):
        os.link(ASSETS / 'source' / name, SOURCE / name)
    write_json(ASSETS / 'asset_manifest.json', {'schema_version': 'medcpt-query-public-assets-v1',
        'status': 'complete', 'assets': entries, 'pins': pins,
        'total_bytes': sum(e['bytes'] for e in entries), 'implicit_token_used': False,
        'user_approved_download': True, 'license': 'public-domain',
        'source_snapshot_is_documentation_only': True})
    print(json.dumps({'status': 'pinned_medcpt_public_assets_complete',
                      'runtime_seconds': round(download_seconds, 3),
                      'asset_manifest_sha256': sha256(ASSETS / 'asset_manifest.json')}),
          file=public_stdout, flush=True)
    for entry in entries:
        require(sha256(ASSETS / entry['relative_path']) == entry['sha256'], 'acquired_asset_changed')
    verify(pins)
    # No model/API/network operation may leave this process after acquisition.
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': ''})
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        load_started = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / 'model'), local_files_only=True,
                                                 trust_remote_code=False)
        model, loading = AutoModel.from_pretrained(str(ASSETS / 'model'), local_files_only=True,
            trust_remote_code=False, output_loading_info=True)
        require(not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
                'complete_native_parameter_binding_required')
        model = model.to('cpu').eval().requires_grad_(False)
        require(not any(p.requires_grad for p in model.parameters()) and
                all(p.device.type == 'cpu' for p in model.parameters()) and
                model.config.model_type == 'bert' and model.config.hidden_size == 768,
                'native_frozen_cpu_bert_required')
        load_seconds = time.monotonic() - load_started
        keys = sorted(texts)
        inputs = [texts[key] for key in keys]
        lengths = [len(tokenizer(text, truncation=False)['input_ids']) for text in inputs]
        require(all(1 <= n <= 64 for n in lengths), 'no_authored_input_truncation_allowed')
        embeddings, replay, token_receipts = {}, {}, []
        passes = 0
        inference_started = time.monotonic()
        for iteration, destination in enumerate((embeddings, replay)):
            for start in range(0, len(keys), 8):
                batch_keys = keys[start:start + 8]
                encoded = tokenizer(inputs[start:start + 8], truncation=True, padding=True,
                    return_tensors='pt', max_length=64)
                with torch.inference_mode():
                    vector = model(**encoded).last_hidden_state[:, 0, :].cpu().to(torch.float32)
                require(list(vector.shape) == [len(batch_keys), 768] and torch.isfinite(vector).all().item(),
                        'finite_native_768_dimensional_embeddings_required')
                destination.update(zip(batch_keys, vector.tolist()))
                passes += 1
                if iteration == 0:
                    for offset, key in enumerate(batch_keys):
                        ids = encoded['input_ids'][offset][encoded['attention_mask'][offset].bool()].tolist()
                        require(len(ids) == lengths[start + offset], 'native_token_receipt_length_changed')
                        token_receipts.append({'text_sha256': key, 'native_token_count': len(ids),
                            'token_ids_sha256': __import__('hashlib').sha256(
                                json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                            'truncated': False, 'adapter_added_prefix': False})
        inference_seconds = time.monotonic() - inference_started
    require(embeddings == replay, 'deterministic_embedding_replay_required')
    rows = score_pairs(plan, embeddings)
    require(score_pairs(plan, replay) == rows and
            all(r['status'] == 'complete' if r['input_nonempty'] else r['status'] == 'empty_input_not_comparable'
                for r in rows) and
            all(abs(r['query_query_cosine'] - 1) < 1e-6 for r in rows if r['variant'] == 'identity'),
            'mechanical_smoke_invariants_required')
    write_json(OUT / 'score_table.json', rows)
    write_json(OUT / 'embeddings.json', {'scope': 'wholly_authored_nonpatient_texts_only',
        'encoding': 'native_CLS_float32', 'vectors_by_text_sha256': embeddings})
    write_json(OUT / 'token_receipts.json', token_receipts)
    summary = {**diagnostic_summary(rows), 'schema_version': 'tricompose-medcpt-authored-smoke-v1',
        'status': 'complete', 'checkpoint_revision': 'd83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc',
        'weight_sha256': '19d78c0d5eaee2f81e6c47c5425bbadcc0c6af016cbb5da4a000d64e59d6e342',
        'device': 'cpu', 'job_id': 12714150, 'new_slurm_submissions': 0, 'new_environments': 0,
        'new_package_installations': 0, 'versions': VERSIONS, 'dtype': 'float32',
        'distinct_encoded_texts': len(keys), 'embedding_dimension': 768,
        'native_forward_passes': passes, 'deterministic_exact_replay': True,
        'frozen_parameters': True, 'native_loading_issues': loading,
        'network_disabled_during_model_calls': True, 'download_seconds': download_seconds,
        'model_initialization_seconds': load_seconds, 'inference_and_replay_seconds': inference_seconds,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'source_patient_inputs_read': False, 'source_clinical_outputs_read': False,
        'candidate_bank_or_scores_modified': False, 'candidate_bank_scored_with_medcpt': False,
        'biolord_or_ratescore_deployed': False, 'scope_or_negation_accuracy_qualified': False}
    write_json(OUT / 'summary.json', summary)
    verify(pins)
    write_json(OUT / 'manifest.json', {'schema_version': 'tricompose-medcpt-authored-smoke-receipt-v1',
        'pins': pins, 'asset_manifest_sha256': sha256(ASSETS / 'asset_manifest.json'),
        'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(OUT.iterdir())
                      if p.is_file() and p.name != 'worker.log']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-download-and-authored-cpu-smoke', action='store_true')
    args = parser.parse_args()
    require(args.approved_download_and_authored_cpu_smoke is True, 'explicit_user_approval_required')
    require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(), 'approved_existing_cpu_allocation_required')
    require(not ASSETS.exists() and not ASSETS.is_symlink() and not SOURCE.exists() and
            not SOURCE.is_symlink() and not OUT.exists() and not OUT.is_symlink(),
            'refuse_existing_asset_source_or_run')
    os.umask(0o007)
    private_dir(BASE / 'medcpt_authored_runs')
    private_dir(OUT, fresh=True)
    public = sys.stdout
    try:
        with (OUT / 'worker.log').open('x') as log:
            (OUT / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = execute(public)
        print(json.dumps({'status': 'medcpt_authored_cpu_smoke_complete',
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'embedding_dimensions': [summary['distinct_encoded_texts'], 768],
            'manifest_sha256': sha256(OUT / 'manifest.json')}))
    except BaseException as error:
        write_json(OUT / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
            'clinical_qualified': False, 'partial_assets_preserved': True})
        print(json.dumps({'status': 'medcpt_authored_cpu_smoke_failed',
                          'error_type': type(error).__name__}), file=public)
        raise SystemExit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'medcpt_preflight_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
        raise SystemExit(2)
