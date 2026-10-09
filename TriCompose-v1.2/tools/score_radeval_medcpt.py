"""Prepare metadata; separately approved internal report-reference CPU test.

No downloads/training/images/EHRs/API; no candidate-bank rescoring or repair.
Preparation never opens the author report CSV or initializes a model.
"""
import argparse
from collections import Counter, defaultdict
import contextlib
import csv
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from unittest.mock import patch

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json
from tricompose_v12.medcpt_authored_probes import cosine
from tricompose_v12.radeval_medcpt_benchmark import join, evaluate, POLICY

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'report_metric_sources/radeval_expert_source_12714150_001'
EXPERT = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
BIOVIL = BASE / 'radeval_image_benchmark_runs/biovil_cpu_12714150_001'
ASSETS = WORKSPACE / 'runtime/models/medcpt-query-v12-12714150-001'
PLANS = BASE / 'radeval_medcpt_plans'
PLAN = PLANS / 'reportref_12714150_001'
RUNS = BASE / 'radeval_medcpt_runs'
RUN = RUNS / 'reportref_12714150_001'
VERSIONS = {'torch': '2.6.0+cpu', 'transformers': '4.44.2', 'numpy': '1.26.4',
            'safetensors': '0.8.0', 'tokenizers': '0.19.1', 'huggingface_hub': '0.36.2'}
RECEIPTS = {
    SOURCE: 'be99b29497d8bf75d52cdd7ca1cc353814bbc18a9ef5c9c1675f03b5752e9ff4',
    EXPERT: '21e1724a5b364eee97b56cdb8f66c0d56194baadc98c333b5c473f7efb37038d',
    BIOVIL: '8e55b223ef22c5da61c20aeb0c53f818ab6c1736eeb2c95ee85b39803ea0c18a',
}
ASSET_SHA = 'f2c23611d0c428d99a27d27858871f8dcd20794f26597574e13979e7463a64aa'
CONTRACT = BASE / 'report_metric_sources/radeval_expert_contract_12714150_003/inventory.json'
CONTRACT_SHA = '29c076bc24fd4c628703cbe1a1ea5049cf090660ff06a2c46e09e59e50b49365'


def allocation_guard():
    require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(), 'existing_cpu_allocation_required')


def metadata():
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/medcpt_authored_probes.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_medcpt_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/report_metric_alignment.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_radeval_medcpt_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/63_radeval_medcpt_existing_cpu.sh',
        WORKSPACE / 'docs/radeval_medcpt_protocol.md', CONTRACT,
        BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json',
        BASE / 'radgraph_literal_evidence_runs/literal_pool960_12714150_001/manifest.json']
    for root, expected in RECEIPTS.items():
        require(sha256(root / 'manifest.json') == expected, 'frozen_existing_receipt_required')
        paths.append(root / 'manifest.json')
    for root, names in ((EXPERT, ('frozen_plan.json', 'scores.json')),
                        (BIOVIL, ('image_scores.json',))):
        receipt = json.loads((root / 'manifest.json').read_text())
        known = {e['path']: e['sha256'] for e in receipt['artifacts']}
        for name in names:
            require(sha256(root / name) == known[name], 'cached_opaque_artifact_required')
            paths.append(root / name)
    require(sha256(CONTRACT) == CONTRACT_SHA and sha256(ASSETS / 'asset_manifest.json') == ASSET_SHA,
            'accepted_expert_contract_and_verified_model_required')
    paths.append(ASSETS / 'asset_manifest.json')
    assets = json.loads((ASSETS / 'asset_manifest.json').read_text())
    require(assets['status'] == 'complete' and assets['license'] == 'public-domain' and
            assets['implicit_token_used'] is False and len(assets['assets']) == 11,
            'approved_public_medcpt_assets_required')
    for item in assets['assets']:
        path = ASSETS / item['relative_path']
        require(path.resolve().is_relative_to(ASSETS) and not path.is_symlink() and
                path.stat().st_size == item['bytes'] and sha256(path) == item['sha256'],
                'native_pinned_asset_required')
        paths.append(path)
    source = json.loads((SOURCE / 'manifest.json').read_text())
    require(source['user_confirmed_authorization'] is True and source['raw_payload_protected'] is True
            and source['file'] == 'reader_study_final_with_annotationsv3.csv', 'authorized_existing_source_required')
    path = SOURCE / source['file']
    # Metadata only here: report CSV bytes are not opened by prepare().
    require(not path.is_symlink() and path.stat().st_size == 691815, 'fixed_source_file_metadata_required')
    expert = json.loads((EXPERT / 'frozen_plan.json').read_text())['inventory']
    require(expert == json.loads(CONTRACT.read_text()) and len(expert['records']) == 624
            and len(expert['graphs']) == 762 and expert['source_groups'] == 180,
            'complete_fixed_expert_inventory_required')
    images = json.loads((BIOVIL / 'image_scores.json').read_text())
    require(len(images) == 624 and sum(r['status'] == 'complete' for r in images) == 132,
            'fixed_cached_image_coverage_required')
    require({n: version(n) for n in VERSIONS} == VERSIONS, 'frozen_compatible_runtime_required')
    return {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}, source, expert


def verify(pins):
    require(all(sha256(WORKSPACE / path) == expected for path, expected in pins.items()),
            'prepared_source_asset_or_old_score_changed')


def prepare():
    allocation_guard()
    pins, source, expert = metadata()
    os.umask(0o007)
    private_dir(PLANS)
    private_dir(PLAN, fresh=True)
    write_json(PLAN / 'plan.json', {'schema_version': 'tricompose-radeval-medcpt-plan-v1',
        'pins': pins, 'policy': dict(POLICY), 'source_payload_sha256': source['sha256'],
        'source_payload_file': source['file'], 'all_attempted_pairs': 624, 'distinct_text_inventory': 762,
        'source_groups': expert['source_groups'], 'cached_image_available_pairs': 132,
        'max_input_tokens': 512, 'oversize_policy': 'unavailable_not_truncated', 'batch_size': 8,
        'encoding': 'native_last_hidden_state_CLS_float32', 'embedding_dimension': 768,
        'replay_scope': 'first_8_hash_sorted_eligible_texts_only', 'bootstrap_resamples': 1000, 'seed': 0,
        'cohort_selected_using_medcpt_scores': False, 'model_calls': 0, 'report_csv_opened': False,
        'patient_input_model_execution_authorized_by_preparation': False,
        'new_slurm_submissions': 0, 'process_timeout_seconds': 1200,
        'existing_allocation': {'job_id': 12714150, 'cpus': 4, 'memory_gib': 32, 'gpus': 0}})
    write_json(PLAN / 'manifest.json', {'schema_version': 'tricompose-radeval-medcpt-plan-receipt-v1',
        'plan_sha256': sha256(PLAN / 'plan.json'), 'model_calls': 0,
        'patient_input_model_execution_authorized': False})
    return {'status': 'expert_medcpt_metadata_prepared_no_model_or_report_csv_read',
            'plan_manifest_sha256': sha256(PLAN / 'manifest.json')}


def recover_texts(rows, plan):
    """Called only inside the separately approved protected model worker."""
    require(len(rows) == plan['source_rows'], 'unchanged_source_row_inventory_required')
    cells = defaultdict(list)
    for cell in plan['annotation_cells']:
        cells[cell['item_id']].append(cell['source_row_index'])
    texts = {}
    for pair in plan['records']:
        require(bool(cells[pair['item_id']]), 'source_row_link_required')
        for index in cells[pair['item_id']]:
            require(type(index) is int and 0 <= index < len(rows), 'bounded_source_row_index_required')
            for field, expected in (('ground_truth', pair['reference_sha256']),
                    (f'prediction{pair["candidate_slot"]}', pair['hypothesis_sha256'])):
                text = rows[index][field]
                require(isinstance(text, str) and len(text) <= 100000 and
                        hashlib.sha256(text.encode()).hexdigest() == expected, 'exact_native_report_input_required')
                require(texts.setdefault(expected, text) == text, 'input_digest_collision')
    require(set(texts) == {g['text_sha256'] for g in plan['graphs']}, 'all_exact_texts_required')
    return texts


def encode(texts, tokenizer, model, torch, public):
    vectors, receipts = {}, {}
    for digest in sorted(texts):
        text = texts[digest]
        receipt = {'text_sha256': digest, 'status': 'failed_embedding', 'failure_type': None,
            'native_token_count': None, 'token_ids_sha256': None,
            'truncated': False, 'adapter_added_prefix': False}
        if not text.strip():
            receipt['status'] = 'empty_input'
        else:
            try:
                ids = tokenizer(text, truncation=False)['input_ids']
                receipt.update(native_token_count=len(ids), token_ids_sha256=hashlib.sha256(
                    json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                    status='over_capacity' if len(ids) > 512 else 'ready')
            except Exception as error:
                receipt['failure_type'] = type(error).__name__
        receipts[digest] = receipt
    keys = [k for k in sorted(texts) if receipts[k]['status'] == 'ready']
    calls, started = 0, time.monotonic()
    for start in range(0, len(keys), 8):
        batch = keys[start:start + 8]
        try:
            encoded = tokenizer([texts[k] for k in batch], truncation=False, padding=True, return_tensors='pt')
            require(encoded['input_ids'].shape[1] <= 512, 'full_input_within_native_capacity_required')
            for offset, key in enumerate(batch):
                ids = encoded['input_ids'][offset][encoded['attention_mask'][offset].bool()].tolist()
                require(len(ids) == receipts[key]['native_token_count'] and hashlib.sha256(
                    json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == receipts[key]['token_ids_sha256'],
                    'tokenized_input_receipt_changed')
            calls += 1
            with torch.inference_mode():
                result = model(**encoded).last_hidden_state[:, 0, :].float().cpu()
            require(tuple(result.shape) == (len(batch), 768) and torch.isfinite(result).all().item(),
                    'native_finite_fulltext_vectors_required')
            vectors.update(zip(batch, result.tolist()))
            for key in batch:
                receipts[key]['status'] = 'complete'
        except Exception as error:
            for key in batch:
                receipts[key].update(status='failed_embedding', failure_type=type(error).__name__)
        if start % 128 == 0:
            print(json.dumps({'status': 'protected_reference_metric_progress',
                'runtime_seconds': round(time.monotonic() - started, 3)}), file=public, flush=True)
    replay_keys = keys[:8]
    replay = {'scope': 'first_8_hash_sorted_eligible_texts_only', 'attempted_texts': len(replay_keys),
              'status': 'unavailable', 'exact_match': None}
    if replay_keys and all(k in vectors for k in replay_keys):
        try:
            encoded = tokenizer([texts[k] for k in replay_keys], truncation=False, padding=True, return_tensors='pt')
            calls += 1
            with torch.inference_mode():
                repeated = model(**encoded).last_hidden_state[:, 0, :].float().cpu().tolist()
            matched = repeated == [vectors[k] for k in replay_keys]
            replay.update(status='complete' if matched else 'mismatch', exact_match=matched)
            require(matched, 'predeclared_small_replay_mismatch')
        except Exception as error:
            replay.update(status='failed', failure_type=type(error).__name__)
    return vectors, receipts, calls, replay


def prediction_rows(plan, vectors, receipts):
    rows = []
    for pair in plan['records']:
        value, status = None, 'empty_input'
        if pair['input_nonempty']:
            keys = (pair['reference_sha256'], pair['hypothesis_sha256'])
            states = [receipts[k]['status'] for k in keys]
            if all(s == 'complete' for s in states) and all(k in vectors for k in keys):
                try:
                    value, status = cosine(*(vectors[k] for k in keys)), 'complete'
                except ValueError:
                    status = 'failed_embedding'
            else:
                status = 'over_capacity' if 'over_capacity' in states else 'failed_embedding'
        rows.append({'item_id': pair['item_id'], 'status': status, 'value': value})
    return rows


def execute(approved, public):
    require(approved is True, 'separate_expert_report_cpu_execution_approval_required')
    allocation_guard()
    require(not RUN.exists() and not RUN.is_symlink(), 'refuse_existing_expert_run')
    plan_receipt = json.loads((PLAN / 'manifest.json').read_text())
    require(sha256(PLAN / 'plan.json') == plan_receipt['plan_sha256'], 'sealed_metadata_plan_required')
    prepared = json.loads((PLAN / 'plan.json').read_text())
    verify(prepared['pins'])
    pins, source, expert = metadata()
    require(pins == prepared['pins'] and prepared['source_payload_sha256'] == source['sha256'],
            'prepared_inventory_unchanged_required')
    os.umask(0o007)
    private_dir(RUNS)
    private_dir(RUN, fresh=True)
    started = time.monotonic()
    try:
        with (RUN / 'worker.log').open('x') as log:
            (RUN / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                path = SOURCE / prepared['source_payload_file']
                require(sha256(path) == prepared['source_payload_sha256'], 'approved_source_bytes_unchanged_required')
                with path.open(newline='', encoding='utf-8-sig') as stream:
                    texts = recover_texts(list(csv.DictReader(stream)), expert)
                write_json(RUN / 'frozen_plan.json', prepared)
                with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
                      patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
                    import torch
                    from transformers import AutoTokenizer, AutoModel
                    torch.set_num_threads(2)
                    torch.set_num_interop_threads(1)
                    torch.manual_seed(0)
                    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / 'model'), local_files_only=True,
                        trust_remote_code=False)
                    model, loading = AutoModel.from_pretrained(str(ASSETS / 'model'), local_files_only=True,
                        trust_remote_code=False, output_loading_info=True)
                    require(not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')),
                            'complete_native_parameter_binding_required')
                    model = model.to('cpu').eval().requires_grad_(False)
                    require(model.config.max_position_embeddings == 512 and model.config.hidden_size == 768 and
                            model.config.model_type == 'bert' and not any(p.requires_grad for p in model.parameters())
                            and all(p.device.type == 'cpu' for p in model.parameters()), 'native_frozen_cpu_model_required')
                    infer_started = time.monotonic()
                    vectors, receipts, calls, replay = encode(texts, tokenizer, model, torch, public)
                    infer_seconds = time.monotonic() - infer_started
                scores = prediction_rows(expert, vectors, receipts)
                # Seal model predictions before integrating cached expert outcomes or comparing metrics.
                write_json(RUN / 'scores.json', scores)
                write_json(RUN / 'token_receipts.json', receipts)
                write_json(RUN / 'private_embeddings.json', {'vectors_by_text_sha256': vectors})
                write_json(RUN / 'prediction_receipt.json', {'schema_version': 'medcpt-expert-predictions-v1',
                    'model_inputs_contain_expert_counts': False, 'native_forward_passes': calls,
                    'replay': replay, 'artifacts': [{'path': name, 'sha256': sha256(RUN / name)}
                        for name in ('scores.json', 'token_receipts.json', 'private_embeddings.json')]})
                previous = json.loads((EXPERT / 'scores.json').read_text())
                images = json.loads((BIOVIL / 'image_scores.json').read_text())
                records = join(expert, scores, previous, images)
                statistics_started = time.monotonic()
                evaluation = evaluate(records, resamples=1000, seed=0)
                statistics_seconds = time.monotonic() - statistics_started
                write_json(RUN / 'score_table.json', records)
                write_json(RUN / 'evaluation.json', evaluation)
                summary = {'schema_version': 'medcpt-expert-reference-run-v1', 'status': 'complete',
                    'policy': dict(POLICY), 'all_attempted_pairs': 624, 'distinct_source_texts': len(texts),
                    'pair_status_counts': dict(Counter(s['status'] for s in scores)),
                    'text_status_counts': dict(Counter(r['status'] for r in receipts.values())),
                    'native_forward_passes': calls, 'small_replay': replay,
                    'full_common_available_pairs': sum(r['full_common_available'] for r in records),
                    'image_common_available_pairs': sum(r['image_common_available'] for r in records),
                    'whole_report_input_limit': 512, 'input_truncation': False,
                    'official_query_example_64_token_recipe_used': False,
                    'model_unchanged_native_CLS_used': True, 'native_loading_issues': loading,
                    'experimental_report_reference_relatedness_not_official_retrieval_metric': True,
                    'released_expert_counts_already_analyzed_before_this_experiment': True,
                    'blind_or_untouched_testing_claimed': False,
                    'device': 'cpu', 'job_id': 12714150, 'versions': VERSIONS,
                    'new_training': False, 'new_slurm_submissions': 0, 'new_downloads': 0,
                    'source_ehr_or_images_read': False, 'raw_report_text_exported': False,
                    'old_bank_scores_or_choices_changed': False, 'network_blocked_during_model_calls': True,
                    'inference_seconds': infer_seconds, 'statistics_seconds': statistics_seconds,
                    'runtime_seconds': time.monotonic() - started,
                    'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
                write_json(RUN / 'summary.json', summary)
                require(sha256(path) == prepared['source_payload_sha256'], 'source_unchanged_after_scoring_required')
                verify(prepared['pins'])
                write_json(RUN / 'manifest.json', {'schema_version': 'medcpt-expert-reference-receipt-v1',
                    'plan_manifest_sha256': sha256(PLAN / 'manifest.json'), 'pins': prepared['pins'],
                    'policy': dict(POLICY), 'artifacts': [{'path': p.name, 'sha256': sha256(p)}
                        for p in sorted(RUN.glob('*.json'))]})
        return {'status': 'protected_medcpt_expert_benchmark_complete',
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3), 'manifest_sha256': sha256(RUN / 'manifest.json')}
    except BaseException as error:
        write_json(RUN / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                        'clinical_qualified': False, 'partial_outputs_preserved': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--approved-expert-report-cpu-execution', action='store_true')
    args = parser.parse_args()
    result = prepare() if args.mode == 'prepare' else execute(args.approved_expert_report_cpu_execution, sys.stdout)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'medcpt_expert_worker_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
