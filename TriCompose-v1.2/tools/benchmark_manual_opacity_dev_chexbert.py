"""Prepared, separately approved CPU Slurm report-label diagnostic only.

All 75 dev texts, including the reference-unavailable slot, are attempted.
Model inputs use only released words. References are decoded after prediction
sealing. Literal annotations and broad CheXbert labels are not equivalent gold.
"""
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

from contracts import new_atomic_run, commit_atomic_run, discard_atomic_run
from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json
from prepare_cxrgraph_manual_cohort import read_documents
from official_report_benchmark import decode, input_normalization
from tricompose_v12.manual_opacity_reader_diagnostic import VERSION, HEADS, evaluate

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
CONFIG = WORKSPACE / 'TriCompose-v1.2/configs/manual_opacity_chexbert_v1.json'
CHECKPOINT = WORKSPACE / '.cache/tricompose_report_eval/models/chexbert/chexbert.pth'
BERT = WORKSPACE / '.cache/tricompose_report_eval/models/bert-base-uncased'
ADAPTER = WORKSPACE / 'cxrmate/tools/chexbert.py'
ASSET_HASHES = {
    CHECKPOINT: '6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1',
    ADAPTER: '5d131d8dc8253211f127f48d8ac61fc689373e64dc92b9ec00a794e2331eafe6',
    BERT / 'config.json': '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20',
    BERT / 'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
    BERT / 'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'}
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'


def sources():
    names = ('tools/benchmark_manual_opacity_dev_chexbert.py',
        'src/tricompose_v12/manual_opacity_reader_diagnostic.py',
        'tests/test_manual_opacity_reader_diagnostic.py', 'tests/test_manual_opacity_reader_worker.py',
        'configs/manual_opacity_chexbert_v1.json', 'slurm/83_manual_opacity_dev_chexbert_cpu.sbatch',
        'tools/prepare_ratescore_assets.py', 'tools/prepare_cxrgraph_manual_cohort.py',
        'src/tricompose_v12/cxrgraph_gold_adapter.py', 'src/tricompose_v12/entity_gold_contract.py',
        'src/tricompose_v12/radgraph_reference_contract.py',
        'src/tricompose_v12/radgraph_reference_contract_v2.py',
        'real_validation/official_report_benchmark.py', 'real_validation/audit_official_report_gold.py')
    return [WORKSPACE / 'TriCompose-v1.2' / name for name in names] + [
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']


def requests_from_documents(documents, expected_count):
    """Do not filter on NER validity, literal words or reference state."""
    require(type(expected_count) is int and 0 < expected_count <= 1024
            and isinstance(documents, list) and len(documents) == expected_count,
            'all_fixed_release_documents_required')
    texts, rows = {}, []
    for index, doc in enumerate(documents):
        key = f'report_{index:04d}'
        row = {'report_id': key, 'source_index': index, 'source_sha256': None,
               'status': 'failed_unavailable', 'failure_type': None, 'finding_states': None,
               'input_token_count': None, 'normalized_input_sha256': None}
        try:
            sentences = doc.get('sentences')
            require(isinstance(sentences, list) and sentences
                    and all(isinstance(s, list) for s in sentences), 'released_sentence_lists_required')
            words = [word for sentence in sentences for word in sentence]
            require(words and all(isinstance(w, str) and w and not any(c.isspace() for c in w) for w in words),
                    'released_literal_words_required')
            text = ' '.join(words)
            require(len(text) <= 100000, 'bounded_canonical_report_required')
            row['source_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
            texts[key] = text
        except Exception as error:
            row['failure_type'] = type(error).__name__
        rows.append(row)
    return texts, rows


def predict_requests(model, texts, rows, inference_context, *, batch_size, maximum_examples, public_stdout):
    """Official model is injected; authored tests inject a non-model fake."""
    require(type(batch_size) is int and 1 <= batch_size <= 4 and
            type(maximum_examples) is int and len(rows) <= maximum_examples <= 75,
            'fixed_small_cpu_encoder_budget_required')
    require(len({r['report_id'] for r in rows}) == len(rows)
            and set(texts).issubset({r['report_id'] for r in rows}), 'complete_unique_request_slots_required')
    require(type(model.bert.config.max_position_embeddings) is int
            and 0 < model.bert.config.max_position_embeddings <= 4096, 'official_bounded_token_limit_required')
    by_id = {r['report_id']: r for r in rows}
    ready = []
    for row in rows:
        key = row['report_id']
        if key not in texts:
            require(row['failure_type'] is not None, 'unreadable_text_needs_explicit_failure')
            continue
        try:
            normalized = input_normalization(texts[key])
            row['normalized_input_sha256'] = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            tokens = model.tokenizer(normalized, truncation=False)['input_ids']
            require(isinstance(tokens, list) and tokens and all(type(t) is int for t in tokens),
                    'official_token_ids_required')
            row['input_token_count'] = len(tokens)
            row['normalization_policy'] = 'unchanged_official_chexbert_wrapper'
            if len(tokens) > model.bert.config.max_position_embeddings:
                row['failure_type'] = 'token_budget_exceeded_no_truncation'
            else:
                ready.append(key)
        except Exception as error:
            row['failure_type'] = type(error).__name__
    examples = batches = 0
    started = time.monotonic()
    for start in range(0, len(ready), batch_size):
        keys = ready[start:start + batch_size]
        require(examples + len(keys) <= maximum_examples, 'no_retry_or_encoder_budget_expansion')
        examples += len(keys)
        batches += 1
        try:
            with inference_context():
                values = model([texts[key] for key in keys]).detach().cpu().tolist()
            require(len(values) == len(keys), 'complete_batch_output_required')
            decoded = [decode(value) for value in values]
            require(all(tuple(states) == HEADS for states in decoded), 'pinned_official_head_order_required')
            for key, states in zip(keys, decoded):
                by_id[key].update(status='complete', failure_type=None, finding_states=states)
        except Exception as error:
            for key in keys:
                by_id[key].update(status='failed_unavailable', failure_type=type(error).__name__, finding_states=None)
        if batches % 5 == 0:
            print(json.dumps({'status': 'protected_opacity_reader_progress',
                'runtime_seconds': round(time.monotonic() - started, 3)}), file=public_stdout, flush=True)
    return {'encoder_examples': examples, 'forward_batches': batches,
            'inference_seconds': time.monotonic() - started}


def verify_request(config):
    expected = {'schema_version': 'manual-opacity-chexbert-request-v1',
        'source_file': 'source/manual_data/dev.json', 'source_bytes': 212072,
        'attempted_reports': 75, 'maximum_encoder_examples': 75, 'batch_size': 4,
        'device': 'cpu', 'model_threads': 2, 'seed': 0, 'retries': 0,
        'token_overflow': 'failed_unavailable_no_truncation',
        'gold_decode_order': 'after_predictions_fsynced_and_hash_sealed',
        'model_receives_annotation_labels': False, 'operator_blinded': False,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False,
        'new_training': False, 'download_allowed': False, 'external_api_allowed': False}
    require(all(config.get(k) == v and type(config.get(k)) is type(v) for k, v in expected.items()),
            'fixed_cpu_report_reader_request_required')


def require_task_scope():
    require(len(sys.argv) == 1, 'fixed_scope_worker_takes_no_arguments')
    job = os.environ.get('SLURM_JOB_ID', '')
    require_slurm(Path('/proc/self/cgroup').read_text(), job)
    require(job != '12784259' and not os.environ.get('SLURM_JOB_GPUS')
            and not os.environ.get('SLURM_STEP_GPUS'), 'new_approved_cpu_task_not_cache_only_allocation_required')
    require(os.environ.get('TRICOMPOSE_MANUAL_OPACITY_READER') == 'manual-opacity-chexbert-request-v1',
            'explicit_approved_reader_batch_scope_required')
    return job


def execute(run, public_stdout):
    require_task_scope()  # A direct imported call must not bypass the batch guard.
    started = time.monotonic()
    config = json.loads(CONFIG.read_text())
    verify_request(config)
    reference = WORKSPACE / config['reference_run']
    intake = WORKSPACE / config['source_intake']
    source = intake / config['source_file']
    require(reference.resolve().is_relative_to(BASE) and intake.resolve().is_relative_to(BASE)
            and source.resolve().is_relative_to(intake) and not source.is_symlink(), 'private_authorized_source_boundary_required')
    require(sha256(reference / 'manifest.json') == config['reference_manifest_sha256']
            and sha256(intake / 'intake_manifest.json') == config['intake_manifest_sha256']
            and sha256(source) == config['source_sha256'] and source.stat().st_size == config['source_bytes']
            and sha256(BANK) == BANK_SHA, 'fixed_dev_reference_and_unchanged_bank_required')
    ref_manifest = json.loads((reference / 'manifest.json').read_text())
    # Hashing the reference bytes is not decoding reference states for inference.
    require(sha256(reference / 'reference_inventory.json') == ref_manifest['artifacts']['reference_inventory.json'],
            'sealed_human_reference_inventory_required')
    require(all(sha256(path) == value for path, value in ASSET_HASHES.items()), 'frozen_official_assets_required')
    paths = [*sources(), source, intake / 'intake_manifest.json', reference / 'manifest.json',
             reference / 'reference_inventory.json', BANK, *ASSET_HASHES]
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    write_json(run / 'frozen_plan.json', {'schema_version': VERSION + '-plan', 'request': config,
        'pins': pins, 'other_thirteen_heads_not_evaluated': True,
        'source_annotations_bundled_but_not_used_for_prediction': True,
        'one_gold_unavailable_text_still_attempted': True,
        'annotations_decoded_only_for_post_prediction_evaluation': False,
        'reference_inventory_decoded_only_after_prediction_seal': True,
        'operator_blinded': False, 'clinical_qualified': False,
        'selection_changed': False, 'regeneration_authorized': False})
    documents, _ = read_documents(source.read_bytes())
    texts, rows = requests_from_documents(documents, config['attempted_reports'])
    del documents  # Released JSON bundles annotations; the request builder ignores them.
    private_dir(run / 'tmp', fresh=True)
    private_dir(run / 'cache', fresh=True)
    private_dir(run / 'cache/loader', fresh=True)
    os.environ.update({'HF_HOME': str(run / 'cache/huggingface'), 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(run / 'tmp'), 'XDG_CACHE_HOME': str(run / 'cache'),
        'TORCH_HOME': str(run / 'cache/torch'), 'MPLCONFIGDIR': str(run / 'cache/matplotlib'),
        'PYTHONPYCACHEPREFIX': str(run / 'cache/bytecode'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        spec = importlib.util.spec_from_file_location('pinned_chexbert_manual_dev_cpu', ADAPTER)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        before_load = time.monotonic()
        # Absolute checkpoint_path retains the existing checkpoint while sending
        # tokenizer cache_dir to this new protected run, not the model directory.
        model = adapter.CheXbert(ckpt_dir=str(run / 'cache/loader'), bert_path=str(BERT),
            checkpoint_path=str(CHECKPOINT), device=torch.device('cpu'))
        model.eval().requires_grad_(False)
        require(not model.training and not any(p.requires_grad for p in model.parameters()), 'frozen_eval_parameters_required')
        load_seconds = time.monotonic() - before_load
        counters = predict_requests(model, texts, rows, torch.inference_mode,
            batch_size=config['batch_size'], maximum_examples=config['maximum_encoder_examples'], public_stdout=public_stdout)
    del texts
    write_json(run / 'chexbert_predictions.json', {'records': rows, 'frozen_parameters': True,
        'device': 'cpu', 'gold_used_for_prediction': False, 'truncation_used': False,
        'head_order': HEADS, 'normalization_policy': 'unchanged_official_chexbert_wrapper'})
    sealed = sha256(run / 'chexbert_predictions.json')
    # Only now decode the sealed reference states. Failed references are retained.
    references = json.loads((reference / 'reference_inventory.json').read_text())['records']
    evaluation, details = evaluate(references, rows)
    support = {s: evaluation['per_reference_state'][s]['reference_support_all_attempted']
               for s in ('positive', 'negative', 'uncertain', 'unknown')}
    support['unavailable'] = evaluation['reference_unavailable']
    require(support == config['known_class_support'], 'unchanged_reference_support_not_retuned_to_predictions')
    write_json(run / 'evaluation.json', evaluation)
    write_json(run / 'comparison_records.json', {'records': details})
    require(sha256(run / 'chexbert_predictions.json') == sealed and
            all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'consumed_inputs_or_predictions_changed')
    summary = {'schema_version': VERSION + '-run', 'status': 'complete_development_reader_diagnostic',
        'attempted_reports': len(rows), **counters,
        'prediction_status_counts': dict(Counter(r['status'] for r in rows)),
        'predictions_fsynced_and_sealed_before_reference_decode': True,
        'source_keys_or_report_text_serialized_as_data_artifacts': False,
        'model_receives_annotation_labels': False, 'operator_blinded': False,
        'frozen_parameters': True, 'device': 'cpu', 'network_disabled': True,
        'truncation_used': False, 'retries': 0, 'new_training': False,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False,
        'initialization_seconds': load_seconds, 'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'torch_version': str(torch.__version__)}
    write_json(run / 'summary.json', summary)
    return summary, pins


def main():
    temporary = None
    try:
        job = require_task_scope()
        os.umask(0o007)
        temporary, target = new_atomic_run(BASE / 'manual_opacity_reader_runs', 'manual_dev_' + job + '_001')
        public_stdout = sys.stdout
        with (temporary / 'framework.log').open('x') as stream:
            (temporary / 'framework.log').chmod(0o660)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result, pins = execute(temporary, public_stdout)
        # These are newly created runtime files only, never asset/source files.
        for path in [temporary, *temporary.rglob('*')]:
            require(not path.is_symlink() and path.resolve().is_relative_to(temporary.resolve())
                    and path.stat().st_gid in (96293, 65534), 'private_new_run_no_external_links_required')
            require(path.is_dir() or path.is_file(), 'new_run_regular_files_only_required')
            path.chmod(0o2770 if path.is_dir() else 0o660)
        write_json(temporary / 'manifest.json', {'schema_version': VERSION + '-receipt', 'pins': pins,
            'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
        temporary = None
        print(json.dumps({'status': 'protected_manual_opacity_reader_diagnostic_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'peak_rss_gib': round(result['peak_rss_gib'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        if temporary is not None:
            discard_atomic_run(temporary)
        print(json.dumps({'status': 'protected_manual_opacity_reader_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
