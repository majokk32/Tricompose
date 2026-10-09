"""Approved existing CPU Slurm: 100 manual reports, frozen CheXbert/cached XL.

Source text and native graph text stay internal to this protected worker. Public
stdout has only sanitized progress, runtime, memory and hashes. No new dataset,
GPU, training, API, case selection, threshold fitting or winner modification.
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

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json
from prepare_cxrgraph_manual_cohort import INTAKE, read_documents
from official_report_benchmark import decode, input_normalization
from tricompose_v12.entity_gold_contract import digest, normalize_graph
from tricompose_v12.manual_literal_assertions import FINDINGS, VERSION, project_reference, evaluate
from tricompose_v12.radgraph_assertion_readout import readout

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
CONTRACT = BASE / 'cxrgraph_gold_contracts/manual_test_12766754_001'
CONTRACT_SHA = '19f61d54c4cfda59212564e615f14518f9e3ce5cab3c71a59b9e1da9cdd56c15'
CACHED = BASE / 'cxrgraph_extraction_runs/manual_xl_12766754_002'
CACHED_SHA = 'a56743b0a3a940184faebbc7a39b0947d227d7f9ed3ffadc00592e10bed7898c'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
CHECKPOINT = WORKSPACE / '.cache/tricompose_report_eval/models/chexbert/chexbert.pth'
BERT = WORKSPACE / '.cache/tricompose_report_eval/models/bert-base-uncased'
ADAPTER = WORKSPACE / 'cxrmate/tools/chexbert.py'
ASSET_HASHES = {CHECKPOINT: '6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1',
    ADAPTER: '5d131d8dc8253211f127f48d8ac61fc689373e64dc92b9ec00a794e2331eafe6',
    BERT / 'config.json': '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20',
    BERT / 'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
    BERT / 'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'}


def requests_from_documents(documents, cohort):
    require(len(documents) == len(cohort) and len(cohort) <= 1024, 'complete_fixed_source_inventory_required')
    texts = {}
    for index, (document, row) in enumerate(zip(documents, cohort)):
        require(row['report_id'] == f'report_{index:04d}' and row['source_index'] == index
                and row['status'] == 'validated', 'fixed_release_order_no_case_selection_required')
        sentences = document['sentences']
        require(isinstance(sentences, list) and sentences and all(isinstance(s, list) for s in sentences),
                'released_sentence_lists_required')
        words = [w for sentence in sentences for w in sentence]
        require(words and all(isinstance(w, str) and w and not any(c.isspace() for c in w) for w in words),
                'released_literal_word_strings_required')
        text = ' '.join(words)
        require(len(text) <= 100000 and hashlib.sha256(text.encode()).hexdigest() == row['source_report_sha256'],
                'same_source_text_as_cached_xl_required')
        texts[row['report_id']] = text
    return texts


def execute(run, public_stdout):
    started = time.monotonic()
    require(sha256(CONTRACT / 'manifest.json') == CONTRACT_SHA
            and sha256(CACHED / 'manifest.json') == CACHED_SHA and sha256(BANK) == BANK_SHA,
            'fixed_completed_manual_sources_and_unchanged_bank_required')
    contract = json.loads((CONTRACT / 'manifest.json').read_text())
    cached = json.loads((CACHED / 'manifest.json').read_text())
    for root, manifest, names in ((CONTRACT, contract, ('cohort.json',)),
            (CACHED, cached, ('native_graphs.json', 'comparisons.json', 'prediction_records.json', 'prediction_receipts.json'))):
        for name in names:
            require(sha256(root / name) == manifest['artifacts'][name], 'completed_manual_artifact_binding_required')
    source = INTAKE / 'source/manual_data/test.json'
    require(sha256(source) == cached['pins'][str(source.relative_to(WORKSPACE))],
            'unchanged_same_hundred_source_reports_required')
    require(all(sha256(path) == value for path, value in ASSET_HASHES.items()),
            'frozen_chexbert_assets_required')
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_literal_assertions.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_manual_literal_assertions.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_manual_literal_worker.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/80_manual_literal_readers_existing_cpu.sh',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_assertion_readout.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/assertion_agreement_diagnostic.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_character_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/cxrgraph_gold_adapter.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_cxrgraph_manual_cohort.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.2/real_validation/official_report_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/real_validation/audit_official_report_gold.py',
        CONTRACT / 'manifest.json', CONTRACT / 'cohort.json', CACHED / 'manifest.json', source, BANK,
        *ASSET_HASHES]
    paths.extend(CACHED / name for name in ('native_graphs.json', 'comparisons.json',
                                           'prediction_records.json', 'prediction_receipts.json'))
    pins = {str(path.relative_to(WORKSPACE)): sha256(path) for path in paths}
    write_json(run / 'frozen_plan.json', {'schema_version': VERSION + '-plan', 'pins': pins,
        'attempted_reports': 100, 'fixed_four_findings': FINDINGS,
        'new_chexbert_inference_examples_maximum': 100, 'batch_size': 4,
        'cached_radgraph_examples': 100, 'new_radgraph_inference_calls': 0,
        'seed': 0, 'device': 'cpu', 'model_threads': 2, 'model_retries': 0,
        'token_overflow': 'failed_unavailable_no_truncation',
        'clinical_reference_role': 'fixed_literal_human_span_projection_not_full_report_labels',
        'synonyms_or_scope_rules_added': False, 'checkpoint_training_overlap': 'unresolved',
        'development_data_not_untouched_final_test': True, 'model_receives_gold_or_baseline': False,
        'selection_changed': False, 'regeneration_authorized': False, 'thresholds_fitted': False})
    cohort = json.loads((CONTRACT / 'cohort.json').read_text())
    documents, _ = read_documents(source.read_bytes())
    require(len(cohort) == len(documents) == 100, 'all_hundred_fixed_reports_required')
    texts = requests_from_documents(documents, cohort)
    del documents  # Bundled annotations are not used to make inference requests.
    private_dir(run / 'tmp', fresh=True)
    private_dir(run / 'cache', fresh=True)
    os.environ.update({'HF_HOME': str(run / 'cache/huggingface'), 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(run / 'tmp'), 'XDG_CACHE_HOME': str(run / 'cache'),
        'TORCH_HOME': str(run / 'cache/torch'), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2'})
    predictions = {name: [] for name in ('chexbert', 'radgraph')}
    with (patch.object(socket.socket, 'connect', side_effect=RuntimeError('network_disabled')),
          patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network_disabled'))):
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        spec = importlib.util.spec_from_file_location('pinned_official_chexbert_manual_cpu', ADAPTER)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        before_load = time.monotonic()
        model = adapter.CheXbert(ckpt_dir=str(CHECKPOINT.parent), bert_path=str(BERT),
            checkpoint_path=CHECKPOINT.name, device=torch.device('cpu'))
        model.eval().requires_grad_(False)
        require(not model.training and not any(p.requires_grad for p in model.parameters()), 'frozen_parameters_required')
        load_seconds = time.monotonic() - before_load
        ready, rows = [], {}
        for entry in cohort:
            key = entry['report_id']
            row = {'report_id': key, 'source_sha256': entry['source_report_sha256'],
                'status': 'failed_unavailable', 'finding_states': None, 'failure_type': None}
            tokens = model.tokenizer(input_normalization(texts[key]), truncation=False)['input_ids']
            row['input_token_count'] = len(tokens)
            row['normalized_input_sha256'] = hashlib.sha256(input_normalization(texts[key]).encode()).hexdigest()
            row['normalization_policy'] = 'unchanged_official_chexbert_wrapper'
            if len(tokens) > model.bert.config.max_position_embeddings:
                row['failure_type'] = 'token_budget_exceeded_no_truncation'
            else:
                ready.append(key)
            rows[key] = row
        before_inference = time.monotonic()
        examples = batches = 0
        for start in range(0, len(ready), 4):
            keys = ready[start:start + 4]
            examples += len(keys)
            batches += 1
            try:
                with torch.inference_mode():
                    values = model([texts[key] for key in keys]).detach().cpu().tolist()
                require(len(values) == len(keys), 'complete_batch_prediction_inventory_required')
                for key, value in zip(keys, values):
                    states = decode(value)
                    rows[key].update(status='complete', finding_states={f: states[f] for f in FINDINGS})
            except Exception as error:
                for key in keys:
                    rows[key]['failure_type'] = type(error).__name__
            if batches % 5 == 0:
                print(json.dumps({'status': 'protected_manual_reader_progress',
                    'runtime_seconds': round(time.monotonic() - before_inference, 3)}),
                    file=public_stdout, flush=True)
        inference_seconds = time.monotonic() - before_inference
        require(examples <= 100, 'frozen_call_budget_required')
        predictions['chexbert'] = [rows[entry['report_id']] for entry in cohort]
    write_json(run / 'chexbert_predictions.json', {'records': predictions['chexbert'],
        'frozen_parameters': True, 'device': 'cpu', 'model_received_annotation_states': False})
    sealed_chexbert = sha256(run / 'chexbert_predictions.json')
    # Predictions are closed and fsynced before human head projection/baseline analysis.
    native = json.loads((CACHED / 'native_graphs.json').read_text())['graphs']
    old_records = {r['report_id']: r for r in json.loads((CACHED / 'prediction_records.json').read_text())}
    require(set(native) == set(old_records) == set(texts), 'all_hundred_cached_graphs_required')
    for entry in cohort:
        key = entry['report_id']
        row = {'report_id': key, 'source_sha256': entry['source_report_sha256'],
               'status': 'failed_unavailable', 'finding_states': None, 'failure_type': None}
        try:
            record = normalize_graph(native[key], native_schema='radgraph_xl', report_id=key,
                source_report_sha256=entry['source_report_sha256'], origin='declared_frozen_prediction')
            require(record == old_records[key], 'native_graph_matches_frozen_prediction_record_required')
            value = readout(texts[key], native[key])
            require(value['source_artifact_sha256'] == row['source_sha256'], 'cached_readout_source_binding_required')
            row.update(status='complete', finding_states=value['finding_states'])
        except Exception as error:
            row['failure_type'] = type(error).__name__
        predictions['radgraph'].append(row)
    write_json(run / 'radgraph_predictions.json', {'records': predictions['radgraph'],
        'frozen_cached_predictions': True, 'new_radgraph_calls': 0})
    comparisons = json.loads((CACHED / 'comparisons.json').read_text())
    require(len(comparisons) == 100 and {r['report_id'] for r in comparisons} == set(texts),
            'all_hundred_manual_span_comparisons_required')
    by_id = {row['report_id']: row for row in comparisons}
    references = []
    for entry in cohort:
        key = entry['report_id']
        comparison = by_id[key]
        require(comparison['comparison_sha256'] == digest({k: v for k, v in comparison.items()
                                                          if k != 'comparison_sha256'}), 'sealed_manual_characters_required')
        ref = project_reference(texts[key], comparison['gold_character_entities'], entry['source_report_sha256'])
        references.append({'report_id': key, 'source_sha256': entry['source_report_sha256'],
            'source_domain': entry['source_domain'], **ref})
    write_json(run / 'reference_projection.json', {'role': 'derived_human_literal_spans_not_official_full_finding_labels',
                                                 'records': references})
    evaluation, details = evaluate(references, predictions)
    groups = {}
    for domain in ('mimic', 'chexpert'):
        ids = {entry['report_id'] for entry in cohort if entry['source_domain'] == domain}
        require(len(ids) == 50, 'fixed_disjoint_domain_counts_required')
        groups[domain], _ = evaluate([r for r in references if r['report_id'] in ids],
            {name: [r for r in records if r['report_id'] in ids] for name, records in predictions.items()})
    write_json(run / 'evaluation.json', evaluation)
    write_json(run / 'domain_evaluation.json', groups)
    write_json(run / 'mask_details.json', {'records': details})
    require(sha256(run / 'chexbert_predictions.json') == sealed_chexbert
            and all(sha256(WORKSPACE / path) == value for path, value in pins.items()),
            'prediction_inputs_or_consumed_program_changed')
    summary = {'schema_version': VERSION + '-run', 'status': 'complete',
        'attempted_reports': 100, 'encoder_examples': examples, 'forward_batches': batches,
        'new_radgraph_forward_calls': 0, 'prediction_status_counts': {
            name: dict(Counter(r['status'] for r in records)) for name, records in predictions.items()},
        'predictions_fsynced_before_reference_projection': True,
        'source_text_or_keys_written': False, 'new_slurm_submissions': 0,
        'network_disabled': True, 'frozen_parameters': True, 'device': 'cpu',
        'truncation_used': False, 'clinical_qualified': False, 'untouched_final_test': False,
        'selection_changed': False, 'regeneration_authorized': False,
        'initialization_seconds': load_seconds, 'inference_seconds': inference_seconds,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
        'torch_version': str(torch.__version__)}
    write_json(run / 'summary.json', summary)
    for directory in (run / 'tmp', run / 'cache'):
        for path in [directory, *directory.rglob('*')]:
            require(not path.is_symlink() and path.resolve().is_relative_to(run),
                    'protected_runtime_no_external_symlinks_required')
            require(path.stat().st_gid in (96293, 65534), 'protected_runtime_project_group_required')
            path.chmod(0o2770 if path.is_dir() else 0o660)
    write_json(run / 'manifest.json', {'schema_version': VERSION + '-receipt', 'pins': pins,
        'artifacts': {p.name: sha256(p) for p in sorted(run.glob('*.json'))},
        'raw_source_text_or_identifiers_exported': False, 'clinical_qualified': False})
    return summary


def main():
    run = None
    try:
        require(len(sys.argv) == 1, 'fixed_manual_reader_worker_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_cpu_allocation_required')
        os.umask(0o007)
        parent = BASE / 'manual_literal_reader_runs'
        private_dir(parent)
        target = parent / 'manual100_12766754_001'
        private_dir(target, fresh=True)
        run = target
        with (run / 'worker.log').open('x') as log:
            (run / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = execute(run, sys.__stdout__)
        print(json.dumps({'status': 'protected_manual_reader_benchmark_complete',
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'manifest_sha256': sha256(run / 'manifest.json')}))
        return 0
    except Exception as error:
        if run is not None and not (run / 'manifest.json').exists():
            write_json(run / 'failure.json', {'status': 'failed', 'failure_type': type(error).__name__})
        print(json.dumps({'status': 'protected_manual_reader_benchmark_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
