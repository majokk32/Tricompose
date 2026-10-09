#!/usr/bin/env python3
"""Prepare/run/analyze a separate fixed-cohort source-span ID diagnostic.

Old exact-quote receipts and every historical score remain immutable. All
model requests contain synthetic report text only. No model inference here
without Slurm AND >=24 GiB CUDA allocation.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT.parent/'src'))
sys.path.insert(0, str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, WORKSPACE, RUN_ID_PATTERN, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
import verify_legacy_report_semantics as old
from check_legacy_report_scope import bounded_json, read_text_inputs
from verify_report_evidence_qwen import infer, summarize
from tricompose_v12.legacy_report_semantics import compare_extractions

INTERFACE_PATH = ROOT/'interfaces/report_span_selection.py'
spec = importlib.util.spec_from_file_location('tricompose_frozen_span_interface_v1', INTERFACE_PATH)
interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interface)
SCHEMA = 'tricompose-frozen-report-span-selection-v1'
PLAN_SCHEMA = SCHEMA+'-plan'
OLD_PLAN_SHA = '2c76baa73aea4b0af0bd4745ff164387727107a627a519524122c1ce84a99d2c'
OLD_REVIEW_SHA = 'd543ac0287b40bc1596944f348a1780b76c742b60a5cdc7decaddc8d2352dca4'


def program_sources():
    return {'span_worker': Path(__file__), 'span_interface': INTERFACE_PATH}


def build_requests(inputs, texts, unavailable):
    expected = {item['report_sha256'] for item in inputs}
    if (set(texts) | set(unavailable) != expected or set(texts) & set(unavailable)
            or len(inputs) != 24 or len(expected) != 22):
        raise ValueError('same_24_slots_22_hashes_required')
    requests = []
    for h in sorted(expected):
        if h in unavailable:
            requests.append({'report_sha256': h, 'input_status': 'unavailable',
                'input_failure_reason': unavailable[h], 'source_text': None,
                'inventory': None, 'messages': None, 'input_text_sha256': None})
            continue
        text = texts[h]
        if interface.digest(text) != h:
            raise ValueError('exact_source_text_hash_required')
        try:
            inventory = interface.build_inventory(text)
            messages = interface.request_messages(text, inventory)
        except interface.SpanContractError as exc:
            requests.append({'report_sha256': h, 'input_status': 'unavailable',
                'input_failure_reason': str(exc), 'source_text': text,
                'inventory': None, 'messages': None, 'input_text_sha256': None})
            continue
        requests.append({'report_sha256': h, 'input_status': 'ready', 'input_failure_reason': None,
            'source_text': text, 'inventory': inventory, 'messages': messages,
            'input_text_sha256': interface.digest(messages[0]['content'][0]['text'])})
    return requests


def prepare(args):
    old_root = require_inside(args.quote_plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(old_root/'manifest.json') != OLD_PLAN_SHA:
        raise ValueError('fixed_exact_quote_parent_plan_required')
    quote_plan, sources = old.load_plan(old_root)
    texts, unavailable, text_sources, verified = read_text_inputs(quote_plan['report_inputs'])
    requests = build_requests(quote_plan['report_inputs'], texts, unavailable)
    sources.update(text_sources, **program_sources())
    plan = {'schema_version': PLAN_SCHEMA, 'interface_version': interface.VERSION,
        'quote_plan_run': str(old_root), 'candidate_slots': 24, 'distinct_report_texts': 22,
        'report_inputs': quote_plan['report_inputs'],
        'model_path': quote_plan['model_path'], 'model_file_sha256': quote_plan['model_file_sha256'],
        'model_file_stats': quote_plan['model_file_stats'], 'prompt_sha256': interface.digest(interface.PROMPT),
        'max_report_chars': interface.MAX_REPORT_CHARS, 'max_spans': interface.MAX_SPANS,
        'max_span_chars': interface.MAX_SPAN_CHARS, 'max_new_tokens': 512, 'seed': 0,
        'max_model_calls': 22, 'model_retries': 0, 'min_vram_gib': 24, 'do_sample': False,
        'source_report_text_opened': True, 'raw_patient_inputs_opened': False,
        'image_pixels_opened': False, 'clinical_validation': False,
        'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False}
    summary = {'candidate_slots': 24, 'distinct_report_texts': 22, 'new_model_calls': 0,
        'verified_synthetic_report_file_slots': verified,
        'ready_distinct_requests': sum(r['input_status'] == 'ready' for r in requests),
        'unavailable_distinct_requests': sum(r['input_status'] != 'ready' for r in requests),
        'input_failure_reasons': dict(Counter(r['input_failure_reason'] for r in requests if r['input_status'] != 'ready')),
        'span_count_distribution_distinct_texts': dict(sorted(Counter(len(r['inventory']['spans'])
            for r in requests if r['input_status'] == 'ready').items())),
        'prompt_content_definition': 'Exact single user text BEFORE the frozen model chat-template wrapper; no second clinical prefix.',
        'synthetic_report_text_opened': True, 'raw_patient_inputs_opened': False,
        'selection_changed': False, 'primary_metric_eligible': False}
    return [('plan.json', plan), ('requests.jsonl', requests), ('summary.json', summary)], sources


def load_staged(plan_run):
    root = require_inside(plan_run, PROTECTED_ROOT, must_exist=True)
    manifest = bounded_json(root/'manifest.json')
    if manifest['schema_version'] != PLAN_SCHEMA or manifest['mode'] != 'prepare':
        raise ValueError('sealed_span_plan_required')
    sources = old.checked_sources(manifest)
    if any(key not in sources or sources[key] != path.resolve() for key, path in program_sources().items()):
        raise ValueError('frozen_span_program_required')
    for name in ('plan.json', 'requests.jsonl', 'summary.json'):
        path = root/name
        if path.stat().st_size > 4*1024*1024 or sha256_file(path) != manifest['artifacts'][name]['sha256']:
            raise ValueError('staged_input_size_or_hash_changed')
        sources['staged_'+name] = path
    plan = bounded_json(root/'plan.json')
    if (plan['interface_version'] != interface.VERSION or plan['prompt_sha256'] != interface.digest(interface.PROMPT)
            or plan['max_new_tokens'] != 512 or plan['min_vram_gib'] != 24
            or plan['do_sample'] is not False or plan['seed'] != 0
            or plan['selection_changed'] is not False or plan['regeneration_authorized'] is not False):
        raise ValueError('fixed_span_protocol_required')
    # Rebuild from the same exact synthetic bytes to verify every ready request,
    # inventory, limit refusal and opaque slot without a model or case reselection.
    outputs, _ = prepare(argparse.Namespace(quote_plan_run=plan['quote_plan_run']))
    rebuilt = dict(outputs)
    requests = [json.loads(line) for line in (root/'requests.jsonl').read_text().splitlines() if line]
    if plan != rebuilt['plan.json'] or requests != rebuilt['requests.jsonl']:
        raise ValueError('same_exact_frozen_requests_required')
    sources['span_plan_manifest'] = root/'manifest.json'
    return plan, requests, sources


def verify(args):
    plan, requests, sources = load_staged(args.plan_run)
    import torch
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24*1024**3:
        raise RuntimeError('gpu_with_at_least_24_gib_required')
    model_path = Path(plan['model_path'])
    assets = [model_path/name for name in old.MODEL_FILES]
    if {p.name: sha256_file(p) for p in assets} != plan['model_file_sha256']:
        raise ValueError('frozen_model_asset_hash_changed')
    before = {k: sha256_file(p) for k, p in sources.items()}
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    records, raw, calls = [], [], 0
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        model, processor, model_type = _load_model(model_path, torch,
            min_pixels=256*28*28, max_pixels=512*28*28)
        model.eval().requires_grad_(False)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('frozen_weights_required')
        dtype = str(next(model.parameters()).dtype)
        for request in requests:
            h = request['report_sha256']
            if request['input_status'] != 'ready':
                records.append({'report_sha256': h, 'model_call_attempted': False,
                    'input_text_sha256': None, **interface.unknown_result('input_unavailable:'+request['input_failure_reason'])})
                continue
            torch.cuda.synchronize()
            started = time.monotonic()
            response, tokens = infer(request['messages'], model, processor, torch, max_new_tokens=512)
            calls += 1
            torch.cuda.synchronize()
            decoded = interface.decode_response(response, request['source_text'], request['inventory'],
                token_limit_reached=tokens['token_limit_reached'])
            records.append({'report_sha256': h, 'model_call_attempted': True,
                'input_text_sha256': request['input_text_sha256'], 'response_sha256': interface.digest(response),
                'elapsed_seconds': round(time.monotonic()-started, 4), **tokens, **decoded})
            raw.append({'report_sha256': h, 'response': response, **tokens})
    if (calls > 22 or before != {k: sha256_file(p) for k, p in sources.items()}
            or any([p.stat().st_size, p.stat().st_mtime_ns] != plan['model_file_stats'][p.name] for p in assets)):
        raise ValueError('call_budget_or_immutable_sources_changed')
    summary = summarize(records)
    summary.update(schema_version=SCHEMA, model_calls=calls, candidate_slots=24,
        evidence_interface=interface.VERSION, model_retries=0, selection_changed=False,
        clinical_accuracy=None, primary_metric_eligible=False)
    payload = {'schema_version': SCHEMA, 'records': records, 'summary': summary,
        'producer': {'frozen': True, 'model_type': model_type, 'model_file_sha256': plan['model_file_sha256'],
            'prompt_sha256': plan['prompt_sha256'], 'interface_version': interface.VERSION,
            'max_new_tokens': 512, 'do_sample': False, 'seed': 0},
        'execution': {'torch_version': str(torch.__version__), 'gpu_name': torch.cuda.get_device_name(0),
            'dtype': dtype, 'slurm_job_id': os.environ['SLURM_JOB_ID']},
        'peak_allocated_vram_gib': round(torch.cuda.max_memory_allocated()/1024**3, 3),
        'verifier_received_ehr_images_scores_or_answer_key': False, 'external_api_used': False,
        'clinical_validation': False, 'selection_changed': False, 'regeneration_authorized': False,
        'primary_metric_eligible': False}
    return [('evidence.json', payload), ('raw_responses.json', {'records': raw}), ('summary.json', summary)], sources


def analyze(args):
    plan, requests, sources = load_staged(args.plan_run)
    root = require_inside(args.review_run, PROTECTED_ROOT, must_exist=True)
    manifest = bounded_json(root/'manifest.json')
    if (manifest['schema_version'] != SCHEMA or manifest['mode'] != 'run'
            or manifest['source_sha256']['span_plan_manifest'] != sha256_file(Path(args.plan_run)/'manifest.json')):
        raise ValueError('sealed_same_span_plan_review_required')
    old.checked_sources(manifest)
    for name in ('evidence.json', 'raw_responses.json'):
        path = root/name
        if sha256_file(path) != manifest['artifacts'][name]['sha256']:
            raise ValueError('immutable_span_review_changed')
        sources['sealed_'+path.stem] = path
    records = bounded_json(root/'evidence.json', 4*1024*1024)['records']
    raw = bounded_json(root/'raw_responses.json', 2*1024*1024)['records']
    lookup = {r['report_sha256']: r for r in requests}
    raw_lookup = {r['report_sha256']: r for r in raw}
    if (len(records) != 22 or len({r['report_sha256'] for r in records}) != 22
            or len(raw_lookup) != len(raw) or set(raw_lookup) != {r['report_sha256'] for r in records if r['model_call_attempted']}):
        raise ValueError('complete_distinct_span_receipts_required')
    for record in records:
        request = lookup[record['report_sha256']]
        if not record['model_call_attempted']:
            if request['input_status'] == 'ready':
                raise ValueError('unexpected_missing_model_call')
            decoded = interface.unknown_result('input_unavailable:'+request['input_failure_reason'])
        else:
            response = raw_lookup[record['report_sha256']]
            if (request['input_status'] != 'ready' or record['response_sha256'] != interface.digest(response['response'])
                    or record['input_text_sha256'] != request['input_text_sha256']
                    or any(record[k] != response[k] for k in ('input_tokens', 'output_tokens', 'token_limit_reached'))):
                raise ValueError('exact_span_request_response_mismatch')
            decoded = interface.decode_response(response['response'], request['source_text'], request['inventory'],
                token_limit_reached=response['token_limit_reached'])
        if any(record[k] != value for k, value in decoded.items()):
            raise ValueError('sealed_span_response_reparse_mismatch')
    quote_plan, _ = old.load_plan(plan['quote_plan_run'])
    parent, scoped, _ = old.load_parents(quote_plan['scope_plan_run'], quote_plan['scope_run'])
    comparison, summary = compare_extractions(parent['candidate_rows'], scoped, records)
    old_root = require_inside(args.quote_review_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(old_root/'manifest.json') != OLD_REVIEW_SHA:
        raise ValueError('fixed_strict_quote_baseline_required')
    old_manifest = bounded_json(old_root/'manifest.json')
    old.checked_sources(old_manifest)
    if sha256_file(old_root/'evidence.json') != old_manifest['artifacts']['evidence.json']['sha256']:
        raise ValueError('strict_quote_baseline_changed')
    baseline = bounded_json(old_root/'evidence.json', 4*1024*1024)['records']
    by_hash = {r['report_sha256']: r for r in baseline}
    if len(by_hash) != len(baseline) or set(by_hash) != set(lookup):
        raise ValueError('same_distinct_text_baseline_required')
    transitions = Counter(by_hash[r['report_sha256']]['contract_status']+'->'+r['contract_status'] for r in records)
    summary.update(schema_version=SCHEMA+'-comparison', evidence_interface=interface.VERSION,
        model_calls=sum(r['model_call_attempted'] for r in records), model_retries=0,
        old_exact_quote_complete=sum(r['contract_status'] == 'complete' for r in baseline),
        interface_status_transitions_distinct_texts=dict(transitions),
        raw_patient_inputs_opened=False, image_pixels_opened=False,
        old_scores_and_winners_unchanged=True, independent_verifier_votes=False,
        interpretation='Separate exploratory source-span interface. Valid source IDs establish reference integrity, not semantic/clinical correctness. Same frozen Qwen checkpoint is not an independent evaluator vote. Availability changes are not clinical improvements.')
    report = '# 原文片段 ID 核查 / Source-span ID review\n\n'
    report += 'Same 24 report slots / 22 texts; fixed four-head interface, same frozen Qwen checkpoint.\n\n'
    report += f"Exact-quote baseline complete: {summary['old_exact_quote_complete']}/22; span-ID complete: {summary['complete_distinct_responses']}/22.\n\n"
    report += '| Finding | Span-ID states (candidate slots) | CheXbert/span comparison |\n|---|---|---|\n'
    for name, values in summary['same_four_heads'].items():
        report += f"| {name} | {json.dumps(values['qwen_states_candidate_slots'], sort_keys=True)} | {json.dumps(values['chexbert_qwen_comparison_candidate_slots'], sort_keys=True)} |\n"
    report += '\n'+summary['interpretation']+'\n\n未改原始 EHR、图像、报告、分数和最优组合。没有临床正确率、已确认错误模态或修复收益结论。\n'
    sources.update(span_review_manifest=root/'manifest.json', strict_quote_review_manifest=old_root/'manifest.json',
        strict_quote_evidence=old_root/'evidence.json')
    return [('assertion_comparison.jsonl', comparison), ('summary.json', summary), ('RESULTS_CN_EN.md', report)], sources


def execute(args):
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise RuntimeError('slurm_required_before_input_or_model_access')
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError('opaque_run_id_required')
    target = require_inside(Path(args.output_root)/args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('refuse_existing_run_before_data_or_model_access')
    if args.mode == 'prepare':
        outputs, sources = prepare(args)
        schema = PLAN_SCHEMA
    elif args.mode == 'run':
        outputs, sources = verify(args)
        schema = SCHEMA
    else:
        outputs, sources = analyze(args)
        schema = SCHEMA+'-comparison'
    before = {k: sha256_file(p) for k, p in sources.items()}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = []
        for name, value in outputs:
            if name.endswith('.jsonl'):
                files.append(write_private_text(temporary/name, ''.join(json.dumps(r, sort_keys=True)+'\n' for r in value)))
            elif name.endswith('.md'):
                files.append(write_private_text(temporary/name, value))
            else:
                files.append(write_private_json(temporary/name, value))
        if before != {k: sha256_file(p) for k, p in sources.items()}:
            raise ValueError('immutable_sources_changed_before_commit')
        write_private_json(temporary/'manifest.json', {'schema_version': schema, 'mode': args.mode,
            'run_id': args.run_id, 'source_paths': {k: str(p) for k, p in sources.items()},
            'source_sha256': before, 'artifacts': {p.name: {'sha256': sha256_file(p)} for p in files},
            'selection_changed': False, 'regeneration_authorized': False, 'primary_metric_eligible': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'analyze'))
    for name in ('output-root', 'run-id'):
        parser.add_argument('--'+name, required=True)
    for name in ('quote-plan-run', 'plan-run', 'review-run', 'quote-review-run'):
        parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        target = execute(args)
    except Exception as exc:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(exc).__name__}))
        return 2
    print(json.dumps({'status': 'completed_'+args.mode, 'manifest_sha256': sha256_file(target/'manifest.json'),
        'elapsed_seconds': round(time.monotonic()-started, 3)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
