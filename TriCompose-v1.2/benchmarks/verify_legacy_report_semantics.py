#!/usr/bin/env python3
"""Frozen Qwen assertion check on the same sealed 24 synthetic report slots.

Prepare is metadata-only. Run requests contain exact report text only. Analyze
reparses sealed responses, without reranking, label changes or regeneration.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT.parent/'src'))
sys.path.insert(0, str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, WORKSPACE, RUN_ID_PATTERN, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
import check_legacy_report_scope as legacy
from verify_report_evidence_qwen import PROMPT, request_messages, decode_evidence, digest_text, infer, summarize
from tricompose_v12.legacy_report_semantics import SCHEMA, HEADS, compare_extractions

PLAN_SHA = '5856c43c78eaa4b8c8959b153a6ee188ef97cfe3d617cce2ce14556f06e4d69c'
SCOPE_SHA = '64b46bd52292462274e4a9ac2f16aade397189f945081060a906875032733e3b'
WEIGHT_SHA = '26fa644c8e61aa8185da5055b61aa28d6897fee102c2f43173acab657c4257a1'
MODEL_FILES = ('config.json', 'generation_config.json', 'processor_config.json',
    'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja', 'model.safetensors')
PLAN_SCHEMA = SCHEMA+'-plan'


def sources_for_program():
    paths = {'semantic_worker': Path(__file__),
        'semantic_comparison': ROOT/'src/tricompose_v12/legacy_report_semantics.py',
        'frozen_evidence_worker': ROOT/'benchmarks/verify_report_evidence_qwen.py',
        'frozen_infer': ROOT/'benchmarks/verify_candidate_findings_qwen.py',
        'frozen_model_adapter': ROOT.parent/'src/tricompose/verifiers/qwenvl_cxr_report.py',
        'imported_expanded_contract': ROOT/'benchmarks/prepare_expanded_polarity.py'}
    # Pin local Python dependencies transitively: importing old diagnostic
    # modules must not introduce an unrecorded code change or new prompt.
    paths.update({'benchmark_dependency_'+p.stem: p for p in sorted((ROOT/'benchmarks').glob('*.py'))
        if p.name != Path(__file__).name})
    paths.update({'contract_dependency_'+p.stem: p for p in sorted((ROOT/'src/tricompose_v12').glob('*.py'))})
    return {**legacy.program_sources(), **paths}


def checked_sources(manifest):
    sources = {key: require_inside(path, WORKSPACE, must_exist=True)
        for key, path in manifest['source_paths'].items()}
    if any(sha256_file(path) != manifest['source_sha256'][key] for key, path in sources.items()):
        raise ValueError('immutable_metadata_or_program_changed')
    return sources


def load_parents(plan_run, scope_run):
    plan_root = require_inside(plan_run, PROTECTED_ROOT, must_exist=True)
    scope_root = require_inside(scope_run, PROTECTED_ROOT, must_exist=True)
    pm, sm = plan_root/'manifest.json', scope_root/'manifest.json'
    if sha256_file(pm) != PLAN_SHA or sha256_file(sm) != SCOPE_SHA:
        raise ValueError('fixed_parent_manifest_required')
    manifest = legacy.bounded_json(pm)
    if (manifest['schema_version'] != legacy.PLAN_SCHEMA or manifest['metadata_only'] is not True
            or sha256_file(plan_root/'plan.json') != manifest['artifacts']['plan.json']['sha256']):
        raise ValueError('sealed_metadata_only_parent_required')
    sources = checked_sources(manifest)  # Parent plan has no report bodies.
    parent = legacy.bounded_json(plan_root/'plan.json', 2*1024*1024)
    args = argparse.Namespace(reliability_run=sources['source_manifest'].parent,
        candidate_run=sources['historical_bank_manifest'].parent)
    recreated, _ = legacy.prepare(args)
    if parent != recreated:
        raise ValueError('same_frozen_metadata_cohort_required')
    scope_manifest = legacy.bounded_json(sm)
    if (scope_manifest['schema_version'] != legacy.SCHEMA
            or scope_manifest['source_sha256']['plan_manifest'] != PLAN_SHA
            or scope_manifest['selection_changed'] is not False):
        raise ValueError('sealed_same_cohort_scope_result_required')
    scope_file = scope_root/'scope_fact_table.jsonl'
    if (scope_file.stat().st_size > 2*1024*1024
            or sha256_file(scope_file) != scope_manifest['artifacts'][scope_file.name]['sha256']):
        raise ValueError('immutable_scope_fact_table_required')
    scoped = [json.loads(line) for line in scope_file.read_text().splitlines() if line]
    if (len(parent['candidate_rows']) != 24 or len(parent['report_inputs']) != 24
            or len({r['report_sha256'] for r in parent['report_inputs']}) != 22 or len(scoped) != 336):
        raise ValueError('fixed_24_slot_22_text_inventory_required')
    # Never hash the scope result's synthetic_report_bytes_* during preparation.
    sources.update(scope_plan_manifest=pm, scope_plan=plan_root/'plan.json',
        scope_result_manifest=sm, scope_fact_table=scope_file)
    return parent, scoped, sources


def prepare(args):
    parent, _, sources = load_parents(args.scope_plan_run, args.scope_run)
    model = Path(args.model_path).resolve(strict=True)
    files = [model/name for name in MODEL_FILES]
    if any(not path.is_file() for path in files):
        raise ValueError('complete_frozen_local_checkpoint_required')
    # Only small config/tokenizer hashes and checkpoint stat here; do not read
    # 15.4 GiB of weights in the CPU metadata preparation step.
    hashes = {p.name: sha256_file(p) for p in files if p.name != 'model.safetensors'}
    hashes['model.safetensors'] = WEIGHT_SHA
    stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    sources.update(sources_for_program())
    payload = {'schema_version': PLAN_SCHEMA, 'scope_plan_run': str(Path(args.scope_plan_run).resolve()),
        'scope_run': str(Path(args.scope_run).resolve()), 'report_inputs': parent['report_inputs'],
        'candidate_slots': 24, 'distinct_report_texts': 22, 'finding_order': list(HEADS),
        'model_path': str(model), 'model_file_sha256': hashes, 'model_file_stats': stats,
        'prompt_sha256': digest_text(PROMPT), 'max_new_tokens': 512, 'do_sample': False, 'seed': 0,
        'min_vram_gib': 24, 'max_model_calls': 22, 'model_retries': 0,
        'frozen': True, 'metadata_only': True, 'report_bodies_opened': False,
        'real_source_or_target_supplied': False, 'selection_changed': False,
        'regeneration_authorized': False, 'primary_metric_eligible': False}
    return payload, sources


def load_plan(run):
    root = require_inside(run, PROTECTED_ROOT, must_exist=True)
    manifest = legacy.bounded_json(root/'manifest.json')
    path = root/'plan.json'
    if (manifest['schema_version'] != PLAN_SCHEMA
            or sha256_file(path) != manifest['artifacts']['plan.json']['sha256']):
        raise ValueError('sealed_semantic_plan_required')
    sources = checked_sources(manifest)
    if any(key not in sources or sources[key] != path.resolve() for key, path in sources_for_program().items()):
        raise ValueError('sealed_semantic_program_required')
    plan = legacy.bounded_json(path)
    rebuilt, _ = prepare(argparse.Namespace(scope_plan_run=plan['scope_plan_run'],
        scope_run=plan['scope_run'], model_path=plan['model_path']))
    if plan != rebuilt:
        raise ValueError('semantic_plan_or_model_metadata_changed')
    sources.update(semantic_plan_manifest=root/'manifest.json', semantic_plan=path)
    return plan, sources


def verify(args):
    plan, sources = load_plan(args.plan_run)
    import torch
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24*1024**3:
        raise RuntimeError('gpu_with_at_least_24_gib_required')
    model_path = Path(plan['model_path'])
    files = [model_path/name for name in MODEL_FILES]
    if {p.name: sha256_file(p) for p in files} != plan['model_file_sha256']:
        raise ValueError('frozen_model_asset_hash_changed')
    texts, unavailable, text_sources, verified_slots = legacy.read_text_inputs(plan['report_inputs'])
    for h, text in list(texts.items()):
        if not text.strip():
            unavailable[h] = 'empty_report'
            del texts[h]
    if set(texts) | set(unavailable) != {r['report_sha256'] for r in plan['report_inputs']}:
        raise ValueError('complete_report_availability_required')
    sources.update(text_sources)
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
        for h in sorted(set(texts) | set(unavailable)):
            if h in unavailable:
                decoded = decode_evidence('{}', '')
                decoded['contract_failure_reason'] = 'source_unavailable:'+unavailable[h]
                records.append({'report_sha256': h, 'model_call_attempted': False, **decoded})
                continue
            torch.cuda.synchronize()
            started = time.monotonic()
            # No scores, EHR, image, model IDs or labels enter this request.
            response, tokens = infer(request_messages(texts[h]), model, processor, torch, max_new_tokens=512)
            calls += 1
            torch.cuda.synchronize()
            decoded = decode_evidence(response, texts[h], token_limit_reached=tokens['token_limit_reached'])
            records.append({'report_sha256': h, 'model_call_attempted': True,
                'response_sha256': digest_text(response), 'elapsed_seconds': round(time.monotonic()-started, 4),
                **tokens, **decoded})
            raw.append({'report_sha256': h, 'response': response, **tokens})
    if (any([p.stat().st_size, p.stat().st_mtime_ns] != plan['model_file_stats'][p.name] for p in files)
            or before != {k: sha256_file(p) for k, p in sources.items()}):
        raise ValueError('immutable_source_or_model_changed_during_inference')
    summary = summarize(records)
    summary.update(model_calls=calls, candidate_slots=24, verified_synthetic_report_file_slots=verified_slots,
        model_retries=0, primary_metric_eligible=False, selection_changed=False, clinical_accuracy=None)
    payload = {'schema_version': SCHEMA, 'records': records, 'summary': summary,
        'producer': {'frozen': True, 'model_type': model_type, 'dtype': dtype,
            'model_file_sha256': plan['model_file_sha256'], 'prompt_sha256': plan['prompt_sha256'],
            'max_new_tokens': 512, 'do_sample': False, 'seed': 0},
        'execution': {'torch_version': str(torch.__version__), 'gpu_name': torch.cuda.get_device_name(0),
            'dtype': dtype, 'slurm_job_id': os.environ['SLURM_JOB_ID']},
        'verifier_received_ehr_images_scores_or_answer_key': False, 'external_api_used': False,
        'selection_changed': False, 'regeneration_authorized': False, 'primary_metric_eligible': False,
        'peak_allocated_vram_gib': round(torch.cuda.max_memory_allocated()/1024**3, 3)}
    return [('evidence.json', payload), ('raw_responses.json', {'records': raw}), ('summary.json', summary)], sources


def analyze(args):
    plan, sources = load_plan(args.plan_run)
    root = require_inside(args.review_run, PROTECTED_ROOT, must_exist=True)
    manifest = legacy.bounded_json(root/'manifest.json')
    if (manifest['schema_version'] != SCHEMA or manifest['mode'] != 'run'
            or manifest['source_sha256']['semantic_plan_manifest'] != sha256_file(Path(args.plan_run)/'manifest.json')):
        raise ValueError('sealed_same_plan_review_required')
    checked_sources(manifest)
    for name in ('evidence.json', 'raw_responses.json'):
        path = root/name
        if sha256_file(path) != manifest['artifacts'][name]['sha256']:
            raise ValueError('immutable_semantic_receipt_changed')
        sources['sealed_'+path.stem] = path
    records = legacy.bounded_json(root/'evidence.json', 4*1024*1024)['records']
    raw = legacy.bounded_json(root/'raw_responses.json', 2*1024*1024)['records']
    texts, unavailable, text_sources, _ = legacy.read_text_inputs(plan['report_inputs'])
    sources.update(text_sources, semantic_review_manifest=root/'manifest.json')
    by_hash = {r['report_sha256']: r for r in raw}
    if len(by_hash) != len(raw) or set(by_hash) != {r['report_sha256'] for r in records if r['model_call_attempted']}:
        raise ValueError('complete_raw_response_receipts_required')
    for record in records:
        h = record['report_sha256']
        if not record['model_call_attempted']:
            if h not in unavailable and texts.get(h, '').strip():
                raise ValueError('unexpected_missing_model_call')
            continue
        response = by_hash[h]
        decoded = decode_evidence(response['response'], texts[h], token_limit_reached=response['token_limit_reached'])
        if (record['response_sha256'] != digest_text(response['response'])
                or any(record[key] != value for key, value in decoded.items())
                or any(record[key] != response[key] for key in ('input_tokens', 'output_tokens', 'token_limit_reached'))):
            raise ValueError('sealed_response_reparse_mismatch')
    parent, scoped, _ = load_parents(plan['scope_plan_run'], plan['scope_run'])
    comparison, summary = compare_extractions(parent['candidate_rows'], scoped, records)
    summary.update(model_calls=sum(r['model_call_attempted'] for r in records),
        raw_patient_inputs_opened=False, image_pixels_opened=False, old_scores_and_winners_unchanged=True)
    report = '# 冻结语义断言核查 / Frozen semantic assertion check\n\n'
    report += 'Same 24 candidate slots / 22 distinct texts, four frozen heads; report text only.\n\n'
    report += '| Finding | Qwen states (candidate slots) | CheXbert/Qwen comparison |\n|---|---|---|\n'
    for name, values in summary['same_four_heads'].items():
        report += f"| {name} | {json.dumps(values['qwen_states_candidate_slots'], sort_keys=True)} | {json.dumps(values['chexbert_qwen_comparison_candidate_slots'], sort_keys=True)} |\n"
    report += '\n'+summary['interpretation']+'\n\n未修改原始 EHR、图像、报告、分数或最优组合。失败与缺失仍为 unknown；不报告临床正确率或修复收益。\n'
    return [('assertion_comparison.jsonl', comparison), ('summary.json', summary), ('RESULTS_CN_EN.md', report)], sources


def execute(args):
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise RuntimeError('slurm_required_before_data_or_model_access')
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError('opaque_run_id_required')
    target = require_inside(Path(args.output_root)/args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('refuse_existing_run_before_data_or_model_access')
    if args.mode == 'prepare':
        payload, sources = prepare(args)
        outputs, schema = [('plan.json', payload)], PLAN_SCHEMA
    elif args.mode == 'run':
        outputs, sources = verify(args)
        schema = SCHEMA
    else:
        outputs, sources = analyze(args)
        schema = SCHEMA+'-comparison'
    fingerprints = {k: sha256_file(p) for k, p in sources.items()}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = []
        for name, payload in outputs:
            if name.endswith('.jsonl'):
                files.append(write_private_text(temporary/name, ''.join(json.dumps(r, sort_keys=True)+'\n' for r in payload)))
            elif name.endswith('.md'):
                files.append(write_private_text(temporary/name, payload))
            else:
                files.append(write_private_json(temporary/name, payload))
        if fingerprints != {k: sha256_file(p) for k, p in sources.items()}:
            raise ValueError('immutable_sources_changed_before_commit')
        write_private_json(temporary/'manifest.json', {'schema_version': schema, 'mode': args.mode,
            'run_id': args.run_id, 'source_paths': {k: str(p) for k, p in sources.items()},
            'source_sha256': fingerprints, 'artifacts': {p.name: {'sha256': sha256_file(p)} for p in files},
            'selection_changed': False, 'primary_metric_eligible': False, 'regeneration_authorized': False})
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
    for name in ('scope-plan-run', 'scope-run', 'model-path', 'plan-run', 'review-run'):
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
