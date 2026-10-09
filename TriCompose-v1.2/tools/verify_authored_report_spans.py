#!/usr/bin/env python3
"""Frozen V2 on an existing authored-language challenge, not clinical gold.

Preparation uses metadata/blinded invented inputs only. GPU run freezes its
predictions on disk before reading authored references or baseline predictions.
No real MIMIC inputs, generator changes, prompt tuning, ranking or repair.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT/'benchmarks', ROOT/'src', ROOT.parent/'src',
        ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(directory))
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, read_json,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
import report_assertion_challenge as challenge
import verify_legacy_report_semantics as frozen

spec = importlib.util.spec_from_file_location('authored_span_interface', ROOT/'interfaces/report_span_selection_v2.py')
interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interface)
SCHEMA = 'tricompose-authored-report-span-v2-diagnostic-v1'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
BANK = BASE/'benchmarks/authored_assertions_20261002_001'
BANK_SHA = '5f76ca3d7f710f4fdf722f87f27afc6548c839f1c4d77e56a6502ec4e3583757'
MODEL_PLAN = BASE/'report_span_v2_plans/span_v2_scope2_12645021_001'
MODEL_PLAN_SHA = '576483c1b4f544b7fd99f5b6e7aca94d8c22c83694294fd41cd3ab9d1aa74afc'
BASELINE = BASE/'verification_runs/authored_assertions_chexbert_12580901'
BASELINE_SHA = '2fde45f8cef8993e7c6f932448c761557a61c4af8c258109e951ba0d36b348c1'


def require_gpu_approval(approved):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not approved or not job.isdigit():
        raise RuntimeError('approved_authored_gpu_slurm_required')
    if f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('actual_slurm_cgroup_required')


def pinned_manifest(root, expected):
    if sha256_file(root/'manifest.json') != expected:
        raise ValueError('fixed_parent_manifest_required')
    return read_json(root/'manifest.json')


def prepare():
    bank_manifest = pinned_manifest(BANK, BANK_SHA)
    parent_manifest = pinned_manifest(MODEL_PLAN, MODEL_PLAN_SHA)
    pinned_manifest(BASELINE, BASELINE_SHA)
    if sha256_file(MODEL_PLAN/'plan.json') != parent_manifest['artifacts']['plan.json']['sha256']:
        raise ValueError('fixed_model_plan_required')
    parent = read_json(MODEL_PLAN/'plan.json')
    if parent['interface_version'] != interface.VERSION or parent['prompt_sha256'] != interface.digest(interface.PROMPT):
        raise ValueError('unchanged_v2_prompt_required')
    resolver, texts, source = challenge.load_inputs(BANK)
    # This loader has no reference fields; preparation never opens answer keys.
    for row in resolver:
        text = texts[row['report_sha256']]
        interface.request_messages(text, interface.build_inventory(text))
    sources = {**frozen.sources_for_program(), 'authored_worker': Path(__file__),
        'span_v2_interface': ROOT/'interfaces/report_span_selection_v2.py',
        'span_v1_inventory_decoder': ROOT/'interfaces/report_span_selection.py',
        'authored_tests': ROOT/'tests/test_authored_report_spans.py',
        'authored_protocol': ROOT.parent/'docs/authored_report_span_protocol.md',
        'bank_manifest': BANK/'manifest.json', 'blinded_resolver': BANK/'resolver.jsonl',
        'model_plan_manifest': MODEL_PLAN/'manifest.json', 'model_plan': MODEL_PLAN/'plan.json',
        'baseline_manifest': BASELINE/'manifest.json'}
    # Pin shared runtime/interface source code to the already-tested V2 plan.
    prior_hashes = {str(Path(path).resolve()): parent_manifest['source_sha256'][key]
        for key, path in parent_manifest['source_paths'].items()}
    for path in sources.values():
        expected = prior_hashes.get(str(path.resolve()))
        if expected is not None and sha256_file(path) != expected:
            raise ValueError('unchanged_frozen_runtime_required')
    sources.update({'authored_text_'+row['item_id']: BANK/row['report_path'] for row in resolver})
    model = Path(parent['model_path'])
    if any([(model/name).stat().st_size, (model/name).stat().st_mtime_ns] != value
            for name, value in parent['model_file_stats'].items()):
        raise ValueError('frozen_model_metadata_changed')
    plan = {'schema_version': SCHEMA+'-plan', 'source': source, 'bank_path': str(BANK),
        'fixed_reports': 56, 'designated_checks': 80, 'full_vector_checks_secondary': 224,
        'interface_version': interface.VERSION, 'prompt_sha256': parent['prompt_sha256'],
        'model_path': parent['model_path'], 'model_file_sha256': parent['model_file_sha256'],
        'model_file_stats': parent['model_file_stats'], 'max_model_calls': 56, 'model_retries': 0,
        'max_new_tokens': 512, 'seed': 0, 'do_sample': False, 'min_vram_gib': 24,
        'reference_sha256': bank_manifest['artifacts']['references.jsonl']['sha256'],
        'answer_key_opened_in_prepare': False, 'new_model_calls': 0,
        'expert_reviewed': False, 'untouched_final_test': False, 'primary_metric_eligible': False,
        'selection_changed': False, 'regeneration_authorized': False, 'real_patient_inputs_used': False,
        'interpretation': 'Existing same-author language fixtures; expected states reflect disclosed annotation policy, not clinical or independent span truth. No post-result rule/prompt tuning.'}
    return plan, sources


def load_plan(root):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    manifest = read_json(root/'manifest.json')
    if manifest['schema_version'] != SCHEMA+'-plan':
        raise ValueError('sealed_authored_plan_required')
    sources = frozen.checked_sources(manifest)
    if sha256_file(root/'plan.json') != manifest['artifacts']['plan.json']['sha256']:
        raise ValueError('sealed_plan_artifact_required')
    plan = read_json(root/'plan.json')
    rebuilt, rebuilt_sources = prepare()
    if plan != rebuilt or sources != {key: path.resolve() for key, path in rebuilt_sources.items()}:
        raise ValueError('exact_frozen_authored_protocol_required')
    sources.update(plan_manifest=root/'manifest.json', sealed_plan=root/'plan.json')
    return plan, sources


def sanitized_decode(response, text, *, token_limit_reached=False):
    decoded = interface.decode_response(response, text, interface.build_inventory(text),
        token_limit_reached=token_limit_reached)
    complete = decoded['contract_status'] == 'complete'
    return {'status': 'complete' if complete else 'failed_unavailable',
        'contract_failure_reason': decoded['contract_failure_reason'],
        'finding_states': {f: value['state'] for f, value in decoded['findings'].items()} if complete else None,
        'source_span_references': {f: {polarity: [
            {key: span[key] for key in ('span_id', 'char_start', 'char_end', 'quote_sha256', 'offset_unit')}
            for span in spans] for polarity, spans in value['evidence'].items()}
            for f, value in decoded['findings'].items()} if complete else None,
        'semantic_correctness_independently_verified': False}


def metrics_for_records(references, records):
    compatible = []
    for row in records:
        if row['status'] == 'failed_unavailable':
            if row['finding_states'] is not None:
                raise ValueError('unavailable_is_not_unknown_prediction')
            compatible.append({**row, 'finding_states': dict.fromkeys(interface.FINDINGS, 'unknown')})
        elif row['status'] == 'complete':
            if not isinstance(row['finding_states'], dict):
                raise ValueError('complete_four_states_required')
            compatible.append(row)
        else:
            raise ValueError('explicit_prediction_status_required')
    # Legacy evaluator counts failed rows as unavailable, never unknown matches.
    return challenge.evaluate_predictions(references, compatible)


def compare_after_freeze(plan, resolver, records):
    references, reference_path = challenge.load_references(BANK, resolver)
    if sha256_file(reference_path) != plan['reference_sha256']:
        raise ValueError('fixed_authored_reference_required')
    metrics, details = metrics_for_records(references, records)
    baseline_manifest = pinned_manifest(BASELINE, BASELINE_SHA)
    path = BASELINE/'predictions.json'
    if sha256_file(path) != baseline_manifest['artifacts']['predictions.json']['sha256']:
        raise ValueError('fixed_blinded_baseline_required')
    baseline = read_json(path)
    if (baseline['source'] != plan['source'] or baseline['model_received_reference_states'] is not False
            or baseline['scorer_read_reference_key'] is not False or baseline['producer']['frozen'] is not True):
        raise ValueError('same_input_blinded_baseline_required')
    baseline_metrics, _ = challenge.evaluate_predictions(references, baseline['records'])
    summary = {'schema_version': SCHEMA, **metrics, 'chexbert_same_input': baseline_metrics,
        'contract_status_counts': dict(Counter(r['status'] for r in records)),
        'complete_all_unknown_reports': sum(r['status'] == 'complete' and
            set(r['finding_states'].values()) == {'unknown'} for r in records),
        'contract_failure_reasons': dict(Counter(r['contract_failure_reason'] for r in records
            if r['contract_failure_reason'] is not None)),
        'expert_reviewed': False, 'independent_clinical_accuracy': None, 'primary_metric_eligible': False,
        'selection_changed': False, 'regeneration_authorized': False, 'real_patient_inputs_used': False,
        'reference_or_baseline_read_after_prediction_fsync': True,
        'model_received_references_baseline_or_scores': False, 'interpretation': plan['interpretation']}
    return summary, details, {'authored_references': reference_path, 'chexbert_predictions': path}


def markdown(summary):
    lines = ['# V2 authored-language diagnostic / 虚构文本核查', '',
        '56 existing invented texts; 80 designated checks; 224 full-vector checks secondary.', '',
        '| Readout | Matches / checks | Macro F1 | Positive/negative flips | Determinate on uncertain/unknown |',
        '|---|---:|---:|---:|---:|']
    for name, result in (('Qwen span V2', summary['designated_targets']),
            ('Frozen CheXbert', summary['chexbert_same_input']['designated_targets'])):
        score = result['macro_f1_present_classes']
        lines.append(f"| {name} | {result['correct']}/{result['checks']} | {score if score is not None else 'NA'} | {result['hard_positive_negative_flips']} | {result['unsafe_commit_on_unknown_or_uncertain']} |")
    lines += ['', '## Per-family diagnostic / 按语言类型', '',
        '| Family | Matches / checks | Unavailable |', '|---|---:|---:|']
    for family, result in summary['per_family'].items():
        lines.append(f"| {family} | {result['correct']}/{result['checks']} | {result['unavailable']} |")
    lines += ['', summary['interpretation'], '',
        'Unknown is not negative; failed availability has no state prediction. Valid span IDs do not establish semantic correctness.',
        '所有原始病例、规则、prompt、权重、score 和 winner 保持不变；不以这份结果自动重生成。', '']
    return '\n'.join(lines)


def run(plan_root, temporary, *, approved=False):
    require_gpu_approval(approved)
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24*1024**3:
        raise RuntimeError('gpu_with_at_least_24_gib_required')
    plan, sources = load_plan(plan_root)
    initial = {key: sha256_file(path) for key, path in sources.items()}
    model_path = Path(plan['model_path'])
    if {name: sha256_file(model_path/name) for name in frozen.MODEL_FILES} != plan['model_file_sha256']:
        raise ValueError('fixed_frozen_model_assets_required')
    resolver, texts, source = challenge.load_inputs(BANK)
    if source != plan['source']:
        raise ValueError('fixed_blinded_input_source_required')
    torch.manual_seed(plan['seed'])
    torch.cuda.reset_peak_memory_stats()
    records, calls = [], 0
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        from tricompose.verifiers.qwenvl_cxr_report import _load_model
        from verify_candidate_findings_qwen import infer
        model, processor, model_type = _load_model(model_path, torch,
            min_pixels=256*28*28, max_pixels=512*28*28)
        model.eval().requires_grad_(False)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('frozen_weights_required')
        dtype = str(next(model.parameters()).dtype)
        for row in resolver:
            if calls >= plan['max_model_calls']:
                raise ValueError('call_budget_exceeded')
            text = texts[row['report_sha256']]
            messages = interface.request_messages(text, interface.build_inventory(text))
            started = time.monotonic()
            response, tokens = infer(messages, model, processor, torch, max_new_tokens=plan['max_new_tokens'])
            calls += 1
            torch.cuda.synchronize()
            records.append({'item_id': row['item_id'], 'report_sha256': row['report_sha256'],
                **sanitized_decode(response, text, token_limit_reached=tokens['token_limit_reached']),
                'input_text_sha256': interface.digest(messages[0]['content'][0]['text']),
                'response_sha256': interface.digest(response), **tokens,
                'elapsed_seconds': round(time.monotonic()-started, 4)})
    # Close and fsync predictions BEFORE parsing references or old predictions.
    path = write_private_json(temporary/'predictions.json', {'schema_version': SCHEMA,
        'records': records, 'model_received_references_baseline_or_scores': False})
    with path.open('rb') as handle:
        os.fsync(handle.fileno())
    prediction_hash = sha256_file(path)
    summary, details, analysis_sources = compare_after_freeze(plan, resolver, records)
    if sha256_file(path) != prediction_hash or any(sha256_file(p) != initial[k] for k, p in sources.items()):
        raise ValueError('immutable_predictions_or_inputs_changed')
    if any([(model_path/name).stat().st_size, (model_path/name).stat().st_mtime_ns] != value
            for name, value in plan['model_file_stats'].items()):
        raise ValueError('frozen_model_metadata_changed')
    summary.update(model_calls=calls, model_retries=0, prompt_sha256=plan['prompt_sha256'],
        interface_version=interface.VERSION, external_api_used=False,
        peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
        execution={'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu_name': torch.cuda.get_device_name(0),
            'dtype': dtype, 'torch_version': str(torch.__version__), 'model_type': model_type})
    sources.update(analysis_sources)
    return [('summary.json', summary), ('details.json', {'records': details}),
        ('RESULTS_CN_EN.md', markdown(summary))], sources


def execute(args):
    if args.mode == 'run':
        require_gpu_approval(args.allow_authored_language_benchmark)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        if args.mode == 'prepare':
            plan, sources = prepare()
            outputs, schema = [('plan.json', plan)], SCHEMA+'-plan'
        else:
            outputs, sources = run(args.plan_run, temporary, approved=args.allow_authored_language_benchmark)
            schema = SCHEMA
        hashes = {key: sha256_file(path) for key, path in sources.items()}
        for name, value in outputs:
            (write_private_text if name.endswith('.md') else write_private_json)(temporary/name, value)
        write_private_json(temporary/'manifest.json', {'schema_version': schema, 'run_id': args.run_id,
            'source_paths': {key: str(path.resolve()) for key, path in sources.items()}, 'source_sha256': hashes,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in sorted(temporary.iterdir())},
            'real_patient_inputs_used': False, 'primary_metric_eligible': False,
            'selection_changed': False, 'regeneration_authorized': False})
        if any(sha256_file(path) != hashes[key] for key, path in sources.items()):
            raise ValueError('immutable_sources_changed_before_commit')
        for path in (temporary, *temporary.iterdir()):
            stat = path.stat()
            if stat.st_gid not in (96293, 65534) or stat.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--plan-run', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-authored-language-benchmark', action='store_true')
    args = parser.parse_args()
    os.umask(0o007)
    started = time.monotonic()
    try:
        target = execute(args)
    except Exception as exc:
        print(json.dumps({'status': 'failed_closed_authored_span_diagnostic', 'error_type': type(exc).__name__}))
        return 2
    print(json.dumps({'status': 'completed_'+args.mode, 'elapsed_seconds': round(time.monotonic()-started, 3),
        'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
