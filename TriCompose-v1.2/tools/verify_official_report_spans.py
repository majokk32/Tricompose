#!/usr/bin/env python3
"""Frozen V2 versus manual report labels; real text only inside approved GPU Slurm.

Preparation reads existing aggregate metadata and source-code/model metadata,
never original annotation/linkage rows or reports. Run exports opaque derived
states, span offsets/hashes and aggregate metrics, never source text/keys/paths,
reference rows or raw model responses. No ranking, calibration or regeneration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for directory in (ROOT/'real_validation', ROOT/'benchmarks', ROOT/'src',
        WORKSPACE/'src', WORKSPACE/'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(directory))
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
import official_report_benchmark as gold_method
import run_official_report_benchmark as gold_runtime
import verify_legacy_report_semantics as frozen

INTERFACE_PATH = ROOT/'interfaces/report_span_selection_v2.py'
spec = importlib.util.spec_from_file_location('official_span_frozen_v2', INTERFACE_PATH)
interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interface)
SCHEMA = 'tricompose-official-report-span-v2-diagnostic-v1'
PLAN_SCHEMA = SCHEMA+'-plan'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
AUDIT = BASE/'real_validation/official_report_gold_coverage_12576792_source1'
SPAN_PLAN = BASE/'report_span_v2_plans/span_v2_scope2_12645021_001'
CHEXBERT = BASE/'real_validation/official_chexbert_reports_12594397'
AUDIT_SHA = 'bf0b5064c5a477bc673196080e2c16b2dc7dc15672f6cf589161761ddfe77fc4'
SPAN_PLAN_SHA = '576483c1b4f544b7fd99f5b6e7aca94d8c22c83694294fd41cd3ab9d1aa74afc'
CHEXBERT_SHA = 'aeea0df50e73c4b57af35d3d686ee5d3677f1663174cf28a570205f5248deec6'
DATASET = WORKSPACE.parent/'datasets'
GOLD = DATASET/'vlm_radiology_report_generation/mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.1.0-test-set-labeled.csv'
LINKAGE = DATASET/'three_modalities/v2_labs_vitals/manifest.csv'
FINDINGS = interface.FINDINGS
STATES = gold_method.STATES
SAFE_FAILURE_CODES = frozenset({'bounded_private_json_required', 'private_json_object_required',
    'fixed_parent_manifest_required', 'fixed_parent_artifact_required', 'fixed_metadata_and_interface_required',
    'frozen_model_metadata_changed', 'sealed_official_span_plan_required', 'sealed_plan_artifact_changed',
    'fixed_official_span_protocol_required', 'exact_prediction_inventory_required', 'exact_baseline_inventory_required',
    'valid_complete_four_states_required', 'unavailable_is_not_unknown_prediction', 'same_impression_bytes_required',
    'gpu_with_at_least_24_gib_required', 'fixed_reference_sources_changed', 'frozen_model_asset_hash_changed',
    'fixed_reference_inventory_required', 'call_budget_exceeded', 'frozen_weights_required',
    'fixed_chexbert_predictions_changed', 'immutable_inputs_or_program_changed', 'refuse_existing_official_span_run',
    'immutable_program_changed_before_commit', 'approved_real_report_slurm_required', 'slurm_cgroup_required'})


def json_file(path, limit=4*1024*1024):
    path = require_inside(path, PROTECTED_ROOT, must_exist=True)
    if path.stat().st_size > limit:
        raise ValueError('bounded_private_json_required')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('private_json_object_required')
    return value


def program_sources():
    return {**frozen.sources_for_program(),
        'official_span_worker': Path(__file__), 'span_v2_interface': INTERFACE_PATH,
        'span_v1_inventory_decoder': ROOT/'interfaces/report_span_selection.py',
        'official_gold_helper': ROOT/'real_validation/official_report_benchmark.py',
        'official_read_helper': ROOT/'real_validation/run_official_report_benchmark.py',
        'official_metadata_helper': ROOT/'real_validation/audit_official_report_gold.py'}


def prepare():
    # Only protected aggregate metadata. Never open original CSV rows here.
    for root, expected in ((AUDIT, AUDIT_SHA), (SPAN_PLAN, SPAN_PLAN_SHA), (CHEXBERT, CHEXBERT_SHA)):
        if sha256_file(root/'manifest.json') != expected:
            raise ValueError('fixed_parent_manifest_required')
    audit_manifest, span_manifest = json_file(AUDIT/'manifest.json'), json_file(SPAN_PLAN/'manifest.json')
    for root, manifest, name in ((AUDIT, audit_manifest, 'summary.json'), (SPAN_PLAN, span_manifest, 'plan.json')):
        if sha256_file(root/name) != manifest['artifacts'][name]['sha256']:
            raise ValueError('fixed_parent_artifact_required')
    audit, parent = json_file(AUDIT/'summary.json'), json_file(SPAN_PLAN/'plan.json')
    if (audit['status'] != 'metadata_coverage_audit_only_not_model_validation'
            or audit['annotation_studies'] != 687
            or audit['splits']['test']['studies_with_nonempty_report_path'] != 27
            or parent['interface_version'] != interface.VERSION
            or parent['prompt_sha256'] != interface.digest(interface.PROMPT)):
        raise ValueError('fixed_metadata_and_interface_required')
    sources = program_sources()
    sources.update(audit_manifest=AUDIT/'manifest.json', audit_summary=AUDIT/'summary.json',
        span_parent_manifest=SPAN_PLAN/'manifest.json', span_parent_plan=SPAN_PLAN/'plan.json',
        chexbert_baseline_manifest=CHEXBERT/'manifest.json')
    stats = {name: [path.stat().st_size, path.stat().st_mtime_ns]
        for name, path in (('gold_annotation_csv', GOLD), ('matched_linkage_manifest', LINKAGE))}
    model = Path(parent['model_path'])
    if any([ (model/name).stat().st_size, (model/name).stat().st_mtime_ns] != value
            for name, value in parent['model_file_stats'].items()):
        raise ValueError('frozen_model_metadata_changed')
    plan = {'schema_version': PLAN_SCHEMA, 'interface_version': interface.VERSION,
        'prompt_sha256': interface.digest(interface.PROMPT), 'finding_order': list(FINDINGS),
        'reference_source_paths': {'gold_annotation_csv': str(GOLD), 'matched_linkage_manifest': str(LINKAGE)},
        'reference_source_sha256': audit['source_sha256'], 'reference_source_stats': stats,
        'annotation_inventory': 687, 'metadata_ready_test_reports': 27,
        'input_policy': 'unchanged_strict_impression_no_fallback',
        'cohort_policy': 'all_annotation_rows_retained_all_linked_test_reports_no_score_or_label_selection',
        'model_path': parent['model_path'], 'model_file_sha256': parent['model_file_sha256'],
        'model_file_stats': parent['model_file_stats'], 'max_model_calls': 27,
        'max_new_tokens': 512, 'min_vram_gib': 24, 'seed': 0, 'do_sample': False, 'model_retries': 0,
        'real_source_rows_or_reports_opened': False, 'new_model_calls': 0,
        'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
        'checkpoint_training_overlap_status': 'unverified',
        'annotation_policy_equivalence_status': 'unverified_strict_impression_only',
        'untouched_final_test': False, 'independent_image_ground_truth': False}
    return plan, sources


def load_plan(root):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    manifest = json_file(root/'manifest.json')
    if manifest['schema_version'] != PLAN_SCHEMA:
        raise ValueError('sealed_official_span_plan_required')
    sources = frozen.checked_sources(manifest)
    if sha256_file(root/'plan.json') != manifest['artifacts']['plan.json']['sha256']:
        raise ValueError('sealed_plan_artifact_changed')
    plan = json_file(root/'plan.json')
    rebuilt, rebuilt_sources = prepare()
    if plan != rebuilt or sources != {key: value.resolve() for key, value in rebuilt_sources.items()}:
        raise ValueError('fixed_official_span_protocol_required')
    sources.update(official_span_plan_manifest=root/'manifest.json', official_span_plan=root/'plan.json')
    return plan, sources


def sanitized_decode(response, text, inventory, *, token_limit_reached=False):
    decoded = interface.decode_response(response, text, inventory, token_limit_reached=token_limit_reached)
    complete = decoded['contract_status'] == 'complete'
    references = None
    if complete:
        references = {name: {polarity: [
            {'span_id': span['span_id'], 'char_start': span['char_start'], 'char_end': span['char_end'],
             'span_sha256': span['quote_sha256'], 'offset_unit': span['offset_unit']}
            for span in fact['evidence'][polarity]] for polarity in interface.POLARITIES}
            for name, fact in decoded['findings'].items()}
    return {'status': 'complete' if complete else 'contract_failed_unavailable',
        'contract_failure_reason': decoded['contract_failure_reason'],
        'finding_states': {name: fact['state'] for name, fact in decoded['findings'].items()} if complete else None,
        'source_span_references': references,
        'semantic_correctness_independently_verified': False}


def summarize(items, records, baseline):
    expected = {item['item_id'] for item in items}
    current = {r['item_id']: r for r in records}
    old = {r['item_id']: r for r in baseline}
    if len(expected) != len(items) or len(current) != len(records) or set(current) != expected:
        raise ValueError('exact_prediction_inventory_required')
    if len(old) != len(baseline) or set(old) != expected:
        raise ValueError('exact_baseline_inventory_required')
    for item in items:
        r = current[item['item_id']]
        states = r['finding_states']
        if r['status'] == 'complete':
            if (item['status'] != 'ready_metadata' or not r['request_ready']
                    or not isinstance(states, dict) or set(states) != set(FINDINGS)
                    or any(value not in STATES for value in states.values())):
                raise ValueError('valid_complete_four_states_required')
        elif states is not None:
            raise ValueError('unavailable_is_not_unknown_prediction')
        previous = old[item['item_id']]
        if (r.get('selected_input_sha256') is not None and previous.get('selected_input_sha256') is not None
                and r['selected_input_sha256'] != previous['selected_input_sha256']):
            raise ValueError('same_impression_bytes_required')
    complete = [item for item in items if current[item['item_id']]['status'] == 'complete']
    paired = [item for item in complete if old[item['item_id']]['status'] == 'complete']
    ready = [item for item in items if current[item['item_id']]['request_ready']]
    def metrics(selected, lookup):
        return {f: gold_method.state_statistics([(i['reference'][gold_method.SOURCE_NAMES[f]],
            lookup[i['item_id']]['finding_states'][f]) for i in selected]) for f in FINDINGS}
    ready_annotated = sum(i['reference'][gold_method.SOURCE_NAMES[f]] != 'unknown' for i in ready for f in FINDINGS)
    recovered = sum(i['reference'][gold_method.SOURCE_NAMES[f]] != 'unknown' and
        i['reference'][gold_method.SOURCE_NAMES[f]] == current[i['item_id']]['finding_states'][f]
        for i in complete for f in FINDINGS)
    return {'schema_version': SCHEMA, 'annotation_inventory': len(items),
        'metadata_status_counts': dict(Counter(i['status'] for i in items)),
        'execution_status_counts': dict(Counter(r['status'] for r in records)),
        'request_ready_reports': len(ready), 'completed_reports': len(complete),
        'failed_ready_requests': sum(current[i['item_id']]['status'] != 'complete' for i in ready),
        'contract_failure_reasons': dict(Counter(r['contract_failure_reason'] for r in records
            if r.get('contract_failure_reason') is not None)),
        'complete_all_unknown_reports': sum(all(value == 'unknown' for value in current[i['item_id']]['finding_states'].values()) for i in complete),
        'per_finding': metrics(complete, current),
        'request_ready_annotated_checks': ready_annotated,
        'request_ready_correct_annotated_states': recovered,
        'failure_aware_annotated_recovery': recovered/ready_annotated if ready_annotated else None,
        'paired_complete_reports': len(paired),
        'paired_comparison': {'qwen_v2': metrics(paired, current), 'chexbert': metrics(paired, old)},
        'reference_rows_or_source_text_exported': False, 'independent_image_ground_truth': False,
        'untouched_final_test': False, 'checkpoint_training_overlap_status': 'unverified',
        'annotation_policy_equivalence_status': 'unverified_strict_impression_only',
        'analysis_unit': 'unique_study_report_not_independent_patient',
        'patient_cluster_bootstrap_performed': False, 'primary_metric_eligible': False,
        'selection_changed': False, 'regeneration_authorized': False, 'thresholds_fitted': False,
        'interpretation': 'Manual report-label diagnostic, not span gold, image truth, untouched final testing or authorization to rank/repair. Failed inputs remain unavailable, unknown is not negative. Same-label accuracy cannot prove the selected span supports the label.'}


def markdown(summary):
    fmt = lambda value: 'NA' if value is None else f'{value:.4f}'
    lines = ['# V2 人工报告标签诊断 / V2 manual report-label diagnostic', '',
        f"Inventory {summary['annotation_inventory']}; request-ready {summary['request_ready_reports']}; complete {summary['completed_reports']}; paired {summary['paired_complete_reports']}.", '',
        '| Finding | Scored checks | Annotated checks | Annotated-state match | Hard polarity flips | Omitted assertions |',
        '|---|---:|---:|---:|---:|---:|']
    for f, m in summary['per_finding'].items():
        lines.append(f"| {f} | {m['checks']} | {m['annotated_checks']} | {fmt(m['annotated_state_accuracy'])} | {m['hard_polarity_flips']} | {m['omitted_annotated_assertions']} |")
    lines += ['', '## 相同输入的并排诊断 / Same-input comparison', '',
        '| Finding | Qwen V2 annotated-state match | CheXbert annotated-state match |', '|---|---:|---:|']
    for f in FINDINGS:
        q, c = (summary['paired_comparison'][model][f] for model in ('qwen_v2', 'chexbert'))
        lines.append(f"| {f} | {fmt(q['annotated_state_accuracy'])} | {fmt(c['annotated_state_accuracy'])} |")
    lines += ['', f"Failure-aware recovery on request-ready annotated checks: {fmt(summary['failure_aware_annotated_recovery'])}.", '',
        summary['interpretation'], '', '未输出真实报告、患者键、源路径、逐行人工标签或原始模型回复。未更改旧分数和排名。', '']
    return '\n'.join(lines)


def run(plan_root, *, allow_real_report_benchmark=False):
    gold_runtime.require_approved_slurm(argparse.Namespace(
        allow_real_report_benchmark=allow_real_report_benchmark))
    plan, sources = load_plan(plan_root)
    # Caller guards explicit approval/cgroup before this function; CUDA is also
    # checked BEFORE any original annotation/linkage rows or reports are opened.
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24*1024**3:
        raise RuntimeError('gpu_with_at_least_24_gib_required')
    input_paths = {'gold_annotation_csv': GOLD, 'matched_linkage_manifest': LINKAGE}
    if {key: sha256_file(path) for key, path in input_paths.items()} != plan['reference_source_sha256']:
        raise ValueError('fixed_reference_sources_changed')
    initial_sources = {key: sha256_file(path) for key, path in sources.items()}
    model_path = Path(plan['model_path'])
    if {name: sha256_file(model_path/name) for name in frozen.MODEL_FILES} != plan['model_file_sha256']:
        raise ValueError('frozen_model_asset_hash_changed')
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(16*1024*1024)
        with GOLD.open(encoding='utf-8-sig', newline='') as g, LINKAGE.open(encoding='utf-8-sig', newline='') as l:
            items, conditions = gold_method.collect_sources(csv.DictReader(g), csv.DictReader(l))
    finally:
        csv.field_size_limit(previous_limit)
    if (len(items) != 687 or sum(i['status'] == 'ready_metadata' for i in items) != 27
            or any(gold_method.SOURCE_NAMES[f] not in conditions for f in FINDINGS)):
        raise ValueError('fixed_reference_inventory_required')
    records, inputs, report_bindings = [], {}, {}
    for item in items:
        record = {'item_id': item['item_id'], 'status': item['status'], 'finding_states': None,
            'source_report_sha256': None, 'selected_input_sha256': None, 'request_ready': False,
            'model_call_attempted': False}
        if item['status'] == 'ready_metadata':
            text, report_hash, error = gold_runtime.read_report(item['report_path'], LINKAGE.parent)
            if error:
                record['status'] = error
            else:
                record['source_report_sha256'] = report_hash
                report_path = Path(item['report_path'])
                if not report_path.is_absolute():
                    report_path = LINKAGE.parent/report_path
                report_bindings[gold_runtime.source_file(report_path)] = report_hash
                selected, reason = gold_method.select_impression(text)
                if selected is None:
                    record['status'] = reason
                else:
                    record['selected_input_sha256'] = interface.digest(selected)
                    try:
                        inventory = interface.build_inventory(selected)
                        messages = interface.request_messages(selected, inventory)
                    except interface.SpanContractError:
                        record['status'] = 'span_input_bounds_unavailable'
                    else:
                        record.update(status='ready_input', request_ready=True)
                        inputs[item['item_id']] = (selected, inventory, messages)
        records.append(record)
    if len(inputs) > plan['max_model_calls']:
        raise ValueError('call_budget_exceeded')
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    calls = 0
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        from tricompose.verifiers.qwenvl_cxr_report import _load_model
        from verify_candidate_findings_qwen import infer
        model, processor, model_type = _load_model(model_path, torch, min_pixels=256*28*28, max_pixels=512*28*28)
        model.eval().requires_grad_(False)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('frozen_weights_required')
        dtype = str(next(model.parameters()).dtype)
        for record in records:
            if not record['request_ready']:
                continue
            selected, inventory, messages = inputs[record['item_id']]
            torch.cuda.synchronize()
            started = time.monotonic()
            response, tokens = infer(messages, model, processor, torch, max_new_tokens=512)
            calls += 1
            torch.cuda.synchronize()
            record.update(sanitized_decode(response, selected, inventory, token_limit_reached=tokens['token_limit_reached']),
                model_call_attempted=True, input_text_sha256=interface.digest(messages[0]['content'][0]['text']),
                response_sha256=interface.digest(response), elapsed_seconds=round(time.monotonic()-started, 4), **tokens)
    # Predictions are frozen before reading the old model's states for analysis.
    cm = json_file(CHEXBERT/'manifest.json')
    cp = CHEXBERT/'predictions.json'
    if sha256_file(cp) != cm['artifacts']['predictions.json']['sha256']:
        raise ValueError('fixed_chexbert_predictions_changed')
    baseline = json_file(cp)['records']
    summary = summarize(items, records, baseline)
    if (initial_sources != {key: sha256_file(path) for key, path in sources.items()}
            or {key: sha256_file(path) for key, path in input_paths.items()} != plan['reference_source_sha256']
            or any(sha256_file(path) != h for path, h in report_bindings.items())
            or any([(model_path/name).stat().st_size, (model_path/name).stat().st_mtime_ns] != value for name, value in plan['model_file_stats'].items())):
        raise ValueError('immutable_inputs_or_program_changed')
    summary.update(model_calls=calls, model_retries=0, interface_version=interface.VERSION,
        prompt_sha256=plan['prompt_sha256'], model_file_sha256=plan['model_file_sha256'],
        execution={'gpu_name': torch.cuda.get_device_name(0), 'dtype': dtype, 'model_type': model_type,
            'torch_version': str(torch.__version__), 'slurm_job_id': os.environ['SLURM_JOB_ID']},
        peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
        model_received_labels_ehr_images_or_scores=False, raw_model_responses_written=False,
        raw_reports_read_in_approved_job_only=True, external_api_used=False)
    sources['chexbert_predictions'] = cp
    return [('summary.json', summary), ('predictions.json', {'schema_version': SCHEMA, 'records': records}),
        ('RESULTS_CN_EN.md', markdown(summary))], sources


def execute(args):
    # The existing real-report guard checks actual process cgroup, not just a
    # spoofable SLURM_JOB_ID. It runs before output creation or real source reads.
    gold_runtime.require_approved_slurm(argparse.Namespace(allow_real_report_benchmark=
        args.mode == 'prepare' or args.allow_real_report_benchmark))
    target = require_inside(Path(args.output_root)/args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('refuse_existing_official_span_run')
    temporary = None
    try:
        temporary, target = new_atomic_run(args.output_root, args.run_id)
        if args.mode == 'prepare':
            plan, sources = prepare()
            outputs, schema = [('plan.json', plan)], PLAN_SCHEMA
        else:
            outputs, sources = run(args.plan_run, allow_real_report_benchmark=args.allow_real_report_benchmark)
            schema = SCHEMA
        hashes = {key: sha256_file(path) for key, path in sources.items()}
        files = [write_private_text(temporary/name, value) if name.endswith('.md')
            else write_private_json(temporary/name, value) for name, value in outputs]
        write_private_json(temporary/'manifest.json', {'schema_version': schema, 'mode': args.mode,
            'run_id': args.run_id, 'source_paths': {key: str(path) for key, path in sources.items()},
            'source_sha256': hashes, 'artifacts': {path.name: {'sha256': sha256_file(path)} for path in files},
            'patient_keys_reports_or_reference_rows_written': False, 'selection_changed': False,
            'regeneration_authorized': False, 'primary_metric_eligible': False})
        if hashes != {key: sha256_file(path) for key, path in sources.items()}:
            raise ValueError('immutable_program_changed_before_commit')
        commit_atomic_run(temporary, target)
    except BaseException:
        if temporary is not None:
            discard_atomic_run(temporary)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    for name in ('output-root', 'run-id'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--plan-run')
    parser.add_argument('--allow-real-report-benchmark', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o007)
    started = time.monotonic()
    try:
        target = execute(args)
    except Exception as exc:
        # Arbitrary exception messages can include real source data: never echo.
        code = str(exc) if str(exc) in SAFE_FAILURE_CODES else 'unclassified_failure'
        print(json.dumps({'status': 'failed_closed_official_span_diagnostic',
            'error_type': type(exc).__name__, 'error_code': code}))
        return 2
    print(json.dumps({'status': 'completed_'+args.mode, 'elapsed_seconds': round(time.monotonic()-started, 3),
        'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
