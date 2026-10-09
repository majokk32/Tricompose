#!/usr/bin/env python3
"""Prepare an authored-only plan; separately approved paired GPU diagnostic.

Never opens real inputs or the synthetic candidate bank. Old consumed sources
remain unchanged. Valid source IDs and authored accuracy are not clinical gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'interfaces'), str(ROOT / 'tools'), str(ROOT / 'benchmarks'),
    str(ROOT.parent / 'src'), str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
import opacity_assertion_stages_v3 as stages
import check_opacity_report_assertions as legacy
from contracts import (PROTECTED_ROOT, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
SCHEMA = 'tricompose-authored-opacity-two-stage-v3-format-only'
TESTS = ROOT / 'tests/test_opacity_assertion_stages_v3.py'
PROTOCOL = ROOT.parent / 'docs/opacity_assertion_stages_v3_protocol.md'
FIXTURES = ROOT / 'benchmarks/opacity_assertion_controls_v2.py'
POLICY = {'reports': 48, 'families': 12, 'replay_item_indices': [0, 1],
    'max_model_calls': 148, 'legacy_primary_calls': 48, 'locator_primary_calls': 48,
    'polarity_primary_max_calls': 48, 'staged_replay_max_calls': 4,
    'locator_max_new_tokens': 128, 'polarity_max_new_tokens': 512,
    'legacy_max_new_tokens': 512, 'do_sample': False, 'seed': 0,
    'format_only_fix_after_v2_result': True, 'retry_failed': False, 'input_source': 'wholly_authored_nonpatient_text',
    'clinical_accuracy_verified': False, 'primary_metric_eligible': False,
    'regeneration_authorized': False, 'selector_enabled': False,
    'dtype_policy': 'unchanged_frozen_loader_including_bf16_emulation',
    'passed_language_gate_only_all_48_states_and_ids_exact_with_stable_replays': True}


def dump_private(path, payload):
    write_private_json(path, payload)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def pinned_sources():
    paths = [Path(__file__), Path(stages.__file__),
        ROOT / 'interfaces/opacity_assertion_stages_v2.py',
        ROOT / 'tools/benchmark_opacity_assertion_stages_v2.py', FIXTURES, TESTS, PROTOCOL,
        Path(legacy.__file__), legacy.TESTS, legacy.PROTOCOL,
        Path(legacy.registry.__file__), Path(legacy.registry.evidence.__file__), Path(legacy.OP.__file__)]
    pins = {str(p): sha256_file(p) for p in paths}
    runtime = legacy.runtime_metadata(pins)
    return pins, runtime


def prepare(args):
    legacy.OP.guard()
    from opacity_assertion_controls_v2 import cases, VERSION
    inputs, refs = [], []
    for case in cases():
        inventory = stages.segments(case['text'])
        stages.validate_ids(case['relevant_segment_ids'], inventory)
        inputs.append({k: case[k] for k in ('item_id', 'text')})
        inputs[-1].update(report_sha256=stages.digest(case['text']), segments=inventory)
        refs.append({k: case[k] for k in
            ('item_id', 'family', 'expected_state', 'relevant_segment_ids', 'legacy_control_id')})
        refs[-1]['report_sha256'] = inputs[-1]['report_sha256']
    controls = {r['control_id']: r for r in legacy.controls()}
    marked = [r for r in refs if r['legacy_control_id']]
    if {r['legacy_control_id'] for r in marked} != set(controls):
        raise ValueError('all_six_known_prior_controls_required')
    for ref in marked:
        control = controls[ref['legacy_control_id']]
        source = next(r for r in inputs if r['item_id'] == ref['item_id'])
        if source['text'] != control['text'] or ref['expected_state'] != control['expected_state']:
            raise ValueError('unchanged_prior_control_required')
    pins, runtime = pinned_sources()
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        dump_private(temporary / 'inputs.json', {'records': inputs})
        dump_private(temporary / 'references.json', {'records': refs})
        plan = {'schema_version': SCHEMA + '-plan', 'fixture_version': VERSION,
            'interface_version': stages.VERSION, 'policy': POLICY, 'runtime': runtime,
            'prompt_sha256s': {'locator': stages.digest(stages.LOCATOR_PROMPT),
                'polarity': stages.digest(stages.POLARITY_PROMPT), 'legacy': stages.digest(legacy.PROMPT)},
            'input_sha256': sha256_file(temporary / 'inputs.json'),
            'reference_sha256': sha256_file(temporary / 'references.json'),
            'new_model_calls': 0, 'patient_or_candidate_inputs_opened': False,
            'authored_references_known_to_investigator': True, 'heldout_test': False}
        dump_private(temporary / 'plan.json', plan)
        legacy.OP.verify_pins(pins)
        dump_private(temporary / 'manifest.json', {'schema_version': SCHEMA + '-plan-manifest',
            'sources': pins, 'artifacts': {name: sha256_file(temporary / name)
                for name in ('inputs.json', 'references.json', 'plan.json')}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'prepared_authored_only_no_inference', 'new_model_calls': 0}


def unavailable(reason):
    return {'status': 'failed_unavailable', 'state': None,
            'failure_reason': reason, 'semantic_correctness_verified': False}


def baseline_outcome(response, text, tokens):
    decoded = legacy.decode_evidence(response, text, token_limit_reached=tokens['token_limit_reached'])
    if decoded['contract_status'] != 'complete':
        return unavailable(decoded['contract_failure_reason'])
    return {'status': 'complete', 'state': decoded['state'], 'failure_reason': None,
        'legacy_opposed_quoted_assertions': decoded['opposed_quoted_assertions'],
        'legacy_evidence': decoded['evidence'], 'semantic_correctness_verified': False}


def run_staged(item, call, prefix):
    # Always keep explicit stage availability and actual attempts, including fails.
    text, inventory = item['text'], item['segments']
    selected, assertions = None, None
    stage_calls = {'locator': 0, 'polarity': 0}
    raw_a = raw_b = None
    try:
        messages = stages.locator_messages(text, inventory)
        stage_calls['locator'] = 1
        raw_a = call(messages, prefix + '_locator', POLICY['locator_max_new_tokens'])
        selected = stages.decode_locator(raw_a['response'], inventory,
            token_limit_reached=raw_a['token_limit_reached'])
        if selected:
            messages = stages.polarity_messages(text, inventory, selected)
            stage_calls['polarity'] = 1
            raw_b = call(messages, prefix + '_polarity', POLICY['polarity_max_new_tokens'])
            assertions = stages.decode_polarity(raw_b['response'], inventory, selected,
                token_limit_reached=raw_b['token_limit_reached'])
        else:
            assertions = []
        result = {'status': 'complete', 'state': stages.reduce_states(assertions),
            'failure_reason': None, 'semantic_correctness_verified': False}
    except Exception as error:
        # Controlled decoder errors contain only fixed reasons; model errors never text.
        reason = str(error) if type(error) is ValueError and str(error) in {
            'token_limit_reached', 'text_response_required', 'invalid_json_or_duplicate_key',
            'object_response_required', 'locator_inventory_mismatch',
            'unique_sorted_bound_segment_ids_required', 'polarity_inventory_mismatch',
            'exhaustive_selected_assertions_required', 'four_state_assertions_required',
            'bounded_nonempty_source_required', 'bounded_segment_inventory_required',
            'exact_source_segment_inventory_required', 'empty_selection_has_no_polarity_call'} else type(error).__name__
        result = unavailable(reason)
    result.update(selected_segment_ids=selected, assertions=assertions,
        source_span_references=stages.evidence(inventory, selected) if selected is not None else None,
        stage_model_calls=stage_calls,
        stage_response_sha256s={'locator': stages.digest(raw_a['response']) if raw_a else None,
            'polarity': stages.digest(raw_b['response']) if raw_b else None})
    return result


def metrics(rows):
    matrix = {s: dict.fromkeys((*stages.STATES, 'unavailable'), 0) for s in stages.STATES}
    for row in rows:
        matrix[row['expected_state']][row['state'] if row['status'] == 'complete' else 'unavailable'] += 1
    f1 = {}
    for state in stages.STATES:
        support = sum(matrix[state].values())
        predicted = sum(matrix[other][state] for other in stages.STATES)
        denominator = support + predicted
        f1[state] = 2 * matrix[state][state] / denominator if denominator else None
    return {'checks': len(rows), 'complete': sum(r['status'] == 'complete' for r in rows),
        'unavailable': sum(r['status'] != 'complete' for r in rows),
        'exact_matches': sum(r['status'] == 'complete' and r['state'] == r['expected_state'] for r in rows),
        'accuracy_all_attempted': sum(r['status'] == 'complete' and r['state'] == r['expected_state'] for r in rows) / len(rows) if rows else None,
        'confusion': matrix, 'per_class_f1': f1,
        'macro_f1_present_classes': sum(v for v in f1.values() if v is not None) / sum(v is not None for v in f1.values()) if any(v is not None for v in f1.values()) else None,
        'hard_positive_negative_flips': sum(r['status'] == 'complete' and
            {r['state'], r['expected_state']} == {'positive', 'negative'} for r in rows),
        'determinate_on_uncertain_unknown': sum(r['status'] == 'complete' and
            r['expected_state'] in ('uncertain', 'unknown') and r['state'] in ('positive', 'negative') for r in rows)}


def score(references, predictions, replays):
    if len(references) != 48 or len(predictions) != 48 or \
            len({r['item_id'] for r in references}) != 48 or len({r['item_id'] for r in predictions}) != 48:
        raise ValueError('all_48_reference_prediction_pairs_required')
    checks = []
    for ref, pred in zip(references, predictions):
        if any(ref[k] != pred[k] for k in ('item_id', 'report_sha256')):
            raise ValueError('exact_authored_join_required')
        if ref['expected_state'] not in stages.STATES:
            raise ValueError('four_state_authored_reference_required')
        for method in ('legacy', 'staged'):
            outcome = pred[method]
            if outcome['status'] not in ('complete', 'failed_unavailable') or \
                    (outcome['status'] == 'complete' and outcome['state'] not in stages.STATES) or \
                    (outcome['status'] != 'complete' and outcome['state'] is not None):
                raise ValueError('unavailable_not_unknown_required')
            checks.append({**ref, 'method': method, 'status': outcome['status'],
                'state': outcome['state'], 'failure_reason': outcome['failure_reason'],
                'locator_exact_authored_ids': outcome.get('selected_segment_ids') == ref['relevant_segment_ids']
                    if method == 'staged' and outcome.get('selected_segment_ids') is not None else None})
    readouts = {method: metrics([r for r in checks if r['method'] == method]) for method in ('legacy', 'staged')}
    staged = [r for r in checks if r['method'] == 'staged']
    locator_matches = sum(r['locator_exact_authored_ids'] is True for r in staged)
    stable = len(replays) == 2 and all(r['same_complete_state_and_ids'] is True for r in replays)
    return {'paired_authored_metrics': readouts,
        'per_family': {family: {method: metrics([r for r in checks if r['method'] == method and r['family'] == family])
            for method in ('legacy', 'staged')} for family in sorted({r['family'] for r in references})},
        'known_six_controls': {method: metrics([r for r in checks if r['method'] == method and r['legacy_control_id']])
            for method in ('legacy', 'staged')},
        'locator_exact_authored_ids': locator_matches, 'locator_checks': 48,
        'language_gate_passed': readouts['staged']['exact_matches'] == 48 and locator_matches == 48 and stable,
        'language_gate_is_not_clinical_qualification': True}, checks


def evaluate(args):
    legacy.OP.guard(gpu=True, approved=args.allow_authored_text_benchmark)
    target = require_inside(Path(args.output_root) / args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('output_run_exists')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('offline_runtime_required')
    pins = {}
    root = require_inside(args.plan_root, PROTECTED_ROOT, must_exist=True)
    manifest = legacy.OP.metadata(root / 'manifest.json', pins, args.plan_manifest_sha256)
    legacy.OP.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    plan = legacy.OP.metadata(root / 'plan.json', pins, manifest['artifacts']['plan.json'])
    inputs = legacy.OP.metadata(root / 'inputs.json', pins, manifest['artifacts']['inputs.json'])['records']
    prompts = {'locator': stages.digest(stages.LOCATOR_PROMPT),
        'polarity': stages.digest(stages.POLARITY_PROMPT), 'legacy': stages.digest(legacy.PROMPT)}
    if plan['schema_version'] != SCHEMA + '-plan' or plan['policy'] != POLICY or \
            plan['prompt_sha256s'] != prompts or plan['interface_version'] != stages.VERSION or \
            plan['runtime'] != legacy.runtime_metadata({}) or \
            plan['input_sha256'] != manifest['artifacts']['inputs.json'] or \
            plan['reference_sha256'] != manifest['artifacts']['references.json'] or \
            len(inputs) != 48 or len({r['item_id'] for r in inputs}) != 48:
        raise ValueError('fixed_authored_only_plan_required')
    for item in inputs:
        if set(item) != {'item_id', 'text', 'report_sha256', 'segments'} or \
                stages.digest(item['text']) != item['report_sha256']:
            raise ValueError('blinded_hash_bound_authored_input_required')
        stages.validate_inventory(item['text'], item['segments'])
    import torch
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
        raise RuntimeError('allocated_gpu_at_least_24g_required')
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    started, predictions, raw, replay, calls = time.monotonic(), [], [], [], Counter()
    try:
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            model, processor, model_type = _load_model(legacy.MODEL, torch, min_pixels=256*28*28, max_pixels=512*28*28)
            model.eval().requires_grad_(False)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError('model_not_frozen')
            dtype = str(next(model.parameters()).dtype)

            def call(messages, call_id, max_new_tokens):
                phase = call_id.rsplit('_', 1)[1]
                calls[phase] += 1
                response, tokens = legacy.infer(messages, model, processor, torch, max_new_tokens=max_new_tokens)
                record = {'call_id': call_id, 'response': response, 'max_new_tokens': max_new_tokens, **tokens}
                raw.append(record)
                return record

            for index, item in enumerate(inputs):
                baseline_calls = 0
                try:
                    messages = legacy.request_messages(item['text'])
                    baseline_calls = 1
                    output = call(messages, item['item_id'] + '_legacy', 512)
                    original = baseline_outcome(output['response'], item['text'], output)
                    original['response_sha256'] = stages.digest(output['response'])
                except Exception as error:
                    original = unavailable(type(error).__name__)
                original['model_calls'] = baseline_calls
                staged = run_staged(item, call, item['item_id'])
                predictions.append({k: item[k] for k in ('item_id', 'report_sha256')})
                predictions[-1].update(legacy=original, staged=staged)
                if index in POLICY['replay_item_indices']:
                    repeated = run_staged(item, call, 'replay_' + item['item_id'])
                    same = all(r['status'] == 'complete' for r in (staged, repeated)) and all(
                        staged[k] == repeated[k] for k in ('state', 'selected_segment_ids', 'assertions'))
                    replay.append({'item_id': item['item_id'], 'same_complete_state_and_ids': same,
                        'same_stage_response_sha256s': staged['stage_response_sha256s'] == repeated['stage_response_sha256s'],
                        'outcome': repeated})
            torch.cuda.synchronize()
        dump_private(temporary / 'predictions.json', {'schema_version': SCHEMA, 'records': predictions})
        dump_private(temporary / 'raw_responses.json', {'records': raw})
        dump_private(temporary / 'replay_checks.json', {'records': replay})
        closed = {name: sha256_file(temporary / name) for name in
            ('predictions.json', 'raw_responses.json', 'replay_checks.json')}
        dump_private(temporary / 'prediction_freeze_receipt.json', {'sha256': closed,
            'authored_reference_semantics_read_before_prediction_fsync': False})
        # No authored keys/families/legacy expected states were sent to the model.
        references = legacy.OP.metadata(root / 'references.json', pins, plan['reference_sha256'])['records']
        readout, checks = score(references, predictions, replay)
        if any(sha256_file(temporary / name) != digest for name, digest in closed.items()):
            raise ValueError('closed_predictions_changed')
        summary = {'schema_version': SCHEMA, **readout, 'policy': POLICY,
            'authored_texts': 48, 'model_calls': sum(calls.values()), 'calls_by_phase': dict(calls),
            'primary_legacy_model_calls': sum(r['legacy']['model_calls'] for r in predictions),
            'primary_locator_model_calls': sum(r['staged']['stage_model_calls']['locator'] for r in predictions),
            'primary_polarity_model_calls': sum(r['staged']['stage_model_calls']['polarity'] for r in predictions),
            'replay_model_calls': sum(sum(r['outcome']['stage_model_calls'].values()) for r in replay),
            'successful_raw_responses': len(raw), 'staged_replays': len(replay),
            'model_type': model_type, 'dtype': dtype, 'gpu_name': torch.cuda.get_device_name(0),
            'elapsed_seconds_before_serialization': round(time.monotonic() - started, 6),
            'peak_allocated_vram_gib_including_load': round(torch.cuda.max_memory_allocated()/1024**3, 3),
            'reference_read_after_prediction_fsync': True,
            'model_received_reference_family_scores_ehr_images_or_case_ids': False,
            'investigator_known_authored_development_set': True, 'clinical_accuracy': None,
            'primary_metric_eligible': False, 'clinical_fault_localization': False,
            'old_scores_and_choices_changed': False, 'candidate_report_text_read': False,
            'patient_inputs_read': False, 'generation_calls': 0, 'training_calls': 0, 'selector_calls': 0}
        if summary['model_calls'] > POLICY['max_model_calls']:
            raise ValueError('fixed_call_cap_exceeded')
        dump_private(temporary / 'scored_checks.json', {'records': checks})
        dump_private(temporary / 'summary.json', summary)
        lines = ['# Authored opacity diagnostic / 虚构文本核查', '',
            'Same-author development tests, not clinical gold or repaired outputs.', '',
            '| Method | State matches / 48 | Unavailable | Hard flips | Determinate on uncertain/unknown |',
            '| --- | ---: | ---: | ---: | ---: |']
        for method, values in summary['paired_authored_metrics'].items():
            lines.append(f"| {method} | {values['exact_matches']}/48 | {values['unavailable']} | {values['hard_positive_negative_flips']} | {values['determinate_on_uncertain_unknown']} |")
        lines += ['', f"Exact authored locator IDs: {summary['locator_exact_authored_ids']}/48.",
            f"Narrow language gate passed: {summary['language_gate_passed']}.",
            'No image/EHR/report regeneration, primary score promotion or fault verdict.', '']
        write_private_text(temporary / 'RESULTS_CN_EN.md', '\n'.join(lines))
        legacy.OP.verify_pins(pins)
        dump_private(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'sources': pins, 'runtime': plan['runtime'], 'plan_manifest_sha256': args.plan_manifest_sha256,
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'completed_authored_assertion_diagnostic_no_clinical_repair',
        'elapsed_seconds_before_serialization': summary['elapsed_seconds_before_serialization'],
        'peak_allocated_vram_gib_including_load': summary['peak_allocated_vram_gib_including_load']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    parser.add_argument('--plan-root')
    parser.add_argument('--plan-manifest-sha256')
    parser.add_argument('--allow-authored-text-benchmark', action='store_true')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    try:
        target, status = prepare(args) if args.action == 'prepare' else evaluate(args)
        print(json.dumps({**status, 'manifest_sha256': sha256_file(target / 'manifest.json')}, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

