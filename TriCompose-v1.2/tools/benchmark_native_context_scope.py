#!/usr/bin/env python3
"""Frozen native/ConText veto probe on 88 authored texts; CPU Slurm only."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'benchmarks'),
               str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, sha256_file, new_atomic_run,
                       commit_atomic_run, discard_atomic_run, write_private_text)
from tricompose_v12 import native_context_scope as scope
from tricompose_v12.opacity_context import metrics
from tricompose_v12.assertion_abstention import authored_readout
import benchmark_opacity_context as frozen_context
import benchmark_chexpert_negbio_authored112 as frozen_native
import score_cached_opacity_candidates as guard

BASE = PROTECTED_ROOT/'tricompose_v1_2'
WORKER = ROOT/'interfaces/native_scope88_legacy_worker_v1.py'
FIXTURE = ROOT/'benchmarks/native_context_scope_controls_v1.py'
TEST = ROOT/'tests/test_native_context_scope.py'
PROTOCOL = WORKSPACE/'docs/native_context_scope_protocol.md'
SCHEMA = 'tricompose-native-context-scope88-v1'
dump = frozen_native.dump


def initialize_context(pins):
    nlp, runtime = frozen_context.initialize(pins)
    component = nlp.get_pipe('medspacy_context')
    component.input_span_type = 'group'
    component.span_group_name = scope.GROUP
    runtime = {**runtime, 'input_span_type': component.input_span_type,
               'span_group_name': component.span_group_name, 'target_inventory': 'exact_native_chexpert_annotations',
               'rules_or_language_configuration_changed': False}
    return nlp, runtime


def prepare(run_id):
    guard.guard()
    pins = {str(path): sha256_file(path) for path in (Path(__file__), WORKER, FIXTURE, TEST, PROTOCOL,
        Path(scope.__file__), Path(frozen_context.__file__), Path(frozen_native.__file__),
        ROOT/'interfaces/chexpert_negbio_legacy_worker_v1.py', ROOT/'src/tricompose_v12/assertion_abstention.py',
        ROOT/'src/tricompose_v12/report_assertions.py', ROOT/'src/tricompose_v12/opacity_context.py',
        ROOT/'tools/score_cached_opacity_candidates.py', WORKSPACE/'TriCompose-v1.0/eval/report_v1_1/contracts.py')}
    deployment = guard.metadata(frozen_native.DEPLOYMENT/'manifest.json', pins, frozen_native.DEPLOYMENT_SHA)
    guard.verify_pins(deployment['sources'])
    pins.update(deployment['sources'])
    _, context_runtime = initialize_context(pins)  # Initialization only, no text parsing.
    from native_context_scope_controls_v1 import cases
    rows = cases()
    inputs = {'records': [{key: row[key] for key in ('item_id', 'text')} for row in rows]}
    for row in inputs['records']:
        row['report_sha256'] = scope.digest(row['text'])
    references = {'records': [{key: row[key] for key in ('item_id', 'finding', 'family', 'expected_state')} for row in rows]}
    for ref, item in zip(references['records'], inputs['records']):
        ref['report_sha256'] = item['report_sha256']
    if len(rows) != 88 or len({row['report_sha256'] for row in inputs['records']}) != 88:
        raise ValueError('fixed_distinct_authored88_inventory_required')
    temporary, target = new_atomic_run(BASE/'native_context_scope_plans', run_id)
    try:
        dump(temporary/'inputs.json', inputs)
        dump(temporary/'references.json', references)
        dump(temporary/'plan.json', {'schema_version': SCHEMA+'-plan', 'authored_count': 88,
            'policy': scope.POLICY, 'context_runtime': context_runtime,
            'deployment_manifest_sha256': frozen_native.DEPLOYMENT_SHA,
            'author_knows_prior_development_results': True, 'clinical_gold': False,
            'authored_reference_semantics_not_in_inference_inputs': True, 'parsed_texts': 0})
        guard.verify_pins(pins)
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-plan-manifest', 'sources': pins,
            'artifacts': {path.name: sha256_file(path) for path in sorted(temporary.iterdir())}})
        frozen_native.private_permissions(temporary)
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'native_context_scope88_plan_frozen', 'authored_texts': 88,
                    'manifest_sha256': sha256_file(target/'manifest.json')}


def checks(references, native, contexts, gates):
    if not len(references) == len(native) == len(contexts) == len(gates) == 88:
        raise ValueError('all_authored_scope_rows_required')
    results = []
    for ref, raw, context, gate in zip(references, native, contexts, gates):
        if any(ref[key] != row[key] for row in (raw, context, gate) for key in ('item_id', 'report_sha256')):
            raise ValueError('exact_source_bound_target_join_required')
        finding = ref['finding']
        output = raw['output']
        signed = gate['output']['findings'][finding]
        state = output['native_labels'][finding]['state'] if output['status'] == 'complete' else None
        if signed['raw_state'] != state or signed['raw_status'] != output['status']:
            raise ValueError('raw_native_state_changed_by_gate')
        results.append({**signed, 'item_id': ref['item_id'], 'report_sha256': ref['report_sha256'],
                       'finding': finding, 'family': ref['family'], 'expected_state': ref['expected_state'],
                       'status': output['status'], 'state': state})
    return results


def readout(rows):
    return {'raw_four_state': metrics(rows), 'veto_only': authored_readout(rows),
            'decisions': dict(sorted(Counter(row['decision'] for row in rows).items()))}


def markdown(summary):
    all_rows = summary['all88']
    native, gated = all_rows['raw_four_state'], all_rows['veto_only']
    lines = ['# Native + ConText scope88 / 多 finding 语境弃权诊断', '',
        '88 wholly authored known development probes, not clinical gold or corrected triples.', '',
        f"Raw four-state matches: {native['exact_matches']}/{native['rows']}.",
        f"Raw signed proposals: {gated['raw_determinate_proposals']}; incorrect: {gated['raw_incorrect_determinate']}.",
        f"Soft retained: {gated['soft_retained']}/88; correct: {gated['soft_correct']}; incorrect: {gated['soft_incorrect']}.",
        f"Incorrect withheld: {gated['incorrect_determinate_withheld']}; correct lost: {gated['correct_determinate_withheld']}.",
        'No abstention is credited as unknown or overall accuracy improvement.', '',
        '| Family | Rows | Raw matches | Soft retained | Correct retained | Wrong retained | Correct lost |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for family, row in summary['per_family'].items():
        raw, r = row['raw_four_state'], row['veto_only']
        lines.append(f"| {family} | {raw['rows']} | {raw['exact_matches']} | {r['soft_retained']} | "
                     f"{r['soft_correct']} | {r['soft_incorrect']} | {r['correct_determinate_withheld']} |")
    lines += ['', 'All native labels preserved. Context is the same source, not an independent clinical vote.',
        'Anatomy and qualified/change-only scope remain unverified; wrong agreements remain visible.',
        'No old EHR/CXR/report, score table, winner, frozen rule or gate changed.',
        'No clinical qualification, selection or regeneration authorized.', '']
    return '\n'.join(lines)


def evaluate(plan_root, expected, run_id):
    guard.guard()
    pins = {}
    root = frozen_native.require_inside(plan_root, BASE/'native_context_scope_plans', must_exist=True)
    manifest = guard.metadata(root/'manifest.json', pins, expected)
    guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    plan = guard.metadata(root/'plan.json', pins, manifest['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA+'-plan' or plan['policy'] != scope.POLICY:
        raise ValueError('frozen_scope_protocol_required')
    nlp, runtime = initialize_context(pins)
    if runtime != plan['context_runtime']:
        raise ValueError('frozen_context_runtime_required')
    rule_identity = runtime['rule_dictionary_sha256']
    temporary, target = new_atomic_run(BASE/'native_context_scope_runs', run_id)
    try:
        tmp = WORKSPACE/'.tmp'/f'native_context_scope_{run_id}'
        tmp.mkdir(mode=0o2770, exist_ok=False)
        os.chmod(tmp, 0o2770)
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.update(TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [str(frozen_native.ENV/'bin/python'), str(WORKER), '--plan-root', str(root),
                   '--plan-manifest-sha256', expected, '--output-directory', str(temporary)]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise RuntimeError('frozen_native_scope_worker_failed_closed')
        native = guard.metadata(temporary/'native_predictions.json', pins)['records']
        native_replays = guard.metadata(temporary/'native_replays.json', pins)['records']
        native_freeze = guard.metadata(temporary/'native_freeze_receipt.json', pins)
        if any(sha256_file(temporary/name) != digest for name, digest in native_freeze['sha256'].items()):
            raise ValueError('frozen_native_predictions_required')
        inputs = guard.metadata(root/'inputs.json', pins, manifest['artifacts']['inputs.json'])['records']
        if len(inputs) != 88 or len(native) != 88 or len(native_replays) != 88:
            raise ValueError('all_fixed_authored_rows_required')
        contexts, gates, replays = [], [], []
        started = time.monotonic()
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            for item, raw in zip(inputs, native):
                if any(item[key] != raw[key] for key in ('item_id', 'report_sha256')):
                    raise ValueError('exact_native_input_join_required')
                context = scope.parse_context(nlp, item['text'], raw['output'])
                replay = scope.parse_context(nlp, item['text'], raw['output'])
                primary_gate, replay_gate = scope.gate(raw['output'], context), scope.gate(raw['output'], replay)
                identity = {key: item[key] for key in ('item_id', 'report_sha256')}
                contexts.append({**identity, 'output': context})
                gates.append({**identity, 'output': primary_gate})
                replays.append({**identity, 'same_full_context_evidence': context == replay,
                    'same_gate': primary_gate == replay_gate, 'context_replay': replay, 'gate_replay': replay_gate})
        context_seconds = round(time.monotonic()-started, 6)
        current_rules = hashlib.sha256(json.dumps([r.to_dict() for r in nlp.get_pipe('medspacy_context').rules],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if current_rules != rule_identity:
            raise ValueError('unchanged_frozen_context_rules_required')
        dump(temporary/'context_predictions.json', {'records': contexts})
        dump(temporary/'scope_gates.json', {'records': gates})
        dump(temporary/'context_replays.json', {'records': replays})
        closed = {name: sha256_file(temporary/name) for name in
            ('native_predictions.json', 'native_replays.json', 'context_predictions.json', 'scope_gates.json', 'context_replays.json')}
        dump(temporary/'prediction_freeze_receipt.json', {'sha256': closed,
            'reference_semantics_decoded_before_prediction_fsync': False,
            'source_provenance_bytes_hashed': True, 'investigator_knows_prior_results': True})
        # Target finding/reference semantics never entered either parser or gate.
        references = guard.metadata(root/'references.json', pins, manifest['artifacts']['references.json'])['records']
        scored = checks(references, native, contexts, gates)
        native_runtime = guard.metadata(temporary/'native_runtime.json', pins)
        summary = {'schema_version': SCHEMA, 'policy': scope.POLICY, 'context_runtime': runtime,
            'native_runtime': native_runtime, 'context_seconds': context_seconds,
            'all88': readout(scored),
            'per_family': {f: readout([r for r in scored if r['family'] == f]) for f in sorted({r['family'] for r in scored})},
            'per_finding': {f: readout([r for r in scored if r['finding'] == f]) for f in sorted({r['finding'] for r in scored})},
            'context_complete': sum(row['output']['status'] == 'complete' for row in contexts),
            'context_unavailable': sum(row['output']['status'] != 'complete' for row in contexts),
            'context_failure_reasons': dict(sorted(Counter(row['output']['failure_reason'] for row in contexts
                if row['output']['status'] != 'complete').items())),
            'native_replay_changed': sum(not row['same_full_evidence'] for row in native_replays),
            'context_replay_changed': sum(not row['same_full_context_evidence'] for row in replays),
            'gate_replay_changed': sum(not row['same_gate'] for row in replays),
            'reference_semantics_decoded_after_prediction_fsync': True,
            'hard_action_eligible': 0, 'clinical_qualification_passed': False,
            'old_scores_labels_winners_changed': False, 'patient_or_candidate_inputs_read': False,
            'gpu_calls': 0, 'external_api': False, 'new_slurm_submissions': 0,
            'selection_changed': False, 'regeneration_authorized': False}
        dump(temporary/'scored_checks.json', {'records': scored})
        dump(temporary/'summary.json', summary)
        write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        guard.verify_pins(pins)
        if any(sha256_file(temporary/name) != digest for name, digest in closed.items()):
            raise ValueError('closed_native_context_gate_predictions_changed')
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-run-manifest',
            'sources': {p: digest for p, digest in pins.items() if not Path(p).is_relative_to(temporary)},
            'plan_manifest_sha256': expected, 'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        frozen_native.private_permissions(temporary)
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'native_context_scope88_diagnostic_complete', 'authored_texts': 88,
        'context_complete': summary['context_complete'], 'context_seconds': context_seconds,
        'soft_retained': summary['all88']['veto_only']['soft_retained'],
        'soft_incorrect': summary['all88']['veto_only']['soft_incorrect'],
        'clinical_qualified': False, 'manifest_sha256': sha256_file(target/'manifest.json')}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('prepare', 'evaluate'))
    cli.add_argument('--run-id', required=True)
    cli.add_argument('--plan-root', type=Path)
    cli.add_argument('--plan-manifest-sha256')
    args = cli.parse_args()
    os.umask(0o007)
    try:
        _, result = prepare(args.run_id) if args.action == 'prepare' else evaluate(args.plan_root, args.plan_manifest_sha256, args.run_id)
        print(json.dumps(result, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'native_context_scope_failed_closed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
