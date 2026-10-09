#!/usr/bin/env python3
"""Cached-native, source-preserving token-bridge replay; CPU Slurm only."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'benchmarks'),
               str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, new_atomic_run, write_private_text,
                       sha256_file, commit_atomic_run, discard_atomic_run)
from tricompose_v12 import native_context_scope_v2 as scope
import benchmark_native_context_scope as v1
import benchmark_chexpert_negbio_authored112 as common
import score_cached_opacity_candidates as guard

BASE = PROTECTED_ROOT/'tricompose_v1_2'
PRIOR_PLAN = BASE/'native_context_scope_plans/scope88_12714150_001'
PRIOR_PLAN_SHA = '7896582f63978b0954e863444a11d0bc023a0fd90abdcd2fc639e512a8c95771'
CACHE = BASE/'native_context_scope_runs/scope88_12714150_001'
CACHE_SHA = 'ab1e93abb349b28b2b7168bbe0d213f67b5950aa95f17dee7b07c310790267dc'
SCHEMA = 'tricompose-native-context-scope88-v2'
PROTOCOL = WORKSPACE/'docs/native_context_scope_v2_protocol.md'
TEST = ROOT/'tests/test_native_context_scope_v2.py'
dump = common.dump


def initialize(pins):
    nlp, runtime = v1.initialize_context(pins)
    site = Path(sys.prefix)/'lib/python3.11/site-packages/spacy/tokens'
    for name in ('_retokenize.pyx', '_retokenize.cpython-311-x86_64-linux-gnu.so'):
        path = site/name
        pins[str(path)] = sha256_file(path)
    return nlp, {**runtime, 'token_boundary_bridge': 'native_character_boundaries_only',
                 'default_token_boundaries_may_change': True, 'native_spans_expanded': False}


def prepare(run_id):
    guard.guard()
    pins = {str(path): sha256_file(path) for path in (Path(__file__), Path(scope.__file__), PROTOCOL, TEST)}
    prior = guard.metadata(PRIOR_PLAN/'manifest.json', pins, PRIOR_PLAN_SHA)
    guard.verify_pins(prior['sources'])
    pins.update(prior['sources'])
    cache = guard.metadata(CACHE/'manifest.json', pins, CACHE_SHA)
    guard.metadata(CACHE/'native_predictions.json', pins, cache['artifacts']['native_predictions.json'])
    _, runtime = initialize(pins)
    temporary, target = new_atomic_run(BASE/'native_context_scope_plans', run_id)
    try:
        for name in ('inputs.json', 'references.json'):
            path = PRIOR_PLAN/name
            if sha256_file(path) != prior['artifacts'][name]:
                raise ValueError('unchanged_scope88_input_or_reference_required')
            write_private_text(temporary/name, path.read_text(encoding='utf-8'))
        dump(temporary/'plan.json', {'schema_version': SCHEMA+'-plan', 'policy': scope.POLICY,
            'context_runtime': runtime, 'prior_plan_manifest_sha256': PRIOR_PLAN_SHA,
            'cached_run_manifest_sha256': CACHE_SHA, 'cached_native_sha256': cache['artifacts']['native_predictions.json'],
            'authored_count': 88, 'prior_scope88_results_known': True, 'heldout_clinical_gold': False,
            'new_native_parser_calls': 0, 'reference_semantics_not_used_by_parsers': True})
        guard.verify_pins(pins)
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-plan-manifest', 'sources': pins,
            'artifacts': {path.name: sha256_file(path) for path in sorted(temporary.iterdir())}})
        common.private_permissions(temporary)
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'scope88_token_bridge_plan_frozen', 'authored_texts': 88,
                    'new_native_parser_calls': 0, 'manifest_sha256': sha256_file(target/'manifest.json')}


def evaluate(plan_root, expected, run_id):
    guard.guard()
    pins = {}
    root = common.require_inside(plan_root, BASE/'native_context_scope_plans', must_exist=True)
    manifest = guard.metadata(root/'manifest.json', pins, expected)
    guard.verify_pins(manifest['sources'])
    pins.update(manifest['sources'])
    plan = guard.metadata(root/'plan.json', pins, manifest['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA+'-plan' or plan['policy'] != scope.POLICY or \
            plan['cached_run_manifest_sha256'] != CACHE_SHA or plan['new_native_parser_calls'] != 0:
        raise ValueError('frozen_engineering_replay_protocol_required')
    nlp, runtime = initialize(pins)
    if runtime != plan['context_runtime']:
        raise ValueError('frozen_context_bridge_runtime_required')
    cache = guard.metadata(CACHE/'manifest.json', pins, CACHE_SHA)
    native = guard.metadata(CACHE/'native_predictions.json', pins, plan['cached_native_sha256'])['records']
    native_replays = guard.metadata(CACHE/'native_replays.json', pins, cache['artifacts']['native_replays.json'])['records']
    historical_runtime = guard.metadata(CACHE/'native_runtime.json', pins, cache['artifacts']['native_runtime.json'])
    inputs = guard.metadata(root/'inputs.json', pins, manifest['artifacts']['inputs.json'])['records']
    if len(native) != len(inputs) or len(inputs) != 88:
        raise ValueError('all_cached_authored88_inputs_required')
    temporary, target = new_atomic_run(BASE/'native_context_scope_runs', run_id)
    try:
        # Exact byte copy: no new native call or altered historical prediction.
        write_private_text(temporary/'native_predictions.json', (CACHE/'native_predictions.json').read_text())
        if sha256_file(temporary/'native_predictions.json') != plan['cached_native_sha256']:
            raise ValueError('cached_native_bytes_changed')
        contexts, gates, replays = [], [], []
        started = time.monotonic()
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            for item, raw in zip(inputs, native):
                if any(item[key] != raw[key] for key in ('item_id', 'report_sha256')):
                    raise ValueError('exact_cached_authored_source_join_required')
                primary = scope.parse_context(nlp, item['text'], raw['output'])
                replay = scope.parse_context(nlp, item['text'], raw['output'])
                primary_gate, replay_gate = scope.gate(raw['output'], primary), scope.gate(raw['output'], replay)
                identity = {key: item[key] for key in ('item_id', 'report_sha256')}
                contexts.append({**identity, 'output': primary})
                gates.append({**identity, 'output': primary_gate})
                replays.append({**identity, 'same_full_context_evidence': primary == replay,
                    'same_gate': primary_gate == replay_gate, 'context_replay': replay, 'gate_replay': replay_gate})
        seconds = round(time.monotonic()-started, 6)
        rules = hashlib.sha256(json.dumps([r.to_dict() for r in nlp.get_pipe('medspacy_context').rules],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if rules != runtime['rule_dictionary_sha256']:
            raise ValueError('unchanged_official_context_rules_required')
        dump(temporary/'context_predictions.json', {'records': contexts})
        dump(temporary/'scope_gates.json', {'records': gates})
        dump(temporary/'context_replays.json', {'records': replays})
        closed = {name: sha256_file(temporary/name) for name in
                  ('native_predictions.json', 'context_predictions.json', 'scope_gates.json', 'context_replays.json')}
        dump(temporary/'prediction_freeze_receipt.json', {'sha256': closed,
            'reference_semantics_decoded_before_prediction_fsync': False,
            'source_provenance_bytes_hashed': True, 'prior_scope88_results_known': True})
        references = guard.metadata(root/'references.json', pins, manifest['artifacts']['references.json'])['records']
        scored = v1.checks(references, native, contexts, gates)
        prior_summary = guard.metadata(CACHE/'summary.json', pins, cache['artifacts']['summary.json'])
        readout = v1.readout(scored)
        if readout['raw_four_state'] != prior_summary['all88']['raw_four_state']:
            raise ValueError('raw_native_metrics_changed_during_bridge_replay')
        summary = {'schema_version': SCHEMA, 'policy': scope.POLICY, 'context_runtime': runtime,
            'historical_native_runtime': historical_runtime, 'new_native_parser_calls': 0,
            'context_seconds': seconds, 'all88': readout,
            'per_family': {f: v1.readout([r for r in scored if r['family'] == f]) for f in sorted({r['family'] for r in scored})},
            'per_finding': {f: v1.readout([r for r in scored if r['finding'] == f]) for f in sorted({r['finding'] for r in scored})},
            'context_complete': sum(row['output']['status'] == 'complete' for row in contexts),
            'context_unavailable': sum(row['output']['status'] != 'complete' for row in contexts),
            'context_failure_reasons': dict(sorted(Counter(row['output']['failure_reason'] for row in contexts
                if row['output']['status'] != 'complete').items())),
            'texts_with_token_splits': sum(row['output'].get('token_boundary_bridge', {}).get('split_tokens', 0) > 0 for row in contexts),
            'split_tokens': sum(row['output'].get('token_boundary_bridge', {}).get('split_tokens', 0) for row in contexts),
            'native_replay_changed_in_cached_run': sum(not row['same_full_evidence'] for row in native_replays),
            'context_replay_changed': sum(not row['same_full_context_evidence'] for row in replays),
            'gate_replay_changed': sum(not row['same_gate'] for row in replays),
            'prior_v1_context_complete': prior_summary['context_complete'],
            'prior_v1_veto_readout': prior_summary['all88']['veto_only'],
            'cached_native_bytes_unchanged': True, 'raw_native_metrics_unchanged': True,
            'prior_scope88_results_known': True, 'reference_semantics_decoded_after_prediction_fsync': True,
            'hard_action_eligible': 0, 'clinical_qualification_passed': False,
            'old_scores_labels_winners_changed': False, 'patient_or_candidate_inputs_read': False,
            'gpu_calls': 0, 'external_api': False, 'new_slurm_submissions': 0,
            'selection_changed': False, 'regeneration_authorized': False}
        dump(temporary/'scored_checks.json', {'records': scored})
        dump(temporary/'summary.json', summary)
        text = v1.markdown(summary).replace('# Native + ConText scope88', '# Token bridge V2 + frozen ConText scope88', 1)
        text += f"\nCached native predictions identical; new native calls 0. V1 context complete {prior_summary['context_complete']}/88; V2 {summary['context_complete']}/88.\n"
        write_private_text(temporary/'RESULTS_CN_EN.md', text)
        guard.verify_pins(pins)
        if any(sha256_file(temporary/name) != digest for name, digest in closed.items()):
            raise ValueError('closed_bridge_predictions_changed')
        dump(temporary/'manifest.json', {'schema_version': SCHEMA+'-run-manifest',
            'sources': {p: digest for p, digest in pins.items() if not Path(p).is_relative_to(temporary)},
            'plan_manifest_sha256': expected, 'cached_run_manifest_sha256': CACHE_SHA,
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        common.private_permissions(temporary)
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, {'status': 'scope88_character_bridge_replay_complete', 'authored_texts': 88,
        'new_native_parser_calls': 0, 'context_complete': summary['context_complete'],
        'context_seconds': seconds, 'soft_retained': readout['veto_only']['soft_retained'],
        'soft_incorrect': readout['veto_only']['soft_incorrect'], 'clinical_qualified': False,
        'manifest_sha256': sha256_file(target/'manifest.json')}


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
        print(json.dumps({'status': 'scope88_bridge_failed_closed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
