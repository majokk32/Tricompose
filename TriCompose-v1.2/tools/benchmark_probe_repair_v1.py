#!/usr/bin/env python3
"""Existing CPU Slurm only: observed probes over the unchanged synthetic cache.

No body/pixel/weight reads. Seal choices before reading the secondary endpoint.
This is development replay, not actual regeneration or independent clinical gold.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ('TriCompose-v1.2/src', 'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(ROOT.parent / relative))
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.legacy_replay_adapter import make_legacy_bank
from tricompose_v12 import probe_repair_v1 as policy_module

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
CAPS = (4, 8, 12, 20, 30)
NEW_METHODS = ('observed_probe_repair', 'probe_without_action_feedback')
OLD_METHODS = ('fixed', 'random', 'static_rerank', 'targeted_heuristic',
               'random_acquisition_score_free_final')
REPLAY = (BASE / 'automatic_replays/automatic_replay_pool80_12576792_001',
    'af82764ac90f9af5b701c577fb2f9bfe6ecf4052310299e8e7cdcb415d2aed5f')
ENDPOINT = (BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001',
    'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c')
UNIFORM = (BASE / 'automatic_replays/score_free_random_12654973_001',
    '771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124')


def cpu_guard():
    job = os.environ.get('SLURM_JOB_ID', '')
    if (not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text()
            or os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')):
        raise RuntimeError('existing_cache_only_cpu_slurm_required')


def checked(path, pin, limit, sources):
    p = require_inside(path, PROTECTED_ROOT, must_exist=True)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > limit or sha256_file(p) != pin:
        raise ValueError('bounded_unchanged_metadata_source_required')
    value = p.read_text()
    if sha256_file(p) != pin:
        raise ValueError('metadata_changed_during_read')
    sources[str(p)] = pin
    return value


def parent(spec, sources):
    root, pin = spec
    value = json.loads(checked(root / 'manifest.json', pin, 1024**2, sources))
    if value.get('original_selection_changed') is not False:
        raise ValueError('historical_selection_must_remain_unchanged')
    return root, value


def artifact(root, manifest, name, sources):
    pin = manifest['artifacts'][name]['sha256']
    return checked(root / name, pin, 32*1024**2, sources)


def load_primary(sources):
    root, manifest = parent(REPLAY, sources)
    policy = json.loads(artifact(root, manifest, 'frozen_policy.json', sources))
    if policy['model_call_budgets'] != list(CAPS):
        raise ValueError('unchanged_budget_grid_required')
    scores = [json.loads(x) for x in checked(manifest['source_paths']['source_scores'],
        manifest['source_sha256']['source_scores'], 4*1024**2, sources).splitlines() if x]
    details = json.loads(checked(manifest['source_paths']['source_edges'],
        manifest['source_sha256']['source_edges'], 8*1024**2, sources))
    bank = make_legacy_bank(scores, details, policy)
    if len(bank) != 80 or sum(map(len, bank.values())) != 960:
        raise ValueError('full_unchanged_eighty_case_bank_required')
    return bank, policy


def replay(grid, policy, cap, feedback):
    initial_slot = (*policy['image_order'][0], policy['report_order'][0])
    controller = policy_module.ProbeController(policy_module.snapshot(grid[initial_slot]),
        policy['image_order'], policy['report_order'], cap, feedback_enabled=feedback)
    while True:
        request = controller.propose()
        if request['action'] == 'stop':
            result = controller.result()
            result['terminal_reason'] = request['reason']
            result['method'] = NEW_METHODS[0 if feedback else 1]
            return result
        # Simulated worker: only the requested slot is projected/observed.
        slot = tuple(request['slot'])
        if slot not in grid:
            controller.complete(request, failure_type='unavailable_cache_slot')
        else:
            controller.complete(request, policy_module.snapshot(grid[slot]))


def numeric_readout(observation, endpoint):
    result = {'artifact_valid': int(observation['artifact_gate_failures'] == 0
        and observation['quality']['cxr_basic_validity_pass']), 'biovil_raw_cosine': endpoint}
    for edge, values in policy_module.edges(observation).items():
        known = len(values['known'])
        for name in ('known', 'comparable', 'support', 'positive_support', 'opposition'):
            result[edge+'_'+name] = len(values[name])
        for name in ('comparable', 'support', 'opposition'):
            result[edge+'_'+name+'_over_known'] = len(values[name])/known if known else None
    return result


def aggregate(trials, observations, endpoints):
    """Average seed replicates WITHIN EHR first; NA never becomes success."""
    groups = defaultdict(list)
    for t in trials:
        cid = t['selected_candidate_id']
        readout = numeric_readout(observations[cid], endpoints[cid]) if cid else None
        groups[(t['case_id'], t['method'], t['model_call_budget'])].append((t, readout))
    cases = []
    keys = list(numeric_readout(next(iter(observations.values())), None))
    for (case, method, cap), rows in sorted(groups.items()):
        result = {'case_id': case, 'method': method, 'model_call_budget': cap,
            'replicates': len(rows), 'mean_simulated_calls': sum(t['calls'] for t, _ in rows)/len(rows),
            'selected_fraction': sum(r is not None for _, r in rows)/len(rows)}
        for key in keys:
            values = [r[key] for _, r in rows if r is not None and r[key] is not None]
            result[key] = sum(values)/len(values) if len(values) == len(rows) else None
        cases.append(result)
    methods = defaultdict(list)
    for row in cases:
        methods[row['method'], row['model_call_budget']].append(row)
    table = []
    for (method, cap), rows in sorted(methods.items()):
        result = {'method': method, 'model_call_budget': cap, 'fixed_ehr_cases': len(rows),
            'mean_simulated_calls': sum(r['mean_simulated_calls'] for r in rows)/len(rows),
            'selected_fraction': sum(r['selected_fraction'] for r in rows)/len(rows),
            'clinical_accuracy': None, 'clinical_repair_success': None}
        for key in keys:
            values = [r[key] for r in rows if r[key] is not None]
            result[key+'_mean'] = sum(values)/len(values) if values else None
            result[key+'_available_ehr_cases'] = len(values)
        table.append(result)
    return cases, table


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def validate_baseline_grid(rows, bank, observations, *, uniform=False):
    """Validate every case/budget/seed identity, not merely the row count."""
    methods = ('random_acquisition_score_free_final',) if uniform else OLD_METHODS[:4]
    expected = set()
    for case in bank:
        for cap in CAPS:
            for method in methods:
                seeds = [(a, b) for a in range(5) for b in range(5)] if uniform else \
                    [(s, None) for s in range(5)] if method == 'random' else [(None, None)]
                expected.update((case, method, cap, a, b) for a, b in seeds)
    found = set()
    for r in rows:
        identity = (r['case_id'], r['method'], r['model_call_budget'],
            r.get('acquisition_seed') if uniform else r.get('random_seed'),
            r.get('final_choice_seed') if uniform else None)
        if identity not in expected or identity in found:
            raise ValueError('complete_unique_frozen_baseline_grid_required')
        found.add(identity)
        calls = r['simulated_model_calls']
        if (type(calls) is not int or not 0 <= calls <= r['model_call_budget']
                or r.get('actual_regeneration_executed') is not False
                or r.get('clinical_acceptance') is not False):
            raise ValueError('bounded_nonclinical_cached_baseline_required')
        cid = r['selected_candidate_id']
        if cid is not None and (cid not in observations
                or observations[cid]['case_id'] != r['case_id']
                or r['selected_ehr_sha256'] != observations[cid]['lineage']['ehr_sha256']):
            raise ValueError('baseline_selection_fixed_case_lineage_required')
    if found != expected:
        raise ValueError('all_frozen_baseline_replicates_required')


def probe_activity(outcomes):
    result = []
    for method in NEW_METHODS:
        for cap in CAPS:
            rows = [r for r in outcomes if r['method'] == method and r['model_call_budget'] == cap]
            history = [h for r in rows for h in r['history']]
            result.append({'method': method, 'model_call_budget': cap, 'fixed_ehr_cases': len(rows),
                'cases_with_accepted_proxy_transition': sum(any(h['accepted'] for h in r['history']) for r in rows),
                'accepted_proxy_transitions': sum(h['accepted'] for h in history),
                'report_probes': sum(h['action'] == 'report_probe' for h in history),
                'image_probes': sum(h['action'] == 'image_probe' for h in history),
                'failed_probes': sum(h['failure_type'] is not None for h in history),
                'clinical_repair_success': None, 'actual_regeneration_executed': False})
    return result


def run(args):
    cpu_guard()
    started = time.monotonic()
    sources = {}
    bank, policy = load_primary(sources)
    code_pins = {str(Path(__file__).resolve()): sha256_file(Path(__file__)),
        str(Path(policy_module.__file__).resolve()): sha256_file(policy_module.__file__)}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        protocol = {'schema_version': policy_module.VERSION+'-protocol', 'deadline': '2026-10-11',
            'methods': list(NEW_METHODS), 'budgets': list(CAPS), 'image_order': policy['image_order'],
            'report_order': policy['report_order'], 'routing': 'observed_proxy_action_effect_only',
            'acceptance': 'strict_target_gain_exact_fact_and_coverage_preservation',
            'all_ehrs_fixed': True, 'heldout_or_clinical_test': False,
            'endpoint_available_to_controller': False, 'training_allowed': False}
        write_private_json(temporary/'frozen_protocol.json', protocol)
        outcomes = [replay(grid, policy, cap, feedback) for _, grid in sorted(bank.items())
            for cap in CAPS for feedback in (True, False)]
        selection_file = write_private_text(temporary/'probe_outcomes.jsonl',
            ''.join(json.dumps(r, sort_keys=True)+'\n' for r in outcomes))
        seal = sha256_file(selection_file)
        # Secondary endpoint and previous outcomes are first read AFTER sealing.
        endpoint_root, endpoint_manifest = parent(ENDPOINT, sources)
        endpoint_doc = json.loads(artifact(endpoint_root, endpoint_manifest, 'scores.json', sources))
        observations = {c['score_record']['triple_candidate_id']: policy_module.snapshot(c)
            for grid in bank.values() for c in grid.values()}
        records = endpoint_doc['records']
        if len(records) != 960 or {r['triple_candidate_id'] for r in records} != set(observations):
            raise ValueError('complete_exact_secondary_inventory_required')
        endpoints = {}
        for record in records:
            cid = record['triple_candidate_id']; obs = observations[cid]
            if record['case_id'] != obs['case_id'] or record['calibrated'] is not False \
                    or any(record[k] != obs['lineage'][k] for k in ('ehr_sha256', 'ehr_facts_sha256',
                        'cxr_sha256', 'report_sha256', 'cxr_candidate_id', 'report_candidate_id')):
                raise ValueError('secondary_source_lineage_required')
            number = record['biovil_raw_cosine']
            if number is None:
                if record['status'] != 'not_available' or not record['reason']:
                    raise ValueError('explicit_secondary_na_required')
            elif type(number) not in (int, float) or not -1.01 <= number <= 1.01 \
                    or record['status'] != 'computed_secondary_uncalibrated' or record['reason'] is not None:
                raise ValueError('valid_uncalibrated_secondary_value_required')
            endpoints[cid] = number
        replay_root, replay_manifest = parent(REPLAY, sources)
        old = [json.loads(x) for x in artifact(replay_root, replay_manifest,
            'replay_outcomes.jsonl', sources).splitlines() if x]
        uniform_root, uniform_manifest = parent(UNIFORM, sources)
        uniform = [json.loads(x) for x in artifact(uniform_root, uniform_manifest,
            'control_outcomes.jsonl', sources).splitlines() if x]
        validate_baseline_grid(old, bank, observations)
        validate_baseline_grid(uniform, bank, observations, uniform=True)
        trials = [{'case_id': r['case_id'], 'method': r['method'],
            'model_call_budget': r['model_call_budget'], 'selected_candidate_id': r['selected_candidate_id'],
            'calls': r['simulated_calls'] if r['method'] in NEW_METHODS else r['simulated_model_calls']}
            for r in (*outcomes, *old, *uniform)]
        if any(r['case_id'] not in bank or r['method'] not in (*NEW_METHODS, *OLD_METHODS)
                or r['model_call_budget'] not in CAPS or r['calls'] > r['model_call_budget']
                or r['selected_candidate_id'] is not None and observations[r['selected_candidate_id']]['case_id'] != r['case_id']
                for r in trials):
            raise ValueError('fixed_case_budget_and_requested_source_choices_required')
        cases, table = aggregate(trials, observations, endpoints)
        if len(cases) != 2800 or len(table) != 35 or any(r['fixed_ehr_cases'] != 80 for r in table):
            raise ValueError('complete_case_method_budget_denominators_required')
        write_private_text(temporary/'case_means.csv', csv_text(cases))
        write_private_text(temporary/'method_comparison.csv', csv_text(table))
        activity = probe_activity(outcomes)
        write_private_text(temporary/'probe_activity.csv', csv_text(activity))
        summary = {'schema_version': policy_module.VERSION+'-run', 'status': 'development_replay_complete',
            'fixed_ehr_cases': 80, 'source_candidate_slots': 960, 'new_policy_replay_trials': len(outcomes),
            'frozen_baseline_trials': len(old)+len(uniform), 'method_rows': len(table),
            'accepted_probe_transitions': dict(Counter(r['method'] for r in outcomes
                for h in r['history'] if h['accepted'])),
            'new_model_calls': 0, 'actual_regeneration_executed': False,
            'actual_gpu_savings': None, 'independent_clinical_truth': False,
            'original_choices_changed': False, 'source_payloads_or_weights_read': False,
            'selection_sha256_sealed_before_endpoint': seal,
            'runtime_seconds': time.monotonic()-started}
        write_private_json(temporary/'summary.json', summary)
        if sha256_file(selection_file) != seal or any(sha256_file(path) != pin
                for path, pin in {**sources, **code_pins}.items()):
            raise ValueError('source_or_choice_changed_during_secondary_evaluation')
        artifacts = {f.name: {'sha256': sha256_file(f)} for f in temporary.iterdir() if f.is_file()}
        write_private_json(temporary/'manifest.json', {'schema_version': policy_module.VERSION+'-receipt',
            'sources': sources, 'code_pins': code_pins, 'artifacts': artifacts,
            'new_model_calls': 0, 'actual_regeneration_executed': False,
            'original_selection_changed': False, 'clinical_qualified': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE/'probe_repair_runs')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, _ = run(args)
        print(json.dumps({'status': 'development_replay_complete',
            'manifest_sha256': sha256_file(target/'manifest.json')}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}, sort_keys=True))
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
