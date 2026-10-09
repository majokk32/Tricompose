#!/usr/bin/env python3
"""All six fixed report schedules, unchanged guard, sealed before endpoints."""
import argparse
from collections import Counter, defaultdict
import csv
import io
from itertools import permutations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_probe_repair_v1 as b
import benchmark_matched_report_prefix_v1 as matched
import diagnose_guard_headroom_v1 as headroom
import diagnose_probe_gap_v1 as gap
from tricompose_v12 import matched_report_prefix_v1 as control
from tricompose_v12.live_workers import check_pins
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

VERSION = 'tricompose-report-order-control-v1'
PROTOCOL = ROOT / 'configs/report_order_control_v1.json'
GROUPS = b.BASE / 'conditioning_robustness_runs/conditioning_groups_12714150_001'
GROUP_SHA = '0d6e3ecf79c5a99cd500ee74cb3fee23695cd447ff768bd1c848593d04ce985c'
HEADROOM = b.BASE / 'guard_headroom_diagnostics/headroom_12784259_001'
HEADROOM_SHA = 'c81239875dd1b66467eea9621d7cc2128d85c6d03d054c5f76b8a97dc1517a46'


def declared_orders(policy, protocol):
    original = policy['report_order']
    expected = [[original[0], *tail] for tail in permutations(original[1:])]
    if (len(original) != 4 or len(set(original)) != 4
            or protocol['orders'] != expected or protocol['caps'] != list(b.CAPS)
            or protocol['initial_report'] != original[0]
            or protocol['initial_image'] != policy['image_order'][0]
            or protocol['commit_guard'] != 'unchanged_probe_repair_v1_action_credit'
            or any(protocol[k] is not False for k in ('choose_order_using_secondary',
                'choose_order_on_final_test', 'clinical_qualified', 'training_allowed',
                'image_regeneration_allowed', 'actual_regeneration_executed', 'old_choices_and_rules_changed'))
            or protocol['all_orders_retained'] is not True
            or protocol['estimands'] != ['case_weighted', 'equal_conditioning_group']):
        raise ValueError('all_six_declared_orders_same_initial_and_guard_required')
    return [('fixed_report_order_%03d' % (n + 1), order) for n, order in enumerate(expected)]


def replay(grid, policy, cap, method, order):
    initial = (*policy['image_order'][0], order[0])
    runner = control.FixedReportPrefix(b.policy_module.snapshot(grid[initial]), order, cap)
    while True:
        request = runner.propose()
        if request['action'] == 'stop':
            result = runner.result()
            result.update(method=method, report_order=list(order), terminal_reason=request['reason'])
            return result
        slot = tuple(request['slot'])
        if slot not in grid:
            runner.complete(request, failure_type='unavailable_cache_slot')
        else:
            runner.complete(request, b.policy_module.snapshot(grid[slot]))


def groups_mapping(rows, cases, expected_groups=49):
    output = {}
    for row in rows:
        case, gid = row['case_id'], row['conditioning_group_id']
        if case not in cases or case in output or not isinstance(gid, str) or not gid:
            raise ValueError('unique_complete_frozen_conditioning_mapping_required')
        output[case] = gid
    if set(output) != set(cases) or len(set(output.values())) != expected_groups:
        raise ValueError('all_cases_and_declared_conditioning_groups_required')
    return output


def grouped_metrics(case_rows, mapping):
    """Do not replace missing EHR edges with zero; group means first."""
    metadata = {'case_id', 'method', 'model_call_budget', 'replicates'}
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[row['method'], row['model_call_budget']].append(row)
    output = []
    for (method, cap), part in sorted(grouped.items()):
        for metric in (k for k in part[0] if k not in metadata):
            values = [(mapping[row['case_id']], row[metric]) for row in part if row[metric] is not None]
            groups = defaultdict(list)
            for gid, value in values:
                groups[gid].append(value)
            output.append({'method': method, 'model_call_budget': cap, 'metric': metric,
                'attempted_ehr_cases': len(part), 'available_ehr_cases': len(values),
                'available_conditioning_groups': len(groups),
                'case_weighted_mean': sum(v for _, v in values) / len(values) if values else None,
                'equal_conditioning_mean': sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None,
                'clinical_accuracy': None})
    return output


def exact_grid(rows, cases, methods):
    expected = {(case, method, cap) for case in cases for method in methods for cap in b.CAPS}
    found = set()
    for row in rows:
        key = (row['case_id'], row['method'], row['model_call_budget'])
        if (key not in expected or key in found or type(row['model_call_budget']) is not int
                or type(row['simulated_calls']) is not int or not 4 <= row['simulated_calls'] <= row['model_call_budget']
                or row['actual_regeneration_executed'] is not False or row['uses_secondary_endpoint'] is not False):
            raise ValueError('unique_bounded_nonclinical_trial_grid_required')
        found.add(key)
    if found != expected:
        raise ValueError('complete_all_case_method_cap_grid_required')


def run(args):
    b.cpu_guard()
    sources = {}
    bank, policy = b.load_primary(sources)
    protocol = json.loads(PROTOCOL.read_text())
    orders = declared_orders(policy, protocol)
    group_manifest = json.loads(b.checked(GROUPS / 'manifest.json', GROUP_SHA, 1024**2, sources))
    group_rows = list(csv.DictReader(io.StringIO(b.artifact(GROUPS, group_manifest, 'case_groups.csv', sources))))
    mapping = groups_mapping(group_rows, bank, protocol['clinical_intent_groups'])
    initial_views = {case: b.policy_module.snapshot(next(iter(grid.values()))) for case, grid in bank.items()}
    if any(row['ehr_sha256'] != initial_views[row['case_id']]['lineage']['ehr_sha256']
            or row['ehr_facts_sha256'] != initial_views[row['case_id']]['lineage']['ehr_facts_sha256'] for row in group_rows):
        raise ValueError('frozen_conditioning_fixed_ehr_lineage_required')
    pins = {str(path): sha256_file(path) for path in (Path(__file__), PROTOCOL,
        ROOT / 'tests/test_report_order_control_v1.py', Path(control.__file__),
        Path(b.__file__), Path(b.policy_module.__file__), Path(matched.__file__), Path(headroom.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', protocol)
        outcomes = [replay(grid, policy, cap, method, order) for _, grid in sorted(bank.items())
            for method, order in orders for cap in b.CAPS]
        exact_grid(outcomes, bank, [method for method, _ in orders])
        choices = write_private_text(temporary / 'order_outcomes.jsonl',
            ''.join(json.dumps(row, sort_keys=True) + '\n' for row in outcomes))
        seal = sha256_file(choices)
        # Old choices/headroom and independent endpoint are read only after seal.
        cm = json.loads(b.checked(headroom.CONTROL / 'manifest.json', headroom.CONTROL_SHA, 1024**2, sources))
        old_controls = [json.loads(s) for s in b.artifact(headroom.CONTROL, cm, 'control_outcomes.jsonl', sources).splitlines()]
        exact_grid(old_controls, bank, ['fixed_image_guarded_report_prefix'])
        baseline = {(r['case_id'], r['model_call_budget']): r for r in old_controls}
        pm = json.loads(b.checked(gap.SOURCE / 'manifest.json', gap.SOURCE_SHA, 1024**2, sources))
        probes = [json.loads(s) for s in b.artifact(gap.SOURCE, pm, 'probe_outcomes.jsonl', sources).splitlines()]
        probes = [r for r in probes if r['method'] == b.NEW_METHODS[0]]
        exact_grid(probes, bank, [b.NEW_METHODS[0]])
        observations = {r['score_record']['triple_candidate_id']: b.policy_module.snapshot(r)
            for grid in bank.values() for r in grid.values()}
        hm = json.loads(b.checked(HEADROOM / 'manifest.json', HEADROOM_SHA, 1024**2, sources))
        head_rows = list(csv.DictReader(io.StringIO(b.artifact(HEADROOM, hm, 'case_headroom.csv', sources))))
        available = {(r['case_id'], int(r['model_call_budget'])): set(json.loads(r['dominating_paths'])) for r in head_rows}
        if len(head_rows) != 400 or len(available) != 400 or set(available) != set(baseline):
            raise ValueError('complete_sealed_headroom_grid_required')
        contrasts = []
        for row in outcomes:
            old = baseline[row['case_id'], row['model_call_budget']]
            if row['method'] == orders[0][0] and any(row[k] != old[k] for k in ('history', 'selected_observation', 'simulated_calls', 'selected_candidate_id')):
                raise ValueError('original_order_exact_replay_required')
            current = observations[row['selected_candidate_id']]
            previous = observations[old['selected_candidate_id']]
            if current != row['selected_observation'] or previous != old['selected_observation']:
                raise ValueError('exact_sealed_source_observations_required')
            contrasts.append({'case_id': row['case_id'], 'method': row['method'], 'model_call_budget': row['model_call_budget'],
                'same_final_candidate': current['candidate_id'] == previous['candidate_id'],
                'strictly_dominates_source_order': headroom.dominates(current, previous),
                'source_order_strictly_dominates': headroom.dominates(previous, current),
                'oracle_dominating_choice_reached': current['candidate_id'] in available[row['case_id'], row['model_call_budget']],
                'oracle_dominating_choice_available': bool(available[row['case_id'], row['model_call_budget']]),
                'clinical_improvement': None})
        endpoint_root, endpoint_manifest = b.parent(b.ENDPOINT, sources)
        endpoints = matched.endpoint_values(json.loads(b.artifact(endpoint_root, endpoint_manifest, 'scores.json', sources))['records'], observations)
        trials = [{'case_id': r['case_id'], 'method': r['method'], 'model_call_budget': r['model_call_budget'],
            'selected_candidate_id': r['selected_candidate_id'], 'calls': r['simulated_calls']} for r in (*outcomes, *probes)]
        cases, methods = b.aggregate(trials, observations, endpoints)
        if len(cases) != 2800 or len(methods) != 35:
            raise ValueError('all_schedules_and_probe_all_caps_required')
        grouped = grouped_metrics(cases, mapping)
        comparison = []
        for method, order in orders:
            for cap in b.CAPS:
                part = [r for r in contrasts if r['method'] == method and r['model_call_budget'] == cap]
                comparison.append({'method': method, 'report_order': ','.join(order), 'model_call_budget': cap,
                    'fixed_ehr_cases': len(part), **{k: sum(r[k] for r in part) for k in (
                        'same_final_candidate', 'strictly_dominates_source_order', 'source_order_strictly_dominates',
                        'oracle_dominating_choice_reached', 'oracle_dominating_choice_available')},
                    'clinical_improvement': None})
        write_private_text(temporary / 'case_means.csv', b.csv_text(cases))
        write_private_text(temporary / 'method_comparison.csv', b.csv_text(methods))
        write_private_text(temporary / 'conditioning_weighted_metrics.csv', b.csv_text(grouped))
        write_private_text(temporary / 'case_order_contrasts.csv', b.csv_text(contrasts))
        write_private_text(temporary / 'order_comparison.csv', b.csv_text(comparison))
        summary = {'schema_version': VERSION, 'status': 'all_six_report_orders_complete',
            'fixed_ehr_cases': 80, 'conditioning_groups': len(set(mapping.values())), 'schedules': 6,
            'new_control_trials': len(outcomes), 'all_caps': list(b.CAPS),
            'selected_sha256_before_endpoint': seal, 'source_order_exact_replay': True,
            'new_model_calls': 0, 'actual_regeneration_executed': False, 'clinical_qualified': False,
            'old_choices_or_rules_changed': False, 'secondary_used_to_choose_order': False,
            'best_order_is_heldout_policy': False, 'source_bodies_pixels_or_weights_read': False,
            'simulated_calls_are_measured_gpu_time': False}
        write_private_json(temporary / 'summary.json', summary)
        if sha256_file(choices) != seal or any(sha256_file(path) != pin for path, pin in sources.items()):
            raise ValueError('sealed_choices_or_source_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION + '-manifest',
            'sources': sources, 'code_pins': pins, 'selected_sha256_before_endpoint': seal,
            'original_selection_changed': False, 'clinical_qualified': False,
            'artifacts': {f.name: {'sha256': sha256_file(f)} for f in temporary.iterdir() if f.is_file()}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(b.BASE / 'report_order_controls'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, summary = run(args)
        print(json.dumps({'status': summary['status'], 'manifest_sha256': sha256_file(target / 'manifest.json'), 'new_model_calls': 0}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
