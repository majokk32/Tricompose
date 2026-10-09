#!/usr/bin/env python3
"""Finite-cache upper bound under unchanged acceptance; not a routing policy."""
import argparse
from collections import Counter
import heapq
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_probe_repair_v1 as b
from tricompose_v12 import probe_repair_v1 as p
from tricompose_v12.live_workers import check_pins
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

VERSION = 'tricompose-unchanged-guard-headroom-v1'
PROTOCOL = ROOT / 'configs/guard_headroom_v1.json'
CONTROL = b.BASE / 'matched_guard_controls/matched_reports_12784259_001'
CONTROL_SHA = '8f4c1fd053f975cbbc229b843df1b97449880afa49da67c8792cbbda8c8d1b4f'


def image(view):
    return tuple(view['lineage'][k] for k in ('cxr_model_id', 'cxr_seed'))


def graph(views):
    """Only accepted, single-modality paired interventions become edges."""
    output = {cid: [] for cid in views}
    for before in views.values():
        for after in views.values():
            if before['candidate_id'] == after['candidate_id']:
                continue
            if image(before) == image(after):
                action = 'report_probe'
            elif before['lineage']['report_model_id'] == after['lineage']['report_model_id']:
                action = 'image_probe'
            else:
                continue
            if p.action_credit(before, after, action)['replacement_allowed_under_proxy_contract']:
                output[before['candidate_id']].append((after['candidate_id'], action, p.COST[action]))
    return output


def shortest_paths(adjacency, initial, cap):
    """Perfect-information cost bound: rejected attempts are avoided, not free."""
    if type(cap) is not int or cap < 4 or initial not in adjacency:
        raise ValueError('registered_initial_and_integer_budget_required')
    distances = {initial: 4}
    paths = {initial: []}
    queue = [(4, initial)]
    while queue:
        cost, cid = heapq.heappop(queue)
        if cost != distances[cid]:
            continue
        for target, action, charge in adjacency[cid]:
            if target not in adjacency or charge != p.COST[action]:
                raise ValueError('registered_target_and_exact_action_cost_required')
            next_cost = cost + charge
            if next_cost <= cap and next_cost < distances.get(target, float('inf')):
                distances[target] = next_cost
                paths[target] = paths[cid] + [{'action': action, 'candidate_id': target}]
                heapq.heappush(queue, (next_cost, target))
    return distances, paths


def dominates(candidate, baseline):
    """Strict improvement in protected facts/oppositions; no scalar endpoint."""
    a, z = p.edges(baseline), p.edges(candidate)
    if (baseline['case_id'] != candidate['case_id'] or any(
            baseline['lineage'][k] != candidate['lineage'][k]
            for k in ('ehr_sha256', 'ehr_facts_sha256'))):
        raise ValueError('same_fixed_ehr_required')
    q0, q1 = (v['quality']['report_structure_quality_score_0_1'] for v in (baseline, candidate))
    if (q0 is None or q1 is None or q1 < q0 or candidate['artifact_gate_failures']
            or not candidate['quality']['cxr_basic_validity_pass']):
        return False
    strict = not baseline['quality']['cxr_basic_validity_pass']
    for edge in p.EDGES:
        protected = 'positive_support' if edge == 'cxr_report' else 'support'
        if (not a[edge]['comparable'] <= z[edge]['comparable']
                or not a[edge][protected] <= z[edge][protected]
                or not z[edge]['opposition'] <= a[edge]['opposition']):
            return False
        strict |= bool(z[edge][protected] - a[edge][protected]
            or a[edge]['opposition'] - z[edge]['opposition'])
    return bool(strict)


def run(args):
    b.cpu_guard()
    sources = {}
    bank, policy = b.load_primary(sources)
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['caps'] != list(b.CAPS) or protocol['secondary_endpoint_read'] is not False:
        raise ValueError('declared_all_caps_no_secondary_required')
    pins = {str(path): sha256_file(path) for path in (Path(__file__), PROTOCOL,
        ROOT / 'tests/test_guard_headroom_v1.py', Path(p.__file__), Path(b.__file__))}
    graphs, views_by_case, distances_by_case = {}, {}, {}
    for case, grid in sorted(bank.items()):
        views = {row['score_record']['triple_candidate_id']: p.snapshot(row) for row in grid.values()}
        initial = grid[(*policy['image_order'][0], policy['report_order'][0])]['score_record']['triple_candidate_id']
        views_by_case[case] = views
        graphs[case] = graph(views)
        distances_by_case[case] = shortest_paths(graphs[case], initial, max(b.CAPS))
    # Only after computing reachability read the existing control choices.
    manifest = json.loads(b.checked(CONTROL / 'manifest.json', CONTROL_SHA, 1024**2, sources))
    controls = [json.loads(line) for line in b.artifact(CONTROL, manifest, 'control_outcomes.jsonl', sources).splitlines()]
    expected = {(case, cap) for case in bank for cap in b.CAPS}
    if len(controls) != 400 or {(r['case_id'], r['model_call_budget']) for r in controls} != expected:
        raise ValueError('complete_unique_fixed_control_grid_required')
    rows = []
    for control in controls:
        case, cap = control['case_id'], control['model_call_budget']
        views = views_by_case[case]
        baseline = views[control['selected_candidate_id']]
        if baseline != control['selected_observation']:
            raise ValueError('exact_sealed_control_observation_required')
        distances, paths = distances_by_case[case]
        reachable = [cid for cid in distances if distances[cid] <= cap]
        better = sorted(cid for cid in reachable if dominates(views[cid], baseline))
        if baseline['candidate_id'] not in reachable:
            raise ValueError('fixed_prefix_choice_must_be_guard_reachable')
        rows.append({'case_id': case, 'model_call_budget': cap,
            'reachable_candidate_count': len(reachable), 'dominating_candidate_count': len(better),
            'fixed_candidate_minimum_oracle_simulated_calls': distances[baseline['candidate_id']],
            'fixed_actual_replay_simulated_calls': control['simulated_calls'],
            'dominating_minimum_oracle_simulated_calls': min((distances[cid] for cid in better), default=None),
            'dominating_paths': json.dumps({cid: paths[cid] for cid in better}, sort_keys=True),
            'clinical_improvement': None})
    summary = {'schema_version': VERSION, 'status': 'unchanged_guard_headroom_diagnostic_complete',
        'fixed_ehr_cases': 80, 'case_cap_rows': 400,
        'cases_with_dominating_path_by_cap': {str(cap): sum(r['dominating_candidate_count'] > 0
            for r in rows if r['model_call_budget'] == cap) for cap in b.CAPS},
        'accepted_graph_edges': sum(len(edges) for adjacency in graphs.values() for edges in adjacency.values()),
        'full_information_oracle': True, 'oracle_avoids_rejected_attempts': True,
        'secondary_endpoint_read': False, 'new_model_calls': 0,
        'actual_regeneration_executed': False, 'clinical_qualified': False,
        'old_choices_or_rules_changed': False, 'source_bodies_pixels_or_weights_read': False}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_json(temporary / 'frozen_protocol.json', protocol)
        write_private_text(temporary / 'case_headroom.csv', b.csv_text(rows))
        write_private_json(temporary / 'summary.json', summary)
        check_pins(pins)
        if any(sha256_file(path) != pin for path, pin in sources.items()):
            raise ValueError('source_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION + '-manifest',
            'sources': sources, 'code_pins': pins, 'original_selection_changed': False,
            'clinical_qualified': False, 'artifacts': {f.name: {'sha256': sha256_file(f)}
                for f in temporary.iterdir() if f.is_file()}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(b.BASE / 'guard_headroom_diagnostics'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, summary = run(args)
        print(json.dumps({'status': summary['status'], 'manifest_sha256': sha256_file(target / 'manifest.json'),
            'cases_with_dominating_path_by_cap': summary['cases_with_dominating_path_by_cap'], 'new_model_calls': 0}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
