#!/usr/bin/env python3
"""Explain sealed replay differences; no new routing, scoring or model calls.

The post-hoc static choice may be unobserved by the controller. It is used only
for this diagnostic, never fed back to routing. Negative support is reported
separately, not dismissed as clinically wrong or promoted to truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import benchmark_probe_repair_v1 as benchmark
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12 import automatic_replay, probe_repair_v1 as p
from tricompose_v12.live_workers import check_pins

VERSION = 'tricompose-probe-gap-diagnostic-v1'
SOURCE = benchmark.BASE / 'probe_repair_runs/pool80_12784259_001'
SOURCE_SHA = '8834a745cc4fec47710be3935eda24b56a42f94af8662044fb3d9d434b908bd7'


def slot(view):
    return tuple(view['lineage'][k] for k in ('cxr_model_id', 'cxr_seed', 'report_model_id'))


def diagnose(probe, static, views, initial_id):
    """Bounded pure reconstruction, including rejected and failed observations."""
    if (probe['case_id'] != static['case_id'] or probe['model_call_budget'] != static['model_call_budget']
            or probe['method'] != benchmark.NEW_METHODS[0] or static['method'] != 'static_rerank'
            or probe['uses_secondary_endpoint'] is not False
            or probe['actual_regeneration_executed'] is not False
            or static['actual_regeneration_executed'] is not False):
        raise ValueError('same_case_cap_and_nonclinical_sealed_methods_required')
    by_slot = {slot(v): v for v in views.values()}
    if len(by_slot) != len(views): raise ValueError('unique_slots_required')
    current = views[initial_id]; p.validate_snapshot(current)
    for key, value in views.items():
        p.validate_snapshot(value)
        if (key != value['candidate_id'] or value['case_id'] != current['case_id']
                or any(value['lineage'][k] != current['lineage'][k] for k in ('ehr_sha256', 'ehr_facts_sha256'))):
            raise ValueError('one_unchanged_ehr_and_exact_candidate_ids_required')
    observed = {initial_id}; attempted = {slot(current)}; events = {}; calls = 4
    for index, h in enumerate(probe['history']):
        requested = tuple(h['slot']); action = h['action']
        if (h['step'] != index or requested in attempted or action not in p.COST
                or h['reserved_simulated_calls'] != p.COST[action]):
            raise ValueError('ordered_unique_reserved_attempts_required')
        attempted.add(requested); calls += p.COST[action]
        if h['failure_type'] is None:
            after = by_slot[requested]; cid = after['candidate_id']
            if cid in observed: raise ValueError('candidate_observed_once_required')
            credit = p.action_credit(current, after, action)
            if h['credit'] != credit or h['accepted'] != credit['replacement_allowed_under_proxy_contract']:
                raise ValueError('exact_original_action_credit_required')
            events[cid] = {'step': index, 'accepted': h['accepted'], 'reasons': credit['reasons']}
            observed.add(cid)
            if h['accepted']: current = after
        else:
            if (h['failure_type'] not in ('worker_failed', 'verification_failed', 'unavailable_cache_slot')
                    or h['credit'] is not None or h['accepted'] is not False):
                raise ValueError('typed_failed_attempt_not_observed_or_free_required')
        if h['committed_candidate_id'] != current['candidate_id']:
            raise ValueError('exact_committed_candidate_required')
    if (calls != probe['simulated_calls'] or calls > probe['model_call_budget']
            or probe['selected_candidate_id'] != current['candidate_id']
            or probe['selected_observation'] != current):
        raise ValueError('sealed_final_observation_and_budget_required')
    sid = static['selected_candidate_id']
    if sid is None: raise ValueError('completed_static_choice_required')
    chosen = views[sid]; event = events.get(sid)
    if sid == current['candidate_id']: category = 'same_final'
    elif sid == initial_id: category = 'initial_superseded'
    elif sid in observed: category = 'accepted_then_superseded' if event['accepted'] else 'observed_rejected'
    else: category = 'not_requested'
    a, b = p.edges(current), p.edges(chosen)
    required = p.needs(current)
    branch_same = slot(current)[:2] == slot(chosen)[:2]
    record = {'case_id': probe['case_id'], 'model_call_budget': probe['model_call_budget'],
        'probe_candidate_id': current['candidate_id'], 'static_candidate_id': sid,
        'static_choice_status': category, 'static_candidate_observed': sid in observed,
        'static_rejection_reasons': ','.join(event['reasons']) if event else '',
        'same_image_branch': branch_same,
        'same_image_bytes': current['lineage']['cxr_sha256'] == chosen['lineage']['cxr_sha256'],
        'static_image_branch_attempted': any(s[:2] == slot(chosen)[:2] for s in attempted),
        'final_report_probe_needed': required['report'], 'final_image_probe_needed': required['image'],
        'probe_simulated_calls': calls, 'static_simulated_calls': static['simulated_model_calls'],
        'static_total_support_is_clinical_superiority': False, 'clinical_fault_location': None,
        'diagnostic_unseen_choices_are_routing_inputs': False}
    for edge in p.EDGES:
        for field in ('known', 'comparable', 'support', 'positive_support', 'opposition'):
            record['static_minus_probe_' + edge + '_' + field] = len(b[edge][field]) - len(a[edge][field])
        record['static_minus_probe_' + edge + '_negative_support'] = (
            len(b[edge]['support'] - b[edge]['positive_support']) - len(a[edge]['support'] - a[edge]['positive_support']))
    record['static_has_higher_raw_cxr_report_support'] = record['static_minus_probe_cxr_report_support'] > 0
    return record


def summarize(records):
    result = []
    for cap in benchmark.CAPS:
        part = [r for r in records if r['model_call_budget'] == cap]
        reasons = Counter(reason for r in part if r['static_choice_status'] == 'observed_rejected'
            for reason in r['static_rejection_reasons'].split(',') if reason)
        result.append({'model_call_budget': cap, 'fixed_ehr_cases': len(part),
            'static_choice_status_case_counts': dict(Counter(r['static_choice_status'] for r in part)),
            'static_higher_support_case_counts_by_status': dict(Counter(r['static_choice_status'] for r in part
                if r['static_has_higher_raw_cxr_report_support'])),
            'observed_static_rejection_reasons_nonexclusive': dict(reasons),
            'unrequested_other_image_cases': sum(r['static_choice_status'] == 'not_requested' and not r['same_image_branch'] for r in part),
            'unrequested_other_image_without_final_image_trigger_cases': sum(r['static_choice_status'] == 'not_requested'
                and not r['same_image_branch'] and not r['final_image_probe_needed'] for r in part),
            **{'sum_static_minus_probe_cxr_report_' + name: sum(r['static_minus_probe_cxr_report_' + name] for r in part)
                for name in ('support', 'positive_support', 'negative_support')},
            'clinical_superiority': None, 'nonexclusive_reasons_are_not_extra_cases': True})
    return result


def run(args):
    benchmark.cpu_guard()  # Before metadata/body-independent cache reads.
    sources = {}; bank, policy = benchmark.load_primary(sources)
    manifest = json.loads(benchmark.checked(SOURCE / 'manifest.json', SOURCE_SHA, 1024**2, sources))
    if (manifest['actual_regeneration_executed'] is not False or manifest['clinical_qualified'] is not False
            or manifest['original_selection_changed'] is not False):
        raise ValueError('unmodified_development_replay_required')
    probes = [json.loads(line) for line in benchmark.artifact(SOURCE, manifest, 'probe_outcomes.jsonl', sources).splitlines()]
    expected = {(case, method, cap) for case in bank for method in benchmark.NEW_METHODS for cap in benchmark.CAPS}
    found = {(r['case_id'], r['method'], r['model_call_budget']) for r in probes}
    if len(probes) != len(expected) or found != expected: raise ValueError('all_eight_hundred_unique_trials_required')
    old_root, old_manifest = benchmark.parent(benchmark.REPLAY, sources)
    old = [json.loads(line) for line in benchmark.artifact(old_root, old_manifest, 'replay_outcomes.jsonl', sources).splitlines()]
    statics = [r for r in old if r['method'] == 'static_rerank']
    static_index = {(r['case_id'], r['model_call_budget']): r for r in statics}
    if len(statics) != 400 or len(static_index) != 400: raise ValueError('all_four_hundred_static_trials_required')
    records = []
    for case, grid in sorted(bank.items()):
        views = {c['score_record']['triple_candidate_id']: p.snapshot(c) for c in grid.values()}
        initial = grid[(*policy['image_order'][0], policy['report_order'][0])]['score_record']['triple_candidate_id']
        for cap in benchmark.CAPS:
            for feedback in (True, False):
                method = benchmark.NEW_METHODS[0 if feedback else 1]
                saved = next(r for r in probes if (r['case_id'], r['method'], r['model_call_budget']) == (case, method, cap))
                if benchmark.replay(grid, policy, cap, feedback) != saved:
                    raise ValueError('sealed_probe_replay_changed')
                if feedback: probe = saved
            static = static_index[case, cap]
            computed = automatic_replay.replay_case(grid, policy, 'static_rerank', cap)
            if any(static[k] != v for k, v in computed.items()): raise ValueError('sealed_static_replay_changed')
            records.append(diagnose(probe, static, views, initial))
    readouts = summarize(records)
    summary = {'schema_version': VERSION, 'status': 'sealed_probe_gap_explained_not_clinical',
        'fixed_ehr_cases': 80, 'case_cap_comparisons': len(records),
        'exact_probe_replays_checked': len(probes), 'exact_static_replays_checked': len(statics),
        'caps': readouts, 'new_model_calls': 0, 'source_bodies_pixels_or_weights_read': False,
        'old_scores_choices_and_rules_changed': False, 'clinical_superiority': None,
        'secondary_endpoints_read_or_used_to_retune': False,
        'posthoc_static_candidates_never_fed_to_controller': True}
    pins = {str(path): sha256_file(path) for path in (Path(__file__),
        ROOT / 'tests/test_probe_gap_diagnostic_v1.py', Path(benchmark.__file__), Path(p.__file__), Path(automatic_replay.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_text(temporary / 'case_gap_diagnostics.csv', benchmark.csv_text(records))
        write_private_json(temporary / 'summary.json', summary)
        if any(sha256_file(path) != pin for path, pin in sources.items()): raise ValueError('source_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION, 'sources': sources,
            'code_pins': pins, 'artifacts': {path.name: {'sha256': sha256_file(path)} for path in temporary.iterdir() if path.is_file()},
            'original_selection_changed': False, 'clinical_qualified': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(benchmark.BASE / 'probe_gap_diagnostics'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, summary = run(args)
        print(json.dumps({'status': summary['status'], 'new_model_calls': 0,
            'manifest_sha256': sha256_file(target / 'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__})); return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
