#!/usr/bin/env python3
"""Same-cap, paired development replay comparison; never retune selections.

Reuse sealed case means and conditioning-only groups, not clinical bodies or
models. Equal cap is NOT equal realized expenditure or measured GPU time.
Group bootstrap intervals are exploratory, not independent clinical evidence.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import check_conditioning_cluster_robustness as grouped
from benchmark_probe_repair_v1 import checked, cpu_guard, csv_text, CAPS, OLD_METHODS, NEW_METHODS
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.live_workers import check_pins

VERSION = 'tricompose-probe-same-cap-comparison-v1'
BASE = grouped.BASE
REPLAY_ROOT = BASE / 'probe_repair_runs/pool80_12784259_001'
REPLAY_SHA = '8834a745cc4fec47710be3935eda24b56a42f94af8662044fb3d9d434b908bd7'
GROUP_ROOT = BASE / 'conditioning_robustness_runs/conditioning_groups_12714150_001'
GROUP_SHA = '0d6e3ecf79c5a99cd500ee74cb3fee23695cd447ff768bd1c848593d04ce985c'
METHODS = (*OLD_METHODS, *NEW_METHODS)
METRICS = (*grouped.METRICS, 'cxr_report_positive_support_over_known')
CONTRASTS = tuple((NEW_METHODS[0], name) for name in
    ('fixed', 'static_rerank', 'targeted_heuristic', NEW_METHODS[1]))


def parse(raw, mapping, *, expected_cases=80):
    cases = {r['case_id'] for r in mapping}
    if len(cases) != len(mapping) or len(cases) != expected_cases:
        raise ValueError('one_frozen_group_per_fixed_case_required')
    expected = {(case, method, cap) for case in cases for method in METHODS for cap in CAPS}
    means = {}
    for row in raw:
        key = (row['case_id'], row['method'], int(row['model_call_budget']))
        if key not in expected or key in means:
            raise ValueError('complete_unique_same_cap_grid_required')
        reps = 25 if row['method'] == OLD_METHODS[-1] else 5 if row['method'] == 'random' else 1
        if str(row['replicates']) != str(reps):
            raise ValueError('seeds_must_be_averaged_within_case_before_comparison')
        values = {metric: grouped.number(row[metric.replace('coverage_over_known', 'comparable_over_known')])
            for metric in grouped.METRICS}
        calls = values['mean_simulated_calls']
        if calls is None or not 0 <= calls <= key[2]:
            raise ValueError('bounded_simulated_call_charge_required')
        for metric, value in values.items():
            if value is not None and metric != 'mean_simulated_calls':
                if not (-1 <= value <= 1 if metric == 'biovil_raw_cosine' else 0 <= value <= 1):
                    raise ValueError('bounded_finite_proxy_required')
        for edge in grouped.EDGES:
            s, o, c = (values[edge + '_' + k] for k in
                ('support_over_known', 'opposition_over_known', 'coverage_over_known'))
            if not (all(v is None for v in (s, o, c)) or
                    all(v is not None for v in (s, o, c)) and abs(s + o - c) < 1e-9):
                raise ValueError('unknown_safe_edge_arithmetic_required')
        known = grouped.number(row['cxr_report_known'])
        positive = grouped.number(row['cxr_report_positive_support'])
        if known is not None and known < 0 or positive is not None and (known is None or not 0 <= positive <= known):
            raise ValueError('positive_support_requires_known_denominator')
        values[METRICS[-1]] = positive / known if known and positive is not None else None
        means[key] = values
    if set(means) != expected:
        raise ValueError('all_cases_methods_and_caps_required')
    return means


def analyze(means, mapping):
    ids = {r['case_id']: r['conditioning_group_id'] for r in mapping}
    method_rows, contrasts = [], []
    for cap in CAPS:
        for method in METHODS:
            for metric in METRICS:
                pairs = [(ids[c], means[c, method, cap][metric]) for c in sorted(ids)
                    if means[c, method, cap][metric] is not None]
                result, _ = grouped.aggregate(pairs, len(ids))
                method_rows.append({'model_call_budget': cap, 'method': method, 'metric': metric, **result})
        for left, right in CONTRASTS:
            for metric in METRICS:
                pairs = [(ids[c], means[c, left, cap][metric] - means[c, right, cap][metric]) for c in sorted(ids)
                    if means[c, left, cap][metric] is not None and means[c, right, cap][metric] is not None]
                result = grouped.paired_interval(pairs, len(ids), draws=1000, seed=0)
                contrasts.append({'model_call_budget': cap, 'left_method': left, 'right_method': right,
                    'metric': metric, 'delta_direction': 'left_minus_right', 'same_budget_cap': True,
                    'equal_realized_calls_claimed': False, 'measured_gpu_savings_claimed': False,
                    'higher_metric_is_better': not (metric.endswith('opposition_over_known') or metric == 'mean_simulated_calls'),
                    **result})
    return method_rows, contrasts


def run(args):
    cpu_guard()  # Before any derived metadata open or directory creation.
    sources = {}
    def doc(root, pin):
        return json.loads(checked(root / 'manifest.json', pin, 1024**2, sources))
    rm, gm = doc(REPLAY_ROOT, REPLAY_SHA), doc(GROUP_ROOT, GROUP_SHA)
    if (rm['actual_regeneration_executed'] is not False or rm['clinical_qualified'] is not False
            or rm['original_selection_changed'] is not False or rm['new_model_calls'] != 0
            or gm['policy']['selection_changed'] is not False or gm['policy']['grouping_uses_outcomes'] is not False):
        raise ValueError('sealed_nonclinical_replay_and_outcome_free_groups_required')
    table = checked(REPLAY_ROOT / 'case_means.csv', rm['artifacts']['case_means.csv']['sha256'], 4*1024**2, sources)
    mapping = list(csv.DictReader(io.StringIO(checked(GROUP_ROOT / 'case_groups.csv',
        gm['artifacts']['case_groups.csv'], 1024**2, sources))))
    raw = list(csv.DictReader(io.StringIO(table)))
    means = parse(raw, mapping)
    methods, contrasts = analyze(means, mapping)
    summary = {'schema_version': VERSION, 'status': 'same_cap_development_replay_comparison_complete',
        'fixed_ehr_cases': 80, 'conditioning_groups': len({r['conditioning_group_id'] for r in mapping}),
        'case_method_cap_rows': len(means), 'method_metric_rows': len(methods), 'paired_contrasts': len(contrasts),
        'caps': CAPS, 'bootstrap_resamples': 1000, 'seed': 0,
        'case_and_equal_conditioning_estimands_retained': True, 'NA_replaced_with_zero': False,
        'budget_cap_is_not_equal_realized_expenditure': True, 'new_model_calls': 0,
        'actual_regeneration_executed': False, 'clinical_qualified': False,
        'posthoc_development_sensitivity': True, 'intervals_multiplicity_adjusted': False,
        'source_selection_unchanged': True, 'source_bodies_pixels_or_weights_read': False}
    pins = {str(p): sha256_file(p) for p in (Path(__file__),
        ROOT / 'tests/test_probe_budget_comparison_v1.py', Path(grouped.__file__))}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_text(temporary / 'method_means.csv', grouped.csv_text(methods))
        write_private_text(temporary / 'paired_contrasts.csv', grouped.csv_text(contrasts))
        write_private_json(temporary / 'summary.json', summary)
        if any(sha256_file(p) != pin for p, pin in sources.items()): raise ValueError('source_changed')
        check_pins(pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION, 'sources': sources,
            'code_pins': pins, 'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir() if p.is_file()},
            'original_selection_changed': False, 'clinical_qualified': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', default=str(BASE / 'probe_budget_comparisons'))
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    try:
        target, summary = run(args)
        print(json.dumps({'status': summary['status'], 'new_model_calls': 0,
            'manifest_sha256': sha256_file(target / 'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__})); return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
