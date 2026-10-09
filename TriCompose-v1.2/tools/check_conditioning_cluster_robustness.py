#!/usr/bin/env python3
"""Grouped robustness over sealed metadata; no models or clinical-body reads."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1')]
from contracts import (WORKSPACE, PROTECTED_ROOT, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
import score_cached_opacity_candidates as guard

BASE = PROTECTED_ROOT/'tricompose_v1_2'
DELIVERY = BASE/'deliverables/first_version_12714150_001'
DELIVERY_SHA = 'a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc'
PROTOCOL = WORKSPACE/'docs/conditioning_cluster_robustness_protocol.md'
TESTS = ROOT/'tests/test_conditioning_cluster_robustness.py'
SCHEMA = 'tricompose-exact-conditioning-robustness-v1'
MODELS = ('chexgenbench_pixart', 'chexgenbench_sana', 'roentgen_v2')
EXPERTS = ('chexagent2', 'cxrmate_single', 'llavarad', 'maira2')
METHODS = ('fixed', 'random', 'static_rerank', 'targeted_heuristic', 'random_acquisition_score_free_final')
CAPS = (4, 8, 12, 20, 30)
EDGES = ('ehr_cxr', 'ehr_report', 'cxr_report')
METRICS = ('biovil_raw_cosine', *(edge+'_'+suffix for edge in EDGES
    for suffix in ('support_over_known', 'opposition_over_known', 'coverage_over_known')), 'mean_simulated_calls')
CONTRASTS = (('static_rerank', 'fixed'), ('static_rerank', 'random_acquisition_score_free_final'),
    ('random', 'random_acquisition_score_free_final'), ('targeted_heuristic', 'static_rerank'),
    ('targeted_heuristic', 'random_acquisition_score_free_final'))
POLICY = {'cohort': 'already_inspected_development_bank', 'posthoc_sensitivity': True,
    'grouping': 'ordered_three_model_prompt_sha256_tuple', 'grouping_uses_outcomes': False,
    'exact_prompt_identity_is_clinical_ehr_equivalence': False, 'all_ehr_cases_retained': True,
    'seed_replicates_are_independent_patients': False, 'bootstrap_resamples': 1000, 'seed': 0,
    'intervals_multiplicity_adjusted': False, 'clinical_qualified': False,
    'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
    'new_model_calls': 0, 'simulated_calls_are_measured_gpu_time': False,
    'new_slurm_submissions': 0, 'clinical_bodies_or_images_read': False}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def groups(index, expected_cases=80):
    """Only identity/hash fields influence grouping; scores are unused."""
    case_rows = defaultdict(list)
    for row in index:
        require(isinstance(row['case_id'], str) and re.fullmatch(r'case_[0-9]{3,4}', row['case_id']), 'opaque_case_required')
        case_rows[row['case_id']].append(row)
    require(len(case_rows) == expected_cases and len(index) == expected_cases*12, 'complete_case_grid_required')
    signatures, output = {}, []
    group_images = defaultdict(set)
    model_prompts, model_images = defaultdict(set), defaultdict(set)
    seen = set()
    for case, rows in sorted(case_rows.items()):
        slots, prompts, images, hashes = set(), {}, {}, set()
        for row in rows:
            model, expert = row['cxr_model_id'], row['report_model_id']
            require(model in MODELS and expert in EXPERTS and str(row['cxr_seed']) == '0', 'exact_model_seed_inventory_required')
            slot = (model, expert)
            require(slot not in slots and row['triple_candidate_id'] not in seen, 'duplicate_slot_or_triple_refused')
            slots.add(slot); seen.add(row['triple_candidate_id'])
            for key in ('prompt_sha256', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256'):
                require(isinstance(row[key], str) and re.fullmatch('[0-9a-f]{64}', row[key]), 'sha256_identity_required')
            require(prompts.setdefault(model, row['prompt_sha256']) == row['prompt_sha256'] and
                images.setdefault(model, row['cxr_sha256']) == row['cxr_sha256'], 'shared_image_prompt_metadata_differs')
            hashes.add((row['ehr_sha256'], row['ehr_facts_sha256']))
        require(len(hashes) == 1 and slots == {(m, e) for m in MODELS for e in EXPERTS}, 'fixed_ehr_complete_grid_required')
        signature = tuple((m, prompts[m]) for m in MODELS)
        gid = signatures.setdefault(signature, 'conditioning_%04d' % len(signatures))
        signature_hash = hashlib.sha256(json.dumps(signature, separators=(',', ':')).encode()).hexdigest()
        output.append({'case_id': case, 'conditioning_group_id': gid, 'conditioning_signature_sha256': signature_hash,
            'ehr_sha256': next(iter(hashes))[0], 'ehr_facts_sha256': next(iter(hashes))[1]})
        group_images[gid].add(tuple(images[m] for m in MODELS))
        for model in MODELS:
            model_prompts[model].add(prompts[model]); model_images[model].add(images[model])
    sizes = Counter(r['conditioning_group_id'] for r in output)
    inventory = {'fixed_ehr_cases': len(case_rows), 'candidate_slots': len(index),
        'exact_conditioning_groups': len(sizes), 'distinct_ehr_hashes': len({r['ehr_sha256'] for r in output}),
        'case_slots_beyond_one_per_conditioning_group': len(output)-len(sizes),
        'group_size_histogram': dict(sorted(Counter(sizes.values()).items())),
        'largest_group_cases': max(sizes.values()),
        'groups_with_identical_three_image_hash_vectors': sum(len(v) == 1 for v in group_images.values()),
        'groups_with_multiple_three_image_hash_vectors': sum(len(v) > 1 for v in group_images.values()),
        'unique_prompt_hashes_per_model': {m: len(model_prompts[m]) for m in MODELS},
        'unique_image_hashes_per_model': {m: len(model_images[m]) for m in MODELS}}
    return output, inventory


def number(value):
    if value in ('', None):
        return None
    require(type(value) in (str, int, float), 'finite_cached_number_required')
    parsed = float(value)
    require(math.isfinite(parsed), 'finite_cached_number_required')
    return parsed


def parse_means(raw, mapping):
    cases = {r['case_id'] for r in mapping}
    expected = {(c, m, cap) for c in cases for m in METHODS for cap in CAPS}
    result = {}
    for row in raw:
        require(str(row['model_call_budget']).isdigit(), 'integer_cap_required')
        cap = int(row['model_call_budget'])
        key = (row['case_id'], row['method'], cap)
        require(key in expected and key not in result, 'exact_unique_method_cap_inventory_required')
        seeds = 25 if row['method'] == METHODS[-1] else 5 if row['method'] == 'random' else 1
        require(str(row['seed_replicates']) == str(seeds), 'all_existing_seed_settings_required')
        values = {metric: number(row[metric]) for metric in METRICS}
        calls = values['mean_simulated_calls']
        require(calls is not None and 0 <= calls <= cap, 'bounded_simulated_calls_required')
        for metric, value in values.items():
            if value is not None and metric != 'mean_simulated_calls':
                require(-1.00001 <= value <= 1.00001 if metric == 'biovil_raw_cosine' else 0 <= value <= 1,
                        'bounded_cached_proxy_required')
        for edge in EDGES:
            support, oppose, coverage = (values[edge+'_'+s] for s in
                ('support_over_known', 'opposition_over_known', 'coverage_over_known'))
            require(all(v is None for v in (support, oppose, coverage)) or
                all(v is not None for v in (support, oppose, coverage)) and math.isclose(support+oppose, coverage, abs_tol=1e-10),
                'unknown_safe_cached_edge_arithmetic_required')
        result[key] = values
    require(set(result) == expected, 'every_case_method_cap_retained_required')
    return result


def aggregate(pairs, attempted):
    """Pairs are (group ID, numeric value); missing pairs were explicitly excluded."""
    grouped = defaultdict(list)
    for gid, value in pairs:
        require(isinstance(gid, str) and type(value) in (float, int) and math.isfinite(value), 'finite_group_values_required')
        grouped[gid].append(value)
    n = len(pairs)
    require(type(attempted) is int and 0 <= n <= attempted, 'explicit_attempted_denominator_required')
    keys = sorted(grouped)
    result = {'attempted_ehr_cases': attempted, 'available_ehr_cases': n, 'available_conditioning_groups': len(keys),
        'case_weighted_mean': sum(v for _, v in pairs)/n if n else None,
        'equal_conditioning_mean': sum(sum(grouped[k])/len(grouped[k]) for k in keys)/len(keys) if keys else None,
        'constant_observed_values': len({v for _, v in pairs}) == 1 if n else None}
    return result, grouped


def quantile(ordered, probability):
    position = (len(ordered)-1)*probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def paired_interval(pairs, attempted, draws=1000, seed=0):
    require(type(draws) is int and 1 <= draws <= 1000 and type(seed) is int, 'bounded_deterministic_bootstrap_required')
    result, grouped = aggregate(pairs, attempted)
    result.update(bootstrap_resamples_requested=draws, bootstrap_resamples_used=0,
        case_weighted_interval_95=None, equal_conditioning_interval_95=None, bootstrap_seed=seed,
        interval_status='insufficient_conditioning_groups')
    if len(grouped) < 3:
        return result
    statistics = [(sum(grouped[k]), len(grouped[k]), sum(grouped[k])/len(grouped[k])) for k in sorted(grouped)]
    rng = random.Random(seed)
    case_values, group_values = [], []
    for _ in range(draws):
        sampled = [statistics[rng.randrange(len(statistics))] for _ in statistics]
        case_values.append(sum(s[0] for s in sampled)/sum(s[1] for s in sampled))
        group_values.append(sum(s[2] for s in sampled)/len(sampled))
    result.update(bootstrap_resamples_used=draws, interval_status='complete',
        case_weighted_interval_95=[quantile(sorted(case_values), p) for p in (.025, .975)],
        equal_conditioning_interval_95=[quantile(sorted(group_values), p) for p in (.025, .975)])
    return result


def analyze(means, mapping):
    ids = {r['case_id']: r['conditioning_group_id'] for r in mapping}
    require(len(ids) == len(mapping), 'unique_group_mapping_required')
    cases = sorted(ids)
    method_rows, contrast_rows = [], []
    for cap in CAPS:
        for method in METHODS:
            for metric in METRICS:
                values = [(ids[c], means[c, method, cap][metric]) for c in cases if means[c, method, cap][metric] is not None]
                result, _ = aggregate(values, len(cases))
                method_rows.append({'model_call_budget': cap, 'method': method, 'metric': metric, **result})
        for left, right in CONTRASTS:
            for metric in METRICS:
                pairs = [(ids[c], means[c, left, cap][metric]-means[c, right, cap][metric]) for c in cases
                    if means[c, left, cap][metric] is not None and means[c, right, cap][metric] is not None]
                result = paired_interval(pairs, len(cases))
                contrast_rows.append({'model_call_budget': cap, 'left_method': left, 'right_method': right,
                    'metric': metric, 'delta_direction': 'left_minus_right',
                    'higher_metric_is_better': not (metric.endswith('opposition_over_known') or metric == 'mean_simulated_calls'),
                    'higher_is_better_is_only_a_proxy_convention': True,
                    'same_acquisition_and_expenditure': left == 'random' and right == METHODS[-1], **result})
    return method_rows, contrast_rows


def source_pins():
    guard.guard()
    pins = {str(p.resolve()): sha256_file(p) for p in (Path(__file__), TESTS, PROTOCOL,
        Path(guard.__file__), WORKSPACE/'TriCompose-v1.0/eval/report_v1_1/contracts.py')}
    manifest = guard.metadata(DELIVERY/'manifest.json', pins, DELIVERY_SHA)
    require(manifest['schema_version'] == 'tricompose-first-version-protected-delivery-v1' and
        manifest['original_selection_changed'] is manifest['clinical_qualified'] is manifest['regeneration_authorized'] is False,
        'unchanged_engineering_delivery_required')
    for name in ('candidate_index.csv', 'baseline_case_means.csv'):
        path = DELIVERY/name
        require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 8*1024**2 and
            sha256_file(path) == manifest['artifacts'][name]['sha256'], 'sealed_bounded_metadata_required')
        pins[str(path)] = manifest['artifacts'][name]['sha256']
    require(sha256_file(guard.BANK/'manifest.json') == guard.BANK_SHA, 'old_bank_changed')
    pins[str(guard.BANK/'manifest.json')] = guard.BANK_SHA
    return pins


def private_json(path, value):
    write_private_json(path, value)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows({k: json.dumps(v, separators=(',', ':')) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    return stream.getvalue()


def prepare(run_id):
    pins = source_pins()
    index = guard.metadata(DELIVERY/'candidate_index.csv', pins, pins[str(DELIVERY/'candidate_index.csv')])
    mapping, inventory = groups(index)
    temporary, target = new_atomic_run(BASE/'conditioning_robustness_plans', run_id)
    try:
        private_json(temporary/'plan.json', {'schema_version': SCHEMA+'-plan', 'policy': POLICY,
            'metrics': METRICS, 'contrasts': CONTRASTS, 'caps': CAPS, 'inventory': inventory,
            'method_case_means_decoded_during_preparation': False, 'grouping_outcome_columns_accessed': False,
            'historical_results_available_to_investigator': True})
        private_json(temporary/'case_groups.json', {'records': mapping})
        guard.verify_pins(pins)
        private_json(temporary/'manifest.json', {'schema_version': SCHEMA+'-plan-manifest', 'sources': pins,
            'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def render(inventory, methods, contrasts):
    show = lambda v: 'NA' if v is None else f'{v:.4f}'
    lines = ['# Conditioning-group robustness / 重复生成条件稳健性', '',
        f"All{inventory['fixed_ehr_cases']} fixed EHRs retained; {inventory['exact_conditioning_groups']} exact three-model prompt signatures.",
        f"Repeated case slots beyond one per signature: {inventory['case_slots_beyond_one_per_conditioning_group']}.",
        'Hash identity is not clinical equivalence; this is exploratory DEVELOPMENT sensitivity, not clinical validation.', '',
        '## All budgets, all methods / 全预算与方法', '',
        '| Cap | Method | Calls case / condition-equal | BioViL case / condition-equal | CXR-report support case / condition-equal |',
        '| ---: | --- | --- | --- | --- |']
    for cap in CAPS:
        for method in METHODS:
            selected = {r['metric']: r for r in methods if r['model_call_budget'] == cap and r['method'] == method}
            columns = [show(selected[k]['case_weighted_mean'])+' / '+show(selected[k]['equal_conditioning_mean'])
                for k in ('mean_simulated_calls', 'biovil_raw_cosine', 'cxr_report_support_over_known')]
            lines.append(f'| {cap} | {method} | '+' | '.join(columns)+' |')
    lines += ['', '## Every-cap paired BioViL contrast / 全预算配对对比', '',
        '| Cap | Left − right | Available EHR / groups | Case-weighted delta [95% cluster CI] | Equal-conditioning delta [95% cluster CI] |',
        '| ---: | --- | --- | --- | --- |']
    for row in contrasts:
        if row['metric'] != 'biovil_raw_cosine':
            continue
        intervals = []
        for prefix in ('case_weighted', 'equal_conditioning'):
            values = row[prefix+'_interval_95']
            ci = 'NA' if values is None else '['+', '.join(show(v) for v in values)+']'
            intervals.append(show(row[prefix+'_mean'])+' '+ci)
        lines.append(f"| {row['model_call_budget']} | {row['left_method']} − {row['right_method']} | "
            f"{row['available_ehr_cases']}/{row['available_conditioning_groups']} | "+' | '.join(intervals)+' |')
    lines += ['', 'All11 metrics and all275 paired contrasts, including EHR-edge NA denominators, remain in CSV/JSON.',
        'Support and opposition are cached classifier/labeler proxies; BioViL is unqualified secondary evidence.',
        'Intervals are descriptive, not multiplicity-adjusted and do not correct scorer bias.',
        'Equal-conditioning weighting changes the estimand; it does not delete EHRs or change generation/selection.',
        'All seeds are already averaged within EHR. No independent-patient, repair-effectiveness or measured GPU-saving claim.',
        'Costs are simulated calls. Only random-acquisition/scored-final vs score-free final shares acquisition/expenditure.',
        '旧score table、80个固定EHR、既有选优、首版下载包和benchmark均未改动。', '']
    return '\n'.join(lines)


def evaluate(plan_root, expected, run_id):
    guard.guard()
    pins = {}
    manifest = guard.metadata(Path(plan_root)/'manifest.json', pins, expected)
    require(manifest['schema_version'] == SCHEMA+'-plan-manifest', 'frozen_cluster_plan_required')
    guard.verify_pins(manifest['sources']); pins.update(manifest['sources'])
    plan = guard.metadata(Path(plan_root)/'plan.json', pins, manifest['artifacts']['plan.json'])
    mapping = guard.metadata(Path(plan_root)/'case_groups.json', pins, manifest['artifacts']['case_groups.json'])['records']
    require(plan['policy'] == POLICY and plan['metrics'] == list(METRICS) and
        plan['caps'] == list(CAPS) and plan['contrasts'] == [list(p) for p in CONTRASTS], 'unchanged_grouping_statistics_required')
    current = source_pins()
    require(all(pins.get(p) == h for p, h in current.items()), 'frozen_source_closure_required')
    index = guard.metadata(DELIVERY/'candidate_index.csv', pins, current[str(DELIVERY/'candidate_index.csv')])
    rebuilt, inventory = groups(index)
    require(rebuilt == mapping and json.loads(json.dumps(inventory)) == plan['inventory'], 'frozen_case_mapping_required')
    raw = guard.metadata(DELIVERY/'baseline_case_means.csv', pins, current[str(DELIVERY/'baseline_case_means.csv')])
    means = parse_means(raw, mapping)
    temporary, target = new_atomic_run(BASE/'conditioning_robustness_runs', run_id)
    started = time.monotonic()
    try:
        methods, contrasts = analyze(means, mapping)
        summary = {'schema_version': SCHEMA, 'policy': POLICY, 'inventory': inventory,
            'case_mean_rows': len(means), 'method_summary_rows': len(methods), 'paired_contrast_rows': len(contrasts),
            'actual_slurm_job_id': os.environ['SLURM_JOB_ID'], 'plan_manifest_sha256': expected,
            'delivery_manifest_sha256': DELIVERY_SHA, 'elapsed_cpu_seconds': round(time.monotonic()-started, 6)}
        private_json(temporary/'summary.json', summary)
        private_json(temporary/'paired_contrasts.json', {'records': contrasts})
        write_private_text(temporary/'case_groups.csv', csv_text(mapping))
        write_private_text(temporary/'method_means.csv', csv_text(methods))
        write_private_text(temporary/'paired_contrasts.csv', csv_text(contrasts))
        write_private_text(temporary/'RESULTS_CN_EN.md', render(inventory, methods, contrasts))
        guard.verify_pins(pins)
        private_json(temporary/'manifest.json', {'schema_version': SCHEMA+'-run-manifest', 'sources': pins,
            'policy': POLICY, 'artifacts': {p.name: sha256_file(p) for p in sorted(temporary.iterdir())}})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('prepare', 'evaluate'))
    cli.add_argument('--run-id', required=True)
    cli.add_argument('--plan-root', type=Path)
    cli.add_argument('--plan-manifest-sha256')
    args = cli.parse_args()
    os.umask(0o007)
    try:
        target = prepare(args.run_id) if args.action == 'prepare' else evaluate(args.plan_root, args.plan_manifest_sha256, args.run_id)
    except Exception as error:
        print(json.dumps({'status': 'conditioning_robustness_failed_closed', 'error_type': type(error).__name__, 'source_text_exposed': False}))
        return 2
    print(json.dumps({'status': 'conditioning_robustness_'+args.action+'_complete', 'new_model_calls': 0,
        'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
