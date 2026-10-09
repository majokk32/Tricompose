#!/usr/bin/env python3
"""Selector-only score-free random control; cached proxies, no clinical truth."""
from __future__ import annotations

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
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

SCHEMA = 'tricompose-score-free-final-choice-control-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
PARENTS = {
    'replay': (BASE / 'automatic_replays/automatic_replay_pool80_12576792_001',
        'af82764ac90f9af5b701c577fb2f9bfe6ecf4052310299e8e7cdcb415d2aed5f',
        ('frozen_policy.json', 'replay_outcomes.jsonl')),
    'complete': (BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001',
        'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c',
        ('candidate_score_table.csv', 'scores.json')),
}
CAPS, SEEDS = (4, 8, 12, 20, 30), (0, 1, 2, 3, 4)
OLD_METHODS = ('fixed', 'random', 'static_rerank', 'targeted_heuristic')
NEW_METHOD = 'random_acquisition_score_free_final'
EDGES = ('ehr_cxr', 'ehr_report', 'cxr_report')
COUNT_FIELDS = ('known_reference_facts', 'comparable_facts', 'supported_facts',
    'supported_positive', 'supported_negative', 'proxy_opposition_facts')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def cpu_guard():
    job = os.environ.get('SLURM_JOB_ID', '')
    if (not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text()
            or os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')):
        raise RuntimeError('existing_cpu_metadata_slurm_required')


def load():
    inputs, sources = {}, {'worker': Path(__file__),
        'tests': ROOT / 'tests/test_score_free_random_control.py',
        'protocol': ROOT.parent / 'docs/score_free_random_control_protocol.md',
        'atomic_contracts': Path(sys.modules['contracts'].__file__)}
    for label, (root, expected, files) in PARENTS.items():
        mp = require_inside(root / 'manifest.json', PROTECTED_ROOT, must_exist=True)
        if mp.stat().st_size > 1024 ** 2 or sha256_file(mp) != expected:
            raise ValueError('fixed_source_manifest_required')
        manifest = json.loads(mp.read_text())
        if manifest['original_selection_changed'] is not False:
            raise ValueError('unchanged_old_selection_required')
        sources[label + '_manifest'] = mp
        for name in files:
            path = require_inside(root / name, root, must_exist=True)
            if (path.stat().st_size > 32 * 1024 ** 2
                    or sha256_file(path) != manifest['artifacts'][name]['sha256']):
                raise ValueError('bounded_sealed_cache_required')
            text = path.read_text()
            if name.endswith('.csv'):
                value = list(csv.DictReader(io.StringIO(text)))
            elif name.endswith('.jsonl'):
                value = [json.loads(line) for line in text.splitlines() if line]
            else:
                value = json.loads(text)
            if sha256_file(path) != manifest['artifacts'][name]['sha256']:
                raise ValueError('cache_changed_during_read')
            inputs[label, name] = value
            sources[label + '_' + name] = path
        if sha256_file(mp) != expected:
            raise ValueError('source_manifest_changed_during_read')
    return inputs, sources


def integer(value):
    if isinstance(value, str):
        if not value.isdecimal():
            raise ValueError('nonnegative_count_required')
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative_count_required')
    return value


def candidate_readout(row):
    prefix = json.loads(row['source_primary_prefix'])
    if not isinstance(prefix, list) or len(prefix) != 4:
        raise ValueError('unchanged_legacy_prefix_inventory_required')
    gate = integer(prefix[0])
    value = None if row['biovil_raw_cosine'] == '' else float(row['biovil_raw_cosine'])
    if value is not None and (not math.isfinite(value) or abs(value) > 1.00001):
        raise ValueError('finite_uncalibrated_endpoint_required')
    result = {'artifact_gate_pass': int(gate == 0), 'biovil_raw_cosine': value}
    for edge in EDGES:
        counts = {field: integer(row[edge + '_' + field]) for field in COUNT_FIELDS}
        known, comparable = counts['known_reference_facts'], counts['comparable_facts']
        if (not 0 <= comparable <= known <= 14
                or counts['supported_facts'] + counts['proxy_opposition_facts'] != comparable
                or counts['supported_positive'] + counts['supported_negative'] != counts['supported_facts']):
            raise ValueError('raw_unknown_safe_counter_arithmetic_required')
        result.update({edge + '_' + k: v for k, v in counts.items()})
        result[edge + '_coverage_over_known'] = comparable / known if known else None
        result[edge + '_support_over_known'] = counts['supported_facts'] / known if known else None
        result[edge + '_opposition_over_known'] = counts['proxy_opposition_facts'] / known if known else None
    return result


def artifact_gate(row):
    prefix = json.loads(row['source_primary_prefix'])
    if not isinstance(prefix, list) or len(prefix) != 4:
        raise ValueError('unchanged_legacy_prefix_inventory_required')
    return integer(prefix[0]) == 0


def uniform_choice(case_id, cap, acquisition_seed, final_seed, observed_gate_records):
    """Blind selector: only IDs and the shared artifact-gate result are accepted."""
    if (not isinstance(case_id, str) or not case_id
            or any(type(x) is not int or x < 0 for x in (cap, acquisition_seed, final_seed))
            or any(set(r) != {'candidate_id', 'artifact_gate_pass'}
                or not isinstance(r['candidate_id'], str) or not r['candidate_id']
                or type(r['artifact_gate_pass']) is not bool for r in observed_gate_records)):
        raise ValueError('score_free_selector_fields_required')
    ids = [r['candidate_id'] for r in observed_gate_records]
    if len(ids) != len(set(ids)):
        raise ValueError('unique_observed_candidates_required')
    eligible = sorted(r['candidate_id'] for r in observed_gate_records if r['artifact_gate_pass'])
    if not eligible:
        return None
    material = f'score-free-final-choice-v1|{case_id}|{cap}|{acquisition_seed}|{final_seed}'
    rng = random.Random(int(hashlib.sha256(material.encode()).hexdigest(), 16))
    return eligible[rng.randrange(len(eligible))]


def validate_trace(row, candidates):
    trace, seen, images, calls = row['action_trace'], set(), set(), Counter()
    total = 0
    for i, action in enumerate(trace):
        cid = action['observed_candidate_id']
        if cid not in candidates or cid in seen or action['step'] != i:
            raise ValueError('unique_existing_observed_trace_required')
        candidate = candidates[cid]
        slot = action['request_slot']
        if (candidate['case_id'] != row['case_id'] or candidate['ehr_sha256'] != row['selected_ehr_sha256']
                or len(slot) != 3 or slot[2] != candidate['report_model_id']):
            raise ValueError('fixed_same_case_ehr_and_report_slot_required')
        image = candidate['cxr_candidate_id']
        charge = 2 + (0 if image in images else 2)
        total += charge
        if action['charged_model_calls'] != charge or action['cumulative_model_calls'] != total:
            raise ValueError('exact_unchanged_simulated_attempt_charges_required')
        if image not in images:
            calls.update(cxr_generator=1, xrv=1)
            images.add(image)
        calls.update(report_generator=1, chexbert=1)
        seen.add(cid)
    if (dict(calls) != row['simulated_calls'] or total != row['simulated_model_calls']
            or total > row['model_call_budget'] or len(seen) != row['observed_candidates']
            or len(images) != row['observed_images']
            or (row['selected_candidate_id'] is not None and row['selected_candidate_id'] not in seen)
            or any(row[k] is not False for k in ('clinical_acceptance', 'clinical_fault_confirmed', 'actual_regeneration_executed'))):
        raise ValueError('unchanged_budget_and_unqualified_source_trial_required')
    return list(seen)


def prepare(inputs):
    policy = inputs['replay', 'frozen_policy.json']
    if (policy['model_call_budgets'] != list(CAPS) or policy['random_seeds'] != list(SEEDS)
            or policy['secondary_scores_used_for_routing'] is not False
            or policy['routing_evidence'] != 'legacy_fourteen_raw_uncalibrated_xrv_chexbert_states'):
        raise ValueError('unchanged_legacy_budget_seed_profile_required')
    rows = inputs['complete', 'candidate_score_table.csv']
    candidates = {r['triple_candidate_id']: r for r in rows}
    if len(candidates) != 960 or len(rows) != 960:
        raise ValueError('full_unique_960_candidate_inventory_required')
    endpoints = inputs['complete', 'scores.json']['records']
    if len(endpoints) != 960 or {r['triple_candidate_id'] for r in endpoints} != set(candidates):
        raise ValueError('exact_complete_endpoint_inventory_required')
    for record in endpoints:
        row = candidates[record['triple_candidate_id']]
        if any(record[k] != row[k] for k in ('case_id', 'cxr_candidate_id', 'report_candidate_id',
                'ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256')):
            raise ValueError('complete_endpoint_artifact_lineage_required')
        if (record['calibrated'] is not False
                or candidate_readout(row)['biovil_raw_cosine'] != record['biovil_raw_cosine']):
            raise ValueError('unchanged_uncalibrated_endpoint_value_required')
    cases = defaultdict(list)
    for row in rows:
        cases[row['case_id']].append(row)
    if len(cases) != 80 or any(len(v) != 12 for v in cases.values()):
        raise ValueError('full_fixed_eighty_by_twelve_case_inventory_required')
    for rows in cases.values():
        if (len({r['ehr_sha256'] for r in rows}) != 1 or len({r['ehr_facts_sha256'] for r in rows}) != 1
                or len({r['cxr_candidate_id'] for r in rows}) != 3
                or len({r['report_model_id'] for r in rows}) != 4):
            raise ValueError('fixed_ehr_three_image_four_report_grid_required')
    old = inputs['replay', 'replay_outcomes.jsonl']
    expected = {(case, method, cap, seed if method == 'random' else None)
        for case in cases for cap in CAPS for method in OLD_METHODS
        for seed in (SEEDS if method == 'random' else (0,))}
    keys = [(r['case_id'], r['method'], r['model_call_budget'], r['random_seed']) for r in old]
    if len(old) != 3200 or len(set(keys)) != 3200 or set(keys) != expected:
        raise ValueError('all_original_method_budget_seed_trials_required')
    for row in old:
        validate_trace(row, candidates)
        if row['selected_candidate_id'] is not None and not candidate_readout(candidates[row['selected_candidate_id']])['artifact_gate_pass']:
            raise ValueError('old_selection_shared_artifact_gate_required')
    return old, candidates


def make_controls(old, candidates):
    controls = []
    for row in old:
        if row['method'] != 'random':
            continue
        visible = [{'candidate_id': a['observed_candidate_id'], 'artifact_gate_pass':
            artifact_gate(candidates[a['observed_candidate_id']])}
            for a in row['action_trace']]
        for seed in SEEDS:
            selected = uniform_choice(row['case_id'], row['model_call_budget'], row['random_seed'], seed, visible)
            controls.append({'case_id': row['case_id'], 'method': NEW_METHOD,
                'model_call_budget': row['model_call_budget'], 'acquisition_seed': row['random_seed'],
                'final_choice_seed': seed, 'selected_candidate_id': selected,
                'parent_acquisition_trial_sha256': digest(row),
                'selected_ehr_sha256': row['selected_ehr_sha256'],
                'input_ehr_assessment_scope': row['input_ehr_assessment_scope'],
                'simulated_calls': dict(row['simulated_calls']), 'simulated_model_calls': row['simulated_model_calls'],
                'observed_candidates': row['observed_candidates'], 'observed_images': row['observed_images'],
                'source_terminal_reason': row['terminal_reason'], 'clinical_scores_used_for_choice': False,
                'endpoint_used_for_choice': False, 'clinical_acceptance': False,
                'actual_regeneration_executed': False})
    return controls


def case_means(outcomes, candidates):
    grouped = defaultdict(list)
    for row in outcomes:
        cid = row['selected_candidate_id']
        readout = candidate_readout(candidates[cid]) if cid is not None else None
        grouped[row['case_id'], row['method'], row['model_call_budget']].append((row, readout))
    fields = list(candidate_readout(next(iter(candidates.values()))))
    result = []
    for (case, method, cap), trials in sorted(grouped.items()):
        expected = 25 if method == NEW_METHOD else 5 if method == 'random' else 1
        if len(trials) != expected or len({r['input_ehr_assessment_scope'] for r, _ in trials}) != 1:
            raise ValueError('all_seed_replicates_with_fixed_ehr_scope_required')
        out = {'case_id': case, 'method': method, 'model_call_budget': cap,
            'ehr_scope': trials[0][0]['input_ehr_assessment_scope'], 'seed_replicates': len(trials),
            'mean_simulated_calls': sum(r['simulated_model_calls'] for r, _ in trials) / len(trials),
            'selected_seed_replicates': sum(v is not None for _, v in trials)}
        for field in fields:
            values = [None if v is None else v[field] for _, v in trials]
            out[field] = sum(values) / len(values) if all(v is not None for v in values) else None
        result.append(out)
    return result


def comparisons(means):
    groups = defaultdict(list)
    for row in means:
        for scope in ('all', row['ehr_scope']):
            groups[scope, row['method'], row['model_call_budget']].append(row)
    fields = [k for k in means[0] if k not in ('case_id', 'method', 'model_call_budget', 'ehr_scope', 'seed_replicates')]
    results = []
    for (scope, method, cap), rows in sorted(groups.items()):
        result = {'ehr_scope': scope, 'method': method, 'model_call_budget': cap, 'fixed_ehr_cases': len(rows)}
        for field in fields:
            available = [r[field] for r in rows if r[field] is not None]
            result[field + '_mean'] = sum(available) / len(available) if available else None
            result[field + '_available_ehr_cases'] = len(available)
        results.append(result)
    index = {(r['case_id'], r['method'], r['model_call_budget']): r for r in means}
    paired = []
    cases = sorted({r['case_id'] for r in means})
    for cap in CAPS:
        for method in OLD_METHODS:
            for scope in ('all', 'explicit_fact_proxy', 'no_direct_comparable_ehr_facts'):
                pairs = [(index[case, method, cap], index[case, NEW_METHOD, cap]) for case in cases
                    if scope == 'all' or index[case, NEW_METHOD, cap]['ehr_scope'] == scope]
                available = [(a, b) for a, b in pairs if a['biovil_raw_cosine'] is not None and b['biovil_raw_cosine'] is not None]
                deltas = [a['biovil_raw_cosine'] - b['biovil_raw_cosine'] for a, b in available]
                paired.append({'ehr_scope': scope, 'model_call_budget': cap, 'baseline': method,
                    'fixed_ehr_cases': len(pairs), 'paired_available_biovil_cases': len(available),
                    'mean_biovil_delta_baseline_minus_score_free': sum(deltas) / len(deltas) if deltas else None,
                    'baseline_higher_ehr_cases': sum(d > 0 for d in deltas),
                    'tied_ehr_cases': sum(d == 0 for d in deltas), 'score_free_higher_ehr_cases': sum(d < 0 for d in deltas),
                    'mean_simulated_calls_delta_baseline_minus_score_free':
                        sum(a['mean_simulated_calls'] - b['mean_simulated_calls'] for a, b in pairs) / len(pairs) if pairs else None,
                    'same_acquisition_and_expenditure': method == 'random', 'clinical_accuracy': None})
    return results, paired


def csv_text(rows):
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def render(summary, comparison, paired):
    def show(v): return 'NA' if v is None else f'{v:.4f}'
    lines = ['# Score-free final-choice control / 不看临床分的最终随机选择', '',
        '80 fixed EHRs; 960 cached triples; 10,000 control seed trials are not new patients.',
        'Same historical acquisition/call charges; five acquisition and five final-choice seeds, all retained.',
        'No model/GPU/source-body reads, regeneration, fitting or changed old winners. DEVELOPMENT proxies only.', '',
        '| Method | Cap | Mean simulated calls | BioViL mean | Paired-available EHR denominator | CXR–report raw support / known | Raw opposition / known |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        if r['ehr_scope'] == 'all':
            lines.append(f"| {r['method']} | {r['model_call_budget']} | {show(r['mean_simulated_calls_mean'])} | "
                f"{show(r['biovil_raw_cosine_mean'])} | {r['biovil_raw_cosine_available_ehr_cases']}/{r['fixed_ehr_cases']} | "
                f"{show(r['cxr_report_support_over_known_mean'])} | {show(r['cxr_report_opposition_over_known_mean'])} |")
    lines += ['', '## Full-cap paired contrasts / 满预算同 EHR 对比', '',
        '| Old baseline | Paired EHRs | BioViL delta baseline − score-free | Baseline higher / tied / score-free higher | Call delta |',
        '|---|---:|---:|---|---:|']
    for r in paired:
        if r['ehr_scope'] == 'all' and r['model_call_budget'] == 30:
            lines.append(f"| {r['baseline']} | {r['paired_available_biovil_cases']}/{r['fixed_ehr_cases']} | "
                f"{show(r['mean_biovil_delta_baseline_minus_score_free'])} | {r['baseline_higher_ehr_cases']} / "
                f"{r['tied_ehr_cases']} / {r['score_free_higher_ehr_cases']} | {show(r['mean_simulated_calls_delta_baseline_minus_score_free'])} |")
    lines += ['', 'The historical `random` means random acquisition + scored final rerank, not this new uniform final choice.',
        'Only their selector-only contrast shares exactly the same acquisition and expenditure.',
        'Costs include historical scorers even when final choice does not use them; this is not a cost-optimal random pipeline.',
        'Raw fourteen-head proxies and global BioViL are not clinical truth. Missing EHR edges remain NA.',
        'All 80 EHRs remain; direct-fact/no-direct-fact groups are cached 8/72, not prompt tiers 15/65.',
        'All seed settings are averaged within EHR before cohort means. No best seed/budget, patient inflation or GPU-savings claim.',
        '本对照只测最终择优相对随机选择的作用；不能据此证明定位错误或 targeted repair 有效。', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    cpu_guard(); started = time.monotonic()
    inputs, sources = load()
    before = {k: sha256_file(p) for k, p in sources.items()}
    old, candidates = prepare(inputs)
    controls = make_controls(old, candidates)
    if len(controls) != 10000:
        raise ValueError('all_final_seed_trials_required')
    means = case_means([*old, *controls], candidates)
    comparison, paired = comparisons(means)
    summary = {'schema_version': SCHEMA, 'fixed_ehr_cases': 80, 'source_triples': 960,
        'original_trials': len(old), 'control_trials': len(controls), 'case_method_budget_rows': len(means),
        'new_model_calls': 0, 'actual_regeneration_executed': False, 'original_selection_changed': False,
        'clinical_accuracy': None, 'primary_metric_eligible': False, 'regeneration_authorized': False,
        'source_bodies_pixels_weights_or_mimic_inputs_read': False, 'actual_gpu_savings': None,
        'cohort_role': 'already_inspected_development_bank', 'all_final_seeds_retained': list(SEEDS),
        'selector_only_same_charged_acquisition': True,
        'elapsed_cpu_seconds_before_serialization': round(time.monotonic() - started, 6)}
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'control_outcomes.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in controls))
        write_private_text(temporary / 'case_means.csv', csv_text(means))
        write_private_text(temporary / 'method_comparison.csv', csv_text(comparison))
        write_private_text(temporary / 'paired_case_comparison.csv', csv_text(paired))
        write_private_text(temporary / 'RESULTS_CN_EN.md', render(summary, comparison, paired))
        if before != {k: sha256_file(p) for k, p in sources.items()}:
            raise ValueError('immutable_sources_changed_during_control')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA,
            'run_id': run_id, 'source_paths': {k: str(p.resolve()) for k, p in sources.items()},
            'source_sha256': before, 'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir()},
            'new_model_calls': 0, 'original_selection_changed': False, 'clinical_accuracy': None,
            'primary_metric_eligible': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=BASE / 'automatic_replays')
    args = parser.parse_args()
    try:
        path = execute(args.output_root, args.run_id)
    except Exception:
        print(json.dumps({'status': 'failed_cached_score_free_control', 'new_model_calls': 0}))
        raise SystemExit(1) from None
    print(json.dumps({'status': 'completed_cached_score_free_control', 'control_trials': 10000,
        'new_model_calls': 0, 'manifest_sha256': sha256_file(path / 'manifest.json')}))


if __name__ == '__main__':
    main()
