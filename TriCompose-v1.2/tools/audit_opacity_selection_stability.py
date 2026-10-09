#!/usr/bin/env python3
"""Post-hoc cached readouts of FROZEN choices, not rescoring/reselecting patients."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
import score_free_random_control as frozen
import score_cached_opacity_candidates as opacity
from contracts import (PROTECTED_ROOT, sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)

SCHEMA = 'tricompose-frozen-selection-opacity-stability-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
CONTROL = BASE / 'automatic_replays/score_free_random_12654973_001'
CONTROL_SHA = '771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124'
OPACITY = BASE / 'candidate_opacity_runs/opacity_pool240_12668204'
OPACITY_SHA = '778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94'
HEADS = ('exact_opacity_0_5', 'same_call_max_0_5_secondary')
METHODS = (*frozen.OLD_METHODS, frozen.NEW_METHOD)
REPLICATES = {m: 25 if m == frozen.NEW_METHOD else 5 if m == 'random' else 1 for m in METHODS}
VALUE_FIELDS = ('selected', 'scored_image', 'comparable', 'proxy_support', 'proxy_opposition',
    'not_comparable', 'image_positive', 'report_positive', 'simulated_calls')


def controls_checked(old, controls, candidates, *, full=True):
    parents = {frozen.digest(r): r for r in old if r['method'] == 'random'}
    if len(parents) != sum(r['method'] == 'random' for r in old):
        raise ValueError('unique_frozen_random_parent_required')
    seen = set()
    for row in controls:
        key = (row['case_id'], row['model_call_budget'], row['acquisition_seed'], row['final_choice_seed'])
        if key in seen or row['method'] != frozen.NEW_METHOD or row['final_choice_seed'] not in frozen.SEEDS:
            raise ValueError('unique_all_frozen_control_seeds_required')
        seen.add(key); parent = parents.get(row['parent_acquisition_trial_sha256'])
        if parent is None or any(row[k] != parent[k] for k in ('case_id', 'model_call_budget',
                'selected_ehr_sha256', 'input_ehr_assessment_scope', 'simulated_calls',
                'simulated_model_calls', 'observed_candidates', 'observed_images')) or \
                row['acquisition_seed'] != parent['random_seed'] or row['source_terminal_reason'] != parent['terminal_reason']:
            raise ValueError('unchanged_same_acquisition_and_expenditure_required')
        if any(row[k] is not False for k in ('clinical_scores_used_for_choice', 'endpoint_used_for_choice',
                'clinical_acceptance', 'actual_regeneration_executed')):
            raise ValueError('unqualified_score_free_control_required')
        chosen = row['selected_candidate_id']
        observed = {r['observed_candidate_id'] for r in parent['action_trace']}
        if chosen is not None and (chosen not in observed or chosen not in candidates or
                candidates[chosen]['case_id'] != row['case_id'] or
                candidates[chosen]['ehr_sha256'] != row['selected_ehr_sha256'] or
                not frozen.artifact_gate(candidates[chosen])):
            raise ValueError('frozen_observed_only_control_choice_required')
    if full:
        expected = {(r['case_id'], r['model_call_budget'], r['random_seed'], seed)
                    for r in parents.values() for seed in frozen.SEEDS}
        if len(controls) != 10000 or seen != expected:
            raise ValueError('all_10000_existing_controls_required')


def validate_table(original, annotated):
    index = {r['triple_candidate_id']: r for r in annotated}
    if len(index) != len(annotated) or set(index) != set(original):
        raise ValueError('exact_original_candidate_inventory_required')
    for cid, source in original.items():
        row = index[cid]
        if any(row[k] != source[k] for k in source):
            raise ValueError('all_original_csv_cells_required')
        if row['opacity_cached_ehr_state'] != 'unknown' or \
                row['opacity_cached_report_state'] not in opacity.STATES or \
                row['opacity_exact_state_0_5'] not in opacity.STATES or \
                row['opacity_same_call_max_state_0_5'] not in opacity.STATES or \
                row['opacity_clinical_accuracy'] != '' or row['opacity_confirmed_faulty_modality'] != '' or \
                row['opacity_selector_used'] != 'False':
            raise ValueError('unchanged_unqualified_opacity_scope_required')
        if row['opacity_scoring_status'] == 'scored':
            exact, infiltration = float(row['opacity_exact_score']), float(row['opacity_infiltration_score'])
            if opacity.state(exact) != row['opacity_exact_state_0_5'] or \
                    opacity.state(max(exact, infiltration)) != row['opacity_same_call_max_state_0_5']:
                raise ValueError('fixed_threshold_raw_score_binding_required')
            opacity.state(infiltration)
        elif row['opacity_scoring_status'] == 'failed_without_replacement':
            if row['opacity_exact_score'] != '' or row['opacity_infiltration_score'] != '' or \
                    row['opacity_exact_state_0_5'] != 'unknown' or row['opacity_same_call_max_state_0_5'] != 'unknown':
                raise ValueError('failed_evidence_retained_unknown_required')
        else:
            raise ValueError('known_image_outcome_required')
    return index


def joined_readouts(trials, candidates):
    """Never choose an ID: attach both definitions to the same frozen chosen ID."""
    output = []
    for trial in trials:
        cid = trial['selected_candidate_id']; row = candidates.get(cid) if cid is not None else None
        if cid is not None and (row is None or row['case_id'] != trial['case_id'] or
                row['ehr_sha256'] != trial['selected_ehr_sha256']):
            raise ValueError('fixed_chosen_candidate_lineage_required')
        trial_hash = frozen.digest(trial)
        for head, field in zip(HEADS, ('opacity_exact_state_0_5', 'opacity_same_call_max_state_0_5'), strict=True):
            image = row[field] if row else 'unknown'
            report = row['opacity_cached_report_state'] if row else 'unknown'
            value = opacity.relation(image, report)
            output.append({'case_id': trial['case_id'], 'method': trial['method'],
                'model_call_budget': trial['model_call_budget'], 'head_definition': head,
                'acquisition_seed': trial.get('acquisition_seed', trial.get('random_seed')),
                'final_choice_seed': trial.get('final_choice_seed'),
                'frozen_trial_sha256': trial_hash, 'selected_candidate_id': cid,
                'selected_ehr_sha256': trial['selected_ehr_sha256'],
                'selected_cxr_sha256': row['cxr_sha256'] if row else None,
                'selected_report_sha256': row['report_sha256'] if row else None,
                'image_state': image, 'report_state': report, 'relation': value,
                'selected': int(row is not None),
                'scored_image': int(row is not None and row['opacity_scoring_status'] == 'scored'),
                'comparable': int(value != 'not_comparable'), 'proxy_support': int(value == 'proxy_support'),
                'proxy_opposition': int(value == 'proxy_opposition'), 'not_comparable': int(value == 'not_comparable'),
                'image_positive': int(image == 'positive'), 'report_positive': int(report == 'positive'),
                'simulated_calls': trial['simulated_model_calls'],
                'clinical_accuracy': None, 'ehr_cxr_clinical_score': None, 'ehr_report_clinical_score': None})
    return output


def case_means(readouts, cases, caps=frozen.CAPS):
    groups = defaultdict(list)
    for row in readouts:
        groups[row['case_id'], row['method'], row['model_call_budget'], row['head_definition']].append(row)
    expected = {(case, method, cap, head) for case in cases for method in METHODS for cap in caps for head in HEADS}
    if set(groups) != expected:
        raise ValueError('all_cases_methods_caps_heads_required')
    output = []
    for (case, method, cap, head), rows in sorted(groups.items()):
        keys = {(r['acquisition_seed'], r['final_choice_seed']) for r in rows}
        if len(rows) != REPLICATES[method] or len(keys) != len(rows) or \
                len({r['selected_ehr_sha256'] for r in rows}) != 1:
            raise ValueError('all_unique_seed_replicates_and_fixed_ehr_required')
        out = {'case_id': case, 'method': method, 'model_call_budget': cap, 'head_definition': head,
            'selected_ehr_sha256': rows[0]['selected_ehr_sha256'], 'seed_replicates': len(rows)}
        for field in VALUE_FIELDS:
            if any(not isinstance(r[field], (int, float)) or isinstance(r[field], bool) or
                   not math.isfinite(r[field]) or r[field] < 0 for r in rows):
                raise ValueError('finite_nonnegative_diagnostic_counts_required')
            out[field] = sum(r[field] for r in rows) / len(rows)
        output.append(out)
    return output


def comparisons(means, cases, caps=frozen.CAPS):
    groups = defaultdict(list); index = {}
    for row in means:
        groups[row['method'], row['model_call_budget'], row['head_definition']].append(row)
        key = row['case_id'], row['method'], row['model_call_budget'], row['head_definition']
        if key in index:
            raise ValueError('unique_case_mean_required')
        index[key] = row
    tables = []
    for (method, cap, head), rows in sorted(groups.items()):
        if len(rows) != len(cases) or {r['case_id'] for r in rows} != set(cases):
            raise ValueError('same_all_case_denominator_required')
        out = {'method': method, 'model_call_budget': cap, 'head_definition': head, 'fixed_ehr_cases': len(rows)}
        for field in VALUE_FIELDS:
            out['mean_' + field] = sum(r[field] for r in rows) / len(rows)
        out['conditional_proxy_agreement'] = out['mean_proxy_support'] / out['mean_comparable'] if out['mean_comparable'] else None
        out['ehr_cases_with_any_comparison'] = sum(r['comparable'] > 0 for r in rows)
        out['clinical_accuracy'] = None; tables.append(out)
    pairs = []
    for cap in caps:
        for head in HEADS:
            contrasts = [(m, 'fixed') for m in METHODS if m != 'fixed'] + [('random', frozen.NEW_METHOD)]
            for method, reference in contrasts:
                values = [(index[c, method, cap, head], index[c, reference, cap, head]) for c in sorted(cases)]
                if any(a['selected_ehr_sha256'] != b['selected_ehr_sha256'] for a, b in values):
                    raise ValueError('paired_fixed_ehr_required')
                out = {'method': method, 'reference': reference, 'model_call_budget': cap, 'head_definition': head,
                    'fixed_ehr_cases': len(values), 'same_acquisition_and_expenditure': method == 'random' and reference == frozen.NEW_METHOD}
                for field in ('proxy_support', 'proxy_opposition', 'comparable', 'simulated_calls'):
                    differences = [a[field] - b[field] for a, b in values]
                    out['mean_delta_' + field] = sum(differences) / len(differences)
                    out[field + '_higher_ehr_cases'] = sum(d > 1e-12 for d in differences)
                    out[field + '_lower_ehr_cases'] = sum(d < -1e-12 for d in differences)
                    out[field + '_tied_ehr_cases'] = sum(abs(d) <= 1e-12 for d in differences)
                out['both_have_any_comparison_ehr_cases'] = sum(a['comparable'] > 0 and b['comparable'] > 0 for a, b in values)
                out['opposition_reduction_with_lost_all_comparison_ehr_cases'] = sum(
                    a['proxy_opposition'] < b['proxy_opposition'] and a['comparable'] == 0 and b['comparable'] > 0 for a, b in values)
                out['clinical_repair_success'] = None
                if out['same_acquisition_and_expenditure'] and any(abs(a['simulated_calls'] - b['simulated_calls']) > 1e-12 for a, b in values):
                    raise ValueError('selector_only_contrast_expenditure_must_match')
                pairs.append(out)
    return tables, pairs


def render(summary, table):
    def percent(x): return 'NA' if x is None else f'{100*x:.2f}%'
    lines = ['# Frozen selection: opacity audit / 已冻结选优的阴影诊断', '',
        'Post-hoc DEVELOPMENT proxies, not clinical accuracy or a new winner. All 80 EHRs retained.',
        'Old random is randomized acquisition + scored final selection; the separate uniform final control is included.',
        'Every historical choice/cost stays unchanged. Repeated seed settings are averaged within EHR first.', '',
        '| Head | Method | Cap | Simulated calls | Support / all EHRs | Opposition / all EHRs | Comparison coverage | Conditional proxy agreement |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['head_definition']} | {row['method']} | {row['model_call_budget']} | "
            f"{row['mean_simulated_calls']:.3f} | {percent(row['mean_proxy_support'])} | "
            f"{percent(row['mean_proxy_opposition'])} | {percent(row['mean_comparable'])} | {percent(row['conditional_proxy_agreement'])} |")
    lines += ['', '## Boundaries / 解释边界', '',
        '- Unknown/uncertain remains not comparable, never a negative or repair. Zero coverage means NA agreement.',
        '- Opacity EHR evidence is unavailable for all 80 anchors; this is not a full-triple clinical score.',
        '- Same checkpoint/extractor, different head definition: not an independent held-out evaluator.',
        '- No best head/budget/seed, retraining, reranking, new acquisition, altered winner or clinical fault judgment.',
        '- Same cap is not equal expenditure. Only scored-vs-uniform final choice shares acquisition/expenditure.',
        '- The 240 already completed diagnostic XRV calls are extra actual work, not historical GPU savings.',
        '- 13,200 frozen trials and 26,400 head readouts are correlated choices over 80 EHRs, not new patients.',
        '- Pair contrasts and lost-comparison counts are descriptive; they do not validate regeneration.', '',
        '## Execution summary', '', '```json', json.dumps(summary, indent=2, sort_keys=True), '```', '']
    return '\n'.join(lines)


def execute(args):
    frozen.cpu_guard(); started = time.monotonic()
    sources = opacity.source_pins()
    for path in (Path(__file__), ROOT / 'tests/test_opacity_selection_stability.py',
                 ROOT.parent / 'docs/opacity_selection_stability_protocol.md', Path(frozen.__file__)):
        sources[str(path)] = sha256_file(path)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        plan = {'schema_version': SCHEMA + '-plan', 'methods': list(METHODS), 'caps': list(frozen.CAPS),
            'seeds': list(frozen.SEEDS), 'heads': list(HEADS), 'fixed_ehr_cases': 80,
            'old_choices_are_frozen': True, 'new_selector_invocations': 0,
            'seed_weighting': 'all_replicates_within_ehr_then_all_80_ehrs',
            'opacity_manifest_sha256': OPACITY_SHA, 'control_manifest_sha256': CONTROL_SHA,
            'clinical_accuracy': None, 'primary_metric_eligible': False,
            'cohort_role': 'post_hoc_already_inspected_development', 'threshold_fitting': False}
        path = write_private_json(temporary / 'audit_plan.json', plan)
        with path.open('rb') as handle:
            os.fsync(handle.fileno())
        inputs, old_sources = frozen.load()
        sources.update({str(p): sha256_file(p) for p in old_sources.values()})
        old, original = frozen.prepare(inputs)
        cm = opacity.metadata(CONTROL / 'manifest.json', sources, CONTROL_SHA)
        if cm['schema_version'] != frozen.SCHEMA or cm['original_selection_changed'] is not False:
            raise ValueError('frozen_control_manifest_required')
        controls = opacity.metadata(CONTROL / 'control_outcomes.jsonl', sources,
            cm['artifacts']['control_outcomes.jsonl']['sha256'], cap=32*1024**2)
        controls_checked(old, controls, original)
        om = opacity.metadata(OPACITY / 'manifest.json', sources, OPACITY_SHA)
        if om['schema_version'] != opacity.SCHEMA + '-manifest' or om['original_selection_changed'] is not False:
            raise ValueError('completed_opacity_sidecar_required')
        opacity.verify_pins(om['sources']); sources.update(om['sources'])
        rows = opacity.metadata(OPACITY / 'candidate_score_table.csv', sources, om['artifacts']['candidate_score_table.csv'])
        if rows != [{**r, **{k: new[k] for k in new if k not in r}} for r, new in zip(inputs['complete', 'candidate_score_table.csv'], rows, strict=True)]:
            raise ValueError('original_source_row_order_required')
        candidates = validate_table(original, rows)
        readouts = joined_readouts([*old, *controls], candidates)
        cases = {r['case_id'] for r in original.values()}
        means = case_means(readouts, cases); table, paired = comparisons(means, cases)
        if len(readouts) != 26400 or len(means) != 4000 or len(table) != 50 or len(paired) != 50:
            raise ValueError('entire_frozen_diagnostic_grid_required')
        summary = {'schema_version': SCHEMA, 'status': 'completed_frozen_choice_diagnostic_unverified',
            'fixed_ehr_cases': 80, 'source_image_slots': 240, 'source_candidate_rows': 960,
            'old_trials': len(old), 'existing_uniform_final_trials': len(controls),
            'head_readouts': len(readouts), 'case_method_cap_head_means': len(means),
            'method_cap_head_comparisons': len(table), 'paired_contrasts': len(paired),
            'new_model_calls': 0, 'new_selector_invocations': 0, 'new_slurm_submissions': 0,
            'source_bodies_or_pixels_opened': False, 'historical_choices_and_costs_changed': False,
            'clinical_accuracy': None, 'clinical_repair_success': None, 'actual_gpu_savings': None,
            'primary_metric_eligible': False, 'independent_clinical_endpoint': False,
            'existing_cpu_job_id': os.environ['SLURM_JOB_ID'],
            'elapsed_cpu_seconds_before_serialization': round(time.monotonic() - started, 6)}
        write_private_text(temporary / 'trial_readouts.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in readouts))
        write_private_text(temporary / 'case_means.csv', opacity.csv_text(means))
        write_private_text(temporary / 'method_comparison.csv', opacity.csv_text(table))
        write_private_text(temporary / 'paired_case_comparison.csv', opacity.csv_text(paired))
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md', render(summary, table))
        opacity.verify_pins(sources)
        names = ('audit_plan.json', 'trial_readouts.jsonl', 'case_means.csv', 'method_comparison.csv',
                 'paired_case_comparison.csv', 'summary.json', 'RESULTS_CN_EN.md')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA, 'sources': sources,
            'artifacts': {n: sha256_file(temporary / n) for n in names},
            'historical_choices_changed': False, 'new_model_calls': 0, 'new_selector_invocations': 0,
            'primary_metric_eligible': False, 'regeneration_authorized': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary); raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=BASE / 'opacity_selection_audits')
    parser.add_argument('--run-id', required=True)
    try:
        target, summary = execute(parser.parse_args(argv))
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__})); return 2
    print(json.dumps({'status': summary['status'], 'fixed_ehr_cases': 80, 'head_readouts': summary['head_readouts'],
        'new_model_calls': 0, 'manifest_sha256': sha256_file(target / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
