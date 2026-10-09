"""Independent cached arithmetic and frozen-choice audit, not clinical truth."""
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

from contracts import new_atomic_run, commit_atomic_run, discard_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'frozen_selection_mask_comparisons/all_caps_12784259_001'
EXPECTED = 'ef24a684fd5e6903dd894f9d60caa0afc9f6ca0fd7056d168b43f21e899b3aae'
OLD = BASE / 'opacity_selection_audits/selection_opacity_12666569_001'
MASK = BASE / 'candidate_image_mask_overlays/pool960_12784259_001'
POLICIES = ('xrv_exact_0_5', 'biovil_fixed_mean', 'agree_fixed_mean', 'agree_all_templates')
METHODS = ('fixed', 'random', 'static_rerank', 'targeted_heuristic', 'random_acquisition_score_free_final')
CAPS = (4, 8, 12, 20, 30)
FIELDS = ('selected', 'accepted_image', 'comparable', 'proxy_support', 'proxy_opposition', 'not_comparable', 'simulated_calls')
BINDINGS = ('case_id', 'method', 'model_call_budget', 'acquisition_seed', 'final_choice_seed',
    'frozen_trial_sha256', 'selected_candidate_id', 'selected_ehr_sha256', 'selected_cxr_sha256',
    'selected_report_sha256', 'simulated_calls')
CONTRASTS = (('xrv_exact_0_5', 'agree_fixed_mean'), ('biovil_fixed_mean', 'agree_fixed_mean'),
             ('agree_fixed_mean', 'agree_all_templates'))


def close(actual, expected):
    if expected is None:
        require(actual in (None, ''), 'unavailable_metric_cannot_be_zero')
    else:
        require(actual not in (None, '') and type(actual) in (str, int, float)
                and math.isfinite(float(actual))
                and math.isclose(float(actual), expected, abs_tol=1e-12, rel_tol=0), 'independent_numeric_replay_mismatch')


def trial_counts(t, row):
    if row is not None:
        require(t['selected_candidate_id'] == row['triple_candidate_id'] and t['case_id'] == row['case_id']
                and all(t[a] == row[b] for a, b in (('selected_ehr_sha256', 'ehr_sha256'),
                    ('selected_cxr_sha256', 'cxr_sha256'), ('selected_report_sha256', 'report_sha256'))),
                'independent_frozen_choice_lineage_mismatch')
    else:
        require(t['selected_candidate_id'] is None, 'missing_candidate_cannot_be_substituted')
    result = {}
    report = row['opacity_cached_report_state'] if row else 'unknown'
    require(report in ('positive', 'negative', 'uncertain', 'unknown'), 'original_four_state_report_required')
    for p in POLICIES:
        state = (row['image_mask_' + p + '_state'] or None) if row else None
        require(state in (None, 'positive', 'negative'), 'explicit_or_missing_mask_required')
        comparable = state is not None and report in ('positive', 'negative')
        support = comparable and state == report
        opposition = comparable and state != report
        result[p] = dict(zip(FIELDS, (int(row is not None), int(state is not None), int(comparable),
            int(support), int(opposition), int(not comparable), t['simulated_calls'])))
    return result


def csv_read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED, 'sealed_completed_comparison_required')
    m = json.loads((RUN / 'manifest.json').read_text())
    for name, value in m['artifacts'].items():
        require(sha256(RUN / name) == value, 'comparison_artifact_changed')
    for name, value in m['pins'].items():
        path = WORKSPACE / name
        require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == value, 'consumed_comparison_source_changed')
    paths = [Path(__file__).resolve(), WORKSPACE / 'TriCompose-v1.2/tests/test_frozen_selection_image_masks_audit.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    temporary, target = new_atomic_run(BASE / 'frozen_selection_mask_audits', 'numeric_12784259_001')
    try:
        write_json(temporary / 'frozen_plan.json', {'schema_version': 'frozen-selection-mask-independent-audit-plan-v1',
            'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
            'production_mask_and_comparison_functions_imported': False, 'clinical_qualified': False,
            'raw_body_pixel_or_weight_asset_read': False})
        with (OLD / 'trial_readouts.jsonl').open() as stream:
            old = [r for r in map(json.loads, stream) if r['head_definition'] == 'exact_opacity_0_5']
        with (RUN / 'trial_mask_readouts.jsonl').open() as stream:
            readouts = list(map(json.loads, stream))
        candidates = {r['triple_candidate_id']: r for r in csv_read(MASK / 'candidate_evidence_table.csv')}
        require(len(old) == len(readouts) == 13200 and len(candidates) == 960, 'all_original_frozen_trial_inventory_required')
        groups = defaultdict(list)
        scalar_checks = 0
        for t, out in zip(old, readouts):
            require(set(out) == {*BINDINGS, 'masks'} and all(out[k] == t[k] for k in BINDINGS),
                    'original_choice_seed_cost_order_or_trial_hash_changed')
            row = candidates.get(t['selected_candidate_id'])
            expected = trial_counts(t, row)
            require(out['masks'] == expected, 'independent_trial_mask_counters_mismatch')
            for p, counts in expected.items():
                groups[t['case_id'], t['method'], t['model_call_budget'], p].append(counts)
                scalar_checks += len(counts)
        cases = sorted({r['case_id'] for r in candidates.values()})
        require(len(cases) == 80 and len(groups) == 8000, 'all_ehr_method_budget_mask_groups_required')
        means = {}
        for key, records in groups.items():
            require(len(records) == (25 if key[1] == METHODS[-1] else 5 if key[1] == 'random' else 1),
                    'within_ehr_seed_denominator_mismatch')
            means[key] = {f: sum(r[f] for r in records) / len(records) for f in FIELDS}
        case_rows = csv_read(RUN / 'case_means.csv')
        seen = set()
        for r in case_rows:
            key = r['case_id'], r['method'], int(r['model_call_budget']), r['mask']
            require(key in means and key not in seen, 'unique_full_case_mean_inventory_required')
            seen.add(key)
            for f, value in means[key].items():
                close(r[f], value)
        require(seen == set(means), 'all_case_means_required')
        aggregates = {(method, cap, p): {f: sum(means[c, method, cap, p][f] for c in cases) / len(cases) for f in FIELDS}
            for method in METHODS for cap in CAPS for p in POLICIES}
        table = csv_read(RUN / 'method_comparison.csv')
        seen = set()
        for r in table:
            key = r['method'], int(r['model_call_budget']), r['mask']
            require(key in aggregates and key not in seen and r['fixed_ehr_cases'] == '80'
                    and r['clinical_accuracy'] == r['clinical_repair_success'] == '', 'aggregate_full_denominator_scope_required')
            seen.add(key)
            a = aggregates[key]
            for f in FIELDS:
                close(r['mean_' + f], a[f])
            close(r['conditional_proxy_agreement'], a['proxy_support'] / a['comparable'] if a['comparable'] else None)
            close(r['ehr_cases_with_any_comparison'], sum(means[c, *key]['comparable'] > 0 for c in cases))
        require(seen == set(aggregates), 'all_100_method_budget_mask_aggregates_required')
        contrasts = [(m, 'fixed') for m in METHODS[1:]] + [('random', METHODS[-1])]
        expected_pairs = {(a, b, cap, p) for a, b in contrasts for cap in CAPS for p in POLICIES}
        seen = set()
        for r in csv_read(RUN / 'paired_method_comparison.csv'):
            method, reference, cap, p = r['method'], r['reference'], int(r['model_call_budget']), r['mask']
            key = method, reference, cap, p
            require(key in expected_pairs and key not in seen, 'full_unique_paired_method_inventory_required')
            seen.add(key)
            a, b = aggregates[method, cap, p], aggregates[reference, cap, p]
            for f in FIELDS:
                close(r['mean_delta_' + f], a[f] - b[f])
            same = (method, reference) == ('random', METHODS[-1])
            require(r['same_acquisition_and_expenditure'] == str(same), 'same_acquisition_flag_mismatch')
            close(r['opposition_reduction_with_lost_all_comparison_ehr_cases'], sum(
                means[c, method, cap, p]['proxy_opposition'] < means[c, reference, cap, p]['proxy_opposition']
                and means[c, method, cap, p]['comparable'] == 0 and means[c, reference, cap, p]['comparable'] > 0 for c in cases))
        require(seen == expected_pairs, 'all_100_paired_method_rows_required')
        expected_losses = {(m, cap, a, b) for m in METHODS for cap in CAPS for a, b in CONTRASTS}
        seen = set()
        for r in csv_read(RUN / 'mask_withholding.csv'):
            method, cap, left, right = r['method'], int(r['model_call_budget']), r['left_mask'], r['right_mask']
            key = method, cap, left, right
            require(key in expected_losses and key not in seen and r['clinical_errors_repaired'] == '',
                    'full_withholding_not_clinical_repair_inventory_required')
            seen.add(key)
            for f in ('accepted_image', 'proxy_support', 'proxy_opposition', 'comparable'):
                value = aggregates[method, cap, left][f] - aggregates[method, cap, right][f]
                require(value >= -1e-12, 'withholding_cannot_add_comparable_support_required')
                close(r['mean_withheld_' + f], value)
        require(seen == expected_losses, 'all_75_nested_mask_withholding_rows_required')
        summary = json.loads((RUN / 'summary.json').read_text())
        require(summary['clinical_qualified'] is False and summary['new_model_calls'] == 0
                and summary['new_selector_invocations'] == 0 and summary['historical_choices_changed'] is False
                and summary['confidence_intervals'] is None, 'unchanged_descriptive_not_clinical_results_required')
        result = {'schema_version': 'frozen-selection-mask-independent-audit-v1', 'status': 'passed',
            'completed_run_manifest_sha256': EXPECTED, 'frozen_trial_bindings_verified': len(old),
            'trial_mask_counter_cells_verified': scalar_checks, 'case_mean_rows_verified': len(means),
            'method_rows_verified': len(aggregates), 'paired_method_rows_verified': len(expected_pairs),
            'nested_mask_rows_verified': len(expected_losses), 'clinical_qualified': False,
            'production_mask_and_comparison_functions_imported': False, 'selection_changed': False,
            'regeneration_authorized': False, 'raw_body_pixel_or_weight_asset_read': False,
            'runtime_seconds': time.monotonic() - started}
        require(all(sha256(WORKSPACE / n) == v for n, v in pins.items()) and sha256(RUN / 'manifest.json') == EXPECTED,
                'audit_sources_changed')
        write_json(temporary / 'audit.json', result)
        write_json(temporary / 'manifest.json', {'schema_version': 'frozen-selection-mask-audit-receipt-v1',
            'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
            'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())}, 'clinical_qualified': False})
        for p in [temporary, *temporary.rglob('*')]:
            require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                    and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'private_numeric_audit_required')
        commit_atomic_run(temporary, target)
        return target, result
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_numeric_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS')
                and not os.environ.get('SLURM_STEP_GPUS'), 'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_frozen_mask_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_frozen_mask_numeric_audit_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
