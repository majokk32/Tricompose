"""Cached uncertainty-aware risk/coverage and report-clustered interval audit.

Uses sealed four-state counters only. No report/source body, original patient
key, image, native graph, model, selector or network service is opened.
"""
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time

from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json
from tricompose_v12.manual_reader_risk_coverage import (
    CONTRASTS, FIELDS, VIEWS, aggregate_reports, analyze,
)

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
MASKS = BASE / 'manual_three_reader_runs/seven_masks_12766754_001'
MASKS_SHA = 'de2439533772e30a68714a7eef6c18b3970c52afee03360aa8e50b45be2fef78'
REF = BASE / 'manual_literal_reader_runs/manual100_12766754_001'
REF_SHA = '030ac27c56217db264bb09c93cba6a3ec2ae7b5322c17323a918fd89ded01a5f'
AUDIT = BASE / 'manual_three_reader_audits/numeric_12766754_001'
AUDIT_SHA = 'a5575d29cb9d996e321a2c4de0aca7bda242eeec45ca1d3c3def19d0c3f91ae5'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'


def csv_write(path, rows):
    require(rows, 'nonempty_fixed_output_required')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)


def tables(result):
    short, long, pairs = [], [], []
    for row in result['rows']:
        m = row['metrics']
        risk = m['conditional_literal_assertion_error']
        limits = risk['percentile_interval']
        short.append({'readout': row['readout'], 'domain': row['domain'], 'policy': row['policy'],
            'reports': row['reports'], 'correct_known': m['known_recovery']['numerator'],
            'known_reference_checks': m['known_recovery']['denominator'],
            'known_recovery': m['known_recovery']['estimate'],
            'correct_positive': row['correct_positive'], 'positive_support': row['positive'],
            'correct_negative': row['correct_negative'], 'negative_support': row['negative'],
            'polarity_flips': row['flips'], 'determinate_on_uncertain': row['commit_uncertain'],
            'uncertain_support': row['uncertain'], 'literal_state_error_numerator': risk['numerator'],
            'adjudicable_determinate_proposals': risk['denominator'],
            'conditional_literal_assertion_error': risk['estimate'],
            'risk_interval_lower': limits[0] if limits else None,
            'risk_interval_upper': limits[1] if limits else None,
            'risk_valid_draws': risk['valid_draws'], 'risk_zero_denominator_draws': risk['zero_denominator_draws'],
            'all_determinate_proposals': row['accepted'], 'attempted_checks': row['attempted'],
            'proposal_coverage': m['proposal_coverage']['estimate'], 'unavailable_checks': row['unavailable'],
            'determinate_on_literal_unknown': row['commit_unknown'], 'literal_unknown_support': row['unknown'],
            'clinical_qualified': False, 'patient_cluster_verified': False})
        for name, value in m.items():
            limits = value['percentile_interval']
            long.append({'readout': row['readout'], 'domain': row['domain'], 'policy': row['policy'],
                'metric': name, 'numerator': value['numerator'], 'denominator': value['denominator'],
                'estimate': value['estimate'], 'lower': limits[0] if limits else None,
                'upper': limits[1] if limits else None, 'valid_draws': value['valid_draws'],
                'zero_denominator_draws': value['zero_denominator_draws'], 'clinical_qualified': False})
    for row in result['paired_contrasts']:
        for metric, value in row['differences'].items():
            limits = value['percentile_interval']
            pairs.append({'readout': row['readout'], 'domain': row['domain'],
                'left': row['left'], 'right': row['right'], 'metric': metric,
                'right_minus_left': value['right_minus_left'], 'lower': limits[0] if limits else None,
                'upper': limits[1] if limits else None, 'valid_draws': value['valid_draws'],
                'zero_denominator_draws': value['zero_denominator_draws'],
                'shared_paired_draws': True, 'clinical_qualified': False})
    return short, long, pairs


def execute():
    started = time.monotonic()
    for root, expected in ((MASKS, MASKS_SHA), (REF, REF_SHA), (AUDIT, AUDIT_SHA)):
        require(sha256(root / 'manifest.json') == expected, 'sealed_completed_parent_required')
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, value in manifest['artifacts'].items():
            require(sha256(root / name) == value, 'completed_parent_artifact_changed')
    require(sha256(BANK) == BANK_SHA, 'original_bank_unchanged_required')
    audit = json.loads((AUDIT / 'audit.json').read_text())
    require(audit['status'] == 'passed' and audit['score_table_rows_checked'] == 42
            and audit['completed_analysis_manifest_sha256'] == MASKS_SHA, 'independent_count_audit_required')
    sources = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_reader_risk_coverage.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_three_reader_agreement.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_manual_reader_risk_coverage.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_manual_reader_risk_coverage_worker.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        MASKS / 'manifest.json', MASKS / 'mask_details.json', MASKS / 'evaluation.json',
        MASKS / 'domain_evaluation.json', REF / 'manifest.json', REF / 'reference_projection.json',
        AUDIT / 'manifest.json', AUDIT / 'audit.json', BANK]
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in sources}
    temporary, target = new_atomic_run(BASE / 'manual_reader_risk_coverage', 'clustered_12784259_001')
    write_json(temporary / 'frozen_plan.json', {'schema_version': 'manual-reader-clustered-diagnostic-plan-v1',
        'pins': pins, 'seed': 0, 'bootstrap_repetitions': 2000, 'confidence': 0.95,
        'stratified_distinct_report_clusters_not_verified_patients': True,
        'primary_readout': 'native_labels', 'secondary_readout': 'mention_conflict_view',
        'fixed_contrasts': CONTRASTS, 'all_existing_seven_masks_retained': True,
        'error_numerator': 'known_polarity_flips_plus_determinate_on_uncertain_human_literal_reference',
        'error_denominator': 'determinate_proposals_on_positive_negative_or_uncertain_literal_reference',
        'literal_unknown_excluded_from_error_not_clinically_negative': True,
        'failed_reader_kept_in_recovery_and_coverage_denominators': True,
        'post_hoc_development_diagnostic': True, 'new_test_set_or_clinical_calibration': False,
        'intervals_do_not_resolve_checkpoint_overlap_scope_or_population_dependence': True,
        'percentile_interval_on_zero_observed_errors_is_not_a_guarantee': True,
        'best_policy_selected': False, 'selection_changed': False, 'regeneration_authorized': False})
    refs = json.loads((REF / 'reference_projection.json').read_text())['records']
    details = json.loads((MASKS / 'mask_details.json').read_text())['records']
    require(len(refs) == 100 and len(details) == 5600, 'same_complete_hundred_report_inventory_required')
    result = analyze(refs, details)
    require(result == analyze(refs, details), 'deterministic_cached_replay_required')
    old = json.loads((MASKS / 'evaluation.json').read_text())
    domains = json.loads((MASKS / 'domain_evaluation.json').read_text())
    for row in result['rows']:
        root = old[row['readout']] if row['domain'] == 'all' else domains[row['readout']][row['domain']]
        m = root['masks'][row['policy']]
        known = row['positive'] + row['negative']
        correct = row['correct_positive'] + row['correct_negative']
        require((correct, known, row['flips'], row['accepted_known'], row['commit_uncertain'], row['commit_unknown'],
                 row['accepted'], row['unavailable']) == (
                m['correct_known_proposals'], m['positive_negative_reference_checks'], m['hard_positive_negative_flips'],
                m['accepted_known_proposals'], m['determinate_on_uncertain_reference'],
                m['determinate_on_unknown_literal_reference'], m['accepted_determinate_proposals'],
                m['unavailable_reader_checks']), 'historical_point_counts_must_remain_identical')
        risk = row['metrics']['conditional_literal_assertion_error']
        require(risk['numerator'] == row['flips'] + row['commit_uncertain']
                and risk['denominator'] == row['accepted'] - row['commit_unknown'],
                'independent_uncertainty_aware_fraction_mismatch')
    short, long, pairs = tables(result)
    require((len(short), len(long), len(pairs)) == (42, 378, 162), 'all_fixed_metrics_domains_and_contrasts_required')
    refs_sorted, policies, counters = aggregate_reports(refs, details)
    write_json(temporary / 'report_cluster_counters.json', {'fields': FIELDS,
        'records': [{'readout': view, 'policy': policy, 'report_id': ref['report_id'],
                     'source_sha256': ref['source_sha256'], 'domain': ref['source_domain'],
                     'counters': counters[v, p, i].tolist()}
                    for v, view in enumerate(VIEWS) for p, policy in enumerate(policies)
                    for i, ref in enumerate(refs_sorted)]})
    write_json(temporary / 'evaluation.json', result)
    csv_write(temporary / 'risk_coverage_table.csv', short)
    csv_write(temporary / 'all_intervals.csv', long)
    csv_write(temporary / 'paired_contrasts.csv', pairs)
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'consumed_source_changed')
    summary = {'schema_version': 'manual-reader-clustered-risk-coverage-run-v1', 'status': 'complete',
        'attempted_reports': 100, 'distinct_report_artifacts': 100, 'report_finding_checks': 400,
        'independent_patients_verified': False, 'table_rows': 42, 'interval_rows': 378, 'paired_metric_rows': 162,
        'bootstrap_repetitions': 2000, 'bootstrap_seed': 0, 'deterministic_replay_equal': True,
        'historical_point_counts_unchanged': True, 'source_report_or_image_bodies_decoded': False,
        'new_model_calls': 0, 'new_slurm_submissions': 0, 'new_downloads_or_external_api': False,
        'original_bank_unchanged': True, 'clinical_qualified': False, 'thresholds_fitted': False,
        'selection_changed': False, 'regeneration_authorized': False,
        'actual_existing_job_id': os.environ['SLURM_JOB_ID'],
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    write_json(temporary / 'summary.json', summary)
    write_json(temporary / 'manifest.json', {'schema_version': 'manual-reader-clustered-risk-coverage-receipt-v1',
        'pins': pins, 'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())},
        'clinical_qualified': False, 'selection_changed': False})
    for p in [temporary, *temporary.rglob('*')]:
        require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'protected_project_modes_required')
    commit_atomic_run(temporary, target)
    return target, summary


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_cached_diagnostic_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS'),
                'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_clustered_reader_diagnostic_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'peak_rss_gib': round(result['peak_rss_gib'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_clustered_reader_diagnostic_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
