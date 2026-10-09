"""Correct dependency inventory; reuse immutable cache-join/table helpers.

The first launcher failed on a nonexistent dependency before creating a run
or decoding cached score rows. It is retained, not silently overwritten.
No reader, threshold, template, reference or statistical endpoint changes.
"""
import json
import os
from pathlib import Path
import resource
import sys
import time

import benchmark_cached_image_consensus as cache
from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json
from tricompose_v12.image_reader_consensus import analyze, POLICIES, CONTRASTS


def sources():
    names = ('tools/benchmark_cached_image_consensus.py',
             'tools/benchmark_cached_image_consensus_v2.py',
             'src/tricompose_v12/image_reader_consensus.py',
             'src/tricompose_v12/manual_reader_risk_coverage.py',
             'src/tricompose_v12/manual_three_reader_agreement.py',
             'src/tricompose_v12/assertion_agreement_diagnostic.py',
             'src/tricompose_v12/manual_literal_assertions.py',
             'src/tricompose_v12/entity_gold_contract.py',
             'src/tricompose_v12/radgraph_assertion_readout.py',
             'src/tricompose_v12/radgraph_reference_contract.py',
             'tests/test_image_reader_consensus.py',
             'tests/test_cached_image_consensus_worker.py',
             'tests/test_cached_image_consensus_worker_v2.py',
             'tools/prepare_ratescore_assets.py')
    return [WORKSPACE / 'TriCompose-v1.2' / name for name in names] + [
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
        cache.XRV / 'manifest.json', cache.XRV / 'scores.json', cache.XRV / 'summary.json',
        cache.BIOVIL / 'manifest.json', cache.BIOVIL / 'scores.json', cache.BIOVIL / 'summary.json', cache.BANK]


def execute():
    started = time.monotonic()
    xm, xs = cache.metadata(cache.XRV, cache.XRV_SHA)
    bm, bs = cache.metadata(cache.BIOVIL, cache.BIOVIL_SHA)
    require(bm['source_manifest_sha256'] == cache.XRV_SHA and xs['reference_kind'] == bs['reference_kind']
            and bs['primary_reduction'] == 'mean_all_three_predeclared'
            and xs['thresholds_fitted_on_this_cohort'] is False and bs['thresholds_fitted'] is False
            and bs['templates_fitted'] is False and bs['selection_changed'] is False,
            'unchanged_reference_threshold_and_template_scope_required')
    require(sha256(cache.BANK) == cache.BANK_SHA, 'original_synthetic_bank_unchanged_required')
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in sources()}
    temporary, target = new_atomic_run(cache.BASE / 'image_reader_consensus_runs', 'ricord50_12784259_002')
    write_json(temporary / 'frozen_plan.json', {'schema_version': 'cached-image-consensus-diagnostic-plan-v1',
        'pins': pins, 'fixed_policies': POLICIES, 'fixed_paired_contrasts': CONTRASTS,
        'attempted_cases': 50, 'finding': 'exact_lung_opacity_not_pneumonia_or_14_labels',
        'reference_kind': xs['reference_kind'], 'reference_supplied_by_model_agreement': False,
        'xrv_threshold': .5, 'xrv_exact_head_not_max_infiltration_proxy': True,
        'biovil_reduction': 'mean_all_three_predeclared', 'stable_templates_are_not_separate_voters': True,
        'seed': 0, 'bootstrap_repetitions': 2000, 'confidence': .95,
        'one_patient_per_case_from_pinned_previous_acquisition_receipts': True,
        'original_patient_metadata_binding_rechecked_here': False,
        'pixels_read_or_models_called': False, 'post_hoc_development': True,
        'new_untouched_test': False, 'clinical_qualified': False,
        'synthetic_domain_transport_validated': False, 'scoring_reference_produces_no_synthetic_labels': True,
        'selection_changed': False, 'regeneration_authorized': False,
        'launcher_v1_failed_before_protected_run_or_cached_row_decode': True,
        'launcher_revision_changes_dependency_inventory_only': True})
    joined = cache.cached_join(json.loads((cache.XRV / 'scores.json').read_text()),
                              json.loads((cache.BIOVIL / 'scores.json').read_text()), xm)
    result = analyze(joined)
    require(result == analyze(joined), 'exact_bootstrap_and_counter_replay_required')
    original = xs['metrics']['direct_lung_opacity_default']
    counts = result['rows'][0]['counts']
    require([counts[f] for f in ('true_positive', 'false_negative', 'true_negative', 'false_positive')]
            == [25, 0, 19, 6], 'historical_exact_head_confusion_must_remain_identical')
    require(original['sensitivity'] == 1 and original['specificity'] == .76,
            'original_fixed_head_summary_unchanged_required')
    table, pairs = cache.tables(result)
    require((len(table), len(pairs), len(result['outcomes'])) == (4, 21, 200),
            'all_fixed_policies_and_cases_required')
    write_json(temporary / 'joined_score_metadata.json', {'records': joined})
    write_json(temporary / 'evaluation.json', {k: v for k, v in result.items() if k != 'outcomes'})
    write_json(temporary / 'policy_outcomes.json', {'records': result['outcomes']})
    cache.csv_write(temporary / 'risk_coverage_table.csv', table)
    cache.csv_write(temporary / 'paired_contrasts.csv', pairs)
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()),
            'consumed_cache_or_program_changed')
    summary = {'schema_version': 'cached-image-reader-consensus-run-v1', 'status': 'complete',
        'attempted_cases': 50, 'reference_positive': 25, 'reference_negative': 25,
        'fixed_policies': 4, 'policy_outcomes': 200, 'paired_metric_rows': 21,
        'frozen_reference_kind': xs['reference_kind'], 'pixel_or_report_bodies_read': False,
        'original_patient_source_identifiers_read': False, 'new_model_calls': 0,
        'new_slurm_submissions': 0, 'patient_grouping_verified_in_pinned_parent': True,
        'patient_binding_independently_reverified_here': False, 'deterministic_replay_equal': True,
        'official_adjudication_reproduced': False, 'clinical_qualified': False,
        'old_exact_head_point_result_unchanged': True, 'original_bank_unchanged': True,
        'selection_changed': False, 'regeneration_authorized': False,
        'actual_existing_cpu_job_id': os.environ['SLURM_JOB_ID'],
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    write_json(temporary / 'summary.json', summary)
    write_json(temporary / 'manifest.json', {'schema_version': 'cached-image-reader-consensus-receipt-v1',
        'pins': pins, 'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())},
        'clinical_qualified': False, 'selection_changed': False})
    for p in [temporary, *temporary.rglob('*')]:
        require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660),
                'private_project_artifacts_required')
    commit_atomic_run(temporary, target)
    return target, summary


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_cached_reference_scope_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS'),
                'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_image_consensus_diagnostic_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'peak_rss_gib': round(result['peak_rss_gib'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_image_consensus_diagnostic_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
