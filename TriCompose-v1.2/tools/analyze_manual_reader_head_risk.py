"""Cached head/domain diagnostic in the existing CPU allocation only.

No new model calls, raw source text/pixels, original keys or policy changes.
The previous pooled run and all candidate selections remain immutable.
"""
import json
import os
from pathlib import Path
import resource
import sys
import time

import analyze_manual_reader_risk_coverage as parent
from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json
from tricompose_v12.manual_reader_head_risk import (
    CONTRASTS, DOMAINS, FIELDS, FINDINGS, VIEWS, aggregate_heads, analyze_heads,
)

BASE = parent.BASE
POOLED = BASE / 'manual_reader_risk_coverage/clustered_12784259_001'
POOLED_SHA = '48da1182d159d57db0d826b10d630c19741baba58ca176f643dc4704d40033dc'
NUMERIC = BASE / 'manual_reader_risk_coverage_audits/numeric_12784259_001'
NUMERIC_SHA = 'f74185da6ec2bf5f7674e50f7fbea5bf8530e3bbee6772c878053d963d47928d'


def sources():
    names = ('tools/analyze_manual_reader_head_risk.py', 'tools/analyze_manual_reader_risk_coverage.py',
             'src/tricompose_v12/manual_reader_head_risk.py',
             'src/tricompose_v12/manual_reader_risk_coverage.py',
             'src/tricompose_v12/manual_three_reader_agreement.py',
             'src/tricompose_v12/assertion_agreement_diagnostic.py',
             'src/tricompose_v12/manual_literal_assertions.py',
             'src/tricompose_v12/entity_gold_contract.py',
             'src/tricompose_v12/radgraph_assertion_readout.py',
             'src/tricompose_v12/radgraph_reference_contract.py',
             'tests/test_manual_reader_head_risk.py', 'tests/test_manual_reader_head_risk_worker.py',
             'tests/test_manual_reader_risk_coverage.py', 'tools/prepare_ratescore_assets.py')
    return [WORKSPACE / 'TriCompose-v1.2' / n for n in names] + [
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
        parent.REF / 'manifest.json', parent.REF / 'reference_projection.json',
        parent.MASKS / 'manifest.json', parent.MASKS / 'mask_details.json',
        POOLED / 'manifest.json', POOLED / 'evaluation.json', POOLED / 'report_cluster_counters.json',
        NUMERIC / 'manifest.json', NUMERIC / 'audit.json', parent.BANK]


def tables(result):
    short, long, pairs = parent.tables(result)
    short = [{'finding': r['finding'], **out} for r, out in zip(result['rows'], short)]
    long = [{'finding': r['finding'], **out}
            for r, out in zip((row for row in result['rows'] for _ in row['metrics']), long)]
    pairs = [{'finding': r['finding'], **out}
             for r, out in zip((row for row in result['paired_contrasts'] for _ in row['differences']), pairs)]
    require((len(short), len(long), len(pairs)) == (168, 1512, 648), 'all_frozen_head_tables_required')
    return short, long, pairs


def execute():
    started = time.monotonic()
    for root, expected in ((POOLED, POOLED_SHA), (NUMERIC, NUMERIC_SHA)):
        require(sha256(root / 'manifest.json') == expected, 'sealed_pooled_numeric_parent_required')
        m = json.loads((root / 'manifest.json').read_text())
        for name, value in m['artifacts'].items():
            require(sha256(root / name) == value, 'sealed_pooled_artifact_changed')
        for name, value in m['pins'].items():
            p = WORKSPACE / name
            require(p.resolve().is_relative_to(WORKSPACE) and sha256(p) == value, 'consumed_pooled_source_changed')
    for root, expected, name in ((parent.REF, parent.REF_SHA, 'reference_projection.json'),
                                 (parent.MASKS, parent.MASKS_SHA, 'mask_details.json')):
        require(sha256(root / 'manifest.json') == expected, 'same_sealed_reference_and_masks_required')
        m = json.loads((root / 'manifest.json').read_text())
        require(sha256(root / name) == m['artifacts'][name]
                and (root / name).stat().st_size < 8 * 1024**2, 'bounded_sealed_derived_rows_required')
    require(sha256(parent.BANK) == parent.BANK_SHA, 'original_candidate_bank_unchanged_required')
    audit = json.loads((NUMERIC / 'audit.json').read_text())
    require(audit['status'] == 'passed' and audit['completed_run_manifest_sha256'] == POOLED_SHA
            and audit['scalar_intervals_checked'] == 378, 'completed_independent_pooled_audit_required')
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in sources()}
    temporary, target = new_atomic_run(BASE / 'manual_reader_head_risk_coverage', 'finding_domains_12784259_001')
    write_json(temporary / 'frozen_plan.json', {'schema_version': 'manual-reader-head-risk-plan-v1',
        'pins': pins, 'findings': FINDINGS, 'readouts': VIEWS, 'domains': DOMAINS,
        'all_seven_existing_masks_retained': True, 'fixed_contrasts': CONTRASTS,
        'seed': 0, 'bootstrap_repetitions': 2000, 'confidence': .95,
        'same_report_stratified_draws_as_previous_pooled_diagnostic': True,
        'patient_clusters_verified': False, 'uncertain_commitments_count_as_literal_errors': True,
        'literal_unknown_is_not_negative_or_adjudicated_error': True,
        'post_hoc_development': True, 'new_untouched_test_or_clinical_calibration': False,
        'support_counts_described_without_inventing_minimum_sample_gate': True,
        'clinical_qualified': False, 'best_policy_selected': False, 'selection_changed': False,
        'regeneration_authorized': False})
    refs = json.loads((parent.REF / 'reference_projection.json').read_text())['records']
    details = json.loads((parent.MASKS / 'mask_details.json').read_text())['records']
    require(len(refs) == 100 and len(details) == 5600, 'same_complete_manual_inventory_required')
    result = analyze_heads(refs, details)
    require(result == analyze_heads(refs, details), 'exact_head_bootstrap_replay_required')
    old = json.loads((POOLED / 'evaluation.json').read_text())
    require(result['sampling']['weights_sha256'] == old['sampling']['weights_sha256'],
            'unchanged_parent_report_draw_hash_required')
    for old_row in old['rows']:
        selected = [r for r in result['rows'] if all(r[k] == old_row[k] for k in ('readout', 'domain', 'policy'))]
        require(len(selected) == 4 and all(sum(r[f] for r in selected) == old_row[f] for f in FIELDS),
                'all_pooled_point_counters_must_remain_identical')
    refs, policies, data = aggregate_heads(refs, details)
    recorded = json.loads((POOLED / 'report_cluster_counters.json').read_text())['records']
    lookup = {(r['readout'], r['policy'], r['report_id']): r for r in recorded}
    for v, view in enumerate(VIEWS):
        for p, policy in enumerate(policies):
            for i, ref in enumerate(refs):
                r = lookup[view, policy, ref['report_id']]
                require(data[v, p, :, i].sum(axis=0).tolist() == r['counters']
                        and r['source_sha256'] == ref['source_sha256'] and r['domain'] == ref['source_domain'],
                        'every_parent_report_counter_and_binding_must_remain_identical')
    short, long, pairs = tables(result)
    write_json(temporary / 'evaluation.json', result)
    write_json(temporary / 'report_head_counters.json', {'fields': FIELDS,
        'records': [{'readout': view, 'policy': policy, 'finding': finding, 'report_id': ref['report_id'],
                     'source_sha256': ref['source_sha256'], 'domain': ref['source_domain'],
                     'counters': data[v, p, h, i].tolist()}
                    for v, view in enumerate(VIEWS) for p, policy in enumerate(policies)
                    for h, finding in enumerate(FINDINGS) for i, ref in enumerate(refs)]})
    parent.csv_write(temporary / 'risk_coverage_table.csv', short)
    parent.csv_write(temporary / 'all_intervals.csv', long)
    parent.csv_write(temporary / 'paired_contrasts.csv', pairs)
    parent.csv_write(temporary / 'reference_support.csv', result['reference_inventory'])
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'head_sources_changed_during_run')
    summary = {'schema_version': 'manual-reader-head-risk-run-v1', 'status': 'complete',
        'reports': 100, 'literal_heads': 4, 'reference_inventory_rows': 12,
        'risk_table_rows': 168, 'scalar_interval_rows': 1512, 'paired_metric_rows': 648,
        'report_head_counters': 5600, 'same_pooled_report_draw_hash': True,
        'all_42_pooled_point_counter_rows_conserved': True, 'all_1400_pooled_report_vectors_conserved': True,
        'source_report_image_or_original_key_read': False, 'new_model_calls': 0, 'new_slurm_submissions': 0,
        'original_bank_unchanged': True, 'clinical_qualified': False, 'thresholds_fitted': False,
        'selection_changed': False, 'regeneration_authorized': False, 'patient_clusters_verified': False,
        'deterministic_replay_equal': True, 'actual_existing_cpu_job': os.environ['SLURM_JOB_ID'],
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    write_json(temporary / 'summary.json', summary)
    write_json(temporary / 'manifest.json', {'schema_version': 'manual-reader-head-risk-receipt-v1',
        'pins': pins, 'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())},
        'clinical_qualified': False, 'selection_changed': False})
    for p in [temporary, *temporary.rglob('*')]:
        require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'project_private_head_outputs_required')
    commit_atomic_run(temporary, target)
    return target, summary


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_manual_head_diagnostic_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS'),
                'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_report_head_risk_diagnostic_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'peak_rss_gib': round(result['peak_rss_gib'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_report_head_risk_diagnostic_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
