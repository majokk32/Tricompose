"""Independent head/domain counters and intervals from cached decisions only.

Reuses the already audited scalar percentile helper, not production analysis
functions. No raw source report, image, annotation span or patient key access.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

import audit_manual_reader_risk_coverage as independent
from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'manual_reader_head_risk_coverage/finding_domains_12784259_001'
EXPECTED = 'b6e0b853e0b0feaf72322fbef94e713dde7203b15ea5bb427a24ae211591d75c'
REF = BASE / 'manual_literal_reader_runs/manual100_12766754_001/reference_projection.json'
DETAILS = BASE / 'manual_three_reader_runs/seven_masks_12766754_001/mask_details.json'
FINDINGS = ('cardiomegaly', 'consolidation', 'pleural_effusion', 'pneumothorax')
FIELDS = independent.FIELDS


def estimate(numerator, denominator):
    return numerator / denominator if denominator else None


def csv_number(value):
    return float(value) if value else None


def verify_numeric_cell(actual, expected):
    value = csv_number(actual)
    require((value is None) == (expected is None), 'head_csv_missing_is_not_zero')
    if expected is not None:
        require(math.isclose(value, expected, rel_tol=0, abs_tol=1e-12), 'head_csv_fraction_or_interval_mismatch')


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED, 'completed_fixed_head_run_required')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(RUN / name) == value, 'completed_head_artifact_changed')
    for name, value in manifest['pins'].items():
        p = WORKSPACE / name
        require(p.resolve().is_relative_to(WORKSPACE) and sha256(p) == value, 'consumed_head_source_changed')
    source_paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/tests/test_manual_reader_head_risk_audit.py',
        WORKSPACE / 'TriCompose-v1.2/audits/audit_manual_reader_risk_coverage.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in source_paths}
    temporary, target = new_atomic_run(BASE / 'manual_reader_head_risk_audits', 'numeric_12784259_001')
    write_json(temporary / 'frozen_plan.json', {'schema_version': 'manual-head-independent-numeric-plan-v1',
        'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
        'production_analysis_functions_imported': False, 'original_source_bodies_read': False,
        'same_report_clusters_and_all_heads_required': True, 'clinical_qualified': False})
    refs = sorted(json.loads(REF.read_text())['records'], key=lambda r: r['report_id'])
    lookup = {r['report_id']: r for r in refs}
    require(len(refs) == len(lookup) == 100 and len({r['source_sha256'] for r in refs}) == 100,
            'same_hundred_distinct_report_artifact_inventory_required')
    details = json.loads(DETAILS.read_text())['records']
    rebuilt = {}
    for row in details:
        ref = lookup[row['report_id']]['finding_states'][row['finding']]
        require(ref == row['human_literal_reference_state'], 'same_cached_literal_reference_required')
        key = tuple(row[k] for k in ('readout', 'policy', 'report_id', 'finding'))
        require(key not in rebuilt and row['finding'] in FINDINGS, 'unique_fixed_head_decision_required')
        c = dict.fromkeys(FIELDS, 0)
        c['attempted'] = 1
        c[ref] = 1
        c['unavailable'] = row['status'] == 'unavailable_reader'
        if row['state'] is not None:
            c['accepted'] = 1
            if ref in ('positive', 'negative'):
                c['accepted_known'] = 1
                c['flips'] = row['state'] != ref
                c['correct_' + ref] = row['state'] == ref
            else:
                c['commit_' + ref] = 1
        rebuilt[key] = [int(c[f]) for f in FIELDS]
    recorded = json.loads((RUN / 'report_head_counters.json').read_text())
    require(tuple(recorded['fields']) == FIELDS and len(recorded['records']) == len(rebuilt) == 5600,
            'all_fixed_report_head_counters_required')
    for row in recorded['records']:
        key = tuple(row[k] for k in ('readout', 'policy', 'report_id', 'finding'))
        ref = lookup[row['report_id']]
        require(row['counters'] == rebuilt[key] and row['source_sha256'] == ref['source_sha256']
                and row['domain'] == ref['source_domain'], 'independent_report_head_vector_or_binding_mismatch')
    result = json.loads((RUN / 'evaluation.json').read_text())
    sampling = result['sampling']
    require(sampling['seed'] == 0 and sampling['repetitions'] == 2000
            and sampling['confidence'] == .95 and sampling['patient_groups_verified'] is False
            and sampling['numpy_version'] == np.__version__, 'same_fixed_head_sampling_required')
    domains = [r['source_domain'] for r in refs]
    weights = np.zeros((2000, 100), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(0))
    for domain in ('mimic', 'chexpert'):
        selected = [i for i, d in enumerate(domains) if d == domain]
        require(len(selected) == 50, 'same_fifty_report_strata_required')
        for i, draws in enumerate(rng.integers(0, 50, size=(2000, 50))):
            for j in draws:
                weights[i, selected[j]] += 1
    require(hashlib.sha256(weights.astype('<i8', copy=False).tobytes()).hexdigest()
            == sampling['weights_sha256'], 'head_paired_report_draw_hash_mismatch')
    points, samples, metric_index = {}, {}, {}
    checked = 0
    for row in result['rows']:
        key = tuple(row[k] for k in ('readout', 'domain', 'policy', 'finding'))
        selected = [i for i, d in enumerate(domains) if row['domain'] == 'all' or d == row['domain']]
        data = np.array([rebuilt[(row['readout'], row['policy'], refs[i]['report_id'], row['finding'])]
                         for i in selected], dtype=np.int64)
        point = data.sum(axis=0)
        require(all(row[f] == int(point[i]) for i, f in enumerate(FIELDS)), 'head_aggregate_counter_mismatch')
        boot = weights[:, selected] @ data
        points[key], samples[key] = {}, {}
        for name, (n, d) in independent.rates(point).items():
            metric = row['metrics'][name]
            value = estimate(int(n), int(d))
            require((metric['numerator'], metric['denominator']) == (int(n), int(d))
                    and metric['estimate'] == value, 'head_point_fraction_mismatch')
            bn, bd = independent.rates(boot)[name]
            vector = np.full(2000, np.nan)
            np.divide(bn, bd, out=vector, where=bd != 0)
            independent.compare_interval(metric, vector)
            points[key][name], samples[key][name] = value, vector
            metric_index[key + (name,)] = metric
            checked += 1
    paired_index = {}
    paired_checks = 0
    for row in result['paired_contrasts']:
        left = tuple(row[k] for k in ('readout', 'domain')) + (row['left'], row['finding'])
        right = tuple(row[k] for k in ('readout', 'domain')) + (row['right'], row['finding'])
        for name, metric in row['differences'].items():
            a, b = points[left][name], points[right][name]
            expected = b - a if a is not None and b is not None else None
            require(metric['right_minus_left'] == expected, 'head_paired_point_or_missing_mismatch')
            independent.compare_interval(metric, samples[right][name] - samples[left][name])
            paired_index[tuple(row[k] for k in ('readout', 'domain', 'left', 'right', 'finding')) + (name,)] = metric
            paired_checks += 1
    csv_checks = 0
    for filename, index, key_names, estimate_name in (
        ('all_intervals.csv', metric_index, ('readout', 'domain', 'policy', 'finding', 'metric'), 'estimate'),
        ('paired_contrasts.csv', paired_index, ('readout', 'domain', 'left', 'right', 'finding', 'metric'), 'right_minus_left')):
        with (RUN / filename).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == len(index), 'head_csv_complete_interval_inventory_required')
        seen = set()
        for row in rows:
            key = tuple(row[k] for k in key_names)
            require(key not in seen, 'head_csv_duplicate_interval_refused')
            seen.add(key)
            metric = index[key]
            verify_numeric_cell(row[estimate_name], metric[estimate_name])
            ci = metric['percentile_interval']
            verify_numeric_cell(row['lower'], ci[0] if ci else None)
            verify_numeric_cell(row['upper'], ci[1] if ci else None)
            require(int(row['valid_draws']) == metric['valid_draws']
                    and int(row['zero_denominator_draws']) == metric['zero_denominator_draws']
                    and row['clinical_qualified'] == 'False', 'head_csv_draw_or_qualification_mismatch')
            csv_checks += 1
    require((checked, paired_checks, csv_checks) == (1512, 648, 2160), 'all_head_numeric_endpoints_required')
    # Verify reference support directly from the cached reference, not predictions.
    support = {}
    for domain in ('all', 'mimic', 'chexpert'):
        selected = [r for r in refs if domain == 'all' or r['source_domain'] == domain]
        for finding in FINDINGS:
            support[domain, finding] = {s: sum(r['finding_states'][finding] == s for r in selected)
                                      for s in ('positive', 'negative', 'uncertain', 'unknown')}
    for row in result['reference_inventory']:
        require(all(row[s] == n for s, n in support[row['domain'], row['finding']].items()),
                'head_reference_support_must_not_come_from_readers')
    audit = {'schema_version': 'manual-reader-head-independent-numeric-audit-v1', 'status': 'passed',
        'completed_run_manifest_sha256': EXPECTED, 'reconstructed_report_head_vectors': 5600,
        'reference_support_groups_verified': 12, 'scalar_intervals_checked': checked,
        'paired_intervals_checked': paired_checks, 'interval_csv_rows_checked': csv_checks,
        'bootstrap_draws_replayed': 2000, 'production_analysis_functions_imported': False,
        'original_report_image_or_patient_key_read': False, 'reference_semantics_independently_adjudicated': False,
        'patient_clusters_verified': False, 'clinical_qualified': False, 'new_model_calls': 0,
        'selection_changed': False, 'regeneration_authorized': False,
        'runtime_seconds': time.monotonic() - started}
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items())
            and sha256(RUN / 'manifest.json') == EXPECTED, 'head_audit_source_changed')
    write_json(temporary / 'audit.json', audit)
    write_json(temporary / 'manifest.json', {'schema_version': 'manual-head-numeric-audit-receipt-v1',
        'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
        'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())}, 'clinical_qualified': False})
    for p in [temporary, *temporary.rglob('*')]:
        require(p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'private_head_audit_required')
    commit_atomic_run(temporary, target)
    return target, audit


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_head_numeric_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259', 'actual_existing_cpu_job_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_report_head_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_report_head_numeric_audit_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
