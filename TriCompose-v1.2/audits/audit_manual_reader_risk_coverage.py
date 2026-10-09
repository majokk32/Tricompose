"""Independent counter/percentile replay, without production analysis imports.

Only sealed opaque indices, derived states and hashes are decoded. Original
reports, original annotation rows, native graph text and images are not opened.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'manual_reader_risk_coverage/clustered_12784259_001'
EXPECTED = '48da1182d159d57db0d826b10d630c19741baba58ca176f643dc4704d40033dc'
REF = BASE / 'manual_literal_reader_runs/manual100_12766754_001/reference_projection.json'
DETAILS = BASE / 'manual_three_reader_runs/seven_masks_12766754_001/mask_details.json'
FIELDS = ('attempted', 'positive', 'negative', 'uncertain', 'unknown', 'correct_positive',
          'correct_negative', 'flips', 'accepted_known', 'commit_uncertain', 'commit_unknown', 'accepted', 'unavailable')


def percentile(values):
    ordered = sorted(float(v) for v in values if np.isfinite(v))
    if not ordered:
        return None, 0
    out = []
    for q in (.025, .975):
        pos = (len(ordered) - 1) * q
        low, high = math.floor(pos), math.ceil(pos)
        out.append(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))
    return out, len(ordered)


def rates(a):
    c = {f: a[..., i] for i, f in enumerate(FIELDS)}
    return {'known_recovery': (c['correct_positive'] + c['correct_negative'], c['positive'] + c['negative']),
        'positive_recovery': (c['correct_positive'], c['positive']),
        'negative_recovery': (c['correct_negative'], c['negative']),
        'conditional_literal_assertion_error': (c['flips'] + c['commit_uncertain'], c['accepted'] - c['commit_unknown']),
        'adjudicable_proposal_coverage': (c['accepted'] - c['commit_unknown'], c['positive'] + c['negative'] + c['uncertain']),
        'proposal_coverage': (c['accepted'], c['attempted']),
        'uncertain_commitment_rate': (c['commit_uncertain'], c['uncertain']),
        'literal_unknown_commitment_rate': (c['commit_unknown'], c['unknown']),
        'unavailable_check_rate': (c['unavailable'], c['attempted'])}


def compare_interval(actual, samples):
    expected, valid = percentile(samples)
    require(valid == actual['valid_draws'] and len(samples) - valid == actual['zero_denominator_draws'],
            'missing_draw_counts_mismatch')
    if expected is None:
        require(actual['percentile_interval'] is None, 'missing_interval_is_not_zero')
    else:
        require(actual['percentile_interval'] is not None
                and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                        for a, b in zip(expected, actual['percentile_interval'])), 'independent_percentile_mismatch')


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED, 'completed_risk_analysis_pin_required')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(RUN / name) == value, 'completed_risk_artifact_changed')
    for name, value in manifest['pins'].items():
        path = WORKSPACE / name
        require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == value, 'consumed_source_changed')
    references = json.loads(REF.read_text())['records']
    refs = sorted(references, key=lambda r: r['report_id'])
    lookup = {r['report_id']: r for r in refs}
    source = json.loads(DETAILS.read_text())['records']
    require(len(refs) == 100 and len(lookup) == 100 and len(source) == 5600, 'complete_original_inventory_required')
    rebuilt = {}
    for row in source:
        key = row['readout'], row['policy'], row['report_id']
        ref = lookup[row['report_id']]['finding_states'][row['finding']]
        require(ref == row['human_literal_reference_state'], 'same_human_projected_state_required')
        a = rebuilt.setdefault(key, dict.fromkeys(FIELDS, 0))
        a['attempted'] += 1
        a[ref] += 1
        a['unavailable'] += row['status'] == 'unavailable_reader'
        if row['state'] is not None:
            a['accepted'] += 1
            if ref in ('positive', 'negative'):
                a['accepted_known'] += 1
                a['flips'] += row['state'] != ref
                a['correct_' + ref] += row['state'] == ref
            else:
                a['commit_' + ref] += 1
    counters = json.loads((RUN / 'report_cluster_counters.json').read_text())
    require(tuple(counters['fields']) == FIELDS and len(counters['records']) == len(rebuilt) == 1400,
            'all_fixed_report_counters_required')
    for row in counters['records']:
        key = row['readout'], row['policy'], row['report_id']
        ref = lookup[row['report_id']]
        require(row['counters'] == [rebuilt[key][f] for f in FIELDS]
                and row['source_sha256'] == ref['source_sha256'] and row['domain'] == ref['source_domain'],
                'independent_per_report_counter_or_hash_mismatch')
    result = json.loads((RUN / 'evaluation.json').read_text())
    protocol = result['sampling']
    require(protocol['repetitions'] == 2000 and protocol['seed'] == 0 and protocol['confidence'] == .95
            and protocol['patient_groups_verified'] is False and protocol['numpy_version'] == np.__version__,
            'same_frozen_report_cluster_protocol_required')
    domains = [r['source_domain'] for r in refs]
    rng = np.random.Generator(np.random.PCG64(0))
    weights = np.zeros((2000, 100), dtype=np.int64)
    for domain in ('mimic', 'chexpert'):
        indices = [i for i, d in enumerate(domains) if d == domain]
        require(len(indices) == 50, 'fixed_fifty_report_strata_required')
        sampled = rng.integers(0, 50, size=(2000, 50))
        for i, selected in enumerate(sampled):
            for j in selected:
                weights[i, indices[j]] += 1
    require(hashlib.sha256(weights.astype('<i8', copy=False).tobytes()).hexdigest() == protocol['weights_sha256'],
            'shared_cluster_draw_replay_mismatch')
    values = {}
    checked = 0
    for row in result['rows']:
        selected = [i for i, d in enumerate(domains) if row['domain'] == 'all' or d == row['domain']]
        data = np.array([[rebuilt[(row['readout'], row['policy'], refs[i]['report_id'])][f] for f in FIELDS]
                         for i in selected], dtype=np.int64)
        point = data.sum(axis=0)
        sample = weights[:, selected] @ data
        samples = {}
        for name, (n, d) in rates(sample).items():
            pn, pd = rates(point)[name]
            metric = row['metrics'][name]
            require((metric['numerator'], metric['denominator']) == (int(pn), int(pd)), 'rate_counter_mismatch')
            estimate = float(pn / pd) if pd else None
            require(metric['estimate'] == estimate, 'point_fraction_mismatch')
            vector = np.full(2000, np.nan)
            np.divide(n, d, out=vector, where=d != 0)
            compare_interval(metric, vector)
            samples[name] = vector
            checked += 1
        values[(row['readout'], row['domain'], row['policy'])] = samples
    paired_checks = 0
    for row in result['paired_contrasts']:
        a = values[(row['readout'], row['domain'], row['left'])]
        b = values[(row['readout'], row['domain'], row['right'])]
        for name, metric in row['differences'].items():
            compare_interval(metric, b[name] - a[name])
            paired_checks += 1
    parent = BASE / 'manual_reader_risk_coverage_audits'
    private_dir(parent)
    target = parent / 'numeric_12784259_001'
    private_dir(target, fresh=True)
    sources = [Path(__file__).resolve(), WORKSPACE / 'TriCompose-v1.2/tests/test_manual_reader_risk_coverage_audit.py']
    audit_pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in sources}
    audit = {'schema_version': 'manual-reader-risk-independent-numeric-audit-v1', 'status': 'passed',
        'completed_run_manifest_sha256': EXPECTED, 'reconstructed_finding_decisions': 5600,
        'report_counter_vectors_checked': 1400, 'scalar_intervals_checked': checked,
        'paired_intervals_checked': paired_checks, 'bootstrap_draws_replayed': 2000,
        'report_sources_or_images_decoded': False, 'production_analysis_functions_imported': False,
        'literal_gold_semantics_reconstructed': False, 'patient_clusters_verified': False,
        'new_model_calls': 0, 'clinical_qualified': False, 'selection_changed': False,
        'regeneration_authorized': False, 'runtime_seconds': time.monotonic() - started}
    require((checked, paired_checks) == (378, 162), 'all_frozen_intervals_required')
    write_json(target / 'audit.json', audit)
    write_json(target / 'manifest.json', {'schema_version': 'manual-reader-risk-numeric-audit-receipt-v1',
        'completed_run_manifest_sha256': EXPECTED, 'pins': audit_pins,
        'artifacts': {'audit.json': sha256(target / 'audit.json')}, 'clinical_qualified': False})
    return target, audit


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_numeric_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259', 'actual_existing_cpu_job_required')
        os.umask(0o007)
        target, audit = execute()
        print(json.dumps({'status': 'protected_reader_risk_numeric_audit_passed',
            'runtime_seconds': round(audit['runtime_seconds'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_reader_risk_numeric_audit_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
