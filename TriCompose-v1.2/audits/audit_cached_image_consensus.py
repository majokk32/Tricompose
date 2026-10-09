"""Separate scalar/counter replay; never imports production consensus code.

Only cached score metadata, opaque indices and previously recorded hashes are
decoded. No original image, annotation row, report or patient key is opened.
This audits arithmetic and bindings, not independent clinical truth.
"""
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'image_reader_consensus_runs/ricord50_12784259_002'
EXPECTED = 'add0c915921d0e2fa0741d604608186ce67faf83d94ad81f370eddc3c5997c29'
XRV = BASE / 'real_validation/ricord_xrv_pilots/ricord_xrv50_12667524'
BIOVIL = BASE / 'real_validation/ricord_biovil_pilots/ricord_biovil50_12669671'
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
POLICIES = ('xrv_exact_0_5', 'biovil_fixed_mean', 'agree_fixed_mean', 'agree_all_templates')
FIELDS = ('attempted', 'reference_positive', 'reference_negative', 'true_positive',
          'false_negative', 'true_negative', 'false_positive', 'accepted', 'abstained', 'unavailable')


def replay_decision(row, policy):
    x = row['xrv_score']
    values = row['margins']
    xstate = None if x is None else 'positive' if x >= .5 else 'negative'
    signs = None if values is None else ['positive' if v > 0 else 'negative' if v < 0 else None for v in values]
    total = None if values is None else sum(values)
    bstate = None if total is None or total == 0 else 'positive' if total > 0 else 'negative'
    if policy == 'xrv_exact_0_5':
        return (xstate, 'accepted_determinate') if xstate else (None, 'unavailable_reader')
    if policy == 'biovil_fixed_mean':
        if values is None:
            return None, 'unavailable_reader'
        return (bstate, 'accepted_determinate') if bstate else (None, 'abstain_mean_tie')
    require(policy in ('agree_fixed_mean', 'agree_all_templates'), 'fixed_audit_policy_required')
    if x is None or values is None:
        return None, 'unavailable_reader'
    if policy == 'agree_all_templates':
        if None in signs:
            return None, 'abstain_template_tie'
        if len(set(signs)) != 1:
            return None, 'abstain_template_sensitive'
    if bstate is None:
        return None, 'abstain_mean_tie'
    return (xstate, 'accepted_determinate') if xstate == bstate else (None, 'abstain_reader_disagreement')


def fractions(c):
    return {'accepted_error_risk': (c['false_positive'] + c['false_negative'], c['accepted']),
        'proposal_coverage': (c['accepted'], c['attempted']),
        'positive_reference_recovery': (c['true_positive'], c['reference_positive']),
        'negative_reference_recovery': (c['true_negative'], c['reference_negative']),
        'conditional_positive_recovery': (c['true_positive'], c['true_positive'] + c['false_negative']),
        'conditional_negative_recovery': (c['true_negative'], c['true_negative'] + c['false_positive']),
        'unavailable_rate': (c['unavailable'], c['attempted'])}


def percentile(values):
    ordered = sorted(float(v) for v in values if math.isfinite(v))
    if not ordered:
        return None, 0
    out = []
    for q in (.025, .975):
        pos = (len(ordered) - 1) * q
        low, high = math.floor(pos), math.ceil(pos)
        out.append(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))
    return out, len(ordered)


def check_interval(metric, values):
    expected, valid = percentile(values)
    require(metric['valid_draws'] == valid and metric['zero_denominator_draws'] == len(values) - valid,
            'audit_valid_draw_count_mismatch')
    actual = metric['percentile_interval']
    require((actual is None) == (expected is None), 'audit_missing_interval_is_not_zero')
    if expected is not None:
        require(len(actual) == 2 and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                                       for a, b in zip(actual, expected)), 'audit_scalar_percentile_mismatch')


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED, 'fixed_completed_consensus_required')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(RUN / name) == value, 'completed_consensus_artifact_changed')
    for name, value in manifest['pins'].items():
        p = WORKSPACE / name
        require(p.resolve().is_relative_to(WORKSPACE) and sha256(p) == value, 'consumed_consensus_source_changed')
    audit_sources = [Path(__file__).resolve(),
                     WORKSPACE / 'TriCompose-v1.2/tests/test_cached_image_consensus_audit.py',
                     WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
                     WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in audit_sources}
    temporary, target = new_atomic_run(BASE / 'image_reader_consensus_audits', 'numeric_12784259_001')
    write_json(temporary / 'frozen_plan.json', {'schema_version': 'cached-image-consensus-numeric-audit-plan-v1',
        'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
        'fixed_policies': POLICIES, 'bootstrap_seed': 0, 'bootstrap_repetitions': 2000,
        'original_pixels_or_patient_source_rows_read': False, 'production_consensus_functions_imported': False,
        'clinical_qualified': False, 'selection_changed': False})
    x = json.loads((XRV / 'scores.json').read_text())
    b = json.loads((BIOVIL / 'scores.json').read_text())
    xm = json.loads((XRV / 'manifest.json').read_text())
    xr = {r['case_id']: r for r in x['records']}
    br = {r['case_id']: r for r in b['records']}
    joined = json.loads((RUN / 'joined_score_metadata.json').read_text())['records']
    require(len(joined) == len(xr) == len(br) == 50, 'all_fifty_cached_sources_required')
    for r in joined:
        cid = r['case_id']
        a, v = xr[cid], br[cid]
        expected_margins = [v['score_pairs'][f]['positive_cosine'] - v['score_pairs'][f]['negative_cosine']
                            for f in FAMILIES]
        require(r['image_sha256'] == a['display_sha256'] == v['source_image_sha256']
                == xm['artifacts']['displays/' + cid + '.png']
                and r['reference_state'] == a['reference_state']
                and r['xrv_score'] == a['direct_lung_opacity_score']
                and r['margins'] == expected_margins, 'independent_join_binding_mismatch')
    require([r['case_id'] for r in joined] == [f'case_{i:03d}' for i in range(50)]
            and len({r['image_sha256'] for r in joined}) == 50
            and Counter(r['reference_state'] for r in joined) == {'positive': 25, 'negative': 25},
            'same_opaque_inventory_and_reference_strata_required')
    outcomes = json.loads((RUN / 'policy_outcomes.json').read_text())['records']
    lookup = {(r['policy'], r['case_id']): r for r in outcomes}
    require(len(lookup) == len(outcomes) == 200, 'all_fixed_policy_decisions_required')
    counters, statuses = {}, {}
    for policy in POLICIES:
        data, status_counts = [], Counter()
        for r in joined:
            state, status = replay_decision(r, policy)
            recorded = lookup[policy, r['case_id']]
            require(recorded == {**{k: r[k] for k in ('case_id', 'image_sha256', 'reference_state')},
                                 'policy': policy, 'state': state, 'status': status}, 'independent_image_decision_mismatch')
            status_counts[status] += 1
            c = dict.fromkeys(FIELDS, 0)
            c['attempted'] = 1
            c['reference_' + r['reference_state']] = 1
            if state is None:
                c['unavailable' if status == 'unavailable_reader' else 'abstained'] = 1
            else:
                c['accepted'] = 1
                c[('true_' if state == r['reference_state'] else 'false_') + state] = 1
            data.append(c)
        counters[policy], statuses[policy] = data, dict(status_counts)
    result = json.loads((RUN / 'evaluation.json').read_text())
    protocol = result['sampling']
    require(protocol['seed'] == 0 and protocol['repetitions'] == 2000 and protocol['confidence'] == .95
            and protocol['numpy_version'] == np.__version__, 'same_fixed_sampling_protocol_required')
    weights = np.zeros((2000, 50), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(0))
    for state in ('positive', 'negative'):
        indices = [i for i, r in enumerate(joined) if r['reference_state'] == state]
        samples = rng.integers(0, 25, size=(2000, 25))
        for i, selected in enumerate(samples):
            for j in selected:
                weights[i, indices[j]] += 1
    require(hashlib.sha256(weights.astype('<i8', copy=False).tobytes()).hexdigest()
            == protocol['weights_sha256'], 'independent_bootstrap_draw_hash_mismatch')
    vectors, points = {}, {}
    scalar_checks = 0
    for row in result['rows']:
        policy = row['policy']
        data = counters[policy]
        total = {f: sum(c[f] for c in data) for f in FIELDS}
        require(row['counts'] == total and row['status_counts'] == statuses[policy], 'independent_policy_counter_mismatch')
        points[policy] = {k: n / d if d else None for k, (n, d) in fractions(total).items()}
        vectors[policy] = {k: [] for k in fractions(total)}
        # Scalar counter reconstruction rather than production einsum/reduction.
        for draws in weights:
            c = {f: sum(int(w) * a[f] for w, a in zip(draws, data)) for f in FIELDS}
            for name, (n, d) in fractions(c).items():
                vectors[policy][name].append(n / d if d else float('nan'))
        for name, (n, d) in fractions(total).items():
            m = row['metrics'][name]
            require((m['numerator'], m['denominator']) == (n, d)
                    and m['estimate'] == points[policy][name], 'independent_point_fraction_mismatch')
            check_interval(m, vectors[policy][name])
            scalar_checks += 1
    paired_checks = 0
    for contrast in result['paired_contrasts']:
        left, right = contrast['left'], contrast['right']
        for name, m in contrast['metrics'].items():
            value = points[right][name] - points[left][name]
            require(m['right_minus_left'] == value, 'independent_difference_direction_mismatch')
            values = [b - a for a, b in zip(vectors[left][name], vectors[right][name])]
            check_interval(m, values)
            paired_checks += 1
    csv_checks = 0
    with (RUN / 'risk_coverage_table.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))
    require(len(table) == 4, 'complete_policy_csv_required')
    for c, row in zip(table, result['rows']):
        require(c['policy'] == row['policy']
                and all(int(c[f]) == row['counts'][f] for f in FIELDS)
                and float(c['accepted_error_risk']) == points[row['policy']]['accepted_error_risk']
                and float(c['proposal_coverage']) == points[row['policy']]['proposal_coverage']
                and c['clinical_qualified'] == c['official_adjudication_reproduced'] == 'False', 'policy_csv_counter_mismatch')
        csv_checks += 1
    with (RUN / 'paired_contrasts.csv').open(newline='') as stream:
        pairs = list(csv.DictReader(stream))
    require(len(pairs) == 21, 'complete_paired_csv_required')
    for c in pairs:
        require(float(c['right_minus_left']) == points[c['right']][c['metric']] - points[c['left']][c['metric']]
                and c['shared_paired_draws'] == 'True', 'paired_csv_difference_mismatch')
        csv_checks += 1
    require((scalar_checks, paired_checks, csv_checks) == (28, 21, 25), 'all_numeric_endpoints_required')
    audit = {'schema_version': 'cached-image-consensus-independent-numeric-audit-v1', 'status': 'passed',
        'completed_run_manifest_sha256': EXPECTED, 'cache_case_bindings_checked': 50,
        'reconstructed_image_decisions': 200, 'scalar_intervals_checked': scalar_checks,
        'paired_intervals_checked': paired_checks, 'csv_rows_checked': csv_checks,
        'bootstrap_draws_replayed': 2000, 'raw_images_or_annotation_bodies_read': False,
        'production_consensus_functions_imported': False, 'patient_source_binding_reverified': False,
        'reference_semantics_independently_adjudicated': False, 'new_model_calls': 0,
        'clinical_qualified': False, 'selection_changed': False,
        'runtime_seconds': time.monotonic() - started}
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items())
            and sha256(RUN / 'manifest.json') == EXPECTED, 'audit_source_or_receipt_changed')
    write_json(temporary / 'audit.json', audit)
    write_json(temporary / 'manifest.json', {'schema_version': 'cached-image-consensus-numeric-audit-receipt-v1',
        'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
        'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())}, 'clinical_qualified': False})
    for p in [temporary, *temporary.rglob('*')]:
        require(p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'project_private_audit_required')
    commit_atomic_run(temporary, target)
    return target, audit


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_cached_image_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259', 'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_image_consensus_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_image_consensus_numeric_audit_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
