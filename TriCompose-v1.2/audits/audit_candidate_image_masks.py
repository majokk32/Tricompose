"""Independent lossless-cell and proxy-mask replay, never clinical adjudication.

Only sealed cached numeric/state metadata is decoded. No source report text,
pixel, EHR body, original patient key or current model asset is opened.
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time

from audit_cached_image_consensus import replay_decision
from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'candidate_image_mask_overlays/pool960_12784259_001'
EXPECTED = '02dc768e868ee4a40ddc9170cb1cdef4a0e6e501c1f81eba5df88105b2b06aea'
SOURCE = BASE / 'candidate_opacity_biovil_runs/biovil_opacity240_12670345'
XRV = BASE / 'candidate_opacity_runs/opacity_pool240_12668204'
POLICIES = ('xrv_exact_0_5', 'biovil_fixed_mean', 'agree_fixed_mean', 'agree_all_templates')
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')
PREFIX = 'image_mask_'


def proxy_relation(state, report):
    require(state in (None, 'positive', 'negative') and report in ('positive', 'negative', 'uncertain', 'unknown'),
            'four_state_proxy_scope_required')
    if state is None or report in ('uncertain', 'unknown'):
        return 'not_comparable'
    return 'proxy_support' if state == report else 'proxy_opposition'


def load_csv(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED, 'completed_fixed_candidate_mask_run_required')
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for name, value in manifest['artifacts'].items():
        require(sha256(RUN / name) == value, 'candidate_mask_artifact_changed')
    for name, value in manifest['pins'].items():
        p = WORKSPACE / name
        require(p.resolve().is_relative_to(WORKSPACE) and sha256(p) == value, 'consumed_candidate_mask_source_changed')
    paths = [Path(__file__).resolve(), WORKSPACE / 'TriCompose-v1.2/tests/test_candidate_image_masks_audit.py',
             WORKSPACE / 'TriCompose-v1.2/audits/audit_cached_image_consensus.py',
             WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
             WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    temporary, target = new_atomic_run(BASE / 'candidate_image_mask_audits', 'numeric_12784259_001')
    write_json(temporary / 'frozen_plan.json', {'schema_version': 'candidate-image-mask-independent-audit-plan-v1',
        'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
        'production_mask_functions_imported': False, 'clinical_qualified': False,
        'raw_body_pixel_or_model_asset_read': False})
    columns, original = load_csv(SOURCE / 'candidate_evidence_table.csv')
    header, rows = load_csv(RUN / 'candidate_evidence_table.csv')
    require(len(columns) == 86 and len(header) == 104 and header[:86] == columns
            and len(rows) == len(original) == 960, 'full_lossless_candidate_and_column_inventory_required')
    preserved = 0
    for a, b in zip(rows, original):
        for key in columns:
            require(a[key] == b[key], 'original_candidate_cell_or_order_changed')
            preserved += 1
        require(a[PREFIX + 'scope'] == 'exact_opacity_image_proposals_and_cached_report_proposal', 'narrow_opacity_scope_required')
        for flag in ('clinical_primary_eligible', 'reference_metric_transferred', 'synthetic_domain_transport_validated',
                     'selector_used', 'regeneration_authorized'):
            require(a[PREFIX + flag] == 'False', 'no_metric_transfer_or_action_authority_required')
    xrv = {r['cxr_candidate_id']: r for r in json.loads((XRV / 'image_scores.json').read_text())['records']}
    biovil = {r['cxr_candidate_id']: r for r in json.loads((SOURCE / 'image_scores.json').read_text())['records']}
    images = json.loads((RUN / 'image_mask_records.json').read_text())['records']
    require(len(images) == len(xrv) == len(biovil) == 240, 'all_original_image_slots_required')
    decisions = {}
    for image in images:
        key = image['cxr_candidate_id']
        x, b = xrv[key], biovil[key]
        for field in ('case_id', 'cxr_candidate_id', 'cxr_sha256', 'cxr_model_id'):
            require(image[field] == x[field] == b[field], 'independent_image_lineage_mismatch')
        margins = [b['score_pairs'][f]['positive_cosine'] - b['score_pairs'][f]['negative_cosine'] for f in FAMILIES]
        require(image['xrv_score'] == x['exact_lung_opacity_score']
                and image['margins'] == margins and image['pairs'] == b['score_pairs'], 'independent_cached_score_binding_mismatch')
        for policy in POLICIES:
            state, status = replay_decision({'xrv_score': x['exact_lung_opacity_score'], 'margins': margins}, policy)
            require(image['decisions'][policy] == {'state': state, 'status': status}, 'independent_image_mask_decision_mismatch')
            decisions[key, policy] = state, status
    relations = {p: Counter() for p in POLICIES}
    checked = 0
    for r in rows:
        for p in POLICIES:
            state, status = decisions[r['cxr_candidate_id'], p]
            rel = proxy_relation(state, r['opacity_cached_report_state'])
            require(r[PREFIX + p + '_state'] == (state or '') and r[PREFIX + p + '_status'] == status
                    and r[PREFIX + p + '_report_relation'] == rel, 'independent_candidate_mask_relation_mismatch')
            relations[p][rel] += 1
            checked += 1
    summary = json.loads((RUN / 'summary.json').read_text())
    for s in summary['policy_summaries']:
        p = s['policy']
        ds = [decisions[key, p] for key in xrv]
        require(s['accepted_image_slots'] == sum(state is not None for state, _ in ds)
                and s['abstained_image_slots'] == sum(status.startswith('abstain_') for _, status in ds)
                and s['unavailable_image_slots'] == sum(status == 'unavailable_reader' for _, status in ds)
                and s['status_counts'] == dict(Counter(status for _, status in ds)), 'independent_mask_coverage_mismatch')
        comparable = relations[p]['proxy_support'] + relations[p]['proxy_opposition']
        require(all(s[key] == relations[p][key] for key in ('proxy_support', 'proxy_opposition', 'not_comparable'))
                and s['comparable_coverage'] == comparable / 960
                and s['agreement_over_comparable'] == (relations[p]['proxy_support'] / comparable if comparable else None),
                'independent_conditional_proxy_fraction_mismatch')
        for model, value in s['by_report_model'].items():
            counts = Counter(proxy_relation(decisions[r['cxr_candidate_id'], p][0], r['opacity_cached_report_state'])
                             for r in rows if r['report_model_id'] == model)
            require(value['candidate_rows'] == 240 and all(value[k] == counts[k] for k in counts), 'report_model_denominator_mismatch')
    for loss in summary['nested_mask_withholding']:
        left, right = loss['left'], loss['right']
        removed = {key for key in xrv if decisions[key, left][0] is not None and decisions[key, right][0] is None}
        counts = Counter(proxy_relation(decisions[r['cxr_candidate_id'], left][0], r['opacity_cached_report_state'])
                         for r in rows if r['cxr_candidate_id'] in removed)
        require(loss['image_slots_withheld'] == len(removed)
                and loss['proxy_support_rows_withheld'] == counts['proxy_support']
                and loss['proxy_opposition_rows_withheld'] == counts['proxy_opposition']
                and loss['not_comparable_rows_withheld'] == counts['not_comparable']
                and loss['removed_correct_or_incorrect_cases_adjudicated'] is False, 'independent_withholding_counts_mismatch')
    require(preserved == 82560 and checked == 3840, 'all_lossless_cells_and_policy_relations_required')
    audit = {'schema_version': 'candidate-image-mask-independent-audit-v1', 'status': 'passed',
        'completed_run_manifest_sha256': EXPECTED, 'original_csv_cells_verified': preserved,
        'image_mask_decisions_verified': 960, 'candidate_mask_relations_verified': checked,
        'full_candidate_rows': 960, 'full_image_slots': 240, 'fixed_ehr_cases': 80,
        'new_model_calls': 0, 'raw_body_pixel_or_model_asset_read': False,
        'production_mask_functions_imported': False, 'clinical_qualified': False,
        'synthetic_reference_semantics_adjudicated': False, 'selection_changed': False,
        'regeneration_authorized': False, 'runtime_seconds': time.monotonic() - started}
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items())
            and sha256(RUN / 'manifest.json') == EXPECTED, 'candidate_mask_audit_source_changed')
    write_json(temporary / 'audit.json', audit)
    write_json(temporary / 'manifest.json', {'schema_version': 'candidate-mask-independent-audit-receipt-v1',
        'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
        'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())}, 'clinical_qualified': False})
    for p in [temporary, *temporary.rglob('*')]:
        require(p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'private_candidate_mask_audit_required')
    commit_atomic_run(temporary, target)
    return target, audit


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_candidate_mask_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259', 'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_candidate_mask_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_candidate_mask_numeric_audit_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
