"""Existing-CPU cached mask annotation; source bodies/pixels never reopened."""
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time

from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json
from tricompose_v12.candidate_image_masks import overlay, POLICIES, CONTRASTS

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
BIOVIL = BASE / 'candidate_opacity_biovil_runs/biovil_opacity240_12670345'
BIOVIL_SHA = '6b3027a012ec146b46a29a911465bb8fe1bdded087b6ee074d5994919c89c828'
XRV = BASE / 'candidate_opacity_runs/opacity_pool240_12668204'
XRV_SHA = '778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94'
BENCH = BASE / 'image_reader_consensus_runs/ricord50_12784259_002'
BENCH_SHA = 'add0c915921d0e2fa0741d604608186ce67faf83d94ad81f370eddc3c5997c29'
AUDIT = BASE / 'image_reader_consensus_audits/numeric_12784259_001'
AUDIT_SHA = '4c4e1a29cf85dc3fbe2d8d24a731b8cfd077b2d69097892bb04d0753df357ab2'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'


def sources():
    names = ('tools/annotate_candidate_image_masks.py', 'src/tricompose_v12/candidate_image_masks.py',
             'src/tricompose_v12/image_reader_consensus.py', 'src/tricompose_v12/manual_reader_risk_coverage.py',
             'src/tricompose_v12/manual_three_reader_agreement.py', 'src/tricompose_v12/manual_literal_assertions.py',
             'src/tricompose_v12/assertion_agreement_diagnostic.py', 'src/tricompose_v12/entity_gold_contract.py',
             'src/tricompose_v12/radgraph_assertion_readout.py', 'src/tricompose_v12/radgraph_reference_contract.py',
             'tests/test_candidate_image_masks.py', 'tests/test_candidate_image_masks_worker.py',
             'tools/prepare_ratescore_assets.py')
    return [WORKSPACE / 'TriCompose-v1.2' / n for n in names] + [
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
        BIOVIL / 'manifest.json', BIOVIL / 'candidate_evidence_table.csv', BIOVIL / 'image_scores.json', BIOVIL / 'summary.json',
        XRV / 'manifest.json', XRV / 'candidate_score_table.csv', XRV / 'image_scores.json',
        BENCH / 'manifest.json', BENCH / 'frozen_plan.json', BENCH / 'summary.json',
        AUDIT / 'manifest.json', AUDIT / 'audit.json', BANK]


def csv_write(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)


def execute():
    started = time.monotonic()
    require(sha256(BANK) == BANK_SHA, 'original_candidate_bank_unchanged_required')
    for root, expected, names in (
        (BIOVIL, BIOVIL_SHA, ('candidate_evidence_table.csv', 'image_scores.json', 'summary.json')),
        (XRV, XRV_SHA, ('candidate_score_table.csv', 'image_scores.json')),
        (BENCH, BENCH_SHA, ('frozen_plan.json', 'summary.json')),
        (AUDIT, AUDIT_SHA, ('audit.json',))):
        require(sha256(root / 'manifest.json') == expected, 'sealed_candidate_or_mask_parent_required')
        m = json.loads((root / 'manifest.json').read_text())
        for name in names:
            require((root / name).stat().st_size < 8 * 1024**2
                    and sha256(root / name) == m['artifacts'][name], 'bounded_sealed_cache_metadata_required')
    bm = json.loads((BIOVIL / 'manifest.json').read_text())
    require(bm['source_manifest_sha256'] == XRV_SHA, 'same_synthetic_cached_parent_required')
    bsummary = json.loads((BIOVIL / 'summary.json').read_text())
    # Parent summaries are sealed provenance; no current pixels or model assets are rehashed.
    require(bsummary['scored_images'] == 240 and bsummary['failed_images'] == 0,
            'same_complete_synthetic_image_cache_required')
    audit = json.loads((AUDIT / 'audit.json').read_text())
    plan = json.loads((BENCH / 'frozen_plan.json').read_text())
    require(audit['status'] == 'passed' and audit['completed_run_manifest_sha256'] == BENCH_SHA
            and tuple(plan['fixed_policies']) == POLICIES and plan['xrv_threshold'] == .5
            and plan['biovil_reduction'] == 'mean_all_three_predeclared'
            and plan['clinical_qualified'] is False and plan['synthetic_domain_transport_validated'] is False,
            'unchanged_audited_masks_not_clinical_calibration_required')
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in sources()}
    temporary, target = new_atomic_run(BASE / 'candidate_image_mask_overlays', 'pool960_12784259_001')
    write_json(temporary / 'frozen_plan.json', {'schema_version': 'candidate-fixed-image-mask-plan-v1',
        'pins': pins, 'fixed_policies': POLICIES, 'fixed_nested_contrasts': CONTRASTS,
        'candidate_rows': 960, 'fixed_ehrs': 80, 'image_slots': 240,
        'no_raw_body_pixel_or_current_weight_hashing': True, 'prior_sealed_provenance_not_current_pixel_reverification': True,
        'reference_metrics_or_labels_transferred': False, 'all_source_cells_and_order_preserved': True,
        'mask_only_image_states_not_report_expert_selection': True,
        'all_ehrs_fixed_including_underconditioned_cases': True, 'post_hoc_development': True,
        'clinical_primary_eligible': False, 'new_model_calls': 0, 'synthetic_transport_validated': False,
        'selection_changed': False, 'regeneration_authorized': False})
    with (BIOVIL / 'candidate_evidence_table.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (XRV / 'candidate_score_table.csv').open(newline='') as stream:
        original = list(csv.DictReader(stream))
    require(len(rows) == len(original) == 960 and len(rows[0]) == 86 and len(original[0]) == 66
            and all(all(r[k] == old[k] for k in old) for r, old in zip(rows, original)),
            'all_original_66_and_86_column_cells_required')
    x = json.loads((XRV / 'image_scores.json').read_text())['records']
    b = json.loads((BIOVIL / 'image_scores.json').read_text())['records']
    annotated, images, summary = overlay(rows, x, b)
    require((summary['candidate_rows'], summary['image_slots'], summary['fixed_ehr_cases'], summary['added_columns'])
            == (960, 240, 80, 18), 'complete_fixed_80_3_4_inventory_required')
    require(len({r['cxr_model_id'] for r in images}) == 3
            and len({r['report_model_id'] for r in rows}) == 4
            and all(sum(r['cxr_candidate_id'] == key for r in rows) == 4 for key in {r['cxr_candidate_id'] for r in rows})
            and all(sum(r['case_id'] == key for r in rows) == 12 for key in {r['case_id'] for r in rows}),
            'unchanged_three_image_four_report_grid_required')
    require((annotated, images, summary) == overlay(rows, x, b), 'deterministic_full_overlay_replay_required')
    summary.update(status='complete', reference_benchmark_manifest_sha256=BENCH_SHA,
        reference_image_metrics_not_read_or_transferred=True,
        current_image_bytes_or_weight_assets_independently_reverified=False,
        actual_existing_cpu_job=os.environ['SLURM_JOB_ID'], new_slurm_submissions=0,
        runtime_seconds=time.monotonic() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    csv_write(temporary / 'candidate_evidence_table.csv', annotated)
    write_json(temporary / 'image_mask_records.json', {'records': images})
    write_json(temporary / 'summary.json', summary)
    flat = [{k: v for k, v in r.items() if k not in ('accepted_image_states', 'status_counts', 'by_report_model')}
            for r in summary['policy_summaries']]
    csv_write(temporary / 'mask_comparison.csv', flat)
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'candidate_mask_source_changed')
    write_json(temporary / 'manifest.json', {'schema_version': 'candidate-image-mask-overlay-receipt-v1',
        'pins': pins, 'source_manifest_sha256': BIOVIL_SHA, 'benchmark_manifest_sha256': BENCH_SHA,
        'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())},
        'clinical_qualified': False, 'selection_changed': False})
    for p in [temporary, *temporary.rglob('*')]:
        require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'project_private_candidate_overlay_required')
    commit_atomic_run(temporary, target)
    return target, summary


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_candidate_mask_overlay_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS'),
                'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_candidate_mask_overlay_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'peak_rss_gib': round(result['peak_rss_gib'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_candidate_mask_overlay_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
