"""Existing CPU Slurm job, cached numeric image-subset controls only."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json, private_dir
from tricompose_v12.radeval_image_length_controls import join, evaluate, POLICY, SELECTORS, METRICS, CONTROLS, TARGETS

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
IMAGE = BASE / 'radeval_image_benchmark_runs/biovil_cpu_12714150_001'
IMAGE_SHA = '8e55b223ef22c5da61c20aeb0c53f818ab6c1736eeb2c95ee85b39803ea0c18a'
IMAGE_AUDIT = BASE / 'radeval_image_benchmark_audits/biovil_cpu_12714150_001/audit.json'
IMAGE_AUDIT_SHA = 'f9af2ec5a0a065e4fc92e174defc8fba474898ee26552a5ca053d98a01e1d1b1'
LENGTH = BASE / 'radeval_length_control_runs/length_controls_12766754_001'
LENGTH_SHA = '7f927552158286e755daa98da81814621580e986607f14e98b75a3266e1c2bff'
LENGTH_AUDIT = BASE / 'radeval_length_control_audits/numeric_12766754_001/audit.json'
LENGTH_AUDIT_SHA = 'c0d06267b1344dd10bc8e6ca8227f7fd374717b9e0719ccfe169b61ff5a8a72b'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
RUN_ID = 'image_length_12766754_001'
PLAN = BASE / 'radeval_image_length_control_plans' / RUN_ID
OUT = BASE / 'radeval_image_length_control_runs' / RUN_ID


def pins():
    paths = [Path(__file__).resolve(), BANK, IMAGE_AUDIT, LENGTH_AUDIT]
    require(sha256(BANK) == BANK_SHA and sha256(IMAGE_AUDIT) == IMAGE_AUDIT_SHA and
            sha256(LENGTH_AUDIT) == LENGTH_AUDIT_SHA, 'fixed_audits_and_bank_required')
    for audit, field, value in ((IMAGE_AUDIT, 'run_manifest_sha256', IMAGE_SHA),
                                (LENGTH_AUDIT, 'worker_manifest_sha256', LENGTH_SHA)):
        receipt = json.loads(audit.read_text())
        require(receipt['status'] == 'passed' and receipt[field] == value, 'independent_source_audit_required')
    for root, expected, names in ((IMAGE, IMAGE_SHA, ('paired_score_table.json', 'evaluation.json')),
                                 (LENGTH, LENGTH_SHA, ('metadata_table.json',))):
        require(not root.is_symlink() and root.stat().st_gid in (96293, 65534) and
                root.stat().st_mode & 0o7777 == 0o2770 and sha256(root / 'manifest.json') == expected,
                'frozen_private_source_manifest_required')
        artifacts = {a['path']: a['sha256'] for a in json.loads((root / 'manifest.json').read_text())['artifacts']}
        paths.append(root / 'manifest.json')
        for name in names:
            p = root / name
            require(p.is_file() and not p.is_symlink() and p.stat().st_gid in (96293, 65534) and
                p.stat().st_mode & 0o7777 == 0o660 and sha256(p) == artifacts[name], 'unchanged_private_metadata_required')
            paths.append(p)
    for relative in ('src/tricompose_v12/radeval_image_length_controls.py',
        'tests/test_radeval_image_length_controls.py', 'audits/audit_radeval_image_length_controls.py',
        'src/tricompose_v12/radeval_length_controls.py', 'src/tricompose_v12/radeval_expert.py',
        'src/tricompose_v12/radeval_image_benchmark.py', 'src/tricompose_v12/radeval_medcpt_benchmark.py',
        'src/tricompose_v12/report_metric_alignment.py', 'src/tricompose_v12/radgraph_reference_contract.py',
        'tools/prepare_ratescore_assets.py', 'audits/audit_radeval_medcpt.py'):
        paths.append(WORKSPACE / 'TriCompose-v1.2' / relative)
    return {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}


def prepare():
    frozen = pins()
    require(not OUT.exists() and not OUT.is_symlink(), 'refuse_existing_run')
    private_dir(PLAN.parent)
    private_dir(PLAN, fresh=True)
    write_json(PLAN / 'plan.json', {'schema_version': 'radeval-image-length-control-plan-v1', 'pins': frozen,
        'policy': dict(POLICY), 'selectors': list(SELECTORS), 'metrics': list(METRICS), 'controls': list(CONTROLS),
        'targets': list(TARGETS), 'bootstrap_resamples': 1000, 'seed': 0, 'all_attempted_pairs': 624,
        'original_image_available_pairs': 132, 'expected_complete_anchors': 44,
        'ranking_direction': 'original_higher_scores_and_negative_native_metadata_costs',
        'effect_direction': 'metric_selected_errors_minus_control_selected_errors',
        'same_complete_anchor_mask': True, 'no_candidate_dropping': True,
        'post_hoc_development_diagnostic': True, 'prior_image_results_already_inspected': True,
        'untouched_or_blinded_test': False, 'multiplicity_adjusted': False,
        'no_report_graph_body_image_or_embedding_decoding': True, 'new_model_calls': 0, 'cpu_job': 12766754})
    write_json(PLAN / 'manifest.json', {'schema_version': 'radeval-image-length-control-plan-receipt-v1',
        'pins': frozen, 'artifacts': [{'path': 'plan.json', 'sha256': sha256(PLAN / 'plan.json')}]})
    print(json.dumps({'status': 'image_length_control_plan_frozen', 'manifest_sha256': sha256(PLAN / 'manifest.json')}))


def execute():
    started = time.monotonic()
    require(PLAN.is_dir() and not PLAN.is_symlink() and not OUT.exists() and not OUT.is_symlink(),
            'fresh_run_and_existing_frozen_plan_required')
    receipt = json.loads((PLAN / 'manifest.json').read_text())
    require(sha256(PLAN / 'plan.json') == receipt['artifacts'][0]['sha256'], 'fixed_plan_bytes_required')
    plan = json.loads((PLAN / 'plan.json').read_text())
    require(receipt['pins'] == plan['pins'] == pins() and plan['policy'] == POLICY and
        plan['selectors'] == list(SELECTORS) and plan['targets'] == list(TARGETS) and
        plan['controls'] == list(CONTROLS) and plan['bootstrap_resamples'] == 1000 and plan['seed'] == 0,
        'immutable_declared_diagnostic_required')
    table = join(json.loads((IMAGE / 'paired_score_table.json').read_text()),
                 json.loads((LENGTH / 'metadata_table.json').read_text()))
    require(len(table) == 624 and sum(r['common_available'] for r in table) == 132 and
            all(r['common_available'] == r['original_image_cohort_available'] for r in table),
            'identical_existing_image_availability_cohort_required')
    private_dir(OUT.parent)
    private_dir(OUT, fresh=True)
    write_json(OUT / 'metadata_table.json', table)
    table_sha = sha256(OUT / 'metadata_table.json')
    write_json(OUT / 'prediction_receipt.json', {'metadata_table_sha256': table_sha,
        'all_selectors_frozen_before_new_statistics': True, 'independently_blinded': False, 'new_model_calls': 0})
    result = evaluate(table, resamples=1000, seed=0)
    require(all(r['complete_anchors'] == 44 for r in result['results']), 'identical_complete_anchor_cohort_required')
    old = json.loads((IMAGE / 'evaluation.json').read_text())
    indexed = {(r['selector'], r['target']): r for r in result['results']}
    replay_checks = 0
    for earlier in old['selection_diagnostic']:
        current = indexed[(earlier['metric'], earlier['target'])]
        require(all(current[k] == earlier[k] for k in ('means', 'attempted_anchors', 'complete_anchors',
            'error_delta_cluster_ci', 'strict_candidate_pairs', 'pairwise_accuracy', 'score_tied_strict_pairs')),
            'original_image_choice_statistics_must_replay')
        previous_corr = old['correlations'][earlier['metric']][earlier['target']]
        require(all(current['correlation'][k] == previous_corr[k] for k in
            ('spearman', 'kendall_tau_b', 'paired_rows')), 'original_image_point_correlations_must_replay')
        replay_checks += 1
    write_json(OUT / 'evaluation.json', result)
    require(sha256(OUT / 'metadata_table.json') == table_sha and pins() == plan['pins'], 'immutable_sources_and_scores_required')
    summary = {'status': 'complete', 'policy': dict(POLICY), 'all_attempted_pairs': 624,
        'common_available_pairs': 132, 'complete_anchors': 44, 'selector_target_cells': len(result['results']),
        'paired_metric_control_cells': len(result['paired_metric_control_comparisons']),
        'original_image_selector_target_cells_exactly_replayed': replay_checks,
        'new_model_calls': 0, 'new_slurm_submissions': 0,
        'raw_report_ehr_image_graph_body_or_embedding_decoded': False,
        'runtime_seconds': time.monotonic() - started, 'job_id': 12766754,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'clinical_score': None, 'selection_changed': False, 'regeneration_authorized': False}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'manifest.json', {'schema_version': 'radeval-image-length-control-run-receipt-v1',
        'plan_manifest_sha256': sha256(PLAN / 'manifest.json'), 'pins': plan['pins'],
        'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(OUT.glob('*.json'))]})
    print(json.dumps({'status': 'image_length_control_diagnostic_complete',
        'runtime_seconds': round(summary['runtime_seconds'], 3), 'peak_rss_gib': round(summary['peak_rss_gib'], 3),
        'manifest_sha256': sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('mode', choices=('prepare', 'run'))
        args = parser.parse_args()
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ.get('SLURM_JOB_ID') == '12766754', 'actual_existing_cpu_job_required')
        os.umask(0o007)
        prepare() if args.mode == 'prepare' else execute()
    except Exception as error:
        print(json.dumps({'status': 'image_length_control_diagnostic_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
