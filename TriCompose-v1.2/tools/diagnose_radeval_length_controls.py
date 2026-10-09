"""Existing-job output-only control experiment; no text/graphs/models decoded."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json, private_dir
from tricompose_v12.radeval_length_controls import build_table, evaluate, POLICY, SELECTORS, CONTROLS

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
LITERAL = BASE / 'radeval_literal_fact_runs/literal_expert_12766754_001'
LITERAL_SHA = '0a05b720644c9c1ece2437a8cde183e5e9ff93f1d2c7309ed8f834effb22a782'
MED = BASE / 'radeval_medcpt_runs/reportref_12714150_001'
MED_SHA = '7005c06b9882317fc8b4148cb06e525c78c07623370e191e5bde2ce88cc8d754'
GRAPH = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
GRAPH_SHA = '21e1724a5b364eee97b56cdb8f66c0d56194baadc98c333b5c473f7efb37038d'
INVENTORY = BASE / 'report_metric_sources/radeval_expert_contract_12714150_003/inventory.json'
INVENTORY_SHA = '29c076bc24fd4c628703cbe1a1ea5049cf090660ff06a2c46e09e59e50b49365'
AUDIT = BASE / 'radeval_literal_fact_audits/numeric_12766754_001/audit.json'
AUDIT_SHA = '481062efef6a4016c4cf0f3356927f576dce72e4916832d339740c4573c6b940'
OLD_BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
OLD_BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
RUN_ID = 'length_controls_12766754_001'
PLAN = BASE / 'radeval_length_control_plans' / RUN_ID
OUT = BASE / 'radeval_length_control_runs' / RUN_ID


def pins():
    require(sha256(INVENTORY) == INVENTORY_SHA and sha256(AUDIT) == AUDIT_SHA
            and sha256(OLD_BANK) == OLD_BANK_SHA, 'fixed_inventory_audit_and_old_bank_required')
    audit = json.loads(AUDIT.read_text())
    require(audit['status'] == 'passed' and audit['worker_manifest_sha256'] == LITERAL_SHA,
            'independently_audited_literal_source_required')
    paths = [Path(__file__).resolve(), INVENTORY, AUDIT, OLD_BANK,
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_length_controls.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_literal_facts.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_medcpt_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/report_metric_alignment.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/fact_comparison_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_literal_evidence.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_radeval_length_controls.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/65_radeval_length_controls_existing_cpu.sh']
    for root, expected, names in (
        (LITERAL, LITERAL_SHA, ('derived_table.json', 'evaluation.json')),
        (MED, MED_SHA, ('token_receipts.json',)),
        (GRAPH, GRAPH_SHA, ('graph_receipts.json',))):
        require(sha256(root / 'manifest.json') == expected, 'frozen_complete_source_manifest_required')
        manifest = json.loads((root / 'manifest.json').read_text())
        artifacts = {a['path']: a['sha256'] for a in manifest['artifacts']}
        paths.append(root / 'manifest.json')
        for name in names:
            p = root / name
            require(p.is_file() and not p.is_symlink() and sha256(p) == artifacts[name], 'unchanged_derived_metadata_required')
            paths.append(p)
    return {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}


def prepare():
    frozen = pins()
    require(not OUT.exists() and not OUT.is_symlink(), 'refuse_existing_run')
    private_dir(PLAN.parent)
    private_dir(PLAN, fresh=True)
    write_json(PLAN / 'plan.json', {'schema_version': 'radeval-length-control-plan-v1', 'pins': frozen,
        'policy': dict(POLICY), 'selectors': list(SELECTORS), 'controls': list(CONTROLS),
        'bootstrap_resamples': 1000, 'seed': 0, 'all_attempted_pairs': 624,
        'ranking_direction': 'minimize_each_separate_count_or_length_cost',
        'paired_effect_direction': 'literal_selected_errors_minus_control_selected_errors',
        'same_complete_anchor_mask_for_all_selectors': True, 'previous_best_literal_feature_known': True,
        'post_hoc_development_diagnostic': True, 'untouched_or_blinded_test': False,
        'no_report_graph_body_or_image_decoding': True, 'new_model_calls': 0, 'cpu_job': 12766754})
    write_json(PLAN / 'manifest.json', {'schema_version': 'radeval-length-control-plan-receipt-v1',
        'pins': frozen, 'artifacts': [{'path': 'plan.json', 'sha256': sha256(PLAN / 'plan.json')}]})
    print(json.dumps({'status': 'length_control_diagnostic_prepared', 'manifest_sha256': sha256(PLAN / 'manifest.json')}))


def execute():
    started = time.monotonic()
    require(PLAN.is_dir() and not PLAN.is_symlink() and not OUT.exists() and not OUT.is_symlink(),
            'fresh_run_and_existing_frozen_plan_required')
    plan_manifest = json.loads((PLAN / 'manifest.json').read_text())
    require(sha256(PLAN / 'plan.json') == plan_manifest['artifacts'][0]['sha256'], 'fixed_plan_bytes_required')
    frozen = json.loads((PLAN / 'plan.json').read_text())
    require(frozen['pins'] == plan_manifest['pins'] == pins() and frozen['policy'] == POLICY
        and frozen['selectors'] == list(SELECTORS) and frozen['controls'] == list(CONTROLS)
        and frozen['bootstrap_resamples'] == 1000 and frozen['seed'] == 0, 'immutable_declared_controls_required')
    private_dir(OUT.parent)
    private_dir(OUT, fresh=True)
    table = build_table(json.loads(INVENTORY.read_text()), json.loads((LITERAL / 'derived_table.json').read_text()),
        json.loads((MED / 'token_receipts.json').read_text()), json.loads((GRAPH / 'graph_receipts.json').read_text()))
    require(len(table) == 624 and all(r['common_available'] for r in table), 'known_complete_cached_metadata_cohort_required')
    write_json(OUT / 'metadata_table.json', table)
    table_sha = sha256(OUT / 'metadata_table.json')
    write_json(OUT / 'prediction_receipt.json', {'metadata_table_sha256': table_sha,
        'costs_frozen_before_new_statistics': True, 'independently_blinded': False, 'new_model_calls': 0})
    result = evaluate(table, resamples=1000, seed=0)
    old = json.loads((LITERAL / 'evaluation.json').read_text())
    current = {(r['selector'], r['target']): r for r in result['results']}
    replay_checks = 0
    for earlier in old['results']:
        r = current[(earlier['feature'], earlier['target'])]
        require(r['correlation'] == earlier['correlation'] and
            r['means'] == earlier['within_anchor_diagnostic']['means'] and
            r['anchor_rows'] == earlier['within_anchor_diagnostic']['anchor_rows'],
            'all_original_literal_statistic_cells_must_replay')
        replay_checks += 1
    write_json(OUT / 'evaluation.json', result)
    require(sha256(OUT / 'metadata_table.json') == table_sha and pins() == frozen['pins'], 'inputs_and_costs_unchanged')
    summary = {'status': 'complete', 'policy': dict(POLICY), 'all_attempted_pairs': len(table),
        'common_available_pairs': result['common_available_pairs'], 'selector_target_cells': len(result['results']),
        'paired_literal_control_cells': len(result['paired_literal_control_comparisons']),
        'original_literal_cells_exactly_replayed': replay_checks,
        'zero_native_entity_hypothesis_pairs': sum(r['native_entity_count'] == 0 for r in table),
        'new_model_calls': 0, 'new_slurm_submissions': 0, 'raw_report_ehr_image_or_native_graph_body_decoded': False,
        'runtime_seconds': time.monotonic() - started, 'job_id': 12766754,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'selection_changed': False, 'clinical_score': None, 'regeneration_authorized': False}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'manifest.json', {'schema_version': 'radeval-length-control-run-receipt-v1',
        'plan_manifest_sha256': sha256(PLAN / 'manifest.json'), 'pins': frozen['pins'],
        'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(OUT.glob('*.json'))]})
    print(json.dumps({'status': 'length_control_diagnostic_complete',
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
        print(json.dumps({'status': 'length_control_diagnostic_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
