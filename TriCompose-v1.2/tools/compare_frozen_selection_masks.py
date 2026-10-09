"""Fixed cached-selection comparison within the already approved CPU allocation."""
import csv
import json
import os
from pathlib import Path
import resource
import sys
import time

from contracts import new_atomic_run, commit_atomic_run, discard_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json
from tricompose_v12.frozen_selection_image_masks import attach, summarize, METHODS, CAPS, POLICIES, CONTRASTS

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
OLD = BASE / 'opacity_selection_audits/selection_opacity_12666569_001'
OLD_SHA = '94889888e3af839b7ab2367bcd413d0819ce5d4ae990c926fddaf199f2c6ea62'
MASK = BASE / 'candidate_image_mask_overlays/pool960_12784259_001'
MASK_SHA = '02dc768e868ee4a40ddc9170cb1cdef4a0e6e501c1f81eba5df88105b2b06aea'
AUDIT = BASE / 'candidate_image_mask_audits/numeric_12784259_001'
AUDIT_SHA = '6d2d48fb3aa246708968c86f50ddd3e6671d9fb5777ba9a1899d5403ee16942e'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'


def sources():
    names = ('tools/compare_frozen_selection_masks.py', 'src/tricompose_v12/frozen_selection_image_masks.py',
        'src/tricompose_v12/candidate_image_masks.py', 'src/tricompose_v12/image_reader_consensus.py',
        'src/tricompose_v12/manual_reader_risk_coverage.py', 'src/tricompose_v12/manual_three_reader_agreement.py',
        'src/tricompose_v12/manual_literal_assertions.py', 'src/tricompose_v12/assertion_agreement_diagnostic.py',
        'src/tricompose_v12/entity_gold_contract.py', 'src/tricompose_v12/radgraph_assertion_readout.py',
        'src/tricompose_v12/radgraph_reference_contract.py', 'tools/prepare_ratescore_assets.py',
        'tests/test_frozen_selection_image_masks.py', 'tests/test_frozen_selection_image_masks_worker.py')
    return [WORKSPACE / 'TriCompose-v1.2' / n for n in names] + [
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py', BANK,
        OLD / 'manifest.json', OLD / 'trial_readouts.jsonl', OLD / 'summary.json',
        MASK / 'manifest.json', MASK / 'candidate_evidence_table.csv',
        AUDIT / 'manifest.json', AUDIT / 'audit.json']


def csv_write(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)


def sealed(root, expected, names):
    require(sha256(root / 'manifest.json') == expected, 'frozen_parent_manifest_required')
    m = json.loads((root / 'manifest.json').read_text())
    for name in names:
        path = root / name
        require(not path.is_symlink() and path.stat().st_size < 64 * 1024**2
                and sha256(path) == m['artifacts'][name], 'bounded_sealed_cached_metadata_required')
    return m


def execute():
    started = time.monotonic()
    require(sha256(BANK) == BANK_SHA, 'original_candidate_bank_unchanged_required')
    old = sealed(OLD, OLD_SHA, ('trial_readouts.jsonl', 'summary.json'))
    mask = sealed(MASK, MASK_SHA, ('candidate_evidence_table.csv',))
    audit = sealed(AUDIT, AUDIT_SHA, ('audit.json',))
    require(old['historical_choices_changed'] is False and old['primary_metric_eligible'] is False
            and old['regeneration_authorized'] is False and mask['clinical_qualified'] is False
            and mask['selection_changed'] is False and audit['completed_run_manifest_sha256'] == MASK_SHA,
            'unchanged_diagnostic_choices_masks_and_audit_required')
    require(json.loads((AUDIT / 'audit.json').read_text())['status'] == 'passed', 'mask_arithmetic_audit_required')
    s = json.loads((OLD / 'summary.json').read_text())
    require((s['old_trials'], s['existing_uniform_final_trials'], s['head_readouts']) == (3200, 10000, 26400),
            'same_frozen_full_trial_inventory_required')
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in sources()}
    temporary, target = new_atomic_run(BASE / 'frozen_selection_mask_comparisons', 'all_caps_12784259_001')
    try:
        write_json(temporary / 'frozen_plan.json', {'schema_version': 'frozen-selection-image-mask-plan-v1',
            'pins': pins, 'fixed_ehr_cases': 80, 'candidate_rows': 960, 'frozen_trials': 13200,
            'methods': METHODS, 'caps': CAPS, 'masks': POLICIES, 'nested_contrasts': CONTRASTS,
            'seed_weighting': 'all_frozen_replicates_within_ehr_then_all_80_ehrs',
            'old_choices_and_costs_immutable': True, 'clinical_primary_eligible': False,
            'report_opacity_proposals_qualified': False, 'ehr_opacity_reference_available': False,
            'report_assertion_v3_gate_passed': False, 'reference_error_rates_transferred': False,
            'no_raw_body_pixel_or_weight_asset_read': True, 'post_hoc_development': True,
            'best_method_mask_budget_or_seed_selected': False, 'new_selector_invocations': 0,
            'new_model_calls': 0, 'regeneration_authorized': False,
            'confidence_intervals': None,
            'interval_reason': 'descriptive_dependent_reused_artifacts_not_verified_independent_patient_clusters'})
        with (MASK / 'candidate_evidence_table.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        with (OLD / 'trial_readouts.jsonl').open() as stream:
            old_readouts = [json.loads(line) for line in stream]
        exact = [r for r in old_readouts if r['head_definition'] == 'exact_opacity_0_5']
        require(len(old_readouts) == 26400 and len(exact) == 13200 and len(rows) == 960,
                'all_cached_choices_not_favorable_subset_required')
        cases = sorted({r['case_id'] for r in rows})
        require(len(cases) == 80, 'all_80_fixed_ehrs_required')
        trials, means = attach(exact, rows, cases=cases)
        table, paired, withheld = summarize(means, cases=cases)
        require((len(trials), len(means), len(table), len(paired), len(withheld)) == (13200, 8000, 100, 100, 75),
                'complete_mask_method_budget_grid_required')
        require((trials, means) == attach(exact, rows, cases=cases), 'deterministic_cached_trial_replay_required')
        with (temporary / 'trial_mask_readouts.jsonl').open('x') as stream:
            for r in trials:
                stream.write(json.dumps(r, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        (temporary / 'trial_mask_readouts.jsonl').chmod(0o660)
        for name, records in (('case_means.csv', means), ('method_comparison.csv', table),
                              ('paired_method_comparison.csv', paired), ('mask_withholding.csv', withheld)):
            csv_write(temporary / name, records)
        summary = {'schema_version': 'frozen-selection-image-mask-comparison-v1', 'status': 'complete',
            'fixed_ehr_cases': 80, 'source_candidate_rows': 960, 'frozen_trials': 13200, 'mask_readouts': 52800,
            'case_means': len(means), 'method_comparisons': len(table), 'paired_method_comparisons': len(paired),
            'mask_withholding_comparisons': len(withheld), 'clinical_accuracy': None, 'clinical_repair_success': None,
            'clinical_qualified': False, 'new_model_calls': 0, 'new_selector_invocations': 0,
            'historical_choices_changed': False, 'all_ehrs_retained': True, 'confidence_intervals': None,
            'source_bodies_pixels_or_weights_read': False, 'new_slurm_submissions': 0,
            'existing_cpu_job': os.environ['SLURM_JOB_ID'], 'actual_gpu_savings': None,
            'runtime_seconds': time.monotonic() - started,
            'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
        write_json(temporary / 'summary.json', summary)
        require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'cached_comparison_source_changed')
        write_json(temporary / 'manifest.json', {'schema_version': 'frozen-selection-mask-comparison-receipt-v1',
            'pins': pins, 'old_selection_manifest_sha256': OLD_SHA, 'mask_manifest_sha256': MASK_SHA,
            'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
        for p in [temporary, *temporary.rglob('*')]:
            require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                    and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'private_comparison_modes_required')
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_selection_comparison_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS')
                and not os.environ.get('SLURM_STEP_GPUS'), 'actual_existing_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_frozen_selection_mask_comparison_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3),
            'peak_rss_gib': round(result['peak_rss_gib'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_frozen_selection_mask_comparison_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
