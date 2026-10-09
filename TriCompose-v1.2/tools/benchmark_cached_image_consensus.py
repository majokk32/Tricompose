"""Existing-CPU cached image-reference benchmark; zero pixel/model access.

Sealed XRV/BioViL outputs on the fixed RICORD 50 case inventory are reused.
Do not call source_receipt(): its historical pixel-file hashing is unnecessary
here. Recorded input hashes are compared, not current source pixels opened.
"""
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from contracts import new_atomic_run, commit_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json
from tricompose_v12.image_reader_consensus import analyze, POLICIES, CONTRASTS

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
XRV = BASE / 'real_validation/ricord_xrv_pilots/ricord_xrv50_12667524'
XRV_SHA = 'dc14a2ba1aed3823f0752a73dda297e874f638b31644d2992227eb3f1cf697a9'
BIOVIL = BASE / 'real_validation/ricord_biovil_pilots/ricord_biovil50_12669671'
BIOVIL_SHA = 'd50e5594e43f7979f385f0e14745d66103714b16693e7656ddcf7812e85480ef'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')


def metadata(root, expected):
    require(sha256(root / 'manifest.json') == expected, 'sealed_image_score_run_required')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in ('scores.json', 'summary.json'):
        require(sha256(root / name) == manifest['artifacts'][name]
                and (root / name).stat().st_size < 4 * 1024**2, 'bounded_sealed_score_metadata_required')
    for group in ('sources', 'dependency_sources'):
        for name, value in manifest.get(group, {}).items():
            path = Path(name) if Path(name).is_absolute() else WORKSPACE / name
            require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == value, 'consumed_source_changed')
    summary = json.loads((root / 'summary.json').read_text())
    require(summary['planned_cases'] == summary['scored_cases'] == 50 and summary['failed_cases'] == 0
            and summary['patient_grouping_verified'] is True and summary['frozen'] is True
            and summary['probability_semantics'] is False and summary['primary_metric_eligible'] is False
            and summary['official_adjudication_reproduced'] is False,
            'same_complete_unqualified_fifty_patient_diagnostic_required')
    return manifest, summary


def cached_join(xrv, biovil, xrv_manifest):
    ids = {f'case_{i:03d}' for i in range(50)}
    xs = {r['case_id']: r for r in xrv['records']}
    bs = {r['case_id']: r for r in biovil['records']}
    require(len(xrv['records']) == len(biovil['records']) == 50 and set(xs) == set(bs) == ids,
            'all_original_fifty_score_records_required')
    require(Counter(r['reference_state'] for r in xs.values()) == {'positive': 25, 'negative': 25},
            'same_derived_reference_denominators_required')
    for scores in (xrv, biovil):
        require(len(scores['outcomes']) == 50 and {r['case_id'] for r in scores['outcomes']} == ids
                and all(r['status'] == 'scored' for r in scores['outcomes']), 'complete_primary_attempt_receipts_required')
    joined = []
    for cid in sorted(ids):
        x, b = xs[cid], bs[cid]
        image_hash = x['display_sha256']
        require(image_hash == b['source_image_sha256'] == xrv_manifest['artifacts']['displays/' + cid + '.png'],
                'same_previously_encoded_display_hash_required')
        pairs = b['score_pairs']
        require(set(pairs) == set(FAMILIES), 'all_three_fixed_templates_required')
        margins = []
        for family in FAMILIES:
            pair = pairs[family]
            require(set(pair) == {'positive_cosine', 'negative_cosine'}
                    and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 1.01 for v in pair.values()),
                    'finite_frozen_cosine_pairs_required')
            margins.append(pair['positive_cosine'] - pair['negative_cosine'])
        joined.append({'case_id': cid, 'image_sha256': image_hash, 'reference_state': x['reference_state'],
                       'xrv_score': x['direct_lung_opacity_score'], 'margins': margins})
    return joined


def csv_write(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)


def tables(result):
    table, pairs = [], []
    for row in result['rows']:
        c, m = row['counts'], row['metrics']
        risk = m['accepted_error_risk']
        ci = risk['percentile_interval']
        table.append({'policy': row['policy'], **c,
            'error_risk_numerator': risk['numerator'], 'error_risk_denominator': risk['denominator'],
            'accepted_error_risk': risk['estimate'], 'risk_interval_lower': ci[0] if ci else None,
            'risk_interval_upper': ci[1] if ci else None, 'risk_valid_draws': risk['valid_draws'],
            'risk_zero_denominator_draws': risk['zero_denominator_draws'],
            'proposal_coverage': m['proposal_coverage']['estimate'],
            'positive_reference_recovery': m['positive_reference_recovery']['estimate'],
            'negative_reference_recovery': m['negative_reference_recovery']['estimate'],
            'conditional_positive_recovery': m['conditional_positive_recovery']['estimate'],
            'conditional_negative_recovery': m['conditional_negative_recovery']['estimate'],
            'clinical_qualified': False, 'official_adjudication_reproduced': False})
    for row in result['paired_contrasts']:
        for name, value in row['metrics'].items():
            ci = value['percentile_interval']
            pairs.append({'left': row['left'], 'right': row['right'], 'metric': name,
                'right_minus_left': value['right_minus_left'], 'lower': ci[0] if ci else None,
                'upper': ci[1] if ci else None, 'valid_draws': value['valid_draws'],
                'zero_denominator_draws': value['zero_denominator_draws'], 'shared_paired_draws': True})
    return table, pairs


def execute():
    started = time.monotonic()
    xm, xs = metadata(XRV, XRV_SHA)
    bm, bs = metadata(BIOVIL, BIOVIL_SHA)
    require(bm['source_manifest_sha256'] == XRV_SHA and xs['reference_kind'] == bs['reference_kind']
            and bs['primary_reduction'] == 'mean_all_three_predeclared'
            and xs['thresholds_fitted_on_this_cohort'] is False and bs['thresholds_fitted'] is False
            and bs['templates_fitted'] is False and bs['selection_changed'] is False,
            'unchanged_reference_threshold_and_template_scope_required')
    require(sha256(BANK) == BANK_SHA, 'original_synthetic_bank_unchanged_required')
    sources = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/image_reader_consensus.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_reader_risk_coverage.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_three_reader_agreement.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/manual_reader_agreement.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_image_reader_consensus.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_cached_image_consensus_worker.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py',
        XRV / 'manifest.json', XRV / 'scores.json', XRV / 'summary.json',
        BIOVIL / 'manifest.json', BIOVIL / 'scores.json', BIOVIL / 'summary.json', BANK]
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in sources}
    temporary, target = new_atomic_run(BASE / 'image_reader_consensus_runs', 'ricord50_12784259_001')
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
        'selection_changed': False, 'regeneration_authorized': False})
    joined = cached_join(json.loads((XRV / 'scores.json').read_text()),
                         json.loads((BIOVIL / 'scores.json').read_text()), xm)
    result = analyze(joined)
    require(result == analyze(joined), 'exact_bootstrap_and_counter_replay_required')
    original = xs['metrics']['direct_lung_opacity_default']
    counts = result['rows'][0]['counts']
    # Reproduce the historical primary confusion, not a post-hoc head/proxy choice.
    require([counts[f] for f in ('true_positive', 'false_negative', 'true_negative', 'false_positive')]
            == [25, 0, 19, 6], 'historical_exact_head_confusion_must_remain_identical')
    require(original['sensitivity'] == 1 and original['specificity'] == .76,
            'original_fixed_head_summary_unchanged_required')
    table, pairs = tables(result)
    require((len(table), len(pairs), len(result['outcomes'])) == (4, 21, 200), 'all_fixed_policies_and_cases_required')
    write_json(temporary / 'joined_score_metadata.json', {'records': joined})
    write_json(temporary / 'evaluation.json', {k: v for k, v in result.items() if k != 'outcomes'})
    write_json(temporary / 'policy_outcomes.json', {'records': result['outcomes']})
    csv_write(temporary / 'risk_coverage_table.csv', table)
    csv_write(temporary / 'paired_contrasts.csv', pairs)
    require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'consumed_cache_or_program_changed')
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
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'private_project_artifacts_required')
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
            'runtime_seconds': round(result['runtime_seconds'], 3), 'peak_rss_gib': round(result['peak_rss_gib'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_image_consensus_diagnostic_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
