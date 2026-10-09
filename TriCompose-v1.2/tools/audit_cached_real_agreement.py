"""Existing Slurm CPU: frozen real-manual count audit, no source/model access.

Opens only explicitly pinned aggregate summaries and sanitized opaque prediction
receipts. Does not follow their original source paths or open annotation rows,
reports, images, native graphs, credentials, model weights or network clients.
"""
from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, require_slurm, sha256, write_json
from tricompose_v12.cached_manual_agreement import FINDINGS, READERS, STATES, POLICIES, evaluate

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
CACHED = {
    'chexbert': (BASE / 'real_validation/official_chexbert_reports_12594397',
        'aeea0df50e73c4b57af35d3d686ee5d3677f1663174cf28a570205f5248deec6'),
    'qwen_span_v2': (BASE / 'real_validation/official_span_v2_12645961',
        'b9274424e49301b8e41e147716a0e751cf3743fc5f1a032f91a4dc1b7baf4b5b'),
}
AUDIT = BASE / 'real_validation/official_report_gold_coverage_12576792_source1'
AUDIT_SHA = 'bf0b5064c5a477bc673196080e2c16b2dc7dc15672f6cf589161761ddfe77fc4'
BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'


def cached(root, expected):
    require(sha256(root / 'manifest.json') == expected, 'pinned_completed_cache_required')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in ('predictions.json', 'summary.json'):
        require(sha256(root / name) == manifest['artifacts'][name]['sha256'],
                'immutable_cached_artifact_required')
    return manifest, json.loads((root / 'summary.json').read_text()), json.loads(
        (root / 'predictions.json').read_text())['records']


def report(result):
    lines = ['# Cached real-manual reader agreement / 缓存真实人工标签一致规则审计', '',
        '15 shared study reports, four fixed heads, 60 checks; 26 explicit positive/negative references.',
        '687 annotation entries retained: 267 unlinked, 393 non-test, 12 strict-Impression rejections.',
        'This reused development resource is not an untouched final test or image/EHR adjudication.', '',
        '| Fixed mask | Determinate proposals / 60 | Known-label correct / 26 | Hard polarity flips | Unknown-reference determinate outputs |',
        '|---|---:|---:|---:|---:|']
    for name, item in result['policies'].items():
        r = item['overall']
        lines.append(f"| {name} | {r['accepted_determinate_proposals']}/60 | "
            f"{r['correct_known_reference_proposals']}/26 | {r['hard_positive_negative_flips']} | "
            f"{r['determinate_on_unknown_reference']} |")
    lines += ['', '## Interpretation / 解释', '',
        'Unknown-reference promotions are differences from annotation policy, NOT adjudicated clinical hallucinations.',
        'Agreement on unknown/uncertain is abstention, not factual support. All sixty checks remain the coverage denominator.',
        'The pair drops the two Qwen known-label flips, but cannot improve on this already-perfect CheXbert known-label baseline; it loses six known assertions.',
        'This is not evidence that extra reader agreement is universally superior or that a report/CXR is the faulty modality.', '',
        'Joint reference counts are certified only by the pinned CheXbert identity confusion on this exact shared cohort,',
        'checked against the complete cached two-reader joint matrix. No per-record reference key is reconstructed/exported.',
        'For a nonidentity baseline, aggregate marginal confusions cannot identify pair accuracy: the implementation returns null.', '',
        'No uncertain reference examples occur in these four heads; zero observed accepted errors is not zero clinical risk.',
        'Checkpoint overlap, temporal/span scope and independent image/EHR truth remain unresolved.',
        'No thresholds/policy priorities were fitted, no best policy was selected, no winner/score was changed and no repair was authorized.',
        '本轮没有新模型推理、GPU、Slurm 提交、下载、API，也没有重读真实报告或原始标签行。', '']
    return '\n'.join(lines)


def execute(run):
    started = time.monotonic()
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/cached_manual_agreement.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_cached_manual_agreement.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/assertion_agreement_diagnostic.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'docs/official_report_span_protocol.md', BANK,
        AUDIT / 'manifest.json', AUDIT / 'summary.json']
    paths.extend(root / name for root, _ in CACHED.values()
                 for name in ('manifest.json', 'summary.json', 'predictions.json'))
    pins = {str(path.relative_to(WORKSPACE)): sha256(path) for path in paths}
    require(sha256(BANK) == BANK_SHA and sha256(AUDIT / 'manifest.json') == AUDIT_SHA,
            'unchanged_bank_and_reference_metadata_required')
    audit_manifest = json.loads((AUDIT / 'manifest.json').read_text())
    require(sha256(AUDIT / 'summary.json') == audit_manifest['artifacts']['summary.json']['sha256'],
            'pinned_reference_metadata_summary_required')
    audit = json.loads((AUDIT / 'summary.json').read_text())
    write_json(run / 'frozen_plan.json', {'schema_version': 'cached-real-agreement-count-plan-v1',
        'pins': pins, 'fixed_readers': READERS, 'fixed_policies': POLICIES,
        'annotation_inventory': 687, 'shared_completed_reports': 15, 'four_head_checks': 60,
        'new_model_calls': 0, 'source_rows_reports_or_images_read': False,
        'use_cached_aggregate_confusions_not_new_reference_labels': True,
        'joint_label_counts_require_exact_cohort_identity_proof': True,
        'nonidentifiable_joint_label_counts_remain_null': True,
        'post_hoc_development_analysis': True, 'best_policy_selection': False,
        'selection_changed': False, 'regeneration_authorized': False})
    manifests, summaries, predictions = {}, {}, {}
    for reader, (root, expected) in CACHED.items():
        manifests[reader], summaries[reader], predictions[reader] = cached(root, expected)
        s = summaries[reader]
        require(s['annotation_inventory'] == 687 and s['completed_reports'] == 15
                and s['primary_metric_eligible'] is False and s['selection_changed'] is False
                and s['regeneration_authorized'] is False, 'fixed_manual_development_scope_required')
    require(manifests['chexbert']['source_sha256'] == audit['source_sha256'],
            'same_original_reference_metadata_bindings_required')
    qm = manifests['qwen_span_v2']
    require(qm['source_sha256']['audit_manifest'] == AUDIT_SHA
            and qm['source_sha256']['audit_summary'] == sha256(AUDIT / 'summary.json')
            and qm['source_sha256']['chexbert_predictions'] == sha256(CACHED['chexbert'][0] / 'predictions.json'),
            'shared_reference_and_paired_baseline_provenance_required')
    for finding in FINDINGS:
        for reader, paired_name in (('chexbert', 'chexbert'), ('qwen_span_v2', 'qwen_v2')):
            require(summaries[reader]['per_finding'][finding] ==
                    summaries['qwen_span_v2']['paired_comparison'][paired_name][finding],
                    'full_completed_and_paired_confusions_identical_required')
    confusions = {name: {f: summaries[name]['per_finding'][f]['confusion_matrix'] for f in FINDINGS}
                  for name in READERS}
    result = evaluate(predictions, confusions)
    require(result['annotation_inventory'] == 687 and result['shared_complete_reports'] == 15,
            'all_frozen_entries_and_ready_reports_required')
    require(all(item['overall']['known_positive_negative_reference_checks'] == 26
                and item['overall']['unknown_reference_checks'] == 34
                and item['overall']['uncertain_reference_checks'] == 0
                for item in result['policies'].values()), 'fixed_manual_reference_support_required')
    # Independent arithmetic replay, without calling the mask decision helper.
    indexed = {name: {r['item_id']: r for r in records} for name, records in predictions.items()}
    ids = sorted(k for k, row in indexed['chexbert'].items() if row['status'] == 'complete')
    replay = 0
    for policy, readers in POLICIES.items():
        for finding in FINDINGS:
            accepted = 0
            for key in ids:
                values = [indexed[name][key]['finding_states'][finding] for name in readers]
                accepted += len(set(values)) == 1 and values[0] in ('positive', 'negative')
                replay += 1
            recorded = result['policies'][policy]['per_finding'][finding]
            require(recorded['accepted_determinate_proposals'] == accepted,
                    'independent_accepted_mask_count_mismatch')
            if len(readers) == 1:
                cm = confusions[readers[0]][finding]
                flips = cm['positive']['negative'] + cm['negative']['positive']
                correct = cm['positive']['positive'] + cm['negative']['negative']
            else:
                require(result['identity_count_proof_by_finding'][finding],
                        'current_cache_identity_count_proof_required')
                cm = confusions['qwen_span_v2'][finding]
                flips, correct = 0, cm['positive']['positive'] + cm['negative']['negative']
            require(recorded['hard_positive_negative_flips'] == flips
                    and recorded['correct_known_reference_proposals'] == correct,
                    'independent_reference_count_replay_mismatch')
    require(all(sha256(WORKSPACE / path) == value for path, value in pins.items()),
            'consumed_cache_or_program_changed')
    write_json(run / 'evaluation.json', result)
    summary = {'schema_version': 'cached-real-agreement-count-run-v1', 'status': 'complete',
        'new_model_calls': 0, 'raw_source_rows_reports_images_read': False,
        'per_record_gold_reconstructed': False, 'policies_evaluated': len(POLICIES),
        'independently_replayed_mask_checks': replay, 'clinical_qualified': False,
        'best_policy_selected': False, 'selection_changed': False,
        'regeneration_authorized': False, 'thresholds_fitted': False,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
    write_json(run / 'summary.json', summary)
    with (run / 'RESULTS_CN_EN.md').open('x', encoding='utf-8') as stream:
        stream.write(report(result))
        stream.flush()
        os.fsync(stream.fileno())
    (run / 'RESULTS_CN_EN.md').chmod(0o660)
    write_json(run / 'manifest.json', {'schema_version': 'cached-real-agreement-count-receipt-v1',
        'pins': pins, 'reference_source_sha256': audit['source_sha256'],
        'artifacts': {p.name: sha256(p) for p in sorted(run.iterdir())
                      if p.suffix in ('.json', '.md')},
        'raw_source_rows_reports_images_read': False, 'per_record_gold_reconstructed': False})
    return summary


def main():
    run = None
    try:
        require(len(sys.argv) == 1, 'fixed_cache_worker_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12766754', 'approved_existing_cpu_allocation_required')
        os.umask(0o007)
        parent = BASE / 'real_agreement_audits'
        private_dir(parent)
        target = parent / 'cached_manual15_12766754_001'
        private_dir(target, fresh=True)
        run = target
        with (run / 'worker.log').open('x') as log:
            (run / 'worker.log').chmod(0o660)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = execute(run)
        print(json.dumps({'status': 'protected_cached_real_agreement_complete',
            'runtime_seconds': round(summary['runtime_seconds'], 3),
            'peak_rss_gib': round(summary['peak_rss_gib'], 3),
            'manifest_sha256': sha256(run / 'manifest.json')}))
        return 0
    except Exception as error:
        if run is not None and not (run / 'manifest.json').exists():
            write_json(run / 'failure.json', {'status': 'failed', 'failure_type': type(error).__name__})
        print(json.dumps({'status': 'protected_cached_real_agreement_failed',
                          'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
