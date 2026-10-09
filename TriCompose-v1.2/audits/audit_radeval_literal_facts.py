"""Independent stdlib numerical audit; no production scoring imports or text IO.

Reuses the independently written MedCPT auditor's rank/statistic utilities,
not the benchmark's implementations. Native extraction itself is not rerun.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from audit_radeval_medcpt import sha, load, close, check, ranks, rho, tau, bootstrap_correlation, percentile
import audit_radeval_medcpt as independent_statistics

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'radeval_literal_fact_runs/literal_expert_12766754_001'
PLAN = BASE / 'radeval_literal_fact_plans/literal_expert_12766754_001'
INVENTORY = BASE / 'report_metric_sources/radeval_expert_contract_12714150_003/inventory.json'
OUT = BASE / 'radeval_literal_fact_audits/numeric_12766754_001'
FEATURES = ('polarity_opposition_proposals', 'different_anatomy_context_concepts',
    'modifier_token_set_differences', 'hypothesis_only_literal_atoms',
    'reference_only_literal_atoms', 'uncertain_or_mixed_literal_atoms')
CATEGORIES = ('false_prediction', 'omission', 'incorrect_location', 'incorrect_severity',
              'unsupported_comparison', 'omitted_change', 'inarticulate_report')


def audit(expected):
    started = time.monotonic()
    check('/job_12766754/' in Path('/proc/self/cgroup').read_text())
    check(sha(RUN / 'manifest.json') == expected and not OUT.exists() and not OUT.is_symlink())
    manifest = load(RUN / 'manifest.json')
    for name, value in manifest['pins'].items():
        check(sha(WORKSPACE / name) == value)
    for a in manifest['artifacts']:
        check(Path(a['path']).name == a['path'] and sha(RUN / a['path']) == a['sha256'])
    check(sha(PLAN / 'manifest.json') == manifest['plan_manifest_sha256'])
    plan_receipt = load(PLAN / 'manifest.json')
    check(plan_receipt['pins'] == manifest['pins'] and
          sha(PLAN / 'plan.json') == plan_receipt['artifacts'][0]['sha256'])
    predictions = load(RUN / 'predictions.json')
    records = load(RUN / 'derived_table.json')
    evaluation = load(RUN / 'evaluation.json')
    summary = load(RUN / 'summary.json')
    source = load(INVENTORY)
    check(len(source['records']) == len(predictions) == len(records) == 624)
    check(load(RUN / 'prediction_receipt.json')['predictions_sha256'] == sha(RUN / 'predictions.json'))
    feature_checks = 0
    for original, pred, row in zip(source['records'], predictions, records):
        check(original['item_id'] == pred['item_id'] == row['item_id'])
        for key in ('source_id', 'source_group_id', 'section_id', 'section_name', 'candidate_slot',
                    'reference_sha256', 'hypothesis_sha256'):
            check(original[key] == row[key])
        check(row['status'] == pred['status'])
        if pred['status'] == 'complete':
            c = pred['comparison']
            payload = {k: v for k, v in c.items() if k != 'comparison_id'}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            check(c['comparison_id'] == 'comparison_' + digest and c['clinical_score'] is None and
                  c['confirmed_faulty_modality'] is None and all(v is False for v in c['policy'].values()))
            check(c['left_source_artifact_sha256'] == original['hypothesis_sha256'] and
                  c['right_source_artifact_sha256'] == original['reference_sha256'])
            counts = Counter(d['relation'] for d in c['presence']['details'])
            check(all(d['clinical_contradiction_verified'] is False for d in c['presence']['details']))
            check(all(counts[key] == value for key, value in c['presence']['counts'].items()))
            values = (counts['explicit_polarity_opposition_proposal'], c['anatomy']['different_context_concepts'],
                      c['modifier_difference_not_severity'], counts['unmentioned_in_right'],
                      counts['unmentioned_in_left'], counts['uncertain_or_mixed_state'] + counts['anatomy_context_not_definite'])
            for feature, value in zip(FEATURES, values):
                check(row['features'][feature] == value)
                feature_checks += 1
        else:
            check(pred['comparison'] is None and all(v is None for v in row['features'].values()))
        errors, totals = {}, []
        for severity, values in original['errors'].items():
            total = sum(values) if None not in values else None
            totals.append(total)
            errors[severity + '_total'] = total
            errors.update({severity + '_' + category: value for category, value in zip(CATEGORIES, values)})
        errors['all_errors_total'] = sum(totals) if None not in totals else None
        check(errors == row['expert_outcomes'])
    anchors = defaultdict(list)
    for row in records:
        anchors[(row['source_id'], row['section_id'], row['reference_sha256'])].append(row)
    point_checks = binary_checks = ci_checks = anchor_checks = 0
    for item in evaluation['results']:
        feature, target = item['feature'], item['target']
        available = [r for r in records if r['status'] == 'complete' and r['expert_outcomes'][target] is not None]
        values = [(r['features'][feature], r['expert_outcomes'][target]) for r in available]
        corr = item['correlation']
        check(corr['paired_rows'] == len(values))
        close(corr['spearman'], rho(values))
        close(corr['kendall_tau_b'], tau(values))
        close(item['paired_coverage'], len(values) / 624)
        point_checks += 1
        flags = Counter((x > 0, y > 0) for x, y in values)
        binary = item['binary_diagnostic']
        for name, key in (('tp', (True, True)), ('fp', (True, False)), ('fn', (False, True)), ('tn', (False, False))):
            check(binary[name] == flags[key])
        tp, fp, fn, tn = [binary[k] for k in ('tp', 'fp', 'fn', 'tn')]
        close(binary['precision'], tp / (tp + fp) if tp + fp else None)
        close(binary['recall'], tp / (tp + fn) if tp + fn else None)
        close(binary['specificity'], tn / (tn + fp) if tn + fp else None)
        positive, negative = sum(y > 0 for _, y in values), sum(y == 0 for _, y in values)
        rr = ranks([x for x, _ in values])
        auc = ((sum(rank for rank, (_, y) in zip(rr, values) if y > 0) - positive * (positive + 1) / 2) /
               (positive * negative)) if positive and negative else None
        close(binary['count_auroc'], auc)
        binary_checks += 1
        if 'cluster_bootstrap' in item:
            interval, usable, groups = bootstrap_correlation([(r['source_group_id'], r['features'][feature], r['expert_outcomes'][target]) for r in available])
            ci = item['cluster_bootstrap']
            check(ci['usable_resamples'] == usable and ci['source_groups'] == groups)
            if interval is None:
                check(ci['spearman_interval_95'] is None)
            else:
                for a, b in zip(interval, ci['spearman_interval_95']):
                    close(a, b)
            ci_checks += 1
        ranking = item['within_anchor_diagnostic']
        check(len(ranking['anchor_rows']) == len(anchors) == ranking['attempted_anchors'])
        eligible_anchors = []
        for saved, candidates in zip(ranking['anchor_rows'], anchors.values()):
            candidates = sorted(candidates, key=lambda r: r['candidate_slot'])
            check([r['candidate_slot'] for r in candidates] == [1, 2, 3])
            complete = all(r['status'] == 'complete' and r['expert_outcomes'][target] is not None for r in candidates)
            check(saved['status'] == ('complete' if complete else 'incomplete_anchor'))
            if not complete:
                check(saved['selected_expected_errors'] is None)
                continue
            counts = [r['features'][feature] for r in candidates]
            errors = [r['expert_outcomes'][target] for r in candidates]
            top = [i for i, c in enumerate(counts) if c == min(counts)]
            selected, random_mean = sum(errors[i] for i in top) / len(top), sum(errors) / 3
            check(saved['top_tie_size'] == len(top))
            close(saved['selected_expected_errors'], selected)
            close(saved['random_expected_errors'], random_mean)
            close(saved['oracle_errors'], min(errors))
            close(saved['metric_minus_random_errors'], selected - random_mean)
            strict = correct = ties = 0
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if errors[i] == errors[j]:
                    continue
                strict += 1
                if counts[i] == counts[j]:
                    correct += .5
                    ties += 1
                else:
                    correct += int((counts[i] < counts[j]) == (errors[i] < errors[j]))
            check(saved['strict_pairs'] == strict and saved['expected_correct_pairs'] == correct and saved['score_tied_strict_pairs'] == ties)
            eligible_anchors.append(saved)
            anchor_checks += 1
        check(ranking['complete_anchors'] == len(eligible_anchors))
        for field, value in ranking['means'].items():
            close(value, sum(r[field] for r in eligible_anchors) / len(eligible_anchors) if eligible_anchors else None)
        strict = sum(r['strict_pairs'] for r in eligible_anchors)
        check(ranking['strict_candidate_pairs'] == strict)
        close(ranking['expected_pairwise_accuracy'], sum(r['expected_correct_pairs'] for r in eligible_anchors) / strict if strict else None)
        if 'error_delta_cluster_ci' in ranking:
            groups = defaultdict(list)
            for a in eligible_anchors:
                groups[a['source_group_id']].append(a['metric_minus_random_errors'])
            keys, rng, boot = sorted(groups), random.Random(0), []
            if len(keys) >= 3:
                for _ in range(1000):
                    data = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
                    boot.append(sum(data) / len(data))
            interval = ranking['error_delta_cluster_ci']['interval95']
            if boot:
                for a, b in zip(interval, (percentile(boot, .025), percentile(boot, .975))):
                    close(a, b)
            else:
                check(interval is None)
            ci_checks += 1
    check(point_checks == binary_checks == 102 and all(v is False for v in evaluation['policy'].values()))
    check(summary['new_model_calls'] == 0 and summary['selection_changed'] is False and summary['clinical_score'] is None)
    for root in (RUN, PLAN):
        for p in (root, *root.rglob('*')):
            check(not p.is_symlink() and p.stat().st_gid in (96293, 65534) and
                  p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660))
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    report = {'status': 'passed', 'worker_manifest_sha256': expected,
        'audit_source_sha256': sha(Path(__file__).resolve()),
        'independent_statistics_helper_sha256': sha(Path(independent_statistics.__file__).resolve()),
        'source_bound_feature_checks': feature_checks, 'point_correlations_recomputed': point_checks,
        'binary_diagnostics_recomputed': binary_checks, 'cluster_intervals_recomputed': ci_checks,
        'eligible_anchor_feature_target_cells_replayed': anchor_checks,
        'protected_modes_verified': True, 'all_source_and_old_result_pins_unchanged': True,
        'raw_report_image_ehr_or_native_graph_body_read': False,
        'native_extraction_independently_rerun': False, 'model_calls': 0,
        'clinical_qualified': False, 'selection_changed': False, 'runtime_seconds': time.monotonic() - started}
    path = OUT / 'audit.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)
    print(json.dumps({'status': 'independent_literal_diagnostic_audit_passed',
        'runtime_seconds': round(report['runtime_seconds'], 3), 'audit_sha256': sha(path)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        audit(args.expected_manifest_sha256)
    except Exception as error:
        print(json.dumps({'status': 'independent_literal_audit_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
