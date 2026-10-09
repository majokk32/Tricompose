"""Independent stdlib numeric/hash audit; zero models or clinical text inputs.

No production benchmark/statistics imports. Does not replay model generation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'radeval_medcpt_runs/reportref_12714150_001'
PLAN = BASE / 'radeval_medcpt_plans/reportref_12714150_001'
EXPERT = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
BIOVIL = BASE / 'radeval_image_benchmark_runs/biovil_cpu_12714150_001'
OUT = BASE / 'radeval_medcpt_audits/numeric_12714150_001'


def check(value):
    if not value:
        raise ValueError('numeric_receipt_or_replay_mismatch')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(left, right):
    if left is None or right is None:
        check(left is right)
    else:
        check(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10))


def ranks(values):
    counts, lookup, earlier = Counter(values), {}, 0
    for value in sorted(counts):
        lookup[value] = earlier + (counts[value] + 1) / 2
        earlier += counts[value]
    return [lookup[value] for value in values]


def rho(pairs):
    if len(pairs) < 3:
        return None
    x, y = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    return sum(a*b for a, b in zip(dx, dy)) / denom if denom else None


def tau(pairs):
    if len(pairs) < 3:
        return None
    pos = neg = tx = ty = 0
    for index, (x, y) in enumerate(pairs):
        for xx, yy in pairs[index+1:]:
            dx, dy = x - xx, y - yy
            if dx == dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                pos += 1
            else:
                neg += 1
    den = math.sqrt((pos+neg+tx) * (pos+neg+ty))
    return (pos-neg) / den if den else None


def percentile(values, fraction):
    data = sorted(values)
    offset = (len(data)-1) * fraction
    i = int(offset)
    return data[i] + (data[min(i+1, len(data)-1)]-data[i]) * (offset-i)


def bootstrap_correlation(values):
    groups = defaultdict(list)
    for group, quality, target in values:
        groups[group].append((quality, target))
    keys, rng, boot = sorted(groups), random.Random(0), []
    if len(keys) >= 3:
        for _ in range(1000):
            sample = [p for _ in keys for p in groups[keys[rng.randrange(len(keys))]]]
            result = rho(sample)
            if result is not None:
                boot.append(result)
    usable = len(boot) >= 900
    return ([percentile(boot, .025), percentile(boot, .975)] if usable else None), len(boot), len(keys)


def audit(expected):
    started = time.monotonic()
    check('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    check(re.fullmatch('[a-f0-9]{64}', expected) and sha(RUN / 'manifest.json') == expected)
    check(not OUT.exists() and not OUT.is_symlink())
    receipt = load(RUN / 'manifest.json')
    for path, digest in receipt['pins'].items():
        check(sha(WORKSPACE / path) == digest)
    for artifact in receipt['artifacts']:
        check(Path(artifact['path']).name == artifact['path'] and
              sha(RUN / artifact['path']) == artifact['sha256'])
    check(sha(PLAN / 'manifest.json') == receipt['plan_manifest_sha256'])
    prediction = load(RUN / 'prediction_receipt.json')
    for artifact in prediction['artifacts']:
        check(sha(RUN / artifact['path']) == artifact['sha256'])
    plan = load(EXPERT / 'frozen_plan.json')['inventory']
    old_graph = load(EXPERT / 'scores.json')
    old_image = load(BIOVIL / 'image_scores.json')
    scores = load(RUN / 'scores.json')
    tokens = load(RUN / 'token_receipts.json')
    vectors = load(RUN / 'private_embeddings.json')['vectors_by_text_sha256']
    records = load(RUN / 'score_table.json')
    result = load(RUN / 'evaluation.json')
    summary = load(RUN / 'summary.json')
    check(len(plan['records']) == len(scores) == len(records) == len(old_graph) == len(old_image) == 624)
    check(len(tokens) == len(plan['graphs']) == 762)
    check(set(tokens) == {g['text_sha256'] for g in plan['graphs']})
    check(set(vectors) == {k for k, r in tokens.items() if r['status'] == 'complete'})
    for key, vector in vectors.items():
        check(len(vector) == 768 and all(type(v) in (int, float) and math.isfinite(v) for v in vector))
        check(1 <= tokens[key]['native_token_count'] <= 512 and tokens[key]['truncated'] is False
              and tokens[key]['adapter_added_prefix'] is False)
    for token in tokens.values():
        check(token['truncated'] is False and token['adapter_added_prefix'] is False)
        if token['status'] == 'over_capacity':
            check(token['native_token_count'] > 512)
    med_metric, image_metric = 'medcpt_query_fulltext_cosine', 'biovil_t_raw_cosine'
    for pair, score, row, graph, image in zip(plan['records'], scores, records, old_graph, old_image):
        check(pair['item_id'] == score['item_id'] == row['item_id'] == graph['item_id'] == image['item_id'])
        for key in ('source_id', 'source_group_id', 'section_id', 'section_name', 'candidate_slot',
                    'reference_sha256', 'hypothesis_sha256'):
            check(pair[key] == row[key])
        check(score['status'] == row['medcpt_status'])
        if score['status'] == 'complete':
            left, right = vectors[pair['reference_sha256']], vectors[pair['hypothesis_sha256']]
            value = sum(a*b for a, b in zip(left, right)) / (
                math.sqrt(sum(a*a for a in left)) * math.sqrt(sum(b*b for b in right)))
            close(value, score['value'])
        else:
            check(score['value'] is None)
        close(score['value'], row['scores'][med_metric])
        close(image['value'], row['scores'][image_metric])
        for metric, value in graph['scores'].items():
            close(value, row['scores'][metric])
        full = score['status'] == graph['status'] == 'complete'
        check(row['full_common_available'] is full and row['image_common_available'] is
              (full and image['status'] == 'complete'))
        categories = ('false_prediction', 'omission', 'incorrect_location', 'incorrect_severity',
                      'unsupported_comparison', 'omitted_change', 'inarticulate_report')
        reference = {}
        totals = []
        for severity, values in pair['errors'].items():
            total = sum(values) if None not in values else None
            totals.append(total)
            reference[severity + '_total'] = total
            reference.update({severity + '_' + category: value for category, value in zip(categories, values)})
        reference['all_errors_total'] = sum(totals) if None not in totals else None
        check(row['expert_outcomes'] == reference)
    point_comparisons = bootstraps = anchor_comparisons = 0
    for cohort_name, mask in (('full_text_common', 'full_common_available'),
                              ('cached_image_common', 'image_common_available')):
        cohort = result['cohorts'][cohort_name]
        common = [r for r in records if r[mask]]
        check(cohort['all_attempted_pairs'] == 624 and cohort['common_score_available_pairs'] == len(common))
        required_metrics = {med_metric, *old_graph[0]['scores']}
        if cohort_name == 'cached_image_common':
            required_metrics.add(image_metric)
        check(set(cohort['metrics']) == required_metrics)
        for metric, evaluation in cohort['metrics'].items():
            for target, corr in evaluation['outcomes'].items():
                values = [(r['source_group_id'], r['scores'][metric], -r['expert_outcomes'][target])
                          for r in common if r['expert_outcomes'][target] is not None]
                pairs = [(v[1], v[2]) for v in values]
                check(corr['paired_rows'] == len(values))
                close(corr['paired_coverage'], len(values) / 624)
                close(corr['spearman'], rho(pairs))
                close(corr['kendall_tau_b'], tau(pairs))
                point_comparisons += 1
                if target in ('clinically_significant_total', 'all_errors_total'):
                    interval, usable, groups = bootstrap_correlation(values)
                    ci = corr['cluster_bootstrap']
                    check(ci['usable_resamples'] == usable and ci['source_groups'] == groups)
                    if interval is None:
                        check(ci['spearman_interval_95'] is None)
                    else:
                        for a, b in zip(interval, ci['spearman_interval_95']):
                            close(a, b)
                    bootstraps += 1
            anchors = defaultdict(list)
            for row in records:
                anchors[(row['source_id'], row['section_id'], row['reference_sha256'])].append(row)
            for target, selection in evaluation['selection_diagnostic'].items():
                expected_rows = []
                for index, candidates in enumerate(anchors.values()):
                    candidates.sort(key=lambda r: r['candidate_slot'])
                    check([r['candidate_slot'] for r in candidates] == [1, 2, 3])
                    complete = all(r[mask] and r['expert_outcomes'][target] is not None for r in candidates)
                    replayed = selection['anchor_records'][index]
                    check(replayed['anchor_id'] == f'anchor_{index:04d}' and
                          replayed['source_group_id'] == candidates[0]['source_group_id'] and
                          replayed['status'] == ('complete' if complete else 'incomplete_anchor'))
                    if not complete:
                        check(replayed['selected_expected_errors'] is None)
                        continue
                    quality = [r['scores'][metric] for r in candidates]
                    errors = [r['expert_outcomes'][target] for r in candidates]
                    top = [i for i, value in enumerate(quality) if value == max(quality)]
                    selected, uniform = sum(errors[i] for i in top) / len(top), sum(errors) / 3
                    check(replayed['top_tie_size'] == len(top))
                    close(replayed['selected_expected_errors'], selected)
                    close(replayed['random_expected_errors'], uniform)
                    close(replayed['oracle_errors'], min(errors))
                    close(replayed['metric_minus_random_errors'], selected-uniform)
                    strict = correct = ties = 0
                    for i, j in ((0, 1), (0, 2), (1, 2)):
                        if errors[i] == errors[j]:
                            continue
                        strict += 1
                        if quality[i] == quality[j]:
                            correct += .5
                            ties += 1
                        else:
                            correct += int((quality[i] > quality[j]) == (errors[i] < errors[j]))
                    check(replayed['strict_pairs'] == strict and replayed['expected_correct_pairs'] == correct
                          and replayed['score_tied_strict_pairs'] == ties)
                    expected_rows.append(replayed)
                    anchor_comparisons += 1
                check(selection['attempted_anchors'] == len(anchors) and
                      selection['complete_anchors'] == len(expected_rows))
                for field, mean in selection['means'].items():
                    close(mean, sum(r[field] for r in expected_rows) / len(expected_rows) if expected_rows else None)
                strict = sum(r['strict_pairs'] for r in expected_rows)
                check(selection['strict_pairs'] == strict and selection['score_tied_strict_pairs'] ==
                      sum(r['score_tied_strict_pairs'] for r in expected_rows))
                close(selection['expected_pairwise_accuracy'],
                      sum(r['expected_correct_pairs'] for r in expected_rows) / strict if strict else None)
                groups = defaultdict(list)
                for r in expected_rows:
                    groups[r['source_group_id']].append(r['metric_minus_random_errors'])
                keys, rng, boot = sorted(groups), random.Random(0), []
                if len(keys) >= 3:
                    for _ in range(1000):
                        sample = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
                        boot.append(sum(sample) / len(sample))
                ci = selection['error_delta_cluster_ci']
                if boot:
                    for a, b in zip([percentile(boot, .025), percentile(boot, .975)], ci['interval95']):
                        close(a, b)
                else:
                    check(ci['interval95'] is None)
    check(summary['pair_status_counts'] == dict(Counter(r['status'] for r in scores)))
    check(summary['text_status_counts'] == dict(Counter(r['status'] for r in tokens.values())))
    check(all(value is False for value in result['policy'].values()))
    check(summary['whole_report_input_limit'] == 512 and summary['input_truncation'] is False and
          summary['old_bank_scores_or_choices_changed'] is False)
    check(summary['small_replay'] == prediction['replay'])
    for root in (RUN, PLAN):
        for path in (root, *root.rglob('*')):
            check(not path.is_symlink() and path.stat().st_gid in (96293, 65534) and
                  path.stat().st_mode & 0o7777 == (0o2770 if path.is_dir() else 0o660))
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    report = {'status': 'passed', 'worker_manifest_sha256': expected,
        'audit_source_sha256': sha(Path(__file__).resolve()), 'all_attempted_pairs': 624,
        'independently_recomputed_cosines': sum(s['status'] == 'complete' for s in scores),
        'point_correlations_recomputed': point_comparisons,
        'clustered_correlation_intervals_recomputed': bootstraps,
        'eligible_anchor_metric_target_cells_replayed': anchor_comparisons,
        'all_prediction_source_and_historical_pins_unchanged': True, 'protected_modes_verified': True,
        'model_calls': 0, 'raw_patient_reports_images_or_ehrs_read': False,
        'tokenization_or_inference_replay_independently_rerun': False,
        'clinical_qualified': False, 'selection_changed': False,
        'runtime_seconds': time.monotonic()-started}
    path = OUT / 'audit.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)
    print(json.dumps({'status': 'independent_medcpt_benchmark_audit_passed',
        'runtime_seconds': round(report['runtime_seconds'], 3), 'audit_sha256': sha(path)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        audit(args.expected_manifest_sha256)
    except Exception as error:
        print(json.dumps({'status': 'independent_medcpt_audit_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
