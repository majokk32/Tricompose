"""Independent cached-number replay, without production metric imports."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import re
import time

from audit_radeval_medcpt import check, load, sha, close, rho, tau, percentile
import audit_radeval_medcpt as independent_statistics

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'radeval_image_length_control_runs/image_length_12766754_001'
PLAN = BASE / 'radeval_image_length_control_plans/image_length_12766754_001'
IMAGE = BASE / 'radeval_image_benchmark_runs/biovil_cpu_12714150_001'
LENGTH = BASE / 'radeval_length_control_runs/length_controls_12766754_001'
OUT = BASE / 'radeval_image_length_control_audits/numeric_12766754_001'
METRICS = ('biovil_t_raw_cosine', 'radgraph_entity_f1', 'radgraph_relation_presence_f1', 'radgraph_full_relation_f1')
CONTROLS = ('shortest_native_token_length', 'closest_reference_token_length', 'fewest_native_entities')
TARGETS = ('clinically_significant_total', 'all_errors_total')


def interval(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['source_group_id']].append(r['metric_minus_random_errors'])
    keys, rng, boot = sorted(groups), random.Random(0), []
    if len(keys) >= 3:
        for _ in range(1000):
            sample = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
            boot.append(sum(sample) / len(sample))
    return ([percentile(boot, .025), percentile(boot, .975)] if boot else None), len(keys)


def check_ci(rows, saved):
    expected, groups = interval(rows)
    check(saved['source_groups'] == groups)
    if expected is None:
        check(saved['interval95'] is None)
    else:
        for a, b in zip(expected, saved['interval95']):
            close(a, b)


def audit(expected):
    started = time.monotonic()
    check('/job_12766754/' in Path('/proc/self/cgroup').read_text() and
          re.fullmatch('[a-f0-9]{64}', expected) and sha(RUN / 'manifest.json') == expected)
    check(not OUT.exists() and not OUT.is_symlink())
    manifest = load(RUN / 'manifest.json')
    for name, value in manifest['pins'].items():
        check(sha(WORKSPACE / name) == value)
    for artifact in manifest['artifacts']:
        check(Path(artifact['path']).name == artifact['path'] and sha(RUN / artifact['path']) == artifact['sha256'])
    check(sha(PLAN / 'manifest.json') == manifest['plan_manifest_sha256'])
    plan_receipt = load(PLAN / 'manifest.json')
    check(plan_receipt['pins'] == manifest['pins'] and sha(PLAN / 'plan.json') == plan_receipt['artifacts'][0]['sha256'])
    image, metadata = load(IMAGE / 'paired_score_table.json'), load(LENGTH / 'metadata_table.json')
    records, result, summary = load(RUN / 'metadata_table.json'), load(RUN / 'evaluation.json'), load(RUN / 'summary.json')
    check(load(RUN / 'prediction_receipt.json')['metadata_table_sha256'] == sha(RUN / 'metadata_table.json'))
    check(len(image) == len(metadata) == len(records) == 624)
    score_checks = 0
    for original, source, row in zip(image, metadata, records):
        for k in ('item_id', 'source_id', 'source_group_id', 'section_id', 'section_name',
                  'candidate_slot', 'reference_sha256', 'hypothesis_sha256'):
            check(row[k] == source[k] == original[k])
        check(row['expert_outcomes'] == original['expert_outcomes'] == source['expert_outcomes'])
        available = original['paired_cohort_available'] and source['common_available']
        check(row['common_available'] is available and
              row['original_image_cohort_available'] is original['paired_cohort_available'] and
              row['image_score_status'] == original['image_score_status'])
        scores = {**original['scores'], **{c: -source['costs'][c] if available else None for c in CONTROLS}}
        if not available:
            scores = dict.fromkeys((*METRICS, *CONTROLS))
        check(row['scores'] == scores)
        score_checks += len(scores)
    check(sum(r['common_available'] for r in records) == 132 and
          all(r['common_available'] == r['original_image_cohort_available'] for r in records))
    anchors = defaultdict(list)
    for r in records:
        anchors[(r['source_id'], r['section_id'], r['reference_sha256'])].append(r)
    index, recomputed = {}, {}
    correlations = anchor_checks = ci_checks = 0
    for item in result['results']:
        selector, target = item['selector'], item['target']
        check(selector in (*METRICS, *CONTROLS) and target in TARGETS and (selector, target) not in index)
        index[(selector, target)] = item
        pairs = [(r['scores'][selector], -r['expert_outcomes'][target]) for r in records
                 if r['common_available'] and r['expert_outcomes'][target] is not None]
        check(item['paired_rows'] == len(pairs) == item['correlation']['paired_rows'])
        close(item['correlation']['spearman'], rho(pairs))
        close(item['correlation']['kendall_tau_b'], tau(pairs))
        correlations += 1
        usable = []
        check(len(item['anchor_rows']) == item['attempted_anchors'] == len(anchors) == 208)
        for saved, (anchor_index, candidates) in zip(item['anchor_rows'], enumerate(anchors.values())):
            candidates = sorted(candidates, key=lambda r: r['candidate_slot'])
            check([r['candidate_slot'] for r in candidates] == [1, 2, 3] and
                  saved['anchor_id'] == f'anchor_{anchor_index:04d}' and
                  saved['source_group_id'] == candidates[0]['source_group_id'])
            complete = all(r['common_available'] and r['expert_outcomes'][target] is not None for r in candidates)
            check(saved['status'] == ('complete' if complete else 'incomplete_anchor'))
            if not complete:
                check(saved['selected_expected_errors'] is None)
                continue
            scores, errors = [r['scores'][selector] for r in candidates], [r['expert_outcomes'][target] for r in candidates]
            top = [i for i, score in enumerate(scores) if score == max(scores)]
            selected, uniform = sum(errors[i] for i in top) / len(top), sum(errors) / 3
            check(saved['top_tie_size'] == len(top))
            close(saved['selected_expected_errors'], selected)
            close(saved['random_expected_errors'], uniform)
            close(saved['oracle_errors'], min(errors))
            close(saved['metric_minus_random_errors'], selected - uniform)
            strict = correct = ties = 0
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if errors[i] == errors[j]:
                    continue
                strict += 1
                if scores[i] == scores[j]:
                    correct += .5
                    ties += 1
                else:
                    correct += int((scores[i] > scores[j]) == (errors[i] < errors[j]))
            check(saved['strict_pairs'] == strict and saved['expected_correct_pairs'] == correct and saved['score_tied_strict_pairs'] == ties)
            usable.append(saved)
            anchor_checks += 1
        recomputed[(selector, target)] = usable
        check(item['complete_anchors'] == len(usable) == 44)
        for field, value in item['means'].items():
            close(value, sum(a[field] for a in usable) / len(usable))
        strict = sum(a['strict_pairs'] for a in usable)
        check(item['strict_candidate_pairs'] == strict)
        close(item['pairwise_accuracy'], sum(a['expected_correct_pairs'] for a in usable) / strict if strict else None)
        check_ci(usable, item['error_delta_cluster_ci'])
        ci_checks += 1
    check(len(index) == 14)
    seen = set()
    for item in result['paired_metric_control_comparisons']:
        metric, control, target = item['metric'], item['control'], item['target']
        check(metric in METRICS and control in CONTROLS and target in TARGETS and (metric, control, target) not in seen)
        seen.add((metric, control, target))
        a, b = recomputed[(metric, target)], recomputed[(control, target)]
        differences = []
        for x, y in zip(a, b):
            check(x['anchor_id'] == y['anchor_id'] and x['source_group_id'] == y['source_group_id'])
            differences.append({'anchor_id': x['anchor_id'], 'source_group_id': x['source_group_id'],
                'metric_minus_random_errors': x['selected_expected_errors'] - y['selected_expected_errors']})
        check(item['complete_anchors'] == len(differences) == 44 and item['paired_anchor_rows'] == differences)
        close(item['metric_minus_control_mean_errors'], sum(r['metric_minus_random_errors'] for r in differences) / len(differences))
        check_ci(differences, item['paired_cluster_ci'])
        ci_checks += 1
    check(len(seen) == 24 and summary['original_image_selector_target_cells_exactly_replayed'] == 8)
    for earlier in load(IMAGE / 'evaluation.json')['selection_diagnostic']:
        current = index[(earlier['metric'], earlier['target'])]
        check(current['means'] == earlier['means'] and current['error_delta_cluster_ci'] == earlier['error_delta_cluster_ci'])
    check(all(v is False for v in result['policy'].values()) and result['clinical_score'] is None and summary['new_model_calls'] == 0)
    for root in (RUN, PLAN):
        for p in (root, *root.rglob('*')):
            check(not p.is_symlink() and p.stat().st_gid in (96293, 65534) and
                  p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660))
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    receipt = {'status': 'passed', 'worker_manifest_sha256': expected,
        'audit_source_sha256': sha(Path(__file__).resolve()),
        'independent_statistics_helper_sha256': sha(Path(independent_statistics.__file__).resolve()),
        'cached_score_cells_recomputed': score_checks, 'point_correlations_recomputed': correlations,
        'eligible_selector_target_anchor_cells_replayed': anchor_checks,
        'paired_metric_control_comparisons_recomputed': len(seen), 'cluster_intervals_recomputed': ci_checks,
        'all_pins_and_private_modes_verified': True, 'raw_report_ehr_image_graph_body_or_embedding_decoded': False,
        'model_calls': 0, 'clinical_qualified': False, 'selection_changed': False,
        'runtime_seconds': time.monotonic() - started}
    path = OUT / 'audit.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)
    check(path.stat().st_gid in (96293, 65534) and path.stat().st_mode & 0o7777 == 0o660)
    print(json.dumps({'status': 'independent_image_length_control_audit_passed',
        'runtime_seconds': round(receipt['runtime_seconds'], 3), 'audit_sha256': sha(path)}))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--expected-manifest-sha256', required=True)
        args = parser.parse_args()
        audit(args.expected_manifest_sha256)
    except Exception as error:
        print(json.dumps({'status': 'independent_image_length_control_audit_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
