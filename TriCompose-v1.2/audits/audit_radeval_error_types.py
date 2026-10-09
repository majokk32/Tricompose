"""Independent numerical replay only; no raw reports/images or model IO."""
from collections import defaultdict
import json
import os
from pathlib import Path
import random

from audit_radeval_biovil_run import close, percentile, require, sha256, spearman, tau

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'radeval_image_benchmark_runs/biovil_cpu_12714150_001'
RUN = BASE / 'radeval_error_type_diagnostics/error_types_12714150_001'
OUT = BASE / 'radeval_error_type_diagnostic_audits/error_types_12714150_001'
CATEGORIES = ('false_prediction', 'omission', 'incorrect_location', 'incorrect_severity',
              'unsupported_comparison', 'omitted_change', 'inarticulate_report')
METRICS = ('biovil_t_raw_cosine', 'radgraph_entity_f1',
           'radgraph_relation_presence_f1', 'radgraph_full_relation_f1')


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUT.exists())
    require(sha256(RUN / 'manifest.json') ==
            '1a3f64824bc43185206dae55ec643cbf8cd2eea460c30c9c06015eba42b8c786')
    receipt = json.loads((RUN / 'manifest.json').read_text())
    hash_checks = 0
    for pin in receipt['pins']:
        require(sha256(ROOT / pin['path']) == pin['sha256'])
        hash_checks += 1
    for pin in receipt['outputs']:
        path = RUN / pin['path']
        require(path.resolve().is_relative_to(RUN) and sha256(path) == pin['sha256'])
        hash_checks += 1
    require(sha256(SOURCE / 'manifest.json') == receipt['source_manifest_sha256'])
    records = json.loads((SOURCE / 'paired_score_table.json').read_text())
    result = json.loads((RUN / 'diagnostic.json').read_text())
    summary = json.loads((RUN / 'summary.json').read_text())
    require(len(records) == result['all_attempted_pairs'] == 624)
    require(result['common_score_available_pairs'] == 132)
    require(result['categories'] == list(CATEGORIES) and result['metrics'] == list(METRICS))
    require(len(result['results']) == 28)
    anchors = defaultdict(list)
    for row in records:
        anchors[(row['source_id'], row['section_id'], row['reference_sha256'])].append(row)
    anchors = {k: sorted(v, key=lambda r: r['candidate_slot']) for k, v in sorted(anchors.items())}
    require(len(anchors) == 208)
    count_checks = correlation_checks = choice_checks = interval_checks = 0
    for category in CATEGORIES:
        target = 'clinically_significant_' + category
        available = [r for r in records if r['paired_cohort_available'] and
                     r['expert_outcomes'][target] is not None]
        for metric in METRICS:
            stored = next(r for r in result['results'] if r['category'] == category and r['metric'] == metric)
            require(stored['all_attempted_pairs'] == len(records) and
                    stored['score_and_category_available_pairs'] == len(available))
            require(stored['error_positive_pairs'] == sum(r['expert_outcomes'][target] > 0 for r in available))
            require(close(stored['category_error_sum'], sum(r['expert_outcomes'][target] for r in available)))
            a, b = [r['scores'][metric] for r in available], [-r['expert_outcomes'][target] for r in available]
            require(close(stored['pooled_correlation']['spearman'], spearman(a, b)) and
                    close(stored['pooled_correlation']['kendall_tau_b'], tau(a, b)))
            count_checks += 1
            correlation_checks += 1
            choices, strict, correct, ties = [], 0, 0.0, 0
            for rows in anchors.values():
                if not all(r['paired_cohort_available'] and
                           r['expert_outcomes'][target] is not None for r in rows):
                    continue
                scores = [r['scores'][metric] for r in rows]
                errors = [r['expert_outcomes'][target] for r in rows]
                top = [i for i, v in enumerate(scores) if v == max(scores)]
                chosen, random_mean = sum(errors[i] for i in top)/len(top), sum(errors)/3
                choices.append((rows[0]['source_group_id'], chosen, random_mean, chosen-random_mean))
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    if errors[i] == errors[j]:
                        continue
                    strict += 1
                    if scores[i] == scores[j]:
                        ties += 1
                        correct += .5
                    else:
                        correct += int((scores[i] > scores[j]) == (errors[i] < errors[j]))
            require(stored['attempted_anchors'] == len(anchors) and stored['complete_anchors'] == len(choices))
            require(stored['strict_candidate_pairs'] == strict and stored['score_tied_strict_pairs'] == ties)
            require(close(stored['pairwise_accuracy'], correct/strict if strict else None))
            require(stored['has_error_discriminating_pairs'] is bool(strict))
            for index, name in enumerate(('selected_errors', 'random_errors', 'metric_minus_random_errors'), 1):
                require(close(stored['means'][name], sum(c[index] for c in choices)/len(choices)))
            choice_checks += len(choices)
            groups = defaultdict(list)
            for group, _, _, delta in choices:
                groups[group].append(delta)
            keys = sorted(groups)
            rng, boot = random.Random(0), []
            for _ in range(1000):
                sample = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
                boot.append(sum(sample)/len(sample))
            require(stored['error_delta_cluster_ci']['source_groups'] == len(keys) == 34)
            require(all(close(a, b) for a, b in zip(stored['error_delta_cluster_ci']['interval95'],
                        [percentile(boot, .025), percentile(boot, .975)])))
            interval_checks += 1
    require(summary['independent_total_error_additivity_checks'] == 12 and
            summary['deterministic_reverse_order_replay'] is True)
    for name in ('clinical_qualified', 'selection_changed', 'weight_or_threshold_fitting',
                 'raw_patient_input_read', 'regeneration_authorized'):
        require(result[name] is False)
    require(result['new_model_calls'] == 0 and result['post_hoc_descriptive_diagnostic'] is True)
    require(result['multiplicity_adjusted'] is False and result['expert_counts_are_image_finding_labels'] is False)
    permissions = 0
    for path in (RUN, *RUN.iterdir()):
        require(path.stat().st_gid in (96293, 65534) and path.stat().st_mode & 0o7777 ==
                (0o2770 if path.is_dir() else 0o660))
        permissions += 1
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    payload = {'status': 'passed', 'source_manifest_sha256': sha256(RUN / 'manifest.json'),
        'code_input_output_hash_checks': hash_checks, 'all_attempted_pairs_retained': len(records),
        'error_type_denominator_replays': count_checks, 'independent_correlations_replayed': correlation_checks,
        'analytical_choices_replayed': choice_checks, 'patient_cluster_intervals_replayed': interval_checks,
        'protected_permissions_checked': permissions, 'new_model_calls': 0,
        'raw_patient_source_access': False, 'clinical_qualified': False, 'selection_changed': False}
    with (OUT / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    (OUT / 'audit.json').chmod(0o660)
    print(json.dumps({'status': 'error_type_diagnostic_audit_passed',
                      'audit_sha256': sha256(OUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
