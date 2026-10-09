"""Independent numerical/count replay; only derived metadata, never text IO."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

from audit_radeval_medcpt import check, load, sha, close, rho, tau, percentile
import audit_radeval_medcpt as independent_statistics

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'radeval_length_control_runs/length_controls_12766754_001'
PLAN = BASE / 'radeval_length_control_plans/length_controls_12766754_001'
LITERAL = BASE / 'radeval_literal_fact_runs/literal_expert_12766754_001'
INVENTORY = BASE / 'report_metric_sources/radeval_expert_contract_12714150_003/inventory.json'
TOKEN = BASE / 'radeval_medcpt_runs/reportref_12714150_001/token_receipts.json'
GRAPH = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001/graph_receipts.json'
OUT = BASE / 'radeval_length_control_audits/numeric_12766754_001'
FEATURES = ('polarity_opposition_proposals', 'different_anatomy_context_concepts',
    'modifier_token_set_differences', 'hypothesis_only_literal_atoms',
    'reference_only_literal_atoms', 'uncertain_or_mixed_literal_atoms')
CONTROLS = ('shortest_native_token_length', 'closest_reference_token_length', 'fewest_native_entities')
CATEGORIES = ('false_prediction', 'omission', 'incorrect_location', 'incorrect_severity',
              'unsupported_comparison', 'omitted_change', 'inarticulate_report')


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
    check('/job_12766754/' in Path('/proc/self/cgroup').read_text())
    check(sha(RUN / 'manifest.json') == expected and not OUT.exists() and not OUT.is_symlink())
    manifest = load(RUN / 'manifest.json')
    for name, value in manifest['pins'].items():
        check(sha(WORKSPACE / name) == value)
    for artifact in manifest['artifacts']:
        check(Path(artifact['path']).name == artifact['path'] and sha(RUN / artifact['path']) == artifact['sha256'])
    check(sha(PLAN / 'manifest.json') == manifest['plan_manifest_sha256'])
    plan_receipt = load(PLAN / 'manifest.json')
    check(plan_receipt['pins'] == manifest['pins'] and
          sha(PLAN / 'plan.json') == plan_receipt['artifacts'][0]['sha256'])
    source, literals = load(INVENTORY), load(LITERAL / 'derived_table.json')
    tokens = load(TOKEN)
    graphs = {g['graph_id']: g for g in load(GRAPH)}
    records, result, summary = load(RUN / 'metadata_table.json'), load(RUN / 'evaluation.json'), load(RUN / 'summary.json')
    check(load(RUN / 'prediction_receipt.json')['metadata_table_sha256'] == sha(RUN / 'metadata_table.json'))
    check(len(source['records']) == len(literals) == len(records) == 624)
    cost_checks = 0
    for original, literal, row in zip(source['records'], literals, records):
        check(original['item_id'] == literal['item_id'] == row['item_id'])
        for key in ('source_id', 'source_group_id', 'section_id', 'section_name', 'candidate_slot',
                    'reference_sha256', 'hypothesis_sha256'):
            check(original[key] == literal[key] == row[key])
        h, r = tokens[original['hypothesis_sha256']], tokens[original['reference_sha256']]
        check(h['text_sha256'] == original['hypothesis_sha256'] and r['text_sha256'] == original['reference_sha256'])
        check(h['truncated'] is r['truncated'] is h['adapter_added_prefix'] is r['adapter_added_prefix'] is False)
        hypothesis_graph, reference_graph = graphs[original['hypothesis_graph_id']], graphs[original['reference_graph_id']]
        check(hypothesis_graph['text_sha256'] == original['hypothesis_sha256'] and
              reference_graph['text_sha256'] == original['reference_sha256'])
        available = (literal['status'] == 'complete' and h['native_token_count'] is not None and
                     r['native_token_count'] is not None and hypothesis_graph['status'] == reference_graph['status'] == 'complete')
        check(row['common_available'] is available)
        check(row['lengths'] == {'hypothesis': h['native_token_count'], 'reference': r['native_token_count']})
        count = hypothesis_graph['metadata']['entity_count'] if hypothesis_graph['status'] == 'complete' else None
        check(row['native_entity_count'] == count)
        expected_costs = {**literal['features'], CONTROLS[0]: h['native_token_count'],
            CONTROLS[1]: abs(h['native_token_count'] - r['native_token_count']) if available else None,
            CONTROLS[2]: count}
        if not available:
            expected_costs = dict.fromkeys((*FEATURES, *CONTROLS))
        check(row['costs'] == expected_costs)
        cost_checks += len(expected_costs)
        errors, totals = {}, []
        for severity, counts in original['errors'].items():
            total = sum(counts) if None not in counts else None
            errors[severity + '_total'] = total
            totals.append(total)
            errors.update({severity + '_' + category: value for category, value in zip(CATEGORIES, counts)})
        errors['all_errors_total'] = sum(totals) if None not in totals else None
        check(errors == row['expert_outcomes'] == literal['expert_outcomes'])
    anchors = defaultdict(list)
    for r in records:
        anchors[(r['source_id'], r['section_id'], r['reference_sha256'])].append(r)
    index, recomputed_anchors = {}, {}
    correlations = anchor_checks = ci_checks = 0
    for item in result['results']:
        selector, target = item['selector'], item['target']
        check((selector, target) not in index)
        index[(selector, target)] = item
        available = [r for r in records if r['common_available'] and r['expert_outcomes'][target] is not None]
        pairs = [(r['costs'][selector], r['expert_outcomes'][target]) for r in available]
        check(item['paired_rows'] == len(pairs) == item['correlation']['paired_rows'])
        close(item['correlation']['spearman'], rho(pairs))
        close(item['correlation']['kendall_tau_b'], tau(pairs))
        correlations += 1
        check(len(item['anchor_rows']) == item['attempted_anchors'] == len(anchors))
        usable = []
        for saved, candidates in zip(item['anchor_rows'], anchors.values()):
            candidates = sorted(candidates, key=lambda r: r['candidate_slot'])
            check([r['candidate_slot'] for r in candidates] == [1, 2, 3])
            complete = all(r['common_available'] and r['expert_outcomes'][target] is not None for r in candidates)
            check(saved['status'] == ('complete' if complete else 'incomplete_anchor'))
            if not complete:
                check(saved['selected_expected_errors'] is None)
                continue
            costs, errors = [r['costs'][selector] for r in candidates], [r['expert_outcomes'][target] for r in candidates]
            top = [i for i, cost in enumerate(costs) if cost == min(costs)]
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
                if costs[i] == costs[j]:
                    correct += .5
                    ties += 1
                else:
                    correct += int((costs[i] < costs[j]) == (errors[i] < errors[j]))
            check(saved['strict_pairs'] == strict and saved['expected_correct_pairs'] == correct and saved['score_tied_strict_pairs'] == ties)
            usable.append({**saved, 'selected_expected_errors': selected, 'metric_minus_random_errors': selected - uniform})
            anchor_checks += 1
        recomputed_anchors[(selector, target)] = usable
        check(item['complete_anchors'] == len(usable))
        for field, value in item['means'].items():
            close(value, sum(a[field] for a in usable) / len(usable) if usable else None)
        strict = sum(a['strict_pairs'] for a in usable)
        close(item['expected_pairwise_accuracy'], sum(a['expected_correct_pairs'] for a in usable) / strict if strict else None)
        if 'error_delta_cluster_ci' in item:
            check_ci(usable, item['error_delta_cluster_ci'])
            ci_checks += 1
    expected_selectors = set((*FEATURES, *CONTROLS))
    check(set(result['selectors']) == expected_selectors and len(index) == 153)
    paired_checks = 0
    seen = set()
    for item in result['paired_literal_control_comparisons']:
        feature, control, target = item['feature'], item['control'], item['target']
        check(feature in FEATURES and control in CONTROLS and (feature, control, target) not in seen)
        seen.add((feature, control, target))
        left, right = recomputed_anchors[(feature, target)], recomputed_anchors[(control, target)]
        check(len(left) == len(right) == item['complete_anchors'])
        differences = []
        for a, b in zip(left, right):
            check(a['anchor_id'] == b['anchor_id'] and a['source_group_id'] == b['source_group_id'])
            differences.append({'anchor_id': a['anchor_id'], 'source_group_id': a['source_group_id'],
                'metric_minus_random_errors': a['selected_expected_errors'] - b['selected_expected_errors']})
        close(item['literal_minus_control_mean_errors'],
              sum(r['metric_minus_random_errors'] for r in differences) / len(differences) if differences else None)
        if 'paired_cluster_ci' in item:
            check(item['paired_anchor_rows'] == differences)
            check_ci(differences, item['paired_cluster_ci'])
            ci_checks += 1
        paired_checks += 1
    check(paired_checks == 306)
    old = load(LITERAL / 'evaluation.json')
    for earlier in old['results']:
        new = index[(earlier['feature'], earlier['target'])]
        check(new['correlation'] == earlier['correlation'] and new['means'] == earlier['within_anchor_diagnostic']['means'])
    check(summary['original_literal_cells_exactly_replayed'] == len(old['results']) == 102)
    check(result['clinical_score'] is None and summary['new_model_calls'] == 0 and all(v is False for v in result['policy'].values()))
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
        'metadata_cost_values_recomputed': cost_checks, 'point_correlations_recomputed': correlations,
        'eligible_selector_target_anchor_cells_replayed': anchor_checks,
        'paired_literal_control_comparisons_recomputed': paired_checks, 'cluster_intervals_recomputed': ci_checks,
        'all_pins_and_protected_modes_verified': True, 'raw_report_ehr_image_native_graph_or_embedding_decoded': False,
        'model_calls': 0, 'clinical_qualified': False, 'selection_changed': False,
        'runtime_seconds': time.monotonic() - started}
    path = OUT / 'audit.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o660)
    print(json.dumps({'status': 'independent_length_control_audit_passed',
        'runtime_seconds': round(receipt['runtime_seconds'], 3), 'audit_sha256': sha(path)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    try:
        audit(args.expected_manifest_sha256)
    except Exception as error:
        print(json.dumps({'status': 'independent_length_control_audit_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
