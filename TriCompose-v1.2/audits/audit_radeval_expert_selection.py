"""Independent within-anchor analytical-choice/cluster-CI replay, no raw data."""
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
RUN = BASE / 'radeval_expert_selection_runs/within_anchor_12714150_001'
OUT = BASE / 'radeval_expert_selection_audits/within_anchor_12714150_001'


def require(value):
    if not value:
        raise ValueError('within_anchor_ranking_audit_failed')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    ix = (len(ordered) - 1)*q
    a, b = math.floor(ix), math.ceil(ix)
    return ordered[a] + (ordered[b]-ordered[a])*(ix-a)


def audit():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    require(not OUT.exists())
    receipt = json.loads((RUN / 'manifest.json').read_text())
    hashes = 0
    for pin in receipt['pins']:
        require(sha256(ROOT / pin['path']) == pin['sha256'])
        hashes += 1
    for output in receipt['outputs']:
        require(sha256(RUN / output['path']) == output['sha256'])
        hashes += 1
    plan = json.loads((SOURCE / 'frozen_plan.json').read_text())['inventory']
    scores = json.loads((SOURCE / 'scores.json').read_text())
    replayed = json.loads((RUN / 'replay.json').read_text())
    summary = json.loads((RUN / 'summary.json').read_text())
    anchors = {}
    for pair, score in zip(plan['records'], scores):
        key = (pair['source_id'], pair['section_id'], pair['reference_sha256'])
        item = anchors.setdefault(key, {'anchor_id': f'anchor_{len(anchors):04d}',
            'source_group_id': pair['source_group_id'], 'pairs': []})
        item['pairs'].append((pair, score))
    lookup = {a['anchor_id']: a for a in anchors.values()}
    require(len(anchors) == replayed['anchor_inventory'] == 208)
    require(len(replayed['records']) == 1248)
    choice_checks = 0
    for record in replayed['records']:
        anchor = lookup[record['anchor_id']]
        require(record['source_group_id'] == anchor['source_group_id'])
        pairs = sorted(anchor['pairs'], key=lambda p: p[0]['candidate_slot'])
        require([p['candidate_slot'] for p, _ in pairs] == [1, 2, 3])
        errors = []
        for pair, _ in pairs:
            cells = pair['errors']['clinically_significant']
            if record['target'] == 'all_errors_total':
                cells = cells + pair['errors']['clinically_insignificant']
            errors.append(sum(cells) if all(v is not None for v in cells) else None)
        values = [s['scores'][record['metric']] for _, s in pairs]
        if None in errors or None in values:
            require(record['status'] == 'incomplete_anchor' and record['metric_minus_random_errors'] is None)
        else:
            require(record['status'] == 'complete')
            winners = [i for i in range(3) if values[i] == max(values)]
            minimums = [i for i in range(3) if errors[i] == min(errors)]
            expected = sum(errors[i] for i in winners)/len(winners)
            require(record['top_tie_size'] == len(winners))
            require(record['metric_top_expected_errors'] == expected)
            require(record['uniform_random_expected_errors'] == sum(errors)/3)
            require(record['oracle_min_errors'] == min(errors))
            require(record['metric_minus_random_errors'] == expected - sum(errors)/3)
            require(record['metric_oracle_hit_probability'] == len(set(winners)&set(minimums))/len(winners))
            require(record['random_oracle_hit_probability'] == len(minimums)/3)
            for i in range(3):
                require(record[f'fixed_slot_{i+1}_errors'] == errors[i])
            comparisons = [(i,j) for i,j in ((0,1),(0,2),(1,2)) if errors[i] != errors[j]]
            tied = sum(values[i] == values[j] for i,j in comparisons)
            correct = sum(.5 if values[i] == values[j] else int(
                (values[i] > values[j]) == (errors[i] < errors[j])) for i,j in comparisons)
            require(record['strict_expert_pair_comparisons'] == len(comparisons)
                    and record['metric_score_tied_strict_pairs'] == tied
                    and record['expected_metric_pairwise_correct'] == correct)
        choice_checks += 1
    for item in summary['summaries']:
        records = [r for r in replayed['records'] if r['metric'] == item['metric']
                   and r['target'] == item['target'] and r['status'] == 'complete']
        require(len(records) == item['eligible_anchors'])
        for name, value in item['means'].items():
            require(value == sum(r[name] for r in records)/len(records))
        groups = defaultdict(list)
        for r in records:
            groups[r['source_group_id']].append(r['metric_minus_random_errors'])
        keys = sorted(groups)
        rng, boot = random.Random(0), []
        for _ in range(1000):
            sample = [v for _ in keys for v in groups[keys[rng.randrange(len(keys))]]]
            boot.append(sum(sample)/len(sample))
        require(item['metric_minus_random_error_delta_ci95'] == [percentile(boot,.025), percentile(boot,.975)])
        strict = sum(r['strict_expert_pair_comparisons'] for r in records)
        require(item['expected_pairwise_accuracy'] == sum(r['expected_metric_pairwise_correct'] for r in records)/strict)
    permissions = 0
    for path in (RUN, *RUN.iterdir()):
        require(path.stat().st_gid in (96293, 65534) and path.stat().st_mode & 0o7777 ==
                (0o2770 if path.is_dir() else 0o660))
        permissions += 1
    require(replayed['post_hoc_diagnostic'] and summary['post_hoc_diagnostic'] and
            summary['selection_changed'] is False and summary['clinical_qualified'] is False)
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    payload = {'status': 'passed', 'source_manifest_sha256': sha256(RUN / 'manifest.json'),
        'source_output_hash_checks': hashes, 'analytical_choice_records_replayed': choice_checks,
        'summaries_and_cluster_intervals_recomputed': len(summary['summaries']),
        'protected_permissions_checked': permissions, 'new_model_calls': 0,
        'post_hoc_diagnostic': True, 'clinical_qualified': False, 'selection_changed': False}
    with (OUT / 'audit.json').open('x') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write('\n')
    (OUT / 'audit.json').chmod(0o660)
    print(json.dumps({'status': 'within_anchor_replay_audit_passed', 'audit_sha256': sha256(OUT / 'audit.json')}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
