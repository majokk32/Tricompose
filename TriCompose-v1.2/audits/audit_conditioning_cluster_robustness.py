#!/usr/bin/env python3
"""Independent metadata/group/statistics audit; no worker or model imports."""
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
BASE = WORKSPACE/'artifacts/protected/tricompose_v1_2'
ROOT = BASE/'conditioning_robustness_runs/conditioning_groups_12714150_001'
DELIVERY = BASE/'deliverables/first_version_12714150_001'
EXPECTED = '0d6e3ecf79c5a99cd500ee74cb3fee23695cd447ff768bd1c848593d04ce985c'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def same(actual, expected):
    assert actual is None if expected is None else actual is not None and math.isclose(actual, expected, abs_tol=1e-12)


def mean(values):
    return sum(values)/len(values) if values else None


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lo, hi = math.floor(position), math.ceil(position)
    return (1-(position-lo))*values[lo]+(position-lo)*values[hi]


def statistics(values, group_for_case):
    groups = defaultdict(list)
    for case, value in values:
        groups[group_for_case[case]].append(value)
    return groups, mean([v for _, v in values]), mean([mean(v) for v in groups.values()])


def bootstrap(groups):
    keys = sorted(groups)
    rng = random.Random(0)
    weighted, equal = [], []
    # Independent implementation: sample group multiplicities, then pool sums.
    group_stats = {key: (sum(groups[key]), len(groups[key])) for key in keys}
    for _ in range(1000):
        counts = dict.fromkeys(keys, 0)
        for _ in keys:
            counts[keys[rng.randrange(len(keys))]] += 1
        weighted.append(sum(group_stats[k][0]*counts[k] for k in keys)/
                        sum(group_stats[k][1]*counts[k] for k in keys))
        equal.append(sum(group_stats[k][0]/group_stats[k][1]*counts[k] for k in keys)/len(keys))
    return [percentile(weighted, q) for q in (.025, .975)], [percentile(equal, q) for q in (.025, .975)]


def main():
    job = os.environ.get('SLURM_JOB_ID', '')
    assert job.isdigit() and f'/job_{job}/' in Path('/proc/self/cgroup').read_text()
    assert digest(ROOT/'manifest.json') == EXPECTED
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for source, expected in manifest['sources'].items():
        assert digest(Path(source)) == expected
    for name, expected in manifest['artifacts'].items():
        assert digest(ROOT/name) == expected
    summary = json.loads((ROOT/'summary.json').read_text())
    source_index = csv_rows(DELIVERY/'candidate_index.csv')
    assert len(source_index) == 960
    model_prompts = defaultdict(dict)
    for row in source_index:
        case, model, prompt = row['case_id'], row['cxr_model_id'], row['prompt_sha256']
        assert model_prompts[case].setdefault(model, prompt) == prompt
    signatures, expected_mapping = {}, {}
    for case in sorted(model_prompts):
        key = tuple(sorted(model_prompts[case].items()))
        gid = signatures.setdefault(key, 'conditioning_%04d' % len(signatures))
        expected_mapping[case] = gid
    assert len(expected_mapping) == 80 and len(signatures) == 49
    actual_mapping = {r['case_id']: r['conditioning_group_id'] for r in csv_rows(ROOT/'case_groups.csv')}
    assert actual_mapping == expected_mapping
    raw = csv_rows(DELIVERY/'baseline_case_means.csv')
    assert len(raw) == 2000
    means = {(r['case_id'], r['method'], int(r['model_call_budget'])): r for r in raw}
    assert len(means) == 2000
    value = lambda row, name: None if row[name] == '' else float(row[name])
    for row in csv_rows(ROOT/'method_means.csv'):
        cap = int(row['model_call_budget'])
        values = [(case, value(means[case, row['method'], cap], row['metric'])) for case in sorted(actual_mapping)]
        available = [(case, v) for case, v in values if v is not None]
        groups, weighted, equal = statistics(available, actual_mapping)
        assert int(row['attempted_ehr_cases']) == 80 and int(row['available_ehr_cases']) == len(available)
        assert int(row['available_conditioning_groups']) == len(groups)
        same(value(row, 'case_weighted_mean'), weighted)
        same(value(row, 'equal_conditioning_mean'), equal)
    contrasts = json.loads((ROOT/'paired_contrasts.json').read_text())['records']
    csv_contrasts = csv_rows(ROOT/'paired_contrasts.csv')
    assert len(contrasts) == len(csv_contrasts) == summary['paired_contrast_rows'] == 275
    intervals = 0
    for row, csv_row in zip(contrasts, csv_contrasts):
        for k, v in row.items():
            assert csv_row[k] == ('' if v is None else json.dumps(v, separators=(',', ':')) if isinstance(v, list) else str(v))
        cap, left, right, metric = row['model_call_budget'], row['left_method'], row['right_method'], row['metric']
        available = []
        for case in sorted(actual_mapping):
            a, b = value(means[case, left, cap], metric), value(means[case, right, cap], metric)
            if a is not None and b is not None:
                available.append((case, a-b))
        groups, weighted, equal = statistics(available, actual_mapping)
        assert row['attempted_ehr_cases'] == 80 and row['available_ehr_cases'] == len(available)
        assert row['available_conditioning_groups'] == len(groups)
        same(row['case_weighted_mean'], weighted); same(row['equal_conditioning_mean'], equal)
        if len(groups) < 3:
            assert row['bootstrap_resamples_used'] == 0
            assert row['case_weighted_interval_95'] is row['equal_conditioning_interval_95'] is None
        else:
            ci_case, ci_equal = bootstrap(groups)
            assert row['bootstrap_resamples_used'] == 1000 and row['bootstrap_seed'] == 0
            for actual, expected in zip(row['case_weighted_interval_95'], ci_case): same(actual, expected)
            for actual, expected in zip(row['equal_conditioning_interval_95'], ci_equal): same(actual, expected)
            intervals += 2
        if left == 'random' and right == 'random_acquisition_score_free_final' and metric == 'mean_simulated_calls':
            assert row['case_weighted_mean'] == row['equal_conditioning_mean'] == 0
        assert row['same_acquisition_and_expenditure'] == (left == 'random' and right == 'random_acquisition_score_free_final')
    for name in ('clinical_qualified', 'primary_metric_eligible', 'selection_changed', 'regeneration_authorized'):
        assert summary['policy'][name] is manifest['policy'][name] is False
    entries = list(ROOT.rglob('*'))+[ROOT, ROOT.parent]
    for path in entries:
        metadata = path.stat()
        assert metadata.st_gid in (96293, 65534)
        assert stat.S_IMODE(metadata.st_mode) == (0o2770 if path.is_dir() else 0o660)
    for name, expected in (
        ('first_version_12714150_001.tar.gz', 'e5538b3ff6537396c0e1fd2ed3ef41fc9df04996e0fdfdab95d5b0ebf9307413'),
        ('first_version_samples_12714150_002.tar.gz', '374c827cece49d03dd2c9ddcf6081f233ccbbc3b1fb615096ad8826e8cc97ace')):
        assert digest(BASE/'deliverables'/name) == expected
    print(json.dumps({'status': 'independent_conditioning_group_audit_passed', 'all_fixed_ehr_cases': 80,
        'conditioning_groups': 49, 'method_mean_rows_checked': 275, 'paired_contrast_rows_checked': 275,
        'intervals_checked': intervals, 'source_pins': len(manifest['sources']), 'protected_entries': len(entries),
        'prior_delivery_archives_unchanged': True, 'new_model_calls': 0, 'clinical_bodies_read': False}))


if __name__ == '__main__':
    main()
