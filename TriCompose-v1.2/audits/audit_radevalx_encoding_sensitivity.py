#!/usr/bin/env python3
"""Independent stdlib numerical audit; no worker, adapter or report imports."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
BASE = WORKSPACE/'artifacts/protected/tricompose_v1_2'
ROOT = BASE/'report_metric_encoding_runs/radevalx_encoding_12714150_001'
SOURCE = BASE/'report_metric_alignment_runs/radevalx_published_v2_12714150_001'
EXPECTED = '8f7f444b92953b0d8c992219ce14d47a66c197bb63adce1f0372cb5a39ce0b99'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def rank(values):
    ordered = sorted(values)
    return [(ordered.index(v)+1 + len(ordered)-list(reversed(ordered)).index(v))/2 for v in values]


def rho(left, right):
    if len(left) < 3:
        return None
    left, right = rank(left), rank(right)
    n = len(left)
    numerator = n*sum(a*b for a, b in zip(left, right))-sum(left)*sum(right)
    denominator = math.sqrt((n*sum(a*a for a in left)-sum(left)**2)*
                            (n*sum(b*b for b in right)-sum(right)**2))
    return numerator/denominator if denominator else None


def tau(left, right):
    signed = sum((1 if (left[i]-left[j])*(right[i]-right[j]) > 0 else -1)
        for i in range(len(left)) for j in range(i) if left[i] != left[j] and right[i] != right[j])
    pairs = len(left)*(len(left)-1)/2
    untied = [pairs-sum(v*(v-1)/2 for v in Counter(values).values()) for values in (left, right)]
    denominator = math.sqrt(untied[0]*untied[1])
    return signed/denominator if denominator else None


def same(actual, expected):
    assert actual is None if expected is None else actual is not None and math.isclose(actual, expected, abs_tol=1e-12)


def interval(left, right):
    rng = random.Random(0)
    values = []
    for _ in range(1000):
        indices = [rng.randrange(len(left)) for _ in left]
        value = rho([left[i] for i in indices], [right[i] for i in indices])
        if value is not None:
            values.append(value)
    values.sort()
    result = []
    for proportion in (.025, .975):
        position = (len(values)-1)*proportion
        low, high = math.floor(position), math.ceil(position)
        result.append(values[low]+(values[high]-values[low])*(position-low))
    return result, len(values)


def main():
    job = os.environ.get('SLURM_JOB_ID', '')
    assert job.isdigit() and f'/job_{job}/' in Path('/proc/self/cgroup').read_text()
    assert digest(ROOT/'manifest.json') == EXPECTED
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, expected in manifest['sources'].items():
        assert digest(Path(path)) == expected
    for name, expected in manifest['artifacts'].items():
        assert digest(ROOT/name) == expected
    summary = json.loads((ROOT/'summary.json').read_text())
    p = json.loads((SOURCE/'published_predictions.json').read_text())
    r = json.loads((SOURCE/'references.json').read_text())
    assert len(p['records']) == len(r['records']) == 100
    assert sum(v is None for row in r['records'] for counts in row['errors'].values() for v in counts) == 1423
    totals = []
    for prediction, reference in zip(p['records'], r['records']):
        assert prediction['item_id'] == reference['item_id']
        assert prediction['source_group_id'] == reference['source_group_id']
        sig = reference['errors']['clinically_significant']
        insig = reference['errors']['clinically_insignificant']
        totals.append((sum(v for v in sig if v is not None), sum(v for v in sig+insig if v is not None)))
    assert sum(total > 3 for _, total in totals) == summary['hypothetical_noisy_subset_pairs'] == 30
    checks = 0
    for scenario, cohorts in summary['scenarios'].items():
        for cohort, details in cohorts.items():
            ids = [i for i, (_, total) in enumerate(totals) if cohort == 'all100' or total > 3]
            assert len(ids) == details['attempted_pairs']
            for metric, outcomes in details['metrics'].items():
                sign = -1 if p['metric_definitions'][metric]['orientation'] == 'lower_is_better' else 1
                for outcome, result in outcomes.items():
                    index = 0 if outcome == 'clinically_significant_total' else 1
                    if scenario == 'blank_is_unresolved':
                        assert result['paired_rows'] == 0 and result['spearman'] is None and result['kendall_tau_b'] is None
                        checks += 1
                        continue
                    left = [sign*p['records'][i]['scores'][metric]['value'] for i in ids]
                    right = [-totals[i][index] for i in ids]
                    assert result['paired_rows'] == len(ids)
                    same(result['spearman'], rho(left, right))
                    same(result['kendall_tau_b'], tau(left, right))
                    if outcome == 'clinically_significant_total' and cohort == 'all100':
                        bounds, draws = interval(left, right)
                        saved = result['conditional_encoding_group_bootstrap']
                        assert saved['usable_resamples'] == draws and saved['seed'] == 0 and draws >= 900
                        for actual, expected in zip(saved['spearman_interval_95'], bounds):
                            same(actual, expected)
                    checks += 1
    for row in summary['paper_table4_reproduction']:
        actual = summary['scenarios']['hypothetical_blank_is_zero'][row['cohort']]['metrics'][row['metric']][row['outcome']]['spearman']
        same(row['hypothetical_spearman'], actual)
        same(row['signed_deviation'], actual-row['paper_spearman'])
        assert row['matches_paper_rounding'] == (abs(actual-row['paper_spearman']) <= .000050000001)
    assert summary['matching_published_values'] == 0 and summary['paper_values_compared'] == 8
    for name in ('encoding_verified', 'clinical_qualified', 'local_implementation_qualified',
                 'primary_metric_eligible', 'selection_changed', 'regeneration_authorized'):
        assert summary['policy'][name] is False and manifest['policy'][name] is False
    entries = list(ROOT.rglob('*'))+[ROOT, ROOT.parent]
    for path in entries:
        metadata = path.stat()
        assert metadata.st_gid in (96293, 65534)
        assert stat.S_IMODE(metadata.st_mode) == (0o2770 if path.is_dir() else 0o660)
    for name, expected in (
        ('first_version_12714150_001.tar.gz', 'e5538b3ff6537396c0e1fd2ed3ef41fc9df04996e0fdfdab95d5b0ebf9307413'),
        ('first_version_samples_12714150_002.tar.gz', '374c827cece49d03dd2c9ddcf6081f233ccbbc3b1fb615096ad8826e8cc97ace')):
        assert digest(BASE/'deliverables'/name) == expected
    print(json.dumps({'status': 'independent_encoding_numeric_audit_passed',
        'outcome_checks': checks, 'paper_target_checks': 8, 'source_pins': len(manifest['sources']),
        'protected_entries': len(entries), 'previous_delivery_archives_unchanged': True,
        'report_bodies_read': False, 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
