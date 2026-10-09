"""Independent stdlib-only replay of authored numeric MedCPT artifacts.

No production probe/scoring module import, model call, clinical input or fitting.
Exact inference replay is recorded by the worker; this audit does not rerun it.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
RUN = WORKSPACE / 'artifacts/protected/tricompose_v1_2/medcpt_authored_runs/query_12714150_001'
ASSETS = WORKSPACE / 'runtime/models/medcpt-query-v12-12714150-001'
OUT = RUN.parent / 'query_12714150_001_audit'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition):
    if not condition:
        raise ValueError('authored_numeric_replay_mismatch')


def audit():
    started = time.monotonic()
    check('/job_12714150/' in Path('/proc/self/cgroup').read_text())
    check(not OUT.exists() and not OUT.is_symlink())
    manifest = load(RUN / 'manifest.json')
    for path, expected in manifest['pins'].items():
        check(digest(WORKSPACE / path) == expected)
    check({item['path'] for item in manifest['artifacts']} ==
          {'frozen_plan.json', 'score_table.json', 'embeddings.json', 'token_receipts.json', 'summary.json'})
    for item in manifest['artifacts']:
        check(digest(RUN / item['path']) == item['sha256'])
    check(digest(ASSETS / 'asset_manifest.json') == manifest['asset_manifest_sha256'])
    assets = load(ASSETS / 'asset_manifest.json')
    check(len(assets['assets']) == 11 and assets['implicit_token_used'] is False)
    for item in assets['assets']:
        path = ASSETS / item['relative_path']
        check(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'])
    for name in ('README.md', 'LICENSE'):
        check(digest(WORKSPACE / 'MedCPT' / name) == digest(ASSETS / 'source' / name))
    plan = load(RUN / 'frozen_plan.json')['authored_inventory']
    rows = load(RUN / 'score_table.json')
    vectors = load(RUN / 'embeddings.json')['vectors_by_text_sha256']
    receipts = load(RUN / 'token_receipts.json')
    summary = load(RUN / 'summary.json')
    check(len(rows) == len(plan['pairs']) == 41 and len(vectors) == len(receipts) == 38)
    check({r['text_sha256'] for r in receipts} == set(vectors))
    for value in vectors.values():
        check(len(value) == 768 and all(type(x) in (int, float) and math.isfinite(x) for x in value))
    for receipt in receipts:
        check(1 <= receipt['native_token_count'] <= 64 and receipt['truncated'] is False and
              receipt['adapter_added_prefix'] is False and len(receipt['token_ids_sha256']) == 64)
    by_variant, by_group = defaultdict(list), defaultdict(dict)
    for pair, row in zip(plan['pairs'], rows):
        check(all(row[key] == value for key, value in pair.items()))
        check(row['clinical_score'] is None and row['clinical_qualified'] is False and
              row['regeneration_authorized'] is False)
        if not pair['input_nonempty']:
            check(row['status'] == 'empty_input_not_comparable' and row['query_query_cosine'] is None)
            continue
        left, right = vectors[pair['reference_sha256']], vectors[pair['candidate_sha256']]
        # Separate implementation: ordinary sum, no production cosine helper.
        value = sum(x * y for x, y in zip(left, right)) / (
            math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right)))
        check(row['status'] == 'complete' and math.isclose(value, row['query_query_cosine'], abs_tol=1e-12))
        if row['variant'] == 'identity':
            check(abs(value - 1) < 1e-6)
        by_variant[row['variant']].append(row['query_query_cosine'])
        by_group[row['group']][row['variant']] = row['query_query_cosine']
    check(dict(Counter(r['status'] for r in rows)) == summary['status_counts'])
    check(set(by_variant) == set(summary['cosine_by_variant']))
    for kind, values in by_variant.items():
        record = summary['cosine_by_variant'][kind]
        check(record['n'] == len(values) and math.isclose(record['mean'], sum(values) / len(values), abs_tol=1e-12)
              and record['minimum'] == min(values) and record['maximum'] == max(values))
    bases = {name: values for name, values in by_group.items() if 'paraphrase' in values}
    check(len(bases) == 6)
    for kind in ('different_finding', 'negation', 'history', 'hypothetical'):
        counts = {'paraphrase_higher': 0, 'paraphrase_lower': 0, 'tie': 0, 'unavailable': 0}
        for values in bases.values():
            diff = values['paraphrase'] - values[kind]
            key = 'tie' if abs(diff) < 1e-12 else 'paraphrase_higher' if diff > 0 else 'paraphrase_lower'
            counts[key] += 1
        check(counts == summary['authored_ordering_diagnostics_not_clinical_accuracy'][kind])
    check(summary['policy'] == plan['policy'] and summary['policy']['clinical_score'] is None)
    check(all(value is False for key, value in summary['policy'].items() if key != 'clinical_score'))
    check(summary['device'] == 'cpu' and summary['native_forward_passes'] == 10 and
          summary['new_slurm_submissions'] == summary['new_package_installations'] == summary['new_environments'] == 0
          and summary['deterministic_exact_replay'] is True and summary['frozen_parameters'] is True
          and summary['candidate_bank_scored_with_medcpt'] is False)
    for root in (RUN, ASSETS, WORKSPACE / 'MedCPT'):
        for path in (root, *root.rglob('*')):
            check(not path.is_symlink() and path.stat().st_gid in (96293, 65534) and
                  path.stat().st_mode & 0o7777 == (0o2770 if path.is_dir() else 0o660))
    os.umask(0o007)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    result = {'status': 'passed', 'worker_manifest_sha256': digest(RUN / 'manifest.json'),
        'audit_source_sha256': digest(Path(__file__).resolve()), 'pair_count': 41,
        'cosines_independently_recomputed': 40, 'empty_inputs_retained_as_null': 1,
        'asset_files_verified': 11, 'all_frozen_source_and_historical_hashes_unchanged': True,
        'all_variant_means_and_orderings_replayed': True, 'protected_modes_verified': True,
        'exact_inference_replay_independently_rerun': False,
        'exact_inference_replay_worker_receipt_verified': True,
        'model_calls': 0, 'clinical_inputs_read': False, 'clinical_qualified': False,
        'selection_changed': False, 'regeneration_authorized': False,
        'runtime_seconds': time.monotonic() - started}
    target = OUT / 'audit.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o660)
    print(json.dumps({'status': 'independent_authored_audit_passed',
                      'runtime_seconds': round(result['runtime_seconds'], 3), 'audit_sha256': digest(target)}))


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        print(json.dumps({'status': 'independent_authored_audit_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
