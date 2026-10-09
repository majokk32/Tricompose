"""Numerical output-only replay in the existing CPU allocation; no model IO."""
import hashlib
import json
import os
from pathlib import Path
import time

from tricompose_v12.radeval_error_types import diagnose
from tricompose_v12.radeval_expert import require
from tricompose_v12.radeval_image_benchmark import METRICS

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'radeval_image_benchmark_runs/biovil_cpu_12714150_001'
AUDIT = BASE / 'radeval_image_benchmark_audits/biovil_cpu_12714150_001/audit.json'
OUT = BASE / 'radeval_error_type_diagnostics/error_types_12714150_001'
SOURCE_HASH = '8e55b223ef22c5da61c20aeb0c53f818ab6c1736eeb2c95ee85b39803ea0c18a'
AUDIT_HASH = 'f9af2ec5a0a065e4fc92e174defc8fba474898ee26552a5ca053d98a01e1d1b1'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def write_json(name, value):
    path = OUT / name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o660)


def execute():
    started = time.monotonic()
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(),
            'existing_cpu_allocation_required')
    require(not OUT.exists(), 'fresh_diagnostic_run_required')
    require(digest(SOURCE / 'manifest.json') == SOURCE_HASH and digest(AUDIT) == AUDIT_HASH,
            'audited_frozen_image_score_source_required')
    audit = json.loads(AUDIT.read_text())
    require(audit['status'] == 'passed' and audit['run_manifest_sha256'] == SOURCE_HASH,
            'passed_source_audit_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    artifacts = {r['path']: r['sha256'] for r in manifest['artifacts']}
    for name in ('paired_score_table.json', 'evaluation.json'):
        require(digest(SOURCE / name) == artifacts[name], 'unchanged_derived_inputs_required')
    paths = [Path(__file__).resolve(),
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radeval_error_types.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_benchmark.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/report_metric_alignment.py',
        ROOT / 'TriCompose-v1.2/tests/test_radeval_error_types.py',
        ROOT / 'docs/radeval_error_type_diagnostic_protocol.md',
        SOURCE / 'manifest.json', SOURCE / 'paired_score_table.json',
        SOURCE / 'evaluation.json', AUDIT]
    pins = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in paths]
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    write_json('frozen_plan.json', {'post_hoc_descriptive_diagnostic': True,
        'pins': pins, 'resamples': 1000, 'seed': 0, 'severity': 'clinically_significant',
        'all_seven_categories_and_four_metrics_required': True,
        'new_model_calls': 0, 'raw_patient_source_access': False, 'selection_changed': False})
    records = json.loads((SOURCE / 'paired_score_table.json').read_text())
    reference = json.loads((SOURCE / 'evaluation.json').read_text())
    result = diagnose(records, resamples=1000, seed=0)
    require(result == diagnose(list(reversed(records)), resamples=1000, seed=0),
            'deterministic_order_invariant_replay_required')
    additivity_checks = 0
    for metric in METRICS:
        original = next(r for r in reference['selection_diagnostic'] if
                        r['metric'] == metric and r['target'] == 'clinically_significant_total')
        components = [r for r in result['results'] if r['metric'] == metric]
        # Counts here are complete for the identical subset; never substitute
        # a per-category denominator for an incomplete total-error anchor.
        require(all(r['complete_anchors'] == original['complete_anchors'] and
                    r['score_and_category_available_pairs'] == reference['common_score_available_pairs']
                    for r in components), 'identical_category_and_total_denominators_required')
        for new_key, old_key in (('selected_errors', 'selected_expected_errors'),
                                 ('random_errors', 'random_expected_errors'),
                                 ('metric_minus_random_errors', 'metric_minus_random_errors')):
            require(abs(sum(r['means'][new_key] for r in components) - original['means'][old_key]) < 1e-12,
                    'category_components_must_reproduce_original_total')
            additivity_checks += 1
    write_json('diagnostic.json', result)
    require(all(digest(ROOT / p['path']) == p['sha256'] for p in pins), 'frozen_source_changed')
    for path in (OUT, *OUT.iterdir()):
        require(path.stat().st_gid in (96293, 65534) and path.stat().st_mode & 0o7777 ==
                (0o2770 if path.is_dir() else 0o660), 'protected_project_permissions_required')
    write_json('summary.json', {'status': 'complete', 'runtime_seconds': time.monotonic() - started,
        'all_attempted_pairs': result['all_attempted_pairs'],
        'common_score_available_pairs': result['common_score_available_pairs'],
        'error_type_metric_diagnostics': len(result['results']),
        'independent_total_error_additivity_checks': additivity_checks,
        'deterministic_reverse_order_replay': True, 'post_hoc_descriptive_diagnostic': True,
        'new_model_calls': 0, 'raw_patient_source_access': False,
        'clinical_qualified': False, 'selection_changed': False})
    write_json('manifest.json', {'schema_version': 'radeval-error-type-diagnostic-receipt-v1',
        'pins': pins, 'source_manifest_sha256': SOURCE_HASH, 'new_model_calls': 0,
        'selection_changed': False, 'outputs': [{'path': p.name, 'sha256': digest(p)}
                                               for p in sorted(OUT.glob('*.json'))]})
    print(json.dumps({'status': 'error_type_diagnostic_complete',
                      'manifest_sha256': digest(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
