"""Job-bound cached-native-graph diagnostic; zero inference or raw source IO.

Cached native graphs contain restricted text, consumed internally only. No
source CSV, real EHR/image, target report file, network or neural model is used.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time

from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json, private_dir
from tricompose_v12.fact_comparison_contract import from_native_report, compare_native_reports
from tricompose_v12.radgraph_literal_evidence import extract
from tricompose_v12.radgraph_reference_contract_v2 import graph_metadata
from tricompose_v12.radeval_literal_facts import join, evaluate, POLICY, FEATURES, TARGETED

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
CACHE = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
CACHE_SHA = '21e1724a5b364eee97b56cdb8f66c0d56194baadc98c333b5c473f7efb37038d'
INVENTORY = BASE / 'report_metric_sources/radeval_expert_contract_12714150_003/inventory.json'
INVENTORY_SHA = '29c076bc24fd4c628703cbe1a1ea5049cf090660ff06a2c46e09e59e50b49365'
OLD_BANK = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
OLD_BANK_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
OLD_LITERAL = BASE / 'radgraph_literal_evidence_runs/literal_pool960_12714150_001/manifest.json'
OLD_LITERAL_SHA = 'da1bfb5c82f1ce51d000843ee11471c6edddaafc20a5ab075227a4c675eb57c9'
RUN_ID = 'literal_expert_12766754_001'
PLAN = BASE / 'radeval_literal_fact_plans' / RUN_ID
OUT = BASE / 'radeval_literal_fact_runs' / RUN_ID


def pins():
    require(sha256(CACHE / 'manifest.json') == CACHE_SHA and sha256(INVENTORY) == INVENTORY_SHA
        and sha256(OLD_BANK) == OLD_BANK_SHA and sha256(OLD_LITERAL) == OLD_LITERAL_SHA,
        'fixed_existing_sources_required')
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    artifacts = {a['path']: a['sha256'] for a in manifest['artifacts']}
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_literal_facts.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/fact_comparison_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_literal_evidence.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/report_metric_alignment.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_medcpt_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radeval_image_benchmark.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_radeval_literal_facts.py',
        WORKSPACE / 'TriCompose-v1.2/slurm/64_radeval_literal_facts_existing_cpu.sh',
        CACHE / 'manifest.json', INVENTORY, OLD_BANK, OLD_LITERAL]
    for name in ('native_graphs.json', 'graph_receipts.json'):
        path = CACHE / name
        require(not path.is_symlink() and path.is_file() and sha256(path) == artifacts[name],
                'immutable_cached_graph_inputs_required')
        paths.append(path)
    return {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}


def prepare():
    frozen = pins()
    require(not OUT.exists() and not OUT.is_symlink(), 'refuse_existing_run')
    private_dir(PLAN.parent)
    private_dir(PLAN, fresh=True)
    write_json(PLAN / 'plan.json', {'schema_version': 'radeval-literal-fact-plan-v1',
        'pins': frozen, 'run_id': RUN_ID, 'cpu_job': 12766754, 'bootstrap_resamples': 1000, 'seed': 0,
        'features': list(FEATURES), 'targeted_hypotheses': TARGETED, 'policy': dict(POLICY),
        'all_attempted_pairs': 624, 'native_graphs': 762,
        'post_hoc_development_diagnostic': True, 'expert_annotations_previously_inspected': True,
        'untouched_independent_test': False, 'new_model_calls': 0, 'raw_source_csv_read': False,
        'ranking_direction': 'minimize_each_separate_literal_count_not_fitted',
        'exact_minimum_count_ties': 'uniform_expected_choice_not_first_slot',
        'restricted_cached_native_graphs_consumed_internally_at_execution': True,
        'scope_severity_and_experiencer_extraction_performed': False})
    write_json(PLAN / 'manifest.json', {'schema_version': 'radeval-literal-fact-plan-receipt-v1',
        'pins': frozen, 'artifacts': [{'path': 'plan.json', 'sha256': sha256(PLAN / 'plan.json')}]})
    print(json.dumps({'status': 'cached_literal_diagnostic_prepared',
        'manifest_sha256': sha256(PLAN / 'manifest.json')}))


def execute():
    started = time.monotonic()
    require(PLAN.is_dir() and not PLAN.is_symlink() and not OUT.exists() and not OUT.is_symlink(),
            'fresh_run_and_existing_plan_required')
    plan_manifest = json.loads((PLAN / 'manifest.json').read_text())
    require(sha256(PLAN / 'plan.json') == plan_manifest['artifacts'][0]['sha256'], 'frozen_plan_required')
    frozen = json.loads((PLAN / 'plan.json').read_text())
    require(frozen['pins'] == plan_manifest['pins'] == pins() and frozen['policy'] == POLICY
        and frozen['features'] == list(FEATURES) and frozen['targeted_hypotheses'] == TARGETED
        and frozen['bootstrap_resamples'] == 1000 and frozen['seed'] == 0, 'unchanged_declared_protocol_required')
    private_dir(OUT.parent)
    private_dir(OUT, fresh=True)
    inventory = json.loads(INVENTORY.read_text())
    receipts = json.loads((CACHE / 'graph_receipts.json').read_text())
    cached = json.loads((CACHE / 'native_graphs.json').read_text())
    require(cached['scope'] == 'authorized_real_reference_benchmark_only', 'approved_cache_scope_required')
    graphs = cached['graphs']
    require(len(inventory['records']) == 624 and len(inventory['graphs']) == len(receipts) == 762,
            'fixed_complete_attempted_inventory_required')
    expected = {g['graph_id']: g for g in inventory['graphs']}
    by_id = {g['graph_id']: g for g in receipts}
    require(set(expected) == set(by_id) and len(by_id) == len(receipts)
        and set(graphs) == {g['graph_id'] for g in receipts if g['status'] == 'complete'},
        'exact_cached_graph_inventory_required')
    native_hashes = {}
    for graph_id, request in expected.items():
        receipt = by_id[graph_id]
        require(receipt['text_sha256'] == request['text_sha256'], 'original_report_hash_binding_required')
        if receipt['status'] == 'complete':
            require(graph_metadata(graphs[graph_id]) == receipt['metadata'], 'unchanged_cached_graph_metadata_required')
            native_hashes[graph_id] = extract(graphs[graph_id])['native_graph_sha256']
    anchors = sorted({(p['source_id'], p['section_id'], p['reference_sha256']) for p in inventory['records']})
    cases = {key: f'case_{i:03d}' for i, key in enumerate(anchors)}
    adapters = {}
    dependencies = (CACHE_SHA, frozen['pins'][str((CACHE / 'native_graphs.json').relative_to(WORKSPACE))])
    def adapter(graph_id, case_id):
        key = (graph_id, case_id)
        if key not in adapters:
            adapters[key] = from_native_report(graphs[graph_id], case_id=case_id,
                report_sha256=expected[graph_id]['text_sha256'], dependencies=dependencies,
                expected_native_graph_sha256=native_hashes[graph_id])
        return adapters[key]
    predictions = []
    for pair in inventory['records']:
        ids = (pair['hypothesis_graph_id'], pair['reference_graph_id'])
        status = ('empty_input' if not pair['input_nonempty'] else 'complete'
            if all(i in graphs for i in ids) else 'unavailable_graph')
        comparison = None
        if status == 'complete':
            case_id = cases[(pair['source_id'], pair['section_id'], pair['reference_sha256'])]
            comparison = compare_native_reports(*(adapter(i, case_id) for i in ids))
            require(comparison == compare_native_reports(*(adapter(i, case_id) for i in ids)),
                    'deterministic_full_comparison_replay_required')
        predictions.append({'item_id': pair['item_id'], 'status': status, 'comparison': comparison})
    write_json(OUT / 'predictions.json', predictions)
    prediction_sha = sha256(OUT / 'predictions.json')
    write_json(OUT / 'prediction_receipt.json', {'predictions_sha256': prediction_sha,
        'sealed_before_statistical_evaluation': True, 'independently_blinded': False,
        'native_model_calls': 0, 'actual_scope_or_severity_extraction': False})
    records = join(inventory, predictions)
    result = evaluate(records, resamples=1000, seed=0)
    write_json(OUT / 'derived_table.json', records)
    write_json(OUT / 'evaluation.json', result)
    require(sha256(OUT / 'predictions.json') == prediction_sha and pins() == frozen['pins'],
            'historical_inputs_and_new_predictions_unchanged')
    summary = {'status': 'complete', 'policy': dict(POLICY), 'all_attempted_pairs': len(records),
        'pair_status_counts': dict(Counter(r['status'] for r in records)), 'cached_graphs': len(graphs),
        'zero_native_entity_graphs': sum(g['metadata']['entity_count'] == 0 for g in receipts if g['status'] == 'complete'),
        'comparisons_replayed': len(predictions), 'diagnostic_feature_target_cells': len(result['results']),
        'new_model_calls': 0, 'new_slurm_submissions': 0, 'cached_native_graph_text_read_internally': True,
        'raw_source_csv_or_ehr_or_image_or_target_report_file_read': False,
        'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'job_id': 12766754, 'clinical_score': None, 'selection_changed': False,
        'regeneration_authorized': False, 'entity_extraction_gold_available': False,
        'post_hoc_development_diagnostic': True}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'manifest.json', {'schema_version': 'radeval-literal-fact-run-receipt-v1',
        'plan_manifest_sha256': sha256(PLAN / 'manifest.json'), 'pins': frozen['pins'],
        'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(OUT.glob('*.json'))]})
    print(json.dumps({'status': 'cached_literal_diagnostic_complete',
        'runtime_seconds': round(summary['runtime_seconds'], 3),
        'peak_rss_gib': round(summary['peak_rss_gib'], 3),
        'manifest_sha256': sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('mode', choices=('prepare', 'run'))
        args = parser.parse_args()
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ.get('SLURM_JOB_ID') == '12766754', 'existing_cpu_job_required')
        os.umask(0o007)
        prepare() if args.mode == 'prepare' else execute()
    except Exception as error:
        print(json.dumps({'status': 'cached_literal_diagnostic_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
