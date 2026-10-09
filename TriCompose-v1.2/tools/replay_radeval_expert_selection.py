"""Derived expert benchmark scores only; no raw input/model/actual selection."""
import json
import os
from pathlib import Path

import smoke_radgraph_xl as installed
from tricompose_v12.radeval_expert_selection import replay, summarize

ROOT = installed.WORKSPACE
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'radeval_expert_radgraph_runs/expert_radgraph_12714150_001'
OUT = BASE / 'radeval_expert_selection_runs/within_anchor_12714150_001'
EXPECTED_SOURCE = '21e1724a5b364eee97b56cdb8f66c0d56194baadc98c333b5c473f7efb37038d'


def execute():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_existing_cpu_allocation_required')
    if installed.sha256(SOURCE / 'manifest.json') != EXPECTED_SOURCE:
        raise ValueError('sealed_expert_score_source_required')
    receipt = json.loads((SOURCE / 'manifest.json').read_text())
    for item in receipt['artifacts']:
        if installed.sha256(SOURCE / item['path']) != item['sha256']:
            raise ValueError('sealed_expert_artifact_mismatch')
    plan = json.loads((SOURCE / 'frozen_plan.json').read_text())['inventory']
    scores = json.loads((SOURCE / 'scores.json').read_text())
    paths = [Path(__file__).resolve(), ROOT / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert_selection.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radeval_expert.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/report_metric_alignment.py',
        ROOT / 'TriCompose-v1.2/tests/test_radeval_expert_selection.py',
        SOURCE / 'manifest.json', SOURCE / 'frozen_plan.json', SOURCE / 'scores.json']
    pins = [{'path': str(p.relative_to(ROOT)), 'sha256': installed.sha256(p)} for p in paths]
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    installed.write_json(OUT / 'frozen_plan.json', {'post_hoc_diagnostic': True,
        'source_benchmark_manifest': EXPECTED_SOURCE, 'pins': pins,
        'metrics': list(scores[0]['scores']), 'targets': ['clinically_significant_total', 'all_errors_total'],
        'resamples': 1000, 'seed': 0, 'new_model_calls': 0,
        'all_three_counts_required': True, 'top_score_ties_uniform_expected_choice': True})
    replayed = replay(plan, scores)
    result = summarize(replayed, resamples=1000, seed=0)
    installed.write_json(OUT / 'replay.json', replayed)
    installed.write_json(OUT / 'summary.json', result)
    if not all(installed.sha256(ROOT / p['path']) == p['sha256'] for p in pins):
        raise ValueError('source_changed')
    installed.write_json(OUT / 'manifest.json', {'schema_version': 'radeval-expert-within-anchor-receipt-v1',
        'pins': pins, 'post_hoc_diagnostic': True, 'new_model_calls': 0, 'selection_changed': False,
        'outputs': [{'path': p.name, 'sha256': installed.sha256(p)} for p in sorted(OUT.glob('*.json'))]})
    print(json.dumps({'status': 'within_anchor_expert_ranking_replay_complete',
                      'manifest_sha256': installed.sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
