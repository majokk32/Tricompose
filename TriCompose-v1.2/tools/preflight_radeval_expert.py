"""Approved CPU-only expert annotation contract preflight, no model calls."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from tricompose_v12.radeval_expert import inventory, outcomes

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
SOURCE = WORKSPACE / 'artifacts/protected/tricompose_v1_2/report_metric_sources/radeval_expert_source_12714150_001'
ROOT = WORKSPACE / 'artifacts/protected/tricompose_v1_2/report_metric_sources'


def execute(run_id):
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('approved_existing_allocation_required')
    if not re.fullmatch(r'radeval_expert_contract_12714150_[0-9]{3}', run_id):
        raise ValueError('opaque_run_id_required')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    path = SOURCE / manifest['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('source_hash_mismatch')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    plan = inventory(rows)
    totals = [outcomes(p['errors']) for p in plan['records']]
    summary = {key: value for key, value in plan.items()
               if key not in ('records', 'graphs', 'annotation_cells')}
    summary.update(pair_inventory=len(plan['records']), unique_report_texts=len(plan['graphs']),
        primary_reference_available=sum(t['clinically_significant_total'] is not None for t in totals),
        all_error_reference_available=sum(t['all_errors_total'] is not None for t in totals))
    os.umask(0o007)
    out = ROOT / run_id
    out.mkdir(mode=0o2770)
    out.chmod(0o2770)
    for name, obj in [('inventory.json', plan), ('summary.json', summary)]:
        with (out / name).open('x') as stream:
            json.dump(obj, stream, sort_keys=True, indent=2)
            stream.write('\n')
        (out / name).chmod(0o660)
    print(json.dumps({'status': 'expert_count_contract_preflight_complete',
        'inventory_sha256': hashlib.sha256((out / 'inventory.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    try:
        execute(parser.parse_args().run_id)
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
