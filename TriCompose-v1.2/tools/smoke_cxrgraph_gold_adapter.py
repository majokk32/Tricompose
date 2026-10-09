"""Authored CXRGraph-schema fixtures only; no clinical files or model calls.

Source graph shapes are invented test fixtures, not the gated manual dataset.
Requires an existing genuine Slurm allocation and refuses run replacement.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import resource
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, sha256, write_json
from test_cxrgraph_gold_adapter import document
from tricompose_v12.cxrgraph_gold_adapter import (
    POLICY, SOURCE, normalize_manual_document, validate_adapter,
)
from tricompose_v12.entity_gold_contract import digest

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'


def main():
    require(len(os.sys.argv) == 1, 'authored_worker_accepts_no_clinical_input_paths')
    job = os.environ.get('SLURM_JOB_ID', '')
    require(re.fullmatch(r'[1-9][0-9]{0,11}', job) and
        re.search(r'/job_' + re.escape(job) + r'(?:/|\n|$)', Path('/proc/self/cgroup').read_text()),
        'actual_existing_cpu_slurm_allocation_required')
    root = BASE / 'cxrgraph_adapter_smokes'
    out = root / ('authored_' + job + '_001')
    require(not out.exists() and not out.is_symlink(), 'refuse_existing_adapter_smoke')
    started = time.monotonic()
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/cxrgraph_gold_adapter.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_cxrgraph_gold_adapter.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/entity_gold_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json',
        BASE / 'entity_gold_contract_smokes/authored_12766754_001/manifest.json']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    os.umask(0o007)
    private_dir(root)
    private_dir(out, fresh=True)
    write_json(out / 'frozen_plan.json', {'schema_version': 'authored-cxrgraph-adapter-plan-v1',
        'pins': pins, 'official_source_definition': SOURCE,
        'input_scope': 'investigator_authored_manual_schema_fixtures',
        'human_dataset_available': False, 'model_calls': 0, 'policy': dict(POLICY)})
    specs = []
    original = document()
    specs.append(('global_offsets_and_bound_attributes', original, 3, 2, 3, 2, 5))
    negative = deepcopy(original)
    negative['ner'][0][0][2] = 'Observation-Absent'
    negative['entity_attributes'][0][0][4] = 'Positive'
    specs.append(('change_not_finding_polarity', negative, 3, 2, 3, 2, 5))
    missing = deepcopy(original)
    missing['entity_attributes'] = [[], []]
    specs.append(('unassigned_attributes', missing, 3, 2, 3, 2, 0))
    chain = deepcopy(original)
    chain['ner'][1][1][2] = 'Location-Attribute'
    chain['relations'] = [[[0, 0, 3, 3, 'located_at']], [[3, 3, 2, 2, 'located_at']]]
    chain['entity_attributes'] = [[], []]
    specs.append(('location_chain_not_flattened', chain, 3, 2, 2, 0, 0))
    part = deepcopy(original)
    part['relations'][1][0][4] = 'part_of'
    specs.append(('part_of_not_modify', part, 3, 2, 3, 1, 5))
    empty = deepcopy(original)
    empty['ner'] = empty['relations'] = empty['entity_attributes'] = [[], []]
    specs.append(('valid_empty_annotations', empty, 0, 0, 0, 0, 0))
    adapters, checks = [], []
    for index, (kind, raw, ne, nr, ce, cr, na) in enumerate(specs):
        kwargs = {'report_id': f'report_{index:04d}',
            'source_report_sha256': digest(['authored_cxrgraph_source', index]),
            'origin': 'authored_fixture'}
        value = normalize_manual_document(raw, **kwargs)
        validate_adapter(value)
        require(value == normalize_manual_document(raw, **kwargs), 'deterministic_manual_adapter_required')
        observed = (len(value['native_entities']), len(value['native_relations']),
            len(value['common_label_record']['entities']), len(value['common_label_record']['relations']),
            len(value['native_attributes']))
        require(observed == (ne, nr, ce, cr, na), 'predeclared_native_and_projected_inventory_required')
        require(value['policy'] == POLICY and value['clinical_score'] is None
            and all(n is None for n in value['native_extra_dimension_scores'].values()),
            'no_extra_dimension_score_or_clinical_qualification')
        if kind == 'change_not_finding_polarity':
            require(value['common_label_record']['entities'][0][2] == 'Observation::definitely absent',
                'native_change_is_not_polarity')
        adapters.append(value)
        checks.append({'fixture_id': f'fixture_{index:03d}', 'kind': kind,
            'predeclared_counts': {'native_entities': ne, 'native_relations': nr,
                'common_entities': ce, 'common_relations': cr, 'assigned_attributes': na},
            'adapter_sha256': value['adapter_sha256']})
    write_json(out / 'adapters.json', adapters)
    write_json(out / 'fixture_checks.json', checks)
    require(all(sha256(WORKSPACE / key) == value for key, value in pins.items()),
        'source_or_historical_manifest_changed')
    summary = {'status': 'authored_cxrgraph_adapter_smoke_complete', 'authored_fixtures': 6,
        'predeclared_inventory_checks': 30, 'models_executed': 0, 'new_slurm_submissions': 0,
        'raw_or_synthetic_patient_artifacts_read': False, 'manual_dataset_downloaded': False,
        'human_gold_evaluated': False, 'official_cxrgraph_metric_run': False,
        'policy': dict(POLICY), 'runtime_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)}
    write_json(out / 'summary.json', summary)
    write_json(out / 'manifest.json', {'schema_version': 'cxrgraph-authored-adapter-smoke-v1',
        'pins': pins, 'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(out.glob('*.json'))]})
    print(json.dumps({'status': summary['status'], 'runtime_seconds': round(summary['runtime_seconds'], 3),
        'peak_rss_gib': round(summary['peak_rss_gib'], 3), 'manifest_sha256': sha256(out / 'manifest.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'authored_cxrgraph_adapter_smoke_failed',
            'error_type': type(error).__name__}))
        raise SystemExit(2)
