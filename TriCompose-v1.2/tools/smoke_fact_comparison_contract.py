"""Typed authored interface fixtures only; no model, patient or bank input.

Not a clinical benchmark. Reuses synthetic test helper graphs to exercise the
native adapter; preserves historical manifests and never overwrites a run.
"""
import json
import os
from pathlib import Path
import time

from prepare_ratescore_assets import WORKSPACE, private_dir, require, sha256, write_json
from tricompose_v12.fact_comparison_contract import compare_facts, compare_native_reports, digest, from_native_report
from tricompose_v12.radgraph_literal_evidence import extract
from test_fact_comparison_contract import fixture
from test_radgraph_literal_evidence import graph, POS, NEG, ANAT

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
ROOT = BASE / 'fact_contract_smokes'
OUT = ROOT / 'authored_interface_12714150_001'


def main():
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(), 'existing_cpu_allocation_required')
    require(not OUT.exists() and not OUT.is_symlink(), 'refuse_existing_fixture_run')
    started = time.monotonic()
    paths = [Path(__file__).resolve(),
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/fact_comparison_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_literal_evidence.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        WORKSPACE / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_fact_comparison_contract.py',
        WORKSPACE / 'TriCompose-v1.2/tests/test_radgraph_literal_evidence.py',
        BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json',
        BASE / 'radgraph_literal_evidence_runs/literal_pool960_12714150_001/manifest.json',
        BASE / 'radeval_medcpt_runs/reportref_12714150_001/manifest.json']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    os.umask(0o007)
    private_dir(ROOT)
    private_dir(OUT, fresh=True)
    write_json(OUT / 'frozen_plan.json', {'pins': pins, 'input_scope': 'invented_typed_and_native_graph_fixtures',
        'clinical_gold': False, 'raw_source_or_generated_patient_inputs_read': False, 'models_executed': 0})
    specs = [
        ('positive', fixture(), fixture(), 'polarity', 'attribute_agreement_proposal'),
        ('negative', fixture(polarity='negative'), fixture(polarity='negative'), 'polarity', 'attribute_agreement_proposal'),
        ('opposition', fixture(), fixture(polarity='negative'), 'polarity', 'polarity_opposition_proposal'),
        ('unknown', fixture(), fixture(polarity='unknown'), 'polarity', 'unknown_not_comparable'),
        ('uncertain', fixture(), fixture(polarity='uncertain'), 'polarity', 'uncertain_not_determinate'),
        ('prior', fixture(), fixture(temporality='prior'), 'polarity', 'not_comparable'),
        ('hypothetical', fixture(), fixture(temporality='hypothetical'), 'polarity', 'not_comparable'),
        ('family', fixture(), fixture(experiencer='family'), 'polarity', 'not_comparable'),
        ('laterality', fixture(laterality='left'), fixture(laterality='right'), 'polarity', 'anatomical_context_not_comparable'),
        ('severity', fixture(severity='mild'), fixture(severity='severe'), 'severity', 'attribute_difference_proposal'),
        ('weak', fixture(modality='ehr', strength='weak_context'), fixture(polarity='negative'), 'polarity', 'not_comparable'),
        ('concept', fixture(concept='authored_a'), fixture(concept='authored_b'), 'polarity', 'not_comparable'),
    ]
    checks = []
    for index, (kind, a, b, attribute, expected) in enumerate(specs):
        result = compare_facts(a, b)
        require(result['comparisons'][attribute]['relation'] == expected and result['clinical_score'] is None
                and result['confirmed_faulty_modality'] is None and all(v is False for v in result['policy'].values()),
                'authored_contract_check_required')
        checks.append({'fixture_id': f'fixture_{index:03d}', 'kind': kind, 'attribute': attribute,
                       'expected_typed_relation': expected, 'comparison': result})
    def native(source, key):
        return from_native_report(source, case_id='case_000', report_sha256=digest(['authored_report', key]),
                                  expected_native_graph_sha256=extract(source)['native_graph_sha256'])
    simple = native(graph([('authored_finding', POS, [])]), 'a')
    inverse = native(graph([('authored_finding', NEG, [])]), 'b')
    polarity = compare_native_reports(simple, inverse)
    require(polarity['presence']['counts']['explicit_polarity_opposition_proposal'] == 1,
            'native_polarity_proposal_preserved')
    a = native(graph([('authored_finding', POS, [['located_at', '2']]), ('left', ANAT, [])]), 'c')
    b = native(graph([('authored_finding', NEG, [['located_at', '2']]), ('right', ANAT, [])]), 'd')
    anatomy = compare_native_reports(a, b)
    require(anatomy['presence']['counts']['explicit_polarity_opposition_proposal'] == 0 and
            anatomy['anatomy']['different_context_concepts'] == 1, 'native_anatomy_partition_preserved')
    write_json(OUT / 'typed_checks.json', checks)
    write_json(OUT / 'native_checks.json', {'polarity': polarity, 'anatomy': anatomy})
    require(all(sha256(WORKSPACE / p) == value for p, value in pins.items()), 'frozen_source_or_old_result_changed')
    summary = {'status': 'complete', 'typed_fixture_comparisons': 12, 'native_fixture_comparisons': 2,
        'checked_dimensions': ['polarity', 'temporality', 'experiencer', 'laterality', 'severity', 'location'],
        'actual_ehr_cxr_report_evaluation_performed': False, 'native_model_calls': 0,
        'source_or_generated_patient_artifacts_read': False, 'new_slurm_submissions': 0,
        'clinical_qualified': False, 'historical_bank_scored': False,
        'selection_changed': False, 'regeneration_authorized': False, 'clinical_score': None,
        'fixture_gold_is_investigator_authored_not_independent': True,
        'runtime_seconds': time.monotonic()-started}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'manifest.json', {'schema_version': 'fact-comparison-authored-smoke-receipt-v1',
        'pins': pins, 'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(OUT.glob('*.json'))]})
    print(json.dumps({'status': 'authored_fact_contract_smoke_complete',
        'runtime_seconds': round(summary['runtime_seconds'], 3), 'manifest_sha256': sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'authored_fact_contract_smoke_failed', 'error_type': type(error).__name__}))
        raise SystemExit(2)
