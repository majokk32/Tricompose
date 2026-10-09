"""Cached synthetic graphs only: no new model, source reports/EHR or pixels."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from tricompose_v12.radgraph_literal_evidence import RELATIONS, candidate_overlay, compare, extract
from tricompose_v12.radgraph_reference_contract import require

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'radgraph_bank_runs/native_pool960_12714150_001'
AUDIT = BASE / 'radgraph_bank_audits/native_pool960_12714150_001/audit.json'
OUT = BASE / 'radgraph_literal_evidence_runs/literal_pool960_12714150_001'
SOURCE_HASH = 'ebc747392d791595901997af38169c0c95944e5b79786876d6c8db9beac295b6'
AUDIT_HASH = 'd425a731cfbd0f0e633b7f5ff5f273b4e81291aa414be2559d2bb97b86adfca6'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(name, payload):
    path = OUT / name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o660)


def execute():
    started = time.monotonic()
    require('/job_12714150/' in Path('/proc/self/cgroup').read_text(), 'existing_cpu_allocation_required')
    require(not OUT.exists(), 'fresh_literal_evidence_run_required')
    require(sha256(SOURCE / 'manifest.json') == SOURCE_HASH and sha256(AUDIT) == AUDIT_HASH,
            'sealed_synthetic_graphs_and_audit_required')
    audit = json.loads(AUDIT.read_text())
    require(audit['status'] == 'passed' and audit['source_manifest_sha256'] == SOURCE_HASH,
            'passed_synthetic_graph_audit_required')
    receipt = json.loads((SOURCE / 'manifest.json').read_text())
    artifacts = {r['path']: r['sha256'] for r in receipt['artifacts']}
    names = ('frozen_plan.json', 'candidate_score_table.csv', 'graph_receipts.json',
             'native_graphs.json', 'summary.json')
    for name in names:
        require(sha256(SOURCE / name) == artifacts[name], 'sealed_cached_source_changed')
    source_summary = json.loads((SOURCE / 'summary.json').read_text())
    require(source_summary['real_mimic_inputs_read'] is False and source_summary['frozen_parameters'] is True,
            'frozen_fully_synthetic_scope_required')
    plan = json.loads((SOURCE / 'frozen_plan.json').read_text())
    require((plan['case_count'], plan['image_slots'], plan['candidate_slots'],
             plan['unique_report_graphs'], plan['same_image_pairs']) == (80, 240, 960, 428, 1440)
            and plan['policy']['selection_changed'] is False, 'complete_unchanged_development_bank_required')
    paths = [Path(__file__).resolve(),
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radgraph_literal_evidence.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py',
        ROOT / 'TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py',
        ROOT / 'TriCompose-v1.2/tests/test_radgraph_literal_evidence.py',
        ROOT / 'TriCompose-v1.2/tests/test_radgraph_bank.py',
        ROOT / 'docs/radgraph_literal_evidence_protocol.md', SOURCE / 'manifest.json', AUDIT,
        *(SOURCE / name for name in names)]
    pins = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in paths]
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    write_json('frozen_plan.json', {'pins': pins, 'cohort_role': 'inspected_synthetic_development',
        'all_case_image_report_slots_retained': True, 'all_same_image_pairs_retained': True,
        'dictionary_or_synonym_mapping': False, 'new_model_calls': 0,
        'source_patient_inputs_read': False, 'selection_changed': False})
    native = json.loads((SOURCE / 'native_graphs.json').read_text())
    require(native['scope'] == 'synthetic_only', 'synthetic_graph_scope_required')
    old_graphs = json.loads((SOURCE / 'graph_receipts.json').read_text())
    require(len(old_graphs) == len(plan['graphs']) == 428 and set(native['graphs']) ==
            {r['graph_id'] for r in old_graphs if r['status'] == 'complete'}, 'complete_native_graph_inventory_required')
    extracted, graph_rows = {}, []
    for request, old in zip(plan['graphs'], old_graphs):
        require((request['graph_id'], request['report_sha256']) == (old['graph_id'], old['report_sha256']),
                'ordered_report_hash_join_required')
        evidence, reason = None, old['failure_reason']
        if old['status'] == 'complete':
            try:
                evidence = extract(native['graphs'][request['graph_id']])
                require(evidence['tokenized_text_sha256'] == old['metadata']['tokenized_text_sha256'],
                        'native_text_hash_binding_required')
            except Exception as error:
                evidence = None
                reason = 'unavailable_' + type(error).__name__
        extracted[request['graph_id']] = evidence
        graph_rows.append({'graph_id': request['graph_id'], 'report_sha256': request['report_sha256'],
            'status': 'complete' if evidence is not None else 'unavailable_graph',
            'failure_type': reason if evidence is None else None, 'evidence': evidence})
    comparisons = []
    for pair in plan['pairs']:
        a, b = extracted[pair['left_graph_id']], extracted[pair['right_graph_id']]
        if a is not None and b is not None:
            evidence = compare(a, b)
        else:
            evidence = {'status': 'unavailable_graph', 'counts': None, 'clinical_score': None,
                        'clinical_qualified': False, 'regeneration_authorized': False}
        comparisons.append({**pair, **evidence})
    additions = {r['triple_candidate_id']: r for r in candidate_overlay(plan, extracted, comparisons)}
    with (SOURCE / 'candidate_score_table.csv').open(newline='') as stream:
        old_rows = list(csv.DictReader(stream))
    require(len(old_rows) == len(additions) == 960, 'complete_old_row_inventory_required')
    table = []
    for row in old_rows:
        extra = {k: v for k, v in additions[row['triple_candidate_id']].items() if k.startswith('rg_literal_')}
        require(not set(row).intersection(extra), 'append_only_new_columns_required')
        table.append({**row, **extra})
    with (OUT / 'candidate_score_table.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (OUT / 'candidate_score_table.csv').chmod(0o660)
    write_json('graph_literal_evidence.json', graph_rows)
    write_json('same_image_literal_evidence.json', comparisons)
    completed = [r for r in comparisons if r['status'] == 'complete']
    candidate_by_id = {r['triple_candidate_id']: r for r in plan['candidates']}
    pairs_with_opposition = [r for r in completed if r['counts']['explicit_polarity_opposition_proposal']]
    pair_model_totals = defaultdict(lambda: {'pairs': 0, 'pairs_with_opposition_proposals': 0,
                                           'opposition_proposal_atoms': 0})
    for r in completed:
        key = '|'.join(sorted((r['left_report_model_id'], r['right_report_model_id'])))
        row = pair_model_totals[key]
        row['pairs'] += 1
        row['pairs_with_opposition_proposals'] += bool(r['counts']['explicit_polarity_opposition_proposal'])
        row['opposition_proposal_atoms'] += r['counts']['explicit_polarity_opposition_proposal']
    summary = {'status': 'complete', 'case_count': 80, 'image_slots': 240, 'candidate_slots': 960,
        'unique_image_hashes': len({r['cxr_sha256'] for r in plan['candidates']}),
        'unique_report_hashes': len(plan['graphs']), 'same_image_pair_slots': len(comparisons),
        'graph_extraction_status_counts': dict(Counter(r['status'] for r in graph_rows)),
        'pair_status_counts': dict(Counter(r['status'] for r in comparisons)),
        'unique_image_native_graph_pair_bindings': len({
            (candidate_by_id[r['left_triple_candidate_id']]['cxr_sha256'],
             tuple(sorted((r['left_native_graph_sha256'], r['right_native_graph_sha256']))))
            for r in completed}),
        'pairs_with_opposition_proposals': len(pairs_with_opposition),
        'image_slots_with_opposition_proposals': len({r['cxr_candidate_id'] for r in pairs_with_opposition}),
        'unique_image_hashes_with_opposition_proposals': len({candidate_by_id[
            r['left_triple_candidate_id']]['cxr_sha256'] for r in pairs_with_opposition}),
        'cases_with_opposition_proposals': len({r['case_id'] for r in pairs_with_opposition}),
        'candidate_slots_with_opposition_proposals': sum((r['rg_literal_opposition_proposal_occurrences'] or 0) > 0
                                                       for r in additions.values()),
        'pair_relation_atom_occurrences': {name: sum(r['counts'][name] for r in completed) for name in RELATIONS},
        'different_anatomy_context_concept_occurrences': sum(r['different_anatomy_context_concepts'] for r in completed),
        'different_modifier_token_set_atom_occurrences': sum(r['modifier_token_set_difference_atoms'] for r in completed),
        'graphs_without_literal_observation_atoms': sum(r is not None and not r['atoms'] for r in extracted.values()),
        'report_model_pair_diagnostics': dict(sorted(pair_model_totals.items())),
        'source_columns_preserved': len(old_rows[0]), 'new_columns': len(table[0])-len(old_rows[0]),
        'runtime_seconds': time.monotonic()-started, 'new_model_calls': 0,
        'source_patient_inputs_read': False, 'new_slurm_submissions': 0, 'clinical_qualified': False,
        'selection_changed': False, 'regeneration_authorized': False}
    write_json('summary.json', summary)
    require(all(sha256(ROOT / pin['path']) == pin['sha256'] for pin in pins), 'source_changed_during_build')
    write_json('manifest.json', {'schema_version': 'radgraph-literal-evidence-receipt-v1',
        'source_manifest_sha256': SOURCE_HASH, 'pins': pins, 'clinical_qualified': False,
        'selection_changed': False, 'artifacts': [{'path': p.name, 'sha256': sha256(p)}
                                               for p in sorted(OUT.iterdir()) if p.is_file()]})
    for path in (OUT, *OUT.iterdir()):
        require(path.stat().st_gid in (96293, 65534) and path.stat().st_mode & 0o7777 ==
                (0o2770 if path.is_dir() else 0o660), 'protected_project_modes_required')
    print(json.dumps({'status': 'synthetic_literal_evidence_complete',
                      'manifest_sha256': sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
