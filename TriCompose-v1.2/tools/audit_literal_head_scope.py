"""Planning-only exact-head coverage; no model, tokens, pixels or source EHR."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from tricompose_v12.radgraph_literal_evidence import digest

ROOT = Path('/project2/ruishanl_1185/inference_3mod')
BASE = ROOT / 'artifacts/protected/tricompose_v1_2'
SOURCE = BASE / 'radgraph_literal_evidence_runs/literal_pool960_12714150_001'
AUDIT = BASE / 'radgraph_literal_evidence_audits/literal_pool960_12714150_001/audit.json'
OUT = BASE / 'radgraph_literal_head_scope_runs/scope37_12714150_001'
HEAD_SOURCE = ROOT / 'TriCompose-v1.2/benchmarks/verify_candidate_findings_qwen.py'


def match_heads(heads, graphs, pairs):
    if not isinstance(heads, (list, tuple)) or not 1 <= len(heads) <= 32 or \
            len(set(heads)) != len(heads) or any(not isinstance(h, str) or
            not re.fullmatch('[a-z][a-z_]{0,47}', h) for h in heads):
        raise ValueError('explicit_existing_head_schema_required')
    lookup = {digest(['literal_observation', head.replace('_', ' ')]): head for head in heads}
    records = []
    for pair in pairs:
        if pair['status'] != 'complete':
            continue
        for detail in pair['details']:
            if detail['relation'] != 'explicit_polarity_opposition_proposal':
                continue
            atom = graphs[pair['left_graph_id']]['atoms'][detail['atom_id']]
            counterpart = graphs[pair['right_graph_id']]['atoms'][detail['atom_id']]
            if atom['concept_sha256'] != counterpart['concept_sha256']:
                raise ValueError('same_literal_concept_required')
            head = lookup.get(atom['concept_sha256'])
            records.append({'pair_id': pair['pair_id'], 'atom_id': detail['atom_id'],
                'case_id': pair['case_id'], 'cxr_candidate_id': pair['cxr_candidate_id'],
                'concept_sha256': atom['concept_sha256'], 'exact_existing_head': head,
                'status': 'exact_literal_head_match' if head is not None else 'no_exact_literal_head_match',
                'current_patient_scope_verified': False, 'clinical_resolved': False,
                'model_request_submitted': False})
    return sorted(records, key=lambda r: (r['pair_id'], r['atom_id']))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def execute():
    if '/job_12714150/' not in Path('/proc/self/cgroup').read_text() or OUT.exists():
        raise ValueError('actual_cpu_allocation_and_fresh_run_required')
    if sha256(SOURCE / 'manifest.json') != 'da1bfb5c82f1ce51d000843ee11471c6edddaafc20a5ab075227a4c675eb57c9' or \
            sha256(AUDIT) != '6c2d1b343b94dd710cdd6cf95f4b4e9f53bbd13d1528377b566125b47a24622e':
        raise ValueError('sealed_source_and_audit_required')
    receipt = json.loads((SOURCE / 'manifest.json').read_text())
    paths = [Path(__file__).resolve(), HEAD_SOURCE, AUDIT, SOURCE / 'manifest.json',
             ROOT / 'TriCompose-v1.2/src/tricompose_v12/radgraph_literal_evidence.py',
             ROOT / 'TriCompose-v1.2/tests/test_literal_head_scope.py']
    artifact_hashes = {r['path']: r['sha256'] for r in receipt['artifacts']}
    for name in ('graph_literal_evidence.json', 'same_image_literal_evidence.json'):
        path = SOURCE / name
        if sha256(path) != artifact_hashes[name]:
            raise ValueError('sealed_literal_artifact_required')
        paths.append(path)
    assignments = [n for n in ast.parse(HEAD_SOURCE.read_text()).body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'FINDINGS' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('single_literal_existing_head_definition_required')
    heads = ast.literal_eval(assignments[0].value)
    if len(heads) != 8:
        raise ValueError('unchanged_eight_head_scope_required')
    pins = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in paths]
    graphs = {r['graph_id']: r['evidence'] for r in
              json.loads((SOURCE / 'graph_literal_evidence.json').read_text())}
    pairs = json.loads((SOURCE / 'same_image_literal_evidence.json').read_text())
    records = match_heads(heads, graphs, pairs)
    if len(records) != 37 or records != match_heads(heads, graphs, pairs[::-1]):
        raise ValueError('all_proposals_and_deterministic_order_required')
    counts = Counter(r['status'] for r in records)
    per_head = {h: sum(r['exact_existing_head'] == h for r in records) for h in heads}
    summary = {'status': 'complete', 'post_hoc_scope_preflight': True,
        'all_opposition_proposal_occurrences': len(records), 'status_counts': dict(counts),
        'per_existing_head_occurrences': per_head,
        'matched_image_slots': len({r['cxr_candidate_id'] for r in records if r['exact_existing_head'] is not None}),
        'unmatched_means_clinically_outside_scope': False, 'synonym_aliases_added': False,
        'source_tokens_or_reports_read': False, 'new_model_calls': 0, 'new_slurm_submissions': 0,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False}
    if not all(sha256(ROOT / p['path']) == p['sha256'] for p in pins):
        raise ValueError('source_changed_during_scope_preflight')
    os.umask(0o007)
    OUT.parent.mkdir(mode=0o2770, exist_ok=True)
    OUT.parent.chmod(0o2770)
    OUT.mkdir(mode=0o2770)
    OUT.chmod(0o2770)
    def write(name, value):
        with (OUT / name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        (OUT / name).chmod(0o660)
    write('records.json', records)
    write('summary.json', summary)
    write('manifest.json', {'schema_version': 'radgraph-literal-existing-head-scope-v1',
        'pins': pins, 'source_manifest_sha256': sha256(SOURCE / 'manifest.json'),
        'clinical_qualified': False, 'selection_changed': False,
        'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(OUT.iterdir())]})
    for path in (OUT, *OUT.iterdir()):
        if path.stat().st_gid not in (96293, 65534) or path.stat().st_mode & 0o7777 != \
                (0o2770 if path.is_dir() else 0o660):
            raise ValueError('protected_project_modes_required')
    print(json.dumps({'status': 'literal_existing_head_scope_complete',
                      'manifest_sha256': sha256(OUT / 'manifest.json')}))


if __name__ == '__main__':
    try:
        execute()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
