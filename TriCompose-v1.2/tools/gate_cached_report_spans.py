#!/usr/bin/env python3
"""Reuse an unchanged scope gate on cached Qwen/CheXbert proposals, CPU only.

Never correct/promote labels, call a model, infer image truth or enable repair.
Authored risk/coverage and synthetic evidence availability remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT/'benchmarks', ROOT/'src', ROOT.parent/'src',
        ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(directory))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.report_assertions import gate_assertion, GATE_VERSION, FINDINGS, STATES
from repair_cached_report_evidence import scope_check
from gate_report_assertion_predictions import selective_counts
from check_legacy_report_scope import read_text_inputs
import report_assertion_challenge as challenge
import verify_legacy_report_semantics as frozen

spec = importlib.util.spec_from_file_location('scope_gate_csv_helpers', ROOT/'tools/diagnose_report_verifier_progress.py')
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)
BASE = PROTECTED_ROOT/'tricompose_v1_2'
SCHEMA = 'tricompose-cached-report-span-scope-gate-v1'
PARENTS = {
    'qwen': (BASE/'verification_runs/authored_span_v2_12646689',
        '2d104e65a350bdca9970d8e4b080e0b5a9c39b93a3572a4f3a30897e94fb82eb', 'predictions.json'),
    'chexbert': (BASE/'verification_runs/authored_assertions_chexbert_12580901',
        '2fde45f8cef8993e7c6f932448c761557a61c4af8c258109e951ba0d36b348c1', 'predictions.json'),
    'bank': (BASE/'benchmarks/authored_assertions_20261002_001',
        '5f76ca3d7f710f4fdf722f87f27afc6548c839f1c4d77e56a6502ec4e3583757', 'resolver.jsonl'),
    'synthetic': (BASE/'verification_runs/span_v2_scope2_12645404_analysis',
        '4defd00b806b3187bc9657521bfae22b6b63e55e7d0b86812e4846344e67121b', 'assertion_comparison.jsonl'),
    'synthetic_plan': (BASE/'report_span_v2_plans/span_v2_scope2_12645021_001',
        '576483c1b4f544b7fd99f5b6e7aca94d8c22c83694294fd41cd3ab9d1aa74afc', 'plan.json'),
}


def require_cpu_slurm():
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('existing_cpu_slurm_required')


def load_fixed_inputs():
    payloads, sources, manifests = {}, {}, {}
    for name, (root, expected, artifact_name) in PARENTS.items():
        mp = require_inside(root/'manifest.json', PROTECTED_ROOT, must_exist=True)
        if sha256_file(mp) != expected:
            raise ValueError('fixed_parent_manifest_required')
        manifest = read_json(mp)
        artifact = require_inside(root/artifact_name, root, must_exist=True)
        if sha256_file(artifact) != manifest['artifacts'][artifact_name]['sha256']:
            raise ValueError('fixed_consumed_artifact_required')
        if artifact.stat().st_size > 4*1024*1024:
            raise ValueError('bounded_artifact_required')
        payloads[name] = ([json.loads(line) for line in artifact.read_text().splitlines()]
            if artifact_name.endswith('.jsonl') else read_json(artifact))
        manifests[name] = manifest
        sources[name+'_manifest'], sources[name+'_artifact'] = mp, artifact
    sources.update(frozen.sources_for_program())
    # This gate/regex/classification code predates the observed Qwen results.
    old_hashes = {str(Path(path).resolve()): manifests['qwen']['source_sha256'][key]
        for key, path in manifests['qwen']['source_paths'].items()}
    for path in sources.values():
        expected = old_hashes.get(str(path.resolve()))
        if expected is not None and sha256_file(path) != expected:
            raise ValueError('preexisting_frozen_program_changed')
    if sha256_file(ROOT/'benchmarks/repair_cached_report_evidence.py') != challenge.FROZEN_GUARD_SHA:
        raise ValueError('preexisting_scope_checker_changed')
    sources.update(worker=Path(__file__), csv_helpers=ROOT/'tools/diagnose_report_verifier_progress.py',
        tests=ROOT/'tests/test_cached_report_scope_gate.py', protocol=ROOT.parent/'docs/cached_report_scope_gate_protocol.md')
    return payloads, sources


def source_gate(text, finding, proposed_state, *, contract_status='complete'):
    base = {'scopegate_version': GATE_VERSION, 'scopegate_retained_state': None,
        'scopegate_scope_verified': False, 'scopegate_evidence': [],
        'scopegate_independent_clinical_validation': False, 'scopegate_regeneration_authorized': False}
    if finding not in FINDINGS:
        return {**base, 'scopegate_decision': 'outside_verifier_scope', 'scopegate_reason': 'no_frozen_head',
            'scopegate_scope_check': None}
    if contract_status != 'complete' or text is None:
        return {**base, 'scopegate_decision': 'verifier_unavailable', 'scopegate_reason': 'input_or_extractor_unavailable',
            'scopegate_scope_check': None}
    result = gate_assertion(text, finding, proposed_state, scope_check)
    committed = result['decision'] == 'scope_commit'
    if committed and (result['state'] != proposed_state or proposed_state == 'unknown'):
        raise ValueError('gate_may_not_flip_or_promote')
    return {**base, 'scopegate_retained_state': result['state'] if committed else None,
        'scopegate_decision': result['decision'], 'scopegate_reason': result['reason'],
        'scopegate_scope_verified': result['scope_verified'], 'scopegate_scope_check': result.get('scope_check'),
        'scopegate_evidence': [{key: span[key] for key in
            ('char_start', 'char_end', 'quote_sha256', 'offset_unit', 'evidence_id')} for span in result['evidence']]}


def authored_rows(resolver, texts, predictions):
    output = []
    for extractor, records in predictions.items():
        index = {r['item_id']: r for r in records}
        if len(index) != len(records) or set(index) != {r['item_id'] for r in resolver}:
            raise ValueError('exact_authored_prediction_inventory_required')
        for item in resolver:
            pred = index[item['item_id']]
            if pred['report_sha256'] != item['report_sha256'] or pred['status'] not in ('complete', 'failed_unavailable'):
                raise ValueError('same_input_prediction_required')
            if pred['status'] == 'complete' and (not isinstance(pred['finding_states'], dict)
                    or set(pred['finding_states']) != set(FINDINGS)
                    or not set(pred['finding_states'].values()) <= STATES):
                raise ValueError('valid_complete_four_states_required')
            for finding in FINDINGS:
                state = pred['finding_states'][finding] if pred['status'] == 'complete' else None
                output.append({'extractor': extractor, 'item_id': item['item_id'], 'finding': finding,
                    'report_sha256': item['report_sha256'], 'raw_state': state,
                    'raw_contract_status': pred['status'], **source_gate(texts[item['report_sha256']], finding,
                        state, contract_status=pred['status'])})
    return output


def synthetic_rows(original, texts, report_inputs):
    inputs = {r['report_candidate_id']: r for r in report_inputs}
    if len(inputs) != len(report_inputs):
        raise ValueError('unique_synthetic_report_slots_required')
    seen, output = set(), []
    for row in original:
        if set(row) != progress.FACT_FIELDS:
            raise ValueError('quote_free_original_fact_schema_required')
        key = (row['triple_candidate_id'], row['finding'])
        if key in seen:
            raise ValueError('unique_candidate_finding_required')
        seen.add(key)
        report_input = inputs.get(row['report_candidate_id'])
        if report_input is None or report_input['report_sha256'] != row['report_sha256']:
            raise ValueError('same_fixed_candidate_report_required')
        if any(row[f] is not False for f in ('independent_clinical_validation', 'selection_changed', 'regeneration_authorized')):
            raise ValueError('unchanged_unqualified_candidate_required')
        output.append({**row, **source_gate(texts.get(row['report_sha256']), row['finding'],
            row['qwen_assertion_state'], contract_status=row['qwen_contract_status'])})
    return output


def analyze_authored(rows, references):
    index = {(r['extractor'], r['item_id'], r['finding']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('unique_extractor_finding_required')
    targets = []
    extractors = sorted({r['extractor'] for r in rows})
    expected = {(name, ref['item_id'], f) for name in extractors for ref in references for f in FINDINGS}
    if set(index) != expected:
        raise ValueError('complete_authored_reference_join_required')
    for name in extractors:
        for ref in references:
            for finding in ref['evaluation_findings']:
                row = index[name, ref['item_id'], finding]
                if row['report_sha256'] != ref['report_sha256']:
                    raise ValueError('authored_reference_hash_required')
                targets.append({'extractor': name, 'item_id': ref['item_id'], 'finding': finding,
                    'family': ref['family'], 'expected': ref['expected_states'][finding],
                    'raw_state': row['raw_state'], 'gated_state': row['scopegate_retained_state'],
                    'decision': row['scopegate_decision'], 'reason': row['scopegate_reason']})
    summary = {name: {'designated_targets': selective_counts([r for r in targets if r['extractor'] == name]),
        'per_finding': {finding: selective_counts([r for r in targets if r['extractor'] == name and r['finding'] == finding])
            for finding in FINDINGS},
        'per_family': {family: selective_counts([r for r in targets if r['extractor'] == name and r['family'] == family])
            for family in sorted({r['family'] for r in targets})}} for name in extractors}
    return summary, targets


def markdown(summary):
    lines = ['# Frozen scope gate / 冻结范围规则与弃权', '',
        '复用旧规则，不纠正原标签；虚构文本的条件匹配与 synthetic 候选可用性分开统计。', '',
        '| Authored extractor | Raw matches / 80 | Commits / 80 | Correct commits | Raw errors not committed | Matching proposals not committed |',
        '|---|---:|---:|---:|---:|---:|']
    for name, value in summary['authored'].items():
        d = value['designated_targets']
        lines.append(f"| {name} | {d['raw_exact_matches']}/80 | {d['scope_commits']}/80 | {d['correct_commits']}/{d['scope_commits']} | {d['raw_errors_not_committed']} | {d['raw_correct_not_committed']} |")
    lines += ['', '## Fixed synthetic slots / 固定合成候选', '',
        '| Gate decision | Candidate/finding rows |', '|---|---:|']
    for decision, count in summary['synthetic_decision_counts'].items():
        lines.append(f'| {decision} | {count} |')
    lines += ['',
        '所有 24 个候选、336 个 finding 行及原始标签/分数保留。只有四头受支持，不能把另外十头算成阴性。',
        'Noncommitted state is null: abstention is not a correct unknown prediction or evidence of normal anatomy.',
        'Conditional authored match is not overall accuracy, independent clinical validation or repaired generation.',
        '同报告模型与规则不是独立临床票数；scope commit 不表示报告正确描述 CXR。',
        'No new model/GPU/API/Slurm call, training, rule/prompt tuning, clinical acceptance, ranking or regeneration.',
        'The fixed cases, original scores/winners and all frozen source artifacts remain unchanged.', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    require_cpu_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        inputs, sources = load_fixed_inputs()
        resolver, authored_texts, source = challenge.load_inputs(PARENTS['bank'][0])
        if (inputs['qwen']['model_received_references_baseline_or_scores'] is not False
                or inputs['chexbert']['source'] != source
                or inputs['chexbert']['model_received_reference_states'] is not False
                or inputs['chexbert']['scorer_read_reference_key'] is not False):
            raise ValueError('blinded_original_extractor_required')
        sources.update({'authored_text_'+r['item_id']: PARENTS['bank'][0]/r['report_path'] for r in resolver})
        synth_texts, unavailable, text_sources, verified = read_text_inputs(inputs['synthetic_plan']['report_inputs'])
        sources.update(text_sources)
        before = {key: sha256_file(path) for key, path in sources.items()}
        authored = authored_rows(resolver, authored_texts, {name: inputs[name]['records'] for name in ('qwen', 'chexbert')})
        synthetic = synthetic_rows(inputs['synthetic'], synth_texts, inputs['synthetic_plan']['report_inputs'])
        if (len(authored), len(synthetic)) != (448, 336):
            raise ValueError('fixed_authored_and_synthetic_inventory_required')
        files = [write_private_text(temporary/'authored_gate_table.jsonl', progress.jsonl_text(authored)),
            write_private_text(temporary/'candidate_gate_fact_table.jsonl', progress.jsonl_text(synthetic))]
        for path in files:
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
        frozen_tables = {path.name: sha256_file(path) for path in files}
        # Reference key is parsed only after blind gate outputs are frozen.
        refs, reference_path = challenge.load_references(PARENTS['bank'][0], resolver)
        sources['authored_references'] = reference_path
        before['authored_references'] = sha256_file(reference_path)
        authored_summary, details = analyze_authored(authored, refs)
        summary = {'schema_version': SCHEMA, 'gate_version': GATE_VERSION, 'authored': authored_summary,
            'authored_rows': len(authored), 'synthetic_rows': len(synthetic),
            'synthetic_candidate_slots': len({r['triple_candidate_id'] for r in synthetic}),
            'synthetic_distinct_report_hashes': len({r['report_sha256'] for r in synthetic}),
            'synthetic_decision_counts': dict(sorted(Counter(r['scopegate_decision'] for r in synthetic).items())),
            'synthetic_input_unavailable': len(unavailable), 'verified_synthetic_report_slots': verified,
            'new_model_calls': 0, 'external_api_used': False, 'real_patient_inputs_opened': False,
            'thresholds_weights_rules_or_prompt_changed': False, 'original_cells_preserved': True,
            'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
            'development_diagnostic_not_heldout': True, 'clinical_faults_confirmed': None,
            'reference_key_read_after_gate_fsync': True}
        write_private_json(temporary/'summary.json', summary)
        write_private_json(temporary/'authored_details.json', {'records': details})
        skinny = ('triple_candidate_id', 'report_candidate_id', 'finding', 'qwen_assertion_state',
            'qwen_contract_status', 'scopegate_decision', 'scopegate_retained_state', 'scopegate_reason',
            'scopegate_scope_verified', 'scopegate_independent_clinical_validation', 'scopegate_regeneration_authorized')
        write_private_text(temporary/'candidate_gate_table.csv', progress.csv_text(synthetic, skinny))
        write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        if any(sha256_file(path) != before[key] for key, path in sources.items()) or any(
                sha256_file(temporary/name) != value for name, value in frozen_tables.items()):
            raise ValueError('immutable_sources_or_blind_tables_changed')
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {key: str(path.resolve()) for key, path in sources.items()}, 'source_sha256': before,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'real_patient_inputs_opened': False})
        for path in (temporary, *temporary.iterdir()):
            stat = path.stat()
            if stat.st_gid not in (96293, 65534) or stat.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE/'verification_gates')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    os.umask(0o007)
    target, summary = execute(args.output_root, args.run_id)
    print(json.dumps({'status': 'completed_cached_scope_gate', 'authored_rows': summary['authored_rows'],
        'synthetic_rows': summary['synthetic_rows'], 'new_model_calls': 0,
        'manifest_sha256': sha256_file(target/'manifest.json')}))


if __name__ == '__main__':
    main()
