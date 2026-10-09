#!/usr/bin/env python3
"""Attach cached report scope availability to the unchanged full-bank tables.

Metadata only: no report/image/EHR bodies, reference keys, models or selection.
Never propagate a checked report hash to an unchecked candidate slot.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, CHEXPERT_FINDINGS, require_inside,
    sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)
from tricompose_v12.reliability_preview import validate_sidecar, REQUEST_DEPENDENCIES

SCHEMA = 'tricompose-full-bank-report-gate-availability-v1'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
HEADS = frozenset(('cardiomegaly', 'consolidation', 'pleural_effusion', 'pneumothorax'))
STATES = frozenset(('positive', 'negative', 'uncertain', 'unknown'))
DECISIONS = frozenset(('scope_commit', 'abstain', 'no_model_assertion',
    'verifier_unavailable', 'outside_verifier_scope'))
GATE_FIELDS = frozenset(('chexbert_qwen_comparison', 'clinical_selection_score',
    'confirmed_faulty_modality', 'finding', 'independent_clinical_validation',
    'literal_scope_decision', 'literal_scope_state', 'qwen_assertion_state',
    'qwen_contract_failure_reason', 'qwen_contract_status', 'raw_chexbert_state',
    'regeneration_authorized', 'report_candidate_id', 'report_sha256',
    'scopegate_decision', 'scopegate_evidence', 'scopegate_independent_clinical_validation',
    'scopegate_reason', 'scopegate_regeneration_authorized', 'scopegate_retained_state',
    'scopegate_scope_check', 'scopegate_scope_verified', 'scopegate_version',
    'selection_changed', 'triple_candidate_id'))
PARENTS = {
    'pool': (BASE/'candidate_reliability_overlays/reliability_pool960_12632006_001',
        'd057efecd06a4db11c6fec27e79f2ecf085b1c7afe0fa7174fce554c8e85b673',
        ('candidate_score_table.csv', 'fact_reliability.jsonl', 'report_dependency_groups.jsonl')),
    'gate': (BASE/'verification_gates/scope_gate_cached_12645021_001',
        '86a3e37a212d06fb665e7797397dd7736a449f92e30bb19c322923cc3f76e66a',
        ('candidate_gate_fact_table.jsonl',)),
    'progress': (BASE/'report_verifier_progress/progress_scope2_12645021_001',
        'b8999d053b6f6adb82127915e8a9194cc4fa3d65ea503ca239cce21bbd861837',
        ('evidence_request_progress.jsonl',)),
}


def require_cpu_slurm():
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('existing_cpu_slurm_required')


def load_fixed_inputs():
    inputs, sources = {}, {'worker': Path(__file__),
        'tests': ROOT/'tests/test_report_gate_availability.py',
        'protocol': ROOT.parent/'docs/report_gate_availability_protocol.md',
        'atomic_contracts': Path(sys.modules['contracts'].__file__)}
    for name in ('reliability_preview', 'scorer_reliability', 'legacy_replay_adapter',
            'invariant_verification', 'decision_preview'):
        sources[name] = ROOT/'src/tricompose_v12'/f'{name}.py'
    for label, (root, expected, names) in PARENTS.items():
        mp = require_inside(root/'manifest.json', PROTECTED_ROOT, must_exist=True)
        if sha256_file(mp) != expected:
            raise ValueError('fixed_parent_manifest_required')
        manifest = json.loads(mp.read_text(encoding='utf-8'))
        if (manifest.get('new_model_calls') != 0
                or manifest.get('primary_metric_eligible') is not False
                or manifest.get('regeneration_authorized') is not False):
            raise ValueError('unqualified_metadata_parent_required')
        sources[label+'_manifest'] = mp
        for name in names:
            path = require_inside(root/name, root, must_exist=True)
            if (not path.is_file() or path.stat().st_size > 32*1024*1024
                    or sha256_file(path) != manifest['artifacts'][name]['sha256']):
                raise ValueError('bounded_fixed_metadata_artifact_required')
            if name.endswith('.csv'):
                with path.open(encoding='utf-8', newline='') as handle:
                    rows = list(csv.DictReader(handle))
                if any(None in r or None in r.values() for r in rows):
                    raise ValueError('complete_csv_cells_required')
            else:
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
            inputs[label, name] = rows
            sources[label+'_'+name] = path
        if sha256_file(mp) != expected:
            raise ValueError('parent_changed_during_read')
    return inputs, sources


def index_gate(rows, facts, gates):
    """Strict slot/finding/report lineage join; not a same-text expansion."""
    candidates = {r['triple_candidate_id']: r for r in rows}
    original = {(r['triple_candidate_id'], r['finding']): r for r in facts}
    if len(candidates) != len(rows) or len(original) != len(facts):
        raise ValueError('unique_pool_candidate_finding_required')
    lookup, grouped = {}, defaultdict(list)
    for gate in gates:
        if set(gate) != GATE_FIELDS:
            raise ValueError('quote_free_gate_fields_required')
        cid, finding = gate['triple_candidate_id'], gate['finding']
        key = cid, finding
        if key in lookup or key not in original or cid not in candidates:
            raise ValueError('unique_existing_gate_slot_required')
        row, fact = candidates[cid], original[key]
        if (gate['report_sha256'] != row['report_sha256']
                or gate['report_candidate_id'] != row['report_candidate_id']
                or gate['raw_chexbert_state'] != fact['states']['chexbert']
                or gate['report_sha256'] != fact['artifact_hashes']['report_sha256']):
            raise ValueError('exact_gate_report_and_raw_state_required')
        if (any(gate[f] is not False for f in ('independent_clinical_validation',
                'regeneration_authorized', 'selection_changed',
                'scopegate_independent_clinical_validation', 'scopegate_regeneration_authorized'))
                or gate['clinical_selection_score'] is not None
                or gate['confirmed_faulty_modality'] is not None):
            raise ValueError('unqualified_unchanged_gate_required')
        decision, state = gate['scopegate_decision'], gate['scopegate_retained_state']
        if decision not in DECISIONS:
            raise ValueError('known_gate_decision_required')
        if finding not in HEADS:
            if decision != 'outside_verifier_scope' or gate['qwen_contract_status'] != 'outside_scope_inventory':
                raise ValueError('unsupported_finding_must_remain_outside_scope')
        elif decision == 'outside_verifier_scope':
            raise ValueError('supported_head_cannot_be_outside_scope')
        elif decision == 'verifier_unavailable':
            if gate['qwen_contract_status'] == 'complete':
                raise ValueError('unavailable_contract_required')
        elif gate['qwen_contract_status'] != 'complete' or gate['qwen_assertion_state'] not in STATES:
            raise ValueError('complete_four_state_contract_required')
        if decision == 'scope_commit':
            if (state not in STATES - {'unknown'} or state != gate['qwen_assertion_state']
                    or gate['scopegate_scope_verified'] is not True):
                raise ValueError('same_nonunknown_retained_proposal_required')
        elif state is not None or gate['scopegate_scope_verified'] is not False:
            raise ValueError('noncommit_is_null_not_new_prediction')
        if decision == 'no_model_assertion' and gate['qwen_assertion_state'] != 'unknown':
            raise ValueError('no_assertion_requires_unknown')
        for evidence in gate['scopegate_evidence']:
            if set(evidence) != {'char_start', 'char_end', 'quote_sha256', 'offset_unit', 'evidence_id'}:
                raise ValueError('quote_free_evidence_fields_required')
        lookup[key] = gate
        grouped[cid].append(gate)
    for candidate in grouped.values():
        if len(candidate) != 14 or {r['finding'] for r in candidate} != set(CHEXPERT_FINDINGS):
            raise ValueError('complete_fixed_gate_slot_inventory_required')
    return lookup, grouped


def fact_extension(gate):
    return {'reportgate_status': 'not_checked' if gate is None else gate['scopegate_decision'],
        'reportgate_raw_qwen_state': None if gate is None else gate['qwen_assertion_state'],
        'reportgate_retained_state': None if gate is None else gate['scopegate_retained_state'],
        'reportgate_reason': 'candidate_slot_not_in_fixed_check' if gate is None else gate['scopegate_reason'],
        'reportgate_source_version': None if gate is None else gate['scopegate_version'],
        'reportgate_independent_clinical_validation': False,
        'reportgate_primary_metric_eligible': False, 'reportgate_regeneration_authorized': False}


def candidate_availability(rows, facts, lookup, grouped):
    counts = Counter(f['triple_candidate_id'] for f in facts if f['states']['ehr'] in ('positive', 'negative'))
    output = []
    for row in rows:
        cid = row['triple_candidate_id']
        checked = cid in grouped
        decisions = Counter(g['scopegate_decision'] for g in grouped.get(cid, ()))
        extension = {'reportgate_status': 'checked_clinically_unqualified' if checked else 'not_checked',
            'reportgate_supported_head_checks': 4 if checked else 0,
            'reportgate_inventory_rows_unchecked': 0 if checked else 14,
            'reportgate_direct_ehr_reference_facts': counts[cid],
            'reportgate_independent_clinical_validation': False,
            'reportgate_primary_metric_eligible': False, 'reportgate_regeneration_authorized': False}
        for decision in sorted(DECISIONS):
            extension['reportgate_'+decision+'_count'] = decisions[decision] if checked else None
        if any(key.startswith('reportgate_') for key in row):
            raise ValueError('already_attached_candidate_refused')
        output.append({**row, **extension})
    annotated_facts = []
    for fact in facts:
        if any(key.startswith('reportgate_') for key in fact):
            raise ValueError('already_attached_fact_refused')
        annotated_facts.append({**fact, **fact_extension(lookup.get((fact['triple_candidate_id'], fact['finding'])))})
    return output, annotated_facts


def request_availability(requests, lookup, candidates):
    output, seen = [], set()
    for row in requests:
        if (row['request_id'] in seen or row['request_kind'] not in REQUEST_DEPENDENCIES
                or row['execution_status'] != 'not_executed'
                or any(row[f] is not False for f in ('clinical_truth_established',
                    'model_execution_allowed', 'reportcheck_clinically_resolved'))
                or row['reportcheck_new_model_calls'] != 0
                or any(k.startswith('reportgate_') for k in row)):
            raise ValueError('unique_unresolved_original_request_required')
        seen.add(row['request_id'])
        consumers = row['consumer_candidate_ids']
        if not consumers or len(set(consumers)) != len(consumers):
            raise ValueError('unique_nonempty_request_consumers_required')
        matched = []
        for cid in consumers:
            candidate = candidates.get(cid)
            if candidate is None or candidate['case_id'] != row['case_id']:
                raise ValueError('existing_same_case_consumer_required')
            if set(row['dependency_hashes']) != set(REQUEST_DEPENDENCIES[row['request_kind']]) or any(
                    candidate[name] != value for name, value in row['dependency_hashes'].items()):
                raise ValueError('exact_request_dependency_hashes_required')
            if row['request_kind'] == 'verify_report_assertion':
                gate = lookup.get((cid, row['finding']))
                if gate is not None:
                    matched.append(gate)
        decisions = Counter(g['scopegate_decision'] for g in matched)
        status = 'not_checked'
        if matched:
            status = (next(iter(decisions)) if len(decisions) == 1 else 'mixed_consumer_availability')
            if len(matched) < len(consumers):
                status = 'partial_consumer_check'
        output.append({**row, 'reportgate_status': status,
            'reportgate_matched_consumer_ids': sorted(g['triple_candidate_id'] for g in matched),
            'reportgate_consumer_decision_counts': dict(sorted(decisions.items())),
            'reportgate_unchecked_consumer_count': len(consumers)-len(matched),
            'reportgate_clinically_resolved': False, 'reportgate_new_model_calls': 0})
    return output


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list))
            else v for k, v in row.items()})
    return stream.getvalue()


def jsonl_text(rows):
    return ''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows)


def markdown(summary):
    lines = ['# Report scope availability / 报告检查可用性', '',
        '保留原始分数和状态，仅附加已检查/弃权/未检查信息；不是重新评分或择优。', '',
        '| Pool / 候选池 | Count |', '|---|---:|',
        f"| Fixed EHR cases | {summary['fixed_ehr_cases']} |",
        f"| Candidate triples | {summary['candidate_rows']} |",
        f"| Original finding rows | {summary['fact_rows']} |",
        f"| Checked candidate slots | {summary['checked_candidate_slots']} |",
        f"| Unchecked candidate slots | {summary['unchecked_candidate_slots']} |",
        f"| Slots with retained assertions | {summary['candidate_slots_with_retained_assertions']} |", '',
        '| Finding availability | Rows |', '|---|---:|']
    lines += [f'| {status} | {count} |' for status, count in summary['fact_status_counts'].items()]
    lines += ['', '| Existing request sidecar status | Logical requests |', '|---|---:|']
    lines += [f'| {status} | {count} |' for status, count in summary['request_status_counts'].items()]
    lines += ['',
        'Clinical requests resolved: 0. Logical requests are not model calls.',
        '相同文本 hash 不能把检查结果传播到未检查的 candidate ID。',
        'Unknown 不变阴性；没有检查与检查后未提及分开。',
        'Report-only scope checks cannot fulfill image/EHR or image/report relation requests.',
        'Retained assertions are not independent clinical truth or calibrated probabilities.',
        'Original scores, ranks, winners, EHRs, request execution history and all source cells remain unchanged.',
        'No new model/GPU/API/Slurm submission, training, reference-key read, raw patient data or report/image body access.', '',
        '`candidate_score_table.csv`: every original score column plus reportgate availability.',
        '`fact_verification_availability.jsonl`: every original fact plus availability.',
        '`evidence_request_availability.jsonl`: every original request/history plus availability.',
        '`summary.json` and `manifest.json`: denominators, source/result hashes and scope boundaries.', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    require_cpu_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        inputs, sources = load_fixed_inputs()
        before = {key: sha256_file(path) for key, path in sources.items()}
        rows, facts, groups = (inputs['pool', name] for name in
            ('candidate_score_table.csv', 'fact_reliability.jsonl', 'report_dependency_groups.jsonl'))
        gates = inputs['gate', 'candidate_gate_fact_table.jsonl']
        requests = inputs['progress', 'evidence_request_progress.jsonl']
        if (len(rows), len(facts), len(groups), len(gates), len(requests)) != (960, 13440, 3360, 336, 2072):
            raise ValueError('fixed_full_bank_inventory_required')
        validate_sidecar(rows, facts, groups)
        lookup, grouped = index_gate(rows, facts, gates)
        annotated, availability = candidate_availability(rows, facts, lookup, grouped)
        attached = request_availability(requests, lookup, {r['triple_candidate_id']: r for r in rows})
        for original, output in ((rows, annotated), (facts, availability), (requests, attached)):
            if len(original) != len(output) or any(any(new[k] != v for k, v in old.items())
                    for old, new in zip(original, output)):
                raise ValueError('all_original_cells_and_order_must_be_preserved')
        if (len({r['case_id'] for r in rows}), len(grouped), len({r['case_id'] for r in annotated
                if r['reportgate_status'] != 'not_checked'})) != (80, 24, 2):
            raise ValueError('unchanged_fixed_case_scope_required')
        summary = {'schema_version': SCHEMA, 'candidate_rows': len(rows), 'fact_rows': len(facts),
            'fixed_ehr_cases': 80, 'image_slots': len({r['cxr_candidate_id'] for r in rows}),
            'checked_candidate_slots': len(grouped), 'unchecked_candidate_slots': len(rows)-len(grouped),
            'candidate_slots_with_retained_assertions': sum(bool(r['reportgate_scope_commit_count']) for r in annotated),
            'fact_status_counts': dict(sorted(Counter(f['reportgate_status'] for f in availability).items())),
            'logical_requests': len(requests),
            'request_status_counts': dict(sorted(Counter(r['reportgate_status'] for r in attached).items())),
            'original_cells_and_order_preserved': True, 'clinically_resolved_requests': 0,
            'new_model_calls': 0, 'new_slurm_submissions': 0, 'existing_cpu_job_id': os.environ['SLURM_JOB_ID'],
            'primary_metric_eligible': False, 'selection_changed': False, 'fixed_ehr_changed': False,
            'regeneration_authorized': False, 'benchmark_matches_transferred_to_candidate_confidence': False,
            'rules_prompts_weights_or_thresholds_changed': False, 'raw_patient_inputs_opened': False,
            'report_image_ehr_bodies_opened': False, 'reference_keys_opened': False,
            'development_diagnostic_not_heldout': True}
        write_private_text(temporary/'candidate_score_table.csv', csv_text(annotated))
        write_private_text(temporary/'fact_verification_availability.jsonl', jsonl_text(availability))
        write_private_text(temporary/'evidence_request_availability.jsonl', jsonl_text(attached))
        write_private_json(temporary/'summary.json', summary)
        write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        if any(sha256_file(path) != before[key] for key, path in sources.items()):
            raise ValueError('consumed_metadata_or_program_changed')
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {key: str(path.resolve()) for key, path in sources.items()}, 'source_sha256': before,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'report_image_ehr_bodies_opened': False})
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
    parser.add_argument('--output-root', type=Path, default=BASE/'report_verification_availability')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, summary = execute(args.output_root, args.run_id)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'metadata_availability_attached',
        'candidate_rows': summary['candidate_rows'], 'checked_candidate_slots': summary['checked_candidate_slots'],
        'new_model_calls': 0, 'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
