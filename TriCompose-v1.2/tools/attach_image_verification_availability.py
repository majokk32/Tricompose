#!/usr/bin/env python3
"""Attach completed image-only evidence without changing scores or histories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import attach_report_gate_availability as reportgate
import verify_cached_image_findings as imageworker
from contracts import (PROTECTED_ROOT, CHEXPERT_FINDINGS, require_inside,
    sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)
from tricompose_v12.reliability_preview import validate_sidecar, REQUEST_DEPENDENCIES

SCHEMA = 'tricompose-full-bank-image-verification-availability-v1'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
HEADS = frozenset(imageworker.image_interface.FINDINGS)
STATES = imageworker.STATES
HASH = re.compile(r'[0-9a-f]{64}\Z')
IMAGE_FIELDS = frozenset(('imagecheck_cxr_candidate_id', 'imagecheck_cxr_sha256',
    'imagecheck_contract_status', 'imagecheck_raw_xrv_state', 'imagecheck_qwen_state',
    'imagecheck_xrv_qwen_relation', 'imagecheck_retained_report_relation',
    'imagecheck_independent_clinical_validation', 'imagecheck_primary_metric_eligible',
    'imagecheck_confirmed_faulty_modality', 'imagecheck_regeneration_authorized'))
RECORD_FIELDS = frozenset(('contract_status', 'cxr_candidate_id', 'cxr_sha256',
    'elapsed_seconds', 'failure_reason', 'independent_clinical_validation',
    'input_tokens', 'output_tokens', 'response_sha256', 'states', 'token_limit_reached'))
RELATIONS = ('explicit_agreement_unqualified', 'explicit_opposition_unqualified',
    'both_unmentioned_or_unassessable', 'uncertainty_not_comparable',
    'single_source_assertion_unqualified', 'outside_image_verifier_scope',
    'image_verifier_unavailable')
PARENTS = {
    'availability': (BASE/'report_verification_availability/reportgate_pool960_12645021_001',
        'efb507376ce8cd8b23b28a6b55d719c7725a66fb9395f79bc70b32e76121299a',
        ('candidate_score_table.csv', 'fact_verification_availability.jsonl',
         'evidence_request_availability.jsonl')),
    'image': (BASE/'verification_runs/image_only_scope2_12649136',
        '07aef0cf22a59ecf6441a07601afca3989c95dd5a5b608be5c4ee05a2bce17e1',
        ('predictions.json', 'image_report_comparison.jsonl', 'summary.json')),
    'pool': (BASE/'candidate_reliability_overlays/reliability_pool960_12632006_001',
        'd057efecd06a4db11c6fec27e79f2ecf085b1c7afe0fa7174fce554c8e85b673',
        ('report_dependency_groups.jsonl',)),
}


def load_fixed_inputs():
    sources = {'worker': Path(__file__), 'tests': ROOT/'tests/test_image_verification_availability.py',
        'protocol': ROOT.parent/'docs/image_verification_availability_protocol.md'}
    for module in (reportgate, imageworker, imageworker.image_interface, imageworker.frozen,
            sys.modules['contracts']):
        sources[module.__name__] = Path(module.__file__)
    for name in ('reliability_preview', 'scorer_reliability', 'legacy_replay_adapter',
            'invariant_verification', 'decision_preview'):
        sources[name] = ROOT/'src/tricompose_v12'/f'{name}.py'
    inputs = {}
    for label, (root, expected, names) in PARENTS.items():
        mp = require_inside(root/'manifest.json', PROTECTED_ROOT, must_exist=True)
        if sha256_file(mp) != expected:
            raise ValueError('fixed_parent_manifest_required')
        manifest = json.loads(mp.read_text(encoding='utf-8'))
        selection_field = 'original_selection_changed' if label == 'pool' else 'selection_changed'
        if (any(manifest.get(k) is not False for k in ('primary_metric_eligible',
                selection_field, 'regeneration_authorized'))
                or manifest.get('new_model_calls') != (6 if label == 'image' else 0)):
            raise ValueError('unqualified_fixed_parent_required')
        sources[label+'_manifest'] = mp
        for name in names:
            path = require_inside(root/name, root, must_exist=True)
            if (not path.is_file() or path.stat().st_size > 32*1024*1024
                    or sha256_file(path) != manifest['artifacts'][name]['sha256']):
                raise ValueError('bounded_fixed_metadata_artifact_required')
            if name.endswith('.csv'):
                with path.open(encoding='utf-8', newline='') as handle:
                    value = list(csv.DictReader(handle))
                if any(None in row or None in row.values() for row in value):
                    raise ValueError('complete_csv_cells_required')
            elif name.endswith('.jsonl'):
                value = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
            else:
                value = json.loads(path.read_text(encoding='utf-8'))
            inputs[label, name] = value
            sources[label+'_'+name] = path
        if sha256_file(mp) != expected:
            raise ValueError('parent_changed_during_read')
    return inputs, sources


def index_evidence(rows, facts, prediction, comparisons):
    """Exact image identity/hash plus exact report slot, never hash-only spread."""
    if (set(prediction) != {'schema_version', 'records', 'frozen', 'image_only',
            'image_prompt_sha256', 'model_received_ehr_reports_scores_or_candidate_ids'}
            or prediction['schema_version'] != imageworker.SCHEMA
            or prediction['frozen'] is not True or prediction['image_only'] is not True
            or prediction['model_received_ehr_reports_scores_or_candidate_ids'] is not False
            or not HASH.fullmatch(prediction['image_prompt_sha256'])):
        raise ValueError('frozen_image_only_prediction_required')
    candidates = {r['triple_candidate_id']: r for r in rows}
    originals = {(f['triple_candidate_id'], f['finding']): f for f in facts}
    if len(candidates) != len(rows) or len(originals) != len(facts):
        raise ValueError('unique_original_inventory_required')
    images = {}
    for row in rows:
        identity = tuple(row[k] for k in ('case_id', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256'))
        if images.setdefault(row['cxr_candidate_id'], identity) != identity:
            raise ValueError('fixed_shared_image_anchor_required')
    records = {}
    for rec in prediction['records']:
        iid = rec['cxr_candidate_id']
        if (set(rec) != RECORD_FIELDS or iid in records or iid not in images
                or rec['cxr_sha256'] != images[iid][1]
                or not HASH.fullmatch(rec['response_sha256'])
                or rec['independent_clinical_validation'] is not False
                or type(rec['token_limit_reached']) is not bool
                or any(type(rec[k]) is not int or rec[k] < 1 for k in ('input_tokens', 'output_tokens'))
                or type(rec['elapsed_seconds']) not in (int, float)
                or not math.isfinite(rec['elapsed_seconds']) or rec['elapsed_seconds'] < 0):
            raise ValueError('exact_quote_free_image_record_required')
        if rec['contract_status'] == 'complete':
            if (not isinstance(rec['states'], dict) or set(rec['states']) != HEADS
                    or not set(rec['states'].values()) <= STATES
                    or rec['token_limit_reached'] is not False or rec['failure_reason'] is not None):
                raise ValueError('complete_eight_state_image_contract_required')
        elif (rec['contract_status'] != 'failed_unavailable' or rec['states'] is not None
                or rec['failure_reason'] not in ('token_cap', 'invalid_eight_state_json')):
            raise ValueError('failed_image_requires_null_states')
        records[iid] = rec
    gates = []
    for comp in comparisons:
        if set(comp) != reportgate.GATE_FIELDS | IMAGE_FIELDS:
            raise ValueError('quote_free_comparison_fields_required')
        gates.append({k: comp[k] for k in reportgate.GATE_FIELDS})
    gate_index, gate_groups = reportgate.index_gate(rows, facts, gates)
    lookup, seen_images = {}, set()
    for comp in comparisons:
        key = comp['triple_candidate_id'], comp['finding']
        fact = originals[key]
        rec = records.get(fact['cxr_candidate_id'])
        if (rec is None or comp['imagecheck_cxr_candidate_id'] != fact['cxr_candidate_id']
                or comp['imagecheck_cxr_sha256'] != fact['artifact_hashes']['cxr_sha256']
                or comp['imagecheck_contract_status'] != rec['contract_status']
                or comp['imagecheck_raw_xrv_state'] != fact['states']['xrv']
                or any(comp[k] is not False for k in ('imagecheck_independent_clinical_validation',
                    'imagecheck_primary_metric_eligible', 'imagecheck_regeneration_authorized'))
                or comp['imagecheck_confirmed_faulty_modality'] is not None):
            raise ValueError('exact_unqualified_image_comparison_required')
        finding, gate = comp['finding'], gate_index[key]
        qwen, image_relation, report_relation = image_relations(rec, fact, gate)
        if (comp['imagecheck_qwen_state'] != qwen
                or comp['imagecheck_xrv_qwen_relation'] != image_relation
                or comp['imagecheck_retained_report_relation'] != report_relation):
            raise ValueError('unchanged_four_state_relations_required')
        expected = reportgate.fact_extension(gate)
        if any(fact.get(k) != v for k, v in expected.items()):
            raise ValueError('original_reportgate_metadata_required')
        lookup[key] = comp
        seen_images.add(fact['cxr_candidate_id'])
    if not records or seen_images != set(records):
        raise ValueError('all_image_results_require_report_inventory')
    return records, lookup, gate_groups


def image_relations(rec, fact, gate=None):
    if fact['finding'] not in HEADS:
        return None, 'outside_image_verifier_scope', 'outside_image_verifier_scope'
    if rec['contract_status'] != 'complete':
        return None, 'image_verifier_unavailable', 'image_verifier_unavailable'
    state = rec['states'][fact['finding']]
    xrv_relation = imageworker.relation(fact['states']['xrv'], state)
    report_relation = ('no_exact_report_check' if gate is None else
        'report_assertion_not_retained' if gate['scopegate_decision'] != 'scope_commit' else
        imageworker.relation(state, gate['scopegate_retained_state']))
    return state, xrv_relation, report_relation


def annotate_pool(rows, facts, records, comparisons):
    annotated, by_candidate, unique = [], defaultdict(list), {}
    for fact in facts:
        if any(k.startswith('imageverify_') for k in fact):
            raise ValueError('already_attached_fact_refused')
        rec = records.get(fact['cxr_candidate_id'])
        comp = comparisons.get((fact['triple_candidate_id'], fact['finding']))
        state, image_relation, report_relation = ((None, 'not_checked', 'not_checked')
            if rec is None else image_relations(rec, fact, comp))
        status = ('not_checked' if rec is None else image_relation if image_relation in
            ('outside_image_verifier_scope', 'image_verifier_unavailable') else 'checked_clinically_unqualified')
        extension = {'imageverify_status': status, 'imageverify_qwen_state': state,
            'imageverify_response_sha256': None if rec is None else rec['response_sha256'],
            'imageverify_xrv_qwen_relation': image_relation,
            'imageverify_retained_report_relation': report_relation,
            'imageverify_exact_report_comparison_available': comp is not None,
            'imageverify_independent_clinical_validation': False,
            'imageverify_primary_metric_eligible': False, 'imageverify_confirmed_faulty_modality': None,
            'imageverify_regeneration_authorized': False}
        output = {**fact, **extension}
        annotated.append(output)
        by_candidate[fact['triple_candidate_id']].append(output)
        if rec is not None:
            key = fact['cxr_candidate_id'], fact['finding']
            value = {'cxr_candidate_id': key[0], 'cxr_sha256': rec['cxr_sha256'], 'finding': key[1],
                'raw_xrv_state': fact['states']['xrv'], 'qwen_state': state,
                'contract_status': rec['contract_status'], 'response_sha256': rec['response_sha256'],
                'xrv_qwen_relation': image_relation, 'independent_clinical_validation': False,
                'primary_metric_eligible': False, 'regeneration_authorized': False}
            if unique.setdefault(key, value) != value:
                raise ValueError('shared_image_finding_must_be_identical')
    candidates = []
    for row in rows:
        if any(k.startswith('imageverify_') for k in row):
            raise ValueError('already_attached_candidate_refused')
        rec = records.get(row['cxr_candidate_id'])
        own = by_candidate[row['triple_candidate_id']]
        if len(own) != 14 or {f['finding'] for f in own} != set(CHEXPERT_FINDINGS):
            raise ValueError('complete_candidate_finding_inventory_required')
        counts = Counter(f['imageverify_xrv_qwen_relation'] for f in own)
        report_counts = Counter(f['imageverify_retained_report_relation'] for f in own)
        extension = {'imageverify_status': 'not_checked' if rec is None else
            'checked_clinically_unqualified' if rec['contract_status'] == 'complete' else 'image_verifier_unavailable',
            'imageverify_shared_image_dependency_id': None if rec is None else row['cxr_candidate_id'],
            'imageverify_supported_head_checks': 0 if rec is None else len(HEADS),
            'imageverify_independent_clinical_validation': False,
            'imageverify_primary_metric_eligible': False, 'imageverify_regeneration_authorized': False}
        for name in RELATIONS:
            extension['imageverify_'+name+'_count'] = counts[name] if rec is not None else None
        for name in ('explicit_agreement_unqualified', 'explicit_opposition_unqualified',
                'report_assertion_not_retained', 'no_exact_report_check', 'uncertainty_not_comparable'):
            extension['imageverify_report_'+name+'_count'] = report_counts[name] if rec is not None else None
        candidates.append({**row, **extension})
    return candidates, annotated, [unique[k] for k in sorted(unique)]


def annotate_requests(requests, candidates, records, comparisons, facts):
    fact_index = {(f['triple_candidate_id'], f['finding']): f for f in facts}
    output, seen = [], set()
    for row in requests:
        kind, consumers = row['request_kind'], row['consumer_candidate_ids']
        if (row['request_id'] in seen or kind not in REQUEST_DEPENDENCIES
                or row['execution_status'] != 'not_executed'
                or any(row[k] is not False for k in ('clinical_truth_established', 'model_execution_allowed',
                    'reportcheck_clinically_resolved', 'reportgate_clinically_resolved'))
                or row['reportgate_new_model_calls'] != 0 or row['reportcheck_new_model_calls'] != 0
                or not consumers or len(set(consumers)) != len(consumers)
                or any(k.startswith('imageverify_') for k in row)):
            raise ValueError('unchanged_unique_unresolved_request_required')
        seen.add(row['request_id'])
        applicable = kind in ('verify_image_finding', 'verify_image_report_relation')
        matched, statuses = [], Counter()
        for cid in consumers:
            candidate = candidates.get(cid)
            if (candidate is None or candidate['case_id'] != row['case_id']
                    or set(row['dependency_hashes']) != set(REQUEST_DEPENDENCIES[kind])
                    or any(candidate[k] != v for k, v in row['dependency_hashes'].items())):
                raise ValueError('exact_existing_consumer_dependencies_required')
            if not applicable:
                continue
            key = cid, row['finding']
            fact = fact_index.get(key)
            if fact is None:
                raise ValueError('existing_request_finding_required')
            rec = records.get(candidate['cxr_candidate_id'])
            comp = comparisons.get(key)
            if rec is not None and (kind == 'verify_image_finding' or comp is not None):
                matched.append(cid)
                statuses[fact['imageverify_xrv_qwen_relation'] if kind == 'verify_image_finding'
                    else fact['imageverify_retained_report_relation']] += 1
        status = 'not_checked' if applicable else 'not_applicable_request_kind'
        if matched:
            status = next(iter(statuses)) if len(statuses) == 1 else 'mixed_consumer_availability'
            if len(matched) < len(consumers):
                status = 'partial_consumer_check'
        output.append({**row, 'imageverify_status': status,
            'imageverify_matched_consumer_ids': sorted(matched),
            'imageverify_consumer_relation_counts': dict(sorted(statuses.items())),
            'imageverify_unchecked_consumer_count': len(consumers)-len(matched) if applicable else 0,
            'imageverify_not_applicable_consumer_count': 0 if applicable else len(consumers),
            'imageverify_clinically_resolved': False, 'imageverify_new_model_calls': 0,
            'imageverify_regeneration_authorized': False})
    return output


def preserve(original, annotated):
    if len(original) != len(annotated) or any(any(new.get(k) != v for k, v in old.items())
            for old, new in zip(original, annotated)):
        raise ValueError('all_original_cells_and_order_must_be_preserved')


def summarize(rows, facts, requests, unique, records):
    checked = [r for r in rows if r['imageverify_status'] != 'not_checked']
    retained = [f for f in facts if f['imageverify_exact_report_comparison_available']
        and f['reportgate_retained_state'] is not None and f['finding'] in HEADS]
    request_by_kind = {}
    for kind in sorted(REQUEST_DEPENDENCIES):
        subset = [r for r in requests if r['request_kind'] == kind]
        request_by_kind[kind] = {'logical_requests': len(subset),
            'with_cached_image_sidecar': sum(bool(r['imageverify_matched_consumer_ids']) for r in subset),
            'status_counts': dict(sorted(Counter(r['imageverify_status'] for r in subset).items()))}
    return {'schema_version': SCHEMA, 'candidate_rows': len(rows), 'fact_rows': len(facts),
        'fixed_ehr_cases': len({r['case_id'] for r in rows}),
        'image_slots': len({r['cxr_candidate_id'] for r in rows}), 'checked_image_slots': len(records),
        'checked_candidate_slots': len(checked), 'unchecked_candidate_slots': len(rows)-len(checked),
        'unique_supported_image_finding_checks': sum(r['finding'] in HEADS for r in unique),
        'unique_image_finding_relation_counts': dict(sorted(Counter(r['xrv_qwen_relation'] for r in unique).items())),
        'candidate_finding_image_status_counts': dict(sorted(Counter(f['imageverify_status'] for f in facts).items())),
        'candidate_finding_report_relation_counts': dict(sorted(Counter(f['imageverify_retained_report_relation'] for f in facts).items())),
        'retained_report_occurrences': len(retained),
        'retained_report_distinct_image_findings': len({(f['cxr_candidate_id'], f['finding']) for f in retained}),
        'logical_requests': len(requests), 'request_availability_by_kind': request_by_kind,
        'requests_with_cached_image_sidecar': sum(bool(r['imageverify_matched_consumer_ids']) for r in requests),
        'clinically_resolved_requests': 0, 'original_cells_and_order_preserved': True,
        'cached_image_run_model_calls': len(records), 'new_model_calls': 0, 'new_slurm_submissions': 0,
        'primary_metric_eligible': False, 'independent_clinical_accuracy': None,
        'selection_changed': False, 'fixed_ehr_changed': False, 'regeneration_authorized': False,
        'scorer_opposition_is_clinical_error': False, 'image_text_agreement_is_independent_truth': False,
        'development_diagnostic_not_heldout': True, 'report_image_ehr_bodies_opened': False,
        'raw_patient_inputs_opened': False, 'reference_keys_opened': False}


def markdown(summary):
    lines = ['# Image verification availability / 图像核查接入候选表', '',
        '保留原始评分、标签、择优结果及请求历史，仅增加 imageverify_ 诊断字段。', '',
        '| Inventory / 范围 | Count |', '|---|---:|',
        f"| Fixed EHR cases | {summary['fixed_ehr_cases']} |",
        f"| Total triples / checked slots | {summary['candidate_rows']} / {summary['checked_candidate_slots']} |",
        f"| Total CXR slots / checked images | {summary['image_slots']} / {summary['checked_image_slots']} |",
        f"| Unique supported image/finding checks | {summary['unique_supported_image_finding_checks']} |",
        f"| Retained report occurrences / distinct image-findings | {summary['retained_report_occurrences']} / {summary['retained_report_distinct_image_findings']} |", '',
        '| Unique image/finding relation / 评分器关系 | Count |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in summary['unique_image_finding_relation_counts'].items()]
    lines += ['', '| Request kind / 请求种类 | Existing requests | With cached image sidecar |', '|---|---:|---:|']
    lines += [f"| {k} | {v['logical_requests']} | {v['with_cached_image_sidecar']} |"
        for k, v in summary['request_availability_by_kind'].items()]
    lines += ['',
        '相同图像的四份报告不是四次独立看图；logical requests 不等于模型调用。',
        'XRV/Qwen 冲突是评分器分歧，不证明图像或报告错误。Unknown/uncertain/未检查不计为阴性或明确一致。',
        'Retained report agreement uses the same Qwen checkpoint across image/text; it is not independent clinical truth.',
        'Clinical requests resolved: 0. EHR edges, clinical scores and repair authorization remain unavailable.',
        'No new inference/API/GPU/submission, source-body access, ranking or regeneration.', '',
        'Start with `candidate_score_table.csv`; inspect per-finding and request JSONL for availability.',
        '`unique_image_finding_table.jsonl` has deduplicated image evidence, not repeated report votes.', '']
    return '\n'.join(lines)


def execute(output_root, run_id):
    reportgate.require_cpu_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        inputs, sources = load_fixed_inputs()
        before = {k: sha256_file(p) for k, p in sources.items()}
        rows, facts, requests = (inputs['availability', name] for name in
            ('candidate_score_table.csv', 'fact_verification_availability.jsonl', 'evidence_request_availability.jsonl'))
        groups = inputs['pool', 'report_dependency_groups.jsonl']
        prediction, comparisons, previous_summary = (inputs['image', name] for name in
            ('predictions.json', 'image_report_comparison.jsonl', 'summary.json'))
        if (len(rows), len(facts), len(requests), len(groups), len(comparisons)) != (960, 13440, 2072, 3360, 336):
            raise ValueError('fixed_full_bank_inventory_required')
        validate_sidecar(rows, facts, groups)
        records, lookup, gate_groups = index_evidence(rows, facts, prediction, comparisons)
        annotated, finding_rows, unique = annotate_pool(rows, facts, records, lookup)
        attached = annotate_requests(requests, {r['triple_candidate_id']: r for r in rows}, records, lookup, finding_rows)
        for old, new in ((rows, annotated), (facts, finding_rows), (requests, attached)):
            preserve(old, new)
        summary = summarize(annotated, finding_rows, attached, unique, records)
        if (summary['fixed_ehr_cases'], summary['image_slots'], len(records), len(gate_groups), len(unique),
                len({r['case_id'] for r in annotated if r['imageverify_status'] != 'not_checked'})) != (80, 240, 6, 24, 84, 2):
            raise ValueError('unchanged_six_image_twentyfour_slot_scope_required')
        if (summary['unique_image_finding_relation_counts'] != previous_summary['unique_image_finding_relation_counts']
                or dict(sorted(Counter(c['imagecheck_retained_report_relation'] for c in comparisons).items()))
                    != previous_summary['retained_report_relation_counts_candidate_findings']
                or previous_summary['model_calls'] != len(records)
                or previous_summary['clinical_requests_resolved'] != 0):
            raise ValueError('cached_result_denominators_must_match')
        summary['existing_cpu_job_id'] = os.environ['SLURM_JOB_ID']
        write_private_text(temporary/'candidate_score_table.csv', reportgate.csv_text(annotated))
        write_private_text(temporary/'fact_verification_availability.jsonl', reportgate.jsonl_text(finding_rows))
        write_private_text(temporary/'evidence_request_availability.jsonl', reportgate.jsonl_text(attached))
        write_private_text(temporary/'unique_image_finding_table.jsonl', reportgate.jsonl_text(unique))
        write_private_json(temporary/'summary.json', summary)
        write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        if any(sha256_file(p) != before[k] for k, p in sources.items()):
            raise ValueError('consumed_metadata_or_program_changed')
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'report_image_ehr_bodies_opened': False})
        for path in (temporary, *temporary.iterdir()):
            st = path.stat()
            if st.st_gid not in (96293, 65534) or st.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE/'image_verification_availability')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, summary = execute(args.output_root, args.run_id)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'cached_image_evidence_attached', 'candidate_rows': summary['candidate_rows'],
        'checked_candidate_slots': summary['checked_candidate_slots'],
        'requests_with_cached_image_sidecar': summary['requests_with_cached_image_sidecar'],
        'new_model_calls': 0, 'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
