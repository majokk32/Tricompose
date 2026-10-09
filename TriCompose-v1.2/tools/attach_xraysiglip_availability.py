#!/usr/bin/env python3
"""Append frozen alternate text-preference readouts, never clinical decisions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_verification_frontier as frontier
import probe_xraysiglip_findings as probe
from contracts import (PROTECTED_ROOT, CHEXPERT_FINDINGS, require_inside,
    sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)

tables, guard = probe.tables, probe.guard
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
SCHEMA = 'tricompose-full-bank-xraysiglip-availability-v1'
RUN_SHA = 'd846602fabba7abca85745e18fd5daba00832ac251ce458bd9f53ff3fa3c54eb'
PARENTS = {
    'frontier': (BASE / 'verification_frontiers/frontier_pool960_12645021_validated',
        '204b499d2eeab6cd7f849ca854a8beefbc433a53876bb03b507a3aa56e0493bc',
        ('candidate_action_table.csv', 'supplemental_image_evidence_requests.jsonl',
            'evidence_request_frontier.jsonl', 'summary.json')),
    'live': (BASE / 'guarded_image_availability/liveimage_pool960_12645021_001',
        '0120146adc043b482f51ef515d70ad3dd1b78132c3b989b22e7bd918b3c68dcb',
        ('fact_verification_availability.jsonl', 'unique_image_finding_table.jsonl')),
    'plan': (BASE / 'xraysiglip_plans/siglip_scope2_12645021_001',
        'fbf4d8bd4e5bf93078533ef90f4e65a2d861a53cc2296269fd31570898be3f89', ('plan.json',)),
    'siglip': (BASE / 'xraysiglip_runs/siglip_scope2_12654030', RUN_SHA,
        ('predictions.json', 'request_readout_comparisons.jsonl', 'summary.json', 'attempt_journal.jsonl')),
}


def load_inputs():
    values, texts = {}, {}
    sources = {'worker': Path(__file__), 'tests': ROOT / 'tests/test_xraysiglip_availability.py',
        'test_fixture_dependency': ROOT / 'tests/test_verification_frontier.py',
        'protocol': ROOT.parent / 'docs/xraysiglip_availability_protocol.md'}
    for module in (frontier, probe, tables, frontier.image_tables, guard, frontier.cached,
            frontier.cached.image_interface, frontier.cached.frozen, sys.modules['contracts']):
        sources[module.__name__] = Path(module.__file__)
    for name in ('reliability_preview', 'scorer_reliability', 'legacy_replay_adapter',
            'invariant_verification', 'decision_preview'):
        sources[name] = ROOT / 'src/tricompose_v12' / (name + '.py')
    for label, (root, expected, names) in PARENTS.items():
        mp = require_inside(root / 'manifest.json', PROTECTED_ROOT, must_exist=True)
        if mp.stat().st_size > 1024 * 1024 or sha256_file(mp) != expected:
            raise ValueError('fixed_parent_manifest_required')
        manifest = json.loads(mp.read_text(encoding='utf-8'))
        if any(manifest[k] is not False for k in
                ('selection_changed', 'primary_metric_eligible', 'regeneration_authorized')):
            raise ValueError('unchanged_unqualified_parent_required')
        sources[label + '_manifest'] = mp
        for name in names:
            path = require_inside(root / name, root, must_exist=True)
            if (not path.is_file() or path.stat().st_size > 64 * 1024 * 1024
                    or sha256_file(path) != manifest['artifacts'][name]['sha256']):
                raise ValueError('bounded_hash_bound_metadata_required')
            text = path.read_text(encoding='utf-8')
            texts[label, name] = text
            if name.endswith('.csv'):
                with path.open(encoding='utf-8', newline='') as handle:
                    value = list(csv.DictReader(handle))
                if not value or any(None in row or None in row.values() for row in value):
                    raise ValueError('complete_csv_cells_required')
            elif name.endswith('.jsonl'):
                value = [json.loads(line) for line in text.splitlines() if line]
            else:
                value = json.loads(text)
            values[label, name] = value
            sources[label + '_' + name] = path
        if sha256_file(mp) != expected:
            raise ValueError('parent_changed_during_read')
    return values, texts, sources


def validate_readouts(plan, predictions, summary, journal):
    if (predictions['schema_version'] != probe.SCHEMA or summary['schema_version'] != probe.SCHEMA
            or predictions['frozen'] is not True or predictions['image_only'] is not True
            or predictions['probe_catalog_sha256'] != guard.digest(probe.probes())):
        raise ValueError('unchanged_frozen_image_only_probe_required')
    records = predictions['records']
    inputs = {r['observation_id']: r for r in plan['inputs']}
    if (len(inputs) != len(plan['inputs']) or len(records) != len(inputs)
            or [r['observation_id'] for r in records] != sorted(inputs)):
        raise ValueError('exact_sorted_unique_input_inventory_required')
    attempts, expected_journal, forward_count, complete_count = 0, [], 0, 0
    result = {}
    for record in records:
        observation = record['observation_id']
        item = inputs[observation]
        receipt = guard.validate_receipt(record['guard'])
        if (record['artifact_sha256'] != item['sha256'] or receipt['artifact_sha256'] != item['sha256']
                or receipt['normalized_pixel_sha256'] not in (None, observation)
                or record['independent_clinical_validation'] is not False
                or type(record['callback_invoked']) is not bool
                or type(record['model_forward_attempted']) is not bool):
            raise ValueError('exact_unqualified_receipt_binding_required')
        if record['callback_invoked']:
            attempts += 1
            if receipt['basic_comparison_permitted'] is not True:
                raise ValueError('blocked_input_cannot_receive_readout')
            if record['pairs'] is not None:
                if (record['contract_status'] != 'complete' or record['failure_reason'] is not None
                        or record['model_forward_attempted'] is not True):
                    raise ValueError('complete_readout_requires_forward')
                complete_count += 1
                probe.summarize_pairs(record['pairs'])
            elif (record['contract_status'] != 'failed_unavailable'
                    or not isinstance(record['failure_reason'], str)
                    or not record['failure_reason'].isidentifier()):
                raise ValueError('failed_attempt_requires_null_unavailable_readout')
            probe.finite_number(record['elapsed_seconds'])
            if record['elapsed_seconds'] < 0:
                raise ValueError('nonnegative_attempt_runtime_required')
            expected_journal.extend([
                {'event': 'reserved', 'scoring_attempt': attempts, 'observation_id': observation},
                {'event': 'completed' if record['pairs'] is not None else 'failed_unavailable',
                    'scoring_attempt': attempts, 'observation_id': observation,
                    'model_forward_attempted': record['model_forward_attempted'],
                    'failure_reason': record['failure_reason'], 'elapsed_seconds': record['elapsed_seconds']}])
        elif (receipt['basic_comparison_permitted'] is not False
                or record['contract_status'] != 'blocked_before_callback'
                or record['pairs'] is not None or record['model_forward_attempted'] is not False
                or record['elapsed_seconds'] is not None or record['failure_reason'] != receipt['reason']):
            raise ValueError('blocked_input_must_have_null_score_and_no_call')
        forward_count += record['model_forward_attempted']
        result[observation] = record
    expected = {'unique_inputs': len(records), 'scoring_attempts': attempts,
        'actual_model_forward_attempts': forward_count, 'blocked_before_callback': len(records) - attempts,
        'complete_model_responses': complete_count, 'failed_unavailable_responses': attempts - complete_count,
        'model_retries': 0, 'clinical_requests_resolved': 0}
    if (any(type(summary[k]) is not int or summary[k] != v for k, v in expected.items())
            or type(plan['max_scoring_attempts']) is not int or attempts > plan['max_scoring_attempts']
            or journal != expected_journal or summary['clinical_accuracy'] is not None
            or summary['posthoc_comparisons_computed_after_prediction_fsync'] is not True
            or summary['shares_vision_encoder_with_chexagent2'] is not True
            or summary['training_overlap_unverified'] is not True
            or summary['distinct_from_qwen_and_xrv_not_independent_clinical_truth'] is not True
            or type(summary['model_load_attempts']) is not int or summary['model_load_attempts'] != int(attempts > 0)
            or any(summary[k] is not False for k in ('primary_metric_eligible', 'selection_changed',
                'regeneration_authorized', 'fixed_ehr_changed', 'calibration_or_policy_fitting',
                'raw_patient_inputs_opened', 'ehr_report_or_real_target_bodies_opened'))):
        raise ValueError('exact_actual_attempt_accounting_and_unqualified_status_required')
    return result


def extension(record, finding):
    value = None
    if record is None:
        status = 'not_checked'
    elif finding not in guard.HEADS:
        status = 'outside_siglip_probe_scope'
    elif record['pairs'] is None:
        status = 'readout_unavailable'
    else:
        value = probe.summarize_pairs(record['pairs'])[finding]
        status = 'checked_template_sensitive' if value['text_preference'] in ('template_sensitive', 'tied_templates') else 'checked_text_preference_unqualified'
    return {'siglip_status': status, 'siglip_readout': value,
        'siglip_run_manifest_sha256': None if record is None else RUN_SHA,
        'siglip_guard_receipt_sha256': None if record is None else record['guard']['receipt_sha256'],
        'siglip_clinical_state': None, 'siglip_calibrated_probability': None,
        'siglip_independent_clinical_validation': False, 'siglip_primary_metric_eligible': False,
        'siglip_regeneration_authorized': False, 'siglip_shares_encoder_with_chexagent2': True}


def direction(state, value):
    if state not in guard.STATES:
        raise ValueError('known_four_state_proxy_required')
    preference = None if value is None else value['text_preference']
    if state not in ('positive', 'negative') or preference not in ('present_prompt_higher', 'absent_prompt_higher'):
        return 'not_comparable'
    return 'same_direction_unqualified' if (state == 'positive') == (preference == 'present_prompt_higher') else 'opposed_direction_unqualified'


def build(rows, facts, unique, supplements, plan, predictions, comparisons, run_summary, journal):
    if any(any(k.startswith('siglip_') for k in row) for group in (rows, facts, unique, supplements) for row in group):
        raise ValueError('repeat_attachment_refused')
    core = [{k: v for k, v in row.items() if not k.startswith('frontier_')} for row in rows]
    candidates, lookup, consumers = frontier.index_inputs(core, facts, unique)
    refs = probe.bind_requests(plan, supplements)
    if refs != plan['request_refs']:
        raise ValueError('exact_sealed_request_refs_required')
    records = validate_readouts(plan, predictions, run_summary, journal)
    expected_comparisons = probe.posthoc(predictions['records'], plan, supplements)
    if comparisons != expected_comparisons or run_summary['logical_requests_compared'] != len(supplements):
        raise ValueError('exact_sealed_posthoc_comparisons_required')
    original_slots = {s['cxr_candidate_id']: s for s in plan['logical_slots'] if s['arm'] == 'original'}
    original_ids = {s['observation_id'] for s in original_slots.values()}
    controls = {s['observation_id'] for s in plan['logical_slots'] if s['arm'] != 'original'}
    if original_ids & controls or any(records[key]['callback_invoked'] for key in controls):
        raise ValueError('controls_cannot_be_candidates_or_model_calls')
    attached_records = {}
    requests_by_image = defaultdict(list)
    for request in supplements:
        requests_by_image[request['cxr_candidate_id']].append(request)
    for image_id, slot in original_slots.items():
        cids = sorted(consumers.get(image_id, ()))
        if not cids:
            raise ValueError('only_existing_image_ids_can_join')
        record = records[slot['observation_id']]
        for cid in cids:
            if any(lookup[cid, head]['liveimage_guard_receipt_sha256'] != record['guard']['receipt_sha256']
                    for head in CHEXPERT_FINDINGS):
                raise ValueError('same_exact_guard_for_every_original_image_finding_required')
        for request in requests_by_image[image_id]:
            if sorted(request['consumer_candidate_ids']) != cids:
                raise ValueError('all_and_only_original_report_consumers_required')
            for cid in cids:
                candidate, fact = candidates[cid], lookup[cid, request['finding']]
                if (candidate['case_id'] != request['case_id'] or candidate['cxr_sha256'] != slot['cxr_sha256']
                        or fact['states']['xrv'] != request['raw_xrv_state']
                        or fact['liveimage_qwen_state'] != request['current_readout']
                        or fact['liveimage_guard_receipt_sha256'] != record['guard']['receipt_sha256']):
                    raise ValueError('exact_case_image_hash_guard_and_proxy_binding_required')
        attached_records[image_id] = record
    fact_output, groups, unique_output = [], defaultdict(list), []
    for fact in facts:
        ext = extension(attached_records.get(fact['cxr_candidate_id']), fact['finding'])
        ext['siglip_vs_current_qwen'] = 'not_comparable' if fact['liveimage_qwen_state'] is None else direction(fact['liveimage_qwen_state'], ext['siglip_readout'])
        ext['siglip_vs_raw_xrv'] = direction(fact['states']['xrv'], ext['siglip_readout'])
        new = {**fact, **ext}
        fact_output.append(new)
        groups[fact['triple_candidate_id']].append(new)
    for value in unique:
        exemplar = next(f for f in groups[consumers[value['cxr_candidate_id']][0]] if f['finding'] == value['finding'])
        unique_output.append({**value, **{k: v for k, v in exemplar.items() if k.startswith('siglip_')}})
    candidate_output = []
    for row in rows:
        record = attached_records.get(row['cxr_candidate_id'])
        own = groups[row['triple_candidate_id']]
        counts = Counter(f['siglip_status'] for f in own)
        candidate_output.append({**row, 'siglip_status': 'not_checked' if record is None else
            'readout_unavailable' if record['pairs'] is None else 'checked_image_only_unqualified',
            'siglip_run_manifest_sha256': None if record is None else RUN_SHA,
            'siglip_supported_readout_count': None if record is None else sum(f['siglip_readout'] is not None for f in own),
            'siglip_stable_text_preference_count': None if record is None else counts['checked_text_preference_unqualified'],
            'siglip_template_sensitive_count': None if record is None else counts['checked_template_sensitive'],
            'siglip_unavailable_count': None if record is None else counts['readout_unavailable'],
            'siglip_outside_scope_count': None if record is None else counts['outside_siglip_probe_scope'],
            'siglip_clinical_selection_score': None, 'siglip_independent_clinical_validation': False,
            'siglip_clinical_selection_eligible': False, 'siglip_primary_metric_eligible': False,
            'siglip_regeneration_authorized': False, 'siglip_report_factuality_verified': False,
            'siglip_shares_encoder_with_chexagent2': True})
    request_output = []
    for request, comparison in zip(supplements, comparisons):
        ext = extension(attached_records[request['cxr_candidate_id']], request['finding'])
        request_output.append({**request, **ext, 'siglip_vs_current_qwen': comparison['vs_current_qwen'],
            'siglip_vs_raw_xrv': comparison['vs_raw_xrv'], 'siglip_independent_evidence_requirement_satisfied': False,
            'siglip_clinically_resolved': False, 'siglip_confirmed_faulty_modality': None})
    for old, new in ((rows, candidate_output), (facts, fact_output), (unique, unique_output), (supplements, request_output)):
        frontier.image_tables.preserve(old, new)
    supported = [r for r in unique_output if r['siglip_readout'] is not None]
    summary = {'schema_version': SCHEMA, 'candidate_rows': len(rows), 'fact_rows': len(facts),
        'fixed_ehr_cases': len({r['case_id'] for r in rows}), 'fixed_image_slots': len(consumers),
        'checked_image_slots': len(attached_records),
        'checked_candidate_slots': sum(r['siglip_status'] != 'not_checked' for r in candidate_output),
        'unchecked_candidate_slots': sum(r['siglip_status'] == 'not_checked' for r in candidate_output),
        'unique_supported_image_findings': len(supported), 'unique_image_finding_inventory': len(unique_output),
        'unique_text_preference_counts': dict(sorted(Counter(r['siglip_readout']['text_preference'] for r in supported).items())),
        'fact_status_counts': dict(sorted(Counter(r['siglip_status'] for r in fact_output).items())),
        'supplemental_logical_requests': len(supplements),
        'supplemental_candidate_finding_links': sum(len(r['consumer_candidate_ids']) for r in supplements),
        'supplemental_text_preference_counts': dict(sorted(Counter(r['siglip_readout']['text_preference'] if r['siglip_readout'] else 'unavailable' for r in request_output).items())),
        'supplemental_vs_current_qwen': dict(sorted(Counter(r['siglip_vs_current_qwen'] for r in request_output).items())),
        'supplemental_vs_raw_xrv': dict(sorted(Counter(r['siglip_vs_raw_xrv'] for r in request_output).items())),
        'historical_siglip_scoring_attempts': run_summary['scoring_attempts'],
        'historical_siglip_actual_forward_attempts': run_summary['actual_model_forward_attempts'],
        'historical_guard_blocks': run_summary['blocked_before_callback'],
        'old_cells_and_order_preserved': True, 'old_execution_history_unchanged': True,
        'fixed_ehr_changed': False, 'selection_changed': False, 'primary_metric_eligible': False,
        'regeneration_authorized': False, 'clinical_requests_resolved': 0, 'clinical_accuracy': None,
        'clinical_selection_score': None, 'independent_evidence_requirement_satisfied': False,
        'new_model_calls': 0, 'new_slurm_submissions': 0,
        'ehr_report_image_or_weight_bodies_opened': False, 'raw_patient_inputs_opened': False}
    return candidate_output, fact_output, unique_output, request_output, summary


def execute(output_root, run_id):
    tables.require_cpu_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        values, texts, sources = load_inputs()
        before = {k: sha256_file(p) for k, p in sources.items()}
        rows = values['frontier', 'candidate_action_table.csv']
        facts = values['live', 'fact_verification_availability.jsonl']
        unique = values['live', 'unique_image_finding_table.jsonl']
        requests = values['frontier', 'supplemental_image_evidence_requests.jsonl']
        original_requests = values['frontier', 'evidence_request_frontier.jsonl']
        if (len(rows), len(facts), len(unique), len(requests), len(original_requests)) != (960, 13440, 84, 21, 2072):
            raise ValueError('fixed_full_bank_denominators_required')
        outputs = build(rows, facts, unique, requests, values['plan', 'plan.json'],
            values['siglip', 'predictions.json'], values['siglip', 'request_readout_comparisons.jsonl'],
            values['siglip', 'summary.json'], values['siglip', 'attempt_journal.jsonl'])
        candidates, annotated, shared, supplemented, summary = outputs
        if (summary['fixed_ehr_cases'], summary['fixed_image_slots'], summary['checked_image_slots'],
                summary['checked_candidate_slots']) != (80, 240, 6, 24):
            raise ValueError('fixed_original_checked_scope_required')
        summary.update(original_logical_requests=len(original_requests),
            original_request_history_byte_preserved=True, existing_cpu_job_id=os.environ['SLURM_JOB_ID'])
        write_private_text(temporary / 'candidate_score_table.csv', tables.csv_text(candidates))
        for name, records in (('fact_verification_availability.jsonl', annotated),
                ('unique_image_finding_table.jsonl', shared), ('supplemental_image_evidence_requests.jsonl', supplemented)):
            write_private_text(temporary / name, tables.jsonl_text(records))
        history = write_private_text(temporary / 'evidence_request_frontier.jsonl', texts['frontier', 'evidence_request_frontier.jsonl'])
        if sha256_file(history) != before['frontier_evidence_request_frontier.jsonl']:
            raise ValueError('original_request_history_byte_preservation_required')
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md',
            '# Full-bank alternate evidence / 全候选辅助证据表\n\n'
            'All original scores and actions stay unchanged; siglip_ columns are availability/readouts only.\n\n'
            '模板偏好不是临床标签；未检查不变 0；不对争议进行多数投票或自动返工。\n\n'
            '48 image/finding readouts are shared among 24 report consumers, not independent votes.\n\n'
            '```json\n' + json.dumps(summary, sort_keys=True, indent=2) + '\n```\n')
        if any(sha256_file(path) != before[key] for key, path in sources.items()):
            raise ValueError('consumed_source_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'selection_changed': False, 'primary_metric_eligible': False,
            'regeneration_authorized': False, 'ehr_report_image_or_weight_bodies_opened': False})
        probe.private_modes(temporary)
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE / 'xraysiglip_availability')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, _ = execute(args.output_root, args.run_id)
        print(json.dumps({'status': 'completed_append_only_no_selection_change',
            'new_model_calls': 0, 'manifest_sha256': sha256_file(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
