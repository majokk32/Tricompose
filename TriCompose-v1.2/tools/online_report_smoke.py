#!/usr/bin/env python3
"""Prepared two-case online report escalation, frozen workers, private output.

GPU execution requires separately approved Slurm. CPU prepare does not infer.
Shadow static-control calls are not online observations or free computation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for relative in ('src', 'TriCompose-v1.0/src', 'TriCompose-v1.1/src',
                 'TriCompose-v1.2/src', 'TriCompose-v1.2/benchmarks',
                 'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(ROOT.parent / relative))
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_output_acceptance as gate
from contracts import (WORKSPACE, PROTECTED_ROOT, RUN_ID_PATTERN, require_inside,
    read_json, sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run,
    private_directory, write_private_json, write_private_text, load_cxr_candidates, load_report_candidates)
from tricompose_v12.invariant_verification import _digest, EXPLICIT
from tricompose_v12.live_workers import (registry, preflight_generator, source_pins,
    check_pins, require_gpu_slurm, single_generated, run_private_process)
from tricompose_v12.live_execution import one_case, attempt, validate_report_binding
from tricompose_v12.live_receipts import image_receipt, completed_receipt
from tricompose_v12.execution_ledger import restore_ledger, CallRequest, CallResult
from tricompose_v12.runtime_dispatch import ProtectedJournal, require_slurm
from tricompose_v11.cxr_contracts import canonical_json_sha256, validate_cxr_request
from tricompose_v11.report_contracts import prepare_report_request_run
from tricompose_v12.full_pool_report_control import validate_endpoint
from run_bounded_regeneration import candidate_row, load as load_parent
from run_fixed_image_reports import normalize_owned_modes
from run_automatic_proxy_replay import render_csv
from score_automatic_replay_biovil import PAIR_FIELDS, REQUEST_SCHEMA

SCHEMA = 'tricompose-online-report-escalation-smoke-v1'
POLICY = {'schema_version': SCHEMA, 'cxr_model': 'roentgen_v2', 'cxr_seed': 2,
    'report_models': ['cxrmate_single'], 'second_report_model': 'chexagent2',
    'case_scope': 'same_two_ehr_only_stratified_development_anchors_as_archived_retry',
    'call_budget_per_case': 6, 'max_retries_per_operation': 0,
    'timeout_seconds_per_worker_process': 180,
    'online_action_scope': 'same_image_report_escalation_only',
    'control': 'always_generate_second_expert_shadow_when_online_stops',
    'maximum_image_regenerations': 0, 'uses_biovil_for_decisions': False,
    'clinical_acceptance': False, 'training_allowed': False,
    'actual_gpu_savings_claim_allowed': False}
SOURCE_PLAN, SOURCE_SHA = gate.SOURCES['plan']
VENDOR = WORKSPACE / 'runtime/vendor/hi_ml_multimodal_0_2_2'


def next_report_action(row, ctx):
    row = gate.controller_view(row)
    facts = gate.validate_observation(row, ctx)
    edges = row['raw_edge_readouts']; s = row['structure']
    if edges['ehr_cxr']['proxy_opposition_facts']:
        return {'action': 'stop_unresolved_image_proxy',
            'reasons': ['direct_ehr_image_proxy_opposition_not_repairable_by_report_only'],
            'clinical_fault_location': None, 'clinical_acceptance': False}
    missing_positive = [f['finding'] for f in facts if f['xrv'] == 'positive' and f['chexbert'] not in EXPLICIT]
    missing_ehr = [f['finding'] for f in facts if f['ehr'] in EXPLICIT and f['chexbert'] not in EXPLICIT]
    reasons = []
    if not gate.report_gate.structure_ok(row): reasons.append('baseline_section_contract_failed')
    if any(s[k] for k in ('generic_report', 'unsupported_temporal_comparison_language')):
        reasons.append('baseline_common_report_risk_flag')
    if s['repeated_sentence_count'] or s['repeated_4gram_ratio']:
        reasons.append('baseline_repetition_flag')
    if edges['ehr_report']['proxy_opposition_facts']: reasons.append('direct_ehr_report_proxy_opposition')
    if edges['cxr_report']['proxy_opposition_facts']: reasons.append('image_report_proxy_opposition')
    if missing_positive: reasons.append('missing_image_positive_report_comparison')
    if missing_ehr: reasons.append('missing_direct_ehr_report_comparison')
    return {'action': 'switch_report_model' if reasons else 'stop_unverified', 'reasons': reasons,
        'missing_image_positive_finding_ids': missing_positive, 'missing_direct_ehr_finding_ids': missing_ehr,
        'clinical_fault_location': None, 'clinical_acceptance': False}


def observations(rows):
    return [{'row': r, 'origin': {'kind': 'completed_ledger_receipt'}} for r in rows]


def initial_decision(base, ctx, book):
    return gate.assess_fresh_output(base['triple_candidate_id'], base['triple_candidate_id'],
        observations([base]), ctx, ledger_snapshot=book)


def source_fingerprints():
    pins = source_pins()
    for path in (Path(__file__), ROOT / 'tests/test_online_report_smoke.py',
        ROOT.parent / 'docs/online_report_smoke_protocol.md',
        ROOT / 'tools/fresh_output_acceptance.py', ROOT / 'tools/score_free_random_control.py'):
        pins[str(path.resolve())] = sha256_file(path)
    return pins


def prepare(args):
    gate.cpu_guard()
    # Authenticate the old EHR-only case selection, source requests, local
    # environments, original inputs and all frozen assets. No model factory.
    parent = load_parent(argparse.Namespace(plan_run=SOURCE_PLAN, plan_manifest_sha256=SOURCE_SHA))
    before = source_fingerprints()
    workers = copy.deepcopy(parent['workers'])
    workers[POLICY['second_report_model']] = preflight_generator(registry()[POLICY['second_report_model']])
    cases = []
    for old in parent['cases']:
        request = copy.deepcopy(old['original_request'])
        request['seed'] = POLICY['cxr_seed']
        request['request_id'] = f"cxrreq_{request['case_id']}_{request['model_id']}_s{request['seed']:06d}"
        if {k: v for k, v in request.items() if k not in ('seed', 'request_id')} != {
                k: v for k, v in old['original_request'].items() if k not in ('seed', 'request_id')}:
            raise ValueError('only_predeclared_seed_may_change')
        validate_cxr_request(request)
        source = old['requests'][0]
        cases.append({'case_id': old['case_id'], 'anchor': old['anchor'],
            'ehr_anchor_sha256': old['ehr_anchor_sha256'], 'opaque_source_index': old['opaque_source_index'],
            'requests': [{'request': request, 'source_path': source['source_path'],
                'source_sha256': source['source_sha256'], 'canonical_request_sha256': canonical_json_sha256(request)}]})
    artifact_pins = {**parent['artifact_pins'], str(SOURCE_PLAN / 'manifest.json'): SOURCE_SHA,
        str(SOURCE_PLAN / 'plan.json'): sha256_file(SOURCE_PLAN / 'plan.json')}
    plan = {'schema_version': SCHEMA, 'policy': POLICY, 'cases': cases, 'workers': workers,
        'source_pins': before, 'artifact_pins': artifact_pins,
        'biovil_python': parent['biovil_python'], 'biovil_model': parent['biovil_model'],
        'biovil_asset_pins': parent['biovil_asset_pins'], 'vendor_path': str(VENDOR),
        'cohort': parent['cohort'], 'source_plan_manifest_sha256': SOURCE_SHA,
        'factory_instantiated': False, 'new_model_calls': 0, 'source_bodies_parsed': False,
        'clinical_acceptance': False, 'historical_pool_is_untouched_test': False,
        'planned_maximum': {'images': 2, 'reports': 4, 'charged_primary_attempts': 12,
            'secondary_image_encodings': 2, 'secondary_text_encodings': 4},
        'minimum_gpu_vram_gib': max(s['minimum_planning_vram_gib'] for s in workers.values())}
    if not (VENDOR / 'health_multimodal/__init__.py').is_file():
        raise ValueError('existing_biovil_vendor_runtime_required')
    validate_plan(plan)
    check_pins(before)
    temp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temp / 'plan.json', plan)
        write_private_json(temp / 'manifest.json', {'schema_version': SCHEMA,
            'status': 'prepared_cpu_only_gpu_not_submitted', 'plan_sha256': sha256_file(p),
            'planned_maximum': plan['planned_maximum'], 'new_model_calls': 0, 'clinical_acceptance': False})
        commit_atomic_run(temp, target)
    except BaseException:
        discard_atomic_run(temp)
        raise
    return target, plan


def validate_plan(plan):
    if (plan['schema_version'] != SCHEMA or plan['policy'] != POLICY or len(plan['cases']) != 2
            or len({c['case_id'] for c in plan['cases']}) != 2
            or any(plan[k] is not False for k in ('factory_instantiated', 'source_bodies_parsed',
                'clinical_acceptance', 'historical_pool_is_untouched_test')) or plan['new_model_calls'] != 0
            or plan['source_plan_manifest_sha256'] != SOURCE_SHA
            or set(plan['workers']) != {'roentgen_v2', 'cxrmate_single', 'xrv', 'chexbert', 'chexagent2'}
            or plan['planned_maximum'] != {'images': 2, 'reports': 4, 'charged_primary_attempts': 12,
                'secondary_image_encodings': 2, 'secondary_text_encodings': 4}
            or plan['minimum_gpu_vram_gib'] != max(s['minimum_planning_vram_gib'] for s in plan['workers'].values())):
        raise ValueError('exact_frozen_two_case_report_only_protocol_required')
    for case in plan['cases']:
        anchor = gate.anchor_from_record(case['anchor'])
        if case['ehr_anchor_sha256'] != anchor.sha256 or case['case_id'] != anchor.case_id or len(case['requests']) != 1:
            raise ValueError('immutable_ehr_case_required')
        row = case['requests'][0]; request = row['request']; validate_cxr_request(request)
        if (request['model_id'] != POLICY['cxr_model'] or request['seed'] != POLICY['cxr_seed']
                or request['case_id'] != anchor.case_id
                or request['inputs']['synthetic_ehr']['sha256'] != anchor.ehr_sha256
                or request['inputs']['ehr_facts']['sha256'] != anchor.ehr_facts_sha256
                or canonical_json_sha256(request) != row['canonical_request_sha256']):
            raise ValueError('fixed_ehr_and_prompt_request_required')
        gate.context(case['anchor'], plan['workers']['xrv'], plan['workers']['chexbert'])
    check_pins(plan['source_pins']); check_pins(plan['artifact_pins'])
    for spec in plan['workers'].values(): check_pins(spec['asset_pins'])
    check_pins(plan['biovil_asset_pins'])


def load(args):
    root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / 'manifest.json') != args.plan_manifest_sha256:
        raise ValueError('approved_plan_manifest_changed')
    m = read_json(root / 'manifest.json')
    if m['schema_version'] != SCHEMA or sha256_file(root / 'plan.json') != m['plan_sha256']:
        raise ValueError('approved_plan_changed')
    plan = read_json(root / 'plan.json'); validate_plan(plan)
    return plan


def additional_report(case, plan, root, baseline_triple, prefix_book):
    """Continue an authenticated prefix without overwriting its old journal."""
    anchor = gate.anchor_from_record(case['anchor']); cr = root / 'cases' / anchor.case_id
    workers = plan['workers']; model = POLICY['second_report_model']
    images = load_cxr_candidates([baseline_triple['cxr_run']]); image = next(iter(images.values()))
    if len(images) != 1 or image['artifact']['sha256'] != baseline_triple['cxr_sha256']:
        raise ValueError('one_fixed_generated_image_required')
    ip = cr / 'operations/xrv_0_a1/scored/cxr_finding_labels.json'
    il = read_json(ip)
    partial = image_receipt(anchor, image, il, label_sha256=sha256_file(ip),
        thresholds_sha256=workers['xrv']['thresholds_sha256'], checkpoint_sha256=workers['xrv']['checkpoint_sha256'])
    pack = prepare_report_request_run(cxr_runs=[baseline_triple['cxr_run']], output_root=cr / 'operations',
        run_id='request_report_second', model_ids=[model])
    pack_root = Path(pack['run_directory']); state = {}; triples = []
    with ProtectedJournal(cr / 'continuation.journal.jsonl') as journal:
        ledger = restore_ledger(prefix_book['events'], case_id=anchor.case_id,
            ehr_anchor_sha256=anchor.sha256, call_budget=prefix_book['call_budget'],
            max_retries=prefix_book['max_retries'], execution_mode=prefix_book['execution_mode'], sink=journal.append)
        if ledger.snapshot() != prefix_book:
            raise ValueError('unchanged_prefix_costs_required')
        def report_validator(payload, operation):
            report_run, report = single_generated(payload, workers[model], pack_root, cxr_candidates=images)
            validate_report_binding(report, image, pack_root, anchor)
            state.update(report=report, report_run=report_run)
            return CallResult(report['artifact']['sha256'])
        request = CallRequest('report_second', anchor.case_id, anchor.sha256, 'report_generator', model,
            _digest(workers[model]), POLICY['cxr_seed'], 'xrv_0', image['artifact']['sha256'])
        result = attempt(ledger, request, workers[model], cr, POLICY, report_validator, request_run=pack_root)
        if result is not None:
            def label_validator(payload, operation):
                if payload['worker_audit_sha256'] != operation.frozen_model_audit_sha256:
                    raise ValueError('frozen_report_scorer_audit_required')
                op_root = require_inside(payload['output_root'], PROTECTED_ROOT, must_exist=True)
                tp = op_root / 'scored/report_finding_labels.json'; tl = read_json(tp)
                receipt = completed_receipt(anchor, partial, image, il, state['report'], tl,
                    image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp),
                    thresholds_sha256=workers['xrv']['thresholds_sha256'],
                    xrv_checkpoint_sha256=workers['xrv']['checkpoint_sha256'],
                    chexbert_checkpoint_sha256=workers['chexbert']['checkpoint_sha256'])
                rp = write_private_json(op_root / 'completed_receipt.json', receipt)
                triples.append({'case_id': anchor.case_id, 'ehr_sha256': anchor.ehr_sha256,
                    'ehr_facts_sha256': anchor.ehr_facts_sha256, 'cxr_run': baseline_triple['cxr_run'],
                    'report_run': str(state['report_run']), 'receipt_path': str(rp), 'receipt_sha256': sha256_file(rp),
                    'cxr_sha256': image['artifact']['sha256'], 'report_sha256': state['report']['artifact']['sha256'],
                    'cxr_path': image['artifact']['path'], 'report_path': state['report']['artifact']['path'],
                    'report_model_id': model, 'cxr_model_id': image['model_id'], 'seed': image['seed']})
                return CallResult(sha256_file(tp), receipt['receipt_id'])
            label_req = CallRequest('chexbert_second', anchor.case_id, anchor.sha256, 'chexbert', 'chexbert',
                _digest(workers['chexbert']), POLICY['cxr_seed'], request.operation_id,
                image['artifact']['sha256'], result.output_artifact_sha256)
            attempt(ledger, label_req, workers['chexbert'], cr, POLICY, label_validator,
                cxr_run=baseline_triple['cxr_run'], report_run=state['report_run'])
        book = ledger.snapshot()
    write_private_json(cr / 'continuation_snapshot.json', book)
    if len(triples) > 1: raise ValueError('one_second_report_only')
    return triples, book


def seal_online(case_root, payload, journal):
    path = write_private_json(case_root / 'online_selection.json', payload)
    sealed = sha256_file(path)
    journal.append({'stage': 'online_selection_sealed', 'case_id': payload['case_id'],
        'selection_sha256': sealed, 'online_charged_attempts': payload['online_charged_attempts']})
    return sealed


def secondary(root, plan, rows, triples, selection_sha, journal):
    if not rows:
        return {'records': [], 'counts': None, 'status': 'not_available_no_completed_pairs',
            'used_for_routing': False, 'clinical_truth_available': False, 'historical_pool_is_untouched_test': False}
    pack = root / 'endpoint_request'; private_directory(pack)
    request = {'schema_version': REQUEST_SCHEMA, 'pairs': [{k: r[k] for k in PAIR_FIELDS} for r in rows],
        'selection_sha256_sealed_before_endpoint': selection_sha, 'modality_source': 'fully_synthetic',
        'selection_used_biovil': False, 'clinical_truth_available': False,
        'routing_or_calibration_update_allowed': False, 'text_policy': 'full_report_no_silent_truncation_overlength_is_na'}
    rp = write_private_json(pack / 'request.json', request)
    write_private_json(pack / 'manifest.json', {'schema_version': REQUEST_SCHEMA,
        'artifacts': {'request.json': {'sha256': sha256_file(rp)}}})
    argv = [plan['biovil_python'], str(Path(__file__)), 'biovil-worker', '--request-run', str(pack),
        '--model-path', plan['biovil_model'], '--output-file', str(root / 'endpoint.json')]
    for name in ('cxr_run', 'report_run'):
        for path in sorted({t[name] for t in triples}): argv += ['--' + name.replace('_', '-'), path]
    runtime = root / 'endpoint_runtime'; private_directory(runtime)
    journal.append({'stage': 'secondary_endpoint', 'status': 'reserved_before_spawn',
        'maximum_image_encodings': 2, 'maximum_text_encodings': 4, 'argv_sha256': _digest(argv)})
    started = time.monotonic()
    try:
        run_private_process(argv, runtime, 180)
        endpoint = read_json(root / 'endpoint.json')
        endpoint['historical_pool_is_untouched_test'] = False
        validate_endpoint(rows, endpoint)
        if (endpoint['request_sha256'] != sha256_file(rp)
                or endpoint['counts']['image_encoder_calls'] > 2 or endpoint['counts']['text_encoder_calls'] > 4):
            raise ValueError('sealed_endpoint_inventory_required')
        journal.append({'stage': 'secondary_endpoint', 'status': 'validated', 'counts': endpoint['counts'],
            'wall_seconds': time.monotonic() - started})
        return endpoint
    except Exception as exc:
        journal.append({'stage': 'secondary_endpoint', 'status': 'failed_charged_not_free',
            'error_type': type(exc).__name__, 'wall_seconds': time.monotonic() - started})
        return {'records': [{k: r[k] for k in PAIR_FIELDS} | {'biovil_raw_cosine': None,
            'status': 'not_available', 'reason': 'secondary_runtime_or_validation_failed', 'calibrated': False} for r in rows],
            'counts': None, 'used_for_routing': False, 'clinical_truth_available': False, 'historical_pool_is_untouched_test': False}


def method_rows(case_results, rows, endpoint):
    index = {r['triple_candidate_id']: r for r in rows}
    scores = {r['triple_candidate_id']: r for r in endpoint['records']}
    result = []
    for case in case_results:
        for method in ('fixed', 'online_report_escalation', 'always_second_static'):
            cid = case['selections'][method]
            row = index.get(cid)
            value = scores[cid]['biovil_raw_cosine'] if cid in scores else None
            result.append({'case_id': case['case_id'], 'method': method, 'selected_candidate_id': cid,
                'method_charged_attempts': case['method_costs'][method],
                'actual_collection_attempts_shared_with_controls': case['actual_collection_attempts'],
                'biovil_raw_cosine': value, 'clinical_acceptance': False,
                'raw_edge_readouts': row['raw_edge_readouts'] if row is not None else None,
                'selection_status': case['statuses'][method]})
    return result


def method_csv(rows):
    """A fixed CSV schema even when a case has no eligible completed output."""
    fields = ('inventory_findings', 'known_reference_facts', 'comparable_facts', 'supported_facts',
        'supported_positive', 'supported_negative', 'proxy_opposition_facts', 'missing_comparisons',
        'coverage_over_known', 'support_over_known', 'clinical_accuracy')
    output = []
    for row in rows:
        record = {k: v for k, v in row.items() if k != 'raw_edge_readouts'}
        edges = row['raw_edge_readouts']
        for edge in ('ehr_cxr', 'ehr_report', 'cxr_report'):
            record.update({edge + '_' + field: edges[edge][field] if edges is not None else None for field in fields})
        output.append({k: 'NA' if v is None else v for k, v in record.items()})
    return render_csv(output)


def run(args):
    require_gpu_slurm()  # Must precede inputs, output creation or CUDA imports.
    plan = load(args)
    import torch
    if torch.cuda.get_device_properties(0).total_memory < plan['minimum_gpu_vram_gib'] * 1024 ** 3:
        raise RuntimeError('allocated_gpu_below_frozen_worker_planning_memory')
    if not RUN_ID_PATTERN.fullmatch(args.run_id): raise ValueError('opaque_run_id_required')
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True); root = output / args.run_id
    private_directory(root); private_directory(root / 'cases')
    started = time.monotonic(); rows, triples, books, results = [], [], [], []
    write_private_json(root / 'start_manifest.json', {'schema_version': SCHEMA, 'policy': POLICY,
        'plan_manifest_sha256': args.plan_manifest_sha256, 'status': 'in_progress',
        'clinical_acceptance': False, 'automatic_resume': False})
    try:
        with ProtectedJournal(root / 'controller.journal.jsonl') as journal:
            for case in plan['cases']:
                initial, prefix, _ = one_case(case, plan, root)
                cr = root / 'cases' / case['case_id']
                if len(initial) > 1: raise ValueError('one_initial_candidate_required')
                if not initial:
                    payload = {'case_id': case['case_id'], 'status': 'unresolved_initial_chain_incomplete',
                        'decision': None, 'online_charged_attempts': prefix['charged_model_attempts']}
                    sealed = seal_online(cr, payload, journal)
                    results.append({'case_id': case['case_id'], 'selections': {m: None for m in
                        ('fixed', 'online_report_escalation', 'always_second_static')},
                        'method_costs': {m: prefix['charged_model_attempts'] for m in ('fixed', 'online_report_escalation', 'always_second_static')},
                        'statuses': {m: 'unresolved_initial_chain_incomplete' for m in ('fixed', 'online_report_escalation', 'always_second_static')},
                        'actual_collection_attempts': prefix['charged_model_attempts'],
                        'online_selection_sha256': sealed, 'online_second_observed': False})
                    books.append(prefix); continue
                base = candidate_row(initial[0]); rows.append(base); triples.extend(initial)
                ctx = gate.context(case['anchor'], plan['workers']['xrv'], plan['workers']['chexbert'])
                base_decision = initial_decision(base, ctx, prefix)
                action = next_report_action(base, ctx)
                journal.append({'stage': 'route', 'case_id': case['case_id'], 'action': action,
                    'prefix_ledger_sha256': _digest(prefix)})
                online_second = action['action'] == 'switch_report_model'
                if not online_second:
                    payload = {'case_id': case['case_id'], 'action': action, 'decision': base_decision,
                        'online_charged_attempts': prefix['charged_model_attempts'], 'status': action['action']}
                    sealed = seal_online(cr, payload, journal)  # BEFORE shadow control.
                    journal.append({'stage': 'shadow_static_only', 'case_id': case['case_id'],
                        'not_visible_to_sealed_online_policy': True})
                journal.append({'stage': 'second_expert_start', 'case_id': case['case_id'],
                    'online_request': online_second, 'prefix_ledger_sha256': _digest(prefix)})
                more, full = additional_report(case, plan, root, initial[0], prefix)
                journal.append({'stage': 'second_expert_finished', 'case_id': case['case_id'],
                    'full_ledger_sha256': _digest(full), 'completed_candidates': len(more)})
                alt = candidate_row(more[0]) if more else None
                if alt is not None: rows.append(alt); triples.extend(more)
                observed = observations([base] + ([alt] if alt is not None else []))
                static_decision = gate.assess_fresh_output(base['triple_candidate_id'],
                    alt['triple_candidate_id'] if alt is not None else None, observed, ctx, ledger_snapshot=full)
                if online_second:
                    online_decision = static_decision
                    payload = {'case_id': case['case_id'], 'action': action, 'decision': online_decision,
                        'online_charged_attempts': full['charged_model_attempts'], 'status': online_decision['status']}
                    sealed = seal_online(cr, payload, journal)
                else:
                    online_decision = base_decision
                if sha256_file(cr / 'online_selection.json') != sealed:
                    raise ValueError('online_selection_changed_after_shadow')
                results.append({'case_id': case['case_id'], 'action': action,
                    'selections': {'fixed': base_decision['selected_candidate_id'],
                        'online_report_escalation': online_decision['selected_candidate_id'],
                        'always_second_static': static_decision['selected_candidate_id']},
                    'method_costs': {'fixed': prefix['charged_model_attempts'],
                        'online_report_escalation': payload['online_charged_attempts'],
                        'always_second_static': full['charged_model_attempts']},
                    'statuses': {'fixed': base_decision['status'], 'online_report_escalation': payload['status'],
                        'always_second_static': static_decision['status']},
                    'actual_collection_attempts': full['charged_model_attempts'], 'online_second_observed': online_second,
                    'online_selection_sha256': sealed, 'static_decision': static_decision})
                books.append(full)
            selection = {'schema_version': SCHEMA, 'policy': POLICY, 'cases': results,
                'clinical_acceptance': False, 'secondary_used': False}
            sp = write_private_json(root / 'selection.json', selection); selection_sha = sha256_file(sp)
            journal.append({'stage': 'all_selections_sealed', 'selection_sha256': selection_sha})
            write_private_json(root / 'score_rows.json', {'records': rows})
            write_private_json(root / 'completed_triplets.json', {'records': triples})
            write_private_json(root / 'execution_summary.json', {'case_ledgers': books})
            endpoint = secondary(root, plan, rows, triples, selection_sha, journal)
            if sha256_file(sp) != selection_sha: raise ValueError('choice_changed_after_secondary')
            write_private_json(root / 'endpoint_sidecar.json', endpoint)
            methods = method_rows(results, rows, endpoint)
            write_private_json(root / 'method_comparison.json', {'records': methods})
            write_private_text(root / 'method_comparison.csv', method_csv(methods))
            check_pins(plan['source_pins']); check_pins(plan['artifact_pins'])
            if sum(b['charged_model_attempts'] for b in books) > 12:
                raise ValueError('whole_experiment_budget_exceeded')
            completed_kinds = Counter()
            for book in books:
                requests = {e['request']['operation_id']: e['request']['kind'] for e in book['events'] if e['event'] == 'attempt_reserved'}
                completed_kinds.update(requests[e['operation_id']] for e in book['events'] if e['event'] == 'attempt_completed')
            summary = {'schema_version': SCHEMA, 'status': 'completed_online_report_smoke_unvalidated',
                'fixed_ehr_cases': 2, 'new_images': completed_kinds['cxr_generator'],
                'new_report_artifacts': completed_kinds['report_generator'],
                'completed_new_reports': len(rows), 'online_second_expert_cases': sum(r['online_second_observed'] for r in results),
                'actual_collection_charged_attempts': sum(b['charged_model_attempts'] for b in books),
                'method_charged_attempts': {m: sum(r['method_costs'][m] for r in results) for m in
                    ('fixed', 'online_report_escalation', 'always_second_static')},
                'shadow_controls_count_as_actual_work': True, 'actual_gpu_savings_demonstrated': False,
                'clinical_acceptance': False, 'clinical_repair_success': False,
                'historical_pool_is_untouched_test': False, 'gpu_seconds': None,
                'runtime_seconds_including_loading_io_and_controls': round(time.monotonic() - started, 3),
                'original_ehr_prompts_or_winners_changed': False, 'plan_manifest_sha256': args.plan_manifest_sha256,
                'selection_sha256_before_secondary': selection_sha}
            write_private_json(root / 'summary.json', summary)
            files = ('selection.json', 'score_rows.json', 'completed_triplets.json', 'execution_summary.json',
                'endpoint_sidecar.json', 'method_comparison.json', 'method_comparison.csv', 'summary.json', 'controller.journal.jsonl')
            write_private_json(root / 'manifest.json', {**summary,
                'artifacts': {name: {'sha256': sha256_file(root / name)} for name in files}})
    except BaseException as exc:
        write_private_json(root / 'failure_manifest.json', {'schema_version': SCHEMA,
            'status': 'failed_or_interrupted_retained', 'error_type': type(exc).__name__,
            'automatic_resume': False, 'clinical_acceptance': False})
        raise
    finally:
        normalize_owned_modes(root)
    return root, summary


def audit(args):
    """Metadata-only audit: no report-structure body re-evaluation or pixels."""
    require_slurm(); plan = load(args)
    root = require_inside(args.source_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(root / 'manifest.json') != args.source_manifest_sha256:
        raise ValueError('completed_source_manifest_changed')
    m = read_json(root / 'manifest.json')
    if (m['schema_version'] != SCHEMA or m['status'] != 'completed_online_report_smoke_unvalidated'
            or m['plan_manifest_sha256'] != args.plan_manifest_sha256
            or any(m[k] is not False for k in ('clinical_acceptance', 'clinical_repair_success',
                'historical_pool_is_untouched_test', 'actual_gpu_savings_demonstrated', 'original_ehr_prompts_or_winners_changed'))):
        raise ValueError('exact_completed_unvalidated_run_required')
    for name, entry in m['artifacts'].items():
        p = require_inside(root / name, root, must_exist=True)
        if p.stat().st_size > 32 * 1024 ** 2 or sha256_file(p) != entry['sha256']:
            raise ValueError('completed_metadata_artifact_changed')
    import stat
    for p in (root, *root.rglob('*')):
        info = p.lstat()
        if (p.is_symlink() or info.st_gid not in (96293, 65534)
                or stat.S_IMODE(info.st_mode) != (0o2770 if p.is_dir() else 0o660)):
            raise ValueError('private_new_run_permissions_required')
    selection = read_json(root / 'selection.json')
    if sha256_file(root / 'selection.json') != m['selection_sha256_before_secondary'] or selection['policy'] != POLICY:
        raise ValueError('sealed_policy_choice_changed')
    rows = read_json(root / 'score_rows.json')['records']; triples = read_json(root / 'completed_triplets.json')['records']
    if len(rows) != len(triples) or len(rows) > 4 or len({r['triple_candidate_id'] for r in rows}) != len(rows):
        raise ValueError('unique_bounded_generated_inventory_required')
    books = read_json(root / 'execution_summary.json')['case_ledgers']
    book_index = {b['case_id']: b for b in books}
    result_index = {c['case_id']: c for c in selection['cases']}
    if set(book_index) != {c['case_id'] for c in plan['cases']} or set(result_index) != set(book_index):
        raise ValueError('all_fixed_ehr_cases_including_failures_required')
    journal = [json.loads(line) for line in (root / 'controller.journal.jsonl').read_text().splitlines()]
    recomputed = 0
    for case in plan['cases']:
        anchor = gate.anchor_from_record(case['anchor']); ctx = gate.context(case['anchor'], plan['workers']['xrv'], plan['workers']['chexbert'])
        cr = root / 'cases' / case['case_id']; book = book_index[case['case_id']]
        gate.validate_ledger(book, anchor)
        if book['call_budget'] != 6 or book['max_retries'] != 0 or book['execution_mode'] != 'approved_slurm_backend':
            raise ValueError('actual_bounded_execution_profile_required')
        prefix = read_json(cr / 'ledger_snapshot.json'); gate.validate_ledger(prefix, anchor)
        original_events = [json.loads(line) for line in (cr / 'execution.journal.jsonl').read_text().splitlines()]
        continued = cr / 'continuation.journal.jsonl'
        added = [json.loads(line) for line in continued.read_text().splitlines()] if continued.exists() else []
        if original_events != prefix['events'] or book['events'] != original_events + added:
            raise ValueError('unchanged_prefix_and_exact_continuation_events_required')
        if added and read_json(cr / 'continuation_snapshot.json') != book:
            raise ValueError('final_cost_snapshot_changed')
        group = [r for r in rows if r['case_id'] == case['case_id']]
        for row in group:
            gate.validate_observation(gate.controller_view(row), ctx); gate.completed_operation(row, book)
            matches = [t for t in triples if t['case_id'] == row['case_id'] and t['report_sha256'] == row['report_sha256']]
            if len(matches) != 1: raise ValueError('exact_generated_triple_binding_required')
            triple = matches[0]
            images = load_cxr_candidates([triple['cxr_run']])
            reports = load_report_candidates([triple['report_run']], cxr_candidates=images)
            if len(images) != 1 or len(reports) != 1: raise ValueError('single_case_generated_artifacts_required')
            image, report = next(iter(images.values())), next(iter(reports.values()))
            from tricompose_v12.live_execution import validate_cxr_binding
            validate_cxr_binding(image, case['requests'][0], anchor)
            if image['artifact']['sha256'] != row['cxr_sha256'] or report['artifact']['sha256'] != row['report_sha256']:
                raise ValueError('generated_artifact_hashes_differ')
            cb_op = 'chexbert_0_0_a1' if row['report_model_id'] == 'cxrmate_single' else 'chexbert_second_a1'
            ip = cr / 'operations/xrv_0_a1/scored/cxr_finding_labels.json'
            tp = cr / 'operations' / cb_op / 'scored/report_finding_labels.json'
            il, tl = read_json(ip), read_json(tp)
            partial = image_receipt(anchor, image, il, label_sha256=sha256_file(ip),
                thresholds_sha256=ctx['thresholds_sha256'], checkpoint_sha256=ctx['xrv_checkpoint_sha256'])
            receipt = completed_receipt(anchor, partial, image, il, report, tl,
                image_labels_sha256=sha256_file(ip), report_labels_sha256=sha256_file(tp),
                thresholds_sha256=ctx['thresholds_sha256'], xrv_checkpoint_sha256=ctx['xrv_checkpoint_sha256'],
                chexbert_checkpoint_sha256=ctx['chexbert_checkpoint_sha256'])
            if (receipt != row['receipt'] or sha256_file(triple['receipt_path']) != triple['receipt_sha256']
                    or read_json(triple['receipt_path']) != receipt):
                raise ValueError('fresh_receipt_not_recomputable_from_frozen_source_labels')
            recomputed += 1
        result = result_index[case['case_id']]
        op = cr / 'online_selection.json'
        if sha256_file(op) != result['online_selection_sha256']: raise ValueError('online_seal_changed')
        payload = read_json(op)
        base = next((r for r in group if r['report_model_id'] == 'cxrmate_single'), None)
        alt = next((r for r in group if r['report_model_id'] == 'chexagent2'), None)
        if base is None:
            if any(result['selections'].values()) or group or result['online_second_observed']:
                raise ValueError('incomplete_prefix_cannot_invent_output')
            expected_cost = prefix['charged_model_attempts']
        else:
            action = next_report_action(base, ctx)
            online_second = action['action'] == 'switch_report_model'
            if result['action'] != action or result['online_second_observed'] != online_second:
                raise ValueError('online_route_changed')
            initial = initial_decision(base, ctx, prefix)
            static = gate.assess_fresh_output(base['triple_candidate_id'], alt['triple_candidate_id'] if alt else None,
                observations([base] + ([alt] if alt else [])), ctx, ledger_snapshot=book)
            expected_online = static if online_second else initial
            if (payload['decision'] != expected_online or result['static_decision'] != static
                    or result['selections'] != {'fixed': initial['selected_candidate_id'],
                        'online_report_escalation': expected_online['selected_candidate_id'],
                        'always_second_static': static['selected_candidate_id']}):
                raise ValueError('source_only_online_and_static_decisions_differ')
            expected_cost = book['charged_model_attempts'] if online_second else prefix['charged_model_attempts']
            events = [e for e in journal if e.get('case_id') == case['case_id']]
            stages = [e['stage'] for e in events]
            seal_index = stages.index('online_selection_sealed'); start_index = stages.index('second_expert_start')
            if (online_second and seal_index < start_index) or (not online_second and seal_index > start_index):
                raise ValueError('online_shadow_chronology_differ')
        if (payload['online_charged_attempts'] != expected_cost
                or result['method_costs'] != {'fixed': prefix['charged_model_attempts'],
                    'online_report_escalation': expected_cost, 'always_second_static': book['charged_model_attempts']}
                or result['actual_collection_attempts'] != book['charged_model_attempts']):
            raise ValueError('cost_prefix_refund_or_hidden_control_charge')
    seal_index = next(i for i, e in enumerate(journal) if e['stage'] == 'all_selections_sealed')
    endpoint_indices = [i for i, e in enumerate(journal) if e['stage'] == 'secondary_endpoint']
    if endpoint_indices and min(endpoint_indices) <= seal_index:
        raise ValueError('endpoint_values_accessed_before_choices_sealed')
    endpoint = read_json(root / 'endpoint_sidecar.json')
    if rows: validate_endpoint(rows, endpoint)
    expected_methods = method_rows(selection['cases'], rows, endpoint)
    if read_json(root / 'method_comparison.json')['records'] != expected_methods:
        raise ValueError('method_readout_changed')
    expected_csv = method_csv(expected_methods)
    if (root / 'method_comparison.csv').read_text() != expected_csv:
        raise ValueError('method_csv_changed')
    if sum(b['charged_model_attempts'] for b in books) != m['actual_collection_charged_attempts']:
        raise ValueError('aggregate_collection_cost_changed')
    receipt = {'schema_version': SCHEMA, 'status': 'metadata_audit_passed_not_clinical',
        'source_manifest_sha256': args.source_manifest_sha256, 'recomputed_completed_receipts': recomputed,
        'fixed_ehr_cases': 2, 'new_model_calls': 0, 'source_bodies_or_pixels_opened': False,
        'actual_gpu_savings_demonstrated': False, 'clinical_acceptance': False}
    temp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temp / 'audit.json', receipt)
        write_private_json(temp / 'manifest.json', {'schema_version': SCHEMA, 'status': receipt['status'],
            'audit_sha256': sha256_file(p), 'new_model_calls': 0})
        commit_atomic_run(temp, target)
    except BaseException:
        discard_atomic_run(temp)
        raise
    return target, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True)
    pre = sub.add_parser('prepare')
    for name in ('output-root', 'run-id'): pre.add_argument('--' + name, required=True)
    launch = sub.add_parser('run')
    for name in ('plan-run', 'plan-manifest-sha256', 'output-root', 'run-id'): launch.add_argument('--' + name, required=True)
    review = sub.add_parser('audit')
    for name in ('plan-run', 'plan-manifest-sha256', 'source-run', 'source-manifest-sha256', 'output-root', 'run-id'):
        review.add_argument('--' + name, required=True)
    worker = sub.add_parser('biovil-worker')
    for name in ('request-run', 'model-path', 'output-file'): worker.add_argument('--' + name, required=True)
    for name in ('cxr-run', 'report-run'): worker.add_argument('--' + name, action='append', required=True)
    args = p.parse_args(); os.umask(0o007)
    try:
        if args.mode == 'biovil-worker':
            require_gpu_slurm(); sys.path.insert(0, str(VENDOR))
            from score_automatic_replay_biovil import score
            write_private_json(require_inside(args.output_file, PROTECTED_ROOT, must_exist=False), score(args))
            return 0
        root, result = prepare(args) if args.mode == 'prepare' else audit(args) if args.mode == 'audit' else run(args)
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1
    print(json.dumps({'status': 'cpu_prepared_gpu_not_submitted' if args.mode == 'prepare' else result['status'],
        'manifest_sha256': sha256_file(root / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
