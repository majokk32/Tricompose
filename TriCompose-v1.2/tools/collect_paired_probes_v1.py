#!/usr/bin/env python3
"""Frozen 2x2 paired-action collection, NOT online adaptive execution.

Prepare reads only hash-bound synthetic plan metadata, not bodies/weights.
Run requires separately approved GPU Slurm. Preserve inputs and existing runs.
Choices on the newly collected bank are sealed before secondary BioViL scoring.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in ('src', 'TriCompose-v1.0/src', 'TriCompose-v1.1/src',
        'TriCompose-v1.2/src', 'TriCompose-v1.2/benchmarks',
        'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(ROOT.parent/directory))
sys.path.insert(0, str(ROOT/'tools'))
from contracts import (PROTECTED_ROOT, WORKSPACE, RUN_ID_PATTERN, require_inside,
    read_json, sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run,
    private_directory, write_private_json, write_private_text)
from tricompose_v12 import probe_repair_v1 as controller
from tricompose_v12.live_workers import require_gpu_slurm, check_pins
from tricompose_v12.live_execution import one_case, score_csv
from tricompose_v12.live_plan import anchor_from_record
from tricompose_v11.cxr_contracts import canonical_json_sha256
from run_bounded_regeneration import candidate_row
from run_fixed_image_reports import normalize_owned_modes
from score_automatic_replay_biovil import PAIR_FIELDS, REQUEST_SCHEMA
import fresh_output_acceptance as fresh
from benchmark_probe_repair_v1 import cpu_guard, checked

VERSION = 'tricompose-paired-probe-collection-v1'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
PARENT = BASE/'online_report_plans/online2_12654973_001'
PARENT_SHA = 'fca888e160ce692eaf100aa5cddf3aabd504c283a1b17e35ce6bf64c628dafac'
POLICY = {'schema_version': VERSION, 'image_order': [['roentgen_v2', 3], ['roentgen_v2', 4]],
    'report_models': ['cxrmate_single', 'chexagent2'], 'controller_budget': 10,
    'call_budget_per_case': 12, 'max_retries_per_operation': 0,
    'timeout_seconds_per_worker_process': 180, 'selection_uses_secondary': False,
    'clinical_acceptance': False, 'training_allowed': False,
    'case_scope': 'unchanged_two_ehr_only_stratified_development_anchors',
    'collection_is_online_controller': False, 'deadline': '2026-10-11'}


def make_plan(parent, pins, code_pins):
    if (parent.get('schema_version') != 'tricompose-online-report-escalation-smoke-v1'
            or len(parent['cases']) != 2 or len({c['case_id'] for c in parent['cases']}) != 2
            or any(parent[k] is not False for k in ('factory_instantiated', 'source_bodies_parsed',
                'clinical_acceptance', 'historical_pool_is_untouched_test'))):
        raise ValueError('authenticated_development_parent_required')
    cases = []
    for old in parent['cases']:
        anchor = anchor_from_record(old['anchor'])
        if old['case_id'] != anchor.case_id or old['ehr_anchor_sha256'] != anchor.sha256 \
                or len(old['requests']) != 1:
            raise ValueError('unchanged_synthetic_ehr_anchor_required')
        source = old['requests'][0]
        original = source['request']
        if original['model_id'] != 'roentgen_v2' or canonical_json_sha256(original) != source['canonical_request_sha256']:
            raise ValueError('authenticated_original_roentgen_request_required')
        rows = []
        for model, seed in POLICY['image_order']:
            request = deepcopy(original)
            request['seed'] = seed
            request['request_id'] = f"cxrreq_{anchor.case_id}_{model}_s{seed:06d}"
            if {k:v for k,v in request.items() if k not in ('seed', 'request_id')} != \
                    {k:v for k,v in original.items() if k not in ('seed', 'request_id')}:
                raise ValueError('only_preregistered_seed_may_change')
            rows.append({**deepcopy(source), 'request': request,
                'canonical_request_sha256': canonical_json_sha256(request)})
        cases.append({**deepcopy(old), 'requests': rows})
    return {'schema_version': VERSION, 'policy': deepcopy(POLICY), 'cases': cases,
        'workers': deepcopy(parent['workers']), 'source_pins': {**parent['source_pins'], **code_pins},
        'artifact_pins': {**parent['artifact_pins'], **pins},
        'biovil_python': parent['biovil_python'], 'biovil_model': parent['biovil_model'],
        'biovil_asset_pins': deepcopy(parent['biovil_asset_pins']),
        'parent_manifest_sha256': PARENT_SHA, 'preflight_asset_pins_reused_not_recomputed': True,
        'minimum_gpu_vram_gib': parent['minimum_gpu_vram_gib'],
        'source_bodies_or_weights_read_during_prepare': False, 'new_model_calls': 0,
        'clinical_acceptance': False, 'planned_maximum': {'images': 4, 'reports': 8,
            'generation_verification_attempts': 24, 'secondary_image_encodings': 4,
            'secondary_text_encodings': 8}}


def prepare(args):
    cpu_guard()
    sources = {}
    receipt = json.loads(checked(PARENT/'manifest.json', PARENT_SHA, 1024**2, sources))
    parent = json.loads(checked(PARENT/'plan.json', receipt['plan_sha256'], 4*1024**2, sources))
    code_paths = [Path(__file__), Path(controller.__file__), ROOT/'tests/test_paired_probe_collection_v1.py']
    code_pins = {str(p.resolve()): sha256_file(p) for p in code_paths}
    plan = make_plan(parent, sources, code_pins)
    validate_plan(plan)
    check_pins({**sources, **code_pins})  # Small metadata/source only; NO weight hashing here.
    temp, target = new_atomic_run(args.output_root, args.run_id)
    try:
        p = write_private_json(temp/'plan.json', plan)
        write_private_json(temp/'manifest.json', {'schema_version': VERSION,
            'status': 'metadata_prepared_gpu_not_submitted', 'plan_sha256': sha256_file(p),
            'planned_maximum': plan['planned_maximum'], 'new_model_calls': 0})
        commit_atomic_run(temp, target)
    except BaseException:
        discard_atomic_run(temp); raise
    return target, {'status': 'metadata_prepared_gpu_not_submitted'}


def validate_plan(plan):
    if (plan['schema_version'] != VERSION or plan['policy'] != POLICY
            or plan['parent_manifest_sha256'] != PARENT_SHA or len(plan['cases']) != 2
            or len({c['case_id'] for c in plan['cases']}) != 2
            or set(plan['workers']) != {'roentgen_v2', 'cxrmate_single', 'chexagent2', 'xrv', 'chexbert'}
            or plan['source_bodies_or_weights_read_during_prepare'] is not False
            or plan['new_model_calls'] != 0 or plan['clinical_acceptance'] is not False
            or plan['planned_maximum'] != {'images': 4, 'reports': 8, 'generation_verification_attempts': 24,
                'secondary_image_encodings': 4, 'secondary_text_encodings': 8}):
        raise ValueError('exact_two_case_paired_collection_plan_required')
    if plan['minimum_gpu_vram_gib'] != max(w['minimum_planning_vram_gib'] for w in plan['workers'].values()):
        raise ValueError('frozen_worker_memory_requirement_required')
    for case in plan['cases']:
        anchor = anchor_from_record(case['anchor'])
        if anchor.case_id != case['case_id'] or anchor.sha256 != case['ehr_anchor_sha256'] \
                or len(case['requests']) != 2:
            raise ValueError('fixed_anchor_and_two_seed_requests_required')
        for row, slot in zip(case['requests'], POLICY['image_order']):
            r = row['request']
            if ([r['model_id'], r['seed']] != slot or r['case_id'] != anchor.case_id
                    or r['inputs']['synthetic_ehr']['sha256'] != anchor.ehr_sha256
                    or r['inputs']['ehr_facts']['sha256'] != anchor.ehr_facts_sha256
                    or canonical_json_sha256(r) != row['canonical_request_sha256']):
                raise ValueError('unchanged_input_bound_seed_request_required')
        a, b = (r['request'] for r in case['requests'])
        if {k:v for k,v in a.items() if k not in ('seed', 'request_id')} != \
                {k:v for k,v in b.items() if k not in ('seed', 'request_id')}:
            raise ValueError('same_final_prompt_for_both_seeds_required')


def fresh_snapshot(row, ctx):
    """Explicit NEW eight-enabled-head profile, never mix with legacy cache."""
    row = fresh.controller_view(row)
    facts = fresh.validate_observation(row, ctx)
    anchor = anchor_from_record(ctx['anchor'])
    structure_ok = fresh.report_gate.structure_ok(row)
    s = row['structure']
    risks = sum(int(bool(s[k])) for k in ('empty', 'generic_report',
        'unsupported_temporal_comparison_language', 'repeated_sentence_count', 'repeated_4gram_ratio'))
    value = {'schema_version': controller.VERSION+'-observation', 'case_id': row['case_id'],
        'candidate_id': row['triple_candidate_id'], 'lineage': {k:row[k] for k in
            ('ehr_sha256', 'ehr_facts_sha256', 'cxr_sha256', 'report_sha256',
                'cxr_candidate_id', 'report_candidate_id', 'cxr_model_id', 'report_model_id')},
        'states': {k:{f['finding']:f[k] for f in facts} for k in ('ehr', 'xrv', 'chexbert')},
        'ehr_sources': {name:sorted(sources) for name, state, sources in anchor.findings},
        'quality': {'cxr_basic_validity_pass': True,
            'report_structure_quality_score_0_1': float(structure_ok)},
        'artifact_gate_failures': risks + int(not structure_ok), 'clinical_qualified': False}
    value['lineage']['cxr_seed'] = row['seed']
    return controller.validate_snapshot(value)


def paired_readout(rows, ctx):
    observed = {controller.ProbeController._slot(v): v for v in (fresh_snapshot(r, ctx) for r in rows)}
    allowed = {(*slot, model) for slot in POLICY['image_order'] for model in POLICY['report_models']}
    if len(observed) != len(rows) or not set(observed) <= allowed:
        raise ValueError('unique_registered_completed_model_seed_slots_required')
    initial_slot = (*POLICY['image_order'][0], POLICY['report_models'][0])
    if initial_slot not in observed:
        return {'status': 'initial_chain_incomplete_retained', 'choices': [], 'action_credits': []}
    choices, credits = [], []
    for feedback in (True, False):
        c = controller.ProbeController(observed[initial_slot], POLICY['image_order'],
            POLICY['report_models'], POLICY['controller_budget'], feedback_enabled=feedback)
        while True:
            request = c.propose()
            if request['action'] == 'stop': break
            obs = observed.get(tuple(request['slot']))
            c.complete(request, obs, failure_type='unavailable_cache_slot' if obs is None else None)
        r = c.result()
        r.update(method='observed_probe_replay' if feedback else 'without_feedback_replay',
            actual_regeneration_executed=False, collected_bank_contains_fresh_outputs=True,
            selection_was_online=False)
        choices.append(r)
    # Same-image report probes AND paired image probes holding report expert.
    for slot, before in sorted(observed.items()):
        for other, after in sorted(observed.items()):
            if slot == other: continue
            action = 'report_probe' if slot[:2] == other[:2] else 'image_probe' if slot[2] == other[2] else None
            if action: credits.append(controller.action_credit(before, after, action))
    choices.insert(0, {'method': 'fixed', 'selected_candidate_id': observed[initial_slot]['candidate_id'],
        'simulated_calls': 4, 'actual_regeneration_executed': False, 'selection_was_online': False})
    return {'status': 'development_paired_readout', 'choices': choices, 'action_credits': credits,
        'fresh_profile': fresh.PROFILE, 'quality_scope': 'binary_official_section_contract_not_clinical_quality',
        'image_validity_scope': 'authenticated_png_metadata_not_anatomy', 'clinical_repair_success': None}


def collect_case(case, plan, root):
    """One stable directory per image seed; each branch pays exactly <=6 calls.

    Do not hand two same-model seeds to the old fixed-path worker: its private
    prompt-copy filenames intentionally refuse a second write. No old code or
    overwrite policy is changed to work around that invariant.
    """
    triples, books = [], []
    for index, request in enumerate(case['requests']):
        branch = root/f'image_branch_{index}'
        private_directory(branch, exist_ok=True)
        private_directory(branch/'cases', exist_ok=True)
        branch_case = {**case, 'requests': [request]}
        branch_plan = {**plan, 'policy': {**plan['policy'], 'call_budget_per_case': 6}}
        completed, book, _ = one_case(branch_case, branch_plan, branch)
        if book['charged_model_attempts'] > 6: raise ValueError('branch_budget_exceeded')
        triples.extend(completed); books.append(book)
    return triples, books


def run(args):
    require_gpu_slurm()  # BEFORE reading plans, CUDA, source bodies or weights.
    p = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
    if sha256_file(p/'manifest.json') != args.plan_manifest_sha256:
        raise ValueError('approved_plan_manifest_changed')
    m = read_json(p/'manifest.json')
    if sha256_file(p/'plan.json') != m['plan_sha256']: raise ValueError('approved_plan_changed')
    plan = read_json(p/'plan.json'); validate_plan(plan)
    check_pins(plan['source_pins']); check_pins(plan['artifact_pins'])
    for w in plan['workers'].values(): check_pins(w['asset_pins'])
    check_pins(plan['biovil_asset_pins'])
    import torch
    if torch.cuda.get_device_properties(0).total_memory < plan['minimum_gpu_vram_gib']*1024**3:
        raise RuntimeError('gpu_below_planning_vram_requirement')
    if not RUN_ID_PATTERN.fullmatch(args.run_id): raise ValueError('opaque_new_run_id_required')
    output = require_inside(args.output_root, PROTECTED_ROOT, must_exist=False)
    private_directory(output, exist_ok=True); root = output/args.run_id
    private_directory(root); private_directory(root/'cases')
    started = time.monotonic(); triples, rows, books, readouts = [], [], [], []
    write_private_json(root/'start_manifest.json', {'schema_version': VERSION, 'status': 'in_progress',
        'plan_manifest_sha256': args.plan_manifest_sha256, 'policy': POLICY})
    try:
        for case in plan['cases']:
            completed, branch_books = collect_case(case, plan, root)
            current = [candidate_row(t) for t in completed]
            ctx = fresh.context(case['anchor'], plan['workers']['xrv'], plan['workers']['chexbert'])
            readouts.append({'case_id': case['case_id'], 'readout': paired_readout(current, ctx)})
            triples.extend(completed); rows.extend(current); books.extend(branch_books)
        sealed = write_private_json(root/'choices_and_action_credits.json', {'records': readouts})
        seal = sha256_file(sealed)
        write_private_json(root/'completed_triplets.json', {'records': triples})
        write_private_json(root/'score_rows.json', {'records': rows})
        write_private_json(root/'execution_summary.json', {'case_ledgers': books})
        write_private_text(root/'score_table.csv', score_csv(triples))
        if sum(b['charged_model_attempts'] for b in books) > 24:
            raise ValueError('whole_collection_budget_exceeded')
        # A subsequent approved endpoint job can consume this request. No
        # secondary inference hidden inside the 24-attempt collection budget.
        request = {'schema_version': REQUEST_SCHEMA, 'pairs': [{k:r[k] for k in PAIR_FIELDS} for r in rows],
            'selection_sha256_sealed_before_endpoint': seal, 'modality_source': 'fully_synthetic',
            'selection_used_biovil': False, 'clinical_truth_available': False,
            'routing_or_calibration_update_allowed': False,
            'text_policy': 'full_report_no_silent_truncation_overlength_is_na'}
        private_directory(root/'endpoint_request')
        req = write_private_json(root/'endpoint_request/request.json', request)
        write_private_json(root/'endpoint_request/manifest.json', {'schema_version': REQUEST_SCHEMA,
            'artifacts': {'request.json': {'sha256': sha256_file(req)}}})
        summary = {'schema_version': VERSION, 'status': 'paired_collection_complete_unvalidated',
            'fixed_ehr_cases': 2, 'completed_triplets': len(triples),
            'charged_model_attempts': sum(b['charged_model_attempts'] for b in books),
            'failed_attempts': sum(b['failed_attempts'] for b in books),
            'actual_frozen_generation_executed': True, 'online_adaptive_repair_executed': False,
            'secondary_endpoint_executed': False, 'clinical_repair_success': None,
            'clinical_acceptance': False, 'selection_sha256_before_secondary': seal,
            'runtime_seconds': time.monotonic()-started, 'old_ehr_prompts_and_winners_changed': False}
        write_private_json(root/'summary.json', summary)
        check_pins(plan['source_pins']); check_pins(plan['artifact_pins'])
        if sha256_file(sealed) != seal: raise ValueError('sealed_choices_changed')
        files = ('choices_and_action_credits.json', 'completed_triplets.json', 'score_rows.json',
            'execution_summary.json', 'score_table.csv', 'summary.json')
        write_private_json(root/'manifest.json', {**summary,
            'artifacts': {k:{'sha256':sha256_file(root/k)} for k in files}})
    except BaseException as exc:
        write_private_json(root/'failure_manifest.json', {'status':'failed_or_interrupted_retained',
            'error_type':type(exc).__name__, 'automatic_resume':False}); raise
    finally:
        normalize_owned_modes(root)
    return root, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--output-root', type=Path, default=BASE/'paired_probe_plans')
    prepare_parser.add_argument('--run-id', required=True)
    launch = sub.add_parser('run')
    for name in ('plan-run','plan-manifest-sha256','output-root','run-id'):
        launch.add_argument('--'+name, required=True)
    args = parser.parse_args(); os.umask(0o007)
    try:
        path, summary = prepare(args) if args.mode == 'prepare' else run(args)
        print(json.dumps({'status':summary['status'], 'manifest_sha256':sha256_file(path/'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status':'failed', 'error_type':type(exc).__name__})); return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
