#!/usr/bin/env python3
"""Prospective mechanical-guard integration with the frozen image-only verifier."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import image_validity_guard as guard
import evaluate_image_validity_guard as inputs
import verify_image_abstention_controls as controls
import verify_cached_image_findings as cached
from contracts import (PROTECTED_ROOT, WORKSPACE, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json,
    write_private_text)

SCHEMA = 'tricompose-prospective-guarded-image-only-v1'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
GUARD_RUN = BASE/'image_validity_guards/pngguard_scope2_12645021_001'
GUARD_SHA = '0c9654d2147b7268890527c5db5f85f0c455217f501f17484ddf23365983ccba'
SCRIPT = ROOT/'slurm/46_guarded_image_findings_flexible_gpu.sbatch'


def inventory(original, realized, manifest):
    """Bind all fixed logical slots without opening pixels or old predictions."""
    expected = controls.logical_slots(original['image_inputs'])
    if ([{k:s[k] for k in ('slot_id', 'cxr_candidate_id', 'cxr_sha256', 'arm')}
            for s in realized['slots']] != expected or len(expected) != 18):
        raise ValueError('same_original_six_image_eighteen_slot_inventory_required')
    sources = {r['cxr_candidate_id']:r for r in original['image_inputs']}
    uniform = {r['observation_id']:r for r in realized['saved_uniform_controls']}
    if len(sources) != 6 or len(uniform) != 4 or len(realized['saved_uniform_controls']) != 4:
        raise ValueError('six_original_four_existing_control_inventory_required')
    unique = {}
    for slot in realized['slots']:
        source = sources[slot['cxr_candidate_id']]
        if slot['arm'] == 'original':
            path = require_inside(source['path'], PROTECTED_ROOT, must_exist=True)
            expected_sha = source['cxr_sha256']
            stats = source['image_file_stats']
        else:
            entry = uniform[slot['observation_id']]
            path = require_inside(inputs.CONTROL_RUN/entry['path'], inputs.CONTROL_RUN, must_exist=True)
            expected_sha = entry['sha256']
            if (manifest['artifacts'][entry['path']]['sha256'] != expected_sha
                    or (entry['width'], entry['height']) != (slot['width'], slot['height'])):
                raise ValueError('sealed_lossless_control_binding_required')
            stats = [path.stat().st_size, path.stat().st_mtime_ns]
        if (not path.is_file() or not 1 <= path.stat().st_size <= guard.MAX_BYTES
                or [path.stat().st_size, path.stat().st_mtime_ns] != stats):
            raise ValueError('bounded_unchanged_source_image_metadata_required')
        item = {'observation_id':slot['observation_id'], 'path':str(path), 'sha256':expected_sha,
            'width':slot['width'], 'height':slot['height'], 'image_file_stats':stats}
        if unique.setdefault(item['observation_id'], item) != item:
            raise ValueError('exact_deduplicated_input_binding_required')
    if len(unique) != 10:
        raise ValueError('ten_unique_existing_inputs_required')
    return [unique[key] for key in sorted(unique)]


def prepare():
    if sha256_file(inputs.ORIGINAL_PLAN/'manifest.json') != inputs.ORIGINAL_PLAN_SHA:
        raise ValueError('fixed_original_image_plan_required')
    original, sources = cached.load_plan(inputs.ORIGINAL_PLAN)
    manifest, metadata, more = inputs.fixed_parent(inputs.CONTROL_RUN, inputs.CONTROL_SHA,
        ('realized_inputs.json',))
    sources.update(more)
    if manifest['new_model_calls'] != 10:
        raise ValueError('same_completed_ten_call_control_baseline_required')
    baseline = require_inside(inputs.CONTROL_RUN/'predictions.json', PROTECTED_ROOT, must_exist=True)
    if (baseline.stat().st_size > 512*1024
            or sha256_file(baseline) != manifest['artifacts']['predictions.json']['sha256']):
        raise ValueError('hash_bound_quote_free_baseline_required')
    gm, _, more = inputs.fixed_parent(GUARD_RUN, GUARD_SHA, ())
    if (gm['new_model_calls'] != 0 or sha256_file(Path(guard.__file__)) != gm['source_sha256']['guard']
            or sha256_file(Path(inputs.__file__)) != gm['source_sha256']['worker']):
        raise ValueError('unchanged_separately_frozen_guard_required')
    sources.update(more)
    sources.update(guarded_worker=Path(__file__), guarded_tests=ROOT/'tests/test_guarded_image_findings.py',
        guarded_protocol=ROOT.parent/'docs/guarded_image_findings_protocol.md',
        guard_core=Path(guard.__file__), guard_evaluator=Path(inputs.__file__),
        controls_worker=Path(controls.__file__), baseline_prediction_bytes=baseline)
    realized = metadata['realized_inputs.json']
    plan = {k:original[k] for k in ('model_path', 'model_file_stats', 'model_file_sha256',
        'finding_order', 'prompt_version', 'image_prompt_sha256', 'min_pixels', 'max_pixels',
        'max_new_tokens', 'model_retries', 'seed', 'do_sample', 'min_vram_gib')}
    plan.update(schema_version=SCHEMA+'-plan', inputs=inventory(original, realized, manifest),
        logical_slots=realized['slots'], max_model_calls=10, fixed_ehr_cases=2,
        decoder_source_paths=gm['decoder_source_paths'], decoder_source_sha256=gm['decoder_source_sha256'],
        metadata_only=True, image_bytes_or_pixels_opened_in_prepare=False,
        model_weights_opened_in_prepare=False, old_prediction_states_parsed_in_prepare=False,
        new_model_calls=0, primary_metric_eligible=False, selection_changed=False,
        regeneration_authorized=False, clinical_truth_available=False,
        development_integration_check_not_heldout=True, verifier_controls_never_used_for_generation=True)
    return plan, sources


def load_plan(root):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    mp, pp = root/'manifest.json', root/'plan.json'
    manifest, plan = cached.bounded_json(mp), cached.bounded_json(pp)
    if (manifest['schema_version'] != SCHEMA+'-plan'
            or sha256_file(pp) != manifest['artifacts']['plan.json']['sha256']):
        raise ValueError('sealed_guarded_integration_plan_required')
    sources = {k:require_inside(v, WORKSPACE, must_exist=True) for k,v in manifest['source_paths'].items()}
    if set(sources) != set(manifest['source_sha256']) or any(
            sha256_file(p) != manifest['source_sha256'][k] for k,p in sources.items()):
        raise ValueError('sealed_program_and_metadata_required')
    rebuilt, dependencies = prepare()
    if plan != rebuilt or sources != {k:p.resolve() for k,p in dependencies.items()}:
        raise ValueError('exact_fixed_guarded_protocol_required')
    sources.update(guarded_plan_manifest=mp, guarded_plan=pp)
    return plan, sources


def invoke_inputs(plan, callback, *, Image):
    """Execute the real guard hook; model callback gets only checked pixels."""
    items = plan['inputs']
    keys = [r['observation_id'] for r in items]
    if keys != sorted(set(keys)) or type(plan['max_model_calls']) is not int or plan['max_model_calls'] < 0:
        raise ValueError('unique_hash_order_and_nonnegative_budget_required')
    records, calls = [], 0
    for item in items:
        def verified_callback(image):
            nonlocal calls
            if (guard.normalized_pixel_sha(image) != item['observation_id']
                    or (image.width, image.height) != (item['width'], item['height'])):
                raise ValueError('exact_checked_input_required_before_inference')
            if calls >= plan['max_model_calls']:
                raise ValueError('fixed_call_budget_exceeded')
            calls += 1
            return callback(image)
        outcome = guard.guarded_invoke(item['path'], item['sha256'], verified_callback, Image=Image)
        current = outcome['guard']
        if (current['artifact_sha256'] != item['sha256']
                or current['normalized_pixel_sha256'] not in (None, item['observation_id'])):
            raise ValueError('same_hash_bound_guard_input_required')
        if outcome['callback_invoked']:
            raw = outcome['callback_result']
            if (not isinstance(raw, dict) or raw.get('independent_clinical_validation') is not False
                    or set(raw) != {'contract_status', 'states', 'failure_reason', 'input_tokens',
                        'output_tokens', 'token_limit_reached', 'response_sha256', 'elapsed_seconds',
                        'independent_clinical_validation'}):
                raise ValueError('sanitized_unqualified_callback_record_required')
            guard.cached_view(current, {**raw, 'observation_id':item['observation_id']})
        else:
            raw = {'contract_status':'blocked_before_model_call', 'states':None,
                'failure_reason':current['reason'], 'input_tokens':None, 'output_tokens':None,
                'token_limit_reached':None, 'response_sha256':None, 'elapsed_seconds':None,
                'independent_clinical_validation':False}
        records.append({'observation_id':item['observation_id'], 'artifact_sha256':item['sha256'],
            'guard':current, 'model_called':outcome['callback_invoked'], **raw})
    if calls != sum(r['model_called'] for r in records):
        raise ValueError('exact_actual_call_ledger_required')
    return records, calls


def analyze(records, plan, baseline, calls):
    fresh = {r['observation_id']:r for r in records}
    old = {r['observation_id']:r for r in baseline['records']}
    expected = {r['observation_id'] for r in plan['inputs']}
    if (len(fresh) != len(records) or set(fresh) != expected or set(old) != expected
            or len(old) != len(baseline['records']) or baseline['frozen'] is not True
            or baseline['image_only'] is not True or baseline['image_prompt_sha256'] != plan['image_prompt_sha256']
            or baseline['model_received_arm_names_ehr_reports_ids_scores_or_expected_answers'] is not False
            or calls != sum(r['model_called'] for r in records)):
        raise ValueError('exact_original_and_guarded_input_contract_required')
    for previous in old.values():
        if previous['contract_status'] == 'complete':
            if (not isinstance(previous['states'],dict) or set(previous['states']) != set(guard.HEADS)
                    or not set(previous['states'].values()) <= guard.STATES
                    or previous['token_limit_reached'] is not False):
                raise ValueError('complete_named_baseline_states_required')
        elif previous['contract_status'] != 'failed_unavailable' or previous['states'] is not None:
            raise ValueError('unavailable_baseline_states_must_be_null')
    comparisons, matches, rows = 0, 0, []
    original_keys = {s['observation_id'] for s in plan['logical_slots'] if s['arm'] == 'original'}
    for key in sorted(original_keys):
        new, previous = fresh[key], old[key]
        complete = new['contract_status'] == previous['contract_status'] == 'complete'
        equal = sum(new['states'][h] == previous['states'][h] for h in guard.HEADS) if complete else None
        comparisons += 8 if complete else 0
        matches += equal or 0
        rows.append({'observation_id':key, 'matching_states':equal,
            'comparable_finding_slots':8 if complete else 0, 'clinical_accuracy':None})
    summary = {'schema_version':SCHEMA, 'unique_inputs':len(records), 'logical_slots':len(plan['logical_slots']),
        'actual_model_calls':calls, 'model_retries':0, 'unguarded_same_input_baseline_calls':len(old),
        'same_input_call_count_difference':len(old)-calls,
        'guard_counts':dict(sorted(Counter(r['guard']['status'] for r in records).items())),
        'blocked_before_model_call':sum(not r['model_called'] for r in records),
        'complete_model_responses':sum(r['contract_status']=='complete' for r in records),
        'failed_model_responses':sum(r['contract_status']=='failed_unavailable' for r in records),
        'original_readout_comparable_finding_slots':comparisons, 'original_readout_matching_states':matches,
        'independent_clinical_accuracy':None, 'clinical_acceptance':False, 'primary_metric_eligible':False,
        'regeneration_authorized':False, 'selection_changed':False, 'fixed_ehr_changed':False,
        'clinical_requests_resolved':0, 'old_prediction_states_parsed_after_prediction_fsync':True,
        'model_received_ehr_reports_ids_scores_arm_names_or_expected_answers':False,
        'verifier_controls_never_used_for_generation':True,
        'mechanical_control_specific_call_difference_not_clinical_efficiency':True,
        'clinical_compute_savings':None, 'gpu_runtime_speedup_established':False,
        'development_integration_check_not_heldout':True}
    return summary, rows


def freeze_then_compare(temporary, records, plan, calls):
    pp = write_private_json(temporary/'predictions.json', {'schema_version':SCHEMA, 'records':records,
        'frozen':True, 'image_only':True, 'image_prompt_sha256':plan['image_prompt_sha256']})
    with pp.open('rb') as handle:
        os.fsync(handle.fileno())
    pp_sha = sha256_file(pp)
    baseline = cached.bounded_json(inputs.CONTROL_RUN/'predictions.json', 512*1024)
    summary, comparisons = analyze(records, plan, baseline, calls)
    return summary, comparisons, pp, pp_sha


def run(plan_root, temporary, *, approved=False):
    cached.require_slurm(gpu=True, approved=approved)
    plan, sources = load_plan(plan_root)
    sources['approved_batch_script'] = SCRIPT
    for index,item in enumerate(plan['inputs']):
        path = require_inside(item['path'], PROTECTED_ROOT, must_exist=True)
        if ([path.stat().st_size, path.stat().st_mtime_ns] != item['image_file_stats']
                or sha256_file(path) != item['sha256']):
            raise ValueError('unchanged_encoded_input_required')
        sources['guarded_input_'+str(index)] = path
    before = {k:sha256_file(p) for k,p in sources.items()}
    model_path = Path(plan['model_path'])
    if {n:sha256_file(model_path/n) for n in cached.frozen.MODEL_FILES} != plan['model_file_sha256']:
        raise ValueError('unchanged_frozen_checkpoint_required')
    import torch
    import PIL
    from PIL import Image, ImageFile, PngImagePlugin, _imaging
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24*1024**3:
        raise RuntimeError('supported_gpu_with_at_least_24_gib_required')
    decoder = {k:Path(v.__file__).resolve() for k,v in
        (('Image',Image), ('ImageFile',ImageFile), ('PngImagePlugin',PngImagePlugin), ('_imaging',_imaging))}
    if (ImageFile.LOAD_TRUNCATED_IMAGES is not False
            or {k:str(p) for k,p in decoder.items()} != plan['decoder_source_paths']
            or {k:sha256_file(p) for k,p in decoder.items()} != plan['decoder_source_sha256']):
        raise ValueError('same_frozen_strict_png_decoder_required')
    torch.manual_seed(plan['seed'])
    torch.cuda.reset_peak_memory_stats()
    started, loaded = time.monotonic(), {}
    def infer_checked(image):
        if not loaded:
            model, processor, kind = _load_model(model_path, torch,
                min_pixels=plan['min_pixels'], max_pixels=plan['max_pixels'])
            model.eval().requires_grad_(False)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError('frozen_model_required')
            loaded.update(model=model, processor=processor, kind=kind)
        messages = cached.image_interface.request_messages('image', image=image)
        torch.cuda.synchronize()
        call_started = time.monotonic()
        response, tokens = cached.image_interface.infer(messages, loaded['model'], loaded['processor'],
            torch, max_new_tokens=plan['max_new_tokens'])
        torch.cuda.synchronize()
        return {**tokens, 'response_sha256':cached.image_interface.digest_text(response),
            **cached.sanitized_decode(response, token_limit_reached=tokens['token_limit_reached']),
            'elapsed_seconds':round(time.monotonic()-call_started, 4)}
    with open(os.devnull,'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        records, calls = invoke_inputs(plan, infer_checked, Image=Image)
    summary, comparisons, pp, pp_sha = freeze_then_compare(temporary, records, plan, calls)
    summary.update(runtime_seconds_excluding_preflight=round(time.monotonic()-started,4),
        peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated()/1024**3,3),
        gpu_name=torch.cuda.get_device_name(0), model_loaded=bool(loaded),
        model_type=loaded.get('kind'), decoder_version=PIL.__version__,
        token_cap_failures=sum(r['token_limit_reached'] is True for r in records),
        raw_patient_inputs_opened=False, ehr_report_or_real_target_bodies_opened=False)
    write_private_json(temporary/'logical_slots.json', {'slots':plan['logical_slots']})
    write_private_json(temporary/'original_repeatability.json', {'records':comparisons})
    write_private_json(temporary/'summary.json', summary)
    write_private_text(temporary/'RESULTS_CN_EN.md',
        '# Prospective guard integration / 调用前护栏验证\n\n'
        'Blocked inputs have null states and no model call; basic pass is not clinical acceptance.\n\n'
        '同一输入的调用次数差仅适用于机械对照，不证明临床效率或 GPU 时间提升。\n\n'
        '原始 EHR、候选分数、择优和历史记录不变；无独立临床真值或修复授权。\n\n'
        '```json\n'+json.dumps(summary,sort_keys=True,indent=2)+'\n```\n')
    if (sha256_file(pp) != pp_sha or any(sha256_file(p) != before[k] for k,p in sources.items())
            or any(sha256_file(p) != plan['decoder_source_sha256'][k] for k,p in decoder.items())
            or any([(model_path/n).stat().st_size,(model_path/n).stat().st_mtime_ns] != v
                for n,v in plan['model_file_stats'].items())):
        raise ValueError('immutable_input_program_decoder_or_predictions_changed')
    return summary, sources, before, plan


def execute(args):
    cached.require_slurm(gpu=args.mode=='run', approved=args.allow_guarded_image_verification)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        if args.mode=='prepare':
            plan, sources = prepare()
            before = {k:sha256_file(p) for k,p in sources.items()}
            write_private_json(temporary/'plan.json',plan)
            summary, schema = {'actual_model_calls':0}, SCHEMA+'-plan'
        else:
            summary, sources, before, plan = run(args.plan_run,temporary,
                approved=args.allow_guarded_image_verification)
            schema=SCHEMA
        if any(sha256_file(p) != before[k] for k,p in sources.items()):
            raise ValueError('consumed_sources_changed')
        write_private_json(temporary/'manifest.json', {'schema_version':schema,'run_id':args.run_id,
            'source_paths':{k:str(p.resolve()) for k,p in sources.items()},'source_sha256':before,
            'decoder_source_paths':plan['decoder_source_paths'],'decoder_source_sha256':plan['decoder_source_sha256'],
            'artifacts':{p.name:{'sha256':sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls':summary['actual_model_calls'],'primary_metric_eligible':False,
            'selection_changed':False,'regeneration_authorized':False,
            'image_pixels_opened':args.mode=='run','ehr_or_report_bodies_opened':False})
        for p in (temporary,*temporary.iterdir()):
            st=p.stat()
            if st.st_gid not in (96293,65534) or st.st_mode & 0o7777 != (0o2770 if p.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary,target)
        return target,summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--plan-run',type=Path)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--allow-guarded-image-verification',action='store_true')
    args=parser.parse_args()
    if args.mode=='run' and args.plan_run is None:
        parser.error('run requires --plan-run')
    try:
        target,summary=execute(args)
    except Exception as error:
        print(json.dumps({'status':'failed_closed','error_type':type(error).__name__}))
        return 2
    print(json.dumps({'status':'guarded_image_'+args.mode+'_completed',
        'actual_model_calls':summary['actual_model_calls'],'manifest_sha256':sha256_file(target/'manifest.json')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
