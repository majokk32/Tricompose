#!/usr/bin/env python3
"""Frozen image-only verifier repeatability and no-information safety controls."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import verify_cached_image_findings as cached
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text,
    private_directory)

SCHEMA = 'tricompose-image-no-information-control-v1'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
SOURCE_PLAN = BASE/'image_only_plans/image_only_scope2_12645021_001'
SOURCE_PLAN_SHA = '5fbcd984de88e7f776f3ef52081a98d978d620848f15feb63cfc19b4bedbcdfe'
BASELINE = BASE/'verification_runs/image_only_scope2_12649136'
BASELINE_SHA = '07aef0cf22a59ecf6441a07601afca3989c95dd5a5b608be5c4ee05a2bce17e1'
ARMS = ('original', 'uniform_black', 'uniform_white')


def logical_slots(images):
    slots = []
    for image in images:
        for arm in ARMS:
            slots.append({'slot_id': cached.image_interface.digest_text(image['cxr_candidate_id']+'|'+arm),
                'cxr_candidate_id': image['cxr_candidate_id'], 'cxr_sha256': image['cxr_sha256'],
                'arm': arm})
    if len({s['slot_id'] for s in slots}) != len(slots):
        raise ValueError('unique_fixed_logical_slots_required')
    return slots


def prepare():
    if sha256_file(SOURCE_PLAN/'manifest.json') != SOURCE_PLAN_SHA:
        raise ValueError('fixed_completed_image_plan_required')
    original, sources = cached.load_plan(SOURCE_PLAN)
    mp, pp = BASELINE/'manifest.json', BASELINE/'predictions.json'
    if sha256_file(mp) != BASELINE_SHA:
        raise ValueError('fixed_completed_baseline_required')
    manifest = cached.bounded_json(mp)
    if (manifest['new_model_calls'] != 6
            or any(manifest[k] is not False for k in ('primary_metric_eligible', 'selection_changed', 'regeneration_authorized'))
            or not pp.is_file() or pp.stat().st_size > 512*1024
            or sha256_file(pp) != manifest['artifacts']['predictions.json']['sha256']):
        raise ValueError('sealed_quote_free_baseline_required')
    # Hash baseline bytes here, but do not parse states/select cases from them.
    sources.update(control_worker=Path(__file__), control_tests=ROOT/'tests/test_image_abstention_controls.py',
        control_protocol=ROOT.parent/'docs/image_abstention_control_protocol.md',
        control_baseline_manifest=mp, control_baseline_predictions=pp)
    plan = {k: original[k] for k in ('image_inputs', 'image_slots', 'fixed_ehr_cases', 'model_path',
        'model_file_sha256', 'model_file_stats', 'finding_order', 'prompt_version', 'image_prompt_sha256',
        'min_pixels', 'max_pixels', 'max_new_tokens', 'model_retries', 'seed', 'do_sample', 'min_vram_gib')}
    plan.update(schema_version=SCHEMA+'-plan', logical_slots=logical_slots(original['image_inputs']),
        logical_slot_count=18, max_model_calls=18, input_deduplication='exact_normalized_RGB_dimensions_and_pixels',
        call_order='normalized_pixel_sha256_ascending', cached_prediction_states_parsed_in_prepare=False,
        metadata_only=True, image_bytes_or_pixels_opened_in_prepare=False, model_weights_opened_in_prepare=False,
        new_model_calls=0, primary_metric_eligible=False, selection_changed=False, regeneration_authorized=False,
        clinical_truth_available=False, development_diagnostic_not_heldout=True,
        verifier_controls_never_used_for_generation=True)
    if len(plan['logical_slots']) != 18 or len(plan['image_inputs']) != 6:
        raise ValueError('same_six_image_eighteen_slot_scope_required')
    return plan, sources


def load_plan(root):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    mp, pp = root/'manifest.json', root/'plan.json'
    manifest, plan = cached.bounded_json(mp), cached.bounded_json(pp)
    if manifest['schema_version'] != SCHEMA+'-plan' or sha256_file(pp) != manifest['artifacts']['plan.json']['sha256']:
        raise ValueError('sealed_control_plan_required')
    sources = {k: require_inside(v, cached.WORKSPACE, must_exist=True) for k, v in manifest['source_paths'].items()}
    if any(sha256_file(p) != manifest['source_sha256'][k] for k, p in sources.items()):
        raise ValueError('sealed_sources_changed')
    rebuilt, rebuilt_sources = prepare()
    if plan != rebuilt or sources != {k: p.resolve() for k, p in rebuilt_sources.items()}:
        raise ValueError('exact_frozen_control_protocol_required')
    sources.update(control_plan_manifest=mp, control_plan=pp)
    return plan, sources


def pixel_sha(image):
    width, height = image.size
    if image.mode != 'RGB' or not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise ValueError('bounded_normalized_RGB_required')
    pixels = image.tobytes()
    if len(pixels) != width*height*3:
        raise ValueError('complete_RGB_pixel_buffer_required')
    h = hashlib.sha256(json.dumps(['RGB', width, height], separators=(',', ':')).encode()+b'\0')
    h.update(pixels)
    return h.hexdigest()


def deduplicate_frames(slots, frames):
    if len(slots) != len(frames):
        raise ValueError('one_realized_frame_per_slot_required')
    unique, realized = {}, []
    for slot, frame in zip(slots, frames):
        key = pixel_sha(frame)
        unique.setdefault(key, frame)
        realized.append({**slot, 'observation_id': key, 'width': frame.width, 'height': frame.height})
    return unique, realized


def build_frames(plan, Image):
    frames, sources = [], {}
    for image in plan['image_inputs']:
        path = require_inside(image['path'], PROTECTED_ROOT, must_exist=True)
        if (not path.is_file() or [path.stat().st_size, path.stat().st_mtime_ns] != image['image_file_stats']
                or sha256_file(path) != image['cxr_sha256']):
            raise ValueError('unchanged_synthetic_source_image_required')
        sources['control_source_image_'+str(len(sources))] = path
        with Image.open(path) as handle:
            if not (64 <= handle.width <= 4096 and 64 <= handle.height <= 4096):
                raise ValueError('bounded_synthetic_dimensions_required')
            original = handle.convert('RGB').copy()
        frames += [original, Image.new('RGB', original.size, (0, 0, 0)),
            Image.new('RGB', original.size, (255, 255, 255))]
    if plan['logical_slots'] != logical_slots(plan['image_inputs']):
        raise ValueError('unchanged_arm_order_required')
    unique, slots = deduplicate_frames(plan['logical_slots'], frames)
    return unique, slots, sources


def save_controls(temporary, unique, slots, Image):
    private_directory(temporary/'control_inputs')
    keys = {s['observation_id'] for s in slots if s['arm'] != 'original'}
    metadata = []
    for index, key in enumerate(sorted(keys)):
        path = temporary/'control_inputs'/f'frame_{index:03d}.png'
        buffer = io.BytesIO()
        unique[key].save(buffer, format='PNG')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o660)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o660)
        with Image.open(path) as handle:
            if pixel_sha(handle.convert('RGB')) != key:
                raise ValueError('lossless_uniform_control_required')
        metadata.append({'path': str(path.relative_to(temporary)), 'sha256': sha256_file(path),
            'observation_id': key, 'width': unique[key].width, 'height': unique[key].height})
    return metadata


def rate(n, d):
    return None if d == 0 else n/d


def analyze(records, slots, baseline):
    """Mechanical safety/readout stability only; no clinical gold or winner."""
    index = {r['observation_id']: r for r in records}
    expected = {s['observation_id'] for s in slots}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError('all_unique_input_predictions_required')
    for rec in records:
        if rec['contract_status'] == 'complete':
            if not isinstance(rec['states'], dict) or set(rec['states']) != set(cached.image_interface.FINDINGS) \
                    or not set(rec['states'].values()) <= cached.STATES or rec['token_limit_reached'] is not False:
                raise ValueError('complete_unchanged_eight_state_contract_required')
        elif rec['contract_status'] != 'failed_unavailable' or rec['states'] is not None:
            raise ValueError('failed_contract_requires_null_not_unknown')
        if rec['independent_clinical_validation'] is not False:
            raise ValueError('no_clinical_truth_promotion')
    arm_stats = {}
    for arm in ARMS:
        own = [s for s in slots if s['arm'] == arm]
        keys = {s['observation_id'] for s in own}
        complete = [index[k] for k in keys if index[k]['contract_status'] == 'complete']
        counts = Counter(state for r in complete for state in r['states'].values())
        total, observed = len(keys)*8, len(complete)*8
        explicit = counts['positive']+counts['negative']
        arm_stats[arm] = {'logical_slots': len(own), 'unique_inputs': len(keys),
            'complete_unique_responses': len(complete), 'unavailable_unique_responses': len(keys)-len(complete),
            'all_finding_slots': total, 'observed_finding_slots': observed,
            'state_counts': {s: counts[s] for s in sorted(cached.STATES)},
            'explicit_assertion_fraction_on_complete': rate(explicit, observed),
            'explicit_assertion_all_slot_lower_bound': rate(explicit, total),
            'explicit_assertion_all_slot_upper_bound': rate(explicit+total-observed, total),
            'unknown_fraction_on_complete': rate(counts['unknown'], observed),
            'interpretable_as_clinical_accuracy': False}
    if (baseline.get('schema_version') != cached.SCHEMA or baseline.get('frozen') is not True
            or baseline.get('image_only') is not True
            or baseline.get('model_received_ehr_reports_scores_or_candidate_ids') is not False):
        raise ValueError('same_blind_frozen_baseline_required')
    previous = {r['cxr_candidate_id']: r for r in baseline['records']}
    originals = [s for s in slots if s['arm'] == 'original']
    if len(previous) != len(baseline['records']) or set(previous) != {s['cxr_candidate_id'] for s in originals}:
        raise ValueError('exact_original_baseline_inventory_required')
    matches = comparisons = unavailable = 0
    original_rows = []
    for slot in originals:
        old, fresh = previous[slot['cxr_candidate_id']], index[slot['observation_id']]
        if old['cxr_sha256'] != slot['cxr_sha256']:
            raise ValueError('exact_original_image_hash_required')
        complete = old['contract_status'] == fresh['contract_status'] == 'complete'
        if old['contract_status'] == 'complete' and (not isinstance(old['states'], dict)
                or set(old['states']) != set(cached.image_interface.FINDINGS)
                or not set(old['states'].values()) <= cached.STATES):
            raise ValueError('complete_cached_original_states_required')
        if old['contract_status'] != 'complete' and (old['contract_status'] != 'failed_unavailable'
                or old['states'] is not None):
            raise ValueError('unavailable_cached_original_requires_null')
        matched = sum(old['states'][f] == fresh['states'][f] for f in cached.image_interface.FINDINGS) if complete else None
        if complete:
            matches += matched
            comparisons += 8
        else:
            unavailable += 8
        original_rows.append({**slot, 'matching_states': matched,
            'comparable_finding_slots': 8 if complete else 0, 'clinical_accuracy': None})
    summary = {'schema_version': SCHEMA, 'logical_slots': len(slots), 'actual_model_calls': len(records),
        'model_retries': 0, 'arms': arm_stats, 'original_readout_matching_states': matches,
        'original_readout_comparable_finding_slots': comparisons,
        'original_readout_unavailable_finding_slots': unavailable,
        'original_readout_match_fraction': rate(matches, comparisons),
        'independent_clinical_accuracy': None, 'primary_metric_eligible': False,
        'selection_changed': False, 'regeneration_authorized': False, 'clinically_resolved_requests': 0,
        'same_geometry_controls_are_not_independent_patients': True,
        'model_received_arm_names_ehr_reports_ids_scores_or_expected_answers': False,
        'verifier_controls_never_used_for_generation': True, 'development_diagnostic_not_heldout': True}
    return summary, original_rows


def run(plan_root, temporary, *, approved=False):
    cached.require_slurm(gpu=True, approved=approved)
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24*1024**3:
        raise RuntimeError('gpu_with_at_least_24_gib_required')
    plan, sources = load_plan(plan_root)
    before = {k: sha256_file(p) for k, p in sources.items()}
    model_path = Path(plan['model_path'])
    if {name: sha256_file(model_path/name) for name in cached.frozen.MODEL_FILES} != plan['model_file_sha256']:
        raise ValueError('frozen_model_bytes_changed')
    from PIL import Image
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    unique, slots, image_sources = build_frames(plan, Image)
    sources.update(image_sources)
    before.update({k: sha256_file(p) for k, p in image_sources.items()})
    controls = save_controls(temporary, unique, slots, Image)
    if len(unique) > plan['max_model_calls']:
        raise ValueError('predeclared_call_budget_exceeded')
    torch.manual_seed(plan['seed'])
    torch.cuda.reset_peak_memory_stats()
    started, records = time.monotonic(), []
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        model, processor, model_type = _load_model(model_path, torch,
            min_pixels=plan['min_pixels'], max_pixels=plan['max_pixels'])
        model.eval().requires_grad_(False)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('frozen_model_required')
        for key in sorted(unique):
            if len(records) >= plan['max_model_calls']:
                raise ValueError('predeclared_call_budget_exceeded')
            messages = cached.image_interface.request_messages('image', image=unique[key])
            torch.cuda.synchronize()
            call_started = time.monotonic()
            response, tokens = cached.image_interface.infer(messages, model, processor, torch,
                max_new_tokens=plan['max_new_tokens'])
            torch.cuda.synchronize()
            records.append({'observation_id': key, **tokens,
                'response_sha256': cached.image_interface.digest_text(response),
                **cached.sanitized_decode(response, token_limit_reached=tokens['token_limit_reached']),
                'elapsed_seconds': round(time.monotonic()-call_started, 4)})
    pp = write_private_json(temporary/'predictions.json', {'schema_version': SCHEMA, 'records': records,
        'image_only': True, 'frozen': True, 'image_prompt_sha256': plan['image_prompt_sha256'],
        'model_received_arm_names_ehr_reports_ids_scores_or_expected_answers': False})
    with pp.open('rb') as handle:
        os.fsync(handle.fileno())
    prediction_hash = sha256_file(pp)
    # First parse of the cached reference states happens after prediction fsync.
    baseline = cached.bounded_json(BASELINE/'predictions.json')
    if baseline['image_prompt_sha256'] != plan['image_prompt_sha256']:
        raise ValueError('unchanged_original_image_prompt_required')
    summary, original_rows = analyze(records, slots, baseline)
    summary.update(runtime_seconds=round(time.monotonic()-started, 4),
        peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
        gpu_name=torch.cuda.get_device_name(0), model_type=model_type,
        token_cap_failures=sum(r['token_limit_reached'] for r in records),
        cached_baseline_parsed_after_prediction_fsync=True,
        raw_patient_inputs_opened=False, ehr_report_or_real_target_bodies_opened=False)
    write_private_json(temporary/'realized_inputs.json', {'slots': slots, 'saved_uniform_controls': controls,
        'normalized_pixel_deduplication': True})
    write_private_json(temporary/'original_repeatability.json', {'records': original_rows})
    write_private_json(temporary/'summary.json', summary)
    write_private_text(temporary/'RESULTS_CN_EN.md',
        '# Uninformative-image safety / 无信息图像安全检查\n\n'
        'Uniform controls support neither disease presence nor absence; unknown is expected.\n\n'
        '原图不是临床真值；复跑一致仅表示稳定性。模型未获得 EHR、报告、干预名称或期望答案。\n\n'
        '机械对照不是临床错误定位或修复成功；现有评分、择优及 EHR 不变。\n\n'
        '```json\n'+json.dumps(summary, sort_keys=True, indent=2)+'\n```\n')
    if (any(sha256_file(p) != before[k] for k, p in sources.items()) or sha256_file(pp) != prediction_hash
            or any([(model_path/n).stat().st_size, (model_path/n).stat().st_mtime_ns] != v
                for n, v in plan['model_file_stats'].items())):
        raise ValueError('immutable_sources_or_frozen_predictions_changed')
    return summary, sources, before


def execute(args):
    cached.require_slurm(gpu=args.mode == 'run', approved=args.allow_image_controls)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        if args.mode == 'prepare':
            plan, sources = prepare()
            before = {k: sha256_file(p) for k, p in sources.items()}
            write_private_json(temporary/'plan.json', plan)
            summary, schema = {'actual_model_calls': 0}, SCHEMA+'-plan'
        else:
            summary, sources, before = run(args.plan_run, temporary, approved=args.allow_image_controls)
            schema = SCHEMA
        if any(sha256_file(p) != before[k] for k, p in sources.items()):
            raise ValueError('immutable_sources_changed')
        write_private_json(temporary/'manifest.json', {'schema_version': schema, 'run_id': args.run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {str(p.relative_to(temporary)): {'sha256': sha256_file(p)}
                for p in sorted(temporary.rglob('*')) if p.is_file()},
            'new_model_calls': summary['actual_model_calls'], 'primary_metric_eligible': False,
            'selection_changed': False, 'regeneration_authorized': False,
            'image_pixels_opened': args.mode == 'run', 'ehr_or_report_bodies_opened': False})
        for path in (temporary, *temporary.rglob('*')):
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
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--plan-run', type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-image-controls', action='store_true')
    args = parser.parse_args()
    if args.mode == 'run' and args.plan_run is None:
        parser.error('run requires --plan-run')
    try:
        target, summary = execute(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'image_control_'+args.mode+'_completed',
        'actual_model_calls': summary['actual_model_calls'], 'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
