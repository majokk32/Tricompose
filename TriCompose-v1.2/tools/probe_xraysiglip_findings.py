#!/usr/bin/env python3
"""Fixed, blinded XraySigLIP polarity probe; alternate evidence, not clinical truth."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import attach_report_gate_availability as tables
import image_validity_guard as guard
from contracts import (PROTECTED_ROOT, WORKSPACE, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
SCHEMA = 'tricompose-xraysiglip-fixed-alternate-polarity-probe-v1'
FRONTIER = BASE / 'verification_frontiers/frontier_pool960_12645021_validated'
FRONTIER_SHA = '204b499d2eeab6cd7f849ca854a8beefbc433a53876bb03b507a3aa56e0493bc'
INPUT_PLAN = BASE / 'guarded_image_plans/guardhook_scope2_12645021_001'
INPUT_SHA = '2864941d29f60e890cdd51e88a7126986027b495345d4cf8a0ff23bdc2b77959'
MODEL = WORKSPACE / 'CheXagent-2/checkpoints/chexagent2-xraysiglip/f0edbf5d90dba44edb7f4f96d8663537cb0749bf'
ENVIRONMENT = WORKSPACE / 'runtime/venvs/chexagent2-srrg-cu121-v1'
SITE = ENVIRONMENT / 'lib/python3.11/site-packages'
SCRIPT = ROOT / 'slurm/47_xraysiglip_findings_flexible_gpu.sbatch'
MODEL_META = ('config.json', 'preprocessor_config.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'spiece.model', 'README.md')
FAMILIES = ('shows_no', 'evidence_no', 'present_absent')


def probes():
    catalog = []
    for head in guard.HEADS:
        name = head.replace('_', ' ')
        pairs = ((f'The chest X-ray shows {name}.', f'The chest X-ray shows no {name}.'),
            (f'There is evidence of {name}.', f'There is no evidence of {name}.'),
            (f'{name.capitalize()} is present.', f'{name.capitalize()} is absent.'))
        for family, (positive, negative) in zip(FAMILIES, pairs):
            catalog.append({'probe_id': head + '_' + family, 'finding': head, 'family': family,
                'positive_text': positive, 'negative_text': negative})
    return catalog


def private_modes(root):
    for path in (root, *root.iterdir()):
        st = path.stat()
        if st.st_gid not in (96293, 65534) or st.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
            raise ValueError('protected_project_modes_required')


def fixed_metadata(root, expected, name, *, jsonl=False):
    mp = require_inside(root / 'manifest.json', PROTECTED_ROOT, must_exist=True)
    if sha256_file(mp) != expected:
        raise ValueError('fixed_parent_manifest_required')
    manifest = json.loads(mp.read_text(encoding='utf-8'))
    if any(manifest[k] is not False for k in ('selection_changed', 'regeneration_authorized', 'primary_metric_eligible')):
        raise ValueError('unchanged_unqualified_parent_required')
    path = require_inside(root / name, root, must_exist=True)
    if path.stat().st_size > 2 * 1024 * 1024 or sha256_file(path) != manifest['artifacts'][name]['sha256']:
        raise ValueError('bounded_fixed_metadata_required')
    value = ([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        if jsonl else json.loads(path.read_text(encoding='utf-8')))
    if sha256_file(mp) != expected:
        raise ValueError('parent_changed_during_read')
    return value, {root.name + '_manifest': mp, root.name + '_' + name: path}


def validate_model_metadata(config, processor, tokenizer):
    if (config.get('architectures') != ['SiglipModel'] or config.get('model_type') != 'siglip'
            or config.get('vision_config', {}).get('image_size') != 512
            or config.get('text_config', {}).get('max_position_embeddings') != 256
            or config.get('torch_dtype') != 'float32'
            or processor.get('processor_class') != 'SiglipProcessor'
            or processor.get('size') != {'height': 512, 'width': 512}
            or processor.get('do_normalize') is not True or processor.get('do_rescale') is not True
            or processor.get('do_resize') is not True or processor.get('resample') != 3
            or processor.get('image_mean') != [0.5] * 3 or processor.get('image_std') != [0.5] * 3
            or processor.get('rescale_factor') != 1 / 255
            or tokenizer.get('model_max_length') != 64 or tokenizer.get('tokenizer_class') != 'SiglipTokenizer'):
        raise ValueError('native_complete_siglip_512_float32_processor_required')


def bind_requests(plan, requests):
    inputs = {r['observation_id']: r for r in plan['inputs']}
    if len(inputs) != len(plan['inputs']) or len({s['slot_id'] for s in plan['logical_slots']}) != len(plan['logical_slots']):
        raise ValueError('unique_fixed_inputs_and_slots_required')
    original = {}
    controls = set()
    for slot in plan['logical_slots']:
        item = inputs.get(slot['observation_id'])
        if item is None or (item['width'], item['height']) != (slot['width'], slot['height']):
            raise ValueError('exact_input_slot_dimensions_required')
        if slot['arm'] == 'original':
            if item['sha256'] != slot['cxr_sha256']:
                raise ValueError('exact_original_encoded_hash_required')
            identity = (slot['observation_id'], slot['cxr_sha256'])
            if original.setdefault(slot['cxr_candidate_id'], identity) != identity:
                raise ValueError('fixed_original_candidate_required')
        elif slot['arm'] in ('uniform_black', 'uniform_white'):
            controls.add(slot['observation_id'])
        else:
            raise ValueError('known_original_or_isolated_control_arm_required')
    if (controls & {v[0] for v in original.values()}
            or set(inputs) != controls | {v[0] for v in original.values()}):
        raise ValueError('controls_cannot_be_candidates')
    seen, keys, requested, refs = set(), set(), set(), []
    for request in requests:
        cid, head = request['cxr_candidate_id'], request['finding']
        if (request['request_id'] in seen or head not in guard.HEADS or cid not in original
                or request['request_kind'] != 'obtain_independent_image_finding_evidence'
                or request['execution_status'] != 'not_executed'
                or request['source_availability_manifest_sha256'] !=
                    '0120146adc043b482f51ef515d70ad3dd1b78132c3b989b22e7bd918b3c68dcb'
                or set(request['dependency_hashes']) != {'cxr_sha256'}
                or request['dependency_hashes']['cxr_sha256'] != original[cid][1]
                or any(request[k] is not False for k in ('model_execution_allowed',
                    'regeneration_authorized', 'clinical_truth_established', 'clinically_resolved', 'primary_metric_eligible'))):
            raise ValueError('exact_unresolved_original_request_required')
        key = (request['case_id'], cid, head)
        if key in keys or not request['consumer_candidate_ids'] or len(set(request['consumer_candidate_ids'])) != len(request['consumer_candidate_ids']):
            raise ValueError('unique_nonempty_supplemental_request_required')
        keys.add(key)
        seen.add(request['request_id'])
        requested.add(cid)
        refs.append({'request_id': request['request_id'], 'case_id': request['case_id'],
            'cxr_candidate_id': cid, 'cxr_sha256': original[cid][1], 'observation_id': original[cid][0],
            'finding': head, 'consumer_candidate_ids': list(request['consumer_candidate_ids'])})
    if requested != set(original):
        raise ValueError('all_and_only_fixed_original_images_required')
    return refs


def prepare():
    original, sources = fixed_metadata(INPUT_PLAN, INPUT_SHA, 'plan.json')
    requests, more = fixed_metadata(FRONTIER, FRONTIER_SHA, 'supplemental_image_evidence_requests.jsonl', jsonl=True)
    sources.update(more)
    refs = bind_requests(original, requests)
    if (len(original['inputs']), len(original['logical_slots']), len(refs),
            len({r['cxr_candidate_id'] for r in refs}), len({r['case_id'] for r in refs})) != (10, 18, 21, 6, 2):
        raise ValueError('fixed_six_image_twentyone_request_development_scope_required')
    sources.update(worker=Path(__file__), tests=ROOT / 'tests/test_xraysiglip_probe.py',
        protocol=ROOT.parent / 'docs/xraysiglip_probe_protocol.md', guard=Path(guard.__file__),
        table_helpers=Path(tables.__file__), contracts=Path(sys.modules['contracts'].__file__))
    for name in MODEL_META:
        path = require_inside(MODEL / name, WORKSPACE, must_exist=True)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('bounded_model_metadata_required')
        sources['model_' + name] = path
    payload = {name: json.loads((MODEL / name).read_text()) for name in
        ('config.json', 'preprocessor_config.json', 'tokenizer_config.json')}
    validate_model_metadata(*payload.values())
    weight = require_inside(MODEL / 'model.safetensors', WORKSPACE, must_exist=True)
    if not weight.is_file() or weight.stat().st_size != 2612631968:
        raise ValueError('existing_complete_siglip_weight_inventory_required')
    runtime_names = ['transformers/' + name for name in ('__init__.py', 'modeling_utils.py',
        'configuration_utils.py', 'processing_utils.py', 'tokenization_utils.py', 'tokenization_utils_base.py',
        'image_utils.py', 'image_transforms.py', 'feature_extraction_utils.py')]
    runtime_names += ['transformers/models/siglip/' + name + '.py' for name in
        ('configuration_siglip', 'image_processing_siglip', 'modeling_siglip', 'processing_siglip', 'tokenization_siglip')]
    runtime_names += ['PIL/' + name for name in ('Image.py', 'ImageFile.py', 'PngImagePlugin.py', '_imaging.cpython-311-x86_64-linux-gnu.so')]
    runtime_names += [name + '/METADATA' for name in ('transformers-4.41.2.dist-info',
        'torch-2.2.1+cu121.dist-info', 'numpy-1.26.4.dist-info', 'pillow-10.4.0.dist-info',
        'sentencepiece-0.2.0.dist-info', 'safetensors-0.4.3.dist-info')]
    for name in runtime_names:
        sources['runtime_' + name] = require_inside(SITE / name, WORKSPACE, must_exist=True)
    for name in ('reliability_preview', 'scorer_reliability', 'legacy_replay_adapter', 'invariant_verification', 'decision_preview'):
        sources[name] = ROOT / 'src/tricompose_v12' / (name + '.py')
    for item in original['inputs']:
        path = require_inside(item['path'], PROTECTED_ROOT, must_exist=True)
        if (not path.is_file() or [path.stat().st_size, path.stat().st_mtime_ns] != item['image_file_stats']
                or not 1 <= path.stat().st_size <= guard.MAX_BYTES):
            raise ValueError('unchanged_bounded_image_file_metadata_required')
    plan = {'schema_version': SCHEMA + '-plan', 'inputs': original['inputs'],
        'logical_slots': original['logical_slots'], 'request_refs': refs, 'probes': probes(),
        'model_path': str(MODEL), 'weight_path': str(weight),
        'weight_stats': [weight.stat().st_size, weight.stat().st_mtime_ns],
        'python_executable': str(ENVIRONMENT / 'bin/python'), 'max_scoring_attempts': 6,
        'model_retries': 0, 'precision': 'float32', 'image_size': [512, 512],
        'text_max_length': 64, 'min_vram_gib': 12, 'seed': 42,
        'metadata_only': True, 'image_bytes_or_pixels_opened_in_prepare': False,
        'model_weights_opened_in_prepare': False, 'new_model_calls': 0,
        'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
        'independent_clinical_validation': False, 'shares_vision_encoder_with_chexagent2': True,
        'training_overlap_unverified': True, 'distinct_from_qwen_and_xrv': True}
    return plan, sources


def load_plan(root):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    path = root / 'plan.json'
    if manifest['schema_version'] != SCHEMA + '-plan' or sha256_file(path) != manifest['artifacts']['plan.json']['sha256']:
        raise ValueError('sealed_siglip_plan_required')
    sources = {k: require_inside(v, WORKSPACE, must_exist=True) for k, v in manifest['source_paths'].items()}
    if set(sources) != set(manifest['source_sha256']) or any(sha256_file(p) != manifest['source_sha256'][k] for k, p in sources.items()):
        raise ValueError('sealed_program_and_metadata_required')
    plan = json.loads(path.read_text())
    rebuilt, expected_sources = prepare()
    if plan != rebuilt or sources != {k: p.resolve() for k, p in expected_sources.items()}:
        raise ValueError('exact_fixed_probe_plan_required')
    sources.update(plan_manifest=root / 'manifest.json', plan=path)
    return plan, sources


def finite_number(value, *, cosine=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (cosine and abs(value) > 1.00001):
        raise ValueError('finite_bounded_numeric_score_required')
    return float(value)


def decode_scores(cosines, logits):
    if len(cosines) != 48 or len(logits) != 48:
        raise ValueError('exact_fortyeight_pair_endpoint_scores_required')
    pairs = []
    for i, probe in enumerate(probes()):
        values = dict(positive_cosine=finite_number(cosines[2 * i], cosine=True),
            negative_cosine=finite_number(cosines[2 * i + 1], cosine=True),
            positive_logit=finite_number(logits[2 * i]), negative_logit=finite_number(logits[2 * i + 1]))
        pairs.append({k: probe[k] for k in ('probe_id', 'finding', 'family')} | values)
    return pairs


def summarize_pairs(pairs):
    if (len(pairs) != 24 or [r['probe_id'] for r in pairs] != [r['probe_id'] for r in probes()]
            or any(set(r) != {'probe_id', 'finding', 'family', 'positive_cosine', 'negative_cosine',
                'positive_logit', 'negative_logit'} for r in pairs)
            or any((r['finding'], r['family']) != (p['finding'], p['family']) for r, p in zip(pairs, probes()))):
        raise ValueError('complete_fixed_probe_inventory_required')
    result = {}
    for head in guard.HEADS:
        subset = [r for r in pairs if r['finding'] == head]
        margins = [finite_number(r['positive_cosine'], cosine=True) - finite_number(r['negative_cosine'], cosine=True) for r in subset]
        for row in subset:
            finite_number(row['positive_logit']); finite_number(row['negative_logit'])
        preference = ('present_prompt_higher' if min(margins) > 0 else
            'absent_prompt_higher' if max(margins) < 0 else
            'tied_templates' if all(m == 0 for m in margins) else 'template_sensitive')
        result[head] = {'cosine_margins': dict(zip(FAMILIES, margins)),
            'mean_cosine_margin': sum(margins) / 3, 'min_cosine_margin': min(margins),
            'max_cosine_margin': max(margins), 'text_preference': preference,
            'clinical_state': None, 'calibrated_probability': None}
    return result


def invoke(plan, callback, *, Image, reserve=lambda *_: None, complete=lambda *_: None):
    keys = [r['observation_id'] for r in plan['inputs']]
    if keys != sorted(set(keys)) or type(plan['max_scoring_attempts']) is not int or plan['max_scoring_attempts'] < 0:
        raise ValueError('sorted_unique_inputs_and_fixed_budget_required')
    records, attempts = [], 0
    for item in plan['inputs']:
        def checked(image):
            nonlocal attempts
            if guard.normalized_pixel_sha(image) != item['observation_id'] or (image.width, image.height) != (item['width'], item['height']):
                raise ValueError('exact_checked_pixels_required')
            if attempts >= plan['max_scoring_attempts']:
                raise ValueError('fixed_scoring_attempt_budget_exceeded')
            attempts += 1
            reserve(attempts, item['observation_id'])
            started = time.monotonic()
            raw = callback(image)
            if set(raw) != {'pairs', 'failure_reason', 'model_forward_attempted'} or type(raw['model_forward_attempted']) is not bool:
                raise ValueError('bounded_sanitized_callback_required')
            if raw['pairs'] is not None:
                summarize_pairs(raw['pairs'])
                if raw['failure_reason'] is not None or raw['model_forward_attempted'] is not True:
                    raise ValueError('complete_scores_require_actual_forward')
            elif not isinstance(raw['failure_reason'], str) or not raw['failure_reason'].isidentifier():
                raise ValueError('unavailable_score_reason_required')
            result = {**raw, 'elapsed_seconds': round(time.monotonic() - started, 6)}
            complete(attempts, item['observation_id'], result)
            return result
        outcome = guard.guarded_invoke(item['path'], item['sha256'], checked, Image=Image)
        if outcome['guard']['normalized_pixel_sha256'] not in (None, item['observation_id']):
            raise ValueError('fixed_normalized_pixels_required')
        raw = outcome['callback_result'] if outcome['callback_invoked'] else {
            'pairs': None, 'failure_reason': outcome['guard']['reason'],
            'model_forward_attempted': False, 'elapsed_seconds': None}
        records.append({'observation_id': item['observation_id'], 'artifact_sha256': item['sha256'],
            'guard': outcome['guard'], 'callback_invoked': outcome['callback_invoked'],
            'contract_status': 'blocked_before_callback' if not outcome['callback_invoked'] else
                'complete' if raw['pairs'] is not None else 'failed_unavailable',
            **raw, 'independent_clinical_validation': False})
    if attempts != sum(r['callback_invoked'] for r in records):
        raise ValueError('exact_scoring_attempt_accounting_required')
    return records, attempts


def posthoc(records, plan, requests):
    lookup = {r['observation_id']: r for r in records}
    refs = {r['request_id']: r for r in plan['request_refs']}
    if len(lookup) != len(records) or set(lookup) != {r['observation_id'] for r in plan['inputs']}:
        raise ValueError('exact_fixed_scored_input_inventory_required')
    if len(refs) != len(requests) or set(refs) != {r['request_id'] for r in requests}:
        raise ValueError('exact_posthoc_request_inventory_required')
    output = []
    for request in requests:
        ref = refs[request['request_id']]
        if (any(request[k] != ref[k] for k in ('case_id', 'cxr_candidate_id', 'finding', 'consumer_candidate_ids'))
                or request['dependency_hashes'] != {'cxr_sha256': ref['cxr_sha256']}):
            raise ValueError('unchanged_exact_posthoc_request_binding_required')
        record = lookup[ref['observation_id']]
        guard.validate_receipt(record['guard'])
        if (record['artifact_sha256'] != ref['cxr_sha256']
                or record['guard']['artifact_sha256'] != ref['cxr_sha256']
                or record['guard']['normalized_pixel_sha256'] != ref['observation_id']
                or record['independent_clinical_validation'] is not False
                or (record['pairs'] is not None and (record['guard']['basic_comparison_permitted'] is not True
                    or record['contract_status'] != 'complete' or record['callback_invoked'] is not True
                    or record['model_forward_attempted'] is not True))):
            raise ValueError('exact_unqualified_guarded_request_readout_required')
        value = None if record['pairs'] is None else summarize_pairs(record['pairs'])[ref['finding']]
        preference = None if value is None else value['text_preference']
        def compare(state):
            if state not in guard.STATES:
                raise ValueError('known_four_state_proxy_required')
            if state not in ('positive', 'negative') or preference not in ('present_prompt_higher', 'absent_prompt_higher'):
                return 'not_comparable'
            same = (state == 'positive') == (preference == 'present_prompt_higher')
            return 'same_direction_unqualified' if same else 'opposed_direction_unqualified'
        output.append({**ref, 'siglip_readout': value, 'vs_current_qwen': compare(request['current_readout']),
            'vs_raw_xrv': compare(request['raw_xrv_state']), 'clinical_request_resolved': False,
            'clinical_truth_available': False, 'confirmed_faulty_modality': None,
            'primary_metric_eligible': False, 'regeneration_authorized': False,
            'shares_encoder_with_chexagent2': True})
    return output


def seal(temporary, run_id, schema, sources, before, *, model_calls=0):
    if any(sha256_file(p) != before[k] for k, p in sources.items()):
        raise ValueError('consumed_source_changed_during_run')
    write_private_json(temporary / 'manifest.json', {'schema_version': schema, 'run_id': run_id,
        'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
        'artifacts': {p.name: {'sha256': sha256_file(p)} for p in sorted(temporary.iterdir())},
        'new_model_calls': model_calls, 'selection_changed': False, 'primary_metric_eligible': False,
        'regeneration_authorized': False, 'independent_clinical_validation': False,
        'ehr_report_or_real_target_bodies_opened': False})
    private_modes(temporary)


def execute_prepare(output_root, run_id):
    tables.require_cpu_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        plan, sources = prepare()
        before = {k: sha256_file(p) for k, p in sources.items()}
        write_private_json(temporary / 'plan.json', plan)
        seal(temporary, run_id, SCHEMA + '-plan', sources, before)
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary)
        raise


def require_gpu_approval(approved):
    tables.require_cpu_slurm()
    if approved is not True:
        raise RuntimeError('separately_approved_gpu_probe_required')


def execute_run(plan_root, output_root, run_id, *, approved=False):
    require_gpu_approval(approved)
    # Import/check CUDA before output creation or any input pixel/weight read.
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 12 * 1024 ** 3:
        raise RuntimeError('cuda_with_at_least_12_gib_required')
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        started = time.monotonic()
        plan, sources = load_plan(plan_root)
        if Path(sys.executable).absolute() != Path(plan['python_executable']).absolute():
            raise ValueError('pinned_existing_python_environment_required')
        sources['approved_batch_script'] = SCRIPT
        for i, item in enumerate(plan['inputs']):
            path = require_inside(item['path'], PROTECTED_ROOT, must_exist=True)
            if [path.stat().st_size, path.stat().st_mtime_ns] != item['image_file_stats'] or sha256_file(path) != item['sha256']:
                raise ValueError('unchanged_hash_bound_image_required')
            sources['image_' + str(i)] = path
        weight = Path(plan['weight_path'])
        if [weight.stat().st_size, weight.stat().st_mtime_ns] != plan['weight_stats']:
            raise ValueError('unchanged_weight_inventory_required')
        sources['complete_existing_weight'] = weight
        before = {k: sha256_file(p) for k, p in sources.items()}
        from PIL import Image, ImageFile
        import transformers.models.siglip.modeling_siglip as implementation
        from transformers import SiglipModel, SiglipProcessor
        if (ImageFile.LOAD_TRUNCATED_IMAGES is not False or Path(implementation.__file__).resolve() !=
                sources['runtime_transformers/models/siglip/modeling_siglip.py']):
            raise ValueError('frozen_native_siglip_and_strict_png_decoder_required')
        torch.manual_seed(plan['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        texts = [p[k] for p in probes() for k in ('positive_text', 'negative_text')]
        loaded, load_attempts = {}, 0
        journal = write_private_text(temporary / 'attempt_journal.jsonl', '')
        def append(event):
            with journal.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, sort_keys=True) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        def reserve(index, observation):
            append({'event': 'reserved', 'scoring_attempt': index, 'observation_id': observation})
        def complete(index, observation, raw):
            append({'event': 'completed' if raw['pairs'] is not None else 'failed_unavailable',
                'scoring_attempt': index, 'observation_id': observation,
                'model_forward_attempted': raw['model_forward_attempted'],
                'failure_reason': raw['failure_reason'], 'elapsed_seconds': raw['elapsed_seconds']})
        def infer(image):
            nonlocal load_attempts
            attempted = False
            try:
                if loaded.get('failed'):
                    return {'pairs': None, 'failure_reason': 'model_unavailable_after_load_failure', 'model_forward_attempted': False}
                if not loaded:
                    load_attempts += 1
                    try:
                        model, info = SiglipModel.from_pretrained(plan['model_path'], local_files_only=True,
                            use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
                        if any(info[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                            raise ValueError('complete_existing_weights_required')
                        model.to('cuda').eval().requires_grad_(False)
                        processor = SiglipProcessor.from_pretrained(plan['model_path'], local_files_only=True)
                        unpadded = processor.tokenizer(texts, padding=False, truncation=False)
                        if len(unpadded['input_ids']) != 48 or max(map(len, unpadded['input_ids'])) > 64:
                            raise ValueError('untruncated_fixed_text_inventory_required')
                        loaded.update(model=model, processor=processor)
                    except Exception:
                        loaded.clear()
                        loaded['failed'] = True
                        raise
                model = loaded['model']
                if model.training or any(p.requires_grad or p.device.type != 'cuda' or p.dtype != torch.float32 for p in model.parameters()):
                    raise ValueError('frozen_float32_cuda_model_required')
                batch = loaded['processor'](text=texts, images=image, padding='max_length',
                    max_length=64, truncation=False, return_tensors='pt')
                if tuple(batch['pixel_values'].shape) != (1, 3, 512, 512) or tuple(batch['input_ids'].shape) != (48, 64):
                    raise ValueError('native_fixed_tensor_shapes_required')
                batch = {k: v.to('cuda') for k, v in batch.items()}
                torch.cuda.synchronize()
                attempted = True
                with torch.inference_mode():
                    output = model(**batch, return_dict=True)
                torch.cuda.synchronize()
                cosine = output.image_embeds @ output.text_embeds.T
                if tuple(cosine.shape) != (1, 48) or tuple(output.logits_per_image.shape) != (1, 48):
                    raise ValueError('fixed_score_dimensions_required')
                pairs = decode_scores(cosine[0].float().cpu().tolist(), output.logits_per_image[0].float().cpu().tolist())
                return {'pairs': pairs, 'failure_reason': None, 'model_forward_attempted': True}
            except Exception as error:
                return {'pairs': None, 'failure_reason': type(error).__name__, 'model_forward_attempted': attempted}
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            records, attempts = invoke(plan, infer, Image=Image, reserve=reserve, complete=complete)
        prediction_path = write_private_json(temporary / 'predictions.json', {
            'schema_version': SCHEMA, 'frozen': True, 'image_only': True,
            'probe_catalog_sha256': guard.digest(probes()), 'records': records})
        with prediction_path.open('rb') as handle:
            os.fsync(handle.fileno())
        requests, _ = fixed_metadata(FRONTIER, FRONTIER_SHA, 'supplemental_image_evidence_requests.jsonl', jsonl=True)
        comparisons = posthoc(records, plan, requests)
        summary = {'schema_version': SCHEMA, 'unique_inputs': len(records),
            'scoring_attempts': attempts, 'actual_model_forward_attempts': sum(r['model_forward_attempted'] for r in records),
            'model_load_attempts': load_attempts, 'blocked_before_callback': sum(not r['callback_invoked'] for r in records),
            'complete_model_responses': sum(r['contract_status'] == 'complete' for r in records),
            'failed_unavailable_responses': sum(r['contract_status'] == 'failed_unavailable' for r in records),
            'logical_requests_compared': len(comparisons), 'model_retries': 0,
            'posthoc_comparisons_computed_after_prediction_fsync': True,
            'clinical_requests_resolved': 0, 'clinical_accuracy': None, 'primary_metric_eligible': False,
            'selection_changed': False, 'regeneration_authorized': False, 'fixed_ehr_changed': False,
            'shares_vision_encoder_with_chexagent2': True, 'training_overlap_unverified': True,
            'distinct_from_qwen_and_xrv_not_independent_clinical_truth': True,
            'runtime_seconds_including_preflight': round(time.monotonic() - started, 6),
            'peak_allocated_vram_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
            'gpu_name': torch.cuda.get_device_name(0), 'calibration_or_policy_fitting': False,
            'raw_patient_inputs_opened': False, 'ehr_report_or_real_target_bodies_opened': False}
        write_private_text(temporary / 'request_readout_comparisons.jsonl', tables.jsonl_text(comparisons))
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md',
            '# Alternate XraySigLIP readout / 补充自动读数\n\n'
            'Only prompt preferences; not clinical labels, probabilities, adjudication or repaired outputs.\n\n'
            '与 Qwen/XRV 不同，但与 CheXagent-2 共用编码器；21 个请求均未临床解决。\n\n'
            '```json\n' + json.dumps(summary, sort_keys=True, indent=2) + '\n```\n')
        seal(temporary, run_id, SCHEMA, sources, before, model_calls=summary['actual_model_forward_attempts'])
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException as error:
        # Retain new private journals for failures/in-flight reconciliation.
        if not (temporary / 'failure_status.json').exists():
            write_private_json(temporary / 'failure_status.json', {'status': 'failed_closed',
                'error_type': type(error).__name__, 'clinical_acceptance': False,
                'regeneration_authorized': False, 'automatic_resume_allowed': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--plan-run', type=Path)
    parser.add_argument('--allow-xraysiglip-image-probe', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            target = execute_prepare(args.output_root or BASE / 'xraysiglip_plans', args.run_id)
            status, calls = 'prepared_no_model_calls', 0
        else:
            if args.plan_run is None:
                raise ValueError('sealed_plan_required')
            target, summary = execute_run(args.plan_run, args.output_root or BASE / 'xraysiglip_runs',
                args.run_id, approved=args.allow_xraysiglip_image_probe)
            status = ('failed_all_readouts_no_selection_change' if summary['complete_model_responses'] == 0 else
                'completed_with_unavailable_readouts_no_selection_change' if summary['failed_unavailable_responses'] else
                'completed_alternate_readout_no_selection_change')
            calls = summary['actual_model_forward_attempts']
        print(json.dumps({'status': status, 'new_model_calls': calls, 'manifest_sha256': sha256_file(target / 'manifest.json')}))
        return 2 if status == 'failed_all_readouts_no_selection_change' else 0
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
