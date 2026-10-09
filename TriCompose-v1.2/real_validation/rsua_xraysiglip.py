#!/usr/bin/env python3
"""Fixed public-cohort polarity readout, not a calibrated clinical adjudicator."""
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
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'real_validation'))
import probe_xraysiglip_findings as probe
import rsua_biovil as baseline
from contracts import (PROTECTED_ROOT, WORKSPACE, require_inside, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

BASE = PROTECTED_ROOT / 'tricompose_v1_2/real_validation'
SCHEMA = 'tricompose-rsua-xraysiglip-polarity-v1'
PRIOR = BASE / 'rsua_biovil_pilots/rsua_biovil50_12637081'
PRIOR_SHA = 'a8eed9db9e20146cb1bc7ca597aed78f8bfb5cd1683b29e82dde99497b08992a'
XRV = BASE / 'rsua_xrv_pilots/rsua_xrv50_12636566'
COHORT = BASE / 'rsua_pilot_cohorts/cohort50_12625457_001/cohort.json'
NATIVE_PLAN = PROTECTED_ROOT / 'tricompose_v1_2/xraysiglip_plans/siglip_scope2_12645021_001'
NATIVE_SHA = 'fbf4d8bd4e5bf93078533ef90f4e65a2d861a53cc2296269fd31570898be3f89'
VALIDATED_WEIGHT_RUN = PROTECTED_ROOT / 'tricompose_v1_2/xraysiglip_runs/siglip_scope2_12654030/manifest.json'
VALIDATED_WEIGHT_MANIFEST_SHA = 'd846602fabba7abca85745e18fd5daba00832ac251ce458bd9f53ff3fa3c54eb'
SCRIPT = ROOT / 'slurm/48_rsua_xraysiglip50_flexible_gpu.sbatch'
REPEATS = ('case_0000', 'case_0001')


def metadata(root, expected, name):
    mp = require_inside(root / 'manifest.json', PROTECTED_ROOT, must_exist=True)
    if mp.stat().st_size > 1024 ** 2 or sha256_file(mp) != expected:
        raise ValueError('fixed_parent_manifest_required')
    manifest = json.loads(mp.read_text())
    path = require_inside(root / name, root, must_exist=True)
    recorded = manifest['artifacts'][name]
    expected_artifact = recorded['sha256'] if isinstance(recorded, dict) else recorded
    if path.stat().st_size > 2 * 1024 ** 2 or sha256_file(path) != expected_artifact:
        raise ValueError('bounded_fixed_metadata_required')
    value = json.loads(path.read_text())
    if sha256_file(mp) != expected:
        raise ValueError('parent_changed_during_read')
    return value, {root.name + '_manifest': mp, root.name + '_' + name: path}


def prepare_payload():
    prior, sources = metadata(PRIOR, PRIOR_SHA, 'scores.json')
    summary, more = metadata(PRIOR, PRIOR_SHA, 'summary.json')
    sources.update(more)
    if (summary['frozen'] is not True or summary['independent_clinical_adjudication'] is not False
            or summary['selection_changed'] is not False or summary['thresholds_fitted'] is not False
            or summary['template_choice_fitted'] is not False or summary['primary_metric_eligible'] is not False
            or summary['execution']['source_cases'] != 50 or summary['execution']['lossless_png_checked'] != 50):
        raise ValueError('unchanged_prior_readout_required')
    if [r['case_id'] for r in prior['records']] != [f'case_{i:04d}' for i in range(50)]:
        raise ValueError('fixed_fifty_opaque_cases_required')
    inputs = []
    for row in prior['records']:
        for field in ('source_image_sha256', 'png_sha256'):
            if not probe.guard.HASH.fullmatch(row[field]):
                raise ValueError('fixed_source_image_hash_required')
        path = require_inside(PRIOR / 'inputs' / (row['case_id'] + '.png'), PRIOR, must_exist=True)
        if not 1 <= path.stat().st_size <= probe.guard.MAX_BYTES:
            raise ValueError('bounded_existing_png_inventory_required')
        inputs.append({'case_id': row['case_id'], 'path': str(path), 'sha256': row['png_sha256'],
            'original_bmp_sha256': row['source_image_sha256'],
            'file_stats': [path.stat().st_size, path.stat().st_mtime_ns]})
    if len({i['sha256'] for i in inputs}) != 50:
        raise ValueError('duplicate_images_no_substitution')
    native, more = metadata(NATIVE_PLAN, NATIVE_SHA, 'plan.json')
    sources.update(more)
    native_manifest = json.loads((NATIVE_PLAN / 'manifest.json').read_text())
    for key, name in native_manifest['source_paths'].items():
        if key.startswith('runtime_'):
            path = require_inside(probe.SITE / key.removeprefix('runtime_'), WORKSPACE, must_exist=True)
        elif key.startswith('model_') and key.removeprefix('model_') in probe.MODEL_META:
            path = require_inside(probe.MODEL / key.removeprefix('model_'), WORKSPACE, must_exist=True)
        else:
            continue
        if (path != Path(name).resolve() or path.stat().st_size > 8 * 1024 ** 2
                or sha256_file(path) != native_manifest['source_sha256'][key]):
            raise ValueError('unchanged_native_runtime_or_model_metadata_required')
        sources[key] = path
    required = {'model_' + name for name in probe.MODEL_META}
    if not required <= set(sources):
        raise ValueError('complete_existing_native_model_metadata_required')
    probe.validate_model_metadata(*(json.loads((probe.MODEL / name).read_text()) for name in
        ('config.json', 'preprocessor_config.json', 'tokenizer_config.json')))
    weight = probe.MODEL / 'model.safetensors'
    if sha256_file(VALIDATED_WEIGHT_RUN) != VALIDATED_WEIGHT_MANIFEST_SHA:
        raise ValueError('validated_existing_weight_receipt_required')
    weight_sha = json.loads(VALIDATED_WEIGHT_RUN.read_text())['source_sha256']['complete_existing_weight']
    if not probe.guard.HASH.fullmatch(weight_sha):
        raise ValueError('validated_full_weight_sha256_required')
    sources['validated_weight_receipt'] = VALIDATED_WEIGHT_RUN
    if (native['model_path'] != str(probe.MODEL) or native['probes'] != probe.probes()
            or native['weight_stats'] != [weight.stat().st_size, weight.stat().st_mtime_ns]
            or weight.stat().st_size != 2612631968 or sha256_file(COHORT) != baseline.COHORT_SHA):
        raise ValueError('fixed_native_assets_or_cohort_required')
    for label, path in {'worker': Path(__file__), 'tests': ROOT / 'tests/test_rsua_xraysiglip.py',
            'protocol': ROOT.parent / 'docs/rsua_xraysiglip_protocol.md',
            'probe': Path(probe.__file__), 'guard': Path(probe.guard.__file__),
            'table_helpers': Path(probe.tables.__file__), 'rsua_biovil_metrics': Path(baseline.__file__),
            'auc_metrics': ROOT / 'real_validation/biovil_matched_pairs.py',
            'biovil_polarity_catalog': ROOT / 'real_validation/biovil_fact_polarity.py',
            'biovil_adapter': WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py',
            'contracts': Path(sys.modules['contracts'].__file__)}.items():
        sources[label] = path
    return {'schema_version': SCHEMA + '-plan', 'inputs': inputs, 'probes': probe.probes(),
        'case_count': 50, 'repeat_ids': list(REPEATS), 'max_callback_attempts': 52,
        'max_forward_attempts': 52, 'model_retries': 0, 'model_path': str(probe.MODEL),
        'weight_path': str(weight), 'weight_stats': native['weight_stats'], 'weight_sha256': weight_sha,
        'python_executable': str(probe.ENVIRONMENT / 'bin/python'), 'image_size': [512, 512],
        'text_max_length': 64, 'precision': 'float32', 'min_vram_gib': 12, 'seed': 42,
        'cohort_path': str(COHORT), 'cohort_sha256': baseline.COHORT_SHA,
        'primary_reduction': baseline.MEAN, 'reference_finding': 'pneumonia',
        'cohort_labels_or_old_scores_in_model_plan': False, 'pixels_opened': False, 'weights_opened': False,
        'clinical_reference_is_published_cohort_proxy': True, 'independent_clinical_validation': False,
        'primary_metric_eligible': False, 'regeneration_authorized': False, 'selection_changed': False}, sources


def prepare(output_root, run_id):
    probe.tables.require_cpu_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        plan, sources = prepare_payload()
        before = {k: sha256_file(p) for k, p in sources.items()}
        write_private_json(temporary / 'plan.json', plan)
        probe.seal(temporary, run_id, SCHEMA + '-plan', sources, before)
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary)
        raise


def analyze(records, references):
    if len(records) != len(references) or len({r['case_id'] for r in records}) != len(records):
        raise ValueError('complete_fixed_reference_inventory_required')
    if set(references) != {r['case_id'] for r in records} or not set(references.values()) <= {'positive', 'negative'}:
        raise ValueError('known_exact_proxy_references_required')
    complete = [r for r in records if r['pairs'] is not None]
    if len(complete) != len(records):
        return {'source_cases': len(records), 'available': len(complete), 'unavailable': len(records) - len(complete),
            'full_cohort_metrics': None, 'stable_text_direction_counts': None,
            'stable_coverage': None, 'stable_proxy_agreement': None}
    pneumonia_rows, preferences = [], Counter()
    stable_wins = 0
    for record in records:
        value = probe.summarize_pairs(record['pairs'])['pneumonia']
        preference = value['text_preference']
        preferences[preference] += 1
        if preference in ('present_prompt_higher', 'absent_prompt_higher'):
            stable_wins += (references[record['case_id']] == 'positive') == (preference == 'present_prompt_higher')
        pairs = [p for p in record['pairs'] if p['finding'] == 'pneumonia']
        pneumonia_rows.append({'case_id': record['case_id'], 'score_pairs': {
            p['family']: {k: p[k] for k in ('positive_cosine', 'negative_cosine')} for p in pairs}})
    stable = preferences['present_prompt_higher'] + preferences['absent_prompt_higher']
    return {'source_cases': len(records), 'available': len(records), 'unavailable': 0,
        'full_cohort_metrics': baseline.summarize(pneumonia_rows, references),
        'stable_text_direction_counts': dict(sorted(preferences.items())),
        'stable_coverage': stable / len(records), 'stable_proxy_wins': stable_wins,
        'stable_proxy_agreement': stable_wins / stable if stable else None}


def repeatability(primary, repeats):
    by_id = {r['case_id']: r for r in primary}
    if [r['case_id'] for r in repeats] != list(REPEATS) or not set(REPEATS) <= set(by_id):
        raise ValueError('only_fixed_first_two_technical_repeats_required')
    result = []
    for repeated in repeats:
        original = by_id[repeated['case_id']]
        if original['artifact_sha256'] != repeated['artifact_sha256']:
            raise ValueError('repeat_must_bind_same_image')
        delta = None
        if original['pairs'] is not None and repeated['pairs'] is not None:
            probe.summarize_pairs(original['pairs']); probe.summarize_pairs(repeated['pairs'])
            delta = max(abs(a[k] - b[k]) for a, b in zip(original['pairs'], repeated['pairs'])
                for k in ('positive_cosine', 'negative_cosine', 'positive_logit', 'negative_logit'))
        result.append({'case_id': repeated['case_id'], 'max_abs_endpoint_score_delta': delta,
            'primary_score_replaced': False})
    return result


def compare_directions(records, biovil_records, xrv_scores):
    ids = {r['case_id'] for r in records}
    if (len(ids) != len(records) or ids != {r['case_id'] for r in biovil_records}
            or len(biovil_records) != len(ids) or ids != set(xrv_scores)):
        raise ValueError('exact_cross_scorer_inventory_required')
    biovil = {r['case_id']: baseline.margins(r)[baseline.MEAN] for r in biovil_records}
    outputs = {}
    for profile, threshold in [('biovil_mean_margin', 0.0), *baseline.PROFILES.items()]:
        counts = Counter()
        for record in records:
            case_id = record['case_id']
            other = biovil[case_id] if profile == 'biovil_mean_margin' else probe.finite_number(xrv_scores[case_id]) - threshold
            if profile != 'biovil_mean_margin' and not 0 <= xrv_scores[case_id] <= 1:
                raise ValueError('bounded_raw_xrv_score_required')
            margin = None if record['pairs'] is None else probe.summarize_pairs(record['pairs'])['pneumonia']['mean_cosine_margin']
            category = ('not_comparable' if margin is None or margin == 0 or (profile == 'biovil_mean_margin' and other == 0) else
                'same_direction_unqualified' if (margin > 0) == (other > 0 if profile == 'biovil_mean_margin' else other >= 0) else
                'opposed_direction_unqualified')
            counts[category] += 1
        outputs[profile] = dict(sorted(counts.items()))
    return outputs


def load_plan(plan_root, expected_sha):
    plan_root = require_inside(plan_root, PROTECTED_ROOT, must_exist=True)
    manifest_path = plan_root / 'manifest.json'
    if sha256_file(manifest_path) != expected_sha:
        raise ValueError('sealed_plan_manifest_required')
    manifest = json.loads(manifest_path.read_text())
    plan_path = plan_root / 'plan.json'
    if (manifest['schema_version'] != SCHEMA + '-plan'
            or sha256_file(plan_path) != manifest['artifacts']['plan.json']['sha256']):
        raise ValueError('unchanged_blind_plan_required')
    rebuilt, sources = prepare_payload()
    if (json.loads(plan_path.read_text()) != rebuilt
            or manifest['source_paths'] != {k: str(p.resolve()) for k, p in sources.items()}
            or any(sha256_file(p) != manifest['source_sha256'][k] for k, p in sources.items())):
        raise ValueError('exact_sealed_program_runtime_and_inputs_required')
    sources.update(plan_manifest=manifest_path, plan=plan_path)
    return rebuilt, sources


def run(args):
    probe.require_gpu_approval(getattr(args, 'allow_rsua_siglip_probe', False))
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 12 * 1024 ** 3:
        raise RuntimeError('cuda_with_at_least_twelve_gib_required')
    plan, sources = load_plan(args.plan_run, args.plan_manifest_sha256)
    if Path(sys.executable).absolute() != Path(plan['python_executable']).absolute():
        raise ValueError('pinned_native_python_environment_required')
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        started = time.monotonic()
        sources['approved_batch_script'] = SCRIPT
        sources['complete_existing_weight'] = Path(plan['weight_path'])
        for item in plan['inputs']:
            path = require_inside(item['path'], PRIOR, must_exist=True)
            if (sha256_file(path) != item['sha256']
                    or [path.stat().st_size, path.stat().st_mtime_ns] != item['file_stats']):
                raise ValueError('fixed_image_bytes_and_stats_required')
            sources['image_' + item['case_id']] = path
        before = {k: sha256_file(p) for k, p in sources.items()}
        if before['complete_existing_weight'] != plan['weight_sha256']:
            raise ValueError('unchanged_validated_full_model_weights_required')
        from PIL import Image, ImageFile
        from transformers import SiglipModel, SiglipProcessor
        if ImageFile.LOAD_TRUNCATED_IMAGES is not False:
            raise ValueError('strict_native_png_decoder_required')
        torch.manual_seed(plan['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        texts = [p[k] for p in probe.probes() for k in ('positive_text', 'negative_text')]
        loaded, counters = {}, {'callback_attempts': 0, 'actual_forward_attempts': 0, 'model_load_attempts': 0}
        journal = write_private_text(temporary / 'attempt_journal.jsonl', '')
        def append(event):
            with journal.open('a') as handle:
                handle.write(json.dumps(event, sort_keys=True) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        def infer(image):
            attempted = False
            try:
                if image.size != (256, 256) or image.mode != 'RGB':
                    raise ValueError('fixed_rsua_native_image_dimensions_required')
                if loaded.get('failed'):
                    return {'pairs': None, 'failure_reason': 'model_unavailable_after_load_failure', 'forward_attempted': False}
                if not loaded:
                    counters['model_load_attempts'] += 1
                    try:
                        model, info = SiglipModel.from_pretrained(plan['model_path'], local_files_only=True,
                            use_safetensors=True, torch_dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
                        if any(info[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                            raise ValueError('complete_existing_siglip_weights_required')
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
                if counters['actual_forward_attempts'] >= plan['max_forward_attempts']:
                    raise ValueError('fixed_forward_budget_exhausted')
                attempted = True
                counters['actual_forward_attempts'] += 1
                with torch.inference_mode():
                    output = model(**batch, return_dict=True)
                torch.cuda.synchronize()
                cosine = output.image_embeds @ output.text_embeds.T
                if tuple(cosine.shape) != (1, 48) or tuple(output.logits_per_image.shape) != (1, 48):
                    raise ValueError('fixed_endpoint_score_shapes_required')
                pairs = probe.decode_scores(cosine[0].float().cpu().tolist(), output.logits_per_image[0].float().cpu().tolist())
                return {'pairs': pairs, 'failure_reason': None, 'forward_attempted': True}
            except Exception as error:
                return {'pairs': None, 'failure_reason': type(error).__name__, 'forward_attempted': attempted}
        def score(item, kind):
            def callback(image):
                if counters['callback_attempts'] >= plan['max_callback_attempts']:
                    raise ValueError('fixed_callback_budget_exhausted')
                counters['callback_attempts'] += 1
                index = counters['callback_attempts']
                append({'event': 'reserved', 'attempt': index, 'case_id': item['case_id'], 'kind': kind})
                call_started = time.monotonic()
                raw = infer(image)
                raw['elapsed_seconds'] = round(time.monotonic() - call_started, 6)
                append({'event': 'completed' if raw['pairs'] is not None else 'failed_unavailable',
                    'attempt': index, 'case_id': item['case_id'], 'kind': kind,
                    'forward_attempted': raw['forward_attempted'], 'failure_reason': raw['failure_reason'],
                    'elapsed_seconds': raw['elapsed_seconds']})
                return raw
            outcome = probe.guard.guarded_invoke(item['path'], item['sha256'], callback, Image=Image)
            raw = outcome['callback_result'] if outcome['callback_invoked'] else {
                'pairs': None, 'failure_reason': outcome['guard']['reason'], 'forward_attempted': False, 'elapsed_seconds': None}
            return {'case_id': item['case_id'], 'kind': kind, 'artifact_sha256': item['sha256'],
                'original_bmp_sha256': item['original_bmp_sha256'], 'guard': outcome['guard'],
                'callback_invoked': outcome['callback_invoked'], **raw,
                'clinical_state': None, 'calibrated_probability': None, 'independent_clinical_validation': False}
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            records = [score(item, 'primary') for item in plan['inputs']]
            repeats = [score(item, 'technical_repeat') for item in plan['inputs'] if item['case_id'] in REPEATS]
        predictions = write_private_json(temporary / 'predictions.json', {
            'schema_version': SCHEMA, 'frozen': True, 'image_only': True, 'records': records, 'technical_repeats': repeats})
        with predictions.open('rb') as handle:
            os.fsync(handle.fileno())
        # Ground-reference/proxy labels are parsed only after predictions are fsynced.
        if sha256_file(COHORT) != plan['cohort_sha256']:
            raise ValueError('unchanged_fixed_reference_cohort_required')
        cohort = json.loads(COHORT.read_text())
        references = {r['case_id']: r['reference_state'] for r in cohort['records']}
        if len(references) != 50 or Counter(references.values()) != {'positive': 25, 'negative': 25}:
            raise ValueError('fixed_balanced_reference_denominators_required')
        baseline.source_receipt(XRV)
        xrv = json.loads((XRV / 'scores.json').read_text())['records']
        if (len(xrv) != 50 or any(r['reference_state'] != references[r['case_id']] for r in xrv)
                or any(r['image_sha256'] != i['original_bmp_sha256'] for r, i in zip(xrv, plan['inputs']))):
            raise ValueError('exact_prior_xrv_reference_and_image_binding_required')
        old_biovil, _ = metadata(PRIOR, PRIOR_SHA, 'scores.json')
        biovil_rows = [{'case_id': r['case_id'], 'score_pairs': r['score_pairs']} for r in old_biovil['records']]
        analysis = analyze(records, references)
        sources.update(reference_cohort=COHORT, xrv_manifest=XRV / 'manifest.json', xrv_scores=XRV / 'scores.json', xrv_summary=XRV / 'summary.json')
        before.update({k: sha256_file(sources[k]) for k in ('reference_cohort', 'xrv_manifest', 'xrv_scores', 'xrv_summary')})
        summary = {'schema_version': SCHEMA, 'status': 'completed_published_cohort_proxy_diagnostic',
            'analysis': analysis, 'cross_scorer': compare_directions(records, biovil_rows, {r['case_id']: r['pneumonia_score'] for r in xrv}),
            'repeatability': repeatability(records, repeats), 'primary_reduction': baseline.MEAN,
            'execution': {**counters, 'source_cases': 50, 'technical_repeat_cases': 2,
                'blocked_before_callback': sum(not r['callback_invoked'] for r in records + repeats),
                'complete_primary_readouts': analysis['available'], 'model_retries': 0, 'generation_calls': 0},
            'frozen': True, 'clinical_reference_is_published_cohort_proxy': True,
            'clinical_accuracy': None, 'independent_clinical_validation': False, 'training_overlap_verified': False,
            'age_domain_verified': False, 'patient_grouping_verified': False, 'other_seven_heads_have_reference_labels': False,
            'primary_metric_eligible': False, 'probability_semantics': False, 'thresholds_fitted': False,
            'template_choice_fitted': False, 'selection_changed': False, 'regeneration_authorized': False,
            'shares_vision_encoder_with_chexagent2': True, 'model_received_reference_labels': False,
            'predictions_fsynced_before_reference_analysis': True, 'mimic_ehr_report_or_target_opened': False,
            'public_reference_images_read': True, 'elapsed_seconds_including_preflight': round(time.monotonic() - started, 6),
            'peak_allocated_vram_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
            'gpu_name': torch.cuda.get_device_name(0)}
        write_private_json(temporary / 'summary.json', summary)
        write_private_json(temporary / 'prompts.json', {'catalog': probe.probes(), 'sha256': probe.guard.digest(probe.probes())})
        write_private_text(temporary / 'RESULTS_CN_EN.md',
            '# RSUA XraySigLIP polarity / 公共分组诊断\n\n'
            'Published cohort proxy, not per-finding clinical adjudication or repair authorization.\n\n'
            '固定全部样本与三种模板；不选最佳模板、不调阈值、不把未知变阴性。\n\n'
            '```json\n' + json.dumps(summary, sort_keys=True, indent=2) + '\n```\n')
        if any(sha256_file(path) != before[key] for key, path in sources.items()):
            raise ValueError('consumed_source_changed_during_run')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA, 'run_id': args.run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls': counters['actual_forward_attempts'], 'selection_changed': False,
            'primary_metric_eligible': False, 'regeneration_authorized': False,
            'independent_clinical_validation': False, 'mimic_ehr_report_or_target_opened': False,
            'public_reference_images_read': True})
        probe.private_modes(temporary)
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException as error:
        if not (temporary / 'failure_status.json').exists():
            write_private_json(temporary / 'failure_status.json', {'status': 'failed_closed',
                'error_type': type(error).__name__, 'regeneration_authorized': False, 'automatic_resume_allowed': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--plan-run', type=Path)
    parser.add_argument('--plan-manifest-sha256')
    parser.add_argument('--allow-rsua-siglip-probe', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            target = prepare(args.output_root or BASE / 'rsua_siglip_plans', args.run_id)
            status = 'prepared_metadata_only_no_model_calls'
        else:
            args.output_root = args.output_root or BASE / 'rsua_siglip_pilots'
            target, summary = run(args)
            status = ('failed_all_readouts_no_selection_change' if summary['analysis']['available'] == 0 else
                'completed_public_cohort_proxy_no_selection_change' if summary['analysis']['unavailable'] == 0 else
                'completed_with_unavailable_readouts_no_selection_change')
        print(json.dumps({'status': status, 'manifest_sha256': sha256_file(target / 'manifest.json')}))
        return 2 if status == 'failed_all_readouts_no_selection_change' else 0
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
