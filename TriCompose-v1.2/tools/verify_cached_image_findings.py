#!/usr/bin/env python3
"""Image-only frozen Qwen check on the existing six-image synthetic subset.

Preparation opens metadata only. GPU execution receives images and the unchanged
image-only prompt, never EHRs, reports, score tables or model/candidate names.
"""
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
for directory in (ROOT/'benchmarks', ROOT/'src', ROOT.parent/'src',
        ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'):
    sys.path.insert(0, str(directory))
from contracts import (PROTECTED_ROOT, WORKSPACE, CHEXPERT_FINDINGS, require_inside,
    sha256_file, new_atomic_run, commit_atomic_run, discard_atomic_run,
    write_private_json, write_private_text)
import verify_candidate_findings_qwen as image_interface
import verify_legacy_report_semantics as frozen
from tricompose_v12.scorer_reliability import digest as canonical_request_sha256

BASE = PROTECTED_ROOT/'tricompose_v1_2'
SCHEMA = 'tricompose-cached-synthetic-image-only-qwen-v1'
MODELS = frozenset(('roentgen_v2', 'chexgenbench_sana', 'chexgenbench_pixart'))
STATES = frozenset(('positive', 'negative', 'uncertain', 'unknown'))
PARENTS = {
    'scope': (BASE/'legacy_report_scope_plans/scope2_plan_12632006_001',
        '5856c43c78eaa4b8c8959b153a6ee188ef97cfe3d617cce2ce14556f06e4d69c', 'plan.json'),
    'model': (BASE/'report_span_v2_plans/span_v2_scope2_12645021_001',
        '576483c1b4f544b7fd99f5b6e7aca94d8c22c83694294fd41cd3ab9d1aa74afc', 'plan.json'),
    'bank': (BASE/'complete_bank_endpoints/complete_bank_secondary_12624822_001',
        'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c', None),
    'gate': (BASE/'verification_gates/scope_gate_cached_12645021_001',
        '86a3e37a212d06fb665e7797397dd7736a449f92e30bb19c322923cc3f76e66a', None),
}


def require_slurm(*, approved=False, gpu=False):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('actual_slurm_allocation_required')
    if gpu and not approved:
        raise RuntimeError('separately_approved_image_gpu_run_required')


def bounded_json(path, cap=2*1024*1024):
    if not path.is_file() or path.stat().st_size > cap:
        raise ValueError('bounded_metadata_required')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('metadata_object_required')
    return value


def parent(label):
    root, expected, name = PARENTS[label]
    mp = require_inside(root/'manifest.json', PROTECTED_ROOT, must_exist=True)
    if sha256_file(mp) != expected:
        raise ValueError('fixed_parent_manifest_required')
    manifest = bounded_json(mp)
    if name is None:
        return manifest, None, {label+'_manifest': mp}
    path = require_inside(root/name, root, must_exist=True)
    if sha256_file(path) != manifest['artifacts'][name]['sha256']:
        raise ValueError('fixed_parent_artifact_required')
    return manifest, bounded_json(path), {label+'_manifest': mp, label+'_plan': path}


def expected_images(rows):
    """All three paths of the same two fixed cases, not score-selected images."""
    if len(rows) != 24 or len({r['case_id'] for r in rows}) != 2:
        raise ValueError('same_fixed_two_case_24_slot_subset_required')
    images = {}
    for row in rows:
        value = {key: row[key] for key in ('case_id', 'cxr_candidate_id',
            'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')}
        cid = row['cxr_candidate_id']
        if images.setdefault(cid, value) != value:
            raise ValueError('shared_image_or_fixed_ehr_changed')
    if len(images) != 6 or any(sum(r['cxr_candidate_id'] == cid for r in rows) != 4 for cid in images):
        raise ValueError('six_images_four_reports_each_required')
    return images


def synthetic_origin_manifest(staged):
    # Historical teammate staging metadata lives under TriCompose-v1.0, not
    # protected/. Read its one hash-bound manifest only, never EHR bodies there.
    root = require_inside(staged['source_v1_staging_run'], WORKSPACE, must_exist=True)
    path = require_inside(root/'run_manifest.json', root, must_exist=True)
    if sha256_file(path) != staged['source_v1_run_manifest_sha256']:
        raise ValueError('original_synthetic_staging_manifest_required')
    origin = bounded_json(path)
    if origin.get('schema_version') != 'tricompose.staging.run.v1' or origin.get('source_generator') not in (
            'synehrgy_gpt2_10bins', 'synehrgy_qwen2_40bins', 'synehrgy_v2'):
        raise ValueError('known_synthetic_ehr_generator_required')
    return path


def resolve_images(rows, bank_manifest):
    expected = expected_images(rows)
    images, sources = {}, {}
    for name in sorted(k for k in bank_manifest['source_paths'] if k.startswith('cxr_manifest_')):
        mp = require_inside(bank_manifest['source_paths'][name], PROTECTED_ROOT, must_exist=True)
        if sha256_file(mp) != bank_manifest['source_sha256'][name]:
            raise ValueError('frozen_cxr_manifest_required')
        manifest = bounded_json(mp)
        if (manifest['schema_version'] != 'tricompose-cxr-candidate-run-v1.1'
                or manifest['frozen_model'] is not True or manifest['model_id'] not in MODELS
                or manifest['candidate_count'] != len(manifest['candidates'])):
            raise ValueError('frozen_text_to_image_candidate_run_required')
        sources[name] = mp
        request_root = require_inside(manifest['source_request_run'], PROTECTED_ROOT, must_exist=True)
        request_manifest = request_root/'manifest.json'
        if sha256_file(request_manifest) != manifest['source_request_run_manifest_sha256']:
            raise ValueError('frozen_input_request_manifest_required')
        requests = bounded_json(request_manifest)
        staging = require_inside(requests['staging_run'], PROTECTED_ROOT, must_exist=True)
        staging_manifest = staging/'run_manifest.json'
        if sha256_file(staging_manifest) != requests['staging_run_manifest_sha256']:
            raise ValueError('frozen_staging_manifest_required')
        staged = bounded_json(staging_manifest)
        if staged['schema_version'] != 'tricompose.staging.run.v1.1':
            raise ValueError('synthetic_staging_schema_required')
        original_manifest = synthetic_origin_manifest(staged)
        sources.update({name+'_requests': request_manifest, name+'_staging': staging_manifest,
            name+'_synthetic_origin': original_manifest})
        for entry in manifest['candidates']:
            cid = entry['candidate_id']
            if cid not in expected:
                continue
            if cid in images:
                raise ValueError('unique_selected_image_candidate_required')
            cp = require_inside(mp.parent/entry['path'], mp.parent, must_exist=True)
            if sha256_file(cp) != entry['sha256']:
                raise ValueError('frozen_image_candidate_metadata_required')
            candidate = bounded_json(cp, 32768)
            ref = expected[cid]
            if (candidate['schema_version'] != 'tricompose-cxr-candidate-v1.1'
                    or candidate['frozen_model'] is not True or candidate['modality'] != 'cxr'
                    or candidate['candidate_id'] != cid or candidate['case_id'] != ref['case_id']
                    or entry['case_id'] != ref['case_id'] or candidate['model_id'] != manifest['model_id']
                    or candidate['artifact']['sha256'] != ref['cxr_sha256']
                    or candidate['ehr_sha256'] != ref['ehr_sha256']
                    or candidate['ehr_facts_sha256'] != ref['ehr_facts_sha256'] or candidate['seed'] != 0):
                raise ValueError('fixed_synthetic_image_lineage_required')
            matches = [r for r in requests['requests'] if r['case_id'] == candidate['case_id']
                and r['model_id'] == candidate['model_id'] and r['seed'] == candidate['seed']]
            if len(matches) != 1:
                raise ValueError('unique_frozen_image_generation_request_required')
            req_path = require_inside(request_root/matches[0]['path'], request_root, must_exist=True)
            if sha256_file(req_path) != matches[0]['sha256']:
                raise ValueError('fixed_image_generation_request_required')
            request = bounded_json(req_path, 32768)
            # The legacy candidate pins canonical JSON, while the request-run
            # manifest pins file bytes. They must not be compared to each other.
            if canonical_request_sha256(request) != candidate['input_request_sha256']:
                raise ValueError('fixed_canonical_image_request_required')
            if (set(request['inputs']) != {'staging_run_manifest', 'synthetic_ehr', 'ehr_facts', 'final_prompt'}
                    or request['inputs']['synthetic_ehr']['sha256'] != ref['ehr_sha256']
                    or request['inputs']['ehr_facts']['sha256'] != ref['ehr_facts_sha256']
                    or request['inputs']['staging_run_manifest']['sha256'] != sha256_file(staging_manifest)
                    or request['frozen_model_required'] is not True):
                raise ValueError('synthetic_text_only_conditioning_required')
            artifact = require_inside(candidate['artifact']['path'], mp.parent, must_exist=True)
            # Stat only in preparation: no image bytes/hash/pixels are opened.
            if not artifact.is_file() or not 1 <= artifact.stat().st_size <= 16*1024*1024:
                raise ValueError('bounded_existing_image_file_required')
            images[cid] = {**ref, 'path': str(artifact), 'model_id': candidate['model_id'],
                'seed': candidate['seed'], 'image_file_stats': [artifact.stat().st_size, artifact.stat().st_mtime_ns]}
            sources['selected_cxr_metadata_'+str(len(images)-1)] = cp
            sources['selected_cxr_request_'+str(len(images)-1)] = req_path
    if set(images) != set(expected) or any({r['model_id'] for r in images.values() if r['case_id'] == case} != MODELS
            for case in {r['case_id'] for r in rows}):
        raise ValueError('complete_same_two_case_three_model_image_inventory_required')
    return sorted(images.values(), key=lambda r: (r['cxr_sha256'], r['cxr_candidate_id'])), sources


def program_sources():
    return {**frozen.sources_for_program(), 'image_worker': Path(__file__),
        'image_tests': ROOT/'tests/test_cached_image_findings.py',
        'image_protocol': ROOT.parent/'docs/cached_image_findings_protocol.md'}


def prepare():
    _, scope, sources = parent('scope')
    _, model, model_sources = parent('model')
    bank_manifest, _, bank_sources = parent('bank')
    gate_manifest, _, gate_sources = parent('gate')
    if (scope['modality_source'] != 'fully_synthetic'
            or scope['source_report_or_real_target_supplied'] is not False
            or gate_manifest['regeneration_authorized'] is not False):
        raise ValueError('same_unqualified_synthetic_subset_required')
    images, metadata = resolve_images(scope['candidate_rows'], bank_manifest)
    model_path = Path(model['model_path']).resolve(strict=True)
    if any([(model_path/name).stat().st_size, (model_path/name).stat().st_mtime_ns] != value
            for name, value in model['model_file_stats'].items()):
        raise ValueError('frozen_model_file_metadata_required')
    if any(sha256_file(model_path/name) != value for name, value in model['model_file_sha256'].items()
            if name != 'model.safetensors'):
        raise ValueError('frozen_small_model_assets_required')
    sources.update({**model_sources, **bank_sources, **gate_sources, **metadata, **program_sources()})
    prompt = image_interface.IMAGE_PROMPT.format(findings=', '.join(image_interface.FINDINGS))
    plan = {'schema_version': SCHEMA+'-plan', 'image_inputs': images, 'image_slots': 6,
        'candidate_slots': 24, 'fixed_ehr_cases': 2, 'finding_order': list(image_interface.FINDINGS),
        'prompt_version': image_interface.PROMPT_VERSION, 'image_prompt_sha256': image_interface.digest_text(prompt),
        'model_path': str(model_path), 'model_file_stats': model['model_file_stats'],
        'model_file_sha256': model['model_file_sha256'], 'max_new_tokens': 384,
        'min_pixels': 256*28*28, 'max_pixels': 512*28*28, 'max_model_calls': 6,
        'model_retries': 0, 'seed': 0, 'do_sample': False, 'min_vram_gib': 24,
        'metadata_only': True, 'image_bytes_or_pixels_opened_in_prepare': False,
        'ehr_or_report_bodies_opened': False, 'new_model_calls': 0, 'primary_metric_eligible': False,
        'selection_changed': False, 'regeneration_authorized': False,
        'independent_clinical_truth_available': False, 'development_subset_not_heldout': True}
    return plan, sources


def load_plan(root):
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    mp, pp = root/'manifest.json', root/'plan.json'
    manifest, plan = bounded_json(mp), bounded_json(pp)
    if manifest['schema_version'] != SCHEMA+'-plan' or sha256_file(pp) != manifest['artifacts']['plan.json']['sha256']:
        raise ValueError('sealed_image_only_plan_required')
    sources = {key: require_inside(path, WORKSPACE, must_exist=True) for key, path in manifest['source_paths'].items()}
    if any(sha256_file(path) != manifest['source_sha256'][key] for key, path in sources.items()):
        raise ValueError('sealed_program_or_metadata_changed')
    rebuilt, rebuilt_sources = prepare()
    if plan != rebuilt or sources != {k: p.resolve() for k, p in rebuilt_sources.items()}:
        raise ValueError('exact_fixed_image_protocol_required')
    sources.update(image_plan_manifest=mp, image_plan=pp)
    return plan, sources


def sanitized_decode(response, *, token_limit_reached=False):
    decoded = image_interface.decode_response(response)
    complete = decoded['contract_status'] == 'complete' and not token_limit_reached
    return {'contract_status': 'complete' if complete else 'failed_unavailable',
        'states': decoded['states'] if complete else None,
        'failure_reason': None if complete else 'token_cap' if token_limit_reached else 'invalid_eight_state_json',
        'independent_clinical_validation': False}


def relation(left, right):
    if left not in STATES or right not in STATES:
        raise ValueError('valid_four_states_required')
    if left in ('positive', 'negative') and right in ('positive', 'negative'):
        return 'explicit_agreement_unqualified' if left == right else 'explicit_opposition_unqualified'
    if left == right == 'unknown':
        return 'both_unmentioned_or_unassessable'
    if 'uncertain' in (left, right):
        return 'uncertainty_not_comparable'
    return 'single_source_assertion_unqualified'


def compare_after_freeze(records, plan):
    _, scope, sources = parent('scope')
    gate_manifest, _, gate_sources = parent('gate')
    gate_path = PARENTS['gate'][0]/'candidate_gate_fact_table.jsonl'
    if gate_path.stat().st_size > 2*1024*1024 or sha256_file(gate_path) != gate_manifest['artifacts'][gate_path.name]['sha256']:
        raise ValueError('sealed_quote_free_report_gate_required')
    gates = [json.loads(line) for line in gate_path.read_text().splitlines() if line]
    index = {r['cxr_candidate_id']: r for r in records}
    if len(index) != 6 or set(index) != {r['cxr_candidate_id'] for r in plan['image_inputs']}:
        raise ValueError('all_six_image_results_required')
    for record in records:
        if record['contract_status'] == 'complete':
            if (not isinstance(record['states'], dict)
                    or set(record['states']) != set(image_interface.FINDINGS)
                    or not set(record['states'].values()) <= STATES):
                raise ValueError('complete_named_eight_state_image_result_required')
        elif record['contract_status'] != 'failed_unavailable' or record['states'] is not None:
            raise ValueError('unavailable_image_must_have_null_states')
    facts = {(r['triple_candidate_id'], r['finding']): r for r in scope['fact_rows']}
    seen, output = set(), []
    for gate in gates:
        key = gate['triple_candidate_id'], gate['finding']
        if key in seen or key not in facts:
            raise ValueError('unique_same_subset_gate_finding_required')
        seen.add(key)
        fact = facts[key]
        result = index[fact['cxr_candidate_id']]
        if (result['cxr_sha256'] != fact['artifact_hashes']['cxr_sha256']
                or gate['report_sha256'] != fact['artifact_hashes']['report_sha256']
                or gate['report_candidate_id'] != fact['report_candidate_id']
                or gate['raw_chexbert_state'] != fact['states']['chexbert']):
            raise ValueError('exact_image_report_and_cached_state_lineage_required')
        finding = gate['finding']
        state = None
        image_relation = report_relation = 'outside_image_verifier_scope'
        if finding in image_interface.FINDINGS:
            if result['contract_status'] != 'complete':
                image_relation = report_relation = 'image_verifier_unavailable'
            else:
                state = result['states'][finding]
                image_relation = relation(fact['states']['xrv'], state)
                report_relation = ('report_assertion_not_retained' if gate['scopegate_decision'] != 'scope_commit'
                    else relation(state, gate['scopegate_retained_state']))
        output.append({**gate, 'imagecheck_cxr_candidate_id': fact['cxr_candidate_id'],
            'imagecheck_cxr_sha256': result['cxr_sha256'], 'imagecheck_contract_status': result['contract_status'],
            'imagecheck_raw_xrv_state': fact['states']['xrv'], 'imagecheck_qwen_state': state,
            'imagecheck_xrv_qwen_relation': image_relation,
            'imagecheck_retained_report_relation': report_relation,
            'imagecheck_independent_clinical_validation': False, 'imagecheck_primary_metric_eligible': False,
            'imagecheck_confirmed_faulty_modality': None, 'imagecheck_regeneration_authorized': False})
    if seen != set(facts) or len(output) != 336:
        raise ValueError('complete_original_336_finding_rows_required')
    unique_image_relations = {}
    for row in output:
        key = row['imagecheck_cxr_candidate_id'], row['finding']
        label = row['imagecheck_xrv_qwen_relation']
        if unique_image_relations.setdefault(key, label) != label:
            raise ValueError('shared_image_comparison_changed')
    summary = {'schema_version': SCHEMA, 'image_slots': 6, 'candidate_slots': 24, 'fact_rows': 336,
        'unique_image_finding_relation_counts': dict(sorted(Counter(unique_image_relations.values()).items())),
        'retained_report_relation_counts_candidate_findings': dict(sorted(Counter(
            r['imagecheck_retained_report_relation'] for r in output).items())),
        'complete_image_responses': sum(r['contract_status'] == 'complete' for r in records),
        'failed_image_responses': sum(r['contract_status'] != 'complete' for r in records),
        'model_calls': len(records), 'model_retries': 0,
        'model_received_ehr_reports_scores_or_candidate_ids': False,
        'comparison_metadata_read_after_prediction_fsync': True,
        'independent_clinical_accuracy': None, 'primary_metric_eligible': False,
        'selection_changed': False, 'regeneration_authorized': False,
        'clinical_requests_resolved': 0, 'development_subset_not_heldout': True}
    return output, summary, {**sources, **gate_sources, 'report_gate_facts': gate_path}


def run(plan_root, temporary, *, approved=False):
    require_slurm(gpu=True, approved=approved)
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 24*1024**3:
        raise RuntimeError('gpu_with_at_least_24_gib_required')
    plan, sources = load_plan(plan_root)
    before = {key: sha256_file(path) for key, path in sources.items()}
    model_path = Path(plan['model_path'])
    if {name: sha256_file(model_path/name) for name in frozen.MODEL_FILES} != plan['model_file_sha256']:
        raise ValueError('frozen_model_bytes_changed')
    from PIL import Image
    from tricompose.verifiers.qwenvl_cxr_report import _load_model
    torch.manual_seed(plan['seed'])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    records = []
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        model, processor, model_type = _load_model(model_path, torch,
            min_pixels=plan['min_pixels'], max_pixels=plan['max_pixels'])
        model.eval().requires_grad_(False)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('frozen_model_required')
        for item in plan['image_inputs']:
            if len(records) >= plan['max_model_calls']:
                raise ValueError('six_call_budget_exceeded')
            path = require_inside(item['path'], PROTECTED_ROOT, must_exist=True)
            if sha256_file(path) != item['cxr_sha256']:
                raise ValueError('sealed_synthetic_image_bytes_changed')
            sources['synthetic_image_'+str(len(records))] = path
            before['synthetic_image_'+str(len(records))] = item['cxr_sha256']
            with Image.open(path) as handle:
                if handle.width > 4096 or handle.height > 4096 or handle.width < 64 or handle.height < 64:
                    raise ValueError('bounded_cxr_dimensions_required')
                image = handle.convert('RGB').copy()
            messages = image_interface.request_messages('image', image=image)
            torch.cuda.synchronize()
            call_started = time.monotonic()
            response, tokens = image_interface.infer(messages, model, processor, torch,
                max_new_tokens=plan['max_new_tokens'])
            torch.cuda.synchronize()
            records.append({'cxr_candidate_id': item['cxr_candidate_id'], 'cxr_sha256': item['cxr_sha256'],
                'response_sha256': image_interface.digest_text(response), **tokens,
                **sanitized_decode(response, token_limit_reached=tokens['token_limit_reached']),
                'elapsed_seconds': round(time.monotonic()-call_started, 4)})
    predictions = write_private_json(temporary/'predictions.json', {'schema_version': SCHEMA, 'records': records,
        'frozen': True, 'image_only': True, 'image_prompt_sha256': plan['image_prompt_sha256'],
        'model_received_ehr_reports_scores_or_candidate_ids': False})
    with predictions.open('rb') as handle:
        os.fsync(handle.fileno())
    prediction_sha = sha256_file(predictions)
    comparison, summary, comparison_sources = compare_after_freeze(records, plan)
    sources.update(comparison_sources)
    for key, path in comparison_sources.items():
        if key not in before:
            before[key] = sha256_file(path)
    summary.update(runtime_seconds=round(time.monotonic()-started, 4),
        peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
        gpu_name=torch.cuda.get_device_name(0), torch_version=str(torch.__version__), model_type=model_type,
        token_cap_failures=sum(r['token_limit_reached'] for r in records),
        model_file_sha256=plan['model_file_sha256'])
    if (any([(model_path/name).stat().st_size, (model_path/name).stat().st_mtime_ns] != value
            for name, value in plan['model_file_stats'].items())
            or any(sha256_file(path) != before[key] for key, path in sources.items())
            or sha256_file(predictions) != prediction_sha):
        raise ValueError('immutable_sources_or_frozen_predictions_changed')
    return comparison, summary, sources, before


def execute(args):
    require_slurm(gpu=args.mode == 'run', approved=args.allow_image_only_verification)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        if args.mode == 'prepare':
            plan, sources = prepare()
            before = {key: sha256_file(path) for key, path in sources.items()}
            write_private_json(temporary/'plan.json', plan)
            schema = SCHEMA+'-plan'
        else:
            comparison, summary, sources, before = run(args.plan_run, temporary,
                approved=args.allow_image_only_verification)
            write_private_text(temporary/'image_report_comparison.jsonl', ''.join(
                json.dumps(row, sort_keys=True)+'\n' for row in comparison))
            write_private_json(temporary/'summary.json', summary)
            write_private_text(temporary/'RESULTS_CN_EN.md',
                '# Image-only verification / 单独看图核查\n\n'
                'Same six synthetic images, unchanged frozen eight-head prompt; no EHR/report in image requests.\n\n'
                'Disagreement is not clinical truth, fault localization or repair authorization.\n\n'
                '```json\n'+json.dumps(summary, sort_keys=True, indent=2)+'\n```\n')
            schema = SCHEMA
        if any(sha256_file(path) != before[key] for key, path in sources.items()):
            raise ValueError('immutable_sources_changed')
        write_private_json(temporary/'manifest.json', {'schema_version': schema, 'run_id': args.run_id,
            'source_paths': {key: str(path.resolve()) for key, path in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls': 0 if args.mode == 'prepare' else summary['model_calls'],
            'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
            'ehr_or_report_bodies_opened': False, 'image_pixels_opened': args.mode == 'run'})
        for path in (temporary, *temporary.iterdir()):
            st = path.stat()
            if st.st_gid not in (96293, 65534) or st.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary, target)
        return target
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--plan-run', type=Path)
    parser.add_argument('--allow-image-only-verification', action='store_true')
    args = parser.parse_args()
    try:
        target = execute(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'image_only_plan_prepared' if args.mode == 'prepare' else 'image_only_check_completed',
        'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
