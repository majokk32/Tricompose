#!/usr/bin/env python3
"""Bounded CPU PNG guard evaluation and lossless candidate-table sidecar."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import image_validity_guard as guard
import verify_cached_image_findings as cached
import attach_report_gate_availability as tables
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

SCHEMA = 'tricompose-fixed-image-validity-guard-evaluation-v1'
BASE = PROTECTED_ROOT/'tricompose_v1_2'
ORIGINAL_PLAN = BASE/'image_only_plans/image_only_scope2_12645021_001'
ORIGINAL_PLAN_SHA = '5fbcd984de88e7f776f3ef52081a98d978d620848f15feb63cfc19b4bedbcdfe'
CONTROL_RUN = BASE/'image_control_runs/noinfo_scope2_12650073'
CONTROL_SHA = '87d1e211e21fe497bda8647ae356a5c5a61246801bd5a288a828263f4f908ecd'
POOL = BASE/'image_verification_availability/imagecheck_pool960_12645021_001'
POOL_SHA = 'd933d9ba37181e2eed31f3f92fd8831d514a313fc225c6a6bb6f4260ad08f1bf'


def fixed_parent(root, expected, names):
    mp = require_inside(root/'manifest.json', PROTECTED_ROOT, must_exist=True)
    if sha256_file(mp) != expected:
        raise ValueError('fixed_parent_manifest_required')
    manifest = cached.bounded_json(mp)
    if any(manifest[k] is not False for k in ('primary_metric_eligible', 'selection_changed', 'regeneration_authorized')):
        raise ValueError('unqualified_unchanged_parent_required')
    sources, inputs = {root.name+'_manifest': mp}, {}
    for name in names:
        path = require_inside(root/name, root, must_exist=True)
        if path.stat().st_size > 4*1024*1024 or sha256_file(path) != manifest['artifacts'][name]['sha256']:
            raise ValueError('bounded_quote_free_parent_artifact_required')
        if name.endswith('.csv'):
            with path.open(encoding='utf-8', newline='') as handle:
                value = list(csv.DictReader(handle))
            if any(None in r or None in r.values() for r in value):
                raise ValueError('complete_original_csv_required')
        else:
            value = json.loads(path.read_text(encoding='utf-8'))
        sources[root.name+'_'+name] = path
        inputs[name] = value
    if sha256_file(mp) != expected:
        raise ValueError('parent_changed_during_read')
    return manifest, inputs, sources


def load_inputs():
    if sha256_file(ORIGINAL_PLAN/'manifest.json') != ORIGINAL_PLAN_SHA:
        raise ValueError('fixed_synthetic_plan_required')
    original, sources = cached.load_plan(ORIGINAL_PLAN)
    control_manifest, control, more = fixed_parent(CONTROL_RUN, CONTROL_SHA,
        ('realized_inputs.json', 'predictions.json'))
    sources.update(more)
    pool_manifest, pool, more = fixed_parent(POOL, POOL_SHA, ('candidate_score_table.csv',))
    sources.update(more)
    if control_manifest['new_model_calls'] != 10 or pool_manifest['new_model_calls'] != 0:
        raise ValueError('fixed_completed_control_and_availability_runs_required')
    sources.update(guard=Path(guard.__file__), worker=Path(__file__),
        tests=ROOT/'tests/test_image_validity_guard.py',
        protocol=ROOT.parent/'docs/image_validity_guard_protocol.md', table_writer=Path(tables.__file__))
    inputs = control['realized_inputs.json']
    pred = control['predictions.json']
    if (len(original['image_inputs']) != 6 or len(inputs['slots']) != 18
            or len(inputs['saved_uniform_controls']) != 4 or len(pred['records']) != 10
            or pred['frozen'] is not True or pred['image_only'] is not True
            or pred['model_received_arm_names_ehr_reports_ids_scores_or_expected_answers'] is not False
            or pred['image_prompt_sha256'] != original['image_prompt_sha256']):
        raise ValueError('fixed_six_original_four_control_input_inventory_required')
    return original, inputs, pred, pool['candidate_score_table.csv'], control_manifest, sources


def bind_input_files(original, inputs, records, control_manifest):
    by_source = {r['cxr_candidate_id']: r for r in original['image_inputs']}
    if len(by_source) != len(original['image_inputs']):
        raise ValueError('unique_original_image_ids_required')
    uniform = {r['observation_id']: r for r in inputs['saved_uniform_controls']}
    if len(uniform) != len(inputs['saved_uniform_controls']):
        raise ValueError('unique_saved_control_pixels_required')
    files, source_map, slot_ids = {}, {}, set()
    for slot in inputs['slots']:
        if slot['slot_id'] in slot_ids or slot['arm'] not in ('original', 'uniform_black', 'uniform_white'):
            raise ValueError('unique_known_logical_slot_required')
        slot_ids.add(slot['slot_id'])
        source = by_source.get(slot['cxr_candidate_id'])
        if source is None or source['cxr_sha256'] != slot['cxr_sha256']:
            raise ValueError('exact_synthetic_source_slot_required')
        if slot['arm'] == 'original':
            path = require_inside(source['path'], PROTECTED_ROOT, must_exist=True)
            if [path.stat().st_size, path.stat().st_mtime_ns] != source['image_file_stats']:
                raise ValueError('unchanged_original_image_stats_required')
            expected = source['cxr_sha256']
        else:
            entry = uniform.get(slot['observation_id'])
            if entry is None:
                raise ValueError('existing_saved_control_required')
            path = require_inside(CONTROL_RUN/entry['path'], CONTROL_RUN, must_exist=True)
            expected = entry['sha256']
            if expected != control_manifest['artifacts'][entry['path']]['sha256']:
                raise ValueError('sealed_control_image_required')
        binding = {'path': str(path), 'sha256': expected,
            'width': slot['width'], 'height': slot['height']}
        key = slot['observation_id']
        if files.setdefault(key, binding) != binding:
            raise ValueError('shared_normalized_image_binding_changed')
        source_map['inspected_input_'+str(len(source_map))] = path
    index = {r['observation_id']: r for r in records}
    if len(index) != len(records) or set(index) != set(files):
        raise ValueError('all_fixed_cached_pixel_records_required')
    return files, index, source_map


def attach_candidates(rows, original, slots, guards):
    references = {r['cxr_candidate_id']: r for r in original['image_inputs']}
    lookup = {}
    for slot in slots:
        if slot['arm'] == 'original':
            lookup[slot['cxr_candidate_id']] = guards[slot['observation_id']]
    output = []
    for row in rows:
        if any(k.startswith('imageguard_') for k in row):
            raise ValueError('already_attached_candidate_refused')
        current = lookup.get(row['cxr_candidate_id'])
        if current is not None:
            ref = references[row['cxr_candidate_id']]
            if (any(row[k] != ref[k] for k in ('case_id', 'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256'))
                    or current['artifact_sha256'] != row['cxr_sha256']):
                raise ValueError('exact_original_candidate_image_and_ehr_required')
        output.append({**row, 'imageguard_status': 'not_checked' if current is None else current['status'],
            'imageguard_reason': None if current is None else current['reason'],
            'imageguard_receipt_sha256': None if current is None else current['receipt_sha256'],
            'imageguard_basic_comparison_permitted': None if current is None else current['basic_comparison_permitted'],
            'imageguard_clinical_acceptance': False, 'imageguard_primary_metric_eligible': False,
            'imageguard_regeneration_authorized': False})
    if len(output) != len(rows) or any(any(new[k] != v for k, v in old.items()) for old, new in zip(rows, output)):
        raise ValueError('every_original_candidate_cell_and_order_must_be_preserved')
    return output


def summarize(views, slots, rows):
    status = {v['raw_verifier_result']['observation_id']: v['guard']['status'] for v in views}
    blocked = [v for v in views if not v['guard']['basic_comparison_permitted']]
    return {'schema_version': SCHEMA, 'unique_checked_inputs': len(views), 'logical_slots': len(slots),
        'unique_guard_status_counts': dict(sorted(Counter(v['guard']['status'] for v in views).items())),
        'logical_guard_status_counts': dict(sorted(Counter(status[s['observation_id']] for s in slots).items())),
        'blocked_unique_input_readout_slots': len(blocked)*len(guard.HEADS),
        'withheld_raw_explicit_assertions_on_blocked_inputs': sum(s in ('positive', 'negative')
            for v in blocked for s in (v['raw_verifier_result']['states'] or {}).values()),
        'retained_cached_finding_states': sum(len(v['guarded_states'] or {}) for v in views),
        'candidate_rows': len(rows), 'candidate_guard_status_counts': dict(sorted(Counter(r['imageguard_status'] for r in rows).items())),
        'source_verifier_records_preserved': True, 'all_original_candidate_cells_and_order_preserved': True,
        'new_model_calls': 0, 'actual_compute_savings': None, 'new_slurm_submissions': 0,
        'clinical_accuracy': None, 'clinical_acceptance': False, 'primary_metric_eligible': False,
        'regeneration_authorized': False, 'selection_changed': False, 'fixed_ehr_changed': False,
        'clinically_resolved_requests': 0, 'raw_patient_inputs_opened': False,
        'ehr_report_or_real_target_bodies_opened': False, 'protected_synthetic_pngs_decoded': True,
        'old_gpu_workers_replaced': False, 'prospective_guard_hook_available_not_executed': True,
        'development_engineering_check_not_heldout_clinical_benchmark': True}


def execute(output_root, run_id):
    cached.require_slurm()
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        original, inputs, pred, rows, control_manifest, sources = load_inputs()
        if len(rows) != 960 or len({r['triple_candidate_id'] for r in rows}) != 960:
            raise ValueError('fixed_full_candidate_inventory_required')
        files, records, image_sources = bind_input_files(original, inputs, pred['records'], control_manifest)
        sources.update(image_sources)
        before = {k: sha256_file(p) for k, p in sources.items()}
        for binding in files.values():
            if sha256_file(Path(binding['path'])) != binding['sha256']:
                raise ValueError('fixed_image_bytes_changed')
        import PIL
        from PIL import Image, ImageFile, PngImagePlugin, _imaging
        if ImageFile.LOAD_TRUNCATED_IMAGES is not False:
            raise ValueError('strict_native_png_decoder_required')
        decoder_paths = {name: Path(module.__file__).resolve() for name, module in
            (('Image', Image), ('ImageFile', ImageFile), ('PngImagePlugin', PngImagePlugin), ('_imaging', _imaging))}
        decoder_hashes = {k: sha256_file(p) for k, p in decoder_paths.items()}
        started, guards, views = time.monotonic(), {}, []
        for key in sorted(files):
            binding = files[key]
            current, _ = guard.inspect_png(binding['path'], binding['sha256'], Image=Image)
            native = current['native_image_metadata']
            if (current['normalized_pixel_sha256'] != key or native is None
                    or (native['width'], native['height']) != (binding['width'], binding['height'])):
                raise ValueError('exact_inspected_cached_input_required')
            guards[key] = current
            views.append(guard.cached_view(current, records[key]))
        for view in views:
            if view['raw_verifier_result'] != records[view['raw_verifier_result']['observation_id']]:
                raise ValueError('source_predictions_must_remain_unchanged')
        annotated = attach_candidates(rows, original, inputs['slots'], guards)
        if (sum(r['imageguard_status'] != 'not_checked' for r in annotated),
                len({r['case_id'] for r in annotated}), len(guards)) != (24, 80, 10):
            raise ValueError('fixed_twocase_siximage_candidate_scope_required')
        summary = summarize(views, inputs['slots'], annotated)
        summary.update(existing_cpu_job_id=os.environ['SLURM_JOB_ID'],
            runtime_seconds=round(time.monotonic()-started, 6),
            decoder={'library': 'Pillow', 'version': PIL.__version__, 'python_executable': sys.executable})
        write_private_text(temporary/'guarded_verifier_views.jsonl', tables.jsonl_text(views))
        write_private_text(temporary/'input_guard_table.jsonl', tables.jsonl_text([
            {'observation_id': key, **files[key], 'guard': guards[key]} for key in sorted(files)]))
        write_private_json(temporary/'logical_slot_guards.json', {'slots': [
            {**s, 'imageguard_status': guards[s['observation_id']]['status']} for s in inputs['slots']]})
        write_private_text(temporary/'candidate_score_table.csv', tables.csv_text(annotated))
        write_private_json(temporary/'summary.json', summary)
        write_private_text(temporary/'RESULTS_CN_EN.md',
            '# Mechanical image guard / 图像有效性护栏\n\n'
            '只检查解码、固定尺寸/模式以及完全均匀像素；通过不代表解剖或临床正确。\n\n'
            '原始标签、候选分数和择优结果不变。Blocked guarded states are null, not negative or unknown.\n\n'
            'No model call, prospective compute savings, clinical truth or automatic regeneration is established.\n\n'
            '```json\n'+json.dumps(summary, sort_keys=True, indent=2)+'\n```\n')
        if (any(sha256_file(p) != before[k] for k, p in sources.items())
                or any(sha256_file(p) != decoder_hashes[k] for k, p in decoder_paths.items())):
            raise ValueError('consumed_source_or_decoder_changed')
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {k: str(p.resolve()) for k, p in sources.items()}, 'source_sha256': before,
            'decoder_source_paths': {k: str(p) for k, p in decoder_paths.items()}, 'decoder_source_sha256': decoder_hashes,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'protected_synthetic_pngs_decoded': True,
            'ehr_or_report_bodies_opened': False})
        for p in (temporary, *temporary.iterdir()):
            st = p.stat()
            if st.st_gid not in (96293, 65534) or st.st_mode & 0o7777 != (0o2770 if p.is_dir() else 0o660):
                raise ValueError('protected_project_permissions_required')
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE/'image_validity_guards')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        target, summary = execute(args.output_root, args.run_id)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'fixed_image_guard_evaluated', 'unique_checked_inputs': summary['unique_checked_inputs'],
        'guard_counts': summary['unique_guard_status_counts'], 'new_model_calls': 0,
        'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
