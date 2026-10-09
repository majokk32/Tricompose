#!/usr/bin/env python3
"""Metadata-only dependency-aware registry; no selection or clinical fault label."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'real_validation'),
               str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1')]
import score_cached_opacity_biovil as evidence
from contracts import (PROTECTED_ROOT, WORKSPACE, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)

BASE = PROTECTED_ROOT / 'tricompose_v1_2'
SOURCE = BASE / 'candidate_opacity_biovil_runs/biovil_opacity240_12670345'
SOURCE_SHA = '6b3027a012ec146b46a29a911465bb8fe1bdded087b6ee074d5994919c89c828'
PLAN = BASE / 'candidate_opacity_biovil_plans/biovil_pool240_12666569_001'
PLAN_SHA = '340c0774ee5f7c593397023f4ea48eb5eb16cffaeab68467d4a6a7720184b187'
PROTOCOL = ROOT.parent / 'docs/opacity_conflict_registry_protocol.md'
TESTS = ROOT / 'tests/test_opacity_conflict_registry.py'
SCHEMA = 'tricompose-opacity-conflict-registry-v1'
LANES = {
    'image_evidence_unavailable': 'missing_image_evidence',
    'image_sources_disagree': 'verify_image_evidence',
    'report_evidence_unavailable': 'missing_report_assertion',
    'three_proxy_sources_agree': 'proxy_agreement_control',
    'report_proposal_opposes_two_image_sources': 'verify_report_assertion',
}
POLICY = {'finding': 'lung_opacity', 'source_ehr_fixed': True,
    'report_dedup_key': 'exact_artifact_sha256', 'report_check_scope': 'text_assertion_only',
    'image_check_scope': 'one_existing_image_slot', 'diagnostic_only': True,
    'new_threshold': False, 'clinical_fault_localization': False,
    'independent_clinical_evidence': False, 'selector_enabled': False,
    'regeneration_authorized': False, 'new_gold_labels': False}


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def checked_rows(rows, images, outcomes, *, full=False):
    """Replay each saved field, including failure outcomes, without any inference."""
    if not rows or any(any(not isinstance(value, str) for value in row.values()) for row in rows):
        raise ValueError('nonempty_csv_named_cells_required')
    original = [{k: v for k, v in row.items() if not k.startswith(evidence.PREFIX)} for row in rows]
    evidence.validate_inventory(original, images, full=full)
    replay = list(csv.DictReader(io.StringIO(evidence.opacity.csv_text(
        evidence.overlay(original, images, outcomes)))))
    if replay != rows:
        raise ValueError('all_saved_evidence_cells_must_replay')
    if full and (any(len(row) != 86 for row in rows) or
                 {i['cxr_model_id'] for i in images} != evidence.opacity.MODELS):
        raise ValueError('full_86_field_three_model_inventory_required')
    groups = defaultdict(list)
    for row in rows:
        if row[evidence.PREFIX + 'joint_proxy_pattern'] not in LANES:
            raise ValueError('known_dependency_pattern_required')
        groups[row['cxr_candidate_id']].append(row)
    shared = ('opacity_exact_score', 'opacity_exact_state_0_5',
              evidence.PREFIX + 'preference_state', evidence.PREFIX + 'template_pattern')
    if any(any(row[key] != members[0][key] for key in shared for row in members)
           for members in groups.values()):
        raise ValueError('same_image_shared_proposals_required')
    if full:
        for image in images:
            members = groups[image['cxr_candidate_id']]
            if len(members) != 4 or {r['report_model_id'] for r in members} != evidence.opacity.EXPERTS:
                raise ValueError('four_distinct_experts_per_image_required')
        cases = defaultdict(list)
        for image in images:
            cases[image['case_id']].append(image)
        if any(len(group) != 3 or {i['cxr_model_id'] for i in group} != evidence.opacity.MODELS
               for group in cases.values()):
            raise ValueError('three_distinct_cxr_models_per_fixed_case_required')
    return groups


def build_registry(rows, images, outcomes, *, full=False):
    groups = checked_rows(rows, images, outcomes, full=full)
    image_index = {r['cxr_candidate_id']: r for r in images}
    outcome_index = {r['cxr_candidate_id']: r for r in outcomes}
    reports = defaultdict(list)
    for row in rows:
        reports[row['report_sha256']].append(row)
    report_ids = {key: f'report_group_{index:04d}' for index, key in enumerate(sorted(reports))}
    candidates = []
    for index, row in enumerate(rows):
        pattern = row[evidence.PREFIX + 'joint_proxy_pattern']
        candidates.append({
            'source_row_index': index, 'source_row_sha256': row_digest(row),
            **{k: row[k] for k in ('case_id', 'triple_candidate_id', 'cxr_candidate_id',
                'report_candidate_id', 'report_model_id', 'ehr_sha256', 'ehr_facts_sha256',
                'cxr_sha256', 'report_sha256')},
            'cxr_model_id': image_index[row['cxr_candidate_id']]['cxr_model_id'],
            'report_group_id': report_ids[row['report_sha256']],
            'report_global_slot_multiplicity': len(reports[row['report_sha256']]),
            'ehr_opacity_state': row['opacity_cached_ehr_state'],
            'xrv_opacity_state': row['opacity_exact_state_0_5'],
            'biovil_opacity_preference': row[evidence.PREFIX + 'preference_state'],
            'report_opacity_state': row['opacity_cached_report_state'],
            'joint_proxy_pattern': pattern, 'diagnostic_lane': LANES[pattern],
            'template_pattern': row[evidence.PREFIX + 'template_pattern'],
            'template_sign_variation': row[evidence.PREFIX + 'template_pattern'] == 'mixed_or_tied',
            'clinical_fault_label': None, 'clinical_accuracy': None,
            'regeneration_authorized': False, 'selector_used': False,
            'verification_executed': False})
    image_rows = []
    for image in images:
        members = groups[image['cxr_candidate_id']]
        first = members[0]
        image_rows.append({
            **{k: image[k] for k in ('case_id', 'cxr_candidate_id', 'cxr_model_id',
                'cxr_sha256', 'ehr_sha256', 'ehr_facts_sha256')},
            'scoring_status': outcome_index[image['cxr_candidate_id']]['status'],
            'xrv_opacity_state': first['opacity_exact_state_0_5'],
            'biovil_opacity_preference': first[evidence.PREFIX + 'preference_state'],
            'image_proxy_relation': first[evidence.PREFIX + 'exact_xrv_proxy_relation'],
            'template_pattern': first[evidence.PREFIX + 'template_pattern'],
            'template_sign_variation': first[evidence.PREFIX + 'template_pattern'] == 'mixed_or_tied',
            'candidate_ids': [r['triple_candidate_id'] for r in members],
            'report_group_ids': [report_ids[r['report_sha256']] for r in members],
            'pattern_slot_counts': dict(Counter(r[evidence.PREFIX + 'joint_proxy_pattern'] for r in members)),
            'clinical_fault_label': None, 'regeneration_authorized': False})
    report_rows = []
    for digest, group_id in sorted(report_ids.items(), key=lambda item: item[1]):
        members = reports[digest]
        report_rows.append({'report_group_id': group_id, 'report_sha256': digest,
            'cached_report_opacity_state': members[0]['opacity_cached_report_state'],
            'candidate_ids': [r['triple_candidate_id'] for r in members],
            'report_candidate_ids': [r['report_candidate_id'] for r in members],
            'image_ids': sorted({r['cxr_candidate_id'] for r in members}),
            'case_ids': sorted({r['case_id'] for r in members}),
            'report_model_ids': sorted({r['report_model_id'] for r in members}),
            'global_slot_multiplicity': len(members),
            'pattern_slot_counts': dict(Counter(r[evidence.PREFIX + 'joint_proxy_pattern'] for r in members)),
            'cross_image_reuse': len({r['cxr_candidate_id'] for r in members}) > 1,
            'cross_pattern_reuse': len({r[evidence.PREFIX + 'joint_proxy_pattern'] for r in members}) > 1,
            'clinical_content_error_established': False, 'verification_executed': False})
    cases = []
    for case_id in sorted({r['case_id'] for r in rows}):
        members = [r for r in rows if r['case_id'] == case_id]
        cases.append({'case_id': case_id, 'ehr_sha256': members[0]['ehr_sha256'],
            'ehr_facts_sha256': members[0]['ehr_facts_sha256'], 'ehr_opacity_state': 'unknown',
            'image_ids': sorted({r['cxr_candidate_id'] for r in members}),
            'candidate_ids': [r['triple_candidate_id'] for r in members],
            'report_group_ids': sorted({report_ids[r['report_sha256']] for r in members}),
            'pattern_slot_counts': dict(Counter(r[evidence.PREFIX + 'joint_proxy_pattern'] for r in members)),
            'clinical_localization_available': False})
    report_requests = []
    for record in report_rows:
        members = [r for r in candidates if r['report_group_id'] == record['report_group_id']
                   and r['diagnostic_lane'] == 'verify_report_assertion']
        if members:
            report_requests.append({'request_id': f'report_assertion_{len(report_requests):04d}',
                'task': 'text_only_opacity_assertion_check', 'report_group_id': record['report_group_id'],
                'report_sha256': record['report_sha256'],
                'trigger_candidate_ids': [r['triple_candidate_id'] for r in members],
                'reattach_all_candidate_ids': record['candidate_ids'],
                'source_contexts_retained': True, 'cached_predictions_exposed_to_verifier': False,
                'status': 'metadata_only_unmaterialized', 'model_calls': 0,
                'regeneration_authorized': False})
    image_requests = []
    for image in image_rows:
        if image['image_proxy_relation'] == 'proxy_opposition':
            image_requests.append({'request_id': f'image_evidence_{len(image_requests):04d}',
                'task': 'same_image_opacity_evidence_check', 'cxr_candidate_id': image['cxr_candidate_id'],
                'cxr_sha256': image['cxr_sha256'], 'candidate_ids': image['candidate_ids'],
                'cached_predictions_exposed_to_verifier': False,
                'status': 'metadata_only_unmaterialized', 'model_calls': 0,
                'regeneration_authorized': False})
    return {'candidates': candidates, 'images': image_rows, 'reports': report_rows,
            'cases': cases, 'requests': {'report_assertion_requests': report_requests,
                                        'image_evidence_requests': image_requests}}


def summarize(registry):
    rows = registry['candidates']
    patterns = {}
    for pattern, lane in LANES.items():
        subset = [r for r in rows if r['joint_proxy_pattern'] == pattern]
        patterns[pattern] = {'diagnostic_lane': lane, 'candidate_slots': len(subset),
            'unique_ehr_cases': len({r['case_id'] for r in subset}),
            'unique_image_slots': len({r['cxr_candidate_id'] for r in subset}),
            'unique_report_hashes': len({r['report_sha256'] for r in subset}),
            'by_report_model_candidate_slots': dict(Counter(r['report_model_id'] for r in subset)),
            'by_cxr_model_candidate_slots': dict(Counter(r['cxr_model_id'] for r in subset))}
    def by_model(records, model_field, label):
        groups = defaultdict(list)
        for record in records:
            groups[record[model_field]].append(record)
        return {key: {label: len(group), 'fixed_ehr_cases': len({r['case_id'] for r in group}),
                      'pattern_slot_counts': dict(Counter(r['joint_proxy_pattern'] for r in group))}
                for key, group in sorted(groups.items())}
    report_groups = registry['reports']
    return {'schema_version': SCHEMA, 'fixed_ehr_cases': len(registry['cases']),
        'image_slots': len(registry['images']), 'candidate_slots': len(rows),
        'unique_report_hashes': len(report_groups), 'dependency_patterns': patterns,
        'report_slot_multiplicity_histogram': dict(sorted(Counter(r['global_slot_multiplicity'] for r in report_groups).items())),
        'report_groups_reused_across_images': sum(r['cross_image_reuse'] for r in report_groups),
        'report_groups_reused_across_patterns': sum(r['cross_pattern_reuse'] for r in report_groups),
        'by_report_model': by_model(rows, 'report_model_id', 'candidate_slots'),
        'by_cxr_model': by_model(rows, 'cxr_model_id', 'candidate_slots'),
        'image_slots_by_cxr_model': dict(Counter(r['cxr_model_id'] for r in registry['images'])),
        'template_sign_variation_image_slots': sum(r['template_sign_variation'] for r in registry['images']),
        'pending_unique_text_assertion_requests': len(registry['requests']['report_assertion_requests']),
        'pending_unique_image_evidence_requests': len(registry['requests']['image_evidence_requests']),
        'resolved_verification_requests': 0, 'new_model_calls': 0, 'generation_calls': 0,
        'training_calls': 0, 'new_slurm_submissions': 0, 'old_scores_and_choices_changed': False,
        'clinical_fault_localization': False, 'clinical_repair_success': None,
        'ehr_opacity_edges_available': False, 'policy': POLICY}


def render(summary):
    lines = ['# Opacity conflict registry / 阴影分歧核验清单', '',
        'Metadata-only development diagnostic. No text/pixels inspected, model executed, winner changed or repair authorized.',
        'All EHRs remain fixed. Shared report bytes and four slots/image are not independent patients.', '',
        '| Pattern | Candidate slots | EHR cases | Image slots | Unique report hashes |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, record in summary['dependency_patterns'].items():
        lines.append(f"| {name} | {record['candidate_slots']} | {record['unique_ehr_cases']} | {record['unique_image_slots']} | {record['unique_report_hashes']} |")
    lines.extend(['', 'Per-pattern unique sets overlap: do not sum them as total distinct reports or cases.',
        '报告正文检查按精确文本hash去重；图像—报告比较仍保留每个上下文，不能按报告文本去重。',
        '相反提案不是已验证报告错误；一致控制不是干净临床gold；模板分歧不是校准概率。',
        'The pending requests contain IDs/hashes only, not executable input paths or verifier-ready text/image inputs.',
        'No cached model predictions are copied into new verifier-facing requests. Clinical accuracy and repair success remain NA.',
        '', '```json', json.dumps(summary, indent=2, sort_keys=True), '```', ''])
    return '\n'.join(lines)


def source_inputs():
    opacity = evidence.opacity
    pins = {str(path): sha256_file(path) for path in (Path(__file__), TESTS, PROTOCOL,
        Path(evidence.__file__), Path(opacity.__file__))}
    manifest = opacity.metadata(SOURCE / 'manifest.json', pins, SOURCE_SHA)
    opacity.verify_pins(manifest['sources']); pins.update(manifest['sources'])
    for name, digest in manifest['artifacts'].items():
        path = evidence.opacity.require_inside(SOURCE / name, SOURCE, must_exist=True)
        if sha256_file(path) != digest:
            raise ValueError('source_artifact_hash_changed')
        pins[str(path)] = digest
    pm = opacity.metadata(PLAN / 'manifest.json', pins, PLAN_SHA)
    opacity.verify_pins(pm['sources']); pins.update(pm['sources'])
    plan = opacity.metadata(PLAN / 'plan.json', pins, pm['artifacts']['plan.json'])
    if manifest['schema_version'] != evidence.SCHEMA + '-manifest' or \
            manifest['plan_manifest_sha256'] != PLAN_SHA or plan['modality_source'] != 'fully_synthetic' or \
            plan['policy'] != evidence.POLICY or plan['old_scores_and_choices_changed'] is not False:
        raise ValueError('same_completed_fully_synthetic_overlay_required')
    values = {name: opacity.metadata(SOURCE / name, pins, manifest['artifacts'][name])
              for name in ('image_scores.json', 'candidate_evidence_table.csv', 'summary.json')}
    if manifest['runtime'] != plan['runtime'] or values['summary.json']['old_scores_and_choices_changed'] is not False:
        raise ValueError('unchanged_bound_runtime_and_scores_required')
    rows, outcomes = values['candidate_evidence_table.csv'], values['image_scores.json']['records']
    replay = list(csv.DictReader(io.StringIO(opacity.csv_text(evidence.overlay(plan['rows'], plan['images'], outcomes)))))
    if rows != replay or evidence.aggregate(rows, outcomes) != {
            key: values['summary.json'][key] for key in evidence.aggregate(rows, outcomes)}:
        raise ValueError('source_row_and_aggregate_replay_required')
    opacity.verify_pins(pins)
    return rows, plan['images'], outcomes, pins


def run(args):
    evidence.opacity.guard()
    rows, images, outcomes, pins = source_inputs()
    registry = build_registry(rows, images, outcomes, full=True)
    summary = summarize(registry)
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_text(temporary / 'candidate_registry.csv', evidence.opacity.csv_text(registry['candidates']))
        for name, key in (('image_registry.json', 'images'), ('report_registry.json', 'reports'),
                          ('case_registry.json', 'cases')):
            write_private_json(temporary / name, {'schema_version': SCHEMA, 'records': registry[key]})
        write_private_json(temporary / 'verification_requests.json', {'schema_version': SCHEMA, **registry['requests']})
        write_private_json(temporary / 'summary.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md', render(summary))
        evidence.opacity.verify_pins(pins)
        artifacts = {path.name: sha256_file(path) for path in sorted(temporary.iterdir())}
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA + '-manifest',
            'source_manifest_sha256': SOURCE_SHA, 'policy': POLICY,
            'sources': pins, 'artifacts': artifacts, 'old_scores_and_choices_changed': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    try:
        target, _ = run(args)
        print(json.dumps({'status': 'completed_metadata_only_registry', 'new_model_calls': 0,
            'manifest_sha256': sha256_file(target / 'manifest.json')}, sort_keys=True))
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
