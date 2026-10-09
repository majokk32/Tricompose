#!/usr/bin/env python3
"""Prepare and execute a frozen CPU-only synthetic report-scope smoke.

Prepare reads metadata only. Run opens only the sealed selected synthetic text,
never real targets, EHR rows, image pixels, checkpoints or model factories.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, WORKSPACE, RUN_ID_PATTERN, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from preview_reliability_evidence import load_sidecar
from tricompose_v12.legacy_report_scope import SCHEMA, SELECTION_RULE, select_cases, build_scope_audit
from repair_cached_report_evidence import scope_check
from report_assertion_challenge import FROZEN_GUARD_SHA

BANK_MANIFEST_SHA = 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c'
PLAN_SCHEMA = SCHEMA+'-plan'


def bounded_json(path, cap=1024*1024):
    if path.stat().st_size > cap:
        raise ValueError('metadata_size_limit_exceeded')
    return read_json(path)


def metadata_resolver(rows, bank_root):
    """Resolve only selected report metadata, without calling image loaders."""
    root = require_inside(bank_root, PROTECTED_ROOT, must_exist=True)
    mp = root/'manifest.json'
    if sha256_file(mp) != BANK_MANIFEST_SHA:
        raise ValueError('historical_bank_manifest_changed')
    manifest = bounded_json(mp)
    if (manifest['schema_version'] != 'tricompose-full-bank-secondary-coverage-v1'
            or manifest['original_selection_changed'] is not False or manifest['clinical_acceptance'] is not False):
        raise ValueError('immutable_historical_bank_required')
    selected = {row['report_candidate_id']: row for row in rows}
    if len(selected) != len(rows):
        raise ValueError('distinct_selected_report_slots_required')
    inputs, sources = {}, {'historical_bank_manifest': mp}
    for name in sorted(k for k in manifest['source_paths'] if k.startswith('report_manifest_')):
        path = require_inside(manifest['source_paths'][name], PROTECTED_ROOT, must_exist=True)
        if sha256_file(path) != manifest['source_sha256'][name]:
            raise ValueError('frozen_report_manifest_changed')
        payload = bounded_json(path)
        if (payload['schema_version'] != 'tricompose-report-candidate-run-v1.1'
                or payload['frozen_model'] is not True or payload['source_report_or_real_target_supplied'] is not False
                or payload['candidate_count'] != len(payload['candidates'])):
            raise ValueError('frozen_synthetic_report_manifest_required')
        sources[name] = path
        for entry in payload['candidates']:
            rid = entry['candidate_id']
            if rid not in selected:
                continue
            if rid in inputs:
                raise ValueError('duplicate_selected_report_metadata')
            row = selected[rid]
            cp = require_inside(path.parent/entry['path'], path.parent, must_exist=True)
            if cp.stat().st_size > 16384 or sha256_file(cp) != entry['sha256']:
                raise ValueError('selected_report_metadata_changed')
            candidate = bounded_json(cp, 16384)
            if (candidate['schema_version'] != 'tricompose-report-candidate-v1.1'
                    or candidate['frozen_model'] is not True or candidate['modality'] != 'report'
                    or candidate['source_report_or_real_target_supplied'] is not False
                    or candidate['candidate_id'] != rid or candidate['case_id'] != row['case_id']
                    or entry['case_id'] != row['case_id']
                    or candidate['model_id'] != row['report_model_id'] or candidate['model_id'] != payload['model_id']
                    or candidate['parent_cxr_candidate_id'] != row['cxr_candidate_id']
                    or entry['parent_cxr_candidate_id'] != row['cxr_candidate_id']
                    or candidate['ehr_sha256_retained_for_lineage'] != row['ehr_sha256']
                    or candidate['ehr_facts_sha256_retained_for_lineage'] != row['ehr_facts_sha256']
                    or candidate['artifact']['sha256'] != row['report_sha256']):
                raise ValueError('selected_synthetic_report_lineage_changed')
            artifact = require_inside(candidate['artifact']['path'], path.parent, must_exist=True)
            inputs[rid] = {'report_candidate_id': rid, 'case_id': row['case_id'],
                'path': str(artifact), 'report_sha256': row['report_sha256']}
            sources['selected_report_metadata_'+str(len(inputs)-1)] = cp
    if set(inputs) != set(selected):
        raise ValueError('complete_selected_report_inventory_required')
    return [inputs[row['report_candidate_id']] for row in rows], sources


def program_sources():
    sources = {'program': Path(__file__), 'scope_contract': ROOT/'src/tricompose_v12/legacy_report_scope.py',
        'scope_checker': ROOT/'benchmarks/repair_cached_report_evidence.py',
        'assertion_contract': ROOT/'src/tricompose_v12/report_assertions.py',
        'relation_contract': ROOT/'src/tricompose_v12/report_scope_table.py',
        'sidecar_loader': ROOT/'benchmarks/preview_reliability_evidence.py',
        'sidecar_validator': ROOT/'src/tricompose_v12/reliability_preview.py',
        'protected_contract': ROOT.parent/'TriCompose-v1.0/eval/report_v1_1/contracts.py'}
    if sha256_file(sources['scope_checker']) != FROZEN_GUARD_SHA:
        raise ValueError('frozen_four_finding_scope_checker_changed')
    return sources


def prepare(args):
    rows, facts, groups, sources = load_sidecar(args.reliability_run)
    rows, facts, groups = select_cases(rows, facts, groups, count=2)
    if len(rows) != 24 or len(facts) != 336 or len(groups) != 84:
        raise ValueError('fixed_two_case_complete_grid_required')
    inputs, registry_sources = metadata_resolver(rows, args.candidate_run)
    sources = {**sources, **registry_sources, **program_sources()}
    payload = {'schema_version': PLAN_SCHEMA, 'selection_rule': SELECTION_RULE,
        'selected_case_ids': sorted({r['case_id'] for r in rows}),
        'candidate_rows': rows, 'fact_rows': facts, 'dependency_groups': groups,
        'report_inputs': inputs, 'frozen_scope_checker_sha256': FROZEN_GUARD_SHA,
        'modality_source': 'fully_synthetic', 'source_report_or_real_target_supplied': False,
        'report_bodies_opened': False, 'selected_by_scores_or_guard_coverage': False,
        'new_model_calls': 0, 'selection_changed': False, 'regeneration_authorized': False}
    return payload, sources


def read_text_inputs(inputs):
    """Bounded bytes, exact UTF-8 without stripping or newline normalization."""
    texts, unavailable, sources = {}, {}, {}
    verified_files = 0
    for index, item in enumerate(inputs):
        path = require_inside(item['path'], PROTECTED_ROOT, must_exist=True)
        h = item['report_sha256']
        if path.stat().st_size > 32768:
            if h in texts:
                raise ValueError('same_report_hash_has_inconsistent_source_availability')
            unavailable[h] = 'overlength_report_bytes'
            continue
        data = path.read_bytes()
        import hashlib
        if hashlib.sha256(data).hexdigest() != h:
            raise ValueError('sealed_synthetic_report_bytes_changed')
        sources['synthetic_report_bytes_'+str(index)] = path
        verified_files += 1
        if h in texts or h in unavailable:
            continue
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            unavailable[h] = 'invalid_utf8'
            continue
        if not text:
            unavailable[h] = 'empty_report'
        elif len(text) > 8192:
            unavailable[h] = 'overlength_report_characters'
        else:
            texts[h] = text
    return texts, unavailable, sources, verified_files


def flat_csv(records):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    for row in records:
        writer.writerow({key: '' if value is None else str(value).lower() if isinstance(value, bool)
            else json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()})
    return stream.getvalue()


def markdown(summary):
    lines = ['# 冻结报告范围核查 / Frozen report-scope smoke', '',
        f"{summary['fixed_ehr_cases']} fixed EHRs, {summary['image_slots']} CXR slots, "
        f"{summary['candidate_rows']} report candidates, {summary['fact_rows']} fourteen-finding rows.",
        'First two sorted opaque cases, all three CXR paths and all four report experts; no score-based case choice.',
        '仅检查已有合成报告文字；未加载 EHR、真实报告、胸片像素或模型。', '',
        '## Scope decisions / 范围核查结果', '', '| Decision | Candidate/finding rows |', '|---|---:|']
    lines += [f'| {kind} | {count} |' for kind, count in summary['scope_decision_counts_candidate_facts'].items()]
    lines += ['', '## Proxy evidence availability / 代理证据可用性', '',
        '| Edge | Stage | Inventory | Explicit reference | Comparable | Support | Opposition |', '|---|---|---:|---:|---:|---:|---:|']
    for stage in ('raw', 'scoped'):
        for edge, value in summary['relations'][stage].items():
            lines.append(f"| {edge} | {stage} | {value['inventory_facts']} | {value['explicit_reference_facts']} | "
                f"{value['comparable_facts']} | {value['support']} | {value['opposition']} |")
    lines += ['', '- Frozen literal scope covers only cardiomegaly, consolidation, pleural effusion and pneumothorax.',
        '- 14 个 finding 及原始状态完整保留；肺炎、水肿、设备等没有新 scope head，保持 unavailable。',
        '- Scope may retain the SAME model proposal or withdraw it to unknown, never invent or flip a finding.',
        '- 撤回冲突/支持表示弃权或覆盖降低，不是图像/报告变好了，也不是已确认模型错了。',
        '- Report text scope is not image truth, EHR fidelity, clinical factuality or fault localization.',
        '- Unknown is not negative. Shared-image report votes are correlated.',
        '- Source offsets/hashes are preserved without copying report quotes into these outputs.',
        '- No thresholds, ranking, winners, original EHR or generated reports are modified.', '',
        'See `candidate_scope_table.csv`, `scope_fact_table.jsonl` and `scope_dependency_groups.jsonl`.',
        'Further GPU work requires a separately approved complete Slurm script; clinical metrics remain unqualified.']
    return '\n'.join(lines)+'\n'


def run(args):
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise RuntimeError('existing_cpu_slurm_required')
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError('opaque_run_id_required')
    target = require_inside(Path(args.output_root)/args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('refuse_existing_run_before_text_access')
    started = time.monotonic()
    if args.mode == 'prepare':
        payload, sources = prepare(args)
        before = {key: sha256_file(path) for key, path in sources.items()}
        outputs = [('plan.json', payload)]
        schema, metadata_only = PLAN_SCHEMA, True
    else:
        root = require_inside(args.plan_run, PROTECTED_ROOT, must_exist=True)
        mp, pp = root/'manifest.json', root/'plan.json'
        manifest, payload = bounded_json(mp), bounded_json(pp, 2*1024*1024)
        if (manifest['schema_version'] != PLAN_SCHEMA or manifest['metadata_only'] is not True
                or sha256_file(pp) != manifest['artifacts']['plan.json']['sha256']
                or payload['schema_version'] != PLAN_SCHEMA or payload['selection_rule'] != SELECTION_RULE
                or payload['modality_source'] != 'fully_synthetic' or payload['source_report_or_real_target_supplied'] is not False
                or payload['report_bodies_opened'] is not False or payload['selected_by_scores_or_guard_coverage'] is not False
                or payload['new_model_calls'] != 0 or payload['selection_changed'] is not False
                or payload['regeneration_authorized'] is not False):
            raise ValueError('sealed_metadata_only_scope_plan_required')
        sources = {key: require_inside(path, WORKSPACE, must_exist=True) for key, path in manifest['source_paths'].items()}
        if (any(sha256_file(path) != manifest['source_sha256'][key] for key, path in sources.items())
                or any(key not in sources or sources[key] != path.resolve() for key, path in program_sources().items())):
            raise ValueError('frozen_scope_plan_source_or_program_changed')
        rows, facts, groups = payload['candidate_rows'], payload['fact_rows'], payload['dependency_groups']
        if len(rows) != 24 or len(facts) != 336 or len(groups) != 84 or len(payload['report_inputs']) != 24:
            raise ValueError('complete_fixed_two_case_plan_required')
        selected_again, _, _ = select_cases(rows, facts, groups, count=2)
        if selected_again != rows or payload['selected_case_ids'] != sorted({r['case_id'] for r in rows}):
            raise ValueError('fixed_source_case_inventory_changed')
        original_rows, original_facts, original_groups, _ = load_sidecar(sources['source_manifest'].parent)
        if select_cases(original_rows, original_facts, original_groups, count=2) != (rows, facts, groups):
            raise ValueError('plan_not_the_first_two_full_source_cases')
        verified_inputs, _ = metadata_resolver(rows, sources['historical_bank_manifest'].parent)
        if verified_inputs != payload['report_inputs']:
            raise ValueError('plan_report_paths_differ_from_frozen_registry')
        sources.update(plan_manifest=mp, plan=pp)
        before = {key: sha256_file(path) for key, path in sources.items()}
        texts, unavailable, text_sources, verified_files = read_text_inputs(payload['report_inputs'])
        sources.update(text_sources)
        before.update({key: sha256_file(path) for key, path in text_sources.items()})
        scoped, candidates, dependencies, summary = build_scope_audit(rows, facts, groups, texts, scope_check,
            text_unavailable=unavailable)
        summary.update(status='completed_frozen_report_scope_smoke', selection_rule=SELECTION_RULE,
            existing_cpu_job_id=os.environ['SLURM_JOB_ID'], new_slurm_submissions=0,
            verified_synthetic_report_file_slots=verified_files, raw_patient_inputs_opened=False,
            synthetic_report_text_opened=True, image_pixels_opened=False,
            frozen_scope_checker_sha256=FROZEN_GUARD_SHA,
            construction_validation_seconds=round(time.monotonic()-started, 6))
        outputs = [('scope_fact_table.jsonl', scoped), ('candidate_scope_table.csv', candidates),
            ('scope_dependency_groups.jsonl', dependencies), ('summary.json', summary), ('RESULTS_CN_EN.md', markdown(summary))]
        schema, metadata_only = SCHEMA, False
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = []
        for name, value in outputs:
            if name.endswith('.jsonl'):
                files.append(write_private_text(temporary/name, ''.join(json.dumps(row, sort_keys=True)+'\n' for row in value)))
            elif name.endswith('.csv'):
                files.append(write_private_text(temporary/name, flat_csv(value)))
            elif name.endswith('.md'):
                files.append(write_private_text(temporary/name, value))
            else:
                files.append(write_private_json(temporary/name, value))
        if before != {key: sha256_file(path) for key, path in sources.items()}:
            raise ValueError('immutable_sources_changed_during_scope_check')
        write_private_json(temporary/'manifest.json', {'schema_version': schema, 'run_id': args.run_id,
            'metadata_only': metadata_only, 'source_paths': {k: str(p) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in files},
            'new_model_calls': 0, 'selection_changed': False, 'regeneration_authorized': False,
            'primary_metric_eligible': False, 'raw_patient_inputs_opened': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    for name in ('output-root', 'run-id'):
        parser.add_argument('--'+name, required=True)
    for name in ('reliability-run', 'candidate-run', 'plan-run'):
        parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    try:
        target = run(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': 'prepared_scope_smoke' if args.mode == 'prepare' else 'completed_frozen_report_scope_smoke',
        'new_model_calls': 0, 'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
