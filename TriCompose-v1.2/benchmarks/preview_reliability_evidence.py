#!/usr/bin/env python3
"""CPU-only evidence-request preview of the sealed 960-row reliability sidecar."""
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
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, require_inside, read_json, sha256_file,
    new_atomic_run, commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.reliability_preview import SCHEMA, SOURCE_SCHEMA, Budget, preview_reliability

SOURCE_MANIFEST_SHA = 'd057efecd06a4db11c6fec27e79f2ecf085b1c7afe0fa7174fce554c8e85b673'


def load_sidecar(root):
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise RuntimeError('existing_cpu_slurm_required')
    root = require_inside(root, PROTECTED_ROOT, must_exist=True)
    mp = root / 'manifest.json'
    if sha256_file(mp) != SOURCE_MANIFEST_SHA:
        raise ValueError('sealed_reliability_manifest_changed')
    manifest = read_json(mp)
    if (manifest['schema_version'] != SOURCE_SCHEMA or manifest['new_model_calls'] != 0
            or manifest['original_selection_changed'] is not False
            or manifest['thresholds_or_weights_changed'] is not False
            or manifest['regeneration_authorized'] is not False
            or manifest['primary_metric_eligible'] is not False):
        raise ValueError('immutable_diagnostic_sidecar_required')
    paths = {'source_manifest': mp}
    for name, cap in (('candidate_score_table.csv', 4*1024*1024),
                      ('fact_reliability.jsonl', 32*1024*1024),
                      ('report_dependency_groups.jsonl', 4*1024*1024)):
        path = require_inside(root/name, root, must_exist=True)
        if path.stat().st_size > cap or sha256_file(path) != manifest['artifacts'][name]['sha256']:
            raise ValueError('sidecar_artifact_changed_or_unbounded')
        paths[name] = path
    with paths['candidate_score_table.csv'].open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    facts = [json.loads(line) for line in paths['fact_reliability.jsonl'].read_text().splitlines() if line]
    groups = [json.loads(line) for line in paths['report_dependency_groups.jsonl'].read_text().splitlines() if line]
    if (len(rows) != 960 or len(facts) != 13440 or len(groups) != 3360
            or len({r['case_id'] for r in rows}) != 80
            or len({(r['case_id'], r['cxr_candidate_id']) for r in rows}) != 240):
        raise ValueError('complete_fixed_full_bank_required')
    case_counts = {}
    image_counts = {}
    for row in rows:
        case_counts[row['case_id']] = case_counts.get(row['case_id'], 0) + 1
        image = (row['case_id'], row['cxr_candidate_id'])
        image_counts[image] = image_counts.get(image, 0) + 1
    if set(case_counts.values()) != {12} or set(image_counts.values()) != {4}:
        raise ValueError('complete_three_image_four_report_grid_required')
    if (sha256_file(mp) != SOURCE_MANIFEST_SHA or any(
            sha256_file(path) != manifest['artifacts'][name]['sha256']
            for name, path in paths.items() if name != 'source_manifest')):
        raise ValueError('sealed_sidecar_changed_during_read')
    return rows, facts, groups, paths


def render_csv(decisions):
    fields = ('case_id', 'triple_candidate_id', 'preview_action', 'explicit_ehr_reference_facts',
              'reason_codes', 'request_ids', 'edge_evidence_status', 'proxy_pattern_counts',
              'clinical_selection_score', 'confirmed_faulty_modality', 'case_rejected',
              'model_execution_allowed', 'regeneration_authorized', 'selection_changed')
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for decision in decisions:
        writer.writerow({key: '' if decision[key] is None else
            str(decision[key]).lower() if isinstance(decision[key], bool) else
            json.dumps(decision[key], sort_keys=True) if isinstance(decision[key], (dict, list)) else decision[key]
            for key in fields})
    return stream.getvalue()


def markdown(summary):
    lines = ['# 保守证据请求预演 / Conservative evidence-request preview', '',
        f"{summary['fixed_ehr_cases']} fixed EHRs; {summary['image_slots']} CXR slots; "
        f"{summary['candidate_rows']} candidate triples; {summary['fact_rows']} cached finding rows.", '',
        '不改 EHR、不拒绝病例、不排名、不重新生成；这不是已验证的错误定位或修复实验。',
        'No EHR changes, rejected cases, ranking, model execution, selection changes or repair.', '',
        '## Preview actions / 预演动作', '', '| Action | Candidate rows |', '|---|---:|']
    lines += [f'| {kind} | {count} |' for kind, count in summary['action_counts'].items()]
    lines += ['', '## Deduplicated logical requests / 去重后的逻辑证据请求', '',
        '| Request kind | Requests |', '|---|---:|']
    lines += [f'| {kind} | {count} |' for kind, count in summary['request_counts_by_kind'].items()]
    lines += ['', f"Candidate/request links: {summary['candidate_request_links']}; "
        f"deduplicated logical requests: {summary['deduplicated_logical_evidence_requests']}.",
        '这些是 finding-level 逻辑请求，不是模型调用次数；不能由此宣称节省了 GPU 时间。',
        'Logical finding requests are NOT model invocations. Batching, assets and measured cost are not specified.', '',
        f"Fixed cases without explicit cached EHR comparison facts: {summary['fixed_ehr_cases_without_explicit_facts']}.",
        '这不等于没有临床内容或生成无效；固定 EHR 保留，不能为了补证据添加疾病/设备。', '',
        '## Boundaries / 使用边界', '',
        '- Unknown/uncertain never become negative; missing comparisons are not clinical agreement.',
        '- EHR=image≠report is a proxy report-verification signal, not a confirmed report error.',
        '- EHR=report≠image is a proxy image-verification signal; reports share the image and are not independent truth.',
        '- Cached report-proposal disagreement requests verification rather than majority voting.',
        '- BioViL raw cosine and RSUA benchmark results never override these patterns or rank candidates.',
        '- No additional budget or verification cost is invented. All requests remain not executed.',
        '- No conditional repair success, localization accuracy or clinical acceptance claim follows.', '',
        '## Files / 文件', '',
        '`decision_preview.csv` / `decisions.jsonl`: candidate reasons, evidence IDs and fixed artifact hashes.',
        '`evidence_requests.jsonl`: dependency-deduplicated requests and all consuming candidates.',
        '`summary.json` / `manifest.json`: counts, boundaries and immutable source/output hashes.', '',
        'Further model work requires its own frozen protocol and separately approved complete Slurm script.']
    return '\n'.join(lines)+'\n'


def run(args):
    started = time.monotonic()
    rows, facts, groups, paths = load_sidecar(args.reliability_run)
    calls, seconds = args.max_additional_calls, args.max_additional_gpu_seconds
    if (calls is None) != (seconds is None):
        raise ValueError('both_budget_limits_or_neither_required')
    budget = None if calls is None else Budget(calls, seconds)
    sources = {**paths, 'program': Path(__file__),
        'preview_contract': ROOT/'src/tricompose_v12/reliability_preview.py',
        'reliability_contract': ROOT/'src/tricompose_v12/scorer_reliability.py',
        'budget_contract': ROOT/'src/tricompose_v12/decision_preview.py',
        'edge_contract': ROOT/'src/tricompose_v12/invariant_verification.py',
        'legacy_profile_contract': ROOT/'src/tricompose_v12/legacy_replay_adapter.py'}
    before = {key: sha256_file(path) for key, path in sources.items()}
    decisions, requests, summary = preview_reliability(rows, facts, groups, budget=budget)
    summary.update(status='completed_conservative_evidence_preview',
        new_slurm_submissions=0, existing_cpu_job_id=os.environ['SLURM_JOB_ID'],
        raw_patient_inputs_opened=False, report_bodies_or_image_pixels_opened=False,
        construction_validation_seconds=round(time.monotonic()-started, 6))
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_text(temporary/'decisions.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in decisions)),
            write_private_text(temporary/'decision_preview.csv', render_csv(decisions)),
            write_private_text(temporary/'evidence_requests.jsonl', ''.join(json.dumps(r, sort_keys=True)+'\n' for r in requests)),
            write_private_json(temporary/'summary.json', summary),
            write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))]
        if before != {key: sha256_file(path) for key, path in sources.items()}:
            raise ValueError('immutable_sources_changed_during_preview')
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA, 'run_id': args.run_id,
            'source_paths': {key: str(path) for key, path in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in files},
            'new_model_calls': 0, 'selection_changed': False, 'fixed_ehr_changed': False,
            'cases_rejected': 0, 'regeneration_authorized': False, 'primary_metric_eligible': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reliability-run', 'output-root', 'run-id'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--max-additional-calls', type=int)
    parser.add_argument('--max-additional-gpu-seconds', type=float)
    args = parser.parse_args(argv)
    try:
        target, summary = run(args)
    except Exception as error:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(error).__name__}))
        return 2
    print(json.dumps({'status': summary['status'], 'candidate_rows': summary['candidate_rows'],
        'new_model_calls': 0, 'manifest_sha256': sha256_file(target/'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
