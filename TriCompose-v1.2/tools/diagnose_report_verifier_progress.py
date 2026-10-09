#!/usr/bin/env python3
"""Metadata-only development diagnostic; never reads reports, images or gold rows.

Append report-check progress to an immutable request preview. A completed
extractor review is not a clinically resolved request, score or repair action.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (CHEXPERT_FINDINGS, PROTECTED_ROOT, commit_atomic_run,
    discard_atomic_run, new_atomic_run, require_inside, sha256_file,
    write_private_json, write_private_text)

SCHEMA = 'tricompose-report-verifier-progress-diagnostic-v1'
BASE = PROTECTED_ROOT / 'tricompose_v1_2'
HEADS = ('cardiomegaly', 'consolidation', 'pleural_effusion', 'pneumothorax')
STATES = ('positive', 'negative', 'uncertain', 'unknown')
EXPLICIT = frozenset(('positive', 'negative'))
PARENTS = {
    'manual': (BASE/'real_validation/official_span_v2_12645961',
        'b9274424e49301b8e41e147716a0e751cf3743fc5f1a032f91a4dc1b7baf4b5b', 'summary.json'),
    'synthetic': (BASE/'verification_runs/span_v2_scope2_12645404_analysis',
        '4defd00b806b3187bc9657521bfae22b6b63e55e7d0b86812e4846344e67121b', 'assertion_comparison.jsonl'),
    'requests': (BASE/'reliability_evidence_previews/evidence_pool960_12632006_001',
        'b38227d1e42bd36e90718558f837a5f442d4cf84bde2795f43b13050db629a9f', 'evidence_requests.jsonl'),
}
FACT_FIELDS = frozenset(('chexbert_qwen_comparison', 'clinical_selection_score',
    'confirmed_faulty_modality', 'finding', 'independent_clinical_validation',
    'literal_scope_decision', 'literal_scope_state', 'qwen_assertion_state',
    'qwen_contract_failure_reason', 'qwen_contract_status', 'raw_chexbert_state',
    'regeneration_authorized', 'report_candidate_id', 'report_sha256',
    'selection_changed', 'triple_candidate_id'))
REQUEST_FIELDS = frozenset(('blocked_by_declared_budget', 'case_id',
    'clinical_truth_established', 'consumer_candidate_ids', 'dependency_hashes',
    'estimated_gpu_seconds', 'estimated_model_calls', 'execution_status', 'finding',
    'model_execution_allowed', 'reason_codes', 'request_id', 'request_kind',
    'source_evidence_ids'))


def state_transition(reference, prediction):
    """Directional reference-policy difference, not a diagnosis of its cause."""
    if reference not in STATES or prediction not in STATES:
        raise ValueError('valid_four_states_required')
    if reference == prediction:
        return ('same_explicit_state' if reference in EXPLICIT else
            'same_uncertain_state' if reference == 'uncertain' else 'same_unmentioned_state')
    if reference in EXPLICIT and prediction in EXPLICIT:
        return 'hard_polarity_flip'
    if prediction == 'unknown':
        return 'annotated_to_unmentioned'
    if reference == 'unknown':
        return 'unmentioned_to_determinate' if prediction in EXPLICIT else 'unmentioned_to_uncertain'
    if reference == 'uncertain':
        return 'uncertain_to_determinate'
    return 'determinate_to_uncertain'


def manual_transitions(summary):
    rows = []
    for extractor in ('chexbert', 'qwen_v2'):
        tables = summary['paired_comparison'][extractor]
        if set(tables) != set(HEADS):
            raise ValueError('fixed_manual_heads_required')
        for finding in HEADS:
            matrix = tables[finding]['confusion_matrix']
            if set(matrix) != set(STATES) or any(set(row) != set(STATES) for row in matrix.values()):
                raise ValueError('complete_four_state_matrix_required')
            for reference in STATES:
                for prediction in STATES:
                    count = matrix[reference][prediction]
                    if type(count) is not int or count < 0:
                        raise ValueError('nonnegative_integer_count_required')
                    rows.append({'extractor': extractor, 'finding': finding,
                        'reference_state': reference, 'predicted_state': prediction,
                        'count': count, 'transition': state_transition(reference, prediction)})
            if sum(sum(r.values()) for r in matrix.values()) != tables[finding]['checks']:
                raise ValueError('manual_check_count_mismatch')
    return rows


def fact_status(row):
    """Symmetric model disagreement, never treats either extractor as truth."""
    if row['finding'] not in HEADS:
        if row['qwen_contract_status'] != 'outside_scope_inventory':
            raise ValueError('outside_scope_contract_required')
        return 'outside_verifier_scope'
    if row['qwen_contract_status'] == 'failed_unavailable':
        return 'verifier_unavailable'
    if row['qwen_contract_status'] != 'complete':
        raise ValueError('supported_head_contract_required')
    left, right = row['raw_chexbert_state'], row['qwen_assertion_state']
    if left not in STATES or right not in STATES:
        raise ValueError('valid_four_states_required')
    if left in EXPLICIT and right in EXPLICIT:
        return ('explicit_agreement_unqualified' if left == right else
            'explicit_polarity_disagreement_unqualified')
    if left == right == 'unknown':
        return 'both_unmentioned'
    if 'uncertain' in (left, right):
        return 'uncertainty_not_contradiction'
    return 'single_extractor_assertion_unqualified'


def annotate_facts(rows):
    output, lookup, grouped = [], {}, defaultdict(list)
    for row in rows:
        if set(row) != FACT_FIELDS or row['finding'] not in CHEXPERT_FINDINGS:
            raise ValueError('quote_free_fact_schema_required')
        if (row['independent_clinical_validation'] is not False
                or row['regeneration_authorized'] is not False or row['selection_changed'] is not False
                or row['clinical_selection_score'] is not None or row['confirmed_faulty_modality'] is not None):
            raise ValueError('unqualified_unchanged_source_required')
        key = (row['triple_candidate_id'], row['finding'])
        if key in lookup:
            raise ValueError('unique_candidate_finding_required')
        annotated = {**row, 'reportcheck_status': fact_status(row),
            'reportcheck_clinical_truth_established': False,
            'reportcheck_primary_metric_eligible': False}
        output.append(annotated)
        lookup[key] = annotated
        grouped[row['triple_candidate_id']].append(annotated)
    candidates = []
    statuses = ('explicit_agreement_unqualified', 'explicit_polarity_disagreement_unqualified',
        'uncertainty_not_contradiction', 'single_extractor_assertion_unqualified',
        'both_unmentioned', 'verifier_unavailable', 'outside_verifier_scope')
    for candidate_id, facts in sorted(grouped.items()):
        if ({r['finding'] for r in facts} != set(CHEXPERT_FINDINGS)
                or len({(r['report_candidate_id'], r['report_sha256']) for r in facts}) != 1):
            raise ValueError('complete_candidate_inventory_and_lineage_required')
        counts = Counter(r['reportcheck_status'] for r in facts)
        candidates.append({'triple_candidate_id': candidate_id,
            'report_candidate_id': facts[0]['report_candidate_id'],
            'report_sha256': facts[0]['report_sha256'], 'fact_rows': len(facts),
            **{s: counts[s] for s in statuses}, 'clinical_selection_score': None,
            'clinical_truth_established': False, 'regeneration_authorized': False})
    return output, candidates, lookup


def request_progress(rows, lookup):
    output, seen = [], set()
    for row in rows:
        if set(row) != REQUEST_FIELDS or row['request_id'] in seen:
            raise ValueError('fixed_unique_request_schema_required')
        seen.add(row['request_id'])
        if (row['execution_status'] != 'not_executed' or row['clinical_truth_established'] is not False
                or row['model_execution_allowed'] is not False):
            raise ValueError('unexecuted_preview_required')
        matched = []
        status = 'not_covered_by_this_report_check'
        # Never apply report-only evidence to an image/relation request, nor
        # expand the fixed cohort to other candidate IDs with identical text.
        if row['request_kind'] == 'verify_report_assertion':
            for candidate_id in row['consumer_candidate_ids']:
                fact = lookup.get((candidate_id, row['finding']))
                if fact is not None:
                    if fact['report_sha256'] != row['dependency_hashes'].get('report_sha256'):
                        raise ValueError('exact_report_dependency_hash_required')
                    matched.append(fact)
            if matched:
                statuses = {r['reportcheck_status'] for r in matched}
                if statuses == {'outside_verifier_scope'}:
                    status = 'outside_verifier_scope'
                elif statuses == {'verifier_unavailable'}:
                    status = 'verifier_unavailable'
                elif 'outside_verifier_scope' in statuses or 'verifier_unavailable' in statuses:
                    status = 'partial_review_unqualified'
                else:
                    status = 'technical_review_complete_clinically_unqualified'
        output.append({**row, 'reportcheck_progress': status,
            'reportcheck_matched_consumer_ids': sorted({r['triple_candidate_id'] for r in matched}),
            'reportcheck_fact_status_counts': dict(sorted(Counter(r['reportcheck_status'] for r in matched).items())),
            'reportcheck_clinically_resolved': False, 'reportcheck_new_model_calls': 0})
    return output


def csv_text(rows, fields=None):
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=list(fields or rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({name: json.dumps(row[name], sort_keys=True) if isinstance(row[name], (dict, list))
            else row[name] for name in writer.fieldnames})
    return handle.getvalue()


def jsonl_text(rows):
    return ''.join(json.dumps(row, sort_keys=True, ensure_ascii=True)+'\n' for row in rows)


def load_fixed_inputs():
    inputs, sources = {}, {'worker': Path(__file__), 'atomic_contracts': Path(sys.modules['contracts'].__file__)}
    for label, (root, expected, name) in PARENTS.items():
        manifest_path = require_inside(root/'manifest.json', PROTECTED_ROOT, must_exist=True)
        if sha256_file(manifest_path) != expected:
            raise ValueError('fixed_parent_manifest_required')
        manifest = json.loads(manifest_path.read_text())
        artifact = require_inside(root/name, root, must_exist=True)
        if sha256_file(artifact) != manifest['artifacts'][name]['sha256']:
            raise ValueError('fixed_consumed_artifact_required')
        if artifact.stat().st_size > 16*1024*1024:
            raise ValueError('bounded_metadata_required')
        # No parent source_paths traversal: some ancestors include source text.
        inputs[label] = ([json.loads(line) for line in artifact.read_text().splitlines()]
            if name.endswith('.jsonl') else json.loads(artifact.read_text()))
        sources[label+'_manifest'], sources[label+'_artifact'] = manifest_path, artifact
    return inputs, sources


def markdown(summary):
    manual = summary['manual_reference_transition_counts']['qwen_v2']
    synthetic = summary['synthetic_fact_status_counts']
    progress = summary['request_progress_counts']
    return f'''# Report verifier progress / 报告核查进度

## 人工报告参考：状态提取诊断

仅重用之前批准的任务的 aggregate confusion matrices，不打开原始报告或参考行。
15 份报告、4 个头、60 次检查；26 个已标注状态中匹配 20 个。
正负反转 {manual.get('hard_polarity_flip', 0)} 个；明确状态变 uncertain
{manual.get('determinate_to_uncertain', 0)} 个；未提及变明确状态
{manual.get('unmentioned_to_determinate', 0)} 个；未提及变 uncertain
{manual.get('unmentioned_to_uncertain', 0)} 个。
后两类是参考策略差异，不是独立裁定的 hallucination。
该样本已用于开发，不能用于 held-out qualification 或候选置信度赋值。

## Synthetic 候选：模型间分歧，不是临床错误定位

固定 {summary['candidate_slots']} 个 triple slots、{summary['distinct_report_hashes']} 个报告 hash，
保留全部 {summary['fact_rows']} 个候选/finding 行。两个提取器明确判断一致
{synthetic.get('explicit_agreement_unqualified', 0)} 个，正负相反
{synthetic.get('explicit_polarity_disagreement_unqualified', 0)} 个；
单个提取器给出明确判断 {synthetic.get('single_extractor_assertion_unqualified', 0)} 个；
双方未提及 {synthetic.get('both_unmentioned', 0)} 个。
Unknown 不等于阴性，uncertain 不算正负矛盾；核查不可用与成功的 unknown 分开。
四头以外的 10 个 finding 保留为 outside scope，不伪装成覆盖。

## 绑定原始 evidence requests

保留全部 {summary['logical_requests']} 个逻辑请求及其原始 execution_status。
{progress.get('technical_review_complete_clinically_unqualified', 0)} 个报告请求有技术核查结果，
{progress.get('outside_verifier_scope', 0)} 个命中候选但 finding 在范围外。
只匹配固定候选 ID、finding 和精确 report hash，不借相同文本扩展病例，
不把纯报告核查当作 image/report relation 的证据。
临床解决请求数为 0；这不是新执行 {summary['logical_requests']} 次模型调用。

## Files / 文件

- `manual_reference_transitions.csv`: aggregate reference→prediction 4×4 matrices.
- `report_verifier_fact_table.jsonl`: all original fact cells plus diagnostic states.
- `report_verifier_candidate_table.csv`: per-slot status counts; no primary score.
- `evidence_request_progress.jsonl/csv`: original request history plus technical progress.

## Boundaries / 边界

No new model/GPU/API/Slurm call, training, threshold fitting, ranking, rejection,
clinical acceptance or regeneration. No real rows/report text/images opened.
The manual diagnostic and synthetic disagreements are separate populations;
76.92% is not a synthetic candidate probability. Agreement is not truth and
report-only reviews do not identify whether an image or report is faulty.
The next useful experiment must measure validity of a specific evidence edge,
with an independently evaluated endpoint; not simply run the same scorer until
its own score improves. Any new GPU submission requires separate approval.
'''


def execute(output_root, run_id):
    temporary, target = new_atomic_run(output_root, run_id)
    try:
        inputs, sources = load_fixed_inputs()
        source_hashes = {key: sha256_file(path) for key, path in sources.items()}
        transitions = manual_transitions(inputs['manual'])
        facts, candidates, lookup = annotate_facts(inputs['synthetic'])
        requests = request_progress(inputs['requests'], lookup)
        if (len(facts), len(candidates), len(requests)) != (336, 24, 2072):
            raise ValueError('fixed_cohort_inventory_required')
        manual_counts = {}
        for name in ('chexbert', 'qwen_v2'):
            counts = Counter()
            for row in transitions:
                if row['extractor'] == name:
                    counts[row['transition']] += row['count']
            manual_counts[name] = dict(sorted(counts.items()))
        summary = {'schema_version': SCHEMA, 'development_diagnostic_not_preregistered': True,
            'candidate_slots': len(candidates), 'distinct_report_hashes': len({r['report_sha256'] for r in facts}),
            'fact_rows': len(facts), 'logical_requests': len(requests),
            'manual_reference_transition_counts': manual_counts,
            'synthetic_fact_status_counts': dict(sorted(Counter(r['reportcheck_status'] for r in facts).items())),
            'request_progress_counts': dict(sorted(Counter(r['reportcheck_progress'] for r in requests).items())),
            'clinically_resolved_requests': 0, 'new_model_calls': 0, 'external_api_used': False,
            'raw_patient_inputs_or_real_targets_opened': False, 'reference_rows_exported': False,
            'primary_metric_eligible': False, 'selection_changed': False, 'regeneration_authorized': False,
            'fixed_ehr_changed': False, 'thresholds_or_weights_fitted': False,
            'manual_accuracy_transferred_to_candidates': False}
        write_private_text(temporary/'manual_reference_transitions.csv', csv_text(transitions))
        write_private_text(temporary/'report_verifier_fact_table.jsonl', jsonl_text(facts))
        write_private_text(temporary/'report_verifier_candidate_table.csv', csv_text(candidates))
        write_private_text(temporary/'evidence_request_progress.jsonl', jsonl_text(requests))
        skinny_fields = ('request_id', 'request_kind', 'finding', 'execution_status', 'reportcheck_progress',
            'reportcheck_matched_consumer_ids', 'reportcheck_fact_status_counts', 'reportcheck_clinically_resolved')
        write_private_text(temporary/'evidence_request_progress.csv', csv_text(requests, skinny_fields))
        write_private_json(temporary/'summary.json', summary)
        write_private_text(temporary/'RESULTS_CN_EN.md', markdown(summary))
        if any(sha256_file(path) != source_hashes[key] for key, path in sources.items()):
            raise ValueError('immutable_consumed_sources_changed')
        manifest = {'schema_version': SCHEMA, 'run_id': run_id,
            'source_paths': {key: str(path.resolve()) for key, path in sources.items()},
            'source_sha256': source_hashes,
            'artifacts': {path.name: {'sha256': sha256_file(path)} for path in sorted(temporary.iterdir())},
            'new_model_calls': 0, 'primary_metric_eligible': False, 'selection_changed': False,
            'regeneration_authorized': False, 'patient_keys_reports_or_reference_rows_written': False}
        write_private_json(temporary/'manifest.json', manifest)
        for path in [temporary, *temporary.iterdir()]:
            stat = path.stat()
            if stat.st_gid not in (96293, 65534) or stat.st_mode & 0o7777 != (0o2770 if path.is_dir() else 0o660):
                raise ValueError('protected_project_group_permissions_required')
        commit_atomic_run(temporary, target)
        return target, summary
    except Exception:
        discard_atomic_run(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE/'report_verifier_progress')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    target, summary = execute(args.output_root, args.run_id)
    print(json.dumps({'status': 'metadata_diagnostic_complete', 'output_root': str(target),
        'candidate_slots': summary['candidate_slots'], 'fact_rows': summary['fact_rows'],
        'logical_requests': summary['logical_requests'], 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
