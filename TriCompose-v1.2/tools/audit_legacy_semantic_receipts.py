#!/usr/bin/env python3
"""Cache-only audit of exact frozen semantic receipts; no model/score changes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT.parent/'TriCompose-v1.0/eval/report_v1_1'))
from contracts import (PROTECTED_ROOT, require_inside, sha256_file, new_atomic_run,
    commit_atomic_run, discard_atomic_run, write_private_json, write_private_text)
from verify_legacy_report_semantics import load_plan, checked_sources
from check_legacy_report_scope import bounded_json, read_text_inputs
from verify_report_evidence_qwen import decode_evidence, digest_text
from repair_cached_report_evidence import decode_aligned, ALIGNMENT_VERSION

SCHEMA = 'tricompose-legacy-semantic-receipt-audit-v1'
REVIEW_SHA = 'd543ac0287b40bc1596944f348a1780b76c742b60a5cdc7decaddc8d2352dca4'
COMPARISON_SHA = '86052b941f55749a16eed660c7bf33e2d086ce43741e4208ef82d67d7741bc6e'


def audit_response(record, raw, text):
    if (record['report_sha256'] != digest_text(text)
            or raw['report_sha256'] != record['report_sha256']
            or digest_text(raw['response']) != record['response_sha256']
            or any(record[k] != raw[k] for k in ('input_tokens', 'output_tokens', 'token_limit_reached'))):
        raise ValueError('exact_source_response_receipt_required')
    strict = decode_evidence(raw['response'], text, token_limit_reached=raw['token_limit_reached'])
    if any(record[k] != value for k, value in strict.items()):
        raise ValueError('original_strict_receipt_reparse_mismatch')
    aligned = decode_aligned(raw['response'], text, token_limit_reached=raw['token_limit_reached'])
    spans = []
    for finding, value in aligned['findings'].items():
        for polarity, evidence in value['evidence'].items():
            for span in evidence:
                quote = text[span['char_start']:span['char_end']]
                if quote != span['quote'] or digest_text(quote) != span['quote_sha256']:
                    raise ValueError('unaligned_original_source_span')
                spans.append({'finding': finding, 'polarity': polarity,
                    **{k: span[k] for k in ('char_start', 'char_end', 'quote_sha256', 'offset_unit', 'alignment_mode')}})
    return {'report_sha256': record['report_sha256'], 'response_sha256': record['response_sha256'],
        'strict_contract_status': strict['contract_status'], 'strict_failure_reason': strict['contract_failure_reason'],
        'whitespace_contract_status': aligned['contract_status'], 'whitespace_failure_reason': aligned['contract_failure_reason'],
        'strict_states': {k: v['state'] for k, v in strict['findings'].items()},
        'whitespace_states': {k: v['state'] for k, v in aligned['findings'].items()},
        'aligned_source_offsets_and_hashes': spans, 'clinical_validation': False,
        'selection_changed': False, 'regeneration_authorized': False}


def run(args):
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise RuntimeError('existing_cpu_slurm_required')
    target = require_inside(Path(args.output_root)/args.run_id, PROTECTED_ROOT, must_exist=False)
    if target.exists():
        raise FileExistsError('refuse_existing_audit_before_text_access')
    plan, sources = load_plan(args.plan_run)
    for label, run, expected in (('review', args.review_run, REVIEW_SHA),
            ('comparison', args.comparison_run, COMPARISON_SHA)):
        root = require_inside(run, PROTECTED_ROOT, must_exist=True)
        if sha256_file(root/'manifest.json') != expected:
            raise ValueError('fixed_sealed_results_required')
        manifest = bounded_json(root/'manifest.json')
        checked_sources(manifest)
        sources[label+'_manifest'] = root/'manifest.json'
        for name, item in manifest['artifacts'].items():
            path = require_inside(root/name, root, must_exist=True)
            if sha256_file(path) != item['sha256']:
                raise ValueError('sealed_result_artifact_changed')
            sources[label+'_'+name] = path
    review_root = Path(args.review_run)
    evidence = bounded_json(review_root/'evidence.json', 4*1024*1024)
    raw = bounded_json(review_root/'raw_responses.json', 2*1024*1024)['records']
    by_hash = {r['report_sha256']: r for r in raw}
    if len(by_hash) != 22 or len(raw) != 22 or len(evidence['records']) != 22:
        raise ValueError('same_22_frozen_model_receipts_required')
    texts, unavailable, text_sources, verified = read_text_inputs(plan['report_inputs'])
    if unavailable or set(texts) != set(by_hash):
        raise ValueError('same_exact_22_synthetic_texts_required')
    sources.update(text_sources, audit_program=Path(__file__))
    before = {key: sha256_file(path) for key, path in sources.items()}
    records = [audit_response(r, by_hash[r['report_sha256']], texts[r['report_sha256']])
        for r in evidence['records']]
    transitions = Counter(r['strict_contract_status']+'->'+r['whitespace_contract_status'] for r in records)
    summary = {'schema_version': SCHEMA, 'candidate_slots': 24, 'distinct_texts': 22,
        'historical_model_calls': 22, 'new_model_calls': 0, 'new_slurm_submissions': 0,
        'alignment_version': ALIGNMENT_VERSION, 'verified_synthetic_report_file_slots': verified,
        'strict_complete': sum(r['strict_contract_status'] == 'complete' for r in records),
        'whitespace_complete': sum(r['whitespace_contract_status'] == 'complete' for r in records),
        'transitions': dict(transitions),
        'remaining_failure_reasons': dict(Counter(r['whitespace_failure_reason'] for r in records
            if r['whitespace_contract_status'] != 'complete')),
        'quote_alignment_modes': dict(Counter(span['alignment_mode'] for r in records
            for span in r['aligned_source_offsets_and_hashes'])),
        'model_execution': evidence['execution'], 'peak_allocated_vram_gib': evidence['peak_allocated_vram_gib'],
        'raw_patient_inputs_opened': False, 'image_pixels_opened': False,
        'quote_or_response_copied_to_output': False, 'selection_changed': False,
        'clinical_accuracy': None, 'primary_metric_eligible': False, 'regeneration_authorized': False,
        'interpretation': 'Technical cached-interface diagnostic only. Whitespace-recovered evidence is not a clinical correction or independent semantic truth. Original strict receipts and all old scores/winners remain unchanged.'}
    report = '# 语义核查回执诊断 / Semantic receipt diagnostic\n\n'
    report += '24 report slots / 22 distinct exact texts. GPU job 12639326 completed on L40S in 1m56s.\n\n'
    report += f"Strict complete: {summary['strict_complete']}/22; whitespace-only diagnostic complete: {summary['whitespace_complete']}/22.\n\n"
    report += 'Only one unavailable response is recoverable by the existing whitespace-only aligner.\n\n'
    report += '| Remaining technical failure | Distinct responses |\n|---|---:|\n'
    for reason, count in sorted(summary['remaining_failure_reasons'].items()):
        report += f'| {reason} | {count} |\n'
    report += '\n'+summary['interpretation']+'\n\n'
    report += '下一步建议：另立协议让模型引用预先编号的原文片段，不重新抄写引文。片段引用有效仍不证明语义正确；不得用此诊断改旧分数或排名。\n'
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        files = [write_private_json(temporary/'audit.json', {'schema_version': SCHEMA, 'records': records}),
            write_private_json(temporary/'summary.json', summary), write_private_text(temporary/'RESULTS_CN_EN.md', report)]
        if before != {key: sha256_file(path) for key, path in sources.items()}:
            raise ValueError('immutable_receipts_changed_during_audit')
        write_private_json(temporary/'manifest.json', {'schema_version': SCHEMA, 'run_id': args.run_id,
            'source_paths': {k: str(p) for k, p in sources.items()}, 'source_sha256': before,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in files}, 'new_model_calls': 0,
            'selection_changed': False, 'primary_metric_eligible': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan-run', 'review-run', 'comparison-run', 'output-root', 'run-id'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    try:
        target = run(args)
    except Exception as exc:
        print(json.dumps({'status': 'failed_closed', 'error_type': type(exc).__name__}))
        return 2
    print(json.dumps({'status': 'completed_cache_only_audit', 'manifest_sha256': sha256_file(target/'manifest.json'),
        'new_model_calls': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
