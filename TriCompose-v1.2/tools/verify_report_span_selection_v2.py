#!/usr/bin/env python3
"""Versioned prompt configuration of the frozen V1 worker; no file changes.

Load a private module instance so V1's on-disk code and all prior runs remain
unchanged. Reuse its guards, staging, model loader, decoder and atomic writes;
only prompt/schema version and separate baseline comparison are configured.
"""
from collections import Counter
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER_V1 = ROOT/'tools/verify_report_span_selection.py'
INTERFACE_V2 = ROOT/'interfaces/report_span_selection_v2.py'
INTERFACE_V1 = ROOT/'interfaces/report_span_selection.py'
spec = importlib.util.spec_from_file_location('tricompose_private_span_v2_worker', WORKER_V1)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
spec = importlib.util.spec_from_file_location('tricompose_private_span_v2_interface', INTERFACE_V2)
interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interface)

# This alters only the fresh module instance, never V1 source/checkpoints/runs.
worker.interface = interface
worker.INTERFACE_PATH = INTERFACE_V2
worker.SCHEMA = 'tricompose-frozen-report-span-selection-v2'
worker.PLAN_SCHEMA = worker.SCHEMA+'-plan'
_sources_v1 = worker.program_sources
_analyze_v1 = worker.analyze
V1_REVIEW = worker.PROTECTED_ROOT/'tricompose_v1_2/verification_runs/span_scope2_12639717'
V1_REVIEW_SHA = '6a0a56a76c04b2f60987cba98b8ef1b82bb999767dd3e98cb9f2ad345319b4d9'


def program_sources():
    return {**_sources_v1(), 'span_v2_driver': Path(__file__),
        'span_v1_inventory_decoder': INTERFACE_V1}


def analyze(args):
    outputs, sources = _analyze_v1(args)
    mp = V1_REVIEW/'manifest.json'
    if worker.sha256_file(mp) != V1_REVIEW_SHA:
        raise ValueError('fixed_span_v1_review_required')
    manifest = worker.bounded_json(mp)
    worker.old.checked_sources(manifest)
    ep = V1_REVIEW/'evidence.json'
    if worker.sha256_file(ep) != manifest['artifacts']['evidence.json']['sha256']:
        raise ValueError('immutable_span_v1_evidence_required')
    baseline = worker.bounded_json(ep, 4*1024*1024)['records']
    current = worker.bounded_json(Path(args.review_run)/'evidence.json', 4*1024*1024)['records']
    by_hash = {r['report_sha256']: r for r in baseline}
    if len(baseline) != 22 or len(by_hash) != 22 or set(by_hash) != {r['report_sha256'] for r in current}:
        raise ValueError('same_fixed_texts_for_span_versions_required')
    summary = next(value for name, value in outputs if name == 'summary.json')
    summary.update(span_v1_complete=sum(r['contract_status'] == 'complete' for r in baseline),
        span_v1_to_v2_status_transitions=dict(Counter(by_hash[r['report_sha256']]['contract_status']+'->'+r['contract_status'] for r in current)),
        complete_responses_with_any_selected_span=sum(r['contract_status'] == 'complete' and
            any(spans for f in r['findings'].values() for spans in f['evidence'].values()) for r in current),
        complete_responses_with_all_heads_unknown=sum(r['contract_status'] == 'complete' and
            all(f['state'] == 'unknown' for f in r['findings'].values()) for r in current),
        decoder_and_segmentation_changed=False, output_schema_semantics_changed=False,
        retrospective_v1_polarity_reconstruction=False, primary_metric_eligible=False)
    for index, (name, value) in enumerate(outputs):
        if name == 'RESULTS_CN_EN.md':
            value += f"\nSpan V1 strict complete: {summary['span_v1_complete']}/22; V2: {summary['complete_distinct_responses']}/22.\n"
            value += 'Only JSON hierarchy instructions/template changed; same source bytes, segmentation, strict decoder and frozen checkpoint. Valid syntax/reference is not semantic correctness.\n'
            outputs[index] = (name, value)
    sources.update(span_v1_review_manifest=mp, span_v1_evidence=ep)
    return outputs, sources


worker.program_sources = program_sources
worker.analyze = analyze


if __name__ == '__main__':
    raise SystemExit(worker.main())
