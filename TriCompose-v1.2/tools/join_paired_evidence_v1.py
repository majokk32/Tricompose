#!/usr/bin/env python3
"""Join authenticated numeric score and text-free conditioning readouts.

Existing cache-only CPU Slurm; no EHR/report/prompt/trace bodies, pixels,
weights, tokenizers, new inference, policy changes or clinical adjudication.
The original failed scoring job stays failed. Input transfer is not truth.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import read_paired_endpoint_cache_v1 as cache
from benchmark_probe_repair_v1 import checked, cpu_guard, csv_text
from contracts import (sha256_file, new_atomic_run, commit_atomic_run,
    discard_atomic_run, write_private_json, write_private_text)
from tricompose_v12.live_workers import check_pins

VERSION = 'tricompose-paired-evidence-join-v1'
SCORE_ROOT = cache.BASE / 'paired_probe_endpoint_readouts/paired2_biovil_12792810_001'
SCORE_SHA = '48b472d148ccffc94ce6e48fbb0d3ed34bac4430fed07a1a824218cba3753c52'
TRACE_ROOT = cache.BASE / 'paired_conditioning_diagnostics/paired_trace_v3_12798919'
TRACE_SHA = 'e91bdd463a3c262da73c9f34a1f7f5c63c3dd06f6a3086e198dc48025dace254'
TRACE_FIELDS = (
    'adapter_token_count', 'attention_token_count', 'padded_token_count',
    'count_check_status', 'source_renderer_version', 'radiographic_id_count',
    'legacy_context_id_count', 'included_phrase_count', 'matched_phrase_count',
    'source_role_mismatch_identified', 'old_combined_guard_would_pass',
    'length_and_hash_checks_pass', 'text_encoder_hook_observed',
)


def join_candidates(scores, lineage, traces):
    """Pure exact-ID join; every existing score/NA cell is preserved verbatim."""
    ids = [r['triple_candidate_id'] for r in scores]
    bindings = {r['triple_candidate_id']: r for r in lineage}
    images = {r['cxr_candidate_id']: r for r in traces}
    if (not scores or len(ids) != len(set(ids)) or len(bindings) != len(lineage)
            or set(ids) != set(bindings) or len(images) != len(traces)):
        raise ValueError('exact_unique_score_lineage_and_trace_inventory_required')
    if {r['cxr_candidate_id'] for r in lineage} != set(images):
        raise ValueError('every_source_image_requires_one_text_free_trace_readout')
    result = []
    for row in scores:
        binding = bindings[row['triple_candidate_id']]
        observed = images[binding['cxr_candidate_id']]
        if (any(str(row[k]) != str(binding[k]) for k in
                ('case_id', 'cxr_model_id', 'seed', 'report_model_id'))
                or row['case_id'] != observed['case_id']
                or str(row['seed']) != str(observed['seed'])
                or binding['cxr_model_id'] != 'roentgen_v2'):
            raise ValueError('same_fixed_case_model_seed_and_candidate_required')
        if (observed['clinical_truth_available'] is not False
                or observed['scores_or_source_inputs_modified'] is not False
                or observed['clinical_context_promoted_to_image_finding'] is not False
                or observed['metadata_projection_is_not_runtime_request'] is not True):
            raise ValueError('diagnostic_readout_must_not_change_inputs_or_assert_truth')
        item = dict(row)
        extras = {'cxr_candidate_id': binding['cxr_candidate_id'],
            'ehr_sha256': binding['ehr_sha256'], 'ehr_facts_sha256': binding['ehr_facts_sha256'],
            'cxr_sha256': binding['cxr_sha256'], 'report_sha256': binding['report_sha256'],
            **{'conditioning_' + k: observed[k] for k in TRACE_FIELDS},
            'conditioning_is_clinical_truth': False,
            'source_gpu_job_state': 'FAILED', 'diagnostic_cpu_job_state': 'COMPLETED',
            'selection_or_primary_acceptance_changed': False}
        if set(item) & set(extras):
            raise ValueError('new_sidecar_columns_cannot_overwrite_original_score_cells')
        item.update(extras)
        result.append(item)
    return result


def run(args):
    cpu_guard()  # Before any metadata open or output creation.
    data = cache.secondary.load_metadata(cache.inputs())
    sources = data['sources']
    def document(root, name, pin):
        return json.loads(checked(root / name, pin, 4 * 1024**2, sources))
    sm = document(SCORE_ROOT, 'manifest.json', SCORE_SHA)
    tm = document(TRACE_ROOT, 'manifest.json', TRACE_SHA)
    if (sm['schema_version'] != cache.VERSION
            or tm['schema_version'] != 'tricompose-paired-conditioning-diagnostic-v3'
            or sm['source_gpu_job_completed_successfully'] is not False
            or any(m['original_selection_changed'] is not False or m['clinical_qualified'] is not False
                for m in (sm, tm))):
        raise ValueError('unchanged_diagnostic_sources_and_failed_gpu_status_required')
    # Compare shared metadata provenance without reopening private trace bodies.
    shared = {str(data['source_root'] / name) for name in
        ('manifest.json', 'score_rows.json', 'choices_and_action_credits.json')}
    for p in shared:
        if sm['sources'].get(p) != sources[p] or tm['sources'].get(p) != sources[p]:
            raise ValueError('score_and_trace_must_share_exact_sealed_source_metadata')
    scores = list(csv.DictReader(io.StringIO(checked(SCORE_ROOT / 'candidate_scores.csv',
        sm['artifacts']['candidate_scores.csv']['sha256'], 4 * 1024**2, sources))))
    traces = document(TRACE_ROOT, 'trace_diagnostics.json',
        tm['artifacts']['trace_diagnostics.json']['sha256'])['records']
    joined = join_candidates(scores, data['rows'], traces)
    if len(joined) != 8 or len(traces) != 4:
        raise ValueError('eight_score_slots_four_images_required')
    methods = list(csv.DictReader(io.StringIO(checked(SCORE_ROOT / 'method_comparison.csv',
        sm['artifacts']['method_comparison.csv']['sha256'], 4 * 1024**2, sources))))
    by_id = {r['triple_candidate_id']: r for r in joined}
    for row in methods:
        selected = by_id[row['selected_candidate_id']]
        if selected['case_id'] != row['case_id']:
            raise ValueError('selected_method_must_retain_same_fixed_case')
        for k, value in selected.items():
            if k.startswith('conditioning_'):
                if k in row: raise ValueError('no_existing_method_cells_overwritten')
                row[k] = value
        row['selection_or_primary_acceptance_changed'] = False
    summary = {'schema_version': VERSION, 'status': 'input_transfer_joined_not_clinical',
        'fixed_ehr_cases': len({r['case_id'] for r in joined}), 'candidate_slots': len(joined),
        'unique_image_candidates': len(traces), 'method_selection_rows': len(methods),
        'length_hash_verified_images': sum(r['length_and_hash_checks_pass'] for r in traces),
        'matched_radiographic_phrases': sum(r['matched_phrase_count'] for r in traces),
        'declared_radiographic_phrases': sum(r['included_phrase_count'] for r in traces),
        'legacy_metadata_role_mismatch_images': sum(r['source_role_mismatch_identified'] for r in traces),
        'primary_score_cells_unchanged': True, 'selection_unchanged': True,
        'source_selection_sha256': data['seal'], 'source_gpu_job_state': 'FAILED',
        'diagnostic_cpu_job_state': 'COMPLETED', 'new_model_calls': 0,
        'source_bodies_pixels_or_weights_read': False, 'clinical_truth_available': False,
        'clinical_repair_success': None, 'clinical_error_localization': None,
        'text_encoder_hook_observed': False,
        'interpretation': 'tokenizer_transfer_checked_clinical_generation_and_reader_errors_not_identified'}
    code_pins = {str(p): sha256_file(p) for p in
        (Path(__file__), ROOT / 'tests/test_paired_evidence_join_v1.py')}
    temporary, target = new_atomic_run(args.output_root, args.run_id)
    try:
        write_private_text(temporary / 'candidate_score_table.csv', csv_text(joined))
        write_private_text(temporary / 'method_comparison.csv', csv_text(methods))
        write_private_json(temporary / 'summary.json', summary)
        if any(sha256_file(p) != pin for p, pin in sources.items()):
            raise ValueError('source_metadata_changed')
        check_pins(code_pins)
        write_private_json(temporary / 'manifest.json', {'schema_version': VERSION,
            'sources': sources, 'code_pins': code_pins,
            'artifacts': {p.name: {'sha256': sha256_file(p)} for p in temporary.iterdir() if p.is_file()},
            'original_selection_changed': False, 'clinical_qualified': False,
            'source_gpu_job_completed_successfully': False})
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', default=str(cache.BASE / 'paired_evidence_tables'))
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    try:
        target, summary = run(args)
        print(json.dumps({'status': summary['status'], 'new_model_calls': 0,
            'manifest_sha256': sha256_file(target / 'manifest.json')}))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
