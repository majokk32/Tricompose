"""Independent counts/bindings of a sealed reference inventory, not semantics.

Reads derived states/coordinates/hashes only. Never reopens the clinical dev
file; source provenance is the completed approved job's sealed receipt.
"""
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time

from contracts import new_atomic_run, commit_atomic_run, discard_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json

BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'
RUN = BASE / 'manual_opacity_reference_inventories/released_dev_12787886_001'
EXPECTED = 'c698dc6c047090142633c886219d98064ed049428fe8bd77d110b4eb191a7dd5'
STATES = ('positive', 'negative', 'uncertain', 'unknown')
RAW = 'artifacts/protected/tricompose_v1_2/reference_datasets/cxrgraph_source_12766754_001/source/manual_data/dev.json'


def replay(records, previous_hashes):
    require(isinstance(records, list) and records and len(records) <= 1024, 'all_bounded_inventory_records_required')
    states, failures, hashes = Counter(), Counter(), Counter()
    mentions = bindings = unannotated = with_mentions = with_unannotated = complete = 0
    for i, r in enumerate(records):
        require(set(r) == {'report_id', 'source_index', 'source_sha256', 'status', 'failure_type', 'projection'}
                and r['report_id'] == f'report_{i:04d}' and r['source_index'] == i,
                'exact_opaque_release_order_and_nontext_fields_required')
        h = r['source_sha256']
        require(h is None or re.fullmatch(r'[a-f0-9]{64}', h), 'source_hash_not_original_key_required')
        if h:
            hashes[h] += 1
        require(r['status'] in ('complete', 'failed_unavailable'), 'all_attempt_outcomes_required')
        if r['status'] == 'failed_unavailable':
            require(r['projection'] is None and isinstance(r['failure_type'], str), 'failure_not_unknown_reference_required')
            failures[r['failure_type']] += 1
            continue
        complete += 1
        p = r['projection']
        require(p['source_sha256'] == h and r['failure_type'] is None
                and p['current_lung_opacity_reference'] is None, 'narrow_bound_literal_reference_required')
        for flag in ('current_patient_scope_verified', 'qualifier_scope_verified', 'pulmonary_anatomy_scope_verified',
                     'generic_normality_expanded', 'synonyms_added', 'clinical_qualified', 'selection_changed', 'regeneration_authorized'):
            require(p[flag] is False, 'scope_or_action_cannot_be_promoted_by_count_audit')
        heads, evidence = p['literal_mentions'], p['human_span_evidence']
        keys = {(h['head_start'], h['head_end_exclusive']) for h in heads}
        require(len(keys) == len(heads), 'unique_literal_head_coordinates_required')
        counts = Counter()
        values = set()
        for e in evidence:
            require(set(e) == {'head_start', 'head_end_exclusive', 'char_start', 'char_end_exclusive', 'human_native_state'}
                and all(type(e[f]) is int for f in ('head_start', 'head_end_exclusive', 'char_start', 'char_end_exclusive'))
                and 0 <= e['char_start'] <= e['head_start'] < e['head_end_exclusive'] <= e['char_end_exclusive'] <= 100000
                and (e['head_start'], e['head_end_exclusive']) in keys
                and e['human_native_state'] in STATES[:3], 'bound_human_state_coordinate_required')
            counts[e['head_start'], e['head_end_exclusive']] += 1
            values.add(e['human_native_state'])
        for head in heads:
            require(set(head) == {'head_start', 'head_end_exclusive', 'annotated_observation_count'}
                and type(head['head_start']) is int and type(head['head_end_exclusive']) is int
                and 0 <= head['head_start'] < head['head_end_exclusive'] <= 100000
                and head['annotated_observation_count'] == counts[head['head_start'], head['head_end_exclusive']],
                'literal_head_evidence_count_mismatch')
        expected = 'unknown' if not values else 'uncertain' if len(values) > 1 or 'uncertain' in values else next(iter(values))
        require(p['manual_literal_state'] == expected and p['literal_mention_count'] == len(heads)
                and p['unannotated_literal_mentions'] == sum(n['annotated_observation_count'] == 0 for n in heads),
                'independent_literal_state_or_mention_pool_mismatch')
        states[expected] += 1
        mentions += len(heads)
        bindings += len(evidence)
        unannotated += p['unannotated_literal_mentions']
        with_mentions += bool(heads)
        with_unannotated += p['unannotated_literal_mentions'] > 0
    return {'attempted_reports': len(records), 'complete_reports': complete,
        'failed_reports': len(records) - complete, 'failure_type_counts': dict(failures),
        'manual_literal_state_counts': {s: states[s] for s in STATES},
        'reports_with_literal_mentions': with_mentions, 'reports_without_literal_mentions': complete - with_mentions,
        'reports_with_unannotated_literal_mentions': with_unannotated,
        'literal_mentions': mentions, 'human_span_evidence_bindings': bindings,
        'duplicate_source_hash_slots_retained': sum(n - 1 for n in hashes.values()),
        'prior_test_source_hash_overlap_slots': sum(n for h, n in hashes.items() if h in previous_hashes),
        'unique_source_artifacts': len(hashes), 'reference_states_available': [s for s in STATES if states[s]]}


def execute():
    started = time.monotonic()
    require(sha256(RUN / 'manifest.json') == EXPECTED, 'sealed_completed_manual_inventory_required')
    m = json.loads((RUN / 'manifest.json').read_text())
    for name, value in m['artifacts'].items():
        require(sha256(RUN / name) == value, 'sealed_inventory_artifact_changed')
    plan = json.loads((RUN / 'frozen_plan.json').read_text())
    require(m['pins'] == plan['pins'] and m['pins'][RAW] == plan['request']['source_file_sha256'],
            'original_sealed_source_receipt_binding_required')
    # Explicitly do not hash/decode clinical source bytes in the cached audit.
    for name, value in m['pins'].items():
        if name == RAW:
            continue
        path = WORKSPACE / name
        require(path.resolve().is_relative_to(WORKSPACE) and sha256(path) == value, 'code_or_cached_metadata_changed')
    prior = WORKSPACE / plan['request']['prior_test_contract']
    old = json.loads((prior / 'cohort.json').read_text())
    require(len(old) == 100, 'all_old_test_hashes_required')
    records = json.loads((RUN / 'reference_inventory.json').read_text())['records']
    counts = replay(records, {r['source_report_sha256'] for r in old})
    summary = json.loads((RUN / 'summary.json').read_text())
    require(all(summary[k] == v for k, v in counts.items()) and summary['new_model_calls'] == 0
        and summary['clinical_qualified'] is False and summary['current_lung_opacity_reference_available'] is False
        and summary['checkpoint_training_overlap_verified'] is False and summary['patient_disjointness_verified'] is False,
        'independent_full_denominator_support_summary_mismatch')
    paths = [Path(__file__).resolve(), WORKSPACE / 'TriCompose-v1.2/tests/test_manual_opacity_inventory_audit.py',
        WORKSPACE / 'TriCompose-v1.2/tools/prepare_ratescore_assets.py',
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    temporary, target = new_atomic_run(BASE / 'manual_opacity_reference_audits', 'numeric_12784259_001')
    try:
        write_json(temporary / 'frozen_plan.json', {'schema_version': 'manual-opacity-inventory-independent-audit-plan-v1',
            'pins': pins, 'completed_run_manifest_sha256': EXPECTED, 'production_projection_or_summary_imported': False,
            'raw_source_bytes_or_native_annotations_reopened': False, 'clinical_qualified': False})
        result = {'schema_version': 'manual-opacity-inventory-independent-audit-v1', 'status': 'passed',
            'completed_run_manifest_sha256': EXPECTED, 'verified_reference_support': counts,
            'all_four_literal_reference_states_observed': all(counts['manual_literal_state_counts'][s] > 0 for s in STATES),
            'raw_source_content_currently_reverified': False, 'human_annotations_reopened_for_adjudication': False,
            'literal_word_heads_independently_reconstructed_from_text': False,
            'clinical_qualified': False, 'new_model_calls': 0, 'selection_changed': False,
            'regeneration_authorized': False, 'runtime_seconds': time.monotonic() - started}
        write_json(temporary / 'audit.json', result)
        require(all(sha256(WORKSPACE / n) == v for n, v in pins.items())
            and sha256(RUN / 'manifest.json') == EXPECTED, 'audit_sources_changed')
        write_json(temporary / 'manifest.json', {'schema_version': 'manual-opacity-reference-audit-receipt-v1',
            'pins': pins, 'completed_run_manifest_sha256': EXPECTED,
            'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())}, 'clinical_qualified': False})
        for p in [temporary, *temporary.rglob('*')]:
            require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'private_reference_audit_required')
        commit_atomic_run(temporary, target)
        return target, result
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_manual_inventory_audit_takes_no_arguments')
        require_slurm(Path('/proc/self/cgroup').read_text(), os.environ.get('SLURM_JOB_ID', ''))
        require(os.environ['SLURM_JOB_ID'] == '12784259' and not os.environ.get('SLURM_JOB_GPUS')
            and not os.environ.get('SLURM_STEP_GPUS'), 'actual_existing_cache_only_cpu_allocation_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_manual_opacity_inventory_numeric_audit_passed',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_manual_opacity_inventory_numeric_audit_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
