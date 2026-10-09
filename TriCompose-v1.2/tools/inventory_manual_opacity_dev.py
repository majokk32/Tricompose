"""Prepared protected CPU task: literal manual references, zero model calls.

Raw reports/annotations are consumed internally only AFTER approval of the
complete Slurm script. Never execute from the cache-only current allocation.
No source text, original keys, patient IDs or response strings are exported.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from contracts import new_atomic_run, commit_atomic_run, discard_atomic_run
from prepare_ratescore_assets import WORKSPACE, require, require_slurm, sha256, write_json
from prepare_cxrgraph_manual_cohort import read_documents
from tricompose_v12.cxrgraph_gold_adapter import normalize_manual_document, validate_adapter
from tricompose_v12.manual_opacity_inventory import VERSION, PATTERN, project, summarize

CONFIG = WORKSPACE / 'TriCompose-v1.2/configs/manual_opacity_inventory_v1.json'
BASE = WORKSPACE / 'artifacts/protected/tricompose_v1_2'


def project_document(document, index):
    rid = f'report_{index:04d}'
    row = {'report_id': rid, 'source_index': index, 'source_sha256': None,
        'status': 'failed_unavailable', 'failure_type': None, 'projection': None}
    try:
        require(isinstance(document, dict) and isinstance(document.get('sentences'), list), 'manual_document_required')
        words = [w for sentence in document['sentences'] for w in sentence]
        require(words and all(isinstance(w, str) and w and not any(c.isspace() for c in w) for w in words),
                'released_literal_word_strings_required')
        text = ' '.join(words)
        require(len(text) <= 100000, 'bounded_manual_text_required')
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        row['source_sha256'] = h
        adapter = normalize_manual_document(document, report_id=rid, source_report_sha256=h,
            origin='declared_human_annotation')
        validate_adapter(adapter)
        bounds, cursor = [], 0
        for word in words:
            bounds.append((cursor, cursor + len(word)))
            cursor += len(word) + 1
        entities = [[bounds[a][0], bounds[b - 1][1], label]
            for a, b, label in adapter['common_label_record']['entities']]
        row.update(status='complete', projection=project(text, entities, h))
    except Exception as error:
        row['failure_type'] = type(error).__name__
    return row


def sources():
    names = ('tools/inventory_manual_opacity_dev.py', 'tools/prepare_cxrgraph_manual_cohort.py',
        'tools/prepare_ratescore_assets.py', 'src/tricompose_v12/manual_opacity_inventory.py',
        'src/tricompose_v12/cxrgraph_gold_adapter.py', 'src/tricompose_v12/entity_gold_contract.py',
        'src/tricompose_v12/radgraph_reference_contract.py', 'src/tricompose_v12/radgraph_reference_contract_v2.py',
        'tests/test_manual_opacity_inventory.py', 'tests/test_manual_opacity_inventory_worker.py',
        'slurm/82_manual_opacity_dev_cpu.sbatch', 'configs/manual_opacity_inventory_v1.json')
    return [WORKSPACE / 'TriCompose-v1.2' / n for n in names] + [
        WORKSPACE / 'TriCompose-v1.0/eval/report_v1_1/contracts.py']


def execute():
    started = time.monotonic()
    config = json.loads(CONFIG.read_text())
    require(config['schema_version'] == 'manual-opacity-inventory-request-v1'
        and config['source_file'] == 'source/manual_data/dev.json' and config['literal_pattern'] == PATTERN
        and config['new_model_calls'] == 0 and config['clinical_qualified'] is False,
        'fixed_manual_dev_language_inventory_required')
    intake = WORKSPACE / config['source_intake']
    prior = WORKSPACE / config['prior_test_contract']
    source = intake / config['source_file']
    require(intake.resolve().is_relative_to(BASE) and prior.resolve().is_relative_to(BASE)
        and source.resolve().is_relative_to(intake) and not source.is_symlink(), 'protected_source_boundaries_required')
    require(sha256(intake / 'intake_manifest.json') == config['source_intake_manifest_sha256']
        and sha256(prior / 'manifest.json') == config['prior_test_manifest_sha256'], 'sealed_manual_source_inventory_required')
    release = json.loads((intake / 'intake_manifest.json').read_text())
    files = {r['path']: r for r in release['files']}
    require(files[config['source_file']]['sha256'] == config['source_file_sha256']
        and files[config['source_file']]['embedded_sha256_verified'] is True
        and source.stat().st_size == config['source_bytes'] < 32_000_000, 'bounded_authorized_release_file_required')
    pm = json.loads((prior / 'manifest.json').read_text())
    require(sha256(prior / 'cohort.json') == pm['artifacts']['cohort.json'], 'same_prior_test_hash_inventory_required')
    bank = BASE / 'complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json'
    require(sha256(bank) == 'ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c', 'original_synthetic_bank_unchanged_required')
    paths = [*sources(), intake / 'intake_manifest.json', source, prior / 'manifest.json', prior / 'cohort.json', bank]
    pins = {str(p.relative_to(WORKSPACE)): sha256(p) for p in paths}
    require(pins[str(source.relative_to(WORKSPACE))] == config['source_file_sha256'], 'fixed_dev_source_content_required')
    temporary, target = new_atomic_run(BASE / 'manual_opacity_reference_inventories',
        'released_dev_' + os.environ['SLURM_JOB_ID'] + '_001')
    try:
        write_json(temporary / 'frozen_plan.json', {'schema_version': VERSION + '-plan', 'request': config,
            'pins': pins, 'reference_role': 'literal_head_human_observation_states_not_global_lung_opacity_gold',
            'already_inspected_test_reports_not_used_as_new_reference': True, 'all_source_documents_retained': True,
            'source_keys_text_or_images_exported': False, 'current_patient_qualifier_scope_unverified': True,
            'checkpoint_training_overlap_unresolved': True, 'new_model_calls': 0, 'clinical_qualified': False})
        documents, storage = read_documents(source.read_bytes())
        require(len(documents) <= config['maximum_reports'], 'predeclared_report_bound_required')
        records = [project_document(d, i) for i, d in enumerate(documents)]
        del documents
        old_cohort = json.loads((prior / 'cohort.json').read_text())
        require(len(old_cohort) == 100, 'all_prior_test_references_for_overlap_check_required')
        summary = summarize(records, previous_test_hashes={r['source_report_sha256'] for r in old_cohort})
        summary.update(status='complete_annotation_inventory_not_model_evaluation',
            storage_format=storage, input_representation='released_words_joined_with_single_spaces',
            annotation_origin='user_uploaded_release_declared_joint_manual_annotation',
            exact_source_hash_overlap_is_not_patient_disjointness=True,
            source_report_text_original_keys_exported=False, actual_cpu_job=os.environ['SLURM_JOB_ID'],
            runtime_seconds=time.monotonic() - started,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
        write_json(temporary / 'reference_inventory.json', {'role': 'human_literal_opacity_span_projection_only', 'records': records})
        write_json(temporary / 'summary.json', summary)
        require(all(sha256(WORKSPACE / name) == value for name, value in pins.items()), 'inventory_sources_changed')
        write_json(temporary / 'manifest.json', {'schema_version': VERSION + '-receipt', 'pins': pins,
            'artifacts': {p.name: sha256(p) for p in sorted(temporary.iterdir())},
            'clinical_qualified': False, 'new_model_calls': 0, 'selection_changed': False, 'regeneration_authorized': False})
        for p in [temporary, *temporary.rglob('*')]:
            require(not p.is_symlink() and p.stat().st_gid in (96293, 65534)
                and p.stat().st_mode & 0o7777 == (0o2770 if p.is_dir() else 0o660), 'project_private_reference_inventory_required')
        commit_atomic_run(temporary, target)
        return target, summary
    except BaseException:
        discard_atomic_run(temporary)
        raise


def main():
    try:
        require(len(sys.argv) == 1, 'fixed_annotation_inventory_takes_no_arguments')
        job = os.environ.get('SLURM_JOB_ID', '')
        require_slurm(Path('/proc/self/cgroup').read_text(), job)
        require(job != '12784259' and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),
                'new_approved_cpu_task_not_current_cache_only_allocation_required')
        require(os.environ.get('TRICOMPOSE_MANUAL_OPACITY_REQUEST') == 'manual-opacity-inventory-request-v1',
                'explicit_scope_batch_entry_required')
        os.umask(0o007)
        target, result = execute()
        print(json.dumps({'status': 'protected_manual_opacity_annotation_inventory_complete',
            'runtime_seconds': round(result['runtime_seconds'], 3), 'peak_rss_gib': round(result['peak_rss_gib'], 3),
            'manifest_sha256': sha256(target / 'manifest.json')}))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'protected_manual_opacity_annotation_inventory_failed', 'failure_type': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
