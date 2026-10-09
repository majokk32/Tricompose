#!/usr/bin/env python3
"""Aggregate-only RICORD annotation audit; no image acquisition or inference.

The unanimous subgroup is a derived diagnostic subset, NOT a claim that the
official majority/adjudication algorithm or independent reader identities were
reproduced. Patient/source keys are never emitted, including hashed keys.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
ACQUIRE_PATH = WORKSPACE / 'TriCompose-v1.2/real_validation/acquire_ricord_annotations.py'
spec = importlib.util.spec_from_file_location('_ricord_acquisition', ACQUIRE_PATH)
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)
from contracts import (commit_atomic_run, discard_atomic_run, new_atomic_run,
                       require_inside, sha256_file, write_private_json,
                       write_private_text)

PROTOCOL = WORKSPACE / 'docs/ricord_reader_group_protocol.md'
TESTS = WORKSPACE / 'TriCompose-v1.2/tests/test_ricord_reader_groups.py'
SCHEMA = 'tricompose-ricord-reader-group-audit-v1'


def audit_groups(payload, removed):
    """Use all studies; no model score, individual label choice or cherry-pick."""
    definitions, groups = {}, []
    for index, group in enumerate(payload['labelGroups']):
        title = group.get('name', '')
        marked_adjudication = isinstance(title, str) and bool(
            re.search(r'\badjudicat\w*\b', title.casefold()))
        groups.append({'index': index, 'adjudication_metadata_marker': marked_adjudication})
        for definition in group['labels']:
            key = definition['id']
            if key in definitions:
                raise ValueError('duplicate_label_definition')
            # Only the four exact, catalog-supported classification labels.
            name = acquire.label_name(definition.get('name'))
            definitions[key] = (index, name if name in acquire.CLASSIFICATIONS else None,
                                definition.get('scope'))

    studies, rows = {}, []
    for dataset in payload['datasets']:
        for study in dataset['studies']:
            key = study['StudyInstanceUID']
            if key in studies:
                raise ValueError('duplicate_study_key')
            studies[key] = study
        rows.extend(dataset.get('annotations', []))
    active = set(studies).difference(removed)
    if not active:
        raise ValueError('active_studies_required')
    by_group = [{} for _ in groups]
    raw_counts, orphan_counts, invalid_counts = Counter(), Counter(), Counter()
    read_coverage = [set() for _ in groups]
    for row in rows:
        definition = definitions.get(row.get('labelId'))
        if definition is None:
            raise ValueError('undefined_label_reference')
        index, name, scope = definition
        key = row.get('StudyInstanceUID')
        if key in removed:
            continue
        raw_counts[index] += 1
        if key not in active:
            orphan_counts[index] += 1
            continue
        # Reading availability is not the number of interpretable diagnoses.
        # Preserve reader abstentions; do not remove an otherwise complete
        # reader simply because more studies have an unknown classification.
        read_coverage[index].add(key)
        if name is None:
            continue
        if scope != 'STUDY' or row.get('data') is not None:
            invalid_counts[index] += 1
            continue
        by_group[index].setdefault(key, set()).add(name)

    for group in groups:
        index = group['index']
        classes = Counter(next(iter(names)) if len(names) == 1 else 'conflicting'
                          for names in by_group[index].values())
        singleton_count = sum(value for name, value in classes.items() if name != 'conflicting')
        group.update(annotation_count_excluding_withdrawn=raw_counts[index],
                     orphan_annotation_count=orphan_counts[index],
                     invalid_classification_annotation_count=invalid_counts[index],
                     annotated_study_count=len(read_coverage[index]),
                     classification_counts=dict(sorted(classes.items())),
                     singleton_classified_studies=singleton_count,
                     complete_reader_candidate=(not group['adjudication_metadata_marker']
                                                and len(read_coverage[index]) / len(active) >= 0.95))
    complete = [group['index'] for group in groups if group['complete_reader_candidate']]
    # Do not invent readers or fill sparse groups. Three is prescribed by the
    # dataset paper; exact participant-to-group identity remains unverified.
    if len(complete) != 3:
        raise ValueError('three_comprehensive_classification_groups_required')

    counts = Counter()
    for key in active:
        observed = [by_group[index].get(key, set()) for index in complete]
        if any(len(value) > 1 for value in observed):
            counts['within_group_conflict'] += 1
        elif any(len(value) == 0 for value in observed):
            counts['missing_group_classification'] += 1
        else:
            labels = {next(iter(value)) for value in observed}
            counts[next(iter(labels)) if len(labels) == 1 else 'between_group_disagreement'] += 1
    positive = counts['typical_appearance'] + counts['indeterminate_appearance']
    negative = counts['negative_for_pneumonia']
    return {
        'active_study_count': len(active),
        'withdrawn_studies_excluded': len(set(studies).intersection(removed)),
        'classification_label_scope': 'STUDY',
        'group_selection_rule': 'non_adjudication_groups_with_annotations_on_at_least_95pct_active_studies',
        'comprehensive_group_indices': complete,
        'groups': groups, 'three_group_classification_counts': dict(sorted(counts.items())),
        'derived_reference_kind': 'unanimous_classification_of_three_comprehensive_label_groups',
        'derived_opacity_positive_studies': positive,
        'derived_opacity_negative_studies': negative,
        'derived_opacity_unknown_or_excluded_studies': len(active) - positive - negative,
        'annotation_counts_sufficient_for_proposed_25_positive_25_negative': min(positive, negative) >= 25,
        'reader_group_identity_verified': False,
        'official_majority_adjudication_reproduced': False,
        'unanimity_subset_selection_bias_disclosed': True,
        'independent_image_reference_not_report_extraction': True,
        'patient_grouping_verified': False, 'image_label_binding_verified': False,
        'pneumonia_diagnosis_reference_validated': False,
        'primary_metric_eligible': False, 'cohort_selected': False,
    }


def run(*, acquisition_root, output_root, run_id):
    source = require_inside(acquisition_root, acquire.PROTECTED, must_exist=True)
    manifest = json.loads((source / 'manifest.json').read_bytes())
    if manifest.get('schema_version') != acquire.SCHEMA + '-manifest':
        raise ValueError('acquisition_manifest_schema_rejected')
    if set(manifest.get('artifacts', {})) != {'annotations.json', 'acquisition.json'}:
        raise ValueError('acquisition_artifact_set_rejected')
    for name, digest in manifest['artifacts'].items():
        path = require_inside(source / name, source, must_exist=True)
        if sha256_file(path) != digest:
            raise ValueError('acquisition_artifact_hash_mismatch')
    for name, digest in manifest['sources'].items():
        path = require_inside(WORKSPACE / name, WORKSPACE, must_exist=True)
        if sha256_file(path) != digest:
            raise ValueError('consumed_source_hash_changed')
    temporary, target = new_atomic_run(output_root, run_id)
    started = time.monotonic()
    try:
        # Re-read ONLY the approved public catalog, not a new annotation/data
        # download. It supplies official withdrawals without logging study keys.
        catalog = acquire.public_get(acquire.public_opener(), acquire.CATALOG, 1500000)
        removed = acquire.catalog_contract(catalog)
        payload = json.loads((source / 'annotations.json').read_bytes(),
                             object_pairs_hook=acquire.reject_duplicate_keys)
        result = audit_groups(payload, removed)
        summary = {'schema_version': SCHEMA,
                   'status': 'derived_label_group_readiness_image_binding_pending',
                   'input_manifest_sha256': sha256_file(source / 'manifest.json'),
                   'input_annotation_sha256': sha256_file(source / 'annotations.json'),
                   'inventory': result, 'images_downloaded': 0, 'model_calls': 0,
                   'slurm_submissions': 0, 'thresholds_changed': False,
                   'selection_changed': False, 'benchmark_executed': False,
                   'elapsed_seconds': round(time.monotonic() - started, 6)}
        write_private_json(temporary / 'audit.json', summary)
        write_private_text(temporary / 'RESULTS_CN_EN.md',
            '# RICORD-1C 标注就绪检查 / annotation readiness\n\n'
            'This is a derived unanimity diagnostic, not official adjudication reproduction.\n\n'
            + json.dumps(result, indent=2, sort_keys=True) + '\n\n'
            '未下载图像、未运行推理、未选择 cohort、未改变阈值或旧选优。\n'
            'No images/inference/cohort selection or historical scoring changes.\n')
        write_private_json(temporary / 'manifest.json', {
            'schema_version': SCHEMA + '-manifest', 'run_id': run_id,
            'input_manifest_sha256': summary['input_manifest_sha256'],
            'sources': {str(path.relative_to(WORKSPACE)): sha256_file(path)
                        for path in (Path(__file__), TESTS, PROTOCOL)},
            'artifacts': {name: sha256_file(temporary / name)
                          for name in ('audit.json', 'RESULTS_CN_EN.md')},
        })
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    return target, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition-root', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', default=str(acquire.PROTECTED / 'tricompose_v1_2/ricord_annotation_audits'))
    args = parser.parse_args(argv)
    try:
        target, summary = run(acquisition_root=args.acquisition_root,
                              output_root=args.output_root, run_id=args.run_id)
    except Exception as error:
        print(json.dumps({'status': 'audit_failed_closed', 'error_type': type(error).__name__, 'model_calls': 0}))
        return 2
    print(json.dumps({'status': summary['status'], 'inventory': summary['inventory'],
                     'model_calls': 0, 'images_downloaded': 0,
                     'manifest_sha256': sha256_file(target / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
