#!/usr/bin/env python3
"""Copy two fixed-order synthetic demonstrations as bytes, never real targets.

No semantic inspection, report rendering, image decoding or model execution.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_first_version_delivery as delivery
from contracts import (PROTECTED_ROOT, new_atomic_run, private_directory,
    commit_atomic_run, discard_atomic_run, sha256_file, write_private_json, write_private_text)

SOURCE = delivery.BASE / 'deliverables/first_version_12714150_001'
SOURCE_PIN = 'a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc'
SCHEMA = 'tricompose-first-version-fixed-synthetic-samples-v1'


def choose_cases(rows, count=2):
    delivery.require(type(count) is int and count == 2 and len(rows) == 80
        and len({r['case_id'] for r in rows}) == 80,
        'fixed_two_of_eighty_case_order_required')
    delivery.require(all(re.fullmatch(r'case_[0-9]+', r['case_id']) for r in rows),
        'opaque_case_names_required')
    return sorted(rows, key=lambda r: (int(r['case_id'].split('_')[1]), r['case_id']))[:count]


def copy_bytes(source, expected_hash, destination, sources, label):
    p = delivery.bounded(source, expected_hash, sources, label, 16 * 1024 ** 2)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(destination, flags, 0o660)
    with p.open('rb') as reader, os.fdopen(fd, 'wb') as writer:
        for block in iter(lambda: reader.read(1024 ** 2), b''):
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    os.chmod(destination, 0o660)
    delivery.require(sha256_file(destination) == expected_hash,
        'copied_synthetic_artifact_hash_differs')


def export(run_id, archive=False):
    delivery.cached.cpu_guard()
    sources = {}
    # This metadata manifest binds >3,000 sources and exceeds the generic 1-MiB
    # historical-manifest limit. Its exact hash is still required, bounded at
    # 4 MiB; no source artifact, clinical rule or cohort is changed.
    mp = delivery.bounded(SOURCE / 'manifest.json', SOURCE_PIN, sources,
        'source_delivery_manifest', 4 * 1024 ** 2)
    m = json.loads(mp.read_text())
    cp = delivery.artifact(SOURCE, m, 'case_index.csv', sources, 'case_index')
    ip = delivery.artifact(SOURCE, m, 'candidate_index.csv', sources, 'candidate_index')
    info_path = delivery.artifact(SOURCE, m, 'generation_inventory.json', sources, 'generation_inventory')
    info = json.loads(info_path.read_text())
    delivery.require(info['source_conditioning'] == 'historical_august_bridge_not_september_repair'
        and info['clinical_best_triple_established'] is False, 'unchanged_development_scope_required')
    with cp.open(newline='') as stream:
        chosen = choose_cases(list(csv.DictReader(stream)))
    with ip.open(newline='') as stream:
        candidates = {r['triple_candidate_id']: r for r in csv.DictReader(stream)}
    delivery.require(len(candidates) == 960, 'full_candidate_index_required')
    parent = delivery.BASE / 'deliverables'
    archive_path = parent / (run_id + '.tar.gz')
    delivery.require(not archive or not archive_path.exists(), 'sample_archive_overwrite_refused')
    temporary, target = new_atomic_run(parent, run_id)
    records, artifacts = [], {}
    try:
        private_directory(temporary / 'cases')
        for ordinal, row in enumerate(chosen):
            case_name = 'case_%03d' % ordinal
            directory = temporary / 'cases' / case_name
            private_directory(directory)
            fixed = candidates[row['fixed_candidate_id']]
            static = candidates[row['historical_selected_candidate_id']]
            delivery.require(fixed['case_id'] == static['case_id'] == row['case_id']
                and fixed['ehr_sha256'] == static['ehr_sha256'] == row['ehr_sha256']
                and static['historical_static_selected'] == 'True', 'same_fixed_synthetic_ehr_required')
            payloads = [('synthetic_ehr.json', static['ehr_path'], static['ehr_sha256'])]
            for label, candidate in (('fixed', fixed), ('static', static)):
                for role, extension in (('cxr', '.png'), ('report', '.txt')):
                    payloads.append((label + '_' + role + extension,
                        candidate[role + '_path'], candidate[role + '_sha256']))
            for name, source_path, digest in payloads:
                destination = directory / name
                copy_bytes(source_path, digest, destination, sources, case_name + '_' + name)
                artifacts[str(destination.relative_to(temporary))] = {'sha256': digest,
                    'content_scope': 'synthetic_model_output_not_real_input_or_target'}
            records.append({'export_case_id': case_name, 'source_opaque_case_id': row['case_id'],
                'ehr_sha256': row['ehr_sha256'], 'fixed_candidate_id': fixed['triple_candidate_id'],
                'historical_static_candidate_id': static['triple_candidate_id'],
                'fixed_models': [fixed['cxr_model_id'], fixed['report_model_id']],
                'historical_static_models': [static['cxr_model_id'], static['report_model_id']],
                'direct_cached_ehr_edge_available': row['direct_ehr_edge_available'] == 'True',
                'clinical_correctness_verified': False})
        text = ('# Fixed-order synthetic examples / 两例合成输出\n\n'
            '病例来自固定 80-EHR 队列，按 opaque case integer 顺序取前两例，不按分数或视觉质量挑样本。\n'
            'These are existing fully synthetic DEVELOPMENT outputs, not real patient inputs or targets.\n'
            'Only authorized project collaborators may access/share this protected packet. Do not put it in Git or external clinical APIs.\n\n'
            'Each cases/case_XXX directory contains:\n'
            '- synthetic_ehr.json: same fixed synthetic EHR for both paths.\n'
            '- fixed_cxr.png + fixed_report.txt: existing ordinary fixed-path result.\n'
            '- static_cxr.png + static_report.txt: historical proxy-selected result.\n\n'
            'Open these files locally to compare. 文件只是字节复制；本任务未阅读正文、解码图像或作临床判读。\n'
            'No new inference, human annotation, score change or true-target comparison. Static means historical scorer choice, not clinically best.\n'
            'Scores, all budgets, source indices and definitions are in first_version_12714150_001.tar.gz.\n'
            'The 80-case bank belongs to the August bridge, not September repair or new October generation.\n')
        p = write_private_text(temporary / 'README_CN_EN.md', text)
        artifacts[p.name] = {'sha256': sha256_file(p)}
        sources['sample_exporter'] = (Path(__file__), sha256_file(Path(__file__)))
        sources['sample_tests'] = (ROOT / 'tests/test_first_version_samples.py',
            sha256_file(ROOT / 'tests/test_first_version_samples.py'))
        delivery.require(all(sha256_file(p) == digest for p, digest in sources.values()),
            'sample_export_source_changed')
        write_private_json(temporary / 'manifest.json', {'schema_version': SCHEMA,
            'run_id': run_id, 'source_paths': {k: str(p) for k, (p, _) in sources.items()},
            'source_sha256': {k: digest for k, (_, digest) in sources.items()}, 'artifacts': artifacts,
            'cases': records, 'case_selection': 'first_two_by_opaque_case_integer_not_score_or_quality',
            'payload_semantics_inspected': False, 'source_mimic_or_real_target_read': False,
            'clinical_qualified': False, 'original_selection_changed': False, 'new_model_calls': 0})
        delivery.require(not target.exists(), 'sample_run_overwrite_refused')
        commit_atomic_run(temporary, target)
    except BaseException:
        discard_atomic_run(temporary)
        raise
    if archive:
        names = sorted([*artifacts, 'manifest.json'])
        fd = os.open(archive_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o660)
        with os.fdopen(fd, 'wb') as stream:
            with tarfile.open(fileobj=stream, mode='w:gz') as package:
                for name in names:
                    p = target / name
                    delivery.require(p.is_file() and not p.is_symlink(), 'sample_archive_regular_file_required')
                    if name in artifacts:
                        delivery.require(sha256_file(p) == artifacts[name]['sha256'], 'sample_archive_hash_changed')
                    metadata = package.gettarinfo(str(p), arcname=run_id + '/' + name)
                    metadata.uid = metadata.gid = metadata.mtime = 0
                    metadata.uname = metadata.gname = ''
                    with p.open('rb') as payload:
                        package.addfile(metadata, payload)
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(archive_path, 0o660)
    return target, archive_path if archive else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args(); os.umask(0o007)
    try:
        target, archive_path = export(args.run_id, args.archive)
    except Exception as error:
        print(json.dumps({'status': 'synthetic_sample_export_failed', 'error_type': type(error).__name__,
            'safe_reason': str(error) if isinstance(error, ValueError)
            and re.fullmatch('[a-z0-9_]+', str(error)) else 'suppressed'}))
        return 1
    print(json.dumps({'status': 'synthetic_fixed_order_examples_exported', 'new_model_calls': 0,
        'manifest_sha256': sha256_file(target / 'manifest.json'),
        'archive_sha256': sha256_file(archive_path) if archive_path else None}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
