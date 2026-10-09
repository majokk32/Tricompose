#!/usr/bin/env python
"""Python 3.6-compatible offline initialization, no report or patient input.

The source/rules are not patched. Explicit resource paths replace unsafe auto
download defaults. PLY disk-output options alone are scoped to avoid source
modification; no target, grammar or negation rule changes. CPU Slurm only.
"""
from __future__ import print_function

import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
ENV = WORKSPACE/'runtime/venvs/chexpert-negbio-py36-12682821-v1'
RESOURCES = WORKSPACE/'runtime/eval_models/chexpert_negbio/frozen_assets_12682821_001'
RESOURCE_MANIFEST = 'c3fa079110adef7392e0363cbad745204b638f23cd391699834f047b2e1326b6'
TMP = WORKSPACE/'.tmp/chexpert_negbio_setup_12682821_001'
EXPECTED = {'numpy': '1.15.4', 'pandas': '0.23.4', 'networkx': '1.11',
            'nltk': '3.3.0', 'ply': '3.11', 'JPype1': '0.6.3',
            'bioc': '1.1.dev3', 'bllipparser': '2016.9.11',
            'pystanforddependencies': '0.3.1', 'lxml': '3.7.3'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def inside(path, root):
    return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(root.resolve())


def guard():
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or '/job_' + job + '/' not in Path('/proc/self/cgroup').read_text() or \
            os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'):
        raise RuntimeError('actual_existing_cpu_slurm_allocation_required')
    if sys.version_info[:3] != (3, 6, 7) or Path(sys.prefix).resolve() != ENV.resolve():
        raise RuntimeError('exact_dedicated_legacy_python_required')
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1' or \
            Path(os.environ.get('TMPDIR', '')).resolve() != TMP.resolve():
        raise RuntimeError('workspace_cache_and_bytecode_policy_required')


@contextlib.contextmanager
def quiet_native():
    # Also captures native C++/JVM output; never expose raw native exceptions.
    sys.stdout.flush()
    sys.stderr.flush()
    descriptors = [os.dup(1), os.dup(2)]
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for descriptor, original in zip((1, 2), descriptors):
            os.dup2(original, descriptor)
            os.close(original)
        os.close(sink)


class ErrorCounter(logging.Handler):
    def __init__(self):
        super(ErrorCounter, self).__init__()
        self.count = 0

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.count += 1


def initialize():
    guard()
    manifest_path = RESOURCES/'resource_manifest.json'
    if sha(manifest_path) != RESOURCE_MANIFEST:
        raise ValueError('sealed_parsing_resource_manifest_required')
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['files'].items():
        path = RESOURCES/relative
        if not inside(path, RESOURCES) or sha(path) != expected:
            raise ValueError('frozen_resource_file_changed')
    import pkg_resources
    versions = {name: pkg_resources.get_distribution(name).version for name in EXPECTED}
    # Conda's NLTK 3.3.0 metadata reports upstream distribution version 3.3.
    # Compare normalized public versions; record the actual strings unchanged.
    if any(pkg_resources.parse_version(versions[name]) != pkg_resources.parse_version(expected)
           for name, expected in EXPECTED.items()):
        raise ValueError('exact_upstream_core_versions_required')
    error_counter = ErrorCounter()
    logger = logging.getLogger()
    logger.handlers = [error_counter]
    logger.setLevel(logging.ERROR)
    started = time.monotonic()
    with quiet_native():
        import nltk
        nltk.data.path[:] = [str(RESOURCES/'nltk_data')]
        for name in ('tokenizers/punkt', 'taggers/universal_tagset/en-ptb.map', 'corpora/wordnet'):
            nltk.data.find(name)
        # PLY otherwise generates tables/debug logs beside the frozen sources.
        import ply.yacc
        native_yacc = ply.yacc.yacc
        def no_source_writes(*args, **kwargs):
            kwargs['write_tables'] = False
            kwargs['debug'] = False
            kwargs['outputdir'] = str(TMP)
            return native_yacc(*args, **kwargs)
        ply.yacc.yacc = no_source_writes
        sys.path[:0] = [str(WORKSPACE/'chexpert-labeler'), str(WORKSPACE/'NegBio')]
        import StanfordDependencies
        native_get_instance = StanfordDependencies.get_instance
        jar = str(RESOURCES/'archives/stanford-corenlp-3.5.2.jar')
        def explicit_offline_jar(*args, **kwargs):
            if args or kwargs.get('backend', 'jpype') != 'jpype':
                raise ValueError('official_default_jpype_backend_required')
            kwargs['jar_filename'] = jar
            kwargs['download_if_missing'] = False
            kwargs['extra_jvm_args'] = ['-Djava.io.tmpdir=' + str(TMP),
                                       '-Djava.util.prefs.userRoot=' + str(TMP/'java_prefs')]
            result = native_get_instance(**kwargs)
            if type(result).__name__ != 'JPypeBackend':
                raise ValueError('silent_backend_fallback_not_allowed')
            return result
        StanfordDependencies.get_instance = explicit_offline_jar
        from stages import classify
        classify.PARSING_MODEL_DIR = str(RESOURCES/'GENIA+PubMed')
        classifier = classify.Classifier(
            WORKSPACE/'chexpert-labeler/patterns/pre_negation_uncertainty.txt',
            WORKSPACE/'chexpert-labeler/patterns/negation.txt',
            WORKSPACE/'chexpert-labeler/patterns/post_negation_uncertainty.txt', verbose=False)
        if classifier.ptb2dep._backend != 'jpype' or error_counter.count:
            raise ValueError('offline_initialization_failed')
        # No sentences/documents, collections, authored inputs or references used.
        result = {'schema_version': 'tricompose-chexpert-negbio-offline-initialization-v1',
                  'status': 'initialized_without_reports', 'python': sys.version.split()[0],
                  'environment': str(ENV), 'versions': versions, 'resource_root': str(RESOURCES),
                  'resource_manifest_sha256': RESOURCE_MANIFEST,
                  'backend': classifier.ptb2dep._backend, 'universal_dependencies': classifier.ptb2dep.universal,
                  'dependency_representation': classifier.ptb2dep.representation,
                  'rules_modified': False, 'default_model_and_jar_identity_retained': True,
                  'workspace_resource_path_overrides_explicit': True,
                  'ply_disk_output_only_override': True, 'detector_errors': error_counter.count,
                  'elapsed_seconds': round(time.monotonic() - started, 6),
                  'reports_or_authored_controls_read': 0, 'report_parser_calls': 0,
                  'end_to_end_report_smoke_passed': False, 'clinical_qualified': False,
                  'training_or_weight_change': False, 'external_api_used': False}
    for relative, expected in manifest['files'].items():
        if sha(RESOURCES/relative) != expected:
            raise ValueError('resource_changed_during_initialization')
    result['worker_sha256'] = sha(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not inside(args.output, WORKSPACE/'artifacts/protected') or args.output.exists():
        raise ValueError('fresh_protected_metadata_output_required')
    if not args.output.parent.is_dir():
        raise ValueError('precreated_private_output_directory_required')
    result = initialize()
    descriptor = os.open(str(args.output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o660)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    os.chmod(str(args.output), 0o660)
    print(json.dumps({'status': result['status'], 'report_parser_calls': 0,
                      'metadata_sha256': sha(args.output), 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'offline_initialization_failed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}), file=sys.stderr)
        raise SystemExit(2)
