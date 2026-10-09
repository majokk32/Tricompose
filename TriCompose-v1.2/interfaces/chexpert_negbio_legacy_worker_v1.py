#!/usr/bin/env python
"""Frozen official report parser on a sealed authored-only plan, Python 3.6.

No input-path mode, candidate loader, reference loader, score or action policy.
Unchanged Classifier.classify runs with observational failure/receipt hooks.
"""
from __future__ import print_function

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time
import types

WORKSPACE = Path('/project2/ruishanl_1185/inference_3mod')
ROOT = WORKSPACE/'TriCompose-v1.2'
PROTECTED = WORKSPACE/'artifacts/protected'
ENV = WORKSPACE/'runtime/venvs/chexpert-negbio-py36-12682821-v1'
DEPLOYMENT = PROTECTED/'tricompose_v1_2/chexpert_negbio_deployments/deployment_12682821_001'
DEPLOYMENT_SHA = 'f8c8d9a02455780de22e913ce7a1c056450c5e75eafffea6279d869583d9ef85'
SCHEMA = 'tricompose-chexpert-negbio-authored112-v1'
STAGES = ('load', 'extract', 'parse', 'dependency', 'syntax_gate', 'detector', 'aggregate', 'serialize')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(path, root):
    return os.path.commonpath([str(Path(path).resolve()), str(Path(root).resolve())]) == str(Path(root).resolve())


def metadata(path, expected):
    path = Path(path)
    if not inside(path, PROTECTED) or path.suffix != '.json' or not 0 < path.stat().st_size < 8*1024**2 or sha(path) != expected:
        raise ValueError('sealed_bounded_protected_json_required')
    result = json.loads(path.read_text(encoding='utf-8'))
    if sha(path) != expected:
        raise ValueError('metadata_changed_during_read')
    return result


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name)
        if not inside(path, WORKSPACE) or not path.is_file() or sha(path) != expected:
            raise ValueError('frozen_workspace_source_changed')


def private_json(path, value):
    if not inside(path, PROTECTED) or path.exists() or path.is_symlink():
        raise ValueError('fresh_protected_output_required')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(str(path), flags, 0o660)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(str(path), 0o660)


def legacy_helpers():
    path = ROOT/'interfaces/initialize_chexpert_negbio_legacy_v1.py'
    spec = importlib.util.spec_from_file_location('_frozen_legacy_helpers', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract(path=None):
    """Remove only an unused Python 3.7 future import in memory, no file edit.

    The sealed contract has no function/variable annotations. Reject a future
    change that would require wider transpilation instead of silently rewriting.
    """
    path = path or ROOT/'src/tricompose_v12/chexpert_negbio_contract.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                (node.returns is not None or any(arg.annotation is not None for arg in
                 node.args.args + node.args.kwonlyargs + ([node.args.vararg] if node.args.vararg else []) +
                 ([node.args.kwarg] if node.args.kwarg else []))):
            raise ValueError('annotation_free_legacy_contract_required')
    removed = [node for node in tree.body if isinstance(node, ast.ImportFrom) and
               node.module == '__future__' and [alias.name for alias in node.names] == ['annotations']]
    if len(removed) != 1:
        raise ValueError('exact_unused_future_import_required')
    tree.body = [node for node in tree.body if node not in removed]
    module = types.ModuleType('_sealed_chexpert_output_contract')
    module.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


class StageFailure(Exception):
    def __init__(self, reason):
        self.reason = reason
        super(StageFailure, self).__init__(reason)


class StageMonitor(object):
    def __init__(self, counter):
        self.counter = counter
        self.stages = {name: {'attempted': False, 'complete': False, 'errors': 0} for name in STAGES}

    def call(self, name, reason, function):
        row = self.stages[name]
        row['attempted'] = True
        before = self.counter.count
        try:
            result = function()
        except StageFailure:
            raise
        except Exception:
            raise StageFailure(reason)
        finally:
            row['errors'] = self.counter.count - before
        if row['errors']:
            raise StageFailure(reason)
        row['complete'] = True
        return result


def node_coverage(document):
    """Use official graph/node lookup; availability, not clinical scope gold."""
    from negbio.neg import semgraph, propagator, neg_detector
    passage = document.passages[0]
    graphs = []
    for sentence in passage.sentences:
        graph = semgraph.load(sentence)
        propagator.propagate(graph)
        graphs.append((sentence, graph))
    for annotation in passage.annotations:
        loc = annotation.locations[0]
        candidates = [(sentence, graph) for sentence, graph in graphs if
                      sentence.offset <= loc.offset and loc.offset + loc.length <= sentence.offset + len(sentence.text)]
        if len(candidates) != 1 or not list(neg_detector.find_nodes(candidates[0][1], loc.offset, loc.offset + loc.length)):
            raise StageFailure('dependency_graph_unavailable')


def classify_checked(classifier, collection, contract, monitor, negdetect, coverage=node_coverage):
    """Run the original method, restoring every observational hook afterwards."""
    parse_doc = classifier.parser.parse_doc
    convert_doc = classifier.ptb2dep.convert_doc
    detect = negdetect.detect
    receipts = []

    def checked_parse(document):
        return monitor.call('parse', 'parse_tree_unavailable', lambda: parse_doc(document))

    def checked_convert(document):
        return monitor.call('dependency', 'dependency_graph_unavailable', lambda: convert_doc(document))

    def checked_detect(document, detector):
        def gate():
            try:
                receipt = contract.syntax_receipt(document)
            except ValueError as error:
                reason = str(error)
                raise StageFailure(reason if reason in contract.FAILURES else 'evidence_alignment_failed')
            coverage(document)
            receipts.append(receipt)
        monitor.call('syntax_gate', 'dependency_graph_unavailable', gate)
        return monitor.call('detector', 'detector_error', lambda: detect(document, detector))

    try:
        classifier.parser.parse_doc = checked_parse
        classifier.ptb2dep.convert_doc = checked_convert
        negdetect.detect = checked_detect
        classifier.classify(collection)
    finally:
        classifier.parser.parse_doc = parse_doc
        classifier.ptb2dep.convert_doc = convert_doc
        negdetect.detect = detect
    if len(receipts) != 1:
        raise StageFailure('dependency_graph_unavailable')
    return receipts[0]


def initialize(runtime, tmp):
    helpers = legacy_helpers()
    import pkg_resources
    versions = {name: pkg_resources.get_distribution(name).version for name in helpers.EXPECTED}
    if versions != runtime['core_versions']:
        raise ValueError('frozen_dependency_versions_required')
    counter = helpers.ErrorCounter()
    logging.getLogger().handlers = [counter]
    logging.getLogger().setLevel(logging.ERROR)
    with helpers.quiet_native():
        import nltk
        resources = Path(runtime['resource_root'])
        nltk.data.path[:] = [str(resources/'nltk_data')]
        for name in ('tokenizers/punkt', 'taggers/universal_tagset/en-ptb.map', 'corpora/wordnet'):
            nltk.data.find(name)
        import ply.yacc
        native_yacc = ply.yacc.yacc
        def official_grammar(*args, **kwargs):
            native_module = sys.modules.get('negbio.ngrex.parser')
            if native_module is None or 'module' in kwargs and kwargs['module'] is not native_module:
                raise ValueError('official_ngrex_grammar_required')
            kwargs.update(module=native_module, write_tables=False, debug=False, outputdir=str(tmp))
            return native_yacc(*args, **kwargs)
        ply.yacc.yacc = official_grammar
        sys.path[:0] = [str(WORKSPACE/'chexpert-labeler'), str(WORKSPACE/'NegBio')]
        import StanfordDependencies
        native_instance = StanfordDependencies.get_instance
        def offline_jar(*args, **kwargs):
            if args or kwargs.get('backend', 'jpype') != 'jpype':
                raise ValueError('official_jpype_backend_required')
            kwargs.update(jar_filename=str(resources/'archives/stanford-corenlp-3.5.2.jar'),
                          download_if_missing=False,
                          extra_jvm_args=['-Djava.io.tmpdir=' + str(tmp),
                                          '-Djava.util.prefs.userRoot=' + str(tmp/'java_prefs')])
            backend = native_instance(**kwargs)
            if type(backend).__name__ != 'JPypeBackend':
                raise ValueError('backend_fallback_not_allowed')
            return backend
        StanfordDependencies.get_instance = offline_jar
        from stages import Classifier, Extractor, Aggregator, classify
        from loader import Loader
        from constants import CATEGORIES
        classify.PARSING_MODEL_DIR = str(resources/'GENIA+PubMed')
        model = Classifier(WORKSPACE/'chexpert-labeler/patterns/pre_negation_uncertainty.txt',
                           WORKSPACE/'chexpert-labeler/patterns/negation.txt',
                           WORKSPACE/'chexpert-labeler/patterns/post_negation_uncertainty.txt', verbose=False)
        extractor = Extractor(WORKSPACE/'chexpert-labeler/phrases/mention',
                              WORKSPACE/'chexpert-labeler/phrases/unmention', verbose=False)
        aggregator = Aggregator(CATEGORIES, verbose=False)
        loader = Loader(None, [], False)
        contract = load_contract()
        if tuple(CATEGORIES) != contract.CATEGORIES or model.ptb2dep._backend != 'jpype' or \
                not model.ptb2dep.universal or model.ptb2dep.representation != 'CCprocessed' or counter.count:
            raise ValueError('frozen_native_runtime_required')
    return helpers, contract, counter, loader, extractor, model, aggregator


def parse_item(item, components):
    helpers, contract, counter, loader, extractor, classifier, aggregator = components
    monitor = StageMonitor(counter)
    with helpers.quiet_native():
        try:
            def load():
                import bioc
                from negbio.pipeline import text2bioc
                document = text2bioc.text2document(item['item_id'], loader.clean(item['text']))
                if not document.passages[0].text.strip():
                    raise StageFailure('empty_cleaned_source')
                document = loader.splitter.split_doc(document)
                collection = bioc.BioCCollection()
                collection.add_document(document)
                return collection, document
            collection, document = monitor.call('load', 'empty_cleaned_source', load)
            monitor.call('extract', 'evidence_alignment_failed', lambda: extractor.extract(collection))
            from negbio.pipeline import negdetect
            receipt = classify_checked(classifier, collection, contract, monitor, negdetect)
            values = monitor.call('aggregate', 'invalid_native_output', lambda: aggregator.aggregate(collection))
            if values.shape != (1, 14):
                raise StageFailure('invalid_native_output')
            result = monitor.call('serialize', 'evidence_alignment_failed', lambda:
                                  contract.serialize(document, item['text'], values[0], receipt,
                                                     detector_error_count=monitor.stages['detector']['errors']))
        except StageFailure as failure:
            result = contract.unavailable(failure.reason)
        except Exception:
            result = contract.unavailable('dependencies_unavailable')
    result['stage_availability'] = monitor.stages
    result['native_classifier_classify_called_unchanged'] = monitor.stages['parse']['attempted']
    return dict(item_id=item['item_id'], report_sha256=item['report_sha256'], output=result)


def guard():
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or '/job_' + job + '/' not in Path('/proc/self/cgroup').read_text() or \
            os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'):
        raise RuntimeError('actual_existing_cpu_slurm_required')
    if sys.version_info[:3] != (3, 6, 7) or Path(sys.prefix).resolve() != ENV.resolve() or \
            os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise RuntimeError('dedicated_frozen_legacy_runtime_required')
    tmp = Path(os.environ.get('TMPDIR', ''))
    if not inside(tmp, WORKSPACE/'.tmp') or not tmp.is_dir():
        raise RuntimeError('workspace_temporary_directory_required')
    return tmp


def validate_inputs(inputs):
    if set(inputs) != {'old48', 'new64'}:
        raise ValueError('two_authored_sets_required')
    for dataset, size in (('old48', 48), ('new64', 64)):
        rows = inputs[dataset]
        prefix = 'authored_' if dataset == 'old48' else 'context_'
        if not isinstance(rows, list) or len(rows) != size:
            raise ValueError('fixed_authored_inventory_required')
        for index, row in enumerate(rows):
            if set(row) != {'item_id', 'text', 'report_sha256'} or row['item_id'] != prefix + str(index).zfill(4) or \
                    not isinstance(row['text'], str) or not row['text'].strip() or len(row['text']) > 8192 or \
                    hashlib.sha256(row['text'].encode('utf-8')).hexdigest() != row['report_sha256']:
                raise ValueError('bounded_fixed_hash_bound_authored_input_required')
    if len({r['report_sha256'] for rows in inputs.values() for r in rows}) != 112:
        raise ValueError('distinct_authored_inputs_required')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan-root', required=True, type=Path)
    cli.add_argument('--plan-manifest-sha256', required=True)
    cli.add_argument('--output-directory', required=True, type=Path)
    args = cli.parse_args()
    tmp = guard()
    if not inside(args.plan_root, PROTECTED/'tricompose_v1_2/chexpert_negbio_authored_plans') or \
            not inside(args.output_directory, PROTECTED/'tricompose_v1_2/chexpert_negbio_authored_runs') or \
            not args.output_directory.is_dir() or any(args.output_directory.iterdir()):
        raise ValueError('fresh_sealed_authored_plan_and_empty_private_run_required')
    deployment = metadata(DEPLOYMENT/'manifest.json', DEPLOYMENT_SHA)
    verify_pins(deployment['sources'])
    runtime = metadata(DEPLOYMENT/'runtime.json', deployment['artifacts']['runtime.json'])
    manifest = metadata(args.plan_root/'manifest.json', args.plan_manifest_sha256)
    verify_pins(manifest['sources'])
    if manifest['sources'].get(str(Path(__file__))) != sha(Path(__file__)):
        raise ValueError('prospectively_frozen_worker_required')
    plan = metadata(args.plan_root/'plan.json', manifest['artifacts']['plan.json'])
    if plan['schema_version'] != SCHEMA + '-plan' or plan['deployment_manifest_sha256'] != DEPLOYMENT_SHA or \
            plan['policy']['authored_only'] is not True or plan['policy']['clinical_qualified'] is not False or \
            plan['policy']['replay_all'] is not True:
        raise ValueError('authored_diagnostic_policy_required')
    started = time.monotonic()
    components = initialize(runtime, tmp)
    initialized = time.monotonic()
    inputs = metadata(args.plan_root/'inputs.json', manifest['artifacts']['inputs.json'])
    validate_inputs(inputs)
    predictions, replays = {}, []
    processed = 0
    for dataset in ('old48', 'new64'):
        predictions[dataset] = []
        for item in inputs[dataset]:
            primary = parse_item(item, components)
            replay = parse_item(item, components)
            predictions[dataset].append(primary)
            replays.append({'dataset': dataset, 'item_id': item['item_id'], 'report_sha256': item['report_sha256'],
                            'same_full_evidence': primary == replay,
                            'both_complete': primary['output']['status'] == replay['output']['status'] == 'complete',
                            'replay_output': replay['output']})
            processed += 1
            if processed % 16 == 0:
                print(json.dumps({'status': 'authored_parser_progress', 'authored_texts_processed': processed,
                                  'parser_passes': processed*2, 'report_bodies_exposed': False}), flush=True)
    parsed = time.monotonic()
    verify_pins(deployment['sources'])
    verify_pins(manifest['sources'])
    private_json(args.output_directory/'predictions.json', {'schema_version': SCHEMA, 'datasets': predictions})
    private_json(args.output_directory/'replays.json', {'schema_version': SCHEMA, 'records': replays})
    closed = {name: sha(args.output_directory/name) for name in ('predictions.json', 'replays.json')}
    private_json(args.output_directory/'prediction_freeze_receipt.json', {
        'sha256': closed, 'references_read_before_prediction_fsync': False,
        'candidate_or_patient_inputs_read': False, 'investigator_knows_previous_development_results': True,
        'predictions_readiness_is_not_clinical_qualification': True})
    private_json(args.output_directory/'runtime.json', {
        'schema_version': SCHEMA + '-runtime', 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'environment': str(ENV), 'python': sys.version.split()[0], 'tmpdir': str(tmp),
        'initialization_seconds': round(initialized-started, 6), 'parsing_seconds': round(parsed-initialized, 6),
        'authored_texts': 112, 'parser_passes_including_replay': 224,
        'deployment_manifest_sha256': DEPLOYMENT_SHA, 'report_parser': 'unchanged_official_classifier',
        'loader_ingestion': 'in_memory_official_clean_text2document_and_splitter_no_csv_io',
        'legacy_contract_transform': 'remove_only_unused_future_annotations_import_in_memory',
        'linguistic_rules_or_weights_changed': False, 'new_slurm_submissions': 0,
        'gpu_calls': 0, 'external_api': False, 'patient_or_candidate_inputs_read': False,
        'clinical_qualified': False, 'selection_changed': False, 'regeneration_authorized': False})
    print(json.dumps({'status': 'authored_predictions_frozen', 'authored_texts': 112,
                      'parser_passes': 224, 'predictions_sha256': closed['predictions.json'],
                      'parsing_seconds': round(parsed-initialized, 6)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'authored_parser_failed_closed', 'error_type': type(error).__name__,
                          'source_text_exposed': False}), file=sys.stderr)
        raise SystemExit(2)
