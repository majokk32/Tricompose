"""Authored mock objects only; no parser/model/patient report is loaded."""
import ast
import copy
import importlib.util
import logging
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from tricompose_v12 import chexpert_negbio_contract as contract

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT/'interfaces/chexpert_negbio_legacy_worker_v1.py'
spec = importlib.util.spec_from_file_location('_tested_legacy_worker', WORKER_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def document():
    text = 'lung opacity'
    mention = NS(id='0', text='opacity', locations=[NS(offset=5, length=7)],
                 infons={'observation': 'Lung Opacity'})
    tokens = [NS(id=f'T{i}', text=word, locations=[NS(offset=offset, length=len(word))],
                 infons={'tag': 'NN', 'lemma': word}) for i, (word, offset) in enumerate((('lung', 0), ('opacity', 5)))]
    rel = NS(infons={'dependency': 'compound'},
             nodes=[NS(role='governor', refid='T1'), NS(role='dependant', refid='T0')])
    sentence = NS(text=text, offset=0, infons={'parse tree': '(ROOT invented)'}, annotations=tokens, relations=[rel])
    return NS(passages=[NS(text=text, offset=0, annotations=[mention], sentences=[sentence])])


class ErrorCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.count += 1


class FakeClassifier:
    def __init__(self, detector_module):
        self.parser = NS(parse_doc=lambda doc: doc)
        self.ptb2dep = NS(convert_doc=lambda doc: doc)
        self.detector = object()
        self.detector_module = detector_module
        self.calls = 0

    def classify(self, collection):
        self.calls += 1
        for doc in collection.documents:
            self.parser.parse_doc(doc)
            self.ptb2dep.convert_doc(doc)
            self.detector_module.detect(doc, self.detector)
            del doc.passages[0].sentences[:]


class LegacyWorkerTests(unittest.TestCase):
    def setUp(self):
        self.counter = ErrorCounter()
        self.previous_handlers = logging.getLogger().handlers
        self.previous_level = logging.getLogger().level
        logging.getLogger().handlers = [self.counter]
        logging.getLogger().setLevel(logging.ERROR)
        self.detector = NS(detect=lambda doc, detector: doc)
        self.classifier = FakeClassifier(self.detector)
        self.original_parse = self.classifier.parser.parse_doc
        self.original_convert = self.classifier.ptb2dep.convert_doc
        self.original_detect = self.detector.detect
        self.doc = document()
        self.monitor = worker.StageMonitor(self.counter)

    def tearDown(self):
        logging.getLogger().handlers = self.previous_handlers
        logging.getLogger().setLevel(self.previous_level)

    def checked(self, coverage=lambda doc: None):
        return worker.classify_checked(self.classifier, NS(documents=[self.doc]), contract,
                                       self.monitor, self.detector, coverage=coverage)

    def test_success_calls_original_method_and_captures_precleanup_receipt(self):
        receipt = self.checked()
        self.assertEqual(self.classifier.calls, 1)
        self.assertEqual(self.doc.passages[0].sentences, [])
        self.assertEqual(receipt['cleaned_text_sha256'], contract.digest('lung opacity'))
        self.assertFalse(receipt['syntax_correctness_verified'])

    def test_original_hooks_restored_on_success(self):
        self.checked()
        self.assertIs(self.classifier.parser.parse_doc, self.original_parse)
        self.assertIs(self.classifier.ptb2dep.convert_doc, self.original_convert)
        self.assertIs(self.detector.detect, self.original_detect)

    def test_swallowed_converter_error_is_not_complete(self):
        self.classifier.ptb2dep.convert_doc = lambda doc: logging.error('invented failure')
        with self.assertRaises(worker.StageFailure) as result:
            self.checked()
        self.assertEqual(result.exception.reason, 'dependency_graph_unavailable')
        self.assertEqual(self.monitor.stages['dependency']['errors'], 1)
        self.assertFalse(self.monitor.stages['detector']['attempted'])

    def test_swallowed_detector_error_is_not_positive_or_unknown(self):
        self.detector.detect = lambda doc, detector: logging.error('invented failure')
        with self.assertRaises(worker.StageFailure) as result:
            self.checked()
        row = contract.unavailable(result.exception.reason)
        self.assertEqual(row['failure_reason'], 'detector_error')
        self.assertIsNone(row['native_labels'])
        self.assertIsNone(row['mention_conflict_view'])

    def test_detector_failure_does_not_clean_up_unavailable_evidence(self):
        self.detector.detect = lambda doc, detector: logging.error('invented failure')
        original = self.detector.detect
        with self.assertRaises(worker.StageFailure):
            self.checked()
        self.assertTrue(self.doc.passages[0].sentences)
        self.assertIs(self.detector.detect, original)
        self.assertIs(self.classifier.parser.parse_doc, self.original_parse)

    def test_missing_tree_is_not_unknown(self):
        self.doc.passages[0].sentences[0].infons['parse tree'] = None
        with self.assertRaises(worker.StageFailure) as result:
            self.checked()
        self.assertEqual(result.exception.reason, 'parse_tree_unavailable')
        self.assertFalse(self.monitor.stages['detector']['attempted'])

    def test_missing_nodes_is_not_complete(self):
        self.doc.passages[0].sentences[0].annotations = []
        with self.assertRaises(worker.StageFailure) as result:
            self.checked()
        self.assertEqual(result.exception.reason, 'dependency_graph_unavailable')

    def test_coverage_failure_gates_detector(self):
        def unavailable(doc):
            raise worker.StageFailure('dependency_graph_unavailable')
        with self.assertRaises(worker.StageFailure):
            self.checked(coverage=unavailable)
        self.assertFalse(self.monitor.stages['detector']['attempted'])

    def test_alignment_failure_restores_hooks(self):
        self.doc.passages[0].annotations[0].text = 'changed'
        with self.assertRaises(worker.StageFailure) as result:
            self.checked()
        self.assertEqual(result.exception.reason, 'evidence_alignment_failed')
        self.assertIs(self.detector.detect, self.original_detect)

    def test_unknown_survives_successful_native_serialization(self):
        self.doc.passages[0].annotations = []
        receipt = self.checked()
        values = [math.nan]*14
        values[0] = 1
        output = contract.serialize(self.doc, 'lung opacity', values, receipt, detector_error_count=0)
        self.assertEqual(output['native_labels']['lung_opacity']['state'], 'unknown')
        self.assertEqual(output['native_labels']['no_finding']['state'], 'positive')

    def test_prior_error_counter_does_not_taint_next_pass(self):
        self.counter.count = 3
        self.checked()
        self.assertEqual(self.monitor.stages['detector']['errors'], 0)

    def test_unexpected_exception_is_sanitized(self):
        def failure(doc):
            raise RuntimeError('never serialize arbitrary input-bearing exception')
        self.classifier.parser.parse_doc = failure
        with self.assertRaises(worker.StageFailure) as result:
            self.checked()
        self.assertEqual(str(result.exception), 'parse_tree_unavailable')
        self.assertIs(self.classifier.parser.parse_doc, failure)

    def test_legacy_contract_only_removes_unused_future_import(self):
        legacy = worker.load_contract()
        self.assertEqual(legacy.CATEGORIES, contract.CATEGORIES)
        self.assertEqual(legacy.VERSION, contract.VERSION)
        for values in ([], ['positive'], ['negative', 'positive'], ['uncertain', 'positive']):
            self.assertEqual(legacy.conflict_view(values), contract.conflict_view(values))

    def test_legacy_contract_serialization_equivalent(self):
        legacy = worker.load_contract()
        doc = document()
        values = [math.nan]*14
        values[4] = 1
        self.assertEqual(legacy.serialize(doc, 'lung opacity', values, legacy.syntax_receipt(doc), detector_error_count=0),
                         contract.serialize(doc, 'lung opacity', values, contract.syntax_receipt(doc), detector_error_count=0))

    def test_future_contract_annotations_are_not_silently_transpiled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'authored_contract.py'
            path.write_text('from __future__ import annotations\ndef authored(value: str):\n    return value\n')
            with self.assertRaises(ValueError):
                worker.load_contract(path)

    def test_input_validation_rejects_reference_or_label_fields(self):
        with self.assertRaises(ValueError):
            worker.validate_inputs({'old48': [{'text': 'authored', 'expected_state': 'positive'}], 'new64': []})

    def test_worker_has_no_reference_or_candidate_input_loader(self):
        tree = ast.parse(WORKER_PATH.read_text())
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module in
                             ('opacity_context_controls_v1', 'opacity_assertion_controls_v2') for node in ast.walk(tree)))
        options = [node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call) and
                   isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument' and
                   node.args and isinstance(node.args[0], ast.Constant)]
        self.assertEqual(set(options), {'--plan-root', '--plan-manifest-sha256', '--output-directory'})

    def test_stage_completion_requires_no_new_error(self):
        self.monitor.call('parse', 'parse_tree_unavailable', lambda: 3)
        self.assertTrue(self.monitor.stages['parse']['complete'])
        self.assertFalse(self.monitor.stages['detector']['attempted'])


if __name__ == '__main__':
    unittest.main()
