"""Invented component inputs only; no parser/model or patient text is loaded."""
import copy
import itertools
import json
import math
from types import SimpleNamespace as NS
import unittest

from tricompose_v12 import chexpert_negbio_contract as contract


def document():
    text = 'opacity opacity'
    mentions, tokens = [], []
    for index, offset in enumerate((0, 8)):
        loc = NS(offset=offset, length=7)
        mentions.append(NS(id=str(index), text='opacity', locations=[loc],
                           infons={'observation': 'Lung Opacity'}))
        tokens.append(NS(id='T' + str(index), text='opacity', locations=[loc],
                         infons={'tag': 'NN', 'lemma': 'opacity'}))
    relation = NS(infons={'dependency': 'dep'},
                  nodes=[NS(role='governor', refid='T0'), NS(role='dependant', refid='T1')])
    sentence = NS(text=text, offset=0, infons={'parse tree': '(ROOT authored)'},
                  annotations=tokens, relations=[relation])
    return NS(passages=[NS(text=text, offset=0, annotations=mentions, sentences=[sentence])])


def labels(opacity=1):
    result = [math.nan] * 14
    result[4] = opacity
    return result


class ChexpertNegbioContractTests(unittest.TestCase):
    def output(self, doc=None, values=None, receipt=None, errors=0):
        doc = doc or document()
        return contract.serialize(doc, 'OPACITY\nOPACITY', values if values is not None else labels(),
                                  receipt or contract.syntax_receipt(doc), detector_error_count=errors)

    def test_exact_category_order(self):
        self.assertEqual(contract.CATEGORIES[4], 'Lung Opacity')
        self.assertEqual(len(set(contract.FINDINGS)), 14)

    def test_native_values(self):
        for value, state in ((1, 'positive'), (0, 'negative'), (-1, 'uncertain'), (math.nan, 'unknown')):
            self.assertEqual(contract.native_value(value)['state'], state)

    def test_invalid_values_not_missing(self):
        for value in (None, True, False, '1', '', 2, math.inf, -math.inf, object()):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                contract.native_value(value)

    def test_wrong_length_rejected(self):
        for size in (0, 13, 15):
            with self.assertRaises(ValueError):
                contract.native_vector([1] * size)

    def test_missing_unknown_not_negative(self):
        self.assertTrue(all(row['state'] == 'unknown' for row in contract.native_vector([math.nan] * 14).values()))

    def test_annotation_key_presence_not_truthiness(self):
        self.assertEqual(contract.mention_state({'negation': 'False'}), 'negative')
        self.assertEqual(contract.mention_state({'uncertainty': ''}), 'uncertain')

    def test_negation_precedes_uncertainty_native_semantics(self):
        self.assertEqual(contract.mention_state({'negation': 'True', 'uncertainty': 'True'}), 'negative')

    def test_all_mention_state_combinations(self):
        for size in range(4):
            for states in itertools.product(('positive', 'negative', 'uncertain'), repeat=size):
                expected = ('uncertain' if 'uncertain' in states or {'positive', 'negative'} <= set(states)
                            else states[0] if states else 'unknown')
                row = contract.conflict_view(states)
                self.assertEqual(row['state'], expected)
                self.assertFalse(row['independent_votes'])

    def test_unknown_not_a_classified_mention(self):
        with self.assertRaises(ValueError):
            contract.conflict_view(['unknown'])

    def test_complete_has_two_distinct_source_hashes(self):
        row = self.output()
        self.assertEqual(row['status'], 'complete')
        self.assertNotEqual(row['original_text_sha256'], row['cleaned_text_sha256'])
        self.assertFalse(row['original_offset_alignment_available'])

    def test_opposing_mentions_do_not_change_native_label(self):
        doc = document()
        doc.passages[0].annotations[1].infons['negation'] = 'True'
        row = self.output(doc)
        self.assertEqual(row['native_labels']['lung_opacity']['state'], 'positive')
        self.assertEqual(row['mention_conflict_view']['lung_opacity']['state'], 'uncertain')
        self.assertTrue(row['mention_conflict_view']['lung_opacity']['opposing_native_mentions'])

    def test_uncertain_mention_can_coexist_with_positive_native_label(self):
        doc = document()
        doc.passages[0].annotations[1].infons['uncertainty'] = 'True'
        row = self.output(doc)
        self.assertEqual(row['native_labels']['lung_opacity']['state'], 'positive')
        self.assertEqual(row['mention_conflict_view']['lung_opacity']['state'], 'uncertain')

    def test_no_finding_not_expanded_or_used_as_assertion(self):
        values = labels(math.nan)
        values[0] = 1
        doc = document()
        doc.passages[0].annotations = []
        row = self.output(doc, values)
        self.assertEqual(row['native_labels']['no_finding']['state'], 'positive')
        self.assertEqual(row['native_labels']['lung_opacity']['state'], 'unknown')
        self.assertIsNone(row['mention_conflict_view']['no_finding'])

    def test_no_quotes_or_text_bodies_in_output(self):
        serialized = json.dumps(self.output(), allow_nan=False)
        self.assertNotIn('OPACITY', serialized)
        self.assertNotIn('opacity opacity', serialized)
        self.assertNotIn('ROOT authored', serialized)

    def test_serialized_offsets_belong_to_cleaned_text(self):
        row = self.output()
        self.assertEqual(row['mentions'][1]['span']['char_start'], 8)
        self.assertEqual(row['mentions'][1]['span']['offset_space'], 'official_cleaned_full_report')

    def test_unicode_span(self):
        self.assertEqual(contract.span('é opacity', 2, 7)['quote_sha256'], contract.digest('opacity'))

    def test_invalid_offsets(self):
        for offset, length in ((-1, 2), (0, 0), (0, 99), (True, 1), (0, 1.0)):
            with self.assertRaises(ValueError):
                contract.span('opacity', offset, length)

    def test_annotation_text_mismatch(self):
        doc = document()
        doc.passages[0].annotations[0].text = 'changed'
        with self.assertRaises(ValueError):
            contract.syntax_receipt(doc)

    def test_repeated_annotation_id(self):
        doc = document()
        doc.passages[0].annotations[1].id = '0'
        with self.assertRaises(ValueError):
            contract.syntax_receipt(doc)

    def test_unknown_category(self):
        doc = document()
        doc.passages[0].annotations[0].infons['observation'] = 'Invented'
        with self.assertRaises(ValueError):
            contract.syntax_receipt(doc)

    def test_missing_parse_tree(self):
        doc = document()
        doc.passages[0].sentences[0].infons['parse tree'] = None
        with self.assertRaisesRegex(ValueError, 'parse_tree_unavailable'):
            contract.syntax_receipt(doc)

    def test_deleted_native_sentences_require_prior_receipt(self):
        doc = document()
        receipt = contract.syntax_receipt(doc)
        doc.passages[0].sentences = []
        self.assertEqual(self.output(doc, receipt=receipt)['status'], 'complete')
        with self.assertRaises(ValueError):
            contract.syntax_receipt(doc)

    def test_missing_graph_nodes(self):
        doc = document()
        doc.passages[0].sentences[0].annotations = []
        with self.assertRaisesRegex(ValueError, 'dependency_graph_unavailable'):
            contract.syntax_receipt(doc)

    def test_missing_graph_edges(self):
        doc = document()
        doc.passages[0].sentences[0].relations = []
        with self.assertRaisesRegex(ValueError, 'dependency_graph_unavailable'):
            contract.syntax_receipt(doc)

    def test_invalid_relation_reference(self):
        doc = document()
        doc.passages[0].sentences[0].relations[0].nodes[0].refid = 'missing'
        with self.assertRaisesRegex(ValueError, 'dependency_graph_unavailable'):
            contract.syntax_receipt(doc)

    def test_mention_without_matching_graph_node(self):
        doc = document()
        sentence = doc.passages[0].sentences[0]
        sentence.annotations = sentence.annotations[:1]
        sentence.relations = []
        with self.assertRaisesRegex(ValueError, 'dependency_graph_unavailable'):
            contract.syntax_receipt(doc)

    def test_receipt_bound_to_annotation_inventory(self):
        doc = document()
        receipt = contract.syntax_receipt(doc)
        doc.passages[0].annotations = doc.passages[0].annotations[:1]
        with self.assertRaises(ValueError):
            self.output(doc, receipt=receipt)

    def test_receipt_bound_to_cleaned_source(self):
        doc = document()
        receipt = contract.syntax_receipt(doc)
        doc.passages[0].text = doc.passages[0].text.upper()
        with self.assertRaises(ValueError):
            self.output(doc, receipt=receipt)

    def test_receipt_not_clinical_syntax_gold(self):
        receipt = contract.syntax_receipt(document())
        self.assertFalse(receipt['syntax_correctness_verified'])
        receipt['syntax_correctness_verified'] = True
        with self.assertRaises(ValueError):
            self.output(receipt=receipt)

    def test_caught_detector_error_is_not_complete_unknown(self):
        row = self.output(errors=1)
        self.assertEqual(row['status'], 'failed_unavailable')
        self.assertEqual(row['failure_reason'], 'detector_error')
        self.assertIsNone(row['native_labels'])

    def test_invalid_error_count(self):
        for value in (False, None, -1):
            with self.assertRaises(ValueError):
                self.output(errors=value)

    def test_unsupported_section_offsets(self):
        doc = document()
        receipt = contract.syntax_receipt(doc)
        doc.passages[0].offset = 10
        self.assertEqual(self.output(doc, receipt=receipt)['failure_reason'], 'unsupported_section_configuration')

    def test_failure_reasons_sanitized(self):
        for reason in contract.FAILURES:
            row = contract.unavailable(reason)
            self.assertIsNone(row['native_labels'])
            self.assertFalse(row['hard_action_eligible'])
        with self.assertRaises(ValueError):
            contract.unavailable('arbitrary source-bearing exception')

    def test_no_score_promotion_or_repair(self):
        row = self.output()
        self.assertIsNone(row['clinical_score'])
        self.assertFalse(row['hard_action_eligible'])
        self.assertFalse(row['regeneration_authorized'])

    def test_replay_exact(self):
        doc = document()
        self.assertEqual(self.output(doc), self.output(copy.deepcopy(doc)))


if __name__ == '__main__':
    unittest.main()
