"""Token-boundary engineering checks only; no trained language model or data."""
import unittest

from tricompose_v12 import native_context_scope as v1
from tricompose_v12 import native_context_scope_v2 as v2
from tricompose_v12.chexpert_negbio_contract import span


def mentions(text, intervals):
    return [{'span': span(text, left, right-left)} for left, right in intervals]


class NativeBoundaryBridgeTests(unittest.TestCase):
    def doc(self, words, spaces=None):
        # spaCy storage/token operations, no pipeline/parser/model inference.
        from spacy.tokens import Doc
        from spacy.vocab import Vocab
        return Doc(Vocab(), words=words, spaces=spaces)

    def test_official_stem_boundary_becomes_strict_without_expansion(self):
        doc = self.doc(['opacity'], [False])
        self.assertIsNone(doc.char_span(0, 5, alignment_mode='strict'))
        result = v2.exact_token_bridge(doc, mentions(doc.text, [(0, 5)]))
        self.assertEqual(doc.text, 'opacity')
        self.assertEqual(doc.char_span(0, 5, alignment_mode='strict').text, 'opaci')
        self.assertEqual([token.text for token in doc], ['opaci', 'ty'])
        self.assertEqual(result['split_tokens'], 1)
        self.assertFalse(result['native_spans_expanded_or_contracted'])

    def test_multiple_cuts_preserve_exact_offsets(self):
        doc = self.doc(['abcdef'], [False])
        intervals = [(1, 4), (2, 6)]
        v2.exact_token_bridge(doc, mentions(doc.text, intervals))
        for left, right in intervals:
            self.assertEqual(doc.char_span(left, right, alignment_mode='strict').text, 'abcdef'[left:right])
        self.assertEqual(doc.text, 'abcdef')

    def test_overlapping_native_spans_do_not_drop_boundaries(self):
        doc = self.doc(['abcdef', 'ghij'], [True, False])
        original = doc.text
        intervals = [(0, 3), (2, 10), (8, 11)]
        v2.exact_token_bridge(doc, mentions(original, intervals))
        self.assertEqual(doc.text, original)
        for left, right in intervals:
            self.assertEqual(doc.char_span(left, right, alignment_mode='strict').text, original[left:right])

    def test_already_aligned_mentions_do_not_change_tokenization(self):
        doc = self.doc(['lung', 'opacity'], [True, False])
        result = v2.exact_token_bridge(doc, mentions(doc.text, [(5, 12)]))
        self.assertEqual(result['split_tokens'], 0)
        self.assertEqual([token.text for token in doc], ['lung', 'opacity'])

    def test_repeated_boundary_deduplicated(self):
        doc = self.doc(['opacity'], [False])
        result = v2.exact_token_bridge(doc, mentions(doc.text, [(0, 5), (0, 5)]))
        self.assertEqual(result['split_tokens'], 1)
        self.assertEqual(result['operations'][0]['split_char_offsets'], [5])

    def test_empty_inventory_does_not_modify_doc(self):
        doc = self.doc(['authored'], [False])
        result = v2.exact_token_bridge(doc, [])
        self.assertFalse(result['default_token_boundaries_changed'])

    def test_all_three_observed_stem_shapes_preserve_source(self):
        for word, stem in (('opacity', 'opaci'), ('atelectasis', 'atelecta'), ('consolidation', 'consolidat')):
            doc = self.doc([word], [False])
            result = v2.exact_token_bridge(doc, mentions(word, [(0, len(stem))]))
            self.assertEqual(doc.text, word)
            self.assertEqual(doc.char_span(0, len(stem), alignment_mode='strict').text, stem)
            self.assertFalse(result['source_text_changed'])

    def test_existing_parsed_doc_is_not_rewritten(self):
        from spacy.tokens import Doc
        from spacy.vocab import Vocab
        doc = Doc(Vocab(), words=['authored'], heads=[0], deps=['ROOT'])
        with self.assertRaises(ValueError):
            v2.exact_token_bridge(doc, [])

    def test_policy_explicitly_discloses_boundary_override_not_rule_fitting(self):
        self.assertTrue(v2.POLICY['default_token_boundaries_may_change'])
        self.assertFalse(v2.POLICY['source_or_native_span_expansion'])
        self.assertFalse(v2.POLICY['new_linguistic_or_anatomy_rules'])
        for key in ('hard_action_eligible', 'clinical_qualified', 'regeneration_authorized'):
            self.assertEqual(v2.POLICY[key], v1.POLICY[key])


if __name__ == '__main__':
    unittest.main()
