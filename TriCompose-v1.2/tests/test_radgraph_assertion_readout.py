"""Authored native graphs only; no model, patient or source artifact access."""
import json
import unittest

from tricompose_v12.entity_gold_contract import count_metrics
from tricompose_v12.radgraph_assertion_readout import PATTERNS, pool, readout, assertion_reliability
from test_radgraph_literal_evidence import graph, POS, NEG, UNC, ANAT


class AssertionReadoutTests(unittest.TestCase):
    def read(self, source):
        return readout(source['text'], source)

    def test_empty_pool_unknown(self):
        self.assertEqual(pool([]), 'unknown')

    def test_known_states_preserved(self):
        for value in ('positive', 'negative', 'uncertain'):
            self.assertEqual(pool([value]), value)

    def test_mixed_assertions_not_majority_vote(self):
        self.assertEqual(pool(['positive', 'positive', 'negative']), 'uncertain')

    def test_uncertain_remains_uncertain(self):
        self.assertEqual(pool(['positive', 'uncertain']), 'uncertain')
        self.assertEqual(pool(['negative', 'uncertain']), 'uncertain')

    def test_unknown_is_not_a_native_label(self):
        with self.assertRaises(ValueError):
            pool(['unknown'])

    def test_literal_heads(self):
        for term, finding in (('cardiomegaly', 'cardiomegaly'), ('consolidation', 'consolidation'),
                              ('pleural effusion', 'pleural_effusion'), ('pneumothorax', 'pneumothorax')):
            row = self.read(graph([(term, POS, [])]))
            self.assertEqual(row['finding_states'][finding], 'positive')
            self.assertEqual(sum(v == 'positive' for v in row['finding_states'].values()), 1)

    def test_negative_head(self):
        row = self.read(graph([('pneumothorax', NEG, [])]))
        self.assertEqual(row['finding_states']['pneumothorax'], 'negative')

    def test_uncertain_head(self):
        row = self.read(graph([('cardiomegaly', UNC, [])]))
        self.assertEqual(row['finding_states']['cardiomegaly'], 'uncertain')

    def test_unmentioned_never_negative(self):
        row = self.read(graph([('authored_finding', POS, [])]))
        self.assertEqual(set(row['finding_states'].values()), {'unknown'})

    def test_no_finding_summary_not_expanded(self):
        row = self.read(graph([('No acute process', NEG, [])]))
        self.assertEqual(set(row['finding_states'].values()), {'unknown'})
        self.assertFalse(row['generic_normality_expanded_to_negatives'])

    def test_pleural_modifier_can_be_separate_entity(self):
        source = graph([('pleural', POS, [['modify', '2']]), ('effusion', NEG, [])])
        self.assertEqual(self.read(source)['finding_states']['pleural_effusion'], 'negative')

    def test_nonpleural_effusion_not_remapped(self):
        source = graph([('pericardial', ANAT, []), ('effusion', POS, [])])
        self.assertEqual(self.read(source)['finding_states']['pleural_effusion'], 'unknown')

    def test_measurement_not_presence(self):
        source = graph([('cardiomegaly', 'Observation::measurement::definitely present', [])])
        self.assertEqual(self.read(source)['finding_states']['cardiomegaly'], 'unknown')

    def test_anatomy_label_not_observation_presence(self):
        self.assertEqual(self.read(graph([('cardiomegaly', ANAT, [])]))['finding_states']['cardiomegaly'], 'unknown')

    def test_other_entity_negation_cannot_flip_target(self):
        source = graph([('consolidation', NEG, []), ('cardiomegaly', POS, [])])
        row = self.read(source)
        self.assertEqual(row['finding_states']['consolidation'], 'negative')
        self.assertEqual(row['finding_states']['cardiomegaly'], 'positive')

    def test_repeated_opposition_uncertain(self):
        source = graph([('pneumothorax', POS, []), ('pneumothorax', NEG, [])])
        self.assertEqual(self.read(source)['finding_states']['pneumothorax'], 'uncertain')

    def test_source_character_alignment_with_punctuation(self):
        source = {'text': 'Pleural effusion .', 'entities': {
            '1': {'tokens': 'effusion', 'start_ix': 1, 'end_ix': 1, 'label': POS, 'relations': []}}}
        row = readout('Pleural effusion.', source)
        evidence = row['source_span_references']['pleural_effusion'][0]
        self.assertEqual((evidence['char_start'], evidence['char_end_exclusive']), (8, 16))

    def test_changed_source_characters_rejected(self):
        with self.assertRaises(ValueError):
            readout('right', graph([('left', POS, [])]))

    def test_no_synonym_mapping(self):
        self.assertEqual(self.read(graph([('enlarged heart', POS, [])]))['finding_states']['cardiomegaly'], 'unknown')

    def test_missing_entities_valid_unknown_not_fabricated_negative(self):
        self.assertEqual(set(self.read({'text': 'cardiomegaly', 'entities': {}})['finding_states'].values()), {'unknown'})

    def test_readout_deterministic_and_unqualified(self):
        source = graph([('cardiomegaly', POS, [])])
        row = self.read(source)
        self.assertEqual(row, self.read(source))
        self.assertNotIn('text', row)
        self.assertFalse(row['clinical_qualified'])
        self.assertFalse(row['scope_semantics_corrected'])
        self.assertFalse(row['selection_changed'])
        self.assertFalse(row['regeneration_authorized'])
        self.assertIsNone(row['clinical_score'])

    def summary(self):
        return {'attempted_reports': 3, 'eligible_reports': 3,
            'entity_by_label': {
                POS: count_metrics(1, 1, 2), NEG: count_metrics(2, 1, 0), UNC: count_metrics(1, 0, 1)},
            'polarity': {'confusion_on_matched_observation_spans': {
                'positive': {'positive': 1, 'negative': 1, 'uncertain': 0},
                'negative': {'positive': 0, 'negative': 2, 'uncertain': 0},
                'uncertain': {'positive': 1, 'negative': 0, 'uncertain': 1}}}}

    def test_reliability_keeps_unmatched_denominator(self):
        row = assertion_reliability({'all': self.summary()})['all']
        self.assertEqual(row['confusion_with_unmatched']['positive']['unmatched'], 1)
        self.assertEqual(row['per_state']['positive']['end_to_end_recall'], 1 / 3)
        self.assertEqual(row['per_state']['positive']['conditional_matched_accuracy'], 1 / 2)

    def test_reliability_separates_uncertainty_and_hard_flips(self):
        row = assertion_reliability({'all': self.summary()})['all']
        self.assertEqual(row['positive_negative_flips_on_matched_spans'], 1)
        self.assertEqual(row['determinate_on_gold_uncertain_matched_spans'], 1)
        self.assertFalse(row['thresholds_fitted'])
        self.assertIsNone(row['scope_or_image_or_ehr_accuracy'])

    def test_reliability_rejects_inconsistent_counts(self):
        summary = self.summary()
        summary['polarity']['confusion_on_matched_observation_spans']['positive']['positive'] = 2
        with self.assertRaises(ValueError):
            assertion_reliability({'all': summary})


if __name__ == '__main__':
    unittest.main()
