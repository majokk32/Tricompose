"""Invented literal strings and annotation/state receipts only."""
from copy import deepcopy
import hashlib
import unittest

from tricompose_v12.manual_literal_assertions import FINDINGS, READERS, evaluate, project_reference

POS = 'Observation::definitely present'
NEG = 'Observation::definitely absent'
UNC = 'Observation::uncertain'
ANAT = 'Anatomy::definitely present'


def project(text, entities):
    return project_reference(text, entities, hashlib.sha256(text.encode()).hexdigest())


def fixtures():
    refs = [{'report_id': 'report_0000', 'source_sha256': 'a' * 64,
             'finding_states': dict.fromkeys(FINDINGS, 'positive')}]
    pred = {name: [{'report_id': 'report_0000', 'source_sha256': 'a' * 64,
                   'status': 'complete', 'finding_states': dict.fromkeys(FINDINGS, 'positive')}]
            for name in READERS}
    return refs, pred


class ManualLiteralAssertionTests(unittest.TestCase):
    def test_literal_annotated_presence(self):
        r = project('Cardiomegaly', [[0, 12, POS]])
        self.assertEqual(r['finding_states']['cardiomegaly'], 'positive')

    def test_no_annotation_is_unknown_not_negative(self):
        self.assertEqual(project('Cardiomegaly', [])['finding_states']['cardiomegaly'], 'unknown')

    def test_human_negative_preserved(self):
        self.assertEqual(project('pneumothorax', [[0, 12, NEG]])['finding_states']['pneumothorax'], 'negative')

    def test_human_uncertain_preserved(self):
        self.assertEqual(project('consolidation', [[0, 13, UNC]])['finding_states']['consolidation'], 'uncertain')

    def test_literal_pleural_modifier_needed(self):
        self.assertEqual(project('pleural effusion', [[8, 16, NEG]])['finding_states']['pleural_effusion'], 'negative')
        self.assertEqual(project('pericardial effusion', [[12, 20, POS]])['finding_states']['pleural_effusion'], 'unknown')

    def test_anatomy_is_not_presence(self):
        self.assertEqual(project('Cardiomegaly', [[0, 12, ANAT]])['finding_states']['cardiomegaly'], 'unknown')

    def test_no_synonyms_added(self):
        self.assertEqual(project('enlarged heart', [[0, 14, POS]])['finding_states']['cardiomegaly'], 'unknown')

    def test_generic_normality_not_expanded(self):
        self.assertEqual(set(project('No acute process', [[3, 16, NEG]])['finding_states'].values()), {'unknown'})

    def test_opposed_mentions_not_majority(self):
        r = project('cardiomegaly cardiomegaly', [[0, 12, POS], [13, 25, NEG]])
        self.assertEqual(r['finding_states']['cardiomegaly'], 'uncertain')

    def test_uncertain_with_positive_not_determinate(self):
        r = project('cardiomegaly cardiomegaly', [[0, 12, POS], [13, 25, UNC]])
        self.assertEqual(r['finding_states']['cardiomegaly'], 'uncertain')

    def test_wrong_source_hash_fails(self):
        with self.assertRaises(ValueError):
            project_reference('cardiomegaly', [[0, 12, POS]], 'a' * 64)

    def test_partial_or_outside_head_span_not_used(self):
        r = project('No cardiomegaly', [[0, 2, NEG]])
        self.assertEqual(r['finding_states']['cardiomegaly'], 'unknown')

    def test_invalid_offset_rejected(self):
        with self.assertRaises(ValueError):
            project('cardiomegaly', [[0, 13, POS]])

    def test_reference_contains_no_text(self):
        r = project('cardiomegaly', [[0, 12, POS]])
        self.assertNotIn('text', r)
        self.assertFalse(r['scope_verified'])
        self.assertFalse(r['official_report_label_reference'])

    def test_three_fixed_masks(self):
        r, p = fixtures()
        result, _ = evaluate(r, p)
        self.assertEqual(len(result['masks']), 3)
        self.assertFalse(result['best_policy_selected'])

    def test_mask_can_compare_imperfect_readers_using_actual_reference(self):
        r, p = fixtures()
        p['radgraph'][0]['finding_states'][FINDINGS[0]] = 'negative'
        result, _ = evaluate(r, p)
        self.assertEqual(result['readers']['radgraph']['overall']['hard_positive_negative_flips'], 1)
        self.assertEqual(result['masks']['agree_radgraph_chexbert']['correct_known_proposals'], 3)

    def test_failed_receipt_retained_not_unknown(self):
        r, p = fixtures()
        p['radgraph'][0].update(status='failed_unavailable', finding_states=None)
        result, _ = evaluate(r, p)
        self.assertEqual(result['readers']['radgraph']['overall']['unavailable_checks'], 4)
        self.assertEqual(result['masks']['agree_radgraph_chexbert']['proposal_coverage'], 0)

    def test_failed_cannot_have_unknown_vector(self):
        r, p = fixtures()
        p['radgraph'][0].update(status='failed_unavailable')
        with self.assertRaises(ValueError):
            evaluate(r, p)

    def test_missing_receipt_rejected(self):
        r, p = fixtures()
        p['radgraph'] = []
        with self.assertRaises(ValueError):
            evaluate(r, p)

    def test_duplicate_reference_rejected(self):
        r, p = fixtures()
        r.append(deepcopy(r[0]))
        with self.assertRaises(ValueError):
            evaluate(r, p)

    def test_different_source_hash_rejected(self):
        r, p = fixtures()
        p['radgraph'][0]['source_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            evaluate(r, p)

    def test_unknown_promotion_not_clinical_false_positive(self):
        r, p = fixtures()
        r[0]['finding_states'] = dict.fromkeys(FINDINGS, 'unknown')
        result, _ = evaluate(r, p)
        row = result['masks']['agree_radgraph_chexbert']
        self.assertEqual(row['determinate_on_unknown_literal_reference'], 4)
        self.assertEqual(row['hard_positive_negative_flips'], 0)
        self.assertFalse(row['unknown_promotions_are_clinical_errors'])

    def test_uncertain_reference_not_forced_negative(self):
        r, p = fixtures()
        r[0]['finding_states'] = dict.fromkeys(FINDINGS, 'uncertain')
        result, _ = evaluate(r, p)
        self.assertEqual(result['masks']['agree_radgraph_chexbert']['determinate_on_uncertain_reference'], 4)
        self.assertEqual(result['readers']['radgraph']['overall']['failure_aware_uncertain_recovery'], 0)

    def test_unknown_agreement_not_support(self):
        r, p = fixtures()
        for name in READERS:
            p[name][0]['finding_states'] = dict.fromkeys(FINDINGS, 'unknown')
        result, _ = evaluate(r, p)
        row = result['masks']['agree_radgraph_chexbert']
        self.assertEqual(row['accepted_determinate_proposals'], 0)
        self.assertIsNone(row['conditional_known_error_rate'])


if __name__ == '__main__':
    unittest.main()
