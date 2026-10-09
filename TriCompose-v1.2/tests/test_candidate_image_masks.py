"""Small invented lineages and score caches; no reports, images or real data."""
import copy
import hashlib
import unittest

from tricompose_v12.candidate_image_masks import overlay, POLICIES, PREFIX, FAMILIES, number, relation


def fixture():
    rows, x, b = [], [], []
    for i, (score, margins) in enumerate(((.8, [.2, .2, .2]), (.8, [-.1, .2, .2]))):
        cid, image_id = f'case_{i:03d}', f'image_{i:03d}'
        digest = hashlib.sha256(image_id.encode()).hexdigest()
        fields = {'case_id': cid, 'cxr_candidate_id': image_id, 'cxr_sha256': digest, 'cxr_model_id': 'invented_model'}
        x.append({**fields, 'status': 'scored', 'exact_lung_opacity_score': score})
        pairs = {f: {'positive_cosine': v, 'negative_cosine': 0.} for f, v in zip(FAMILIES, margins)}
        b.append({**fields, 'status': 'scored', 'score_pairs': pairs})
        for j, state in enumerate(('positive', 'negative', 'unknown', 'uncertain')):
            r = {k: fields[k] for k in ('case_id', 'cxr_candidate_id', 'cxr_sha256')}
            r.update(triple_candidate_id=f'triple_{i}_{j}', ehr_sha256=digest, ehr_facts_sha256=digest,
                report_sha256=hashlib.sha256(f'report_{i}_{j}'.encode()).hexdigest(), report_model_id=f'report_model_{j}',
                opacity_cached_report_state=state, opacity_cached_ehr_state='unknown', opacity_selector_used='False',
                opacity_biovil_selector_used='False', opacity_clinical_accuracy='', opacity_biovil_clinical_accuracy='',
                opacity_exact_score=str(score), opacity_exact_state_0_5='positive', opacity_biovil_preference_state='positive',
                opacity_biovil_mean_margin=str(sum(margins)/3), preserved_legacy_value='do_not_change')
            for f, pair in pairs.items():
                for k, v in pair.items():
                    r['opacity_biovil_' + f + '_' + k] = str(v)
                r['opacity_biovil_' + f + '_margin'] = str(pair['positive_cosine'] - pair['negative_cosine'])
            rows.append(r)
    return rows, x, b


class CandidateMaskOverlayTests(unittest.TestCase):
    def test_all_source_cells_and_order_preserved(self):
        rows, x, b = fixture()
        original = copy.deepcopy((rows, x, b))
        output, _, summary = overlay(rows, x, b)
        self.assertEqual((rows, x, b), original)
        self.assertTrue(summary['original_cells_preserved'])
        self.assertTrue(summary['original_row_order_preserved'])
        self.assertTrue(all(out['preserved_legacy_value'] == 'do_not_change' for out in output))
        self.assertEqual(summary['added_columns'], 18)

    def test_template_stability_is_distinct_from_mean_agreement(self):
        output, _, s = overlay(*fixture())
        self.assertEqual(s['policy_summaries'][2]['accepted_image_slots'], 2)
        self.assertEqual(s['policy_summaries'][3]['accepted_image_slots'], 1)
        self.assertIsNone(output[-1][PREFIX + POLICIES[-1] + '_state'])

    def test_unknown_and_uncertain_reports_not_negatives(self):
        for state in ('unknown', 'uncertain'):
            self.assertEqual(relation('positive', state), 'not_comparable')
            self.assertEqual(relation('negative', state), 'not_comparable')

    def test_withheld_proposal_not_agreement(self):
        self.assertEqual(relation(None, 'negative'), 'not_comparable')

    def test_no_metric_or_truth_transfer(self):
        output, _, s = overlay(*fixture())
        for key in ('clinical_qualified', 'reference_metric_transferred', 'synthetic_domain_transport_validated',
                    'selection_changed', 'regeneration_authorized', 'ehr_opacity_reference_available'):
            self.assertFalse(s[key])
        self.assertTrue(all(r[PREFIX + 'clinical_primary_eligible'] is False for r in output))

    def test_four_report_slots_not_four_image_votes(self):
        _, _, s = overlay(*fixture())
        self.assertEqual((s['image_slots'], s['candidate_rows']), (2, 8))
        self.assertEqual(s['policy_summaries'][0]['accepted_image_slots'], 2)

    def test_withholding_retains_full_denominator(self):
        _, _, s = overlay(*fixture())
        last = s['policy_summaries'][-1]
        self.assertEqual(last['candidate_rows'], 8)
        self.assertEqual(last['not_comparable'], 6)
        self.assertEqual(last['comparable_coverage'], 2 / 8)

    def test_nested_loss_counts_do_not_claim_correctness(self):
        _, _, s = overlay(*fixture())
        loss = s['nested_mask_withholding'][-1]
        self.assertEqual(loss['image_slots_withheld'], 1)
        self.assertEqual(loss['proxy_support_rows_withheld'], 1)
        self.assertEqual(loss['proxy_opposition_rows_withheld'], 1)
        self.assertFalse(loss['removed_correct_or_incorrect_cases_adjudicated'])

    def test_duplicate_candidate_refused(self):
        rows, x, b = fixture()
        with self.assertRaises(ValueError):
            overlay(rows + [rows[0]], x, b)

    def test_missing_image_receipt_refused(self):
        rows, x, b = fixture()
        with self.assertRaises(ValueError):
            overlay(rows, x[:-1], b)

    def test_cross_case_join_refused(self):
        rows, x, b = fixture()
        rows[0]['case_id'] = 'other_case'
        with self.assertRaises(ValueError):
            overlay(rows, x, b)

    def test_fixed_ehr_hash_cannot_change(self):
        rows, x, b = fixture()
        rows[0]['ehr_sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            overlay(rows, x, b)

    def test_cached_score_mismatch_refused(self):
        rows, x, b = fixture()
        rows[0]['opacity_exact_score'] = '.3'
        with self.assertRaises(ValueError):
            overlay(rows, x, b)

    def test_cached_mean_mismatch_refused(self):
        rows, x, b = fixture()
        rows[0]['opacity_biovil_mean_margin'] = '.6'
        with self.assertRaises(ValueError):
            overlay(rows, x, b)

    def test_no_template_subset(self):
        rows, x, b = fixture()
        b[0]['score_pairs'].pop(FAMILIES[0])
        with self.assertRaises(ValueError):
            overlay(rows, x, b)

    def test_nonfinite_score_refused(self):
        for value in ('nan', 'inf', True):
            with self.assertRaises(ValueError):
                number(value)

    def test_duplicate_report_artifact_keeps_same_state(self):
        rows, x, b = fixture()
        rows[1]['report_sha256'] = rows[0]['report_sha256']
        with self.assertRaises(ValueError):
            overlay(rows, x, b)

    def test_preannotated_table_refused(self):
        rows, x, b = fixture()
        for r in rows:
            r[PREFIX + 'selector_used'] = 'False'
        with self.assertRaises(ValueError):
            overlay(rows, x, b)

    def test_failed_xrv_not_negative_or_dropped(self):
        rows, x, b = fixture()
        x[0].update(status='failed_without_replacement', exact_lung_opacity_score=None)
        for r in rows[:4]:
            r.update(opacity_exact_score='', opacity_exact_state_0_5='unknown')
        output, _, s = overlay(rows, x, b)
        self.assertIsNone(output[0][PREFIX + POLICIES[0] + '_state'])
        self.assertEqual(s['candidate_rows'], 8)
        self.assertEqual(s['policy_summaries'][0]['unavailable_image_slots'], 1)

    def test_failed_biovil_not_mean_tie_or_success(self):
        rows, x, b = fixture()
        b[0].update(status='failed_without_replacement', score_pairs=None)
        for r in rows[:4]:
            r.update(opacity_biovil_preference_state='unknown', opacity_biovil_mean_margin='')
            for f in FAMILIES:
                for suffix in ('positive_cosine', 'negative_cosine', 'margin'):
                    r['opacity_biovil_' + f + '_' + suffix] = ''
        output, _, s = overlay(rows, x, b)
        self.assertEqual(output[0][PREFIX + POLICIES[1] + '_status'], 'unavailable_reader')
        self.assertEqual(s['policy_summaries'][0]['accepted_image_slots'], 2)

    def test_inconsistent_duplicate_image_hash_remains_visible(self):
        rows, x, b = fixture()
        duplicate = x[0]['cxr_sha256']
        x[1]['cxr_sha256'] = b[1]['cxr_sha256'] = duplicate
        for r in rows[4:]:
            r['cxr_sha256'] = duplicate
        _, _, s = overlay(rows, x, b)
        self.assertEqual(s['distinct_image_artifacts'], 1)
        self.assertEqual(s['duplicate_image_hash_decision_variations'], 1)
        self.assertEqual(s['image_slots'], 2)

    def test_source_column_loss_refused(self):
        rows, x, b = fixture()
        rows[0].pop('preserved_legacy_value')
        with self.assertRaises(ValueError):
            overlay(rows, x, b)


if __name__ == '__main__':
    unittest.main()
