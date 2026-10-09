"""Invented numeric fixtures, never patient inputs or model generations."""
import copy
import json
import unittest

from test_radeval_length_controls import fixtures
from tricompose_v12.radeval_length_controls import build_table, CONTROLS
from tricompose_v12.radeval_image_length_controls import join, evaluate, METRICS, SELECTORS


def inputs():
    metadata = build_table(*fixtures())
    image = [{**{k: r[k] for k in ('item_id', 'source_id', 'source_group_id', 'section_id',
        'section_name', 'candidate_slot', 'reference_sha256', 'hypothesis_sha256')},
        'image_score_status': 'complete', 'paired_cohort_available': True,
        'scores': {m: r['candidate_slot'] / 4 for m in METRICS},
        'expert_outcomes': copy.deepcopy(r['expert_outcomes'])} for r in metadata]
    return image, metadata


class ImageLengthControlTests(unittest.TestCase):
    def setUp(self):
        self.inputs = inputs()

    def table(self):
        return join(*self.inputs)

    def result(self):
        return evaluate(self.table(), resamples=20, seed=0)

    def test_all_attempts_and_seven_selectors(self):
        r = self.result()
        self.assertEqual((r['all_attempted_pairs'], r['common_available_pairs']), (12, 12))
        self.assertEqual(r['selectors'], list(SELECTORS))
        self.assertEqual(len(r['results']), 14)
        self.assertEqual(len(r['paired_metric_control_comparisons']), 24)

    def test_fixed_higher_score_and_lower_control_directions(self):
        t = self.table()[0]
        self.assertEqual(t['scores'][METRICS[0]], .25)
        self.assertEqual([t['scores'][c] for c in CONTROLS], [-30, -20, -19])

    def test_shared_mask_keeps_missing_pair_and_whole_anchor_unavailable(self):
        self.inputs[0][0].update(image_score_status='unavailable_source_image',
            paired_cohort_available=False, scores=dict.fromkeys(METRICS))
        r = self.result()
        self.assertEqual(r['all_attempted_pairs'], 12)
        self.assertEqual(r['common_available_pairs'], 11)
        self.assertTrue(all(i['complete_anchors'] == 3 for i in r['results']))
        self.assertTrue(all(i['anchor_rows'][0]['status'] == 'incomplete_anchor' for i in r['results']))

    def test_missing_metadata_masks_image_scores_too(self):
        self.inputs[1][0].update(common_available=False,
            costs=dict.fromkeys(self.inputs[1][0]['costs']))
        self.assertTrue(all(v is None for v in self.table()[0]['scores'].values()))
        self.assertEqual(self.result()['original_image_available_pairs'], 12)
        self.assertEqual(self.result()['common_available_pairs'], 11)

    def test_paired_difference_is_score_selected_errors_minus_control(self):
        r = next(i for i in self.result()['paired_metric_control_comparisons']
            if i['metric'] == METRICS[0] and i['control'] == CONTROLS[0] and i['target'] == 'clinically_significant_total')
        self.assertEqual(r['metric_minus_control_mean_errors'], 2)
        self.assertEqual(r['paired_cluster_ci']['interval95'], [2, 2])

    def test_ties_use_expected_choice_not_first_slot(self):
        for r in self.inputs[0]:
            r['scores'] = {m: .5 for m in METRICS}
        r = next(i for i in self.result()['results']
            if i['selector'] == METRICS[0] and i['target'] == 'clinically_significant_total')
        self.assertEqual(r['means']['metric_minus_random_errors'], 0)
        self.assertEqual(r['pairwise_accuracy'], .5)

    def test_null_annotation_is_not_zero(self):
        for source in self.inputs:
            e = source[0]['expert_outcomes']
            e.update(clinically_significant_false_prediction=None,
                clinically_significant_total=None, all_errors_total=None)
        r = self.result()
        self.assertTrue(all(i['paired_rows'] == 11 and i['complete_anchors'] == 3 for i in r['results']))

    def test_no_available_scores_is_not_perfect(self):
        for r in self.inputs[0]:
            r.update(image_score_status='unavailable_source_image', paired_cohort_available=False,
                scores=dict.fromkeys(METRICS))
        result = self.result()
        self.assertEqual(result['common_available_pairs'], 0)
        self.assertTrue(all(r['means']['selected_expected_errors'] is None for r in result['results']))
        self.assertTrue(all(r['paired_cluster_ci']['interval95'] is None
                            for r in result['paired_metric_control_comparisons']))

    def rejected(self, side, key, value):
        self.inputs[side][0][key] = value
        with self.assertRaises(ValueError):
            self.table()

    def test_reordered_source_rejected(self):
        self.inputs[0].reverse()
        with self.assertRaises(ValueError):
            self.table()

    def test_duplicate_identity_rejected(self):
        for source in self.inputs:
            source[1] = copy.deepcopy(source[0])
        with self.assertRaises(ValueError):
            self.table()

    def test_extra_report_body_rejected(self):
        self.inputs[0][0]['report_text'] = 'Invented extra payload'
        with self.assertRaises(ValueError):
            self.table()

    def test_extra_metadata_payload_rejected(self):
        self.inputs[1][0]['report_text'] = 'Invented extra payload'
        with self.assertRaises(ValueError):
            self.table()

    def test_unequal_hash_binding_rejected(self):
        self.rejected(0, 'hypothesis_sha256', 'a' * 64)

    def test_raw_identity_rejected(self):
        for source in self.inputs:
            source[0]['source_id'] = 'raw_source_key'
        with self.assertRaises(ValueError):
            self.table()

    def test_extra_score_rejected(self):
        self.inputs[0][0]['scores']['unplanned_score'] = .5
        with self.assertRaises(ValueError):
            self.table()

    def test_missing_score_rejected(self):
        self.inputs[0][0]['scores'].pop(METRICS[0])
        with self.assertRaises(ValueError):
            self.table()

    def test_nonfinite_score_rejected(self):
        self.inputs[0][0]['scores'][METRICS[0]] = float('nan')
        with self.assertRaises(ValueError):
            self.table()

    def test_bool_score_rejected(self):
        self.inputs[0][0]['scores'][METRICS[0]] = True
        with self.assertRaises(ValueError):
            self.table()

    def test_out_of_range_reference_score_rejected(self):
        self.inputs[0][0]['scores'][METRICS[1]] = -.1
        with self.assertRaises(ValueError):
            self.table()

    def test_missing_status_with_numeric_score_rejected(self):
        self.inputs[0][0]['paired_cohort_available'] = False
        with self.assertRaises(ValueError):
            self.table()

    def test_complete_score_with_unavailable_image_rejected(self):
        self.rejected(0, 'image_score_status', 'unavailable_source_image')

    def test_unknown_status_rejected(self):
        self.rejected(0, 'image_score_status', 'not_applicable')

    def test_numeric_availability_rejected(self):
        self.rejected(0, 'paired_cohort_available', 1)

    def test_negative_metadata_rejected(self):
        self.inputs[1][0]['costs'][CONTROLS[0]] = -1
        with self.assertRaises(ValueError):
            self.table()

    def test_inconsistent_metadata_control_rejected(self):
        self.inputs[1][0]['costs'][CONTROLS[1]] += 1
        with self.assertRaises(ValueError):
            self.table()

    def test_modified_expert_count_rejected(self):
        self.inputs[0][0]['expert_outcomes']['clinically_significant_total'] += 1
        with self.assertRaises(ValueError):
            self.table()

    def test_extra_expert_field_rejected(self):
        for source in self.inputs:
            source[0]['expert_outcomes']['text'] = 'Invented extra payload'
        with self.assertRaises(ValueError):
            self.table()

    def test_invalid_slot_rejected(self):
        for source in self.inputs:
            source[0]['candidate_slot'] = 0
        with self.assertRaises(ValueError):
            self.table()

    def test_all_three_slots_required(self):
        self.inputs = tuple(source[:-1] for source in self.inputs)
        with self.assertRaises(ValueError):
            self.result()

    def test_deterministic_and_source_preserving(self):
        before = copy.deepcopy(self.inputs)
        self.assertEqual(self.result(), self.result())
        self.assertEqual(self.inputs, before)

    def test_no_patient_payload_in_outputs(self):
        encoded = json.dumps(self.table()) + json.dumps(self.result())
        for text in ('Invented reference', 'Invented candidate', '/invented/', '90000000'):
            self.assertNotIn(text, encoded)

    def test_no_clinical_promotion_or_new_model_calls(self):
        r = self.result()
        self.assertTrue(all(v is False for v in r['policy'].values()))
        self.assertTrue(r['post_hoc_development_diagnostic'])
        self.assertFalse(r['multiplicity_adjusted'])
        self.assertIsNone(r['clinical_score'])
        self.assertEqual(r['new_model_calls'], 0)

    def test_bounds_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.table(), resamples=0)

    def test_zero_entities_valid_metadata_not_perfect_quality(self):
        r = self.inputs[1][0]
        r['costs'][CONTROLS[2]] = r['native_entity_count'] = 0
        self.assertEqual(self.table()[0]['scores'][CONTROLS[2]], 0)
        self.assertFalse(self.result()['policy']['entity_count_is_clinical_quality'])
