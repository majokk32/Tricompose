"""Invented image scores and reference labels only, no pixels or real inputs."""
import copy
import unittest

from tricompose_v12.image_reader_consensus import analyze, decision, POLICIES


def row(x=.8, margins=None, truth='positive', i=0):
    return {'case_id': f'case_{i:03d}', 'image_sha256': str(i) * 64,
            'xrv_score': x, 'margins': [.2, .1, .3] if margins is None else margins,
            'reference_state': truth}


def cohort():
    return [row(i=0), row(i=1), row(.2, [-.2, -.1, -.3], 'negative', 2),
            row(.8, [-.2, -.1, -.3], 'negative', 3)]


class ImageConsensusDecisionTests(unittest.TestCase):
    def test_xrv_exact_threshold_inclusive(self):
        self.assertEqual(decision(row(.5), POLICIES[0])['state'], 'positive')
        self.assertEqual(decision(row(.499), POLICIES[0])['state'], 'negative')

    def test_mean_uses_all_templates(self):
        self.assertEqual(decision(row(margins=[-.1, .2, .2]), POLICIES[1])['state'], 'positive')

    def test_mean_tie_abstains(self):
        value = decision(row(margins=[1., -1., 0.]), POLICIES[1])
        self.assertIsNone(value['state'])
        self.assertEqual(value['status'], 'abstain_mean_tie')

    def test_reader_disagreement_not_truth_vote(self):
        value = decision(row(.1), POLICIES[2])
        self.assertIsNone(value['state'])
        self.assertEqual(value['status'], 'abstain_reader_disagreement')

    def test_template_stability_is_not_mean_agreement(self):
        r = row(margins=[-.1, .2, .2])
        self.assertEqual(decision(r, POLICIES[2])['state'], 'positive')
        self.assertEqual(decision(r, POLICIES[3])['status'], 'abstain_template_sensitive')

    def test_template_tie_abstains_even_if_mean_positive(self):
        r = row(margins=[0., .2, .2])
        self.assertEqual(decision(r, POLICIES[3])['status'], 'abstain_template_tie')

    def test_failure_not_negative(self):
        r = row()
        r['xrv_score'] = None
        self.assertEqual(decision(r, POLICIES[0]), {'state': None, 'status': 'unavailable_reader'})
        self.assertEqual(decision(r, POLICIES[2])['status'], 'unavailable_reader')
        self.assertEqual(decision(r, POLICIES[1])['state'], 'positive')

    def test_biovil_failure_does_not_remove_xrv(self):
        r = row()
        r['margins'] = None
        self.assertEqual(decision(r, POLICIES[0])['state'], 'positive')
        self.assertEqual(decision(r, POLICIES[1])['status'], 'unavailable_reader')

    def test_no_favorable_template_subset(self):
        r = row()
        r['margins'] = [.2]
        with self.assertRaises(ValueError):
            decision(r, POLICIES[1])

    def test_nonfinite_or_out_of_range_scores_rejected(self):
        for value in (float('nan'), float('inf'), 1.1, True):
            with self.assertRaises(ValueError):
                decision(row(value), POLICIES[0])

    def test_nonfinite_template_rejected(self):
        with self.assertRaises(ValueError):
            decision(row(margins=[.2, float('nan'), -.2]), POLICIES[1])

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            decision(row(), 'best_observed_template')


class ImageReferenceDenominatorTests(unittest.TestCase):
    def test_external_reference_not_xrv_vote(self):
        result = analyze(cohort(), repetitions=100)
        xrv, _, mean, _ = result['rows']
        self.assertEqual(xrv['counts']['false_positive'], 1)
        self.assertEqual(xrv['metrics']['accepted_error_risk']['estimate'], .25)
        self.assertEqual(mean['counts']['accepted'], 3)
        self.assertEqual(mean['counts']['abstained'], 1)
        self.assertEqual(mean['metrics']['negative_reference_recovery']['estimate'], .5)

    def test_failed_image_kept_in_reference_recovery(self):
        rows = cohort()
        rows[0]['xrv_score'] = None
        r = analyze(rows, repetitions=100)['rows'][0]
        self.assertEqual(r['counts']['attempted'], 4)
        self.assertEqual(r['counts']['unavailable'], 1)
        self.assertEqual(r['metrics']['positive_reference_recovery']['estimate'], .5)

    def test_all_mean_unknown_risk_is_null(self):
        rows = cohort()
        for r in rows:
            r['margins'] = [1., -1., 0.]
        r = analyze(rows, repetitions=100)['rows'][1]
        self.assertIsNone(r['metrics']['accepted_error_risk']['estimate'])
        self.assertIsNone(r['metrics']['accepted_error_risk']['percentile_interval'])
        self.assertEqual(r['metrics']['accepted_error_risk']['zero_denominator_draws'], 100)

    def test_duplicate_image_cluster_rejected(self):
        rows = cohort()
        rows[1]['image_sha256'] = rows[0]['image_sha256']
        with self.assertRaises(ValueError):
            analyze(rows, repetitions=100)

    def test_duplicate_case_rejected(self):
        rows = cohort()
        rows[1]['case_id'] = rows[0]['case_id']
        with self.assertRaises(ValueError):
            analyze(rows, repetitions=100)

    def test_unknown_reference_not_negative(self):
        rows = cohort()
        rows[0]['reference_state'] = 'unknown'
        with self.assertRaises(ValueError):
            analyze(rows, repetitions=100)

    def test_one_class_only_rejected(self):
        with self.assertRaises(ValueError):
            analyze(cohort()[:2], repetitions=100)

    def test_deterministic_case_order_and_replay(self):
        rows = cohort()
        self.assertEqual(analyze(rows, repetitions=100), analyze(list(reversed(rows)), repetitions=100))

    def test_inputs_not_mutated(self):
        rows = cohort()
        original = copy.deepcopy(rows)
        analyze(rows, repetitions=100)
        self.assertEqual(rows, original)

    def test_four_masks_and_three_shared_draw_contrasts(self):
        r = analyze(cohort(), repetitions=100)
        self.assertEqual(len(r['rows']), 4)
        self.assertEqual(len(r['outcomes']), 16)
        self.assertEqual(len(r['paired_contrasts']), 3)
        self.assertTrue(all(x['shared_paired_draws'] for x in r['paired_contrasts']))

    def test_no_clinical_qualification_selection_or_repair(self):
        r = analyze(cohort(), repetitions=100)
        for name in ('clinical_qualified', 'best_policy_selected', 'thresholds_fitted', 'selection_changed', 'regeneration_authorized'):
            self.assertFalse(r[name])
        self.assertFalse(r['all_templates_are_independent_models'])
        self.assertFalse(r['sampling']['patient_source_binding_rechecked_by_this_module'])


if __name__ == '__main__':
    unittest.main()
