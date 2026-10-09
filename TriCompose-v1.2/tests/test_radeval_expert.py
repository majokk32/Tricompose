"""Invented schema fixtures only, never copied source patient examples."""
import copy
import json
import unittest

from tricompose_v12.radeval_expert import (
    CATEGORIES, METRICS, SEVERITIES, annotation_counts, evaluate, inventory,
    internal_group, outcomes,
)

DESCRIPTIONS = (
    'False prediction of finding', 'Omission of finding',
    'Incorrect location/position of finding', 'Incorrect severity of finding',
    'Mention of comparison that is not present in the reference',
    'Omission of a change from a previous study',
    'Inarticulate report (grammar and readability issues)',
)


def annotation(significant=0, insignificant=0):
    return '\n'.join([header + '\n' + '\n'.join(
        f'{i+1}. {label}: {value if i == 0 else 0}' for i, label in enumerate(DESCRIPTIONS))
        for header, value in [('**Significant:**', significant), ('Insignificant:', insignificant)]])


def row(index=0, reader='author_reader_a'):
    return {'annotator': reader, 'type': 'findings',
        'images_path': f'/invented/p90000000/s{90000001+index}/image.png',
        'ground_truth': f'Invented reference fixture number {index}.',
        **{f'prediction{s}': f'Invented candidate fixture {index} slot {s}.' for s in range(1, 4)},
        **{f'annotation{s}': annotation(s) for s in range(1, 4)}}


class ExpertCountsTests(unittest.TestCase):
    def test_full_explicit_counts(self):
        parsed = annotation_counts(annotation(2, 3))
        self.assertEqual(parsed['status'], 'complete')
        self.assertEqual(parsed['errors'][SEVERITIES[0]], [2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(outcomes(parsed['errors'])['all_errors_total'], 5)

    def test_blank_not_zero(self):
        parsed = annotation_counts('  ')
        self.assertEqual(parsed['status'], 'blank_unavailable')
        self.assertIsNone(outcomes(parsed['errors'])['all_errors_total'])

    def test_missing_terminal_count_is_null(self):
        text = annotation().replace('1. False prediction of finding: 0',
                                    '1. False prediction of finding:', 1)
        parsed = annotation_counts(text)
        self.assertEqual(parsed['errors'][SEVERITIES[0]][0], None)
        self.assertIsNone(outcomes(parsed['errors'])['clinically_significant_total'])
        self.assertEqual(outcomes(parsed['errors'])['clinically_insignificant_total'], 0)

    def test_negative_not_accepted(self):
        parsed = annotation_counts(annotation().replace('finding: 0', 'finding: -1', 1))
        self.assertIsNone(parsed['errors'][SEVERITIES[0]][0])

    def test_duplicate_category_invalidates_cell(self):
        text = annotation() + '\n1. False prediction of finding: 3'
        parsed = annotation_counts(text)
        self.assertIn('duplicate_category', parsed['issues'])
        self.assertTrue(all(x is None for v in parsed['errors'].values() for x in v))

    def test_duplicate_header_invalidates_cell(self):
        parsed = annotation_counts(annotation() + '\nSignificant:')
        self.assertIn('duplicate_severity_header', parsed['issues'])
        self.assertIsNone(outcomes(parsed['errors'])['all_errors_total'])

    def test_wrong_label_for_index_invalidates(self):
        parsed = annotation_counts(annotation().replace('1. False prediction of finding', '1. Omission of finding'))
        self.assertIn('unsupported_category_description', parsed['issues'])

    def test_missing_severity_invalidates(self):
        parsed = annotation_counts(annotation().split('Insignificant:')[0])
        self.assertIn('missing_severity_header', parsed['issues'])

    def test_count_limit(self):
        parsed = annotation_counts(annotation(1001))
        self.assertIn('count_out_of_bounds', parsed['issues'])

    def test_alphabetic_count_is_missing(self):
        parsed = annotation_counts(annotation().replace('finding: 0', 'finding: none', 1))
        self.assertIsNone(parsed['errors'][SEVERITIES[0]][0])

    def test_schema_bounded(self):
        with self.assertRaises(ValueError):
            annotation_counts('x' * 100001)


class ExpertInventoryTests(unittest.TestCase):
    def test_no_clinical_text_or_source_ids_exported(self):
        plan = inventory([row()])
        exported = json.dumps(plan)
        for forbidden in ('Invented reference', 'Invented candidate', '90000000', '90000001', 'author_reader_a', 'image.png'):
            self.assertNotIn(forbidden, exported)
        self.assertEqual(len(plan['records']), 3)

    def test_deterministic(self):
        self.assertEqual(inventory([row(0), row(1)]), inventory([row(0), row(1)]))

    def test_same_patient_cluster_across_studies(self):
        plan = inventory([row(0), row(1)])
        self.assertEqual(plan['source_keys'], 2)
        self.assertEqual(plan['source_groups'], 1)

    def test_chexpert_patient_cluster(self):
        self.assertEqual(internal_group('/fixture/patient100/study3/a.jpg'),
                         internal_group('/fixture/patient100/study4/b.jpg'))

    def test_unknown_path_explicit_fallback(self):
        self.assertEqual(internal_group('/fixture/unmapped/a.jpg')[0], 'author_source_key')

    def test_all_released_readers_mean(self):
        first, second = row(), row(reader='author_reader_b')
        second['annotation1'] = annotation(3)
        plan = inventory([first, second])
        self.assertEqual(len(plan['records']), 3)
        self.assertEqual(plan['records'][0]['released_reader_cells'], 2)
        self.assertEqual(plan['records'][0]['errors'][SEVERITIES[0]][0], 2)

    def test_missing_second_reader_not_silently_dropped(self):
        first, second = row(), row(reader='author_reader_b')
        second['annotation1'] = ''
        plan = inventory([first, second])
        self.assertIsNone(plan['records'][0]['errors'][SEVERITIES[0]][0])
        self.assertEqual(plan['records'][0]['released_reader_cells'], 2)

    def test_duplicate_reader_pair_rejected(self):
        with self.assertRaises(ValueError):
            inventory([row(), row()])

    def test_conflicting_source_texts_not_merged(self):
        first, second = row(), row(reader='author_reader_b')
        second['prediction1'] = 'A different invented candidate.'
        plan = inventory([first, second])
        self.assertEqual(len(plan['records']), 4)
        self.assertEqual(plan['source_slot_text_conflicts'], 1)

    def test_unknown_author_section_not_renamed_full_report(self):
        item = row()
        item['type'] = 'unfamiliar_source_enum'
        plan = inventory([item])
        self.assertEqual(plan['section_counts'], {'other_author_section': 3})

    def test_exact_schema(self):
        item = row()
        item['extra'] = 'x'
        with self.assertRaises(ValueError):
            inventory([item])

    def test_graph_cache_deduplicates_by_text_hash(self):
        item = row()
        item['prediction2'] = item['prediction1']
        plan = inventory([item])
        self.assertEqual(len(plan['graphs']), 3)
        self.assertEqual(len(plan['records']), 3)


class ExpertEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.plan = inventory([row(i) for i in range(4)])
        self.scores = [{'item_id': p['item_id'], 'status': 'complete',
            'scores': {m: (4 - p['candidate_slot']) / 4 for m in METRICS}}
            for p in self.plan['records']]

    def test_quality_correlates_with_negative_errors(self):
        result = evaluate(self.plan, self.scores, resamples=0)
        for metric in METRICS:
            self.assertAlmostEqual(result['metrics'][metric]['outcomes']['clinically_significant_total']['spearman'], 1)
        self.assertFalse(result['local_implementation_qualified'])

    def test_missing_scores_not_zero(self):
        self.scores[0] = {**self.scores[0], 'status': 'unavailable_graph', 'scores': {m: None for m in METRICS}}
        result = evaluate(self.plan, self.scores, resamples=0)
        self.assertEqual(result['metrics'][METRICS[0]]['complete_scores'], 11)

    def test_incomplete_reference_reduces_coverage(self):
        self.plan['records'][0]['errors'][SEVERITIES[0]][0] = None
        result = evaluate(self.plan, self.scores, resamples=0)
        self.assertEqual(result['metrics'][METRICS[0]]['outcomes']['clinically_significant_total']['paired_rows'], 11)

    def test_swapped_pair_join_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.plan, self.scores[::-1], resamples=0)

    def test_nan_rejected(self):
        self.scores[0]['scores'][METRICS[0]] = float('nan')
        with self.assertRaises(ValueError):
            evaluate(self.plan, self.scores, resamples=0)

    def test_unavailable_with_numeric_score_rejected(self):
        self.scores[0]['status'] = 'unavailable_graph'
        with self.assertRaises(ValueError):
            evaluate(self.plan, self.scores, resamples=0)

    def test_cluster_interval_does_not_claim_patient_independence(self):
        result = evaluate(self.plan, self.scores, resamples=100, seed=0)
        ci = result['metrics'][METRICS[0]]['outcomes']['clinically_significant_total']['cluster_bootstrap']
        self.assertEqual(ci['status'], 'insufficient_source_groups')
        self.assertEqual(ci['source_groups'], 1)


if __name__ == '__main__':
    unittest.main()
