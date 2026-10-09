"""Wholly invented strings/annotations; never read protected patient inputs."""
from copy import deepcopy
import hashlib
import unittest

from tricompose_v12.manual_opacity_inventory import project, summarize

LABELS = {'positive': 'Observation::definitely present', 'negative': 'Observation::definitely absent',
          'uncertain': 'Observation::uncertain'}


def projected(text, word, state):
    start = text.index(word)
    entities = [[start, start + len(word), LABELS[state]]] if state else []
    return project(text, entities, hashlib.sha256(text.encode()).hexdigest())


def record(index, projection=None, failure=None):
    return {'report_id': f'report_{index:04d}', 'source_index': index,
        'source_sha256': projection['source_sha256'] if projection else None,
        'status': 'complete' if projection else 'failed_unavailable',
        'failure_type': failure, 'projection': projection}


class ManualOpacityInventoryTests(unittest.TestCase):
    def test_positive_negative_uncertain_use_annotated_spans(self):
        for state in LABELS:
            p = projected('Lung opacity is described.', 'opacity', state)
            self.assertEqual(p['manual_literal_state'], state)
            self.assertEqual(p['literal_mention_count'], 1)
            self.assertEqual(len(p['human_span_evidence']), 1)
            self.assertIsNone(p['current_lung_opacity_reference'])

    def test_plural_and_case_preserved(self):
        text = 'Bilateral OPACITIES are described.'
        p = projected(text, 'OPACITIES', 'positive')
        self.assertEqual(p['manual_literal_state'], 'positive')
        e = p['human_span_evidence'][0]
        self.assertEqual(text[e['head_start']:e['head_end_exclusive']], 'OPACITIES')

    def test_unknown_is_not_negative(self):
        for text in ('The heart size is normal.', 'No acute findings.', 'There is an opacity.', 'Opacity is absent.'):
            p = project(text, [], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(p['manual_literal_state'], 'unknown')
            self.assertIsNone(p['current_lung_opacity_reference'])

    def test_unannotated_head_visible(self):
        p = project('There is an opacity.', [], hashlib.sha256(b'There is an opacity.').hexdigest())
        self.assertEqual(p['literal_mention_count'], 1)
        self.assertEqual(p['unannotated_literal_mentions'], 1)

    def test_synonyms_do_not_create_new_heads(self):
        for word in ('infiltrate', 'pneumonia', 'opacification'):
            p = projected(f'An {word} is described.', word, 'positive')
            self.assertEqual(p['manual_literal_state'], 'unknown')
            self.assertEqual(p['literal_mention_count'], 0)

    def test_qualifiers_anatomy_and_time_not_clinical_scope_gold(self):
        for text, state in (('No large opacity.', 'negative'), ('A bone opacity is seen.', 'positive'),
                            ('Previously an opacity was described.', 'positive')):
            p = projected(text, 'opacity', state)
            self.assertEqual(p['manual_literal_state'], state)
            self.assertFalse(p['qualifier_scope_verified'])
            self.assertFalse(p['pulmonary_anatomy_scope_verified'])
            self.assertFalse(p['current_patient_scope_verified'])
            self.assertFalse(p['regeneration_authorized'])
            self.assertIsNone(p['current_lung_opacity_reference'])

    def test_opposing_human_observations_pool_uncertain(self):
        text = 'Opacity. Opacities.'
        entities = [[0, 7, LABELS['positive']], [9, 18, LABELS['negative']]]
        p = project(text, entities, hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(p['manual_literal_state'], 'uncertain')

    def test_anatomy_entity_is_not_observation(self):
        text = 'Opacity is described.'
        p = project(text, [[0, 7, 'Anatomy::definitely present']], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(p['manual_literal_state'], 'unknown')

    def test_hash_bound_deterministic_no_input_mutation(self):
        text = 'An opacity is described.'
        entities = [[3, 10, LABELS['positive']]]
        original = deepcopy(entities)
        h = hashlib.sha256(text.encode()).hexdigest()
        self.assertEqual(project(text, entities, h), project(text, entities, h))
        self.assertEqual(entities, original)
        with self.assertRaises(ValueError):
            project(text, entities, '0' * 64)

    def test_invalid_or_duplicate_spans_rejected(self):
        text = 'An opacity is described.'
        h = hashlib.sha256(text.encode()).hexdigest()
        for entities in ([[[3, 10, LABELS['positive']]]], [[3, 10, LABELS['positive']], [3, 10, LABELS['negative']]],
                         [[True, 10, LABELS['positive']]], [[3, 100, LABELS['positive']]], [[3, 10, 'fake']]):
            with self.assertRaises(ValueError):
                project(text, entities, h)

    def test_summary_keeps_failures_duplicates_and_overlap(self):
        p = projected('Opacity is described.', 'Opacity', 'positive')
        q = projected('Opacity is uncertain.', 'Opacity', 'uncertain')
        rows = [record(0, p), record(1, p), record(2, q), record(3, failure='ValueError')]
        s = summarize(rows, previous_test_hashes={p['source_sha256']})
        self.assertEqual(s['attempted_reports'], 4)
        self.assertEqual(s['complete_reports'], 3)
        self.assertEqual(s['failed_reports'], 1)
        self.assertEqual(s['manual_literal_state_counts']['unknown'], 0)
        self.assertEqual(s['duplicate_source_hash_slots_retained'], 1)
        self.assertEqual(s['prior_test_source_hash_overlap_slots'], 2)
        self.assertFalse(s['patient_disjointness_verified'])
        self.assertFalse(s['checkpoint_training_overlap_verified'])

    def test_summary_cannot_drop_or_reorder_release_slots(self):
        p = projected('Opacity is described.', 'Opacity', 'positive')
        for rows in ([record(1, p)], [record(0, p), record(0, p)], [record(0, failure=None)]):
            with self.assertRaises(ValueError):
                summarize(rows, previous_test_hashes=set())


if __name__ == '__main__':
    unittest.main()
