import unittest

from tricompose_v12.radgraph_reference_contract_v2 import (
    NATIVE_LABELS, graph_metadata, score_table,
)


def graph(label):
    return {'text': 'fixture', 'entities': {'1': {
        'tokens': 'fixture', 'label': label, 'start_ix': 0, 'end_ix': 0,
        'relations': []}}}


class NativeXLVocabularyTests(unittest.TestCase):
    def test_all_eleven_native_labels_preserved(self):
        self.assertEqual(len(NATIVE_LABELS), 11)
        for label in NATIVE_LABELS:
            meta = graph_metadata(graph(label))
            self.assertEqual(meta['native_label_counts'][label], 1)
            self.assertEqual(sum(meta['native_label_counts'].values()), 1)

    def test_measurements_are_not_disease_assertions(self):
        for label in NATIVE_LABELS:
            if '::measurement::' in label:
                meta = graph_metadata(graph(label))
                self.assertEqual(meta['measurement_entity_count'], 1)
                self.assertEqual(sum(meta['nonmeasurement_observation_state_counts'].values()), 0)

    def test_anatomy_negative_is_not_negative_finding(self):
        meta = graph_metadata(graph('Anatomy::definitely absent'))
        self.assertEqual(meta['nonmeasurement_observation_state_counts']['negative'], 0)

    def test_measurement_graph_valid_with_native_score(self):
        g = graph('Observation::measurement::definitely present')
        native = ([1., 1., 1.], [[1.], [1.], [1.]], [g], [g])
        table = score_table(['pair_0000'], [True], native)
        self.assertEqual(table['records'][0]['hypothesis_graph']['measurement_entity_count'], 1)
        self.assertFalse(table['policy']['clinical_qualified'])
        self.assertNotIn('fixture', repr(table))

    def test_unsupported_ontology_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'unsupported_xl_label'):
            graph_metadata(graph('invented::positive'))

    def test_empty_pairs_remain_null(self):
        native = ([0., 0., 0.], [[0.], [0.], [0.]], [], [])
        table = score_table(['pair_0000'], [False], native)
        self.assertEqual(table['eligible_pairs'], 0)
        self.assertTrue(all(value is None for value in table['records'][0]['scores'].values()))


if __name__ == '__main__':
    unittest.main()
