"""Invented native-output fixtures only; no models or clinical source files."""
import copy
import unittest

from tricompose_v12.radgraph_reference_contract import graph_metadata, score_table


def graph(label="Observation::definitely present"):
    return {"text": "edema left lung", "entities": {
        "1": {"tokens": "edema", "label": label, "start_ix": 0, "end_ix": 0,
              "relations": [["located_at", "2"]]},
        "2": {"tokens": "left lung", "label": "Anatomy::definitely present",
              "start_ix": 1, "end_ix": 2, "relations": []}}}


def result():
    return ([0.5, 0.25, 0.125], [[1., 0.], [.5, 0.], [.25, 0.]],
            [graph()], [graph()])


class RadGraphReferenceContractTests(unittest.TestCase):
    def test_entity_states_and_relations_preserved(self):
        for label, state in (("Observation::definitely present", "positive"),
                             ("Observation::definitely absent", "negative"),
                             ("Observation::uncertain", "uncertain")):
            meta = graph_metadata(graph(label))
            self.assertEqual(meta["native_state_counts"][state], 1)
            self.assertEqual(meta["relation_count"], 1)
            self.assertFalse(meta["scope_verified"])

    def test_missing_entity_is_not_negative(self):
        meta = graph_metadata({"text": "invented fixture", "entities": {}})
        self.assertEqual(meta["native_state_counts"]["negative"], 0)

    def test_unsupported_ontology_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported_xl_label"):
            graph_metadata(graph("OBS-DP"))

    def test_offsets_and_tokens_must_match(self):
        for field, value in (("start_ix", -1), ("end_ix", 50),
                             ("start_ix", True), ("tokens", "private fixture")):
            bad = graph()
            bad["entities"]["1"][field] = value
            with self.assertRaises(ValueError) as error:
                graph_metadata(bad)
            self.assertNotIn("private fixture", str(error.exception))

    def test_dangling_relation_rejected(self):
        bad = graph()
        bad["entities"]["1"]["relations"][0][1] = "3"
        with self.assertRaisesRegex(ValueError, "valid_relation_destination_required"):
            graph_metadata(bad)

    def test_score_components_not_collapsed(self):
        table = score_table(["pair_0000", "pair_0001"], [True, False], result())
        scores = list(table["records"][0]["scores"].values())
        self.assertEqual(scores, [1., .5, .25])
        self.assertFalse(table["policy"]["clinical_qualified"])
        self.assertFalse(table["policy"]["image_factuality_verified"])
        self.assertFalse(table["policy"]["ehr_consistency_verified"])

    def test_empty_pair_is_null_not_model_zero(self):
        table = score_table(["pair_0000", "pair_0001"], [True, False], result())
        row = table["records"][1]
        self.assertEqual(row["status"], "empty_input_not_eligible")
        self.assertTrue(all(v is None for v in row["scores"].values()))
        self.assertEqual(table["eligible_pairs"], 1)

    def test_valid_zero_score_is_retained(self):
        native = ([0., 0., 0.], [[0.], [0.], [0.]], [graph()], [graph()])
        row = score_table(["pair_0000"], [True], native)["records"][0]
        self.assertEqual(row["status"], "complete")
        self.assertEqual(set(row["scores"].values()), {0.})

    def test_missing_annotation_is_not_empty_or_unknown(self):
        native = list(result())
        native[2] = []
        with self.assertRaisesRegex(ValueError, "complete_nonempty_annotation_inventory_required"):
            score_table(["pair_0000", "pair_0001"], [True, False], native)

    def test_nan_and_boolean_scores_rejected(self):
        for value in (float("nan"), float("inf"), True, -0.1, 1.1):
            native = copy.deepcopy(result())
            native[1][0][0] = value
            with self.assertRaises(ValueError):
                score_table(["pair_0000", "pair_0001"], [True, False], native)

    def test_inventory_and_means_must_match(self):
        for change in ("mean", "width", "empty"):
            native = copy.deepcopy(result())
            if change == "mean":
                native[0][0] = 0.1
            elif change == "width":
                native[1][0].pop()
            else:
                native[1][0][1] = 0.2
                native[0][0] = 0.6
            with self.assertRaises(ValueError):
                score_table(["pair_0000", "pair_0001"], [True, False], native)

    def test_no_text_in_score_table_or_input_mutation(self):
        native = result()
        original = copy.deepcopy(native)
        table = score_table(["pair_0000", "pair_0001"], [True, False], native)
        self.assertNotIn("edema left lung", repr(table))
        self.assertNotIn("located_at", repr(table))
        self.assertEqual(native, original)

    def test_patient_identifier_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "bounded_unique_opaque_pair_ids_required"):
            score_table(["patient_private", "pair_0001"], [True, False], result())


if __name__ == "__main__":
    unittest.main()
