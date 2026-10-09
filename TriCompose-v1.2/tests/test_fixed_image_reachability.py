"""Invented states only: no clinical artifact, model or network reads."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from tricompose_llm import fixed_image_reachability as s


def invented(ehr="positive", xrv="negative"):
    facts = [{"finding": n, "ehr": "unknown", "xrv": "unknown"} for n in s.FINDINGS]
    scores = dict.fromkeys(s.FINDINGS)
    thresholds = {n: {"enabled": i < 8, "negative_max": .3, "positive_min": .7}
        for i, n in enumerate(s.FINDINGS)}
    f = next(f for f in facts if f["finding"] == "edema")
    f.update(ehr=ehr, xrv=xrv)
    scores["edema"] = {"positive": .9, "negative": .1, "uncertain": .5, "unknown": None}[xrv]
    observer = {n: "unknown" for n in s.FINDINGS[:8]}
    return facts, scores, thresholds, observer


def analyze(values, **kwargs):
    return s.analyze(*values, observer_scope=list(s.FINDINGS[:8]), observer_status="complete", **kwargs)


class ReachabilityTests(unittest.TestCase):
    def test_all_sixteen_fixed_state_pairs_have_exact_four_symbolic_rows(self):
        for e, x in product(s.REPORT_STATES, repeat=2):
            with self.subTest(e=e, x=x):
                rows = s.truth_table(e, x)
                self.assertEqual([r["symbolic_report_state"] for r in rows], list(s.REPORT_STATES))
                self.assertEqual(sum(r["all_three_explicit_support"] for r in rows), int(e in s.EXPLICIT and x == e))
                self.assertTrue(all(r["symbolic_only_not_generated_or_selected"] for r in rows))

    def test_both_explicit_opposition_directions_are_unrepairable_by_report_only(self):
        for e, x in (("positive", "negative"), ("negative", "positive")):
            result = analyze(invented(e, x))
            b = result["bounds"]
            self.assertEqual(b["maximum_all_three_support_if_only_report_changes"], 0)
            self.assertEqual(b["minimum_remaining_fixed_ehr_cxr_oppositions"], 1)
            for row in s.truth_table(e, x):
                if row["symbolic_report_state"] in s.EXPLICIT:
                    self.assertEqual([row["ehr_report_relation"], row["cxr_report_relation"]].count("proxy_opposition"), 1)
                else:
                    self.assertFalse(row["known_ehr_report_comparable"])
                self.assertEqual(row["immutable_ehr_cxr_relation"], "proxy_opposition")

    def test_missing_image_label_is_missing_not_opposition_or_support(self):
        for state in ("unknown", "uncertain"):
            b = analyze(invented(xrv=state))["bounds"]
            self.assertEqual(b["known_ehr_facts"], 1)
            self.assertEqual(b["fixed_ehr_cxr_missing_facts"], 1)
            self.assertEqual(b["fixed_ehr_cxr_opposition_facts"], 0)
            self.assertEqual(b["maximum_all_three_support_over_known"], 0)

    def test_unknown_ehr_does_not_produce_zero_or_perfect_consistency(self):
        for state in ("unknown", "uncertain"):
            b = analyze(invented(ehr=state))["bounds"]
            self.assertEqual(b["known_ehr_facts"], 0)
            self.assertIsNone(b["maximum_all_three_support_over_known"])
            self.assertIsNone(b["every_known_ehr_fact_can_have_triple_support_under_fixed_labels"])

    def test_disabled_head_cannot_become_negative(self):
        values = invented(xrv="unknown")
        values[2]["edema"]["enabled"] = False
        values[1]["edema"] = .1
        result = analyze(values)
        edema = next(r for r in result["records"] if r["finding"] == "edema")
        self.assertIsNone(edema["unchanged_threshold_readout"]["score_minus_negative_boundary"])
        values[0][3]["xrv"] = "negative"
        with self.assertRaises(ValueError): analyze(values)

    def test_unavailable_observer_is_none_not_fabricated_states(self):
        values = invented(); values = (*values[:3], None)
        result = s.analyze(*values, observer_scope=list(s.FINDINGS[:8]), observer_status="failed_unavailable")
        self.assertTrue(all(r["observer_state"] is None for r in result["records"]))
        self.assertTrue(all(r["observer_ehr_relation"] is None for r in result["records"]))
        with self.assertRaises(ValueError):
            s.analyze(*invented(), observer_scope=list(s.FINDINGS[:8]), observer_status="failed_unavailable")

    def test_observer_agreement_does_not_change_bound_or_create_truth(self):
        values = invented(); before = deepcopy(values)
        a = analyze(values); values[3]["edema"] = "positive"; b = analyze(values)
        self.assertEqual(a["bounds"], b["bounds"])
        self.assertFalse(b["clinical_acceptance"])
        self.assertFalse(b["majority_vote_used"])
        self.assertIsNone(b["clinical_fault_location"])
        values[3]["edema"] = before[3]["edema"]
        self.assertEqual(values, before)

    def test_signed_distances_are_not_probabilities_or_confidence(self):
        record = s.threshold_readout(.1, {"enabled": True, "negative_max": .3, "positive_min": .7}, "negative")
        self.assertAlmostEqual(record["score_minus_negative_boundary"], -.2)
        self.assertFalse(record["probability_semantics"])
        self.assertFalse(record["boundary_distances_are_clinical_confidence"])
        self.assertFalse(record["threshold_changed"])

    def test_thresholds_and_scores_must_replay_exactly(self):
        for value in (float("nan"), float("inf"), -.1, 1.1, True):
            with self.assertRaises(ValueError):
                s.threshold_readout(value, {"enabled": True, "negative_max": .3, "positive_min": .7}, "negative")
        with self.assertRaises(ValueError):
            s.threshold_readout(.9, {"enabled": True, "negative_max": .3, "positive_min": .7}, "negative")
        with self.assertRaises(ValueError):
            s.threshold_readout(.5, {"enabled": True, "negative_max": .8, "positive_min": .7}, "uncertain")

    def test_exact_boundary_uses_existing_positive_first_tie_semantics(self):
        threshold = {"enabled": True, "negative_max": .5, "positive_min": .5}
        self.assertEqual(s.threshold_readout(.5, threshold, "positive")["state"], "positive")
        with self.assertRaises(ValueError): s.threshold_readout(.5, threshold, "negative")

    def test_symbolic_count_not_model_calls_and_inputs_unchanged(self):
        values = invented(); old = deepcopy(values); result = analyze(values)
        self.assertEqual(values, old)
        self.assertEqual(result["bounds"]["symbolic_state_slots"], 56)
        self.assertEqual(result["new_model_calls"], 0)
        self.assertEqual(result["report_states_generated"], 0)
        self.assertFalse(result["installed_as_policy"])

    def test_bound_matches_independent_exhaustive_small_invented_case(self):
        values = invented(ehr="unknown", xrv="unknown")
        refs = ("positive", "negative", "positive")
        images = ("positive", "positive", "unknown")
        for i, (e, x) in enumerate(zip(refs, images, strict=True)):
            values[0][i].update(ehr=e, xrv=x)
            values[1][s.FINDINGS[i]] = {"positive": .9, "negative": .1, "unknown": None}[x]
        b = analyze(values)["bounds"]
        counts = [sum(e in ("positive", "negative") and e == x == r for e, x, r in zip(refs, images, report))
            for report in product(s.REPORT_STATES, repeat=3)]
        self.assertEqual(b["maximum_all_three_support_if_only_report_changes"], max(counts))
        self.assertEqual(b["known_ehr_facts"], 3)
        self.assertEqual(b["maximum_all_three_support_over_known"], 1/3)

    def test_missing_duplicate_inventory_and_invalid_observer_scope_rejected(self):
        for which in ("missing", "duplicate", "observer"):
            values = invented()
            if which == "missing": values[0].pop()
            elif which == "duplicate": values[0][-1] = deepcopy(values[0][0])
            else: values[3].pop("edema")
            with self.assertRaises(ValueError): analyze(values)

    def test_actual_report_readout_does_not_count_silence_as_repair(self):
        from test_fresh_output_acceptance import fixture
        for state in s.REPORT_STATES:
            row, _ = fixture(image_state="negative", report_state=state)
            before = deepcopy(row)
            result = s.report_readout(row, row["receipt"]["fact_states"])
            self.assertEqual(result["all_three_supported_facts"], 0)
            self.assertTrue(result["actual_report_not_symbolic"])
            self.assertEqual(row, before)
            changed = deepcopy(row["receipt"]["fact_states"]); changed[3]["xrv"] = "positive"
            with self.assertRaises(ValueError): s.report_readout(row, changed)


if __name__ == "__main__": unittest.main()
