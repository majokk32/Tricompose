"""Small invented cached metadata only; no data/model/GPU work."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import build_finding_review as builder
from test_separated_qwen_review import fixture, states
import analyze_qwen_evidence as analysis


def review_fixture(two_reports=False):
    table, evidence, scores = fixture()
    table[0]["score"] = "70.0"
    if two_reports:
        other = {**table[0], "triple_candidate_id": "report_b", "report_candidate_id": "report_b",
                 "report_model_id": "llavarad", "selected": "False"}
        table.append(other)
        evidence.append({**evidence[0], "report_candidate_id": "report_b", "report_model_id": "llavarad",
                         "report_sha256": "c" * 64})
        scores["records"].append({**scores["records"][1], "candidate_id": "report_b", "artifact_sha256": "c" * 64})
    joined = analysis.join_rows(table, evidence, scores)
    source = {row["report_candidate_id"]: row for row in evidence}
    lineages = {row["triple_candidate_id"]: {
        "ehr_sha256": "e" * 64, "ehr_facts_sha256": "f" * 64,
        "cxr_candidate_id": row["cxr_candidate_id"], "report_candidate_id": row["report_candidate_id"],
        "cxr_sha256": source[row["triple_candidate_id"]]["image_sha256"],
        "report_sha256": source[row["triple_candidate_id"]]["report_sha256"]} for row in table}
    direct = {fact: {"state": "unknown", "evidence": [], "source_fields": []}
              for fact in builder.DIRECT_EHR_TO_CHEXPERT}
    direct["pneumonia"] = {"state": "positive", "evidence": ["diagnosis:invented"],
                           "source_fields": ["diagnoses[0]"]}
    staged = {"synthetic_a": {"schema_version": "tricompose-facts-v1.1", "missing_is_unknown": True,
                               "case_id": "synthetic_a", "direct_facts": direct}}
    return joined, source, lineages, staged


class FindingReviewTests(unittest.TestCase):
    def test_unknown_and_uncertain_are_neither_support_nor_negative(self):
        self.assertEqual(builder.relation("positive", "unknown")[0], "unknown")
        self.assertEqual(builder.relation("negative", "uncertain")[0], "not_comparable")
        self.assertEqual(builder.relation("unknown", "unknown")[0], "unknown")
        self.assertEqual(builder.relation("uncertain", "uncertain")[0], "not_comparable")

    def test_unavailable_verifier_is_distinct_from_unmentioned_finding(self):
        self.assertEqual(builder.relation("positive", "unknown", available=False),
                         ("not_comparable", "verifier_contract_unavailable"))

    def test_fixed_evidence_and_original_selection_are_retained(self):
        inputs = review_fixture()
        before = copy.deepcopy(inputs)
        facts, reviews = builder.build_rows(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(len(facts), 8)
        self.assertTrue(all(row["original_selected"] for row in facts))
        pneumonia = next(row for row in facts if row["finding"] == "pneumonia")
        self.assertEqual(pneumonia["ehr_provenance"]["source_fields"], ["diagnoses[0]"])
        self.assertEqual(pneumonia["artifact_hashes"]["ehr_sha256"], "e" * 64)
        self.assertIsNone(reviews[0]["confirmed_faulty_modality"])

    def test_scorer_disagreement_never_assigns_clinical_fault_or_repair(self):
        facts, reviews = builder.build_rows(*review_fixture(two_reports=True))
        self.assertTrue(any(row["relations"]["image_evaluators"] == "contradiction" for row in facts))
        self.assertTrue(all(not row["clinical_error_confirmed"] for row in facts))
        self.assertTrue(all(not row["automatic_repair_eligible"] for row in facts))
        self.assertTrue(all(row["confirmed_faulty_modality"] is None for row in reviews))

    def test_image_finding_counts_are_not_multiplied_by_report_count(self):
        facts, reviews = builder.build_rows(*review_fixture(two_reports=True))
        summary = builder.summarize(facts, reviews)
        self.assertEqual(summary["counts"]["fact_rows"], 16)
        self.assertEqual(summary["counts"]["unique_image_finding_pairs"], 8)
        self.assertEqual(summary["counts"]["independent_ehr_cases"], 1)
        self.assertEqual(summary["relation_counts"]["image_evaluators"]["contradiction"], 1)
        self.assertFalse(summary["clinical_fault_assigned"])
        self.assertFalse(summary["policy_executed"])

    def test_changed_image_hash_or_case_hash_is_rejected(self):
        inputs = review_fixture()
        inputs[2]["report_a"]["cxr_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            builder.build_rows(*inputs)
        inputs = review_fixture(two_reports=True)
        inputs[2]["report_b"]["ehr_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "fixed EHR hashes"):
            builder.build_rows(*inputs)

    def test_ehr_assertions_require_provenance_and_cannot_change_state(self):
        inputs = review_fixture()
        inputs[3]["synthetic_a"]["direct_facts"]["pneumonia"]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "lacks evidence"):
            builder.build_rows(*inputs)
        inputs = review_fixture()
        inputs[3]["synthetic_a"]["direct_facts"]["pneumonia"]["state"] = "negative"
        with self.assertRaisesRegex(ValueError, "differs from cached"):
            builder.build_rows(*inputs)

    def test_failed_report_contract_has_no_opposition_evidence(self):
        inputs = review_fixture()
        inputs[0][0]["qwen_report_contract_status"] = "failed_unavailable"
        import json
        inputs[0][0]["qwen_report_states"] = json.dumps(states())
        facts, reviews = builder.build_rows(*inputs)
        self.assertTrue(all(row["relations"]["qwen_image_report"] == "not_comparable" for row in facts))
        self.assertIn("secondary_report_contract_unavailable", reviews[0]["review_reasons"])

    def test_evidence_ids_are_deterministic_and_unique_per_finding(self):
        left, _ = builder.build_rows(*review_fixture())
        right, _ = builder.build_rows(*review_fixture())
        self.assertEqual(left, right)
        self.assertEqual(len({row["evidence_id"] for row in left}), 8)

    def test_oversized_cohort_is_refused_as_lightweight_pilot(self):
        table, source, lineages, facts = review_fixture()
        with self.assertRaisesRegex(ValueError, "bounded"):
            builder.build_rows(table * 49, source, lineages, facts)


if __name__ == "__main__":
    unittest.main()
