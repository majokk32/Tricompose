import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "benchmarks/analyze_candidate_scorers.py"
spec = importlib.util.spec_from_file_location("analyze_candidate_scorers", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def candidate(case, image, model, support, cosine, rank):
    return {
        "candidate_id": f"{case}_{image}_{model}", "case_id": case,
        "cxr_id": image, "cxr_model": "image_generator", "cxr_seed": 0,
        "report_model": model, "biovil_cosine": cosine,
        "support_recall": support, "total_hard_conflicts": 0,
        "eligible": True, "existing_rank": rank, "selected": rank == 1,
        "temporal_flag": False, "token_count": 20,
        "cxr_positive": 4, "cxr_negative": 0,
        "positive_support": int(support * 4), "negative_support": 0,
        "explicit_conflicts": 0,
    }


class CandidateScorerTests(unittest.TestCase):
    def test_unknown_is_neither_negative_support_nor_conflict(self):
        image = dict.fromkeys(module.CHEXPERT_FINDINGS, "unknown")
        report = image.copy()
        image["pneumonia"] = "positive"
        image["pneumothorax"] = "negative"
        image["cardiomegaly"] = report["cardiomegaly"] = "negative"
        report["support_devices"] = "positive"
        counts = module.finding_counts(image, report)
        self.assertEqual(counts["positive_support"], 0)
        self.assertEqual(counts["negative_support"], 1)
        self.assertEqual(counts["explicit_conflicts"], 0)
        self.assertEqual(counts["unresolved_positive"], 1)
        self.assertEqual(counts["unresolved_negative"], 1)
        self.assertEqual(counts["report_positive_cxr_unknown"], 1)

    def test_pair_comparisons_hold_image_fixed_and_keep_case_denominator(self):
        rows = [candidate("case_a", "image_a", "r1", 0.75, 0.1, 1),
                candidate("case_a", "image_a", "r2", 0.25, 0.8, 2),
                candidate("case_b", "image_b", "r1", 0.25, 0.1, 2),
                candidate("case_b", "image_b", "r2", 0.75, 0.8, 1)]
        result = module.summarize(rows)
        self.assertEqual(result["counts"]["independent_ehr_cases"], 2)
        self.assertEqual(result["counts"]["same_image_report_pairs"], 2)
        self.assertEqual([r["ordering"] for r in result["pair_comparisons"]],
                         ["discordant", "concordant"])

    def test_label_tie_is_not_counted_as_discordance(self):
        rows = [candidate("case_a", "image_a", "r1", 0.5, 0.1, 1),
                candidate("case_a", "image_a", "r2", 0.5, 0.8, 2)]
        result = module.summarize(rows)
        self.assertEqual(result["pair_comparisons"][0]["ordering"], "tied")
        self.assertEqual(result["per_case"][0]["strict_pairs"], 0)

    def test_one_image_cannot_have_different_case_lineage(self):
        rows = [candidate("case_a", "same_image", "r1", 0.5, 0.1, 1),
                candidate("case_b", "same_image", "r2", 0.5, 0.8, 2)]
        with self.assertRaisesRegex(ValueError, "inconsistent reference"):
            module.summarize(rows)

    def test_cannot_join_scores_to_a_different_image_hash(self):
        table = [{"triple_candidate_id": "r", "case_id": "case_a",
                  "cxr_candidate_id": "image_a", "cxr_model_id": "generator",
                  "cxr_seed": "0", "report_model_id": "reporter"}]
        lineage = [{"triple_candidate_id": "r", "case_id": "case_a", "lineage": {
            "cxr_candidate_id": "image_a", "cxr_sha256": "expected",
            "report_sha256": "report_hash", "report_model_id": "reporter",
            "cxr_model_id": "generator", "cxr_seed": 0}}]
        evidence = {"report_candidate_id": "r", "case_id": "case_a",
                    "parent_cxr_candidate_id": "image_a", "image_sha256": "different",
                    "report_sha256": "report_hash", "report_model_id": "reporter",
                    "source_cxr_model_id": "generator"}
        structure = {**evidence, "image_sha256": "expected"}
        with self.assertRaisesRegex(ValueError, "lineage/hash mismatch"):
            module.joined_rows(table, lineage, [evidence], [structure])


if __name__ == "__main__":
    unittest.main()
