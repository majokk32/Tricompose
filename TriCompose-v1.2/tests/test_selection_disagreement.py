import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1] / "benchmarks" / "audit_selection_disagreement.py"
spec = importlib.util.spec_from_file_location("audit_selection_disagreement", MODULE)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _row(name, *, image, model, cosine, selected=False, fixed=False):
    return {
        "case_id": "case_000",
        "triple_candidate_id": name,
        "cxr_candidate_id": image,
        "cxr_model_id": "chexgenbench_sana" if fixed else "chexgenbench_pixart",
        "cxr_seed": "0" if fixed else "1",
        "report_model_id": "maira2" if fixed else model,
        "biovil_raw_cosine_secondary": str(cosine),
        "report_cxr_support_recall": "0.25",
        "total_hard_contradiction_count": "0",
        "selected": str(selected),
    }


class SelectionDisagreementTests(unittest.TestCase):
    def test_flags_low_rank_same_image_disagreement_without_changing_selection(self):
        rows = [
            _row("chosen", image="image_1", model="chexagent2", cosine=-0.02, selected=True),
            _row("alternate", image="image_1", model="llavarad", cosine=0.64),
            _row("fixed", image="image_0", model="maira2", cosine=0.83, fixed=True),
            _row("other", image="image_0", model="llavarad", cosine=0.71),
        ]
        result = audit.audit_rows(
            rows, fixed_cxr_model="chexgenbench_sana", fixed_seed=0,
            fixed_report_model="maira2"
        )
        self.assertEqual(len(result), 1)
        self.assertIs(result[0]["secondary_disagreement_review_flag"], True)
        self.assertEqual(result[0]["selected_triple_candidate_id"], "chosen")
        self.assertEqual(result[0]["selected_biovil_rank_within_case"], 4)

    def test_does_not_flag_minor_secondary_difference(self):
        rows = [
            _row("chosen", image="image_1", model="llavarad", cosine=0.75, selected=True),
            _row("alternate", image="image_1", model="chexagent2", cosine=0.8),
            _row("fixed", image="image_0", model="maira2", cosine=0.83, fixed=True),
            _row("other", image="image_0", model="llavarad", cosine=0.71),
        ]
        result = audit.audit_rows(
            rows, fixed_cxr_model="chexgenbench_sana", fixed_seed=0,
            fixed_report_model="maira2"
        )
        self.assertIs(result[0]["secondary_disagreement_review_flag"], False)

    def test_rejects_missing_fixed_path(self):
        rows = [
            _row("chosen", image="image_1", model="llavarad", cosine=0.75, selected=True),
            _row("alternate", image="image_1", model="chexagent2", cosine=0.8),
        ]
        with self.assertRaisesRegex(ValueError, "one selected and one fixed"):
            audit.audit_rows(
                rows, fixed_cxr_model="chexgenbench_sana", fixed_seed=0,
                fixed_report_model="maira2"
            )


if __name__ == "__main__":
    unittest.main()
