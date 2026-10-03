"""Invented-fixture tests only: no model imports, real data or GPU work."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import verify_candidate_findings_qwen as verifier
import analyze_qwen_evidence as analysis


def states(**changes):
    return {**dict.fromkeys(verifier.FINDINGS, "unknown"), **changes}


def fixture():
    table = [{"triple_candidate_id": "report_a", "report_candidate_id": "report_a",
              "case_id": "synthetic_a", "cxr_candidate_id": "image_a",
              "cxr_model_id": "chexgenbench_sana", "cxr_seed": "0",
              "report_model_id": "maira2", "selected": "True", "score": "70.0"}]
    evidence = [{"report_candidate_id": "report_a", "case_id": "synthetic_a",
                 "parent_cxr_candidate_id": "image_a", "source_cxr_model_id": "chexgenbench_sana",
                 "report_model_id": "maira2", "image_sha256": "a" * 64,
                 "report_sha256": "b" * 64,
                 "ehr_finding_states": states(pneumonia="positive"),
                 "cxr_finding_states": states(pneumonia="positive"),
                 "report_finding_states": states(pneumonia="negative")}]
    scores = {"schema_version": verifier.SCHEMA, "primary_metric_eligible": False,
              "targeted_repair_approved": False, "selection_changed": False,
              "image_report_separation_enforced": True,
              "verifier_received_ehr_scores_or_winner_flags": False,
              "finding_order": list(verifier.FINDINGS), "records": [
                  {"candidate_id": "image_a", "input_kind": "image", "artifact_sha256": "a" * 64,
                   "contract_status": "complete", "states": states(pneumonia="negative")},
                  {"candidate_id": "report_a", "input_kind": "report", "artifact_sha256": "b" * 64,
                   "contract_status": "complete", "states": states(pneumonia="negative")}]}
    return table, evidence, scores


class SeparatedQwenReviewTests(unittest.TestCase):
    def test_image_request_has_no_report_or_score_prompt(self):
        sentinel = object()
        messages = verifier.request_messages("image", image=sentinel)
        self.assertIs(messages[0]["content"][0]["image"], sentinel)
        text = messages[0]["content"][1]["text"]
        self.assertNotIn("untrusted_report", text)
        self.assertNotIn('"score"', text)
        self.assertNotIn("candidate_id", text)

    def test_report_request_contains_no_image(self):
        messages = verifier.request_messages("report", report="Synthetic fixture text.")
        self.assertEqual([item["type"] for item in messages[0]["content"]], ["text"])
        self.assertIn("Missing findings are unknown", messages[0]["content"][0]["text"])

    def test_cannot_send_both_modalities(self):
        with self.assertRaises(ValueError):
            verifier.request_messages("image", image=object(), report="synthetic")
        with self.assertRaises(ValueError):
            verifier.request_messages("report", image=object(), report="synthetic")

    def test_named_finding_parser_preserves_unknown_and_uncertain(self):
        value = states(pneumonia="uncertain", edema="negative")
        self.assertEqual(verifier.parse_states(json.dumps(value)), value)
        self.assertEqual(verifier.parse_states("```json\n" + json.dumps(value) + "\n```"), value)

    def test_partial_vector_and_invalid_states_never_become_negative(self):
        for text in ('{"pneumonia":"positive"}', '"+-+-+-+-"',
                     json.dumps(states(pneumonia=True)), json.dumps(states(pneumonia="absent"))):
            result = verifier.decode_response(text)
            self.assertEqual(result["contract_status"], "failed_unavailable")
            self.assertEqual(set(result["states"].values()), {"unknown"})

    def test_extra_or_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            verifier.parse_states(json.dumps({**states(), "score": 1.0}))
        duplicate = json.dumps(states())[:-1] + ', "pneumonia": "negative"}'
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verifier.parse_states(duplicate)

    def test_slurm_guard_precedes_data_paths_and_torch_import(self):
        with patch.dict(verifier.os.environ, {}, clear=True), patch.object(verifier, "load_cxr_candidates") as load:
            with self.assertRaisesRegex(RuntimeError, "approved Slurm"):
                verifier.run(None)
            load.assert_not_called()

    def test_fixed_inventory_rejects_missing_or_duplicate_report_expert(self):
        images, reports = {}, {}
        experts = ("maira2", "cxrmate_single", "llavarad", "chexagent2")
        for number in range(12):
            image = f"image_{number}"
            case = "synthetic_a" if number < 6 else "synthetic_b"
            images[image] = {"case_id": case}
            for expert in experts:
                reports[f"{image}_{expert}"] = {"case_id": case, "parent_cxr_candidate_id": image,
                                               "model_id": expert}
        verifier.validate_inventory(images, reports, expected_images=12, expected_reports=48)
        reports["image_0_maira2"]["model_id"] = "llavarad"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verifier.validate_inventory(images, reports, expected_images=12, expected_reports=48)

    def test_unknown_does_not_supply_support_or_contradiction(self):
        result = analysis.compare(states(pneumonia="positive"), states(pneumonia="uncertain"))
        self.assertEqual(result["known"], 1)
        self.assertEqual(result["comparable"], 0)
        self.assertEqual(result["supported"], 0)
        self.assertEqual(result["contradictions"], 0)
        self.assertIsNone(result["agreement_on_comparable"])
        self.assertIsNone(analysis.compare(states(), states())["support_over_known"])

    def test_join_preserves_original_scores_and_selection_even_when_qwen_disagrees(self):
        table, evidence, scores = fixture()
        row = analysis.join_rows(table, evidence, scores)[0]
        self.assertEqual({key: row[key] for key in table[0]}, table[0])
        self.assertEqual(row["qwen_ehr_cxr_contradictions"], 1)
        self.assertEqual(row["qwen_cxr_report_supported"], 1)
        self.assertFalse(row["qwen_primary_metric_eligible"])
        self.assertFalse(row["qwen_targeted_repair_approved"])
        summary = analysis.summarize([row])
        self.assertEqual(summary["independent_ehr_cases"], 1)
        self.assertFalse(summary["selection_changed"])

    def test_changed_artifact_hash_is_rejected(self):
        table, evidence, scores = fixture()
        scores["records"][0]["artifact_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            analysis.join_rows(table, evidence, scores)

    def test_forged_primary_scope_or_missing_extraction_is_rejected(self):
        table, evidence, scores = fixture()
        changed = copy.deepcopy(scores)
        changed["primary_metric_eligible"] = True
        with self.assertRaisesRegex(ValueError, "verifier scope"):
            analysis.join_rows(table, evidence, changed)
        changed = copy.deepcopy(scores)
        changed["records"].pop()
        with self.assertRaisesRegex(ValueError, "fixed candidate inventory"):
            analysis.join_rows(table, evidence, changed)

    def test_failed_extraction_cannot_supply_positive_facts(self):
        table, evidence, scores = fixture()
        scores["records"][0]["contract_status"] = "failed_unavailable"
        with self.assertRaisesRegex(ValueError, "must not supply"):
            analysis.join_rows(table, evidence, scores)


if __name__ == "__main__":
    unittest.main()
