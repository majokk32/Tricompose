"""Invented language/contract fixtures; no checkpoint, patient data or GPU."""
import copy
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"benchmarks"))
import report_assertion_challenge as challenge
import score_report_assertion_challenge as scorer


def prediction(ref, status="complete"):
    return {"item_id": ref["item_id"], "report_sha256": ref["report_sha256"],
        "status": status, "finding_states": dict(ref["expected_states"])}


class AuthoredFixtureTests(unittest.TestCase):
    def test_fixed_inventory_and_distinct_texts(self):
        rows = challenge.authored_cases()
        self.assertEqual(len(rows), 56)
        self.assertEqual(len({row["report_sha256"] for row in rows}), 56)
        self.assertEqual(sum(len(row["evaluation_findings"]) for row in rows), 80)
        self.assertEqual({row["item_id"] for row in rows}, {f"fixture_{i:03d}" for i in range(56)})

    def test_deterministic_authoring(self):
        self.assertEqual(challenge.authored_cases(), challenge.authored_cases())

    def test_four_states_and_balanced_finding_targets(self):
        rows = challenge.authored_cases()
        states = {row["expected_states"][name] for row in rows for name in row["evaluation_findings"]}
        self.assertEqual(states, set(challenge.STATES))
        for name in challenge.FINDINGS:
            self.assertEqual(sum(name in row["evaluation_findings"] for row in rows), 20)

    def test_unmentioned_is_not_negative(self):
        for row in challenge.authored_cases():
            if row["family"] == "unmentioned_or_unassessed":
                self.assertEqual(set(row["expected_states"].values()), {"unknown"})

    def test_qualified_absence_is_uncertain(self):
        rows = [row for row in challenge.authored_cases() if row["family"] == "qualified_absence"]
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["expected_states"][row["evaluation_findings"][0]], "uncertain")

    def test_opposed_assertions_and_missing_assessment(self):
        rows = challenge.authored_cases()
        self.assertEqual(rows[-1]["expected_states"]["pleural_effusion"], "uncertain")
        self.assertTrue(all(row["expected_states"][row["evaluation_findings"][0]] == "unknown" for row in rows if row["family"] == "assessment_missing_not_absence"))

    def test_authored_reference_not_created_by_edit_constructor(self):
        source = inspect.getsource(challenge.authored_cases)
        self.assertNotIn("polarity_edit", source)
        self.assertNotIn("scope_check", source)
        self.assertNotIn("load_report", source)

    def test_full_responses_never_use_head_order_guess(self):
        values = [0]*14
        values[scorer.CHEXBERT_ORDER.index("cardiomegaly")] = 1
        values[scorer.CHEXBERT_ORDER.index("pleural_effusion")] = 2
        values[scorer.CHEXBERT_ORDER.index("pneumothorax")] = 3
        labels = scorer.decode_predictions(values)
        self.assertEqual(labels, {"cardiomegaly": "positive", "consolidation": "unknown", "pleural_effusion": "negative", "pneumothorax": "uncertain"})

    def test_invalid_head_classes_refused(self):
        for length, index, value in ((13, 0, 0), (14, 0, True), (14, 0, 1.0), (14, 0, 4), (14, 13, 2)):
            values = [0]*length
            values[index] = value
            with self.assertRaises(ValueError):
                scorer.decode_predictions(values)


class MetricsTests(unittest.TestCase):
    def test_perfect_authored_predictions_still_not_clinical_truth(self):
        refs = challenge.authored_cases()
        metrics, details = challenge.evaluate_predictions(refs, [prediction(ref) for ref in refs])
        self.assertEqual(metrics["designated_targets"]["checks"], 80)
        self.assertEqual(metrics["designated_targets"]["correct"], 80)
        self.assertEqual(metrics["designated_targets"]["macro_f1_present_classes"], 1.0)
        self.assertEqual(metrics["full_vectors_secondary"]["checks"], 224)
        self.assertEqual(len(details), 80)

    def test_four_class_macro_not_unknown_dominated_accuracy(self):
        rows = [{"expected": a, "predicted": b} for a,b in (("positive","positive"),("negative","positive"),("uncertain","unavailable"),("unknown","unknown"))]
        result = challenge.classification_counts(rows)
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["macro_f1_present_classes"], 5/12)
        self.assertEqual(result["unavailable"], 1)
        self.assertEqual(result["hard_positive_negative_flips"], 1)

    def test_invalid_contract_cannot_get_credit_for_unknown(self):
        ref = next(row for row in challenge.authored_cases() if row["family"] == "unmentioned_or_unassessed")
        metrics, _ = challenge.evaluate_predictions([ref], [prediction(ref, "failed_unavailable")])
        self.assertEqual(metrics["designated_targets"]["correct"], 0)
        self.assertEqual(metrics["designated_targets"]["unavailable"], 4)

    def test_missing_or_duplicate_predictions_not_dropped(self):
        refs = challenge.authored_cases()
        predictions = [prediction(ref) for ref in refs]
        for rows in (predictions[:-1], predictions+[predictions[0]]):
            with self.assertRaises(ValueError):
                challenge.evaluate_predictions(refs, rows)

    def test_hash_inventory_status_and_state_forgery_refused(self):
        refs = challenge.authored_cases()[:1]
        for field, value in (("report_sha256", "wrong"), ("status", "skipped"), ("finding_states", {})):
            row = prediction(refs[0]); row[field] = value
            with self.assertRaises(ValueError):
                challenge.evaluate_predictions(refs, [row])
        row = prediction(refs[0]); row["finding_states"]["cardiomegaly"] = "absent"
        with self.assertRaises(ValueError):
            challenge.evaluate_predictions(refs, [row])

    def test_uncertain_and_unknown_unsafe_commit_count(self):
        rows = [{"expected": a, "predicted": b} for a,b in (("unknown","negative"),("uncertain","positive"),("negative","unknown"))]
        self.assertEqual(challenge.classification_counts(rows)["unsafe_commit_on_unknown_or_uncertain"], 2)
        self.assertEqual(challenge.classification_counts(rows)["hard_positive_negative_flips"], 0)

    def test_empty_support_is_null_not_perfect(self):
        result = challenge.classification_counts([])
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["macro_f1_present_classes"])

    def test_metric_json_roundtrip_and_sources_immutable(self):
        refs = challenge.authored_cases()
        predictions = [prediction(ref) for ref in refs]
        before = copy.deepcopy((refs, predictions))
        result, _ = challenge.evaluate_predictions(refs, predictions)
        self.assertEqual(result, json.loads(json.dumps(result)))
        self.assertEqual(before, (refs, predictions))


class SafetyTests(unittest.TestCase):
    def test_slurm_guard_precedes_input_and_model_access(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(scorer, "load_inputs") as inputs:
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                scorer.run(SimpleNamespace())
            inputs.assert_not_called()

    def test_scorer_does_not_read_reference_key(self):
        source = inspect.getsource(scorer.run)
        self.assertNotIn("load_references", source)
        self.assertNotIn("references.jsonl", source)
        self.assertNotIn("authored_cases", source)
        self.assertIn("load_inputs", source)

    def test_blinded_resolver_rejects_reference_fields(self):
        rows = [{"item_id": f"fixture_{i:03d}", "report_path": f"reports/fixture_{i:03d}.txt", "report_sha256": str(i)} for i in range(56)]
        rows[0]["expected_state"] = "positive"
        manifest = {"schema_version": challenge.BANK_SCHEMA, "reports": 56, "patient_data_used": False,
            "primary_metric_eligible": False, "artifacts": {"resolver.jsonl": {"sha256": "pinned"}}}
        with patch.object(challenge, "require_inside", return_value=Path("invented")), patch.object(challenge, "read_json", return_value=manifest), patch.object(challenge, "sha256_file", return_value="pinned"), patch.object(Path, "read_text", return_value="\n".join(json.dumps(row) for row in rows)):
            with self.assertRaisesRegex(ValueError, "reference information leaked"):
                challenge.load_inputs("invented")

    def test_oracle_quote_audit_not_a_blind_model_accuracy(self):
        refs = challenge.authored_cases()[:1]
        refs.append(next(row for row in challenge.authored_cases() if row["family"] == "modal_may"))
        texts = {row["report_sha256"]: row["text"] for row in refs}
        decision = {"veto": False, "veto_reason": None, "scope_reasons": ["invented_unchecked"], "literal_mentions_checked": 0}
        def scope_stub(*_args):
            return decision

        with patch.object(challenge, "sha256_file", return_value=challenge.FROZEN_GUARD_SHA), patch.object(challenge, "load_inputs", return_value=([], texts, {})), patch.object(challenge, "load_references", return_value=(refs, Path("invented_key"))), patch("repair_cached_report_evidence.scope_check", new=scope_stub):
            summary, details, _ = challenge.oracle_quote_audit(SimpleNamespace(bank_run="invented"))
            self.assertEqual(len(details), 6)
            self.assertTrue(summary["oracle_quote_selection"])
            self.assertEqual(summary["unsafe_determinate_probes"], 3)
            self.assertEqual(summary["model_calls"], 0)
            self.assertIsNone(summary["independent_clinical_accuracy"])

    def test_changing_guard_after_challenge_is_refused(self):
        with patch.object(challenge, "sha256_file", return_value="changed"), patch.object(challenge, "load_inputs") as inputs:
            with self.assertRaisesRegex(ValueError, "pre-challenge veto code changed"):
                challenge.oracle_quote_audit(SimpleNamespace())
            inputs.assert_not_called()

    def test_nonoverwrite_precedes_staging(self):
        argv = ["challenge", "--mode", "stage", "--output-root", "invented", "--run-id", "invented"]
        with patch.object(sys, "argv", argv), patch.object(challenge, "new_atomic_run", side_effect=FileExistsError), patch.object(challenge, "stage") as stage, patch("builtins.print"):
            self.assertEqual(challenge.main(), 1)
            stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
