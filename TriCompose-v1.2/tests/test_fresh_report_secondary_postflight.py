"""Numeric fixture postflight; never loads weights or clinical artifacts."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import audit_fresh_report_secondary as audit
from test_fresh_report_secondary import data, endpoint


def fixture():
    rows, sels, ctxs, books = data()
    controls = audit.secondary.freeze_controls(rows, sels, ctxs, books, expected_cases=1)
    plan = {"rows": rows, "controls": controls, "maximum_image_encodings": 2, "maximum_text_encodings": 8}
    ep = endpoint(rows)
    ep.update(producer={"frozen": True, "checkpoint_sha256": audit.secondary.MODEL_HASHES,
        "text_policy": "full_report_no_silent_truncation_overlength_is_na"},
        original_selection_changed=False, primary_clinical_metric=False, peak_vram_gib=.5,
        counts={"requested_pairs": 4, "image_encoder_calls": 1, "text_encoder_calls": 4, "unavailable_reports": 0})
    _, _, summary = audit.secondary.endpoint_tables(rows, controls, ep)
    manifest = {"schema_version": audit.secondary.VERSION,
        "status": "completed_independent_endpoint_unvalidated", "charged_endpoint_worker_attempts": 1,
        "new_generator_calls": 0, "endpoint_used_for_selection": False, "clinical_acceptance": False,
        "encoding_counts": ep["counts"], "runtime_seconds_including_load_io": 2.0, "peak_vram_gib": .5}
    return plan, manifest, ep, summary


class FreshReportSecondaryPostflightTests(unittest.TestCase):
    def test_frozen_secondary_and_sealed_tables_verified(self):
        args = fixture(); before = copy.deepcopy(args)
        table, pairs, diagnostics = audit.verify_payload(*args)
        self.assertEqual(args, before)
        self.assertEqual(len(table), 4)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual(pairs[0]["llm_report_model"], "maira2")

    def test_higher_embedding_alternative_does_not_replace_choice(self):
        plan, manifest, ep, summary = fixture()
        _, pairs, diagnostics = audit.verify_payload(plan, manifest, ep, summary)
        self.assertTrue(any(d["higher_cosine_but_gate_blocked"] for d in diagnostics))
        self.assertEqual(pairs[0]["llm_report_model"], "maira2")
        self.assertEqual(plan["controls"]["choices"][0]["llm_candidate_id"], plan["rows"][1]["triple_candidate_id"])

    def test_forged_summary_and_encoding_costs_rejected(self):
        for field in ("summary", "encodings", "bool"):
            plan, manifest, ep, summary = fixture()
            if field == "summary": summary["llm_static_same_candidate_count"] = 0
            elif field == "encodings": ep["counts"]["text_encoder_calls"] -= 1
            else: ep["counts"]["image_encoder_calls"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.verify_payload(plan, manifest, ep, summary)

    def test_training_routing_or_clinical_claim_rejected(self):
        for field in ("training", "routing", "truth"):
            plan, manifest, ep, summary = fixture()
            if field == "training": ep["producer"]["frozen"] = False
            elif field == "routing": manifest["endpoint_used_for_selection"] = True
            else: manifest["clinical_acceptance"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.verify_payload(plan, manifest, ep, summary)

    def test_wrong_model_or_truncated_text_profile_rejected(self):
        for field, value in (("checkpoint_sha256", {}), ("text_policy", "silent_truncation")):
            plan, manifest, ep, summary = fixture()
            ep["producer"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.verify_payload(plan, manifest, ep, summary)

    def test_unavailable_report_remains_in_denominator_and_has_no_text_encoding(self):
        plan, manifest, ep, summary = fixture()
        ep["records"][1].update(biovil_raw_cosine=None, status="not_available",
            reason="full_report_exceeds_text_context_no_truncation")
        ep["counts"].update(text_encoder_calls=3, unavailable_reports=1)
        _, _, summary = audit.secondary.endpoint_tables(plan["rows"], plan["controls"], ep)
        _, pairs, diagnostics = audit.verify_payload(plan, manifest, ep, summary)
        self.assertIsNone(pairs[0]["llm_minus_fixed"])
        self.assertEqual(summary["requested_report_pairs"], 4)
        self.assertEqual(summary["available_pairs"], 3)

    def test_invalid_resource_measurement_rejected(self):
        for value in (float("inf"), -1.0, True):
            plan, manifest, ep, summary = fixture()
            manifest["runtime_seconds_including_load_io"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.verify_payload(plan, manifest, ep, summary)

    def test_cpu_guard_precedes_metadata_reads(self):
        with patch.object(audit.secondary.gate, "cpu_guard", side_effect=RuntimeError("invented_login")), \
             patch.object(audit.postflight, "MetadataReader") as reader:
            with self.assertRaises(RuntimeError): audit.audit(object())
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
