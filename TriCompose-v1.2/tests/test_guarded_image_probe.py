"""Invented pending intents and numeric image states; no models or artifacts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import run_guarded_image_probe as worker
from test_fresh_cxr_secondary import package
from test_image_evidence_guard import fixture, proposal
from tricompose_v12.invariant_verification import _digest


def invented():
    _, generation, _, _ = package(False)
    generation.update(data_origin="original_fully_synthetic_pool80", cached_triplets=[])
    cases, records = [], []
    for ordinal, case in enumerate(generation["cases"]):
        for row in case["cached_rows"]:
            row["cxr_sha256"] = _digest(["invented_probe_reference", ordinal])
            for fact in row["receipt"]["fact_states"]:
                if fact["finding"] == "edema": fact["xrv"] = "negative"
            generation["cached_triplets"].append({**{k: row[k] for k in
                ("case_id", "ehr_sha256", "ehr_facts_sha256", "cxr_sha256", "report_sha256",
                 "cxr_model_id", "report_model_id", "seed")}, "cxr_path": f"/invented/image_{ordinal}.png"})
        case["reference"] = case["cached_rows"][0]
        state, _ = fixture()
        cases.append({"reference": case["reference"], "cached_rows": case["cached_rows"],
            "numeric_state": state, "intent": proposal(state)})
        states = dict.fromkeys(worker.cached.observer.existing.image_interface.FINDINGS, "negative")
        states["edema"] = "positive"
        records.append({"cxr_candidate_id": case["reference"]["cxr_candidate_id"],
            "cxr_sha256": case["reference"]["cxr_sha256"], "contract_status": "complete", "states": states})
    return generation, cases, records


class GuardedImageProbeTests(unittest.TestCase):
    def test_reference_images_are_fixed_and_order_deterministic_not_probe_winners(self):
        generation, _, _ = invented()
        images = worker.two_reference_images(generation)
        self.assertEqual(len(images), 2)
        self.assertTrue(all(set(i) == {"cxr_candidate_id", "cxr_sha256", "path"} for i in images))
        generation["cases"].reverse(); generation["cached_triplets"].reverse()
        self.assertEqual(images, worker.two_reference_images(generation))

    def test_duplicate_case_image_and_nonreference_seed_rejected(self):
        for kind in ("case", "image", "seed"):
            generation, _, _ = invented()
            if kind == "case": generation["cases"][1]["case_id"] = generation["cases"][0]["case_id"]
            elif kind == "image":
                old = generation["cases"][0]["reference"]["cxr_sha256"]
                generation["cases"][1]["reference"]["cxr_sha256"] = old
                for triple in generation["cached_triplets"]:
                    if triple["case_id"] == generation["cases"][1]["case_id"]: triple["cxr_sha256"] = old
            else: generation["cases"][0]["reference"]["seed"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): worker.two_reference_images(generation)

    def test_fresh_disagreement_is_applied_before_backend_construction(self):
        _, cases, records = invented(); before = deepcopy((cases, records)); events = []
        results = worker.dispatch_cases(cases, records, events.append)
        self.assertEqual(len(results), 2)
        self.assertEqual(before, (cases, records))
        self.assertTrue(all(r["dispatch"]["status"] == "guard_abstained_unverified" for r in results))
        self.assertTrue(all(r["dispatch"]["actual_worker_model_calls"] == 0 for r in results))
        self.assertFalse(any(e["event"] == "backend_dispatch_reserved" for e in events))
        self.assertTrue(all(r["source_numeric_intent_is_cached"] for r in results))

    def test_unblocked_intent_is_deferred_not_executed_on_verifier_gpu(self):
        _, cases, records = invented()
        for record in records: record["states"]["edema"] = "negative"
        results = worker.dispatch_cases(cases, records, lambda event: None)
        self.assertTrue(all(r["dispatch"]["status"] == "deferred_separately_approved_backend_required" for r in results))
        self.assertTrue(all(r["dispatch"]["backend_dispatch_attempts"] == 0 for r in results))
        self.assertTrue(all(r["clinical_acceptance"] is False for r in results))

    def test_failed_or_uncertain_observer_never_becomes_disease_absence(self):
        for kind in ("failed", "uncertain", "unknown"):
            _, cases, records = invented()
            if kind == "failed": records[0].update(contract_status="failed_unavailable", states=None)
            else: records[0]["states"]["edema"] = kind
            results = worker.dispatch_cases(cases, records, lambda event: None)
            first = results[0]
            self.assertEqual(first["guard"]["counts"]["observer_ehr_opposition"], 0)
            self.assertEqual(first["dispatch"]["status"], "guard_abstained_unverified")

    def test_missing_duplicate_and_foreign_responses_fail_closed(self):
        _, cases, records = invented()
        for bad in (records[:1], [records[0], records[0]]):
            with self.assertRaises(ValueError): worker.dispatch_cases(cases, bad, lambda event: None)
        records[0]["cxr_sha256"] = "9" * 64
        with self.assertRaises(ValueError): worker.dispatch_cases(cases, records, lambda event: None)

    def test_wrong_current_intent_binding_fails_before_dispatch(self):
        _, cases, records = invented()
        cases[0]["cached_rows"][0] = deepcopy(cases[0]["cached_rows"][1])
        with self.assertRaises(ValueError): worker.dispatch_cases(cases, records, lambda event: None)

    def test_prepare_and_gpu_guards_precede_reads_models_and_writes(self):
        with patch.object(worker.cached.observer.source.previous.gate, "cpu_guard", side_effect=RuntimeError("invented_login")), \
             patch.object(worker.cached.observer.source.previous.postflight, "MetadataReader") as read, \
             patch.object(worker, "new_atomic_run") as write:
            with self.assertRaises(RuntimeError): worker.prepare(object())
            read.assert_not_called(); write.assert_not_called()
        with patch.object(worker, "gpu_guard", side_effect=RuntimeError("invented_no_gpu")), \
             patch.object(worker.cached.observer.source, "read_json") as read, \
             patch.object(worker, "new_atomic_run") as write:
            with self.assertRaises(RuntimeError): worker.run(object())
            read.assert_not_called(); write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
