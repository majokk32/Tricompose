"""Invented images/reports/labels only; no deployed model is instantiated."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_validation"))
import rsua_medical_report_observer as w
from test_rsua_qwen_observer import inputs_and_inspector


def generated(*, tokens=30, text="FINDINGS:\nInvented fixture."):
    return {"text": text, "metadata": {"fixture_only": True},
        "tokens": {"input_tokens": 10, "output_tokens": tokens, "token_limit_reached": tokens >= 512}}


def report_rows():
    inputs, inspect, _ = inputs_and_inspector()
    return w.generate_records(inputs, inspect=lambda i: inspect(i)[0],
        generate=lambda i: generated(), save=lambda i, t: {"report_path": "reports/" + i["case_id"] + ".txt",
            "report_sha256": "b"*64}, journal=lambda e: None)


class MedicalObserverTests(unittest.TestCase):
    def reports(self, **kwargs):
        inputs, inspect, _ = inputs_and_inspector(uniform_first=kwargs.pop("uniform_first", False))
        default = dict(inspect=lambda i: inspect(i)[0], generate=Mock(return_value=generated()),
            save=lambda i, t: {"report_path": "reports/" + i["case_id"] + ".txt", "report_sha256": "b"*64},
            journal=lambda e: None)
        default.update(kwargs)
        return w.generate_records(inputs, **default)

    def labels(self, reports=None, **kwargs):
        default = dict(load=Mock(), length=lambda r: (20, 512),
            infer=Mock(return_value=dict.fromkeys(w.CHEXPERT_FINDINGS, "unknown")), journal=lambda e: None)
        default.update(kwargs)
        return w.label_records(reports or report_rows(), **default)

    def test_all_generated_reports_preserve_hash_order_and_actual_calls(self):
        rows = self.reports()
        self.assertEqual([r["case_id"] for r in rows], ["invented_0", "invented_1", "invented_2"])
        self.assertTrue(all(r["model_called"] and r["report_status"] == "generated_available" for r in rows))
        self.assertTrue(all(r["source_image_sha256"] and r["png_sha256"] and r["report_sha256"] for r in rows))

    def test_uniform_image_is_blocked_without_model_call(self):
        cb = Mock(return_value=generated()); rows = self.reports(uniform_first=True, generate=cb)
        self.assertEqual(cb.call_count, 2)
        self.assertFalse(rows[0]["model_called"])
        self.assertEqual(rows[0]["report_status"], "blocked_before_model_call")
        self.assertIsNone(rows[0]["report_path"])

    def test_report_reservation_happens_before_callback(self):
        events = []
        def cb(item):
            self.assertEqual(events[-1]["event"], "report_request_reserved")
            return generated()
        self.reports(generate=cb, journal=events.append)
        self.assertEqual(len(events), 6)

    def test_journal_failure_prevents_report_call(self):
        cb = Mock()
        with self.assertRaises(OSError): self.reports(generate=cb, journal=Mock(side_effect=OSError()))
        cb.assert_not_called()

    def test_report_runtime_failure_is_charged_then_stops_without_retry(self):
        cb = Mock(side_effect=RuntimeError("invented")); events = []
        rows = self.reports(generate=cb, journal=events.append)
        self.assertEqual(cb.call_count, 1)
        self.assertEqual([r["report_status"] for r in rows], ["runtime_failed_unavailable"] +
            ["not_attempted_after_runtime_failure"]*2)
        self.assertEqual(sum(r["model_called"] for r in rows), 1)
        self.assertEqual(sum(e["event"] == "report_request_reserved" for e in events), 1)

    def test_token_cap_and_empty_are_unavailable_not_negative(self):
        for value, expected in ((generated(tokens=512), "token_cap_unavailable"),
            (generated(text=" "), "empty_unavailable")):
            rows = self.reports(generate=lambda item: deepcopy(value))
            load = Mock(); output, _ = self.labels(rows, load=load)
            load.assert_not_called()
            self.assertTrue(all(r["report_status"] == expected and r["model_called"] for r in output))
            self.assertTrue(all(r["contract_status"] == "failed_unavailable" and r["states"] is None for r in output))

    def test_changed_guard_hash_rejected_before_generation(self):
        inputs, inspect, _ = inputs_and_inspector(); cb = Mock()
        def bad(item):
            receipt = inspect(item)[0]
            return w.prior.guard.receipt("f"*64, receipt["status"], receipt["reason"],
                evidence=receipt["native_image_metadata"], pixel_sha256=receipt["normalized_pixel_sha256"])
        with self.assertRaises(ValueError): w.generate_records(inputs, inspect=bad, generate=cb, save=Mock(), journal=Mock())
        cb.assert_not_called()

    def test_duplicate_empty_and_over_budget_scope_rejected(self):
        inputs, inspect, _ = inputs_and_inspector(); cb = Mock()
        for invalid in ([], [inputs[0], inputs[0]], [dict(inputs[0], case_id=str(i), png_sha256=format(i, "064x")) for i in range(51)]):
            with self.assertRaises(ValueError): w.generate_records(invalid, inspect=inspect, generate=cb, save=Mock(), journal=Mock())
        cb.assert_not_called()

    def test_invalid_token_receipt_cannot_claim_report_completion(self):
        for change in ({"token_limit_reached": True}, {"output_tokens": 513}, {"input_tokens": 0}):
            value = generated(); value["tokens"].update(change)
            with self.assertRaises(ValueError): self.reports(generate=lambda item: value)

    def test_four_states_project_by_name_without_inventing_negatives(self):
        for state in w.prior.guard.STATES:
            rows, stopped = self.labels(infer=lambda r: dict.fromkeys(w.CHEXPERT_FINDINGS, state))
            self.assertFalse(stopped)
            self.assertTrue(all(r["contract_status"] == "complete" and r["label_called"] for r in rows))
            self.assertTrue(all(set(r["states"]) == set(w.prior.guard.HEADS) and set(r["states"].values()) == {state} for r in rows))
            self.assertTrue(all(len(r["finding_states"]) == 14 for r in rows))

    def test_no_available_reports_means_no_labeler_load(self):
        rows = self.reports(uniform_first=True, generate=Mock(side_effect=RuntimeError("invented")))
        load = Mock(); cb = Mock(); output, _ = self.labels(rows, load=load, infer=cb)
        load.assert_not_called(); cb.assert_not_called()
        self.assertEqual(output[0]["contract_status"], "blocked_before_model_call")
        self.assertEqual(output[2]["contract_status"], "not_attempted_after_runtime_failure")

    def test_labeler_load_failure_stays_unavailable_with_zero_forwards(self):
        load = Mock(side_effect=RuntimeError("invented")); cb = Mock(); events = []
        rows, stopped = self.labels(load=load, infer=cb, journal=events.append)
        self.assertTrue(stopped); self.assertEqual(load.call_count, 1); cb.assert_not_called()
        self.assertEqual(events, [{"event": "label_load_reserved"}])
        self.assertTrue(all(r["contract_status"] == "failed_unavailable" and r["states"] is None and not r["label_called"] for r in rows))

    def test_label_token_limit_skips_without_forward_or_negative(self):
        cb = Mock(); rows, stopped = self.labels(length=lambda r: (513, 512), infer=cb)
        cb.assert_not_called(); self.assertFalse(stopped)
        self.assertTrue(all(r["label_status"] == "overlength_unavailable_no_truncation" and r["states"] is None for r in rows))

    def test_exact_label_token_bound_is_allowed(self):
        rows, _ = self.labels(length=lambda r: (512, 512))
        self.assertTrue(all(r["contract_status"] == "complete" for r in rows))

    def test_label_forward_failure_stops_and_preserves_all_report_charges(self):
        cb = Mock(side_effect=RuntimeError("invented")); events = []
        rows, stopped = self.labels(infer=cb, journal=events.append)
        self.assertTrue(stopped); self.assertEqual(cb.call_count, 1)
        self.assertEqual(sum(r["label_called"] for r in rows), 1)
        self.assertTrue(all(r["model_called"] and r["states"] is None for r in rows))
        self.assertEqual([r["label_status"] for r in rows], ["runtime_failed_unavailable"] + ["not_attempted_after_label_runtime_failure"]*2)

    def test_label_reservation_precedes_inference(self):
        events = []
        def cb(report):
            self.assertEqual(events[-1]["event"], "label_request_reserved")
            return dict.fromkeys(w.CHEXPERT_FINDINGS, "unknown")
        self.labels(infer=cb, journal=events.append)
        self.assertEqual(len(events), 7)

    def test_invalid_head_vector_and_bad_state_are_rejected(self):
        for result in ({"pneumonia": "negative"}, dict.fromkeys(w.CHEXPERT_FINDINGS, "not_a_state"), None):
            with self.assertRaises((ValueError, TypeError)): self.labels(infer=lambda r: result)

    def test_label_scope_duplicate_rejected_before_load(self):
        load = Mock(); rows = report_rows()
        with self.assertRaises(ValueError): w.label_records([rows[0], rows[0]], load=load, length=Mock(), infer=Mock(), journal=Mock())
        load.assert_not_called()

    def test_normalization_matches_existing_labeler_literal_operations(self):
        text = "  Findings:\nfixture  double space\\s+ and\\s+(?=[\\.,]) end  "
        expected = text.strip().replace("\n", " ").replace("\\s+", " ").replace("\\s+(?=[\\.,])", "").strip()
        self.assertEqual(w.normalized_chexbert_input(text), expected)
        self.assertIn("double space", expected)

    def test_tracking_forwards_original_arguments_and_restores_method(self):
        original = Mock(return_value=[SimpleNamespace(shape=(35,))])
        runtime = SimpleNamespace(_model=SimpleNamespace(generate=original))
        ids = SimpleNamespace(shape=(1, 5))
        def generate(path):
            runtime._model.generate(ids, do_sample=False, max_new_tokens=512)
            return SimpleNamespace(canonical_text="invented report", metadata={"fixture_only": True})
        runtime.generate = generate
        result = w.tracked_official_generate(runtime, Path("invented.png"))
        self.assertIs(runtime._model.generate, original)
        original.assert_called_once_with(ids, do_sample=False, max_new_tokens=512)
        self.assertEqual(result["tokens"], {"input_tokens": 5, "output_tokens": 30, "token_limit_reached": False})

    def test_tracking_restores_model_method_after_runtime_exception(self):
        original = Mock(); runtime = SimpleNamespace(_model=SimpleNamespace(generate=original),
            generate=Mock(side_effect=RuntimeError("invented")))
        with self.assertRaises(RuntimeError): w.tracked_official_generate(runtime, Path("invented.png"))
        self.assertIs(runtime._model.generate, original)

    def test_preparation_guard_precedes_all_plan_reads(self):
        with patch.object(w.prior.observer.source.previous.gate, "cpu_guard", side_effect=RuntimeError("blocked")), \
                patch.object(w.prior, "manifest") as read:
            with self.assertRaises(RuntimeError): w.prepare(Mock())
            read.assert_not_called()

    def test_execution_guards_precede_reads_and_framework_imports(self):
        for entry in (w.run, w.phase):
            with patch.object(w, "require_gpu_slurm", side_effect=RuntimeError("blocked")), patch.object(w, "load_plan") as read:
                with self.assertRaises(RuntimeError): entry(Mock())
                read.assert_not_called()

    def test_label_inputs_and_report_records_are_not_mutated(self):
        reports = report_rows(); before = deepcopy(reports)
        self.labels(reports)
        self.assertEqual(reports, before)


if __name__ == "__main__": unittest.main()
