"""Invented references/pixels/hashes only, mocked model and source inspection."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_validation"))
import rsua_qwen_observer as w
from test_guarded_image_findings import Image, guarded


def record(i, state="unknown", status="complete"):
    return {"case_id": f"invented_{i}", "model_called": status not in ("blocked_before_model_call", "not_attempted_after_runtime_failure"),
        "contract_status": status, "states": dict.fromkeys(w.guard.HEADS, state) if status == "complete" else None}


def callback_result(state="unknown", status="complete"):
    return {**{k: v for k, v in record(0, state, status).items() if k in ("contract_status", "states")},
        "failure_reason": None if status == "complete" else "invalid_eight_state_json",
        "response_sha256": "b"*64, "input_tokens": 600, "output_tokens": 85, "token_limit_reached": False}


def inputs_and_inspector(*, uniform_first=False):
    images = [Image(uniform_first, 0), Image(False, 2), Image(False, 4)]
    inputs = [{"case_id": f"invented_{i}", "png_sha256": format(i+1, "064x"),
        "original_bmp_sha256": format(i+4, "064x")} for i in range(3)]
    def inspect(item):
        image = images[inputs.index(item)]
        return guarded(image, item["png_sha256"]), image
    return inputs, inspect, images


class RsuaQwenTests(unittest.TestCase):
    def test_four_states_have_distinct_full_denominator_counts(self):
        rows = [record(i, state) for i, state in enumerate(("positive", "negative", "uncertain", "unknown"))]
        refs = {r["case_id"]: "positive" for r in rows}
        result = w.summarize(rows, refs); p = result["reference_groups"]["positive"]
        self.assertEqual(p["reference_slots"], 4)
        self.assertEqual([p[k] for k in ("positive", "negative", "uncertain", "unknown")], [1, 1, 1, 1])
        self.assertEqual(p["correct_explicit_support_over_all"], .25)
        self.assertEqual(p["conditional_support_over_explicit"], .5)
        self.assertEqual(p["missingness_correctness_upper"], .75)

    def test_no_explicit_answer_is_unavailable_conditionally_not_perfect(self):
        rows = [record(0, "unknown"), record(1, "uncertain")]
        result = w.summarize(rows, {"invented_0": "positive", "invented_1": "negative"})
        self.assertEqual(result["explicit_coverage"], 0)
        self.assertEqual(result["correct_explicit_support_over_all"], 0)
        self.assertIsNone(result["conditional_support_over_explicit"])
        self.assertIsNone(result["clinical_accuracy"])
        self.assertIsNone(result["auroc"])
        self.assertFalse(result["primary_metric_eligible"])

    def test_failure_blocked_unattempted_are_not_unknown_or_negative(self):
        rows = [record(0, status="failed_unavailable"), record(1, status="blocked_before_model_call"),
            record(2, status="not_attempted_after_runtime_failure")]
        result = w.summarize(rows, {r["case_id"]: "negative" for r in rows})["reference_groups"]["negative"]
        self.assertEqual(result["unknown"], 0)
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["correct_explicit_slots"], 0)
        self.assertEqual(result["unresolved_slots"], 3)
        self.assertEqual(result["missingness_correctness_upper"], 1)

    def test_opposite_explicit_states_remain_errors_not_missingness(self):
        result = w.summarize([record(0, "negative"), record(1, "positive")],
            {"invented_0": "positive", "invented_1": "negative"})
        self.assertEqual(result["confusion_explicit_only"], {"tp": 0, "fn": 1, "tn": 0, "fp": 1})
        self.assertEqual(result["balanced_correct_explicit_support_over_all"], 0)
        self.assertEqual(result["reference_groups"]["negative"]["missingness_correctness_upper"], 0)

    def test_one_reference_group_has_no_balanced_support_invented(self):
        result = w.summarize([record(0, "positive")], {"invented_0": "positive"})
        self.assertIsNone(result["balanced_correct_explicit_support_over_all"])
        self.assertIsNone(result["reference_groups"]["negative"]["explicit_coverage"])

    def test_missing_duplicate_unknown_reference_invalid_vector_rejected(self):
        for kind in ("missing_ref", "duplicate", "unknown_ref", "partial", "false_call"):
            row = record(0); refs = {"invented_0": "positive"}; rows = [row]
            if kind == "missing_ref": refs["extra"] = "positive"
            elif kind == "duplicate": rows.append(deepcopy(row))
            elif kind == "unknown_ref": refs["invented_0"] = "unknown"
            elif kind == "partial": row["states"].pop("pneumonia")
            else: row["model_called"] = False
            with self.assertRaises(ValueError): w.summarize(rows, refs)

    def test_summary_does_not_mutate_inputs_or_create_other_finding_references(self):
        rows = [record(0, "positive"), record(1, "negative")]; refs = {"invented_0": "positive", "invented_1": "negative"}
        before = deepcopy((rows, refs)); result = w.summarize(rows, refs)
        self.assertEqual((rows, refs), before)
        self.assertTrue(result["other_seven_findings_reference_unavailable"])
        self.assertFalse(result["regeneration_authorized"])

    def test_source_order_hash_lineage_but_no_score_or_label_in_plan(self):
        rows = [{"case_id": f"case_{i:04d}", "png_sha256": format(i+1, "064x"),
            "source_image_sha256": format(i+100, "064x"), "score_pairs": {"ignored": .5}} for i in range(50)]
        before = deepcopy(rows); inputs = w.public_inputs(rows, stat_path=lambda p: [100, 1000])
        self.assertEqual(len(inputs), 50)
        self.assertEqual(rows, before)
        self.assertTrue(all(set(r) == {"case_id", "path", "png_sha256", "original_bmp_sha256", "file_stats"} for r in inputs))
        rows.reverse()
        with self.assertRaises(ValueError): w.public_inputs(rows, stat_path=lambda p: [100, 1000])

    def test_duplicate_hashes_cannot_replace_one_source_image(self):
        rows = [{"case_id": f"case_{i:04d}", "png_sha256": "a"*64,
            "source_image_sha256": format(i+100, "064x")} for i in range(50)]
        with self.assertRaises(ValueError): w.public_inputs(rows, stat_path=lambda p: [100, 1000])

    def test_uniform_guard_blocks_before_any_model_call(self):
        inputs, inspect, images = inputs_and_inspector(uniform_first=True)
        callback = Mock(return_value=callback_result()); events = []
        rows, calls, failed = w.invoke_inputs(inputs, callback, journal=events.append, inspect=inspect)
        self.assertEqual(calls, 2); self.assertFalse(failed)
        self.assertEqual(rows[0]["contract_status"], "blocked_before_model_call")
        self.assertIsNone(rows[0]["states"])
        self.assertEqual(callback.call_count, 2)
        self.assertIs(callback.call_args_list[0].args[0], images[1])

    def test_reservation_is_durable_before_callback(self):
        inputs, inspect, _ = inputs_and_inspector(); events = []
        def callback(image):
            self.assertEqual(events[-1]["event"], "call_reserved")
            return callback_result()
        rows, charged, failed = w.invoke_inputs(inputs, callback, journal=events.append, inspect=inspect)
        self.assertEqual(charged, 3); self.assertFalse(failed)
        self.assertEqual([e["event"] for e in events], ["call_reserved", "call_finished"] * 3)

    def test_journal_failure_prevents_call_and_does_not_resume(self):
        inputs, inspect, _ = inputs_and_inspector(); callback = Mock()
        with self.assertRaises(OSError):
            w.invoke_inputs(inputs, callback, journal=Mock(side_effect=OSError()), inspect=inspect)
        callback.assert_not_called()

    def test_runtime_failure_charged_stops_and_keeps_all_slots(self):
        inputs, inspect, _ = inputs_and_inspector(); events = []
        callback = Mock(side_effect=RuntimeError("invented failure"))
        rows, charged, failed = w.invoke_inputs(inputs, callback, journal=events.append, inspect=inspect)
        self.assertEqual(charged, 1); self.assertTrue(failed); self.assertEqual(len(rows), 3)
        self.assertEqual([r["contract_status"] for r in rows], ["failed_unavailable"] + ["not_attempted_after_runtime_failure"] * 2)
        self.assertEqual(callback.call_count, 1)
        self.assertTrue(events[-1]["runtime_failure_stop"])

    def test_invalid_json_response_is_unavailable_not_retry(self):
        inputs, inspect, _ = inputs_and_inspector(); cb = Mock(return_value=callback_result(status="failed_unavailable"))
        rows, calls, failed = w.invoke_inputs(inputs, cb, journal=lambda e: None, inspect=inspect)
        self.assertEqual(calls, 3); self.assertFalse(failed); self.assertEqual(cb.call_count, 3)
        self.assertTrue(all(r["states"] is None for r in rows))

    def test_duplicate_or_insufficient_budget_refused_before_callback(self):
        inputs, inspect, _ = inputs_and_inspector(); cb = Mock()
        for values, budget in ((inputs, 2), ([inputs[0], inputs[0]], 3)):
            with self.assertRaises(ValueError): w.invoke_inputs(values, cb, journal=lambda e: None, inspect=inspect, maximum_calls=budget)
        cb.assert_not_called()

    def test_changed_guard_pixel_identity_refused_before_model(self):
        inputs, inspect, _ = inputs_and_inspector(); cb = Mock()
        def wrong(item):
            current, _ = inspect(item); return current, Image(False, 10)
        with self.assertRaises(ValueError): w.invoke_inputs(inputs, cb, journal=lambda e: None, inspect=wrong)
        cb.assert_not_called()

    def test_unsanitized_callback_extra_prose_or_token_cap_rejected(self):
        inputs, inspect, _ = inputs_and_inspector()
        for kind in ("prose", "tokens", "hash"):
            raw = callback_result()
            if kind == "prose": raw["free_text"] = "invented unacceptable content"
            elif kind == "tokens": raw["token_limit_reached"] = True
            else: raw["response_sha256"] = "invalid"
            with self.assertRaises(ValueError): w.invoke_inputs(inputs, Mock(return_value=raw), journal=lambda e: None, inspect=inspect)

    def test_xrv_same_image_profiles_unchanged_and_not_consensus_truth(self):
        inputs, _, _ = inputs_and_inspector()
        rows = [record(0, "positive"), record(1, "negative"), record(2, "unknown")]
        refs = {"invented_0": "positive", "invented_1": "negative", "invented_2": "positive"}
        sources = [{"case_id": i["case_id"], "image_sha256": i["original_bmp_sha256"],
            "reference_state": refs[i["case_id"]], "pneumonia_score": score} for i, score in zip(inputs, (.8, .2, .52))]
        before = deepcopy((rows, refs, sources, inputs)); result = w.compare_xrv(rows, refs, sources, inputs)
        self.assertEqual((rows, refs, sources, inputs), before)
        self.assertEqual(result["xrv_default_0_5"]["qwen_unresolved"], 1)
        self.assertEqual(result["xrv_default_0_5"]["xrv_proxy_readout"]["confusion_explicit_only"]["tp"], 2)
        self.assertEqual(result["xrv_unchanged_weak_reference_transport"]["xrv_proxy_readout"]["confusion_explicit_only"]["tp"], 1)
        self.assertFalse(result["xrv_default_0_5"]["same_state_is_clinical_truth"])
        sources[0]["image_sha256"] = "d"*64
        with self.assertRaises(ValueError): w.compare_xrv(rows, refs, sources, inputs)

    def test_cpu_and_gpu_guards_precede_any_plan_or_patient_source_read(self):
        with patch.object(w.observer.source.previous.gate, "cpu_guard", side_effect=RuntimeError("blocked")), \
                patch.object(w, "manifest") as read:
            with self.assertRaises(RuntimeError): w.prepare(Mock())
            read.assert_not_called()
        with patch.object(w.observer, "gpu_guard", side_effect=RuntimeError("blocked")), patch.object(w, "manifest") as read:
            with self.assertRaises(RuntimeError): w.run(Mock())
            read.assert_not_called()


if __name__ == "__main__": unittest.main()
