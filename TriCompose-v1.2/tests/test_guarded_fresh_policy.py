"""Invented same-EHR image branches, numeric receipts and untried expert menu."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"agent"))
import run_guarded_fresh_policy as worker
from test_fresh_output_acceptance import fixture
from tricompose_llm.contracts import DECISION_SCHEMA
from tricompose_llm.guarded_action_dispatch import GuardedActionDispatcher


def ingredients():
    rows = [fixture(image_state="negative", report_state="unknown", model=model,
        report_id=f"invented_cached_{index}")[0] for index, model in enumerate(worker.MODELS)]
    probe, context = fixture(image_state="negative", report_state="unknown", seed=1,
        image_id="invented_trial_image", report_id="invented_trial_report", model=worker.MODELS[0])
    old = {"case_id": rows[0]["case_id"], "anchor": context["anchor"], "reference": rows[0]}
    states = dict.fromkeys(worker.previous.cached.observer.existing.image_interface.FINDINGS, "unknown")
    states["edema"] = "positive"
    observation = {"cxr_candidate_id": probe["cxr_candidate_id"], "cxr_sha256": probe["cxr_sha256"],
        "contract_status": "complete", "states": states}
    specs = {"xrv": {"thresholds": {name: {"enabled": name in context["enabled_xrv_findings"]}
        for name in worker.previous.cached.observer.source.previous.gate.FINDINGS},
        "thresholds_sha256": context["thresholds_sha256"], "checkpoint_sha256": context["xrv_checkpoint_sha256"]},
        "chexbert": {"checkpoint_sha256": context["chexbert_checkpoint_sha256"]}}
    return old, probe, rows, observation, specs


def case_fixture():
    return worker.make_case(*ingredients())


def decision(packet, action="regenerate_report"):
    state = packet["observation"]
    return {"schema_version": DECISION_SCHEMA, "step_id": state["step_id"], "action": action,
        "target_id": state["tools"][0]["tool_id"] if action == "regenerate_report" else None,
        "evidence_ids": ["e0004"], "reason_code": "report_mismatch" if action == "regenerate_report" else "insufficient_evidence"}


class FreshGuardedPolicyTests(unittest.TestCase):
    def test_case_keeps_ehr_and_old_selection_but_uses_distinct_trial_image(self):
        inputs = ingredients(); before = deepcopy(inputs)
        result = worker.make_case(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(result["ehr_sha256"], inputs[0]["reference"]["ehr_sha256"])
        self.assertEqual(result["retained_original_candidate_id"], inputs[0]["reference"]["triple_candidate_id"])
        self.assertNotEqual(result["trial_candidate_id"], result["retained_original_candidate_id"])
        self.assertFalse(result["original_selection_change_allowed"])

    def test_three_real_request_models_exclude_already_observed_current_expert(self):
        result = case_fixture(); packet = result["packet"]
        tools = packet["observation"]["tools"]
        self.assertEqual(len(tools), 3)
        self.assertTrue(all(t["action"] == "regenerate_report" and t["cost_units"] == 2 for t in tools))
        self.assertTrue(all(t["model_id"] != packet["current_report_model_id"] for t in tools))
        self.assertTrue(all(not item["backend_installed_in_this_job"] for item in result["tool_catalog"].values()))

    def test_history_is_other_image_not_future_report_label(self):
        packet = case_fixture()["packet"]
        self.assertEqual(len(packet["expert_history"]), 4)
        self.assertEqual(packet["observation"]["current_candidate_id"], "c0004")
        self.assertEqual(packet["candidate_image_groups"][-1]["image_group_id"], "i0001")
        self.assertTrue(all(g["image_group_id"] == "i0000" for g in packet["candidate_image_groups"][:4]))

    def test_report_request_is_deferred_never_fake_generation(self):
        packet = case_fixture()["packet"]; events = []
        result, value = GuardedActionDispatcher(sink=events.append).dispatch(
            decision(packet), packet["observation"], packet["image_guard"], backend_factory=None)
        self.assertIsNone(value)
        self.assertEqual(result["status"], "deferred_separately_approved_backend_required")
        self.assertEqual(result["actual_worker_model_calls"], 0)
        self.assertFalse(result["clinical_acceptance"])

    def test_unresolved_stop_is_abstain_not_clinical_acceptance(self):
        packet = case_fixture()["packet"]
        result, _ = GuardedActionDispatcher(sink=lambda event: None).dispatch(
            decision(packet, "stop"), packet["observation"], packet["image_guard"])
        self.assertEqual(result["effective_decision"]["action"], "abstain")
        self.assertFalse(result["clinical_acceptance"])

    def test_changed_ehr_original_image_or_seed_rejected(self):
        for kind in ("ehr", "image", "seed", "model"):
            inputs = list(ingredients())
            if kind == "ehr": inputs[1]["ehr_sha256"] = "8"*64
            elif kind == "image": inputs[1]["cxr_sha256"] = inputs[0]["reference"]["cxr_sha256"]
            elif kind == "seed": inputs[1]["seed"] = 0
            else: inputs[1]["report_model_id"] = worker.MODELS[1]
            with self.subTest(kind=kind), self.assertRaises(ValueError): worker.make_case(*inputs)

    def test_observer_binding_and_unavailable_semantics_preserved(self):
        inputs = list(ingredients()); inputs[3]["cxr_sha256"] = "8"*64
        with self.assertRaises(ValueError): worker.make_case(*inputs)
        inputs = list(ingredients()); inputs[3].update(contract_status="failed_unavailable", states=None)
        result = worker.make_case(*inputs)
        self.assertEqual(result["packet"]["image_guard"]["counts"]["observer_unavailable_slots"], 8)
        self.assertEqual(len(result["packet"]["observation"]["tools"]), 3)

    def test_gpu_and_cpu_guards_precede_readers_and_model_construction(self):
        with patch.object(worker, "gpu_guard", side_effect=RuntimeError("invented_cpu")), \
                patch.object(worker, "GuardAwareQwenPlanner") as planner, patch.object(worker, "require_inside") as reader:
            with self.assertRaises(RuntimeError): worker.run(None)
            reader.assert_not_called(); planner.assert_not_called()
        with patch.object(worker.previous.cached.observer.source.previous.gate, "cpu_guard",
                side_effect=RuntimeError("invented_login")), \
                patch.object(worker.previous.cached.observer.source.previous.postflight, "MetadataReader") as reader:
            with self.assertRaises(RuntimeError): worker.prepare(None)
            reader.assert_not_called()


if __name__ == "__main__": unittest.main()
