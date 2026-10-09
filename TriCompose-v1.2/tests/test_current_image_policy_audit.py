"""Authored numeric replay only; never instantiate a real Qwen model."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import audit_current_image_policy as audit_worker
from test_current_image_policy import case_fixture, decision


def ingredients(*, failed=False):
    worker = audit_worker.worker
    cases = [case_fixture(), case_fixture()]
    cases[1]["case_id"] = "case_999"
    plan = {"cases": cases, "qwen": {"asset_pins": {"/authored/model.safetensors": {"sha256": "0" * 64}}}}
    journal = [{"event": "planner_load_reserved", "charged_load_attempts": 1}]
    results = []; deferred = []
    for i, case in enumerate(cases):
        journal.append({"event": "current_policy_reserved", "ordinal": i,
            "policy_input_sha256": case["policy_input_sha256"], "charged_policy_units": 1})
        packet = case["packet"]
        if failed and i == 1:
            result = {"case_id": case["case_id"], "status": "planner_failed_charged_no_retry",
                "decision": None, "dispatch": None, "rule_control": case["rule_control"],
                "clinical_acceptance": False, "original_winner_changed": False}
            journal.append({"event": "current_policy_failed_charged", "ordinal": i})
        else:
            d = decision(packet)
            journal.append({"event": "current_policy_validated", "ordinal": i, "decision": d})
            dispatch = worker.dispatch_record(d, packet,
                sink=lambda event: journal.append({"case_id": case["case_id"], **event}))
            result = {"case_id": case["case_id"], "status": "current_llm_decision_validated_unverified",
                "source": "new_frozen_qwen_generate_call", "decision": d, "dispatch": dispatch,
                "policy_input_sha256": case["policy_input_sha256"], "rule_control": case["rule_control"],
                "same_effective_action_and_target_as_rule": True, "clinical_acceptance": False,
                "original_winner_changed": False, "response_metadata": {"input_tokens": 3,
                    "output_tokens": 10, "token_limit_reached": False, "response_sha256": "1" * 64}}
        results.append(result)
        for policy, dispatch in (("qwen", result["dispatch"]), ("rule", case["rule_control"]["dispatch"])):
            if dispatch is not None:
                deferred.append({"case_id": case["case_id"], "policy": policy,
                    "request": case["tool_catalog"][dispatch["effective_decision"]["target_id"]],
                    "clinical_acceptance": False})
    pending = {"schema_version": worker.VERSION, "records": deferred,
        "automatic_submission_allowed": False, "requires_complete_script_and_explicit_approval": True}
    audit = {"frozen": True, "interface_version": worker.PACKET_VERSION, "local_attempts": 2,
        "generate_attempts": 2, "historical_other_image_experts_supplied": False, "endpoint_scores_supplied": False,
        "failure_counts": {"state_contract": 0, "tokenize": 0, "context_bound": 0, "generate": 0,
            "decode": 0, "json_contract": int(failed), "decision_contract": 0},
        "asset_pins": {"model.safetensors": "0" * 64}}
    summary = {"charged_new_policy_requests": 2, "fresh_generate_attempts": 2,
        "validated_fresh_decisions": 2 - int(failed), "same_effective_action_and_target_as_rule": 2 - int(failed),
        "actual_rule_model_calls": 0, "new_worker_model_calls": 0, "llm_superiority_demonstrated": False,
        "clinical_acceptance": False, "measured_saved_model_calls": None, "original_winner_changed": False}
    return plan, results, pending, audit, journal, summary


class CurrentImageAuditTests(unittest.TestCase):
    def test_exact_replay_and_zero_rule_calls_not_saved_generator_calls(self):
        data = ingredients(); original = deepcopy(data)
        result = audit_worker.replay(*data)
        self.assertTrue(result["exact_durable_dispatch_replay_pass"])
        self.assertEqual(result["charged_new_policy_requests"], 2)
        self.assertEqual(result["same_effective_action_and_target_as_rule"], 2)
        self.assertEqual(data, original)

    def test_invalid_llm_output_remains_charged_no_rule_fallback(self):
        result = audit_worker.replay(*ingredients(failed=True))
        self.assertEqual(result["charged_new_policy_requests"], 2)
        self.assertEqual(result["validated_decisions"], 1)

    def test_pre_generate_failure_does_not_become_generate_attempt(self):
        data = list(ingredients(failed=True))
        data[3]["generate_attempts"] = data[5]["fresh_generate_attempts"] = 1
        data[3]["failure_counts"]["json_contract"] = 0
        data[3]["failure_counts"]["context_bound"] = 1
        self.assertEqual(audit_worker.replay(*data)["charged_new_policy_requests"], 2)

    def test_tampered_journal_rule_cost_pending_and_success_claim_rejected(self):
        for kind in ("journal", "rule", "pending", "claim", "cost", "asset", "response"):
            data = list(ingredients())
            if kind == "journal": data[4].pop(1)
            elif kind == "rule": data[1][0]["rule_control"]["actual_model_calls"] = 1
            elif kind == "pending": data[2]["automatic_submission_allowed"] = True
            elif kind == "claim": data[5]["llm_superiority_demonstrated"] = True
            elif kind == "cost": data[5]["charged_new_policy_requests"] = 1
            elif kind == "asset": data[3]["asset_pins"]["model.safetensors"] = "2" * 64
            else: data[1][0]["response_metadata"]["token_limit_reached"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError): audit_worker.replay(*data)


if __name__ == "__main__": unittest.main()
