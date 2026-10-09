"""Invented four-state fixtures; no models, clinical artifacts or network."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from tricompose_llm.contracts import ContractError, DECISION_SCHEMA, PUBLIC_SCHEMA, validate_public_state
from tricompose_llm.image_evidence_guard import (
    build_guard, guarded_decision, guarded_state, validate_guard,
)

STATES = ("positive", "negative", "uncertain", "unknown")


def fixture(ehr="positive", xrv="negative", observer="positive"):
    def edge(ref, target):
        known = int(ref in STATES[:2])
        comparable = int(known and target in STATES[:2])
        support = int(comparable and ref == target)
        return {"known": known, "comparable": comparable, "supported": support,
                "opposed": comparable - support,
                "positive_supported": int(support and ref == "positive")}
    state = {"schema_version": PUBLIC_SCHEMA,
        "step_id": "s0000", "current_candidate_id": "c0000", "evidence": [{
            "evidence_id": "e0000", "candidate_id": "c0000",
            "edges": {"ehr_cxr": edge(ehr, xrv), "ehr_report": edge(ehr, "unknown"),
                      "cxr_report": edge(xrv, "unknown")},
            "quality": {"image_basic_valid": True, "report_structure": 1.0, "artifact_failures": 0},
            "uncertainty": {"ehr_unknown": int(ehr == "unknown"), "ehr_uncertain": int(ehr == "uncertain"),
                "image_unknown": int(xrv == "unknown"), "image_uncertain": int(xrv == "uncertain"),
                "report_unknown": 1, "report_uncertain": 0}}],
        "tools": [{"tool_id": "t0000", "action": "regenerate_cxr", "model_id": "m0000", "seed": 1, "cost_units": 4},
                  {"tool_id": "t0001", "action": "regenerate_report", "model_id": "m0001", "seed": 0, "cost_units": 2}],
        "budget": {"limit_units": 9, "spent_units": 1, "planner_units_per_call": 1}, "history": []}
    status = "failed_unavailable" if observer is None else "complete"
    guard = build_guard({"edema": ehr}, {"edema": xrv}, None if observer is None else {"edema": observer},
                        candidate_id="c0000", evidence_id="e0000", observer_status=status)
    return validate_public_state(state), guard


def proposal(state, action="regenerate_cxr"):
    tool = next((t for t in state["tools"] if t["action"] == action), None)
    return {"schema_version": DECISION_SCHEMA, "step_id": state["step_id"], "action": action,
        "target_id": tool["tool_id"] if tool else None, "evidence_ids": ["e0000"],
        "reason_code": "explore_alternative"}


class ImageEvidenceGuardTests(unittest.TestCase):
    def test_all_four_state_combinations_preserve_unknown_and_uncertain(self):
        for ehr, xrv, observer in itertools.product(STATES, repeat=3):
            with self.subTest(ehr=ehr, xrv=xrv, observer=observer):
                _, guard = fixture(ehr, xrv, observer)
                c = guard["counts"]
                known = int(ehr in STATES[:2])
                joint = int(known and xrv in STATES[:2] and observer in STATES[:2])
                self.assertEqual(c["known_ehr"], known)
                self.assertEqual(c["joint_known_comparable"], joint)
                self.assertEqual(c["known_scorer_disagreement"], int(joint and xrv != observer))
                self.assertEqual(c["known_missing_comparison"], known - joint)
                self.assertEqual(guard["withhold_image_attribution"], not joint or xrv != observer)
                self.assertFalse(guard["clinical_acceptance"])

    def test_disagreement_removes_only_image_tools_without_mutating_original(self):
        state, guard = fixture(); original = deepcopy(state)
        view = guarded_state(state, guard)
        self.assertEqual([t["action"] for t in view["tools"]], ["regenerate_report"])
        self.assertEqual({k: v for k, v in view.items() if k != "tools"},
                         {k: v for k, v in state.items() if k != "tools"})
        self.assertEqual(state, original)

    def test_conflicting_request_and_stop_become_explicit_abstain(self):
        state, guard = fixture()
        for action in ("regenerate_cxr", "stop"):
            decision = proposal(state, action); original = deepcopy(decision)
            result = guarded_decision(decision, state, guard)
            self.assertTrue(result["decision_withheld"])
            self.assertEqual(result["effective_decision"]["action"], "abstain")
            self.assertEqual(result["effective_decision"]["reason_code"], "insufficient_evidence")
            self.assertEqual(result["raw_decision"], original)
            self.assertEqual(decision, original)

    def test_report_retry_and_abstain_preserved(self):
        state, guard = fixture()
        for action in ("regenerate_report", "abstain"):
            decision = proposal(state, action)
            result = guarded_decision(decision, state, guard)
            self.assertFalse(result["decision_withheld"])
            self.assertEqual(result["effective_decision"], decision)

    def test_agreement_does_not_authorize_acceptance_or_change_old_gate(self):
        for state_name in STATES[:2]:
            state, guard = fixture(observer=state_name, xrv=state_name)
            self.assertEqual(guard["status"], "scorers_agree_unvalidated")
            self.assertEqual(guarded_state(state, guard), state)
            result = guarded_decision(proposal(state), state, guard)
            self.assertFalse(result["decision_withheld"])
            self.assertFalse(result["clinical_acceptance"])
            self.assertIsNone(result["clinical_fault_location"])
            self.assertTrue(result["original_acceptance_gate_must_still_run"])

    def test_failed_observer_is_unavailable_not_negative_or_successful_abstention(self):
        state, guard = fixture(observer=None)
        self.assertEqual(guard["counts"]["observer_unavailable_slots"], 1)
        self.assertEqual(guard["counts"]["observer_ehr_opposition"], 0)
        self.assertEqual(guard["counts"]["known_missing_comparison"], 1)
        self.assertTrue(guarded_decision(proposal(state), state, guard)["decision_withheld"])
        with self.assertRaises(ContractError):
            build_guard({"edema": "positive"}, {"edema": "negative"}, {"edema": "unknown"},
                        candidate_id="c0000", evidence_id="e0000", observer_status="failed_unavailable")

    def test_unknown_ehr_is_not_a_normal_patient_constraint(self):
        state, guard = fixture(ehr="unknown", xrv="negative", observer="negative")
        self.assertEqual(guard["counts"]["known_ehr"], 0)
        self.assertEqual(guard["counts"]["joint_ehr_support"], 0)
        self.assertEqual(guard["status"], "unresolved_no_known_ehr_constraints")
        self.assertTrue(guarded_decision(proposal(state, "stop"), state, guard)["decision_withheld"])

    def test_noncomparable_ehr_findings_do_not_create_known_disagreement(self):
        guard = build_guard({"edema": "positive", "pneumonia": "unknown"},
                            {"edema": "positive", "pneumonia": "negative"},
                            {"edema": "positive", "pneumonia": "positive"},
                            candidate_id="c0000", evidence_id="e0000")
        self.assertEqual(guard["counts"]["outside_known_ehr_scorer_disagreement"], 1)
        self.assertEqual(guard["counts"]["known_scorer_disagreement"], 0)
        self.assertFalse(guard["withhold_image_attribution"])

    def test_dictionary_order_has_no_effect(self):
        a = {"edema": "positive", "pneumonia": "unknown"}
        b = {"edema": "negative", "pneumonia": "uncertain"}
        c = {"edema": "positive", "pneumonia": "unknown"}
        original = build_guard(a, b, c, candidate_id="c0000", evidence_id="e0000")
        self.assertEqual(original, build_guard(dict(reversed(list(a.items()))), b, c,
                                              candidate_id="c0000", evidence_id="e0000"))

    def test_wrong_candidate_evidence_scope_and_primary_counts_rejected(self):
        state, guard = fixture()
        for key, value in (("current_candidate_id", "c0001"), ("current_evidence_id", "e0001")):
            bad = deepcopy(guard); bad[key] = value
            with self.assertRaises(ContractError): guarded_state(state, bad)
        state["evidence"][0]["edges"]["ehr_cxr"].update(supported=1, opposed=0, positive_supported=1)
        with self.assertRaises(ContractError): guarded_state(state, guard)

    def test_closed_projection_has_no_clinical_text_paths_hashes_or_finding_names(self):
        _, guard = fixture()
        serialized = json.dumps(guard)
        for value in ("edema", "pneumonia", "sha256", "/project2", "report_text"):
            self.assertNotIn(value, serialized)
        for field in ("path", "ehr_text", "patient_id", "report_text", "score"):
            bad = deepcopy(guard); bad[field] = "invented_forbidden"
            with self.assertRaises(ContractError): validate_guard(bad)

    def test_algebra_bool_nan_and_unavailable_claim_tampering_rejected(self):
        _, guard = fixture()
        for field, value in (("known_ehr", True), ("known_ehr", float("nan")),
                             ("known_scorer_disagreement", 0), ("observer_unavailable_slots", 1)):
            bad = deepcopy(guard); bad["counts"][field] = value
            with self.assertRaises(ContractError): validate_guard(bad)
        for field, value in (("clinical_acceptance", True), ("clinical_fault_location", "cxr"),
                             ("majority_vote_used", True), ("withhold_image_attribution", 1)):
            bad = deepcopy(guard); bad[field] = value
            with self.assertRaises(ContractError): validate_guard(bad)

    def test_invalid_raw_decision_is_not_repaired_into_valid_output(self):
        state, guard = fixture()
        bad = proposal(state); bad["raw_text"] = "invented_invalid"
        with self.assertRaises(ContractError): guarded_decision(bad, state, guard)
        bad = proposal(state); bad["target_id"] = "t9999"
        with self.assertRaises(ContractError): guarded_decision(bad, state, guard)

    def test_partial_foreign_and_invalid_finding_vectors_rejected(self):
        for bad in ({}, {"unknown_name": "positive"}, {"edema": "invalid"}, {"edema": True}):
            with self.assertRaises(ContractError):
                build_guard(bad, {"edema": "negative"}, {"edema": "positive"},
                            candidate_id="c0000", evidence_id="e0000")
        with self.assertRaises(ContractError):
            build_guard({"edema": "positive"}, {"edema": "negative"}, {},
                        candidate_id="c0000", evidence_id="e0000")


if __name__ == "__main__":
    unittest.main()
