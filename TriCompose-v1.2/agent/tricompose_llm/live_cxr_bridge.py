"""One prospective CXR action, not clinical fault localization.

Only authenticated past numeric observations enter planning. A new image needs
its own XRV, report, and CheXbert chain. The existing cross-image preservation
gate is reused unchanged; old reports cannot be attached to a new image.
"""
from __future__ import annotations

import copy

import fresh_output_acceptance as gate
from tricompose_v12 import bounded_regeneration as image_gate
from tricompose_v12.invariant_verification import _digest, _HASH
from tricompose_v12.report_expert_control import structure_ok

from .contracts import (PUBLIC_SCHEMA, require, validate_decision,
                        validate_public_state)
from .live_report_bridge import FreshReportSession

VERSION = "tricompose-fresh-cxr-action-v1"


class FreshCXRSession:
    """A single bounded seed-one probe after the sealed report-only run.

    Cached work is shared historical cost, not free newly acquired evidence.
    Qwen can request the installed action, stop, or abstain. This deliberately
    does not present a one-action interface as general autonomous repair.
    """

    def __init__(self, observations, current_id, context, ledger, *,
                 source_manifest_sha256, sink):
        require(callable(sink) and isinstance(source_manifest_sha256, str)
                and _HASH.fullmatch(source_manifest_sha256), "authenticated_cache_and_sink_required")
        self.context = copy.deepcopy(context)
        self.anchor = gate.anchor_from_record(context["anchor"])
        gate.validate_ledger(ledger, self.anchor)
        require(ledger["charged_model_attempts"] == 0 and ledger["pending_attempts"] == 0
                and ledger["call_budget"] == 4 and ledger["max_retries"] == 0,
                "new_empty_four_call_no_retry_ledger_required")
        require(observations and len(observations) <= 4,
                "bounded_completed_historical_inventory_required")
        self.observed = [{"row": gate.controller_view(row), "origin": {
            "kind": "pinned_cached_reference", "source_manifest_sha256": source_manifest_sha256}}
            for row in observations]
        # Checks fixed EHR/scorer, shared-image labels, identities and arithmetic.
        initial_id = self.observed[0]["row"]["triple_candidate_id"]
        gate.assess_fresh_output(initial_id, initial_id, self.observed, self.context,
                                ledger_snapshot=ledger)
        rows = {item["row"]["triple_candidate_id"]: item["row"] for item in self.observed}
        require(current_id in rows and all(row["cxr_model_id"] == "roentgen_v2"
                and row["seed"] == 0 for row in rows.values()), "sealed_seed_zero_reference_required")
        require(len({row["cxr_candidate_id"] for row in rows.values()}) == 1,
                "one_fixed_historical_image_required")
        self.initial = self.observed[0]["row"]
        self.reference = rows[current_id]
        self.current = self.reference if structure_ok(self.reference) else None
        self.ledger = copy.deepcopy(ledger)
        self.sink = sink
        self.policy_requests = 0
        self.active_state = None
        self.active_tool = False
        self.history = []
        self.terminal = None
        self.frozen_rule = image_gate.route(self.reference)

    def begin_planning(self):
        require(self.active_state is None and not self.active_tool and self.terminal is None,
                "one_serial_prospective_policy_request_required")
        if self.current is None:
            self.terminal = "abstain_no_section_eligible_reference"
            return None
        if self.frozen_rule["action"] != "regenerate_cxr":
            self.terminal = self.frozen_rule["action"] + "_no_image_probe_basis_unverified"
            return None
        require(self.policy_requests == 0, "single_image_probe_only")
        self.sink({"event": "policy_reserved", "step_id": "s0000",
            "policy_request_number": 1, "ledger_sha256": _digest(self.ledger)})
        self.policy_requests = 1
        index = next(i for i, item in enumerate(self.observed)
            if item["row"]["triple_candidate_id"] == self.current["triple_candidate_id"])
        state = {"schema_version": PUBLIC_SCHEMA, "step_id": "s0000",
            "current_candidate_id": f"c{index:04d}",
            "evidence": [FreshReportSession._numeric(self, item["row"], i)
                         for i, item in enumerate(self.observed)],
            "tools": [{"tool_id": "t0000", "action": "regenerate_cxr",
                "model_id": "m0000", "seed": 1, "cost_units": 4}],
            "budget": {"limit_units": 5, "spent_units": 1,
                "planner_units_per_call": 1}, "history": []}
        self.active_state = validate_public_state(state)
        return copy.deepcopy(state)

    def receive_decision(self, decision):
        require(self.active_state is not None and not self.active_tool,
                "reserved_policy_request_required")
        validate_decision(decision, self.active_state)
        self.sink({"event": "validated_policy_decision", "decision": copy.deepcopy(decision)})
        if decision["action"] in ("stop", "abstain"):
            self.terminal = decision["action"] + "_unverified"
            self.active_state = None
            return False
        require(decision["action"] == "regenerate_cxr", "only_installed_image_action_allowed")
        self.active_tool = True
        return True

    def policy_failed(self):
        require(self.active_state is not None and not self.active_tool,
                "reserved_policy_request_required")
        self.sink({"event": "policy_failed_charged", "step_id": "s0000"})
        self.active_state = None
        self.terminal = "abstain_policy_failed"

    def finish_image(self, proposal, ledger):
        require(self.active_tool, "requested_image_action_required")
        gate.validate_ledger(ledger, self.anchor)
        require(all(ledger[key] == self.ledger[key] for key in
                ("call_budget", "max_retries", "execution_mode"))
                and ledger["pending_attempts"] == 0, "unchanged_execution_contract_required")
        requests = [event["request"] for event in ledger["events"]
                    if event["event"] == "attempt_reserved"]
        kinds = ["cxr_generator", "xrv", "report_generator", "chexbert"]
        models = ["roentgen_v2", "xrv", self.reference["report_model_id"], "chexbert"]
        require(1 <= len(requests) <= 4
                and [r["kind"] for r in requests] == kinds[:len(requests)]
                and [r["model_id"] for r in requests] == models[:len(requests)]
                and all(r["seed"] == 1 for r in requests)
                and ledger["charged_model_attempts"] == len(requests),
                "one_seed_one_fresh_image_report_verification_chain_required")
        result = transition = None
        passed = False
        if proposal is not None:
            proposal = gate.controller_view(proposal)
            gate.validate_observation(proposal, self.context)
            gate.completed_operation(proposal, ledger)
            require(len(requests) == 4 and ledger["failed_attempts"] == 0
                    and proposal["seed"] == 1 and proposal["cxr_model_id"] == "roentgen_v2"
                    and proposal["report_model_id"] == self.reference["report_model_id"],
                    "completed_new_image_with_own_report_required")
            inventory = self.observed + [{"row": proposal,
                "origin": {"kind": "completed_ledger_receipt"}}]
            result = gate.assess_fresh_output(self.initial["triple_candidate_id"],
                proposal["triple_candidate_id"], inventory, self.context, ledger_snapshot=ledger)
            # Preserve both the original fixed baseline and the retained report.
            transition = image_gate.compare(self.reference, proposal)
            passed = result["proposal_passes_proxy_preservation"] and transition["exploratory_gate_pass"]
            self.observed = inventory
            if passed:
                self.current = proposal
        else:
            require(ledger["failed_attempts"] == 1
                    and ledger["events"][-1]["event"] == "attempt_failed",
                    "missing_output_requires_durable_charged_failure")
        history = {"step_id": "s0000", "action": "regenerate_cxr", "tool_id": "t0000",
            "accepted_proxy_transition": bool(passed), "failed": proposal is None,
            "resolved_count": sum(len(v) for v in transition["removed_opposition_fact_ids"].values()) if transition else 0,
            "new_opposition_count": sum(len(v) for v in transition["new_opposition_fact_ids"].values()) if transition else 0,
            "lost_comparison_count": sum(len(v["comparable"]) for v in transition["lost_fact_ids"].values()) if transition else 0}
        self.sink({"event": "fresh_image_finished", "history": history,
            "charged_new_worker_attempts": len(requests), "full_ledger_sha256": _digest(ledger),
            "original_baseline_veto": result, "retained_reference_transition": transition})
        self.history.append(history)
        self.ledger = copy.deepcopy(ledger)
        self.active_state = None
        self.active_tool = False
        self.terminal = "proxy_preserving_image_change_unverified" if passed else "unresolved_reference_retained"
        return bool(passed)

    def result(self):
        require(self.terminal is not None and self.active_state is None and not self.active_tool,
                "sealed_terminal_session_required")
        return {"schema_version": VERSION, "case_id": self.anchor.case_id,
            "status": self.terminal, "baseline_candidate_id": self.initial["triple_candidate_id"],
            "reference_candidate_id": self.reference["triple_candidate_id"],
            "selected_candidate_id": self.current["triple_candidate_id"] if self.current else None,
            "observed_candidate_ids": [item["row"]["triple_candidate_id"] for item in self.observed],
            "accepted_proxy_transitions": sum(h["accepted_proxy_transition"] for h in self.history),
            "charged_new_worker_attempts": self.ledger["charged_model_attempts"],
            "charged_policy_requests": self.policy_requests, "history": copy.deepcopy(self.history),
            "frozen_rule_action": self.frozen_rule["action"],
            "historical_cost_scope": "shared_sunk_not_measured_not_zero",
            "clinical_repair_success": False, "clinical_accuracy": None,
            "confirmed_faulty_modality": None, "training_performed": False,
            "input_scope": "numeric_state_only_not_clinical_critic",
            "action_scope": "one_seed_one_roentgen_probe_with_own_report_and_verifiers",
            "image_regeneration_installed": True, "efficient_dynamic_stop_demonstrated": False}
