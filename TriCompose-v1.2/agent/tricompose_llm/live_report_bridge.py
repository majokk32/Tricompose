"""New fresh-report session; historical cache controller stays immutable.

Pure numeric/receipt logic, no model factories or patient artifact reads.
The existing fresh-output gate is reused, not retuned. A second gate compares
against the currently retained report to prevent a later accepted regression.
Only switching CXR-only frozen report experts is supported in this version.
"""
from __future__ import annotations

import copy

import fresh_output_acceptance as gate
from tricompose_v12.invariant_verification import _digest
from tricompose_v12.report_expert_control import MODELS, compare, structure_ok

from .contracts import (PUBLIC_SCHEMA, require, validate_decision,
                        validate_public_state)

VERSION = "tricompose-fresh-report-bridge-v1"


class FreshReportSession:
    """Fresh receipts must extend the charged ledger, not appear from a cache.

    Policy requests have a separate durable counter. The numeric planning
    units equal worker reservations plus policy requests, not GPU seconds.
    Failed policy/worker attempts are never refunded. Real backend execution
    and filesystem authentication are enforced by the separate live runner.
    """

    def __init__(self, baseline, context, ledger, *, max_steps=3, sink):
        require(type(max_steps) is int and 1 <= max_steps <= 3 and callable(sink),
                "bounded_fresh_report_session_required")
        self.context = copy.deepcopy(context)
        self.anchor = gate.anchor_from_record(context["anchor"])
        gate.validate_ledger(ledger, self.anchor)
        require(ledger["pending_attempts"] == 0 and ledger["max_retries"] == 0,
                "serial_no_retry_prefix_required")
        base = gate.controller_view(baseline)
        gate.validate_observation(base, context)
        gate.completed_operation(base, ledger)
        require(base["report_model_id"] in MODELS and ledger["charged_model_attempts"] == 4,
                "one_completed_fresh_baseline_chain_required")
        self.initial = base
        self.current = base if structure_ok(base) else None
        self.observed = [{"row": base, "origin": {"kind": "completed_ledger_receipt"}}]
        self.ledger = copy.deepcopy(ledger)
        self.max_steps = max_steps
        self.sink = sink
        self.policy_requests = 0
        self.used = {base["report_model_id"]}
        self.history = []
        self.active_state = None
        self.active_tool = None
        self.terminal = None
        self.models = {m: f"m{i:04d}" for i, m in enumerate(MODELS)}
        self.tools = {f"t{i:04d}": m for i, m in enumerate(MODELS)}

    def _numeric(self, row, index):
        facts = gate.validate_observation(row, self.context)
        e = row["raw_edge_readouts"]
        uncertainty = {}
        for modality, field in (("ehr", "ehr"), ("image", "xrv"), ("report", "chexbert")):
            for state in ("unknown", "uncertain"):
                uncertainty[modality + "_" + state] = sum(f[field] == state for f in facts)
        return {"evidence_id": f"e{index:04d}", "candidate_id": f"c{index:04d}",
            "edges": {name: {"known": value["known_reference_facts"],
                "comparable": value["comparable_facts"], "supported": value["supported_facts"],
                "opposed": value["proxy_opposition_facts"],
                "positive_supported": value["supported_positive"]} for name, value in e.items()},
            "quality": {"image_basic_valid": True, "report_structure": float(structure_ok(row)),
                "artifact_failures": 0}, "uncertainty": uncertainty}

    def begin_planning(self):
        require(self.active_state is None and self.active_tool is None and self.terminal is None,
                "serial_policy_request_required")
        if self.current is None:
            self.terminal = "abstain_no_section_eligible_baseline"
            return None
        available = [(tid, m) for tid, m in self.tools.items() if m not in self.used]
        if self.policy_requests >= self.max_steps or self.ledger["call_budget"] - self.ledger["charged_model_attempts"] < 2:
            self.terminal = "abstain_budget_limit"
            return None
        if not available:
            self.terminal = "abstain_no_untried_report_expert"
            return None
        step = f"s{self.policy_requests:04d}"
        # The sink must fsync before the Qwen subprocess can be started.
        self.sink({"event": "policy_reserved", "step_id": step,
            "policy_request_number": self.policy_requests + 1, "ledger_sha256": _digest(self.ledger)})
        self.policy_requests += 1
        current_index = next(i for i, item in enumerate(self.observed)
            if item["row"]["triple_candidate_id"] == self.current["triple_candidate_id"])
        state = {"schema_version": PUBLIC_SCHEMA, "step_id": step,
            "current_candidate_id": f"c{current_index:04d}",
            "evidence": [self._numeric(item["row"], i) for i, item in enumerate(self.observed)],
            "tools": [{"tool_id": tid, "action": "regenerate_report", "model_id": self.models[m],
                "seed": 0, "cost_units": 2} for tid, m in available],
            "budget": {"limit_units": self.ledger["call_budget"] + self.max_steps,
                "spent_units": self.ledger["charged_model_attempts"] + self.policy_requests,
                "planner_units_per_call": 1}, "history": copy.deepcopy(self.history)}
        self.active_state = validate_public_state(state)
        return copy.deepcopy(state)

    def receive_decision(self, decision):
        require(self.active_state is not None and self.active_tool is None,
                "reserved_policy_request_required")
        validate_decision(decision, self.active_state)
        self.sink({"event": "validated_policy_decision", "decision": copy.deepcopy(decision)})
        if decision["action"] in ("stop", "abstain"):
            self.terminal = decision["action"] + "_unverified"
            self.active_state = None
            return None
        require(decision["action"] == "regenerate_report", "image_regeneration_not_installed")
        model = self.tools[decision["target_id"]]
        self.used.add(model)  # A failed expert is not automatically retried.
        self.active_tool = (decision["target_id"], model)
        return model

    def policy_failed(self):
        require(self.active_state is not None and self.active_tool is None, "reserved_policy_request_required")
        self.sink({"event": "policy_failed_charged", "step_id": self.active_state["step_id"]})
        self.terminal = "abstain_policy_failed"
        self.active_state = None

    def finish_report(self, proposal, ledger):
        require(self.active_tool is not None, "registered_report_tool_required")
        gate.validate_ledger(ledger, self.anchor)
        old = self.ledger
        require(ledger["events"][:len(old["events"])] == old["events"]
            and all(ledger[k] == old[k] for k in ("call_budget", "max_retries", "execution_mode"))
            and ledger["pending_attempts"] == 0, "unchanged_durable_prefix_required")
        suffix = ledger["events"][len(old["events"]):]
        calls = [e["request"] for e in suffix if e["event"] == "attempt_reserved"]
        tid, model = self.active_tool
        require(1 <= len(calls) <= 2 and [r["kind"] for r in calls] in
            (["report_generator"], ["report_generator", "chexbert"])
            and calls[0]["model_id"] == model
            and all(r["input_image_sha256"] == self.initial["cxr_sha256"] for r in calls)
            and ledger["charged_model_attempts"] - old["charged_model_attempts"] == len(calls),
            "one_charged_fresh_report_and_verifier_only")
        passed, result, transition = False, None, None
        if proposal is not None:
            proposal = gate.controller_view(proposal)
            gate.validate_observation(proposal, self.context)
            position = gate.completed_operation(proposal, ledger)
            require(position >= len(old["events"]) and len(calls) == 2
                and proposal["report_model_id"] == model
                and proposal["cxr_candidate_id"] == self.initial["cxr_candidate_id"]
                and proposal["cxr_sha256"] == self.initial["cxr_sha256"]
                and all(proposal["triple_candidate_id"] != x["row"]["triple_candidate_id"] for x in self.observed),
                "new_completed_same_image_report_required")
            proposed_inventory = self.observed + [{"row": proposal, "origin": {"kind": "completed_ledger_receipt"}}]
            result = gate.assess_fresh_output(self.initial["triple_candidate_id"],
                proposal["triple_candidate_id"], proposed_inventory, self.context, ledger_snapshot=ledger)
            transition = compare(self.current, proposal)
            passed = result["proposal_passes_proxy_preservation"] and transition["exploratory_gate_pass"]
            self.observed = proposed_inventory
            if passed:
                self.current = proposal
        else:
            require(any(e["event"] == "attempt_failed" for e in suffix),
                    "missing_output_requires_charged_failure")
        record = {"step_id": self.active_state["step_id"], "action": "regenerate_report", "tool_id": tid,
            "accepted_proxy_transition": bool(passed), "failed": proposal is None,
            "resolved_count": sum(len(v) for v in transition["removed_opposition_fact_ids"].values()) if transition else 0,
            "new_opposition_count": sum(len(v) for v in transition["new_opposition_fact_ids"].values()) if transition else 0,
            "lost_comparison_count": sum(len(v) for k, v in transition["lost_fact_ids"].items()
                if k.endswith("_comparable")) if transition else 0}
        self.sink({"event": "fresh_report_finished", "history": record,
            "charged_new_worker_attempts": len(calls), "full_ledger_sha256": _digest(ledger),
            "baseline_veto": result, "current_transition": transition})
        self.history.append(record)
        self.ledger = copy.deepcopy(ledger)
        self.active_state = self.active_tool = None
        return bool(passed)

    def result(self):
        require(self.active_state is None and self.active_tool is None and self.terminal is not None,
                "sealed_terminal_session_required")
        return {"schema_version": VERSION, "execution_mode": self.ledger["execution_mode"],
            "status": self.terminal, "case_id": self.anchor.case_id,
            "baseline_candidate_id": self.initial["triple_candidate_id"],
            "selected_candidate_id": self.current["triple_candidate_id"] if self.current else None,
            "accepted_proxy_transitions": sum(h["accepted_proxy_transition"] for h in self.history),
            "observed_candidate_ids": [x["row"]["triple_candidate_id"] for x in self.observed],
            "charged_worker_attempts": self.ledger["charged_model_attempts"],
            "charged_policy_requests": self.policy_requests, "history": copy.deepcopy(self.history),
            "clinical_repair_success": False, "clinical_accuracy": None,
            "input_scope": "numeric_state_only_not_clinical_critic",
            "action_scope": "same_image_frozen_report_expert_switch_only",
            "image_regeneration_installed": False, "training_performed": False}
