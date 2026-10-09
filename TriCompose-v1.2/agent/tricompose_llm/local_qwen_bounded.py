"""Pre-call feasibility guard, not a repair of invalid model responses.

With no affordable untried tool, terminate explicitly in code before model
generation. With any tool available, the real frozen Qwen and unchanged
decision validator are used; an invalid decision still fails closed.
"""
from .contracts import DECISION_SCHEMA, validate_decision, validate_public_state
from .local_qwen_v2 import LocalQwenPlanner as TypedQwenPlanner

INTERFACE_VERSION = "tricompose-local-qwen-affordable-menu-v3"


class LocalQwenPlanner(TypedQwenPlanner):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.programmatic_terminal_stops = 0
        self.proposal_sources = []

    def propose(self, state):
        validate_public_state(state)
        no_tool = not state["tools"]
        self.proposal_sources.append({"request_index": len(self.proposal_sources),
            "step_id": state["step_id"], "source": "guard_empty_menu" if no_tool else "llm"})
        if no_tool:
            self.programmatic_terminal_stops += 1
            current = next(r for r in state["evidence"] if r["candidate_id"] == state["current_candidate_id"])
            return validate_decision({"schema_version": DECISION_SCHEMA, "step_id": state["step_id"],
                "action": "abstain", "target_id": None, "evidence_ids": [current["evidence_id"]],
                "reason_code": "insufficient_evidence"}, state)
        return super().propose(state)

    def audit(self):
        return {**super().audit(), "interface_version": INTERFACE_VERSION,
            "programmatic_terminal_stops": self.programmatic_terminal_stops,
            "proposal_sources": list(self.proposal_sources),
            "empty_menu_terminal_is_not_llm_decision": True,
            "policy_request_units_include_conservative_guard_overhead": True}
