"""LLM selects interventions; local immutable fact guards accept or roll back.

Only the initial snapshot and completed requested observations are exposed.
This controller does not impose the legacy deterministic top-score choice.
It does not infer clinical truth or grant live-worker authorization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from tricompose_v12.probe_repair_v1 import action_credit, digest, edges, validate_snapshot

from . import VERSION
from .contracts import PUBLIC_SCHEMA, require, validate_decision, validate_public_state

ACTION_MAP = {"regenerate_report": "report_probe", "regenerate_cxr": "image_probe"}
TOOL_UNITS = {"regenerate_report": 2, "regenerate_cxr": 4}
SUPPORTED_MODES = ("authored_fixture", "cache_replay")


@dataclass(frozen=True)
class Tool:
    """Local menu binding; model/seed never chosen by a shell string from LLM."""
    tool_id: str
    action: str
    model_id: str
    seed: int = 0


@dataclass(frozen=True)
class ProbeRequest:
    step_id: str
    tool_id: str
    action: str
    slot: tuple
    parent_sha256: str
    reserved_units: int


class CachedExecutor:
    """Offline snapshots ONLY; never call this fresh generation."""
    mode = "cache_replay"

    def __init__(self, grid):
        self._grid = grid

    def execute(self, request, current):
        from tricompose_v12.probe_repair_v1 import snapshot
        require(digest(current) == request.parent_sha256, "executor_parent_changed")
        require(request.slot in self._grid, "cache_slot_unavailable")
        return snapshot(self._grid[request.slot])


class LLMRepairController:
    def __init__(self, initial, tools, *, budget_units=16, initial_units=4,
                 planner_units=1, max_steps=8, event_sink=None):
        validate_snapshot(initial)
        require(type(budget_units) is int and 1 <= budget_units <= 10000
                and type(initial_units) is int and 0 <= initial_units <= budget_units
                and type(planner_units) is int and 1 <= planner_units <= budget_units
                and type(max_steps) is int and 1 <= max_steps <= 64, "bounded_budget_and_steps")
        require(0 < len(tools) <= 256 and all(isinstance(t, Tool) for t in tools), "registered_tools_required")
        self.tools = {}
        for index, tool in enumerate(tools):
            require(tool.tool_id == f"t{index:04d}" and tool.action in ACTION_MAP
                    and isinstance(tool.model_id, str) and bool(tool.model_id)
                    and type(tool.seed) is int and 0 <= tool.seed < 2**63
                    and (tool.action != "regenerate_report" or tool.seed == 0), "typed_local_tool")
            self.tools[tool.tool_id] = tool
        require(len({(t.action, t.model_id, t.seed) for t in tools}) == len(tools), "unique_tool_bindings")
        self.initial = deepcopy(initial)
        self.current = deepcopy(initial)
        self.budget, self.spent = budget_units, initial_units
        self.initial_units, self.planner_units, self.max_steps = initial_units, planner_units, max_steps
        self.observed = [deepcopy(initial)]
        self.ids = {initial["candidate_id"]}
        self.attempted = set()
        self.trace, self.history = [], []
        self.planner_calls = 0
        self.tool_attempts = 0
        self.sink = event_sink
        self.ran = False

    @staticmethod
    def slot(value):
        lin = value["lineage"]
        return (lin["cxr_model_id"], lin["cxr_seed"], lin["report_model_id"])

    def _target_slot(self, tool):
        model, seed, report = self.slot(self.current)
        return (model, seed, tool.model_id) if tool.action == "regenerate_report" else (tool.model_id, tool.seed, report)

    def _event(self, value):
        event = deepcopy(value)
        if self.sink is not None: self.sink(deepcopy(event))
        self.trace.append(event)

    def _evidence(self):
        result = []
        for index, value in enumerate(self.observed):
            uncertainty = {}
            for public, local in (("ehr", "ehr"), ("image", "xrv"), ("report", "chexbert")):
                for state in ("unknown", "uncertain"):
                    uncertainty[public+"_"+state] = sum(s == state for s in value["states"][local].values())
            result.append({"evidence_id": f"e{index:04d}", "candidate_id": f"c{index:04d}",
                "edges": {edge: {"known": len(v["known"]), "comparable": len(v["comparable"]),
                    "supported": len(v["support"]), "opposed": len(v["opposition"]),
                    "positive_supported": len(v["positive_support"])} for edge, v in edges(value).items()},
                "quality": {"image_basic_valid": value["quality"]["cxr_basic_validity_pass"],
                    "report_structure": value["quality"]["report_structure_quality_score_0_1"],
                    "artifact_failures": value["artifact_gate_failures"]}, "uncertainty": uncertainty})
        return result

    def public_state(self, step):
        require(type(step) is int and 0 <= step < self.max_steps, "bounded_step")
        # Only public ordinal IDs; never reuse source case IDs or artifact IDs.
        models = {name: f"m{i:04d}" for i, name in enumerate(sorted({t.model_id for t in self.tools.values()}))}
        menu = []
        for tool in self.tools.values():
            slot = self._target_slot(tool)
            if (slot == self.slot(self.current) or (tool.action, slot) in self.attempted
                    or self.spent + TOOL_UNITS[tool.action] > self.budget):
                continue
            menu.append({"tool_id": tool.tool_id, "action": tool.action,
                         "model_id": models[tool.model_id], "seed": slot[1], "cost_units": TOOL_UNITS[tool.action]})
        current_index = next(i for i, v in enumerate(self.observed) if v["candidate_id"] == self.current["candidate_id"])
        state = {"schema_version": PUBLIC_SCHEMA, "step_id": f"s{step:04d}",
            "current_candidate_id": f"c{current_index:04d}", "evidence": self._evidence(),
            "tools": menu, "budget": {"limit_units": self.budget, "spent_units": self.spent,
                "planner_units_per_call": self.planner_units}, "history": deepcopy(self.history)}
        return validate_public_state(state)

    def run(self, planner, executor):
        require(not self.ran, "new_controller_required_no_replay_resume")
        require(getattr(executor, "mode", None) in SUPPORTED_MODES, "live_generation_not_enabled_in_this_release")
        self.ran = True
        terminal = "step_limit"
        for step in range(self.max_steps):
            if self.spent + self.planner_units > self.budget:
                terminal = "budget_exhausted"; break
            self.spent += self.planner_units
            self.planner_calls += 1
            self._event({"event": "planner_reserved", "step_id": f"s{step:04d}",
                         "cost_units": self.planner_units, "spent_units": self.spent})
            state = self.public_state(step)
            try:
                decision = validate_decision(planner.propose(deepcopy(state)), state)
            except Exception:
                self._event({"event": "planner_failed", "step_id": state["step_id"], "charged": True})
                terminal = "planner_failed_or_invalid"; break
            self._event({"event": "proposal_validated", "state": state, "decision": decision})
            if decision["action"] in ("stop", "abstain"):
                terminal = decision["action"]; break
            tool = self.tools[decision["target_id"]]
            slot = self._target_slot(tool)
            units = TOOL_UNITS[tool.action]
            self.spent += units
            self.tool_attempts += 1
            self.attempted.add((tool.action, slot))
            request = ProbeRequest(state["step_id"], tool.tool_id, tool.action, slot, digest(self.current), units)
            self._event({"event": "tool_reserved", "request": request.__dict__, "spent_units": self.spent})
            # A planner cannot mutate local snapshots, and cannot bypass this guard.
            before = deepcopy(self.current)
            credit, accepted, failed = None, False, False
            try:
                after = executor.execute(request, deepcopy(before))
                validate_snapshot(after)
                require(self.slot(after) == slot and after["candidate_id"] not in self.ids,
                        "new_requested_observation_required")
                credit = action_credit(before, after, ACTION_MAP[tool.action])
                self.observed.append(deepcopy(after)); self.ids.add(after["candidate_id"])
                accepted = credit["replacement_allowed_under_proxy_contract"]
                if accepted: self.current = deepcopy(after)
            except Exception:
                failed = True  # No exception/body logged; reserved cost is retained.
            history = {"step_id": state["step_id"], "action": tool.action, "tool_id": tool.tool_id,
                       "accepted_proxy_transition": accepted, "failed": failed,
                       "resolved_count": sum(map(len, credit["resolved_proxy_conflicts"].values())) if credit else 0,
                       "new_opposition_count": sum(map(len, credit["new_proxy_conflicts"].values())) if credit else 0,
                       "lost_comparison_count": sum(map(len, credit["lost_comparable_fact_ids"].values())) if credit else 0}
            self.history.append(history)
            self._event({"event": "tool_completed", **history, "credit": credit,
                         "current_observation_sha256": digest(self.current)})
        return {"schema_version": VERSION, "mode": executor.mode, "terminal_reason": terminal,
            "initial_observation_sha256": digest(self.initial), "selected_observation": deepcopy(self.current),
            "budget_units": self.budget, "spent_units": self.spent,
            "planner_calls": self.planner_calls, "tool_attempts": self.tool_attempts,
            "accepted_proxy_transitions": sum(h["accepted_proxy_transition"] for h in self.history),
            "trace": deepcopy(self.trace), "history": deepcopy(self.history),
            "actual_generator_calls": 0, "actual_regeneration_executed": False,
            "clinical_fault_location": None, "clinical_repair_success": None,
            "cost_units_are_not_measured_gpu_time_or_api_price": True}


def tools_from_orders(image_order, report_order):
    rows = [("regenerate_report", model, 0) for model in report_order]
    rows += [("regenerate_cxr", model, seed) for model, seed in image_order]
    return [Tool(f"t{i:04d}", action, model, seed) for i, (action, model, seed) in enumerate(rows)]
