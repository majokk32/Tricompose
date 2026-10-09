"""Execution-time image-evidence veto before any backend construction.

This new entry boundary leaves consumed sessions and acceptance gates unchanged.
Authentication and GPU/Slurm authorization belong to the caller/backend factory.
No approved backend means an explicit deferred request, not fictitious inference.
One dispatch attempt is not one model call; worker costs stay in their own ledger.
"""
from __future__ import annotations

from copy import deepcopy

from .contracts import require
from .image_evidence_guard import guarded_decision

VERSION = "tricompose-guarded-action-dispatch-v1"


class GuardedActionDispatcher:
    def __init__(self, *, sink):
        require(callable(sink), "durable_guard_event_sink_required")
        self.sink = sink
        self.started = False
        self.finished = False
        self.backend_dispatch_attempts = 0

    def dispatch(self, decision, state, guard, *, backend_factory=None):
        """Exactly one decision; never instantiate an uncertain or absent tool.

        The sink must fsync each event before returning. Callback output stays
        private/unvalidated and cannot become a clinical verdict here. The caller
        must apply the original receipt/cost/acceptance gate to any fresh output.
        """
        require(not self.started, "single_serial_guarded_dispatch_only")
        outcome = guarded_decision(decision, state, guard)
        # Even a later fsync failure must not permit replaying this dispatch.
        self.started = True
        self.sink({"event": "execution_guard_applied", "version": VERSION,
            "guard": deepcopy(guard), **deepcopy(outcome)})
        effective = outcome["effective_decision"]
        payload = None
        if effective["action"] in ("stop", "abstain"):
            status = "guard_abstained_unverified" if outcome["decision_withheld"] else "terminal_unverified"
        elif backend_factory is None:
            status = "deferred_separately_approved_backend_required"
        else:
            require(callable(backend_factory), "registered_lazy_backend_factory_required")
            # Reservation is durable before even constructing a backend.
            self.sink({"event": "backend_dispatch_reserved", "ordinal": 0,
                "action": effective["action"], "target_id": effective["target_id"]})
            self.backend_dispatch_attempts = 1
            try:
                backend = backend_factory()
                require(callable(backend), "registered_backend_callback_required")
                payload = backend(deepcopy(effective))
                status = "backend_returned_unvalidated_acceptance_gate_required"
            except Exception:
                status = "backend_failed_attempt_retained_no_retry"
                payload = None
            self.sink({"event": "backend_dispatch_finished", "ordinal": 0, "status": status})
        result = {"schema_version": VERSION, "status": status, **outcome,
            "backend_dispatch_attempts": self.backend_dispatch_attempts,
            "actual_worker_model_calls": None if self.backend_dispatch_attempts else 0,
            "clinical_fault_location": None, "clinical_acceptance": False,
            "automatic_retry_allowed": False}
        self.sink({"event": "guarded_dispatch_sealed", "result": deepcopy(result)})
        self.finished = True
        return result, payload
