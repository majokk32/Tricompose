"""Bounded, hash-chained single-case call accounting; no model implementation.

Reservation must be journaled BEFORE a backend starts. Failed/in-flight calls
keep their cost. This module does not select models or infer clinical success.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
import math

from .invariant_verification import _digest, _HASH, _ID

SCHEMA = "tricompose-bounded-execution-ledger-v1"
KINDS = {"cxr_generator", "xrv", "report_generator", "chexbert"}
PARENT_KIND = {"cxr_generator": None, "xrv": "cxr_generator",
    "report_generator": "xrv", "chexbert": "report_generator"}
FAILURES = {"runtime_exception", "timeout", "interrupted", "invalid_result", "cuda_out_of_memory"}


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


@dataclass(frozen=True)
class CallRequest:
    operation_id: str
    case_id: str
    ehr_anchor_sha256: str
    kind: str
    model_id: str
    frozen_model_audit_sha256: str
    seed: int
    parent_operation_id: str | None = None
    input_image_sha256: str | None = None
    input_report_sha256: str | None = None

    def __post_init__(self):
        if (any(not isinstance(v, str) or not _ID.fullmatch(v) for v in
                (self.operation_id, self.case_id, self.model_id))
                or self.kind not in KINDS or not _hash(self.ehr_anchor_sha256)
                or not _hash(self.frozen_model_audit_sha256) or type(self.seed) is not int or self.seed < 0
                or self.parent_operation_id is not None and
                (not isinstance(self.parent_operation_id, str) or not _ID.fullmatch(self.parent_operation_id))
                or any(v is not None and not _hash(v) for v in (self.input_image_sha256, self.input_report_sha256))):
            raise ValueError("invalid immutable call request")
        if self.kind == "cxr_generator":
            if any(v is not None for v in (self.parent_operation_id, self.input_image_sha256, self.input_report_sha256)):
                raise ValueError("cold-start CXR call cannot have image/report parents")
        elif (self.parent_operation_id is None or self.input_image_sha256 is None
                or (self.input_report_sha256 is not None) != (self.kind == "chexbert")):
            raise ValueError("incomplete call dependency/input contract")

    def record(self):
        return dict(self.__dict__)

    @property
    def sha256(self):
        return _digest(self.record())


@dataclass(frozen=True)
class CallResult:
    output_artifact_sha256: str
    verification_receipt_id: str | None = None

    def __post_init__(self):
        if not _hash(self.output_artifact_sha256) or self.verification_receipt_id is not None and not _hash(self.verification_receipt_id):
            raise ValueError("invalid output/receipt hash")

    def record(self):
        return dict(self.__dict__)


class BudgetExhausted(RuntimeError):
    pass


class RetryExhausted(RuntimeError):
    pass


class BoundedCallLedger:
    """Serial case-level ledger. Durable sink exceptions stop execution.

    `execution_mode` is a recorded claim, NOT GPU authorization. The deployed
    backend and Slurm approval boundary are outside this accounting primitive.
    A receipt ID is supplied by an independently validating adapter; a hash is
    not itself proof of checkpoint freezing or clinical correctness.
    """

    def __init__(self, *, case_id, ehr_anchor_sha256, call_budget, max_retries, execution_mode, sink):
        if (not isinstance(case_id, str) or not _ID.fullmatch(case_id) or not _hash(ehr_anchor_sha256)
                or type(call_budget) is not int or call_budget < 0
                or type(max_retries) is not int or max_retries < 0
                or execution_mode not in {"invented_fixture_no_models", "approved_slurm_backend"}
                or not callable(sink)):
            raise ValueError("invalid frozen ledger contract")
        self._settings = (case_id, ehr_anchor_sha256, call_budget, max_retries, execution_mode)
        self._sink = sink; self._requests = {}; self._attempts = Counter()
        self._results = {}; self._failures = {}; self._pending = None; self._events = []

    @property
    def case_id(self): return self._settings[0]

    @property
    def anchor(self): return self._settings[1]

    @property
    def budget(self): return self._settings[2]

    @property
    def max_retries(self): return self._settings[3]

    @property
    def mode(self): return self._settings[4]

    @property
    def contract_sha256(self): return _digest([SCHEMA, *self._settings])

    def _emit(self, payload):
        base = {"schema_version": SCHEMA, "execution_mode": self.mode, "case_id": self.case_id,
            "ehr_anchor_sha256": self.anchor, "event_index": len(self._events),
            "execution_contract_sha256": self.contract_sha256,
            "previous_event_sha256": self._events[-1]["event_sha256"] if self._events else None, **payload}
        event = {**base, "event_sha256": _digest(base)}
        # Never run a backend if persisting its reservation fails. A sink must
        # durably append/fsync in prospective use; in-memory sinks are tests.
        self._sink(copy.deepcopy(event))
        self._events.append(event)

    @property
    def charged_attempts(self):
        return sum(self._attempts.values())

    def _validate_request(self, request):
        if not isinstance(request, CallRequest) or request.case_id != self.case_id or request.ehr_anchor_sha256 != self.anchor:
            raise ValueError("fixed case/EHR anchor changed")
        previous = self._requests.get(request.operation_id)
        if previous is not None and request.sha256 != previous.sha256:
            raise ValueError("retry/reuse changed model seed lineage or audit")
        if request.kind != "cxr_generator":
            parent_id = request.parent_operation_id
            if parent_id not in self._results or self._requests[parent_id].kind != PARENT_KIND[request.kind]:
                raise ValueError("successful phase dependency required")
            parent = self._requests[parent_id]; result = self._results[parent_id]
            expected_image = result.output_artifact_sha256 if request.kind == "xrv" else parent.input_image_sha256
            if request.input_image_sha256 != expected_image:
                raise ValueError("parent image hash changed")
            if request.kind == "chexbert" and request.input_report_sha256 != result.output_artifact_sha256:
                raise ValueError("parent report hash changed")

    def reserve(self, request):
        self._validate_request(request)
        if self._pending is not None:
            raise RuntimeError("unfinished attempt must be resolved; no parallel case calls")
        if request.operation_id in self._results:
            raise ValueError("successful call requires explicit zero-cost reuse")
        if self.charged_attempts >= self.budget:
            raise BudgetExhausted("model-call budget exhausted")
        attempts = self._attempts[request.operation_id]
        if attempts and (not self._failures[request.operation_id]["retryable"] or attempts > self.max_retries):
            raise RetryExhausted("failed call retry limit exhausted")
        reservation_id = _digest([SCHEMA, request.sha256, attempts + 1])
        self._emit({"event": "attempt_reserved", "reservation_id": reservation_id,
            "request": request.record(), "request_sha256": request.sha256,
            "attempt_number": attempts + 1, "charged_model_attempts": 1,
            "cumulative_charged_model_attempts": self.charged_attempts + 1})
        self._requests[request.operation_id] = request
        self._attempts[request.operation_id] += 1
        self._pending = (reservation_id, request.operation_id)
        return reservation_id

    def _request_for(self, reservation_id):
        if self._pending is None or self._pending[0] != reservation_id:
            raise ValueError("unknown or completed reservation")
        return self._requests[self._pending[1]]

    def _elapsed(self, value):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid backend elapsed time")
        return float(value)

    def complete(self, reservation_id, result, *, elapsed_seconds):
        request = self._request_for(reservation_id)
        if not isinstance(result, CallResult) or (result.verification_receipt_id is not None) != (request.kind in {"xrv", "chexbert"}):
            raise ValueError("generator/scorer result phase differs")
        self._emit({"event": "attempt_completed", "reservation_id": reservation_id,
            "operation_id": request.operation_id, "result": result.record(),
            "backend_elapsed_seconds": self._elapsed(elapsed_seconds), "charged_model_attempts": 0,
            "cumulative_charged_model_attempts": self.charged_attempts,
            "clinical_acceptance": False, "clinical_repair_success": False})
        self._results[request.operation_id] = result
        self._pending = None

    def fail(self, reservation_id, *, error_code, retryable, elapsed_seconds):
        request = self._request_for(reservation_id)
        if error_code not in FAILURES or type(retryable) is not bool:
            raise ValueError("sanitized operational failure required")
        self._emit({"event": "attempt_failed", "reservation_id": reservation_id,
            "operation_id": request.operation_id, "error_code": error_code, "retryable": retryable,
            "backend_elapsed_seconds": self._elapsed(elapsed_seconds), "charged_model_attempts": 0,
            "cumulative_charged_model_attempts": self.charged_attempts,
            "failure_is_clinical_contradiction": False, "clinical_acceptance": False})
        self._failures[request.operation_id] = {"error_code": error_code, "retryable": retryable}
        self._pending = None

    def reuse(self, request):
        self._validate_request(request)
        if self._pending is not None or request.operation_id not in self._results:
            raise ValueError("only committed successful calls can be reused")
        self._emit({"event": "successful_result_reused", "operation_id": request.operation_id,
            "request_sha256": request.sha256, "result": self._results[request.operation_id].record(),
            "charged_model_attempts": 0, "cumulative_charged_model_attempts": self.charged_attempts})
        return self._results[request.operation_id]

    def snapshot(self):
        attempts = Counter()
        for name, count in self._attempts.items(): attempts[self._requests[name].kind] += count
        return {"schema_version": SCHEMA, "case_id": self.case_id, "ehr_anchor_sha256": self.anchor,
            "execution_contract_sha256": self.contract_sha256,
            "execution_mode": self.mode, "call_budget": self.budget, "max_retries": self.max_retries,
            "charged_model_attempts": self.charged_attempts, "charged_attempts_by_kind": dict(sorted(attempts.items())),
            "completed_operations": len(self._results),
            "failed_attempts": sum(e["event"] == "attempt_failed" for e in self._events),
            "pending_attempts": int(self._pending is not None),
            "model_execution_allowed_by_ledger": False, "clinical_acceptance": False,
            "clinical_repair_success": False, "measured_gpu_seconds": None,
            "actual_model_calls": 0 if self.mode == "invented_fixture_no_models" else None,
            "events": copy.deepcopy(self._events)}


def restore_ledger(events, *, case_id, ehr_anchor_sha256, call_budget, max_retries, execution_mode, sink):
    """Replay canonical journal events; pending reservations keep their charge.

    No in-flight operation is automatically retried on restart. First reconcile
    an abandoned attempt with its backend. Restoring doesn't call a model or
    append historical events into the prospective sink.
    """
    captured = []
    ledger = BoundedCallLedger(case_id=case_id, ehr_anchor_sha256=ehr_anchor_sha256,
        call_budget=call_budget, max_retries=max_retries, execution_mode=execution_mode, sink=captured.append)
    for expected in events:
        previous_count = len(captured)
        kind = expected["event"]
        if kind == "attempt_reserved":
            ledger.reserve(CallRequest(**expected["request"]))
        elif kind == "attempt_completed":
            ledger.complete(expected["reservation_id"], CallResult(**expected["result"]),
                elapsed_seconds=expected["backend_elapsed_seconds"])
        elif kind == "attempt_failed":
            ledger.fail(expected["reservation_id"], error_code=expected["error_code"],
                retryable=expected["retryable"], elapsed_seconds=expected["backend_elapsed_seconds"])
        elif kind == "successful_result_reused":
            ledger.reuse(ledger._requests[expected["operation_id"]])
        else:
            raise ValueError("unsupported journal event")
        if len(captured) != previous_count + 1 or _digest(captured[-1]) != _digest(expected):
            raise ValueError("journal hash chain/arithmetic/order differs")
    if not callable(sink): raise ValueError("prospective journal sink required")
    ledger._sink = sink
    return ledger
