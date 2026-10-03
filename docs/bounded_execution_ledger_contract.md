# Bounded call ledger / 有预算的执行账本

This layer accounts for operations; it does not implement a new agent, score,
model priority or clinically validated repair policy. The archived full/partial
verification contracts and existing V1.0/V1.1 generators remain unchanged.

## Invariants

- Every case retains its fixed EHR anchor hash. Requests bind opaque operation
  IDs, model audit hash, model ID, seed, parent operation and image/report hashes.
  Retry/reuse cannot silently change any of these. Changing model/seed is a NEW
  operation, not a hidden retry or clinically confirmed repair.
- Phase order is CXR generator → XRV → report generator → CheXbert. A report
  call requires a successful image-scoring operation; a hash-bound partial
  receipt must be supplied by its validating adapter. Unknown/unvalidated
  evidence is not a clinically passing gate. The ledger never checks pixels,
  generated text or clinical findings, and does not turn a scorer result into
  clinical acceptance. The verifier, not this ledger, checks receipt contents.
- `reserve()` durably journals BEFORE an operation starts. It charges one
  single-case model-invocation ATTEMPT, even if launch later fails. This is
  deliberately conservative; a reservation is not evidence that GPU inference
  occurred. Completed/failed events add no second charge. No batching or hidden
  multi-model backend is supported by this unit-cost contract.
- Failed and in-flight reservations retain their charge. A failed generator or
  scorer cannot create a usable result. Retry limits apply to each immutable
  operation; all retries also consume budget. A successful output is explicitly
  reused at zero NEW model cost, with its original cost still in the ledger.
- Budget exhaustion leaves unfinished modalities unresolved. Operational OOM,
  timeout, malformed result or runtime failure is not clinical contradiction.
  No EHR is discarded, enriched, replaced or automatically labeled a failure.
- A serial case ledger has one in-flight operation. Hash-chained journals can
  be restored exactly. An interrupted reservation remains pending/charged and
  is not automatically replayed. Reconcile the worker status before resolving
  it; never refund or resume a potentially still-running model blindly.
- A prospective sink must append/fsync below `artifacts/protected/` before
  returning. A failed sink prevents the caller from receiving a reservation.
  The ledger alone is NOT a launcher, checkpoint-freezing audit or Slurm
  authorization. The deployment adapter must enforce Slurm, model freezing,
  artifact hashes, private sanitized logs, protected paths and one call/result.

## Current test boundary

`bounded_execution_fixture_v1.json` fixes TWO INVENTED IDs before execution.
The smoke program supplies no patient EHR, generated report or image, reads no
model/checkpoint and calls no model. It exercises partial/full verification
hooks on invented metadata/vectors and tests failure/retry accounting. Report
and image artifact hashes are explicitly invented, not files produced by
generators. Its elapsed time is CPU fixture overhead, not inference/GPU time.

The fixture run cannot establish clinical success, image quality, actual GPU
savings or a completed prospective generation backend. Backend integration
with authentic frozen scorer provenance and durable worker/log handling remains
subsequent work. Do not mark fresh inference as the historical legacy cache or
silently mix fourteen-head and eight-head calibration profiles.

No new GPU submission is authorized by these tests or this contract. Every
prospective inference job still requires its full script/resource request and
explicit approval. No external API, training or human feedback is required to
exercise this engineering layer.

## Dispatch boundary

`runtime_dispatch.py` supplies `ProtectedJournal` (exclusive creation, private
mode and flush/fsync) and `dispatch_operation` (Slurm guard BEFORE backend/log
activity). The backend must match the request's audit hash, frozen flag and
execution-mode declaration. Reservation precedes invocation. Timeout/runtime/
invalid-result outcomes are sanitized, charged and unresolved; no automatic
retry or scorer-to-clinical-error conversion occurs. A completion-journal error
leaves the original attempt pending/charged, not launched a second time.

Python stdout/stderr is redirected into the project-private operation log.
An actual subprocess/native runtime adapter must separately capture child file
descriptors and enforce approved resources, actual checkpoints, artifact hash
validation and scorer provenance. These model adapters are NOT yet installed;
the only supplied backend is explicitly invented, with no model input/body.

Journal restore checks the canonical event hash chain, phase dependencies,
attempt/cost arithmetic and a frozen contract hash. Budget, retry limit, case,
anchor or execution-mode changes cannot silently resume the same journal.
Both model-call accounting and validation/dispatch overhead must be measured
prospectively; fixture backend elapsed times are not GPU times. The existing
scorer/ranking policy is unchanged. No model script or sbatch is supplied or
submitted by this accounting smoke.
