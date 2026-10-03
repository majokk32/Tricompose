# Invariant verification interface / 固定 EHR 验证接口

This integrates evidence accounting into generation-history events. It does
not rerank candidates, tune a policy, execute a model or certify clinical repair.
The old eight-head scoped decision preview stays unchanged. This adapter uses
the separately named historical uncalibrated fourteen-head cache only.

## Contract and pipeline hook

1. Construct ONE immutable EHR anchor for each case. Bind the existing EHR/facts
   hashes, all four-state finding values and cached source categories. Reference
   evidence IDs depend on EHR evidence only, not the subsequently chosen image
   or report. No new EHR extraction, disease/device insertion or weak-context
   promotion is permitted.
2. After a requested generator/scorer slot returns, call
   `verify_candidate(candidate, anchor)` BEFORE passing its evidence receipt to
   a controller. Receipt contents bind candidate/artifact hashes and raw states,
   and expose support/opposition/missing coverage for all three edges. They
   contain no text/pixels, BioViL selection input or clinical truth.
3. On a proposed image replacement call
   `verify_transition(before_receipt, after_receipt, anchor)` before calling it
   a repair. Track added/lost fixed-EHR image support, added/removed opposition,
   missing observations and all-three support separately. Changed report-only
   artifacts do not establish image repair.
4. No direct EHR constraint means `unverified_no_direct_ehr_constraints`, NOT
   zero score, clinical failure, normal disease or permission to change EHR.
   Proxy improvements, regressions and mixed changes remain explicitly
   unvalidated. A positive weighted/global score alone cannot grant success.
5. Keep operational model/seed exploration separate from clinical acceptance.
   The interface never chooses the next model, rewrites the original action,
   filters a case, grants GPU authorization or supplies a clinical fault label.

Example integration with a validated cache adapter:

```python
anchor = anchor_from_cached_candidate(fixed_ehr_candidate)
receipt = verify_candidate(new_candidate, anchor)
transition = verify_transition(previous_receipt, receipt, anchor)
# Store the receipt/transition with the controller's original call ledger.
# A clinical_success flag cannot be inferred from proxy score improvement.
```

## Missingness and clinical scope

Explicit positive/negative states may support or oppose a reference. Unknown
and uncertain are missing comparison states, never negative. Unmentioned EHR
findings are not hallucination labels. Support/coverage rates with no explicit
reference remain NA. A normal/negative agreement can be valid. Device/location/
severity information outside this cached finding schema is not invented.

The historical global No-Finding source metric is NOT applied to raw receipt
states. Raw report labels are unverified; no new syntax guard or calibrated head
is claimed. Same-image report disagreement is correlated, not independent image
truth. Clinical acceptance, natural error localization and clinical repair
success stay unavailable even if every cached label agrees.

## Engineering execution and output

Freeze `TriCompose-v1.2/configs/invariant_verification_v1.json` before the cache
integration test. Inside an EXISTING Slurm CPU allocation, stream the original
requested slots from every registered case/method/budget/seed trial. Construct
receipts only for those observed candidates, deduplicate identical receipts,
and associate final selections with receipt IDs without changing them. Retain
all original invocation ledgers, action traces, terminal reasons and EHRs.
Repeated steps/seeds are not independent cases. CPU receipt work does not save
historical generation cost; no model call is hidden in the verification hook.

Write a new atomic protected run with immutable anchors, candidate receipts,
event/transition bindings, trial receipt bindings, per-method/budget status
counts, bilingual handoff and source/result hashes. Require Slurm before any
protected reads. Do not open raw patient inputs, report bodies or image pixels,
run new GPU inference, download weights or submit a job. Actual prospective
generation still requires its complete script/resources and explicit approval.
This integration is an engineering test on previously inspected development
data, not a new independent evaluation or demonstrated quality improvement.
