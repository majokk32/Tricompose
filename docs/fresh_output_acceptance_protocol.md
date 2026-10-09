# Fresh observed-output adapter / 新推理回执验收接口

This separately frozen DEVELOPMENT interface adapts the already frozen fresh
eight-enabled-XRV-head receipt profile. It does NOT convert that profile into
the legacy fourteen-head cache, retune thresholds, train, generate data or
change existing live workers. No model submission is authorized by this file.

## Inputs and trust

Callable: `TriCompose-v1.2/tools/fresh_output_acceptance.py::assess_fresh_output`.
Pass an initial baseline ID, proposed ID (or None), an explicitly ordered list
of observation metadata, the fixed EHR/scorer context and the full durable
case-ledger snapshot. Only already available observations can be proposed.
The first observation is the initial baseline. Alternate endpoints and old
winner/rank fields are excluded by the row projection.

Observation origins distinguish pinned, previously authenticated cached
references (shared sunk costs, NOT zero historical cost or fresh calls) from
completed ledger receipts. Cached references precede new acquisitions. A live
observation must bind to one successful CheXbert completion and its report,
XRV and image-generator dependencies, including artifact/label hashes,
partial/completed receipt IDs, model, seed, input image and fixed EHR anchor.
Every ledger is replayed exactly; failed or pending attempts keep their cost.

Self-hashes and an origin marker do not establish trust. The controller or
auditor MUST authenticate source manifests, artifacts, frozen models, scorer
configuration and PNG metadata before supplying records. This pure layer
rechecks identities/arithmetic, not source bytes, anatomy or report assertions.
The archived smoke additionally pins the previous plan/run/audit manifests and
selected metadata artifacts. It does not re-open checkpoints or report/image
bodies, or rerun the prior audit's complete source-artifact scan.

Fourteen named states remain in the record, but only the frozen eight XRV heads
may be observable. The other six must remain unknown. Preserve anchor source
category provenance, unknown/uncertain, weak-context boundaries, NA denominators
and all three raw edges. No global No-Finding expansion or probability claim.

## Decision

1. An unchanged section-valid, nonempty reference stays unchanged/unverified.
2. Same-image proposals reuse the immutable cross-expert report predicate:
   fact-ID and comparison preservation, no new proxy opposition, strict evidence
   gain, nonduplicate output and no worse common structure/risk flags.
3. For a changed image, require the frozen explicit direct-EHR/XRV opposition
   branch basis. Reuse the frozen cross-image predicate against BOTH the initial
   reference and the first gate-passing observed same-image report, if distinct.
   Select that comparison reference by fixed expert priority, then observation
   order. Never use an unseen full-bank winner or BioViL.
4. Failed/missing proposals fall back to the initial section-eligible reference,
   not a new reranking operation. If that reference is ineligible, return no
   output. Context/lineage tampering is an error, not a clinical veto.
5. Passes mean only proxy-preserving change/unverified. Always keep clinical
   acceptance/repair flags false and confirmed faulty modality None. The
   interface cannot authorize a model call or claim independent fault evidence.

This is a final-output adapter, NOT an online action loop. Its fallback is
explicitly the initial reference, like the preceding legacy output veto; this
does not modify the historical bounded-retry selector's static fallback.
Before future sequential use, a separately reviewed controller must declare
its incumbent/stop/acquisition policy. Do not reinterpret these post-selection
checks as measured early stopping or new regeneration successes.

## Test and archived smoke

Use invented fixtures for provenance/mask, cross-case and same-artifact drift,
hash-preserving state tampering, missingness, true fact gain versus silence,
both-reference comparisons, observation boundaries, endpoint isolation, ledger
refund/failure/pending/reuse behavior and all safety flags.

In the existing CPU allocation only, authenticate the immutable completed
two-case bounded-retry DEVELOPMENT run and its audit. For each fixed EHR, make
two independent checks: observed same-image static proposal, and its already
completed new-image proposal. Four checks are NOT four cases, new calls,
chained online outcomes, a new efficacy experiment or an untouched holdout.
Report the two real ledgers' eight attempts ONCE; keep all old outputs intact.

Write a new atomic protected run: decisions, summary, bilingual metadata-only
report and manifest. Refuse overwrite; directories 2770/files 0660. Freeze the
worker/tests/protocol and source fingerprints. No report/EHR body, image pixel,
model/checkpoint, external API, new threshold, model download or GPU execution.
Every future Slurm submission still requires the full script/resources shown
first and explicit subsequent approval.
