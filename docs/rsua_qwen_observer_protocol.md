# Same-cohort frozen Qwen image observer: pneumonia diagnostic

Prepared follow-up, not executed. Two current synthetic development cases have
explicit EHR–XRV pneumonia opposition; their image guard is unresolved. A
report-only switch cannot eliminate that fixed-label conflict. Before relying
on the Qwen image observer, test its pneumonia readout on the SAME existing
50 RSUA images used for XRV/BioViL-T/XraySigLIP. No new acquisition required.

## Reference and independence limits

Reuse the sealed seed-0 cohort, 25 published pneumonia and 25 paper-described
normal-cohort proxies. Official release calls the latter Non-Covid; preserve
this documented distinction. These are NOT independently adjudicated per-image
clinical labels. Lung segmentation validation is not disease adjudication.
Patient grouping, age domain and training overlap remain unverified. Existing
results have already been inspected: DEVELOPMENT, not untouched final testing.
The new Qwen predictions have not been observed when this protocol is written.
Even a perfect proxy result will not qualify clinical repair on synthetic CXRs.

Reuse all 50 existing lossless L PNGs from the BioViL run in original opaque
source order. Verify original BMP/PNG hash lineage against the prior XRV run.
No new cohort, substitution, easy-case filtering, conversion or preprocessing
tuning. CPU prepare parses only manifest/aggregate and derived prediction
metadata, performs stat/hash checks, and never opens pixels or cohort labels.

## Frozen execution contract

Use the same read-only local Qwen2.5-VL checkpoint, loader and exact eight-
finding IMAGE_PROMPT as the current image observer. Pixel bounds 200704–401408,
greedy decoding, seed 0, max_new_tokens 384, no retries. All weights eval/frozen.
The existing exact-spatial-uniformity PNG guard runs before a lazy model call;
it supplies the identical checked in-memory RGB object. Basic pass does not
certify anatomy. Model receives ONLY image pixels and the fixed generic prompt,
never reference labels, source IDs, old scores, reports or EHR content.

Reserve/fsync each load/call attempt before execution. At most 50 model calls
and one lazy model load; invalid JSON/token cap is unavailable, not negative.
For runtime failure, charge the failed attempt, stop, and retain all remaining
slots explicitly unattempted; no retry/replacement/resume. Blocked guard slots
retain null states and no call. Preserve all 50 denominators.
Persist/fsync predictions before opening the cohort labels or old XRV scores.
All source real images/labels are consumed only internally in the separately
approved Slurm job. No MIMIC EHR, report, previous/target CXR, external API,
download, training, new threshold/scorer or old-winner update.

## Predeclared metrics

Evaluate ONLY pneumonia against the published class proxy. Other seven
predicted findings have no reference here, remain unevaluated, and cannot be
assumed absent in the normal-proxy group.

- Full attempted/available/explicit/uncertain/unknown/blocked/failed/unattempted
  counts, including positive and negative reference groups separately.
- Explicit four-cell TP/FN/TN/FP counts; unresolved is a separate category.
- Correct explicit support over ALL reference-positive and ALL reference-
  negative slots, and their equal-weight mean. Report explicit coverage and
  conditional support accuracy separately, with the actual denominators.
- Conservative lower/upper correctness bounds retaining unavailable and
  abstained slots. These are missingness bounds, NOT confidence intervals or
  clinical accuracy. Explicit wrong states cannot become correct in the upper.
- Frozen-XRV default 0.5 and transported 0.55457607 readouts on the same image
  set, with coverage and same-state/disagreement counts, not a majority vote.
- No AUROC/AP/Brier/ECE from Qwen's discrete states; no invented probabilities
  for unknown/uncertain. No threshold fitting or best-template selection.
- Actual charged load/calls, failures, runtime, tokens, GPU dtype and allocated
  VRAM. No measured generation savings, LLM decision advantage, clinical fault
  localization or repair success is claimed.

Do not retrofit a qualification threshold after seeing this run. Compare the
class-specific signals alongside existing scorer results, retain limitations,
and keep the existing clinical attribution guard closed.

## Proposed resources / approval boundary

One debug A40, two CPUs, 32 GiB host RAM, ten-minute allocation cap, worker
process cap 540 seconds. This is a resource ceiling, not queue/runtime promise.
Earlier eight-head observer runs peak near 15.6 GiB allocated VRAM; the pinned
worker requires at least 24 GiB device memory. A 16 GiB P100/V100 cannot satisfy
that unchanged memory contract. Do not alter dtype/quantization to force fit.

Entry: `TriCompose-v1.2/real_validation/rsua_qwen_observer.py`.
Script: `TriCompose-v1.2/agent/slurm/17_rsua_qwen50_debug_a40.sbatch`.
The complete script/resources must be shown and explicitly approved before
sbatch. This protocol and CPU preparation do not authorize submission.
Public logs contain sanitized status/hash/runtime only. All outputs, attempts,
caches and private diagnostics use project group and 2770/0660 protected modes.

## Test contract

Invented fixtures test all four states, unavailable vs unknown, no implied
negative, full denominators, exact matched image/reference scope, guard blocking,
lazy calls, charge-before-call, runtime failures, no retries, source immutability
and pixel-only requests. No real reference row is printed or read by tests.
