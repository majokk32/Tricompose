# V1.2 protocol: error localization and targeted regeneration

Status (2026-09-22): specification only; no repair policy has been executed.
Entry gate: review and approve calibrated V1.1 evidence, prompt-response smoke,
and static baselines. See `version_roadmap.md`. Do not bypass that gate by
renaming the present uncalibrated selector as an Agent.
The first scientific check is frozen-scorer discrimination on protected real
matched pairs versus independently checked mismatches, using patient-disjoint
calibration and final-test splits. The eight-case synthetic intervention smoke
only checks plumbing and cannot replace that real-data gate. See
`further_development.md` for the corrected execution order.

## What the current bank can and cannot establish

The August pool has 80 fixed synthetic EHRs, three CXRs per EHR and four
CXR-only reports per image. `cxrmate_single` is not CXRMate-ED. Agreement between
two reports conditioned on the same image is correlated evidence, not an
independent confirmation that the image or classifier is wrong.

Only eight EHRs currently provide explicit comparable radiographic evidence.
The remaining EHRs must not be fabricated, enriched with target labels, dropped,
or regenerated to make localization easier. Report-CXR verification can still
operate, but EHR-supported attribution must abstain where information is absent.
XRV has no device head; a report device statement cannot be declared false just
because this classifier returns unknown.

## First deliverable after V1.1 approval: a controlled benchmark

Freeze separate development, threshold-calibration and final-test case groups.
Build a protected corruption manifest; never overwrite generation artifacts.
Each record must retain source candidate hashes, corruption type, expected
target modality, changed evidence, and independent adjudication status.

| Intervention | What it tests | Required safeguard |
|---|---|---|
| Different-case report swap | Report mismatch | A random swap alone may remain clinically compatible; retain accidental matches |
| Different-case CXR swap, reports held fixed | CXR mismatch | Record that this is an intervention, not natural model-error attribution |
| Same-disease swap | Specificity beyond common diagnoses | Require an independently verified distinguishing fact; otherwise mark unavailable |
| Minimal finding presence/negation change | Explicit contradiction detection | Only mutate an already explicit statement; preserve unrelated facts |
| Laterality/severity/device/temporal change | Fine-grained clinical error | Run only when independent evidence/extraction supports that attribute |
| No corruption | False-positive and unnecessary-repair rate | Include normal, abnormal and underconditioned cases |

Altering only a CheXbert/XRV label vector tests scoring arithmetic, not clinical
localization. A benchmark must intervene on the artifact and run an independent
check. Original generated pairs are not automatically positive clinical gold.
Real-anchor benchmark construction and all learned evaluator inference require
separately approved Slurm scripts and protected outputs.

## Policy contract to implement after benchmark review

Input state:

```text
fixed case_id + ehr_sha256
candidate IDs, parent hashes, model revisions and seeds
per-edge direct support / explicit contradiction / unknown / coverage
weak-context flags (never hard contradiction)
scorer provenance + calibration status + uncertainty
call history, observed GPU time, remaining budgets
```

Allowed actions: `select`, `regenerate_cxr`, `regenerate_report`,
`switch_report_model`, `verify_more`, `reject`. No `replace_ehr` action.
Each action records the actual evidence IDs and rule version that triggered it.

Conservative decision order:

1. Reject invalid/empty artifacts as artifact failures, not disease disagreement.
2. If no direct comparable evidence exists, abstain on the relevant EHR edge;
   never convert missingness to success or contradiction.
3. If an independently supported image finding contradicts one report, try
   another report candidate/model within budget.
4. Attribute a CXR error only with adequate calibrated, independently supported
   evidence; a classifier-report disagreement alone is insufficient.
5. If evidence is conflicting or sparse, verify more or abstain. Optional
   report-to-CXR cycles are secondary and cannot establish correctness of EHR.
6. Keep the EHR fixed. At a predeclared budget limit, return an accepted triple
   or a rejection with reason; never hide a rejected case from the denominator.

Do not set clinical thresholds by guessing constants in this document. Freeze
them using reviewed calibration data, then record the policy/config hash.

## Required comparisons and accounting

Compare operational fixed path, preseeded random, all fixed paths, static
reranking and targeted repair on the same fixed cases and budget grid. The best
fixed path chosen on final-test data is only a descriptive upper bound.

An offline replay of the existing full candidate bank is useful for debugging,
but it does not save the computation already spent creating the bank. Report
replay-simulated cost separately from actual new-run GPU time. Count generation,
verification, failed calls, retries and optional cycles; do not charge only the
winning candidate.

Report per-class localization confusion, abstention/coverage, false-repair rate,
repair success against independent evidence, contradictions, image/report
quality, diversity, rejection and complete cost. Progress to V1.3 only if
targeted repair improves the quality-cost trade-off over static reranking.
