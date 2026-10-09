# Cached scorer coverage and tie diagnostic

This CPU-only diagnostic follows the completed, audited four-image/two-report
expert control. It explains score availability and discrimination; it is not
a new selector, calibration, clinical fault diagnosis or regeneration loop.

## Immutable evidence and separate profiles

1. The fresh engineering control retains two synthetic EHRs, four fixed CXRs
   and eight reports. Require its completed, hash-bound metadata audit and
   exactly recompute its frozen pre-BioViL selection and paired comparison.
   Its fourteen-field schema has eight enabled XRV heads; disabled heads remain
   unknown. Preserve the exact source two-term proxy prefix.
2. The already inspected 80-EHR development bank retains all 960 candidates
   on 240 image slots and four report experts. Reuse its authenticated legacy
   loader and original fourteen-field proxy evidence. Do NOT apply the fresh
   thresholds, refit labels or pool the two profiles. Retain the original
   four-component ranking prefix (gate, opposition, EHR support, image/report
   support). Structure/runtime/ID are excluded only to diagnose prefix ties;
   original full-key winners and actions are unchanged.

## Predeclared readouts

- Count EHR states once per fixed case, image states once per case/image ID,
  report states once per report ID. Also report distinct artifact hashes.
- Count no-direct-EHR cases, image positive/negative/unknown/uncertain coverage,
  reports with no comparable image evidence, and zero-opposition/no-comparison
  rows. Zero opposition with no comparison is not a good report.
- Enumerate every unordered pair of reports on each same-image slot. Pair
  counts are not independent patients. Preserve duplicate text artifacts.
- Describe source-prefix ties and absolute BioViL gaps on available tied pairs.
  Average gaps within an EHR, then over available EHRs; expose availability
  counts. Do not call this an all-cohort estimate, clinical accuracy, significance,
  benefit, or endpoint-oracle selection.
- Fresh endpoint lineage must match every pair. Historical endpoints cover
  only the previously scored selected union. All other candidates stay NA with
  an explicit reason; do not substitute, rank only scored rows or impute zero.

## Execution and output

`TriCompose-v1.2/benchmarks/diagnose_score_coverage.py` requires an existing CPU
Slurm allocation before opening caches. No new job submission, model runtime,
raw patient input, report body, image pixels, external API or checkpoint change
is involved. The output is a NEW atomic directory beneath
`artifacts/protected/tricompose_v1_2/score_coverage_diagnostics/`, with protected
permissions, source/result SHA256s, separate per-profile candidate/model/pair
CSVs and a bilingual `RESULTS_CN_EN.md`. Existing runs cannot be overwritten.

This analysis is development evidence. Do not alter masks, thresholds, model
priorities, original candidates or clinical claims to improve the observed
endpoint. Future prospective controls need separately frozen case inventories,
policies, independent endpoints and actual cost accounting. GPU submissions
still require full script/resource review and explicit approval.
