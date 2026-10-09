# Cached report scope gate: completed development result

The [sealed protocol](cached_report_scope_gate_protocol.md), existing gate and
frozen predictions remain unchanged. This separate result note contains only
aggregate diagnostics, no generated report text or real patient data.

## What ran

`TriCompose-v1.2/tools/gate_cached_report_spans.py` applied the unchanged
`report_assertions.gate_assertion()` and cached literal scope checker to:

1. Both frozen extractors on all 56 existing wholly invented texts / 80
   designated language checks.
2. Qwen V2 on the same fixed 24 fully synthetic candidate slots / 22 exact
   report hashes, preserving all 336 original candidate/finding rows.

The standard-library CPU worker used existing Slurm allocation 12645021.
No new GPU/model/API call, download, training, prompt/rule tuning or submission
occurred. Full-vector gate outputs were written, fsynced and hash-frozen before
the authored reference key was parsed. The gate was previously developed after
examining this authored challenge; this follow-up is therefore post-hoc
development, not preregistration, untouched testing or independent clinical gold.

## Authored risk and coverage

The 80-check denominator includes every designated check, including unknown,
uncertain and noncommitted results. Conditional matches refer only to assertions
the gate retained, not overall accuracy.

| Fixed authored diagnostic | Qwen V2 | CheXbert |
|---|---:|---:|
| Raw state matches | 50/80 | 48/80 |
| Retained assertions / all checks | 24/80 (30.0%) | 30/80 (37.5%) |
| Retained assertions matching the authored reference | 24/24 | 30/30 |
| Raw errors not retained | 30 | 32 |
| Matching non-unknown assertions not retained | 7 | 4 |
| Matching unknown checks with no assertion to release | 19 | 14 |
| Total noncommitted checks | 56 | 50 |

For Qwen, the seven matching non-unknown assertions not retained comprise four
positive, two negative and one uncertain assertion. The CheXbert four are
positive. Thus the earlier total of 26 Qwen matching noncommitted checks does
not mean 26 supported clinical facts were lost: 19 had no model assertion.

All raw errors are withheld on this fixed language challenge, but much of the
input remains unusable. Noncommitted retained states are null; they are not
corrected unknown/negative predictions and cannot be counted as successful
normal findings. Zero retained errors here does not establish a clinical error
rate, calibration, generalization or 100% overall accuracy. The gate and model
both use the same report, so their agreement is not independent clinical evidence.

## Fixed synthetic candidate availability

| Decision | Candidate/finding rows |
|---|---:|
| Scope commit | 17 |
| Abstain | 41 |
| No model assertion | 38 |
| Outside the four-head verifier scope | 240 |
| Total, all original rows retained | 336 |

There are **17/96 (17.7%)** retained assertions within the four supported heads,
occurring in **9/24 candidate slots**. Sixteen are negative and one positive.
These are retained extractor assertions, not independently confirmed clinical
facts. The 22 hashes versus 24 slots reflect exact text deduplication, not missing
cases. No unavailable input is silently treated as an unknown success.

| Supported finding | Commit | Abstain | No assertion |
|---|---:|---:|---:|
| Cardiomegaly | 0 | 17 | 7 |
| Consolidation | 1 | 12 | 11 |
| Pleural effusion | 8 | 8 | 8 |
| Pneumothorax | 8 | 4 | 12 |

Every raw label, contract status, primary score, row ordering and original false/
null clinical-validation field remains unchanged; added fields begin with
`scopegate_`. A commit retains exactly the original non-unknown proposal. It
does not inspect or adjudicate an image, validate an EHR edge, mark an evidence
request clinically resolved, change rankings/winners or authorize regeneration.
The authored conditional match is never transferred to a synthetic candidate
as a confidence score.

## Artifacts and independent replay

Protected run, relative to `artifacts/protected/tricompose_v1_2/`:
`verification_gates/scope_gate_cached_12645021_001/`.

Artifacts: `authored_gate_table.jsonl`, `authored_details.json`,
`candidate_gate_fact_table.jsonl`, `candidate_gate_table.csv`, `summary.json`,
`RESULTS_CN_EN.md`, `manifest.json`. Evidence contains opaque IDs, offsets and
hashes, not quotes. Directories are 2770, files 0660, project-group access only.

Manifest SHA256:
`86a3e37a212d06fb665e7797397dd7736a449f92e30bb19c322923cc3f76e66a`.

Independent audit verified 209 source bindings, six result artifact hashes,
448 authored rows, all 336 preserved original candidate rows, exact replay of
the unchanged gate, designated denominators and selective counts, evidence
field restrictions, CSV/Markdown replay and protected ownership/modes. Historical
full-bank and report-progress manifests are unchanged. Full test suite:
**1,233 passing tests**, including 17 new invented/mock fixtures.

## Consequence for the pipeline

This follow-up establishes a reproducible distinction between a cached raw
proposal and a source-scope-retained assertion. It is useful as an availability/
abstention sidecar, not a new primary consistency score or a clinical judge.
The coverage cost is large; simply reporting perfect conditional matches would
hide it. CXR factuality, EHR compatibility and causal error localization remain
separate unresolved questions. Automatic targeted regeneration stays disabled;
any future selection policy or GPU experiment requires a separate frozen protocol
and the applicable execution approval.
