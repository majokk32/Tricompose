# Cached V2 report scope gate: development risk/coverage test

This CPU-only follow-up reuses the **unchanged** existing
`report_assertions.gate_assertion()` and `repair_cached_report_evidence.scope_check()`.
It does not add clinical rules, flip/promote an extractor proposal, repair a
report, fit weights/thresholds, change prompts or update primary scores/winners.
The gate was previously designed after inspecting the same authored fixture
challenge. Reapplying it after observing V2 is post-hoc development, not held-out
clinical qualification, preregistration or independent evidence.

## Separate cohorts and boundaries

1. Existing 56 wholly invented texts with 80 designated checks: blind gate
   the frozen Qwen V2 and CheXbert proposals separately, all four heads. Freeze
   448 full-vector gate rows before parsing the existing authored reference key.
   Assess error retention and lost matching assertions as well as conditional
   match/coverage. Do not omit failed/unavailable or noncommitted checks.
2. Fixed 24 synthetic candidate slots / 22 report hashes: all 336 original
   finding rows remain unchanged; append scope availability for Qwen V2 only.
   Ten unsupported findings per candidate remain outside scope, not negative.
   This cohort has no independent clinical references or truth labels.

Exact sealed manifests and artifact hashes bind all inputs. The original span
plan supplies previously verified fully synthetic report paths, not real targets,
EHR rows, images or checkpoints. Source texts are bounded and checked against
their hashes. Only authored/synthetic report text is consumed; no raw MIMIC data
is opened. No new model, GPU, API, download or sbatch call. The worker requires
an actual existing Slurm cgroup and uses standard-library CPU aggregation.

## Output semantics

Keep raw extractor state/contract status and every original candidate fact cell.
`scopegate_decision` is scope_commit, abstain, no_model_assertion,
verifier_unavailable or outside_verifier_scope. A scope commit retains exactly
the original non-unknown proposal. Otherwise `scopegate_retained_state` is null,
not a new negative/unknown clinical assertion. Unknown is never promoted; a
non-veto without complete literal scope coverage is not a pass.

Evidence exports source offsets, hashes and opaque evidence IDs, not quotes.
The gate uses whole-source context and its existing limited literal vocabulary;
synonyms and mixed/historical context may be uncovered. Scope coverage is not
semantic or image truth, and source parser/model agreement is not independent
clinical votes. Every row keeps independent_clinical_validation and
regeneration_authorized false. Nothing is automatically marked clinically
resolved or attached as a new primary scalar score.

## Fixed diagnostic endpoints

For each extractor, all 80 authored designated checks: raw matches, committed
coverage, correct/incorrect commits, noncommitted checks, raw errors not committed,
raw matching proposals not committed, committed polarity flips and determinate
commits on unknown/uncertain references. Retain all family/finding denominators.
Do not report perfect conditional match as overall accuracy or call an abstention
a corrected unknown prediction. On synthetic candidates, report scope-decision
counts and unchanged state/hash lineage only, never clinical accuracy.

## Implementation and outputs

Worker: `TriCompose-v1.2/tools/gate_cached_report_spans.py`.
Tests: `TriCompose-v1.2/tests/test_cached_report_scope_gate.py`; invented fixtures
and mocks only. Old modules/checkpoints/runs are read-only. Output uses a new
opaque atomic run under protected `tricompose_v1_2/verification_gates/`;
directories 2770, files 0660 and project access group only.

Artifacts: `authored_gate_table.jsonl`, `authored_details.json`,
`candidate_gate_fact_table.jsonl`, `candidate_gate_table.csv`, `summary.json`,
`RESULTS_CN_EN.md`, `manifest.json`. Original generations, cases, labels, prompt
versions, numerical scores and winners remain untouched. Any future selection
or GPU experiment needs its own separately frozen policy and execution approval.
