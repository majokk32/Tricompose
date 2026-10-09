# Frozen span V2: existing authored-language challenge

Status: prepared development protocol, no new GPU inference authorized or run.
This follows the report-verifier progress sidecar. It tests report assertion
extraction, not image factuality or clinically confirmed error localization.

## Why this test

The prior manual-report diagnostic has no uncertain reference states in its
four heads. Its annotated-state match therefore does not establish uncertainty,
unmentioned or contextual-negation handling. The existing wholly invented
language challenge contains those states and provides an already fixed,
disclosed reference policy. Reuse it, rather than authoring new rules/cases
to improve the observed Qwen results.

Expected states are same-team authors' judgments, not radiologist labels,
independent evidence-span gold or untouched final evaluation. The fixtures and
previous CheXbert outcomes were already inspected during development. This
is a diagnostic comparison, not preregistered/held-out clinical validation.

## Frozen inputs and model

- All 56 original challenge texts, 80 designated finding checks; 224 full-vector
  checks are secondary. No score-based case/family filtering or new text edits.
- Source: protected `benchmarks/authored_assertions_20261002_001/` under
  `artifacts/protected/tricompose_v1_2/`. Manifest SHA256:
  `5f76ca3d7f710f4fdf722f87f27afc6548c839f1c4d77e56a6502ec4e3583757`.
- Four heads: cardiomegaly, consolidation, pleural effusion and pneumothorax.
  Positive, negative, uncertain and unknown remain distinct.
- Unchanged frozen Qwen2.5-VL-7B, V2 prompt, segmentation, decoder, processor,
  and greedy inference; seed 0, 512 output-token cap, no retry. Maximum 56 calls.
  No model/API download, training, thresholds, prompt updates or case changes.
- Reuse the installed model and Python environment used by jobs 12645404 and
  12645961; model assets remain externally read-only.

## Reference policy and separation

Reuse the existing challenge exactly: modal or qualified absence is uncertain;
unmentioned/unassessed is unknown; explicit opposed assertions are uncertain.
Families include literal positives, four absence forms, modal/rule-out language,
qualified absence, no worsening, other-finding negation scope, missing assessment,
generic summaries and mixed/conflicting assertions. These conventions are
disclosed language labels, not claims that an actual image has these findings.

Preparation reads manifests, blinded resolver/invented texts and model file
metadata; it does not parse references or baseline predictions. Every prompt
contains only the original authored text and numbered original spans. There
are no family/expected-state/baseline/clinical-score fields in model messages.
Predictions are closed, fsynced and hash-frozen before reference/baseline reads.
The same-input CheXbert comparison uses its existing frozen predictions, not
new CheXbert calls. Model responses/quotes are not exported; states and source
offsets/hash references are retained below the protected root.

## Endpoints fixed before this execution

Primary diagnostic: four-state accuracy and macro F1 on all 80 designated
checks, with failed availability retained in the denominator. Separate positive/
negative polarity flips and determinate outputs on uncertain/unknown references.
Report all family and finding results; do not select a favorable family.
Secondary: all 224 state checks, contract failures, all-unknown responses,
token-cap failures, actual calls/retries/runtime and peak allocated GPU memory.
Compare on the exact same authored inputs with CheXbert; do not mix this language
test with patient/image clinical endpoints or candidate-confidence calibration.

Any failed request is unavailable, not four successful unknown predictions.
Internal legacy-metric compatibility may temporarily construct placeholder
unknown vectors with failure status; the unchanged evaluator counts them as
unavailable. Persisted failed predictions have null states and null evidence.
No clinical acceptance, score/rank/winner replacement or regeneration follows.

## Implementation and execution

Worker: `TriCompose-v1.2/tools/verify_authored_report_spans.py`, modes `prepare`
and `run`; tests use additional wholly invented metadata/text only. Existing
challenge loaders/evaluator and frozen Qwen loader/inference are reused without
editing them. A fresh atomic run refuses overwrites. Output directories use
2770 and files 0660 with project-group access; caches/temporary/logs stay inside
the workspace, external directories remain read-only.

Proposed script: `TriCompose-v1.2/slurm/43_authored_report_spans_flexible_gpu.sbatch`.
Resource request: gpu partition, A40/A100/L40S-compatible node, one GPU, two CPUs,
32 GiB host RAM, 10-minute wall limit. Worker requires at least 24 GiB GPU memory.
The time limit is an upper bound, not a queue-start prediction. Before sbatch,
the complete script and resources must be shown and separately approved.

Expected outputs: `predictions.json`, `details.json`, `summary.json`,
`RESULTS_CN_EN.md`, and hash-bound `manifest.json`. No primary metric or clinical
repair qualification is conferred by a high language-challenge result.
