# V1.3 protocol: independent final evaluation

Status (2026-09-22): protocol and reusable bootstrap utility only. No new final
test, reviewer study, or paper-level superiority result exists. Entry gate:
V1.2 passes controlled localization and budget-matched repair comparisons.

## Freeze before running final test

Record hashes for generator checkpoints, preprocessing, fact extraction,
prompt renderers, selection metrics, thresholds, policy and stopping budgets.
Use patient-disjoint development/calibration/final-test/human-review splits for
real-anchor studies. Keep all candidates of one synthetic case together. Do not
use a synthetic case's final-test scores to choose a model, threshold or rule.

Publish an evaluator-role matrix: selection, threshold calibration, independent
final evaluation, or human review. Reusing the selection score can show that an
optimizer optimized its objective, but cannot independently establish quality.

## Outcomes and valid units

| Scope | Primary reporting rule |
|---|---|
| EHR | Preserve the fixed cohort; descriptive validity and fidelity use a declared common schema and reference population |
| Image quality | Distribution metrics are cohort-level; include sample-size uncertainty and pathology strata, not per-image FID |
| Synthetic reports | Reference-free structure plus independent image-grounded clinical assessment; no invented real reference report |
| Real-anchor reports | Reference overlap/factuality only against the correctly paired protected reference; report image drift separately |
| Three edges | Report each separately with known/comparable/supported/contradictory denominators and coverage |
| Weak clinical context | Separate support/incompatibility analysis; CHF or diuretic use cannot require visible edema |
| Rare/abnormal cases | Give counts and uncertainty; no stable rare-disease claim from one or two positives |
| Cost | Full generation and evaluation calls, GPU time, failures/retries and rejection rate |

The current 960 candidates correspond to 80 EHR cases, not 960 independent
patients. After each method makes one selection per case, compare paired
case-level outcomes. For repeated real-patient studies, bootstrap patient
groups rather than studies. The dependency-free implementation is
`TriCompose-v1.0/eval/report_v1_1/paired_case_bootstrap.py`.

That function preserves missing pairs and coverage. Its interval concerns the
observed comparable subset, not all cases. Report missing/rejected cases
separately and preregister any failure-aware endpoint before seeing outcomes.
Do not quietly omit difficult cases to inflate accepted-sample performance.

## Independent evidence

Use held-out human judgments or genuinely independent frozen evaluators with
documented limitations. A second model can share training data/biases with the
selector; label the degree of independence, not just the model name. A text-only
report scorer is not evidence that the image contains the described finding.

Blind reviewers to generator, policy and selected-vs-baseline status. Preserve
case pairing during statistical analysis and randomize display order. Review
image realism, image-report support, omissions, explicit contradictions and
unsupported specificity; report inter-reviewer agreement and adjudication.

No new evaluator checkpoint is downloaded or external API is called by this
protocol. Real MIMIC artifacts remain inside approved protected computation.
TSTR/TRTS require downstream training and remain out of scope under the current
no-training constraint; they need a separate explicit scope decision.

## Decision report

Show paired changes and confidence intervals for independent clinical quality,
contradiction and coverage, together with quality/diversity versus total compute
curves. Compare equal-budget methods; show exhaustive reranking as a separate
higher-cost reference when appropriate. Never pool the eight EHR-known cases
and all 960 report edges into one purported clinical denominator.

If static selection matches repair at equal/lower cost, report that outcome and
keep the validated static-composition contribution. The goal is a defensible
result, not completing version numbers by assertion.
