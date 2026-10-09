# Opacity assertion stages V2: authored-only paired development diagnostic

Prepared scope. No new GPU job, source candidate report inspection, training,
download, rule fitting, reranking, score replacement or repair is authorized by
preparation. The complete batch script/resources need subsequent explicit
approval. Consumed V1 worker, protocol, plan and result remain immutable.

## Motivation and interpretation

V1 job 12675347 had 13/16 quote-contract completions but only 3/6 known authored
control matches. Possible -> positive, unmentioned -> negative and qualified
absence -> negative require a semantic diagnostic, not CXR/EHR replacement.
Quote existence and deterministic replay never established linguistic accuracy.

V2 separates locating statements from classifying their assertion polarity.
Mechanical source segment IDs remove free quote invention and disambiguate
repeated exact sentences. They cannot verify relevance, negation interpretation,
omitted mentions or whether the report describes an image accurately.
This changes both interface and prompt, not only architecture; a paired result
does not isolate which change caused a numerical difference.

## Fixed authored suite and annotation policy

Exactly 48 distinct wholly invented texts, 12 families of four, one opacity
check per text. Reuse all six known V1 controls verbatim with explicit legacy
markers, plus 42 authored extensions. No protected source reports, patient EHR,
CXR, reference targets or generated candidate bank is read by this benchmark.
Code fixtures: `TriCompose-v1.2/benchmarks/opacity_assertion_controls_v2.py`.

Families: explicit presence, explicit absence, possible finding, unmentioned
finding, qualified absence, opposing assertions, other-finding negation scope,
other-disease-only, generic summary, change language, multisegment context and
repeated assertion. These are **investigator-known development fixtures**, not
held-out language tests, independent clinical annotations or disease prevalence.

State policy: explicit presence/absence -> positive/negative; possible or
qualified absence/change-only language -> uncertain; unrelated disease or
unmentioned/generic statement -> unknown. Explicit unchanged presence stays
positive. Opposed positives/negatives and any uncertain assertion reduce to
uncertain. Other-disease-only does not weakly imply opacity. These conventions
are disclosed task labels, not universal radiological ontology or image truth.
Relevant mechanical segment IDs are also authored expectations, not independent
gold spans. Preserve every text and every unavailable prediction.

## Model separation, interface and frozen comparison

Use the exact existing local frozen Qwen2.5-VL-7B checkpoint/runtime/loader.
Weight SHA256 `26fa644c8e61aa8185da5055b61aa28d6897fee102c2f43173acab657c4257a1`.
No model modification, dtype-policy switch, quantization, training or API.
The loader's unchanged BF16 support check includes emulation in Torch 2.9.1;
record actual dtype/device, not an assumed V100 fp16 fallback.

Model inputs contain only source text, fixed instructions, mechanical segment
IDs and, for stage B, stage A selected IDs. Never send expected states/spans,
families, legacy-control metadata, case IDs, clinical scores, old predictions,
EHR or images. Project-group access is not process-level investigator blinding.
Store answer keys separately and read their semantics only after predictions,
raw responses and replays are fsynced and hash-frozen.

- Baseline: unchanged V1 quote prompt/parser, once on each same authored text.
- Stage A: locate explicitly relevant sentence/line segment IDs; include
  negative/possible/qualified/change/opposed mentions, skip unrelated statements.
  Response exactly `{"segment_ids": [...]}`; unique bounded ascending integers.
- Stage B: classify every selected segment in its original whole-report context.
  Response exactly `{"assertions": [{"segment_id": ..., "state": ...}]}`;
  exhaustive selected IDs, no added/omitted/duplicate IDs or invented quotes.
- Empty completed stage A -> completed unknown, stage B skipped. An incorrect
  empty selection is an omission in the authored span/state diagnostic, not
  proof of a normal image. Any stage failure -> unavailable with null state,
  never a successful unknown. Preserve stage availability and attempts.
- Character limit 8192, mechanical segment cap 64; refuse rather than truncate.
  No clinical regex classifier, silent rescue, relabeling, favorable retries,
  sentence filtering, case replacement or parameter selection.

Greedy seed 0. Locator cap 128 output tokens; polarity/baseline caps 512 each.
Repeat the staged path for original items 0/1 once, without replacing primary
outputs. **Maximum 148 attempted calls**: 48 baseline, 48 primary locators,
up to 48 primary polarities, up to four calls for two staged replays. Actual
counts are lower if selections are empty or stages fail; never invent attempts.
This benchmark is not a claimed computation-saving repair method.

## Predeclared metrics and narrow gate

Paired four-state accuracy over all 48 attempts, failure-aware per-class/macro
F1, full confusion with unavailable column, positive/negative flips and
determinate outputs on uncertain/unknown. Report all 12 families, all six known
controls, exact authored segment-ID match, failures, replay changes, attempted
phase counts, runtime and peak allocated CUDA memory. Missing predictions remain
in the denominator; repeated statements are not removed for easier evaluation.

The narrow **language gate** passes only with 48/48 staged state matches,
48/48 authored selected-ID matches and both complete replay states/IDs stable.
Even a pass does not qualify clinical truth, primary scoring, error localization
or regeneration. A failure remains a failure, not grounds to revise the same
sealed suite/prompt. Any next candidate check needs a new plan and approval.

## Artifacts and implementation

Interface: `TriCompose-v1.2/interfaces/opacity_assertion_stages_v2.py`.
Worker: `TriCompose-v1.2/tools/benchmark_opacity_assertion_stages_v2.py`.
Tests: `TriCompose-v1.2/tests/test_opacity_assertion_stages_v2.py`.
Fixtures/protocol/code/runtime and all model assets are hash-bound at preparation.

Atomic protected plan: inputs.json, references.json, plan.json, manifest.json.
Atomic new results: predictions.json, raw_responses.json, replay_checks.json,
prediction_freeze_receipt.json, scored_checks.json, summary.json, RESULTS_CN_EN.md,
manifest.json. Group-only directories 2770/files 0660; caches/temporary/logs
workspace-local, external directories read-only. No overwriting of old runs.
Raw responses/source text never enter public logs/chat/Git; public runtime
status contains sanitized status, elapsed time, tensor memory and hashes only.

Post-run independent decoding/cost/reference-join/matrix/hash/mode checks are
required. CPU oracle fixtures test software contracts, not frozen-model quality.
