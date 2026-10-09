# Frozen-selection opacity stability audit

## Question and scope

Keep every historical selection fixed and attach the completed exact-opacity
sidecar as a **post-hoc DEVELOPMENT diagnostic**, not an independent clinical
endpoint or a new selector. The exact and max heads share a checkpoint and the
reports share their cached CheXbert extractor. No clinical superiority, error
localization, repair success, probability calibration or actual GPU saving can
be established from this comparison.

All **80 fixed EHRs / 240 existing CXR slots / 960 report slots** remain. Use
every already frozen budget (4, 8, 12, 20, 30) and seed (0 through 4). Retain
all 3,200 old trials and all 10,000 already frozen uniform-final-choice trials;
13,200 trials are repeated choices over 80 EHRs, not new patients/generations.

The old `random` is **random acquisition plus scored final selection**, not
uniform final choice. Include the separately frozen score-free final control
without drawing new seeds or calling any selector. Its acquisition trace and
charged simulated expenditure must match the old random parent exactly.

## Frozen bindings, before this audit's joined readouts

- Old replay manifest:
  `af82764ac90f9af5b701c577fb2f9bfe6ecf4052310299e8e7cdcb415d2aed5f`.
- Completed full-bank endpoint manifest:
  `ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`.
- Existing uniform-final control manifest:
  `771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124`.
- New completed opacity sidecar manifest:
  `778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94`.

The old choices were frozen before new exact-opacity predictions. This audit
protocol is written after the 64.55%/27.92% overall sidecar readout was seen;
it is not a previously unseen confirmatory test. Do not choose a head, budget,
seed, model, subset, metric weight or next winner from the result.

## Measurements and denominators

Two readouts of the **same selected candidate**:

1. Exact Lung Opacity at unchanged **0.5**.
2. Same-call max(Lung Opacity, Infiltration) at unchanged **0.5**, secondary.

Retain report positive/negative/uncertain/unknown, candidate status/failures,
artifact hashes, case, old choice, method/cap/seeds and old simulated costs.
Only explicit positive/negative on both sides are comparable. Unknown/uncertain
is unavailable, not negative, a zero clinical score, or successful repair.
No global No Finding expansion, pneumonia-to-opacity conversion, EHR enrichment,
threshold fit or new report extraction. Opacity EHR evidence remains unknown
for all 80 anchors; both EHR-related clinical edge scores remain NA.

For each case/method/budget/head, average all seed replicates **within the
EHR first**: 1 for fixed/static/targeted, 5 for old random/scored final, 25 for
uniform final. Then average the same 80 EHR means. Report:

- support, opposition and comparable fractions over all fixed EHRs;
- agreement conditional on comparable = mean support / mean comparable;
- comparison availability, selected-output availability and missingness;
- mean **simulated** historical calls, never measured GPU time;
- selected-image positive and report positive fractions, descriptive only.

Zero comparable means NA conditional agreement, not perfect consistency.
Conditional agreement alone can favor sparse assertions; show coverage next
to it. All cases including unavailable ones stay in unconditional denominators.
This is a single-finding diagnostic, not a 14-label or full-triple score.

Compare every non-fixed method with fixed at each cap/head. Also compare old
scored-final random with uniform-final random: only this selector contrast has
exactly matched acquisition/expenditure. Average per-EHR support/opposition/
coverage/call differences; retain direction counts and common-comparability
denominators. Separately count opposition reduction accompanied by zero new
comparison while baseline had comparison. Call this lost evidence, not repair.
Other same-cap methods can incur different expenditure; disclose the gap.

No p-value/clinical significance or patient-independent sample inflation.
All caps/seeds are reported; no retrospective winner ranking or Pareto claim.

## Execution and provenance

Worker: `TriCompose-v1.2/tools/audit_opacity_selection_stability.py`.
Tests: `TriCompose-v1.2/tests/test_opacity_selection_stability.py`, invented
vectors/choices/hashes only.

Existing CPU Slurm allocation only; no new `sbatch`, model factory, inference,
selector invocation, regeneration, API or download. Read bounded hash-bound
synthetic choice/score metadata only. Do not open EHR/report/prompt text, images,
real source rows or benchmark patient records. Hashing already pinned artifacts
does not load a model. The old trace validator verifies observed-only choices,
fixed EHR and exact cost; uniform controls bind their immutable parent hash.

Write the diagnostic plan and fsync before attaching choices to new readouts.
Atomic non-overwriting new run, protected group modes 2770/0660; keep all old
choices/source code/outputs unchanged. Verify sources before and after work.
No synthetic clinical text or individual source key is put in public docs/logs.
After completion, independently recompute inventory, joining, seed-weighting,
paired counts, artifact/source hashes and permissions.

Any later GPU request needs a separate full-script/resource display/approval.
This audit cannot authorize clinical repair or replace the old score table.
