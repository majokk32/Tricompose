# Frozen evaluator scorecard: cached RSUA comparison

This is a **post-hoc developmental analysis of already observed results**, not
a new clinical validation, an untouched test or a scoring-policy update. It
uses the completed XRV (12636566), BioViL-T (12637081) and XraySigLIP
(12654662) runs on the same fixed 50 public RSUA reference images. No image,
model weight, EHR, report body or MIMIC reference is opened. No model, API,
training, threshold fitting, template selection or new Slurm submission occurs.
Execution requires an existing CPU Slurm allocation and private atomic output.

## Fixed inputs and alignment

Pin the three result manifests and the original cohort metadata by SHA256.
Read only their bounded aggregate summaries and cached prediction metadata.
Require the same 50 unique opaque case IDs, 25 published pneumonia and 25
paper-described normal-proxy references, matching original BMP hashes across
all scorers, and matching lossless PNG hashes for both vision-language models.
Do not follow image paths, recursively open manifest sources, copy report text
or reuse raw MIMIC references. Refuse missing/failed predictions, a changed
source or an unknown/uncertain reference; never substitute another image.

Reproduce every existing template's AUROC/AP and polarity counts, both unchanged
XRV operating points, and the fixed mean of all three VLM templates. Retain all
templates without selecting the best. Ties are not negative findings. A
zero-margin text preference is not a calibrated clinical threshold. Record
template-stable coverage and conditional proxy wins with the full denominator.

## Descriptive paired uncertainty

Use exactly **2,000 stratified paired bootstrap resamples, seed 0**, resampling
25 positive-proxy and 25 negative-proxy image indices with replacement. Use the
same resample indices for XRV, the fixed BioViL-T mean and the fixed XraySigLIP
mean. Compute the existing tie-aware AUROC and average precision, percentile
intervals at 2.5%/97.5% (linear interpolation), and all three fixed pairwise
differences. Do not bootstrap the two XRV thresholds as independent scorers.

This conditions on the deliberately balanced developmental sample. Its unit is
an **image**, not an independently verified patient. Patient grouping, age
domain and checkpoint-training overlap are unknown. Intervals do not account
for those limitations or establish clinical significance. AP describes the
balanced sample, not natural prevalence. No p-values, optimal threshold,
best-template claim or prospective superiority claim is produced.

## Fit-to-purpose interpretation

Publish a score vector rather than a manufactured universal weighted score:
artifact validity; explicit fixed-EHR evidence and coverage; raw per-edge proxy
support/opposition; alternate retrieval/text margins; uncertainty/dependency;
actual cost. Existing lexicographic selection remains unchanged. All models'
probability-calibration and independent clinical-truth qualifications remain
unavailable. Correlated same-image reports are not independent votes; SigLIP
shares CheXagent-2's vision encoder.

The generated usage contract records diagnostic/ranking scope and disallowed
interpretations. It is **documentation, not an executable deployment policy**:
it does not alter weights, historical winners, action permissions or repair
authorization. Ranking discrimination alone never clears a modality-repair
gate. A mechanical blank-image guard detects basic invalidity, not anatomy or
clinical truth. No-comparable-EHR edges remain unavailable rather than zero.

## Outputs and verification

Entry point: `TriCompose-v1.2/tools/summarize_frozen_evaluators.py`.
Fixture tests: `TriCompose-v1.2/tests/test_frozen_evaluator_scorecard.py`.

Fresh protected `real_validation/evaluator_scorecards/<opaque_run_id>/`:

```text
scorecard.csv
bootstrap_comparison.json
score_usage_contract.json
summary.json
RESULTS_CN_EN.md
manifest.json
```

Seal the worker, tests, this protocol, exact imported metric/atomic helpers and
consumed metadata hashes. Recheck source hashes before atomic commit, refuse
overwrite, and keep directories 2770/files 0660 in the CARC project group.
The output contains aggregates only: no individual reference labels, prediction
vectors, images, report text or patient IDs. CPU time is not model GPU time.
Invented-fixture tests and exact arithmetic/source audits are engineering
checks, not clinical validation. Older code, manifests, score tables and
selected triples must stay immutable.
