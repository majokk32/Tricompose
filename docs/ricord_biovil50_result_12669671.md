# RICORD + BioViL-T opacity diagnostic / 完成与使用边界

Approved job **12669671** completed 2026-10-05, V100 **d13-04**, exit **0:0**,
allocation elapsed **38 seconds**. Worker elapsed before serialization
**23.220128 seconds**, peak allocated CUDA tensors including model load
**0.608 GiB**; neither is complete lifetime GPU/context accounting.

## Completed without changing the pipeline

- **50/50** fixed bound RICORD PNGs scored; zero failures/substitutions.
- 50 main image encodings plus original first-two image replays: **52 calls**.
  One text batch: six generic polarity probes plus two duplicate inputs.
- Zero training, generation, threshold fitting, template choice, new selector,
  external API, download or old winner replacement.
- Reused original PNGs, with unchanged official BioViL min/max-remap and
  resize512/crop448 preprocessing. No new DICOM decode/window selection.
- Reference states and prior exact-XRV scores join after all encodings; neither
  is an encoder input. Source patient identifiers are not exported.
- All records/denominators/ties stay represented. Unknown/failure is not a
  negative or repair success. The predeclared mean uses all three templates.

The protected metric report contains all three families and the fixed mean,
AUROC/AP, positive/negative reference wins, coverage, actual cost/replays and
same-image exact-XRV disagreement. These are opacity-specific diagnostic
readouts; do not confuse binary phrase preference with full-report factuality.

```text
artifacts/protected/tricompose_v1_2/real_validation/ricord_biovil_pilots/ricord_biovil50_12669671/
  scores.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`d50e5594e43f7979f385f0e14745d66103714b16693e7656ddcf7812e85480ef`.
Approval/script/job details: `docs/ricord_biovil50_submission_12669671.md`.
Frozen protocol: `docs/ricord_biovil50_protocol.md`.

## Interpretation / 如何使用

The fixed-mean opacity probe shows a useful ranking signal on this chosen
reference, while its negative-assertion direction remains imperfect. Template
form also affects that direction. Keep every profile visible; do not elevate
the most favorable template or tune margin zero using these labels.

This is a more directly relevant single-finding check than generic image/report
cosine. It does not establish that BioViL understands all negations, that global
similarity is a clinical contradiction score, or that past generated reports
are correct. The earlier weaker pneumonia task and this opacity task have
different references/domains/findings; neither should be generalized to all
14 labels or used to declare an overall scorer winner.

The same-image classifier comparison tests differing model proposals; shared
image dependence and possible training overlap remain. Agreement is not a new
clinical reference, and disagreement does not identify the faulty modality.
No natural-error localization or repair success was evaluated here.

This reference has 50 selected patients with derived strict unanimity, deliberate
class balance and one image per patient. Original-order/unanimity bias, official
adjudication, clinical display adequacy, checkpoint training overlap and
synthetic-domain transport remain unresolved. The XRV result was already seen;
the new probe is post-hoc DEVELOPMENT, not an untouched confirmatory final test.
No p-values, population prevalence/PPV or clinical superiority are claimed.

Both EHR-related opacity edges still lack direct cached evidence for all 80
synthetic anchors. This benchmark does not fill those fields, qualify cached
CheXbert opacity assertions, or create a validated total triple score.

## Independent engineering audit

Verified exact source/plan/result hashes, 50 distinct opaque record joins,
positive/negative 25/25 reference denominators, 50 original outcomes, four
predeclared reductions, class wins/recovery/missingness, finite cosines,
fixed-mean/XRV direction counts, 52 image calls/one text batch, two original
replay indices, artifact hashes and protected project-group 2770/0660 modes.
Separate pairwise-ranking AUROC and tied-group AP reproduced the saved metrics.
No model/runtime helper was used to manufacture a reference or break ties.

Duplicate-text embedding delta and both image-replay score deltas are zero.
All result artifacts remain Git-ignored. Native stderr is empty. Sealed
worker/tests/protocol/plan and all previous choices remain unchanged. The
pre-submission full invented-fixture regression passed **1,865 tests**.

## Next boundary

The justified next use is a separately predeclared opacity evidence sidecar,
not an automatic clinical repair judge or a silently revised winner table.
Keep exact-XRV, fixed-mean image/text margin, report proposal, EHR availability,
coverage and provenance separate. Model disagreement must remain unresolved
unless additional qualified evidence supplies a reference. Unknown cannot be
rewarded as removed contradiction.

Any candidate GPU scoring/regeneration or changed selection policy needs its
own protocol, complete script/resources and separate explicit approval. Do not
fit template weights/thresholds, replace hard EHRs, narrow the cohort or change
old scores based on this successful component diagnostic.
