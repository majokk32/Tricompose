# Frozen evaluator scorecard / 当前评分器结论

Completed in existing CPU Slurm allocation **12645021** using only cached
public-reference prediction metadata. No new model, GPU, download, API,
patient-source read, image/weight read, threshold fitting, training or selection
change. The [analysis protocol](frozen_evaluator_scorecard_protocol.md) was fixed
before this analysis, but the source results were already inspected. This is
explicitly **post-hoc DEVELOPMENT**, not retrospective preregistration or a new
independent validation.

## Same-image comparison

All 50 original opaque RSUA images remain: 25 published pneumonia and 25
paper-described normal proxies. The reference is a cohort class, not
independently adjudicated per-finding disease truth. Patient grouping, age
domain and training overlap are unverified. The dataset's segmentation-mask
validation must not become a disease-adjudication claim.

| Frozen readout | AUROC | Descriptive image-level 95% interval | AP | Positive-reference wins | Negative-reference wins | Balanced wins |
|---|---:|---|---:|---:|---:|---:|
| XRV default 0.5 | 0.7248 | [0.5632, 0.8640] | 0.6989 | 12/25 | 19/25 | 0.62 |
| XRV unchanged transported threshold | 0.7248 | Same underlying ranking | 0.6989 | 6/25 | 23/25 | 0.58 |
| BioViL-T fixed three-template mean | 0.5520 | [0.3856, 0.7152] | 0.5844 | 21/25 | 8/25 | 0.58 |
| XraySigLIP fixed three-template mean | 0.7552 | [0.6064, 0.8816] | 0.7167 | 8/25 | 21/25 | 0.58 |

All three original VLM template profiles are also retained in `scorecard.csv`,
for ten rows in total. No best-template choice or changed operating point.
XRV uses a proxy-score threshold; VLM wins use the sign of an authored-text
margin. These have different semantics and are not calibrated probabilities.

Exactly 2,000 class-stratified paired bootstrap resamples, seed 0, use shared
image indices across the three underlying ranking readouts. The sample's
deliberate 25/25 balance is preserved; AP does not estimate natural-population
precision. Intervals are descriptive and do not address unknown patient
clustering or cohort/training overlap. No p-values or clinical significance.

| Fixed contrast | AUROC difference | Descriptive 95% interval | AP difference | Descriptive 95% interval |
|---|---:|---|---:|---|
| SigLIP minus XRV | +0.0304 | [−0.1105, +0.1648] | +0.0178 | [−0.1436, +0.1624] |
| BioViL-T minus XRV | −0.1728 | [−0.3376, −0.0048] | −0.1145 | [−0.2800, +0.0392] |
| SigLIP minus BioViL-T | +0.2032 | [+0.0032, +0.3968] | +0.1323 | [−0.0493, +0.3105] |

**中文结论：SigLIP 的点估计略高于 XRV，但二者配对差值区间包含零，不能
据此宣布 SigLIP 更好。BioViL-T 的这个“单一肺炎正负文本”任务较弱，也不能
外推为全报告图文检索无用。排序能力、否定理解与临床正确性必须分开。**

BioViL-T has stable template direction on 36/50, agreeing with the class proxy
on 19/36 of those stable cases. SigLIP is stable on only 2/50 (4%), with 2/2
conditional proxy wins. Neither conditional fraction becomes full-cohort
clinical accuracy. Neither stability nor model consensus qualifies a repair
decision. SigLIP shares CheXagent-2's visual encoder.

## What the pipeline can use now

| Evidence component | Legitimate present use | Not established |
|---|---|---|
| Mechanical image/report validity | Detect a bounded basic-invalidity condition | Anatomy correctness or clinical image quality |
| Direct fixed-EHR facts and provenance | Preserve actual positive/negative constraints and missingness | CHF/medication/lab context as a required image finding |
| XRV and report-label proposals | Transparent raw edge support/opposition and coverage | Independent image truth or qualified fault attribution |
| BioViL-T | Secondary retrieval/compatibility readout | Global cosine as a finding-level negation verdict |
| XraySigLIP | Alternate template-preference diagnostic | Margin sign as clinical presence/absence |
| Template/scorer/dependency uncertainty | Mark unresolved evidence, preserve denominators | Independent majority-vote truth |
| Actual execution ledger | Count calls, failures, retries and measured runtime | Simulated replay calls as actual GPU savings |

Keep this evidence as a vector; do not add incompatible scales into a new
weighted clinical total. An EHR edge with no comparable direct fact remains
unavailable, not zero or perfect consistency. Historical lexicographic scoring
and selected triples remain unchanged. `score_usage_contract.json` documents
these limits; it is **not an installed routing or regeneration policy**.

The next method experiment should freeze an explicitly exploratory,
dependency/uncertainty-aware policy separately and compare fixed, score-free
random, static reranking and bounded targeted actions at equal cost. Preserve
the fixed EHR and report every unresolved case. Do not present this already
inspected bank as confirmatory testing, or claim clinical repair success from
these proxy readouts. Further model execution still requires the full Slurm
script/resources and explicit approval.

## Protected output and verification

```text
artifacts/protected/tricompose_v1_2/real_validation/evaluator_scorecards/scorecard50_12645021_001/
  scorecard.csv
  bootstrap_comparison.json
  score_usage_contract.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`91a2899919a14cb03e6f43ac0bc58f37bb0290e553628831b80da689ec2833f1`.
Analysis/construction before commit took **0.297866 CPU seconds**; this is not
model runtime. The output is aggregate-only and private (2770/0660, project
NFS GID 65534). No old source artifact, score table or winner was overwritten.

The post-run audit checked **16 bounded source hashes, five artifact hashes,
ten CSV rows, exact template/usage/report replay and project permissions**.
A separate numeric implementation using pairwise Mann–Whitney ranking and
tied-group AP reproduced **12 model/contrast interval checks** with all 2,000
paired samples. It did not reopen pixels, weights or MIMIC patient inputs.
These are arithmetic checks, not independent clinical adjudication.

**26 new invented-fixture tests passed; the full V1.2 suite passed 1,544 tests
in 8.951 seconds.** Sealed worker/tests/protocol and old run dependencies stay
immutable. Only this aggregate note and the V1.2 README are updated afterward.
