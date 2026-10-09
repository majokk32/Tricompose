# RSUA same-cohort BioViL-T follow-up

Status: **approved, completed and verified as job 12637081**. All existing XRV
results, thresholds, candidate banks and selected triples remain unchanged.
Every further GPU execution needs the entire script/resources displayed and
a new explicit approval.

## Question and claim boundary

The [completed XRV diagnostic](rsua_pneumonia_pilot.md) yielded pneumonia AUROC
0.7248 on the fixed 25+25 public RSUA cohort, but sensitivity was 0.48 at the
default 0.5 and 0.24 at the unchanged transported threshold. It does not justify
using XRV-negative as a stand-alone clinical absence or regeneration decision.

This follow-up asks whether the existing frozen BioViL-T provides a distinct
image-text signal on **exactly the same 50 images**, and where its polarity
ordering differs from XRV. Different architecture does not mean statistically
independent errors or independent clinical adjudication. RSUA's Non-Covid
comparison group remains the paper-described normal-cohort proxy. Unknown age
domain, patient grouping and training-image overlap remain unresolved. No
clinical reference, disease prevalence estimate or qualified repair gate is
created by agreeing model outputs.

## Frozen protocol before new scores

Program: `TriCompose-v1.2/real_validation/rsua_biovil.py`.

1. Bind the completed XRV run to manifest SHA256
   `92a64874818f37e0d363e8406064d4619cb9edefa01ccab43881cf1a1df7d0e0`.
   Keep all original opaque case IDs, image hashes and fixed cohort SHA256
   `ed33c7708ac737aa8e270da2219457def542186f02bcc085c7188c7a135779c1`.
   Do not substitute, reselect, duplicate or drop a difficult case.
2. Reuse the existing pneumonia-only slice of the original frozen polarity
   catalog, without modifying that original worker or any previous result:

   ```text
   The chest X-ray shows pneumonia.
   The chest X-ray shows no pneumonia.

   There is evidence of pneumonia.
   There is no evidence of pneumonia.

   Pneumonia is present.
   Pneumonia is absent.
   ```

   These are generic authored texts, not real reports, extracted patient facts
   or new generation prompts. They add no severity, view, device or history.
3. Pin the existing BioViL image/text checkpoints, tokenizer/configuration,
   vendored runtime and helper source hashes. Reuse eval mode, disabled
   parameter gradients, inference mode, L2-normalized 128-dimensional global
   embeddings, official resize 512 and center crop 448. No download, API,
   training, fine-tuning or new classifier.
4. The official BioViL loader does not support BMP. Only inside approved GPU
   Slurm, convert each protected BMP to an opaque grayscale PNG, verify exact
   256x256 grayscale-byte equality after reopening it, then use the unchanged
   official image loader/normalization. Constant images or pixel changes cause
   a fail-closed run, never a case replacement. Original BMPs stay read-only.
   This PNG preserves the grayscale input used by the historical XRV adapter;
   BioViL's own min/max remap is still part of its official preprocessing.
5. Encode six authored texts plus two identical duplicate controls in one
   batch. Encode all 50 images once and replay the first two in fixed source
   order: 52 image-encoder calls, one eight-input text batch. Replay does not
   replace primary scores. Source labels are consulted only after encodings,
   never sent to the model as text, features or labels.

## Metrics and non-selection rules

For each fixed image/template:

```text
margin = cosine(image, positive_text) - cosine(image, negative_text)
```

Report each of the three template families and their **predeclared arithmetic
mean**, without selecting the best template after seeing the results. Margin
AUROC/AP measure published-cohort discrimination; they do not establish
negation understanding. Separately report positive-reference wins (`margin>0`),
negative-reference wins (`margin<0`), their balanced mean, exact ties, class
denominators and mean margin. A tie is unknown, neither a negative nor a win.
The zero margin is a pairwise text-ordering boundary, not a fitted clinical
threshold. High AUROC with poor negative wins remains an important failure
mode, not a reason to redefine the metric.

Compare the fixed mean-margin polarity with both original XRV operating
profiles, 0.5 and 0.55457607. Keep counts for both-positive, both-negative,
XRV-positive/BioViL-negative, XRV-negative/BioViL-positive and BioViL tie,
including separate published-reference-class counts. This is **scorer
disagreement**, not evidence that one model or a generated modality is wrong.
Do not turn agreement into majority-vote clinical truth or change thresholds,
weights, winner tables or regeneration acceptance. No Brier/ECE on these
non-probability scores. `primary_metric_eligible` and `regeneration_authorized`
both remain false.

## Prepared plan and output

```text
artifacts/protected/tricompose_v1_2/real_validation/rsua_biovil_plans/plan_12636566_001/
  plan.json
  manifest.json
```

- Plan-manifest SHA256: `043e8f8ad708fa84fe2529a48ad0a0c2139e211e9814199fffff673ccb922e5b`.
- Plan SHA256: `ee411221af829d7843b825c2638553901cb732ec7480a56c7870bc60df1b4017`.
- Worker SHA256: `91556a13e97c1c64253b3ca6c15927ebfbba1c159966e994dfe289ec0fe08c0c`.
- Image checkpoint SHA256: `b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77`.
- Text checkpoint SHA256: `6d86a8d760eaa09c9a55d57cc6f6bb01b0cbccb8b827fc775a79f37a8fbda76c`.

The metadata-only plan passed independent hash/permission checks, with zero
model calls or decoded pixels. Full V1.2 suite: 1,027 invented-fixture tests
passed in 13.056 seconds, including 16 new score/guard/mock-conversion tests.
These are interface tests, not clinical validation.

Prepared script: `TriCompose-v1.2/slurm/38_rsua_biovil50_gpu.sbatch`.
Request: account ruishanl_1185, gpu partition, one flexible GPU, two CPUs,
8G host RAM and five-minute execution cap, with offline assets and all
cache/temporary/log output redirected into new protected workspace paths.
The cap does not include queue wait; no immediate-start guarantee is made.

After complete script/resource display and explicit approval, the script was
submitted unchanged. Results were atomically committed to:

```text
artifacts/protected/tricompose_v1_2/real_validation/rsua_biovil_pilots/rsua_biovil50_12637081/
  summary.json
  scores.json
  prompts.json
  score_table.csv
  manifest.json
  inputs/case_####.png
```

Protected scores contain opaque IDs, image hashes and cosines, not copied
source EHR/report text, source patient keys or disease-reference rows. All
directories/files retain project-private modes 2770/0660. Public Slurm output
contains fixed status only. The original XRV job and its report remain sealed.
Every further submission still requires a separately shown script and approval.

## Completion: weak polarity discrimination, not repair authorization

The P100 job on e21-03 waited approximately 23 seconds and completed in 15
seconds, exit `0:0`. Worker runtime was 8.8523s and Torch peak allocated VRAM
including load was 0.606 GiB. All 50 conversions preserved grayscale pixels;
the two image replays and duplicate-text embeddings had maximum differences
of zero. Artifact hashes, protected modes and source/plan binding passed an
independent metadata check. No historical metric or selection was changed.

| Frozen reduction | Margin AUROC | AP | Positive wins | Negative wins | Balanced wins |
|---|---:|---:|---:|---:|---:|
| shows / shows no | 0.5696 | 0.59075810 | 23/25 | 4/25 | 0.54 |
| evidence / no evidence | 0.5312 | 0.57047564 | 21/25 | 6/25 | 0.54 |
| present / absent | 0.5440 | 0.60688239 | 16/25 | 10/25 | 0.52 |
| Predeclared mean of all three | 0.5520 | 0.58437032 | 21/25 | 8/25 | 0.58 |

There were no ties. The main mean-margin readout shows weak discrimination
and a tendency to prefer presence on this specific published-cohort proxy
task. This is descriptive, not a significance test or a claim that BioViL-T
fails at all retrieval tasks. Do not choose a different template after scoring.

Mean-margin BioViL disagreed with default XRV on 22/50 images and with unchanged
transported-threshold XRV on 32/50. The default-profile 21 XRV-negative/
BioViL-positive disagreements included ten pneumonia and eleven control cases;
that pattern cannot identify which evaluator is correct. Default-profile
agreement also included six jointly positive control cases and three jointly
negative pneumonia cases. Agreement is not clinical truth. Do not use this
follow-up to make BioViL override XRV or activate regeneration.

Protected bilingual report and aggregate comparison CSV:
`artifacts/protected/tricompose_v1_2/real_validation/rsua_pilot_reviews/review_biovil_12637081_001/`.
Result-manifest SHA256:
`a8eed9db9e20146cb1bc7ca597aed78f8bfb5cd1683b29e82dde99497b08992a`.
The next safe integration is scoped reliability/disagreement metadata, not new
clinical labels or retrospective threshold/weight changes. Both clinical
primary-metric eligibility and regeneration authorization stay false.
