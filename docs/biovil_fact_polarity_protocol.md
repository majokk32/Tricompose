# BioViL-T single-fact polarity diagnostic

Status: implemented and fixture-tested; **approved, completed and verified as job 12597637**.
The complete script and resource request were shown before separate explicit
approval. Initial scheduler status was `RUNNING` on `e23-02`; it subsequently
completed with exit `0:0` in 49 seconds. Program runtime was 45.827 seconds,
peak allocated VRAM including model load 0.576 GiB. Metrics recomputed exactly
from protected derived scores/cache; no raw source input was reopened in review.
This protocol was frozen before observing the new polarity scores. It extends the completed
real matched/random and shared-finding hard-negative retrieval diagnostics;
those existing experiments are not rerun as baselines.

## Question

Does the unchanged frozen image-text scorer distinguish presence from absence
of the **same** finding on a fixed CXR? Strong whole-report retrieval need not
imply sensitivity to negation. A scorer that always prefers the positive text
may have high positive-case agreement or even margin AUROC, while failing to
rank an explicit negative statement correctly.

This is a necessary diagnostic for a reference-free consistency framework,
not proof of clinical truth or a new generation/repair method.

## Frozen inputs and templates

1. Reuse the existing 128 validation + 128 test cohort from
   `real_biovil_valtest128_20260930_001/scores.json`; one frontal study per
   patient, original opaque IDs, membership, source hashes and split assignment.
   Do not select cases by finding, difficulty, score or image quality.
2. Reuse report-derived four-state references from
   `real_xrv_weak_valtest128_20261001_001/scores.json`. XRV outputs are not used
   as reference labels; only the earlier cached source-report label states.
   These are **not independent image annotations**. No new source report/EHR
   text is read. Linkage keys and image paths stay transient inside approved
   Slurm; no source ID, path or per-row reference label is exported.
3. Eight fixed findings: atelectasis, cardiomegaly, consolidation, edema,
   lung opacity, pleural effusion, pneumonia, pneumothorax. This cached source
   has Lung Opacity; it is not the manual Airspace Opacity source from the
   separate CheXbert diagnostic. Do not merge the two inventories.
4. Three fixed positive/negative template pairs per finding:

   ```text
   The chest X-ray shows {finding}.
   The chest X-ray shows no {finding}.

   There is evidence of {finding}.
   There is no evidence of {finding}.

   {Finding} is present.
   {Finding} is absent.
   ```

   These are authored generic probes, not patient-specific reports, a
   pseudo-report-to-CXR generator, or excerpts from MIMIC. They add no view,
   laterality, severity, history or device details. Their polarity differs;
   their concept remains fixed. No language model creates or chooses prompts.
5. Encode all 48 prompts, plus two identical generic positive-control prompts,
   without reference labels. Score every image against all 48 probes. Labels
   are consulted only afterwards by the summary computation.
6. Use the same pinned BioViL-T checkpoints and local inference adapter,
   tokenizer/model metadata and vendored runtime. Resize 512, center crop 448,
   L2-normalized 128-dimensional global embeddings; parameters eval/frozen.
   Use `torch.inference_mode()` and offline assets; no training/download/API.

## Predeclared metrics

For each image/finding/template:

```text
margin = cosine(image, positive_text) - cosine(image, negative_text)
```

- A positive weak reference ranks correctly iff `margin > 0`; a negative weak
  reference iff `margin < 0`. Exact ties are not wins and are reported.
- Report positive-reference and negative-reference win rates separately,
  their equal-weight balanced win rate, full/reference-known denominators,
  and excluded unknown/uncertain counts. Unknown never becomes negative.
- Report margin AUROC/AUPRC only when each binary reference class has at
  least ten examples. AUROC applies to a signed ranking score, not calibrated
  probability. Sparse-class metrics remain unavailable; raw descriptive
  win rates retain their support counts.
- Report all three template families separately and the predeclared mean of
  their margins. **Do not select the best template after seeing test results.**
- The zero point is the within-image ordering of two statements, not a fitted
  clinical threshold. Do not fit thresholds, weights, a router or a new scorer.
- Record positive/negative text-embedding cosine to diagnose lexical proximity,
  duplicate-prompt embedding difference, and score differences when replaying
  the first four image inputs in source order. Replays do not replace scores.
- No patient bootstrap, clinical localization accuracy, regeneration success,
  independent image factuality or calibrated probability is claimed.

## Interpretation and integration

Both source splits have already been used in previous diagnostic pilots;
neither is an untouched final clinical benchmark. References are report-
derived, and possible MIMIC training overlap remains unresolved. A high probe
score cannot clear the clinical-adjudication gate. A low balanced polarity win
rate would diagnose a limitation of this image-text score, not automatically
prove that a generator or report is wrong. Do not mutate synthetic winners or
rules to improve results on this source.

The permitted next integration is a secondary evidence diagnostic alongside
XRV/CheXbert and scope/coverage. Before targeted repair, obtain independent
image/report adjudication and validate localization on controlled interventions.

## Proposed execution

Program: `TriCompose-v1.2/real_validation/biovil_fact_polarity.py`.
Script: `TriCompose-v1.2/slurm/27_real_biovil_fact_polarity_debug_p100.sbatch`.
Request: debug, one P100, two CPUs, 16 GiB host RAM, ten-minute cap,
text batch 16, replay four, minimum ten per class for AUROC/AUPRC.
This is a resource ceiling, not a runtime prediction or reservation.

Before every submission show the **complete script and resource request**, then
obtain explicit approval. Earlier jobs or a general "go" before a new script
is shown are not new-job approval. No GPU model was loaded during preparation.
The approved script SHA256 is
`f025c1fae6c7e8e383302aa7072fb0f6f079b16312cac3c6caf633f17e8b3543`;
the approved program SHA256 is
`e0a3ff342866f076c40d0687cf79962ac1f87cb6afc516baecdb65f4fe0be379`.

After approval only, submit:

```bash
ALLOW_REAL_IMAGE_POLARITY=1 sbatch TriCompose-v1.2/slurm/27_real_biovil_fact_polarity_debug_p100.sbatch
```

The submitted job's non-overwriting output will be
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_fact_polarity_12597637/`
with `summary.json/md`, `scores.json`, `prompts.json`, `manifest.json`.
Scores contain only opaque IDs, split, image hashes and model cosines, not
reference rows or raw source contents. Job-local caches/logs stay protected;
directories 2770/files 0660/project group. Public logs contain sanitized
status, elapsed runtime, allocated VRAM and manifest hash only.

## Completion receipt, not a protocol change

All original case membership, templates, minimum class support and metrics
above were retained. The completed original manifest SHA256 is
`5aeedee4f0aaaf8e5afcd534cc7cdf6ef44b0b4f9f0d544cedb2d5682c5e935c`.
Original artifacts, fixed cohort/reference bindings, derived metric replay,
declared-order Markdown, project modes/group and unchanged old score/winner
hashes passed verification. Numerical image/prompt replay is retained in the
protected result; it does not replace primary predictions.

The readable protected bilingual report is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_polarity_review_12597637_001/RESULTS_CN_EN.md`,
with an execution receipt and artifact manifest. Results are descriptive weak-
reference evidence with finding/template-dependent polarity performance, not
independent clinical adjudication. No threshold/template choice, scorer weight,
score table, winner or regeneration authorization was changed.
