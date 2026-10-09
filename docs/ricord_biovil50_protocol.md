# Frozen RICORD opacity polarity diagnostic

PREPARATION ONLY. No new inference or `sbatch` is authorized by this file.
Show the complete script/resource request and obtain separate explicit approval.

## Question

Can the existing frozen BioViL-T image/text scorer distinguish an opacity
assertion from its negation on the same fixed reference images? Ranking
(AUROC/AP) and assertion direction are separate measurements. Good retrieval
or AUROC does not imply that margin sign qualifies a contradiction decision.

Reuse exactly the completed RICORD XRV 50-image run, manifest
`dc14a2ba1aed3823f0752a73dda297e874f638b31644d2992227eb3f1cf697a9`.
All 50 patients, 25 opacity-positive and 25 opacity-negative under the frozen
derived three-group unanimity rule, remain. This is not official adjudication,
a pneumonia reference, natural-prevalence sampling or an untouched final set.
The earlier XRV results were already seen; disclose post-hoc DEVELOPMENT.
Training overlap, display clinical adequacy and synthetic-domain transport
remain unverified.

## Immutable inputs and model

Read the existing bound opaque `displays/case_000.png` through `case_049.png`
inside a separately approved GPU job only. No DICOM decode, PNG modification,
alternate window, source patient key, EHR, real report or new download.
Prepare reads manifests/aggregate receipts and hashes, never pixels or reference
rows. All inputs/external environments/checkpoints are read-only.

Reuse the sealed BioViL loader/model/vendor asset bindings from the completed
RSUA helper. Freeze/eval both encoders and use `torch.inference_mode()`.
Image checkpoint SHA256:
`b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77`.
Text checkpoint SHA256:
`6d86a8d760eaa09c9a55d57cc6f6bb01b0cbccb8b827fc775a79f37a8fbda76c`.

The official BioViL PNG loader internally min/max-remaps to uint8, then resizes
to 512 and center crops 448; retain that behavior and disclose it. Source PNG
bytes are unchanged, but preprocessing is NOT identical to XRV. Do not call
an architecture-only comparison or optimize display values for a higher score.

## Predeclared prompts and reduction

Reuse all three existing `Lung Opacity` pairs in `biovil_fact_polarity.probes()`:

1. `The chest X-ray shows lung opacity.` / `The chest X-ray shows no lung opacity.`
2. `There is evidence of lung opacity.` / `There is no evidence of lung opacity.`
3. `Lung opacity is present.` / `Lung opacity is absent.`

Do not pick a template using this reference or substitute diseases/severity/view.
L2-normalized 128-dimensional embeddings produce positive/negative cosines.
Each margin is cosine(positive) minus cosine(negative); primary diagnostic
reduction is the arithmetic mean of ALL THREE margins. Margin > 0 means
positive preference, < 0 negative preference, exact zero remains unknown.
This is not a calibrated probability, fitted threshold or clinical diagnosis.

One text batch with six prompts plus two duplicate copies of the first prompt.
One image encoding per fixed case; replay original indices 0 and 1 once each,
not the easiest successful images. Full completion expects 52 image-encoder
calls, one eight-input text batch, zero retries, training or generation.
Only opaque image paths and generic prompts enter the encoder. Reference labels
and old XRV predictions join AFTER all image/text encodings.

## Metrics, missingness and scope

Retain all outcomes with sanitized error types and their original denominator.
Report all three profiles plus the fixed mean: AUROC, grouped-tie AP, positive
and negative reference wins separately, balanced win rate, ties and mean margin.
Failed/zero-margin items are not negative predictions or successful repair.
Report scored/failed and per-class coverage; conditional ranking uses only
scored records and is labeled as such. Also show class recovery over all planned
25 positive / 25 negative patients so failures cannot inflate recovery.
All failed means ranking NA, not a perfect all-unknown readout.

Compare mean-margin direction with the SAME-DISPLAY cached exact XRV head at
unchanged 0.5; retain ties and disagreement. Do not rerun XRV, use its max head,
fit its threshold, turn model consensus into reference or select a new model.
Report duplicate-text geometry and first-two replay deltas separately.
No p-values/superiority claims, best-template score, policy fitting, scorer
weight, candidate reranking, fault-localization or regeneration authorization.

## Execution and privacy

Worker: `TriCompose-v1.2/real_validation/ricord_biovil50.py`.
Tests: `TriCompose-v1.2/tests/test_ricord_biovil50.py`, invented fixtures only.
Prepare is CPU Slurm metadata-only. Run requires explicit approval flag, actual
Slurm job cgroup and CUDA before any pixel/model access. Offline existing assets,
new job-local workspace cache/temp only. New opaque runs, atomic writes, no
overwrite; protected directories 2770, files 0660, project group only.

Scores/aggregates remain protected; no source labels, patient keys, raw image,
report or EHR text in public logs/chat/Git. Public stdout only sanitized status,
runtime/memory and artifact hash. Native worker stdout/stderr is protected.
After completion independently verify hashes, all 50-case lineages, fixed-mean
algebra, missingness, cost/replay and modes. Never edit consumed sources or the
previous selection/opacity results to improve this diagnostic.
