# RSUA same-cohort frozen XraySigLIP polarity diagnostic

Prospective model readout on the already inspected developmental RSUA 50-image
cohort, not an untouched validation cohort or independent clinical adjudication.
Keep all old cases, predictions, thresholds, candidate scores and winners.
The [dataset](https://data.mendeley.com/datasets/2jg8vfdmpm/1) calls the comparison
group Non-Covid; the [data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10570962/)
describes it as normal. Physician validation concerns lung segmentation masks,
not independently adjudicated per-finding clinical labels. Age domain, patient
grouping and checkpoint training overlap remain unverified. Do not claim clinical
localization accuracy, validated absence, probability calibration or repair.

## Fixed inputs before this model's scores

- Existing RSUA 25 pneumonia / 25 normal-proxy cases, in original opaque order;
  no new cohort selection, downloads, substitute or difficult-case removal.
- Use the exact losslessly converted grayscale PNGs from BioViL job 12637081,
  bound to manifest `a8eed9db9e20146cb1bc7ca597aed78f8bfb5cd1683b29e82dde99497b08992a`.
  Preserve original XRV image hashes and cohort hash
  `ed33c7708ac737aa8e270da2219457def542186f02bcc085c7188c7a135779c1`.
- Frozen native SiglipModel/processor, existing full local XraySigLIP weights,
  float32/eager, actual 512x512 inputs, max-length 64-token text padding.
  Use the exact eight-head / three-template catalog of the completed six-image
  probe, all 48 text endpoints per image. No fitted head, template choice,
  truncation, adapter, quantization, new threshold or negative-to-positive flip.
- Only pneumonia has the published binary cohort proxy. The other seven heads
  are recorded readouts without reference evaluation or invented labels.

Metadata preparation uses the existing CPU Slurm allocation and reads bounded
receipts, opaque image/hash inventories, native runtime source and model-small
files only. Check image/weight file stats, not bytes or pixels. Record known
hashes from previous approved jobs. The blind plan contains no class labels,
XRV scores or BioViL cosines. Do not recursively follow manifest source trees.

## Approved GPU execution (requires new explicit approval)

Before image/weight access verify the actual Slurm cgroup, explicit allow flag,
CUDA and minimum 12 GiB usable VRAM. Authenticate plan/program/runtime/assets.
Use the mechanical image guard before every callback; unavailable/uniform inputs
receive null scores, never accepted or substituted. Keep all 50 primary inputs.
Replay only the first two fixed images as technical checks, never best-of-two.
At most **52 callback attempts / joint image-text forwards**, zero retries;
reserve/fsync each attempt before callback, record actual forward/load/failure
counts, guard blocks and elapsed time. A failed attempt consumes its budget.
Do not repeatedly reload a failed model.

Predictions and technical repeats are written and fsynced **before** parsing
cohort references/XRV predictions or computing comparisons. No label, score,
expected answer, report, EHR, patient ID or arm name is passed to the model.
Failure/interruption journals remain private for reconciliation, not blind
resume. Full image/weight hashing occurs only in the separately approved GPU
allocation. Public output is sanitized status/runtime/memory/hashes only.

## Predeclared readouts

For pneumonia report AUROC/AP of each original positive-minus-negative cosine
template margin and the fixed arithmetic mean of all three. Do not select the
best template, reverse a poor AUROC or fit a threshold. Separately report:

- positive-reference wins (margin > 0), negative-reference wins (margin < 0),
  their balanced mean, ties, class denominators and availability;
- direction stability across all three templates, stable coverage and polarity
  agreement with the cohort proxy among stable cases, with full denominator;
- maximum endpoint/margin changes on the two technical repeats;
- unqualified mean-margin direction comparisons with BioViL-T and both frozen
  XRV profiles (0.5 and 0.55457607), preserving ties/unknown as not comparable.

Zero margin is an authored-text ordering boundary, not a fitted clinical cutoff.
Raw cosine/logit/margin is not a probability; no Brier/ECE or clinical confidence.
If any primary readout is unavailable, keep all 50 rows and full denominators,
mark full-cohort AUROC/AP and rates unavailable rather than reporting a selected
complete-case accuracy. Numerical repeatability is not clinical validity.

Outputs use fresh atomic protected `real_validation/rsua_siglip_plans/` and
`rsua_siglip_pilots/` runs, directories 2770/files 0660, refuse overwrite. Seal
consumed source/artifact hashes. Worker `real_validation/rsua_xraysiglip.py`;
invented-fixture tests `tests/test_rsua_xraysiglip.py`. All clinical acceptance,
primary-metric, regeneration, selection-change and probability-semantic flags
stay false. No real MIMIC EHR/report/target or external API is used.
