# Public RSUA frozen-XRV pneumonia diagnostic

Status: acquisition, metadata-only staging and approved GPU diagnostic completed
as **job 12636566**. This does not clear a clinical scorer-validation or repair
gate. No historical candidate, threshold, score or selected triple is changed.

## Why this dataset, and what its labels mean

The official [RSUA dataset](https://data.mendeley.com/datasets/2jg8vfdmpm/1)
is public, version 1, CC BY 4.0, DOI `10.17632/2jg8vfdmpm.1`. Its associated
[data paper](https://doi.org/10.1016/j.dib.2023.109640) describes a lung
segmentation dataset with 53 pneumonia, 32 normal and 207 COVID-19 images.
The download page instead calls the 32-image group Non-Covid. We preserve that
distinction: its pneumonia-negative state is a **paper-described normal-cohort
proxy**, not an independently adjudicated per-image disease label.

Clinician validation described by these sources concerns lung segmentation
masks; it must not be claimed as blinded disease-label adjudication. Patient
grouping, age population and checkpoint training overlap are unverified. The
checkpoint's declared training-source list does not list RSUA, but that alone
does not prove no shared images. This small resource offers an accessible
single-finding diagnostic, not a substitute for an independent adult-CXR
benchmark or a qualified three-modal clinical evaluator.

## Acquisition and metadata receipt

`TriCompose-v1.2/real_validation/acquire_rsua.py` discovers the versioned public
file list from the official public endpoint, checks the page's public/license
metadata, and fetches only the Validated ZIP. No credential, cookie or gated
mirror is used. Exact published size and SHA256 are verified. The archive is
34,738,327 bytes (about 33.1 MiB), downloaded in 10.449 seconds.

Protected acquisition run:

```text
artifacts/protected/tricompose_v1_2/reference_datasets/rsua/acquisition_12625457_001/
```

Acquisition-manifest SHA256:
`94e926014414ec59e712276ff7250d2fdf5b2493318bc3d8065fb4cf38eccc59`.

Directory/header inspection establishes 292 chest images and 292 lung masks,
each represented in BMP and NPY. These are **not 1,168 independent images**.
BMP chest-image headers consistently specify 256x256, 8-bit, uncompressed.
NPY headers specify 256x256x1 float32; no NPY values or pickles were loaded.

| Published class | Chest images | Corresponding masks |
|---|---:|---:|
| Pneumonia | 53 | 53 |
| Non-Covid, described as normal in the paper | 32 | 32 |
| COVID-19 | 207 | 207 |

Only archive metadata and array/image headers have been inspected. Source
names are not emitted: the receipt keeps opaque ZIP member indices and name
hashes. No image pixels were decoded, no model was called and no real target
was used in generation.

## Sealed pilot before model scores

`real_validation/rsua_pilot.py stage` fixes 25 pneumonia and 25 normal-proxy
images using deterministic seed-0 hash ranking. Mask members, COVID-19 and
alternate NPY copies are excluded. There is no score-dependent case selection
or substitution. This image-level, label-stratified cohort is not a patient
split or an estimate of natural prevalence.

```text
artifacts/protected/tricompose_v1_2/real_validation/rsua_pilot_cohorts/cohort50_12625457_001/
  cohort.json
  manifest.json
```

- Cohort SHA256: `ed33c7708ac737aa8e270da2219457def542186f02bcc085c7188c7a135779c1`.
- Manifest SHA256: `776727f4da1af9bde231e662f57c9edc2a214ad2b0d4b1c7005e85724896af82`.
- Worker SHA256: `c57276dbe9281e10b6602fe98b333ca6d5be48c6f7697b1f391fe6002e122266`.
- Headers, class/role inventory, hashes and protected modes passed preflight.
- Model calls: zero. `primary_metric_eligible` stays false.

## Proposed GPU run and metrics

Prepared script: `TriCompose-v1.2/slurm/37_rsua_xrv50_gpu.sbatch`.
Account `ruishanl_1185`; partition `gpu`; one unrestricted GPU type; two CPUs;
8G host RAM; five-minute execution cap. Queue wait is separate from that cap.
There is no reason to reserve an A40 exclusively for this DenseNet diagnostic.
The entire script/resources were shown and explicitly approved, then submitted
unchanged on 2026-10-03. Future submissions still require separate approval.

The worker uses the existing read-only Torch/XRV environment and the exact
frozen `densenet121-res224-all` checkpoint. It reuses the saved PIL grayscale,
255 normalization, center crop and 224 resize protocol and validates its
fingerprint/head mapping against the saved threshold bundle. Both checkpoint
and threshold bytes are pinned. Network model fetching is disabled; caches
and diagnostics stay project-private inside the workspace.

After approval, only the selected original BMPs are copied internally to opaque
protected filenames and decoded for XRV. Duplicated selected image bytes cause
a fail-closed run, not cohort replacement. Runtime calls are inference-only;
the model is in eval mode with all parameter gradients disabled.

Report pneumonia AUROC and average precision, then TP/FN/TN/FP, sensitivity,
specificity and balanced accuracy under two **unchanged** operating profiles:

1. Historical default 0.5.
2. The saved report-derived weak-reference threshold, 0.55457607, transported
   without fitting on RSUA.

Precision/F1 describe this artificially balanced 25+25 pilot only. The scores
are XRV operating-point-normalized outputs, not calibrated clinical
probabilities: do not compute probability-semantic Brier/ECE here. No threshold
optimization, synthetic winner replacement or clinical repair decision follows.

Results were atomically committed to:

```text
artifacts/protected/tricompose_v1_2/real_validation/rsua_xrv_pilots/rsua_xrv50_12636566/
  summary.json
  scores.json
  score_table.csv
  manifest.json
  inputs/case_####.bmp
```

All directories/files use project-boundary modes 2770/0660. Public Slurm
output contains fixed status only; model stdout/stderr are private. Full V1.2
regression before submission: 1,011 invented-fixture tests passed in 13.019
seconds, including 12 acquisition and 11 pilot tests. Unit test success is not
model or clinical validation.

## Completed single-finding results

The job started on a P100 after approximately six seconds in queue and completed
with exit `0:0` in 23 seconds. The worker took 11.687416 seconds, making exactly
50 frozen-model calls and no generation calls. Torch peak allocated VRAM was
0.045 GiB; this is not device-wide occupancy or reserved memory. Artifact hashes
and protected group/modes were independently checked without opening pixels.

| Operating profile | AUROC | AP | TP/FN/TN/FP | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---|---:|---:|---:|
| Default 0.5 | 0.7248 | 0.69888206 | 12/13/19/6 | 0.48 | 0.76 | 0.62 |
| Unchanged weak-reference 0.55457607 | 0.7248 | 0.69888206 | 6/19/23/2 | 0.24 | 0.92 | 0.58 |

The ranking signal is nontrivial but false-negative risk is substantial on this
small proxy task. An XRV-negative result alone cannot establish that a generated
image clinically lacks pneumonia. Conversely, scorer limitations do not prove
the generated image is correct. No acceptance rule or threshold was adjusted;
`primary_metric_eligible` remains false.

Result-manifest SHA256:
`92a64874818f37e0d363e8406064d4619cb9edefa01ccab43881cf1a1df7d0e0`.
Protected bilingual handoff:
`artifacts/protected/tricompose_v1_2/real_validation/rsua_pilot_reviews/review_12636566_001/RESULTS_CN_EN.md`.
