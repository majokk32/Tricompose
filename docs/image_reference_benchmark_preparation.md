# Image-reference benchmark: resource audit and proposed integration

Status: resource audit complete; no new dataset acquired or model benchmark
executed. This follows the [cached EHR/image reliability audit](ehr_cxr_reliability_audit.md).
Do not treat a proposed dataset or protocol as an available local asset.

## Scoped local result

Read-only metadata checks covered the shared `datasets`, `local_data` and
`EHRXDiff_baseline/local_data` roots below `/project2/ruishanl_1185`, to depth
three, with an explicit 12,000-entry bound. The targeted inventory scanned 433
entries, was not truncated and had no inaccessible directories. Image/patient
directories were excluded from descent. This does not establish absence in
other CARC directories; unrelated personal repositories were not investigated.

Three candidate metadata/archive files were identified: existing MIMIC
automatic labels, manual report-label annotations and a source-report ZIP.
Only recognized CSV headers and cryptographic hashes were checked; no data row
or image/report content was opened, and the ZIP was not extracted. The manual
labels evaluate report mentions, not independent image adjudication. The
[official MIMIC-CXR-JPG description](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
documents this distinction.

Protected evidence:
`artifacts/protected/tricompose_v1_2/reference_resource_audits/image_reference_12625457_001/`.
It contains `inventory.json`, `RESULTS_CN_EN.md` and `manifest.json`; manifest
SHA256 `7a29156768423f9099e2bcda5ea41bfff80a8fe6c920df680f6de6c8d61cc4e5`.
No source was modified, no credentials were accessed, and no job/model ran.

## Why VinDr-CXR is the proposed first candidate

[VinDr-CXR v1.0.0](https://physionet.org/content/vindr-cxr/1.0.0/) provides
radiologist-annotated images, including a global pneumonia diagnostic
impression and local findings. Its 3,000-image test reference uses five-reader
consensus. It is DICOM data requiring credentialed access and a dataset-specific
DUA; existing MIMIC access does not establish access to this resource.

The existing all-source XRV runtime describes its training sources as
`nih-pc-chex-mimic_ch-google-openi-rsna`. Thus RSNA/NIH or CheXpert annotations
cannot establish unseen-image independence without checking actual splits and
training overlap. VinDr is not listed in that inspected description, which is
a useful lead, NOT proof of zero overlap. Keep checkpoint-training independence
as an explicit unresolved provenance question.

[CheXpert's official competition protocol](https://stanfordmlgroup.github.io/competitions/chexpert/)
emphasizes five findings that do not include pneumonia. It can be useful for
those findings but is not by itself the requested pneumonia diagnostic.
[RSNA's pneumonia annotation paper](https://pubs.rsna.org/doi/10.1148/ryai.2019180041)
labels pneumonia-suspicious lung opacity, not confirmed pneumonia. Do not
silently substitute opacity for pneumonia or claim independent evaluation of
a checkpoint declaring RSNA/NIH training.

## Integration requirements before inference

1. Obtain a user-confirmed authorized local path or separate dataset-acquisition
   authorization. Confirm the dataset-specific DUA without reading or asking
   the user to paste credentials. All acquired files stay project-private below
   `artifacts/protected/`; do not mirror into public Git or another project.
   Before using the shared project-group boundary, confirm that everyone granted
   data access meets that dataset's access requirements. Project membership or
   MIMIC authorization alone is not evidence of VinDr authorization.
2. Check the actual official image-label CSV schema, version, integrity and
   accessible image inventory internally. Use `image_labels_test.csv`, not a
   Kaggle bounding-box-only substitute. The installed XRV `VinBrain_Dataset`
   helper exposes fourteen box-label pathologies without a pneumonia head;
   its label parser is not the official global-pneumonia reference parser.
3. Freeze exact, named correspondence for the eight existing scorer heads:
   atelectasis, cardiomegaly, consolidation, edema, lung opacity, pleural
   effusion, pneumonia and pneumothorax. Preserve other/missing states as
   unknown. Diagnose radiologist image impressions, not microbiologically
   confirmed disease or EHR causality. Missing boxes alone are not negatives.
4. Review and pin DICOM modality LUT, windowing, photometric polarity, scaling,
   center crop and resize. The current saved-image adapter opens PNGs through
   PIL; it is not a validated direct DICOM loader. A new DICOM protocol needs
   its own provenance. Do not forge equality with old PIL/PNG fingerprints.
5. Check whether patient grouping is available. Image IDs are not patient IDs;
   do not claim a patient-disjoint resplit or patient-level confidence interval
   solely from unique image IDs. Preserve official partitions and document any
   unavailable grouping/training-overlap evidence.
6. Seal the case inventory before any model scores. A proposed 50-image pilot
   is an engineering/diagnostic cohort, not a paper-final efficacy estimate.
   Any class stratification must be declared from labels before inference,
   not chosen for easy images or favorable predictions. Show class counts and
   missingness; if a finding lacks both classes, retain it with AUROC/AP NA.
7. Initially test the unchanged frozen classifier and explicitly declared
   threshold-transport profile. Do not refit thresholds, alter head masks or
   change current synthetic choices based on this test. DICOM/domain transport
   does not establish validity on synthetic images automatically.

## Required readouts

- Per-finding explicit-reference counts and coverage, raw-score AUROC/AP where
  both classes exist, confusion counts, sensitivity and specificity.
- Fixed operating-point results with uncertainty intervals and excluded-label
  counts; no Brier/ECE for operating-point-normalized nonprobability scores.
- Reference type, patient-group availability, acquisition/decoder/checkpoint
  hashes, runtime, failures and memory. No real rows, keys or images in logs.
- A separate qualification statement: scorer discrimination versus image
  annotations is not end-to-end EHR fidelity, validated error localization or
  repair success. Preserve negative results and all old outputs.

Only after dataset/schema/decoder preflight should an exact script be prepared.
A small XRV pilot is expected to fit a single P100-class GPU, but actual host
memory and wall-time requests depend on image decoding and I/O readiness.
Show the complete batch script/resources and obtain explicit approval before
submission. This document authorizes no download, inference or new submission.

## Implemented readiness entry point (not a benchmark)

`TriCompose-v1.2/real_validation/vindr_readiness.py` implements a stdlib-only
readiness receipt. Its default invocation opens no local dataset, does not
authenticate, and writes an immutable protected run:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  TriCompose-v1.2/real_validation/vindr_readiness.py \
  --run-id <new_opaque_run_id> --probe-public-access
```

The optional probe uses HEAD only against one fixed official annotation URL,
without cookies, authentication, proxies, redirects or response-body reads.
HTTP 200 would not verify dataset identity or a DUA. HTTP 403 means anonymous
access did not succeed at that URL; it says nothing about the user's
authenticated access. No token or credential is requested or read.

The actual readiness run is
`artifacts/protected/tricompose_v1_2/vindr_readiness_runs/readiness_12625457_001/`.
It contains `readiness.json`, `RESULTS_CN_EN.md` and `manifest.json`; manifest
SHA256 `c9d01274d9465a6c20ff43fae2fa6d4f003a61ff9bd45a2eebe66bad9e3b3ca7`.
The unauthenticated probe returned 403 in 0.713 seconds. Hashes and private
directory/file modes were independently rechecked. This run consumed no real
annotation row or image pixel, and made zero model calls or job submissions.
Status: `blocked_authorized_local_data_required`.

### Optional header-only inspection after access confirmation

This branch is not executed in the readiness run above. It requires a private
attestation under `artifacts/protected/`, the explicit
`--allow-authorized-header` flag, and an existing approved Slurm allocation.
It opens exactly `annotations/image_labels_test.csv` below the attested local
root, only for one bounded, unbuffered physical header line. It never reads
annotation rows or opens DICOM images. The source must remain within the CARC
project; external source directories remain read-only.

An attestation is a user's recorded confirmation, NOT automatic proof of
PhysioNet authorization. Do not populate true confirmations merely because
the user said “go” or has MIMIC access. Its exact non-secret contract is:

```json
{
  "schema_version": "tricompose-vindr-access-attestation-v1",
  "dataset": "vindr-cxr",
  "dataset_version": "1.0.0",
  "purpose": "header_only_readiness",
  "local_dataset_root": "/actual/authorized/local/root",
  "dataset_specific_dua_confirmed": false,
  "required_training_confirmed": false,
  "all_project_group_readers_authorized_confirmed": false
}
```

The example deliberately has false confirmations and is rejected. A genuine
receipt needs explicit user-confirmed true values; it must have directory mode
2770 and file mode 0660 under the project-group boundary. Extra account/token
fields, missing confirmations, or integer substitutes for booleans fail closed.
Never paste a credential into this receipt or chat.

The candidate named mapping covers the eight unchanged XRV heads. It recognizes
only declared exact column aliases; unknown column text is hashed. Duplicate
aliases, bounding-box schemas, undocumentable label-vector ordering and missing
required heads stop preflight. `rad_ID` is rejected for this test-consensus
entry point. No absent column or cell is turned into a negative reference.
The real official header and label-value encoding have NOT been observed, so
even a compatible named header remains pending semantics/DICOM review; it
cannot authorize a GPU benchmark or confer primary-metric eligibility.

Twenty-five invented-fixture tests cover privacy guards, one-line reads,
schema rejection, missingness, mocked HTTP behavior, immutable writes and
permissions. The fixtures contain no real patient data. Patient grouping,
checkpoint overlap, complete label decoding, acquisition integrity and DICOM
preprocessing remain unresolved. No direct-DICOM inference adapter is claimed.
