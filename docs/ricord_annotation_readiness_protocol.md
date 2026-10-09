# RICORD-1C: annotation acquisition and aggregate readiness

This is not a benchmark, image acquisition, or permission to submit a job.
The user approved only the official approximately 2.4 MB annotation download
after the public, body-free HEAD check returned 200 and 2,449,004 bytes.

## Sources and scientific scope

- [TCIA collection](https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/)
- [Official catalog and annotation link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281)
- [Primary dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7993245/)
- Dataset DOI: `10.7937/91ah-v663`; CC BY-NC 4.0, noncommercial research and
  attribution required.

The catalog describes adult COVID-positive patients, image interpretations by
thoracic radiology specialists, examination classification, and airspace
disease extent. It reports 998 examinations and 1,257 images; these units are
not interchangeable. COVID positivity alone is not a positive radiographic
pneumonia reference. Typical/indeterminate appearance and airspace grades
must not silently become general clinical pneumonia gold labels.

The official catalog names two withdrawn examinations. Their keys are parsed
in memory for exclusion; no key, per-patient hash, or row is logged or added
to a manifest. No raw catalog or identifiers are copied into Git.

## Approved operation

`TriCompose-v1.2/real_validation/acquire_ricord_annotations.py`:

1. Fetches only the fixed public catalog and its fixed annotation URL, without
   credentials, proxies, cookies, arbitrary endpoints or redirects.
2. Checks the catalog license/dataset/link and withdrawal notice. Bounds the
   catalog and annotation transfers, requires the previously observed byte
   size, and rejects duplicate JSON keys.
3. Saves `annotations.json` unchanged inside a new, non-overwriting protected
   run. Records local SHA256 and observed byte size. No published annotation
   checksum has been verified; TLS/size/local hashing is not a published
   content-integrity guarantee.
4. Emits only allowlisted field names, recognized clinical label definitions,
   and aggregate counts. Unrecognized label names are not printed or forced
   into known classes. Missing annotations remain unknown; multiple distinct
   classification/grade labels on one study are flagged as conflicting.
5. Separates study, series and image metadata counts. Does not assume that a
   study-level annotation applies to every image, or that unknown scope means
   image scope. Patient grouping and label-to-image binding remain unverified.
6. Writes `acquisition.json` and a hash-binding `manifest.json`, with private
   files 0660 and directories 2770 inside the CARC project boundary.

The approved default root is
`artifacts/protected/tricompose_v1_2/reference_datasets/ricord_1c/`.
No image, clinical spreadsheet, patient example, model or checkpoint is
downloaded/opened. No inference, new Slurm submission, threshold fitting,
candidate reranking or synthetic-output modification occurs.

## Gates after this operation

- Inspect actual annotation scope, patient grouping availability, image
  binding, conflicts and class coverage using aggregate-only diagnostics.
- Explicitly predeclare a lung-opacity reference mapping from image
  interpretation labels; use unknown for atypical or ambiguous cases unless
  an explicit reviewed annotation supports a mapping. General pneumonia
  remains a different endpoint.
- RICORD is not listed in the current XRV DenseNet training-source
  description. That is a provenance lead, not verified no-overlap evidence.
  The RSNA 2018 opacity challenge is not synonymous with RICORD 2021 merely
  because the same organization appears in both names.
- Obtain separate approval before acquiring a small fixed image cohort.
  Resolve DICOM display transforms/photometric polarity and frontal-image
  inclusion before running the unchanged frozen scorer.
- Show the complete batch script/resources and obtain explicit approval
  before any future GPU Slurm submission. Do not refit thresholds or alter
  old selections after reading this diagnostic set.

The invented-fixture tests cover approval gating, network/size restrictions,
schema privacy, unknown/conflicting semantics, withdrawal exclusion and
sanitized CLI errors. An acquired annotation file does not establish scorer
validity, clinical fault localization, successful repair, or paper readiness.
