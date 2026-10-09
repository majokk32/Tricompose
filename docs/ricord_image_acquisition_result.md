# RICORD-1C 小样本影像准备结果 / bounded image acquisition

Completed 2026-10-05. This is data preparation for independent image-label
scorer validation, **not a new model evaluation result**. No inference,
pixel decoding, GPU task, new Slurm submission, threshold fitting or old
candidate-selection change occurred.

## 实际完成 / completed

| Check | Result |
| --- | ---: |
| Frozen planned images | 50 |
| Acquired and DICOM-header-bound images | 50 |
| Failed / replaced / unattempted | 0 / 0 / 0 |
| Unique patients / studies | 50 / 50 |
| Derived lung-opacity positive / negative | 25 / 25 |
| Image response bytes | 471,794,146 (449.938 MiB) |
| Plan metadata + image response bytes | 472,905,031 (450.997 MiB) |
| Approved ceiling | 50 images, 1 GiB |
| Acquisition + header checks | 89.791385 seconds |
| New invented-fixture tests / full V1.2 suite | 19 / 1,762 passing |
| Model calls / pixels decoded / Slurm submissions | 0 / 0 / 0 |

The tiny official pydicom 3.0.2 wheel (approximately 2.4 MB) was installed in a
new workspace-local dependency directory, with no dependencies, model weights
or external environment changes. It is separate from the above dataset-body
byte count; even including it, acquisition remains well below 1 GiB.

The fixed cohort was selected before any image request: traverse original
annotation study order; first 25 eligible per class; one study per patient
across both classes; only whole studies containing exactly one CR/DX image.
No visual inspection, scorer output or download success chose the cases.
The series inventory contains 998 studies / 1,241 series / 1,257 images,
including 759 single-image studies. Of the derived reference studies,
88 positives and 113 negatives met single-image/metadata eligibility before
selected-patient deduplication. All 50 selected images passed; none was replaced.

## 保存位置 / locations

These are relative to `/project2/ruishanl_1185/inference_3mod`:

```text
artifacts/protected/tricompose_v1_2/reference_datasets/ricord_1c/
├── annotations_12654973_001/           # unchanged official annotations
├── image_plan_12666569_001/
│   ├── plan.json                      # fixed opaque indices and reference
│   ├── raw_series_inventory.json      # protected-only source metadata
│   ├── raw_sop_inventory.json         # protected-only binding metadata
│   └── manifest.json
└── images_12666569_001/
    ├── acquisition.json               # status + opaque-case header checks
    ├── manifest.json
    ├── case_000/image.dcm
    └── ... case_049/image.dcm
```

Plan manifest SHA256:
`b88b46f9e1b9c8620432f868325d6e6f48a0435ab7ef2356cd0e3bd0f6012142`.

Image acquisition manifest SHA256:
`de8e303594a3dba71da36185b4272d516b577a0bb5ed567720fdf517b6b95284`.

All artifact hashes, source pins, header-parser source hashes, annotation/plan
lineage, case uniqueness, aggregate arithmetic and private modes were checked.
Directories are 2770 and files 0660, with project-group ownership exposed as
65534 on NFS. The raw source identifiers occur only in protected source
metadata/DICOMs, never filenames, manifests or public logs. Git ignores all
protected artifacts and temporary dependencies. Reusing the same image run ID
was rejected with FileExistsError before any new image request or overwrite.
The old online/conflict run manifest hashes remain unchanged.

## 显示转换还没完成 / display preparation remains

Header-only reading follows [pydicom's documented stop-before-pixels interface](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.filereader.dcmread.html).
The [NBIA official public API](https://wiki.cancerimagingarchive.net/display/Public/NBIA%2BSearch%2BREST%2BAPI%2BGuide)
provided series/SOP membership and single-image downloads. Internal study,
series, SOP and patient comparisons all passed. This checks binding, not
clinical truth, anatomy or pixel display.

| DICOM header property | Images |
| --- | ---: |
| AP / PA | 46 / 4 |
| DX / CR | 20 / 30 |
| MONOCHROME2 / MONOCHROME1 | 42 / 8 |
| JPEG Lossless SV1 transfer syntax / explicit VR little endian | 48 / 2 |
| Rescale slope/intercept pair | 50 |
| Window center/width pair | 35 |
| VOI LUT present | 15 |
| Modality LUT present | 0 |

Do **not** open these as ordinary PNGs or silently pass raw integer pixels
through the existing PNG preprocessing. A separate frozen conversion contract
must handle JPEG-lossless decoding, modality rescale, VOI LUT/window selection,
MONOCHROME1/presentation polarity, dynamic range and the frozen classifier's
normalization/crop/resize. The eight MONOCHROME1 images make ignoring polarity
particularly unsafe. Display-transform validity is still false; no pixel was
decoded or displayed in this acquisition.

## 科学边界与下一步 / scientific limits and next step

The reference is **derived unanimity of three comprehensively populated label
groups**, not verified reader identity or reproduced official adjudication.
It tests **lung opacity**, not general pneumonia. Atypical, disagreement and
missing/uninterpretable readings were excluded, not converted to negatives.
The 25+25 balance, unanimity and original-export ordering introduce selection
bias; this is a small diagnostic, not a representative paper-final benchmark.
Local hashes bind artifacts; a published image checksum was not verified.
Checkpoint training overlap remains unresolved and primary-metric eligibility
stays false.

Next: freeze and test the DICOM display/preprocessing implementation, then show
the complete proposed Slurm script and resources for explicit approval. Only
after approval run the unchanged frozen XRV lung-opacity head on this fixed
cohort and report AUROC/AUPRC plus fixed-threshold sensitivity/specificity and
uncertainty. Do not fit thresholds on these 50 cases, relabel old synthetic
outputs, or claim a clinical generation repair before scorer reliability is
demonstrated. No GPU job is authorized by the image-download approval.
