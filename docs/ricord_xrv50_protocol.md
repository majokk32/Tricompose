# Frozen RICORD-1C display and XRV diagnostic

This operation is prepared, not submitted. User approval for image acquisition
did not approve pixel decoding/inference. Show the COMPLETE Slurm script and
resource request, then obtain explicit approval. No training, threshold fitting,
case substitution, generator optimization, original-file edits or clinical
image inspection. Protected data only; no external API or download in the job.

## Fixed inputs and endpoint

Use exactly the 50 acquired, bound RICORD-1C images (25+25, 50 patients).
The plan and image acquisition manifests, all artifact/source hashes and
header parser source hashes must verify. Preserve opaque IDs and source integer
offsets; never emit source patient/study/SOP keys, including hashed IDs.
Reference is derived three-group strict unanimity, not official adjudication
or general pneumonia. Ordering/unanimity/stratification bias and unresolved
training overlap remain disclosed; paper-primary eligibility is false.

Diagnostic endpoint: frozen XRV DenseNet121-res224-all **exact Lung Opacity
head**, AUROC and average precision, default 0.5 sensitivity/specificity,
balanced accuracy, confusion counts and Wilson 95% intervals. AP is not
trapezoidal PR area and does not estimate natural-prevalence PPV here.
Use a fixed-seed class-stratified patient bootstrap (2,000 replicates) for
AUROC/AP uncertainty; one image per patient, no resampling to select a result.
These intervals do not account for selection bias or annotation uncertainty.

The historical 14-finding adapter uses max(Lung Opacity, Infiltration) for its
lung_opacity field. Keep that distinct from the exact head. As a SECONDARY
legacy-proxy transport diagnostic, show that unchanged max-score at 0.5 and
the pre-existing weak-reference threshold 0.642907085. Verify the old bundle
hash/checkpoint/mapping; do not fit it here or apply it to the exact head.
The new DICOM display path changes preprocessing provenance: transported
thresholds are NOT newly validated calibration/probabilities. No Brier/ECE.
Pneumonia predictions are not scored against the opacity reference.

## Frozen display contract

Implement with official [pydicom processing](https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html)
and [JPEG-lossless-supported plugins](https://pydicom.github.io/pydicom/stable/guides/plugin_table.html):
pydicom 3.0.2, pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0. Install only into new
workspace-local targets, never the external model environment. Hash installed
Python/compiled-plugin sources and pin environment package versions.

1. For the acquired JPEG Lossless SV1 syntax use the explicit pylibjpeg plugin;
   for explicit-VR little endian use the native decoder. No silent plugin switch.
2. Validate single-frame grayscale CR/DX, decoded dimensions/storage range,
   source membership and unchanged file SHA256.
3. Apply modality LUT/rescale first. Select first VOI LUT if present, otherwise
   first complete window pair, as prescribed by the predeclared policy, not
   visually/scorer-selected. Respect LINEAR/LINEAR_EXACT/SIGMOID.
4. Map the metadata-defined output range to [0,1], never image min/max or fitted
   percentiles. Do not change histogram/contrast/geometry. Reject missing VOI
   rather than invent a dynamic-range fallback. Integral LUT input is required;
   do not round nonintegral rescaled pixels into LUT indices.
5. Follow [DICOM DX presentation/polarity semantics](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.11.3.html):
   use IDENTITY or INVERSE presentation shape ONCE when present, otherwise
   photometric interpretation. Shape/photometric contradiction fails closed.
   Do not invert a second time for MONOCHROME1 or use intensity-relationship
   sign as a display transform. Unsupported presentation LUT sequence fails.
6. Mask explicit pixel padding to black; no additional cropping. Quantize to
   uint8 [0,255] with rounding and save opaque protected PNGs. Constant/all-padding
   displays fail, not negative labels; no replacement or best-window search.
7. Reuse unchanged frozen runtime: PIL L -> XRV normalize(maxval=255) ->
   XRayCenterCrop -> XRayResizer(224) -> frozen operating-point-normalized model.

`prepare` uses metadata/header-only checks and invented tests; it seals the
display policy, code/tests, dependencies, checkpoint, old threshold bundle and
input manifests before any real decoding/scoring. `evaluate` checks explicit
approval flag + Slurm + CUDA BEFORE any pixel/model access. A single model
load, one inference per decoded case, at most 50; no alternate seeds/models.
Case-level errors remain protected and sanitized, original denominator 50 stays
visible. If a case fails, show incomplete status/coverage and do not promote a
conditional subset to a complete 50-case evaluation.

PNG images, individual scores, summaries and manifest stay under protected,
2770 directories / 0660 files / project group. Public logs show only sanitized
status/dimensions/runtime/memory/hashes. The worker's native stdout/stderr stays
in private per-job logs. Fixed inputs, legacy selections and consumed historical
code/protocols remain immutable. Passing invented transform/codec tests verifies
engineering behavior, not clinical adequacy of the real displays or scorer.
