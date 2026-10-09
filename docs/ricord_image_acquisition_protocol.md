# Bounded RICORD-1C acquisition (no inference)

User approval covers at most 50 corresponding images, at most 1 GiB, not the
whole collection and not a GPU submission. This protocol is frozen before
selecting a cohort or downloading images. Historical scoring and outputs stay
unchanged. Data and raw API metadata remain protected; no source key, including
a hashed patient key, is included in a manifest, filename or public log.

## Reference and selection

Reuse the verified annotation acquisition and reader-group audit. The reference
is **derived strict unanimity of three comprehensive label groups**, not a
reproduced official adjudication or a clinical pneumonia diagnosis. Typical or
indeterminate appearance maps to positive lung opacity; negative-for-pneumonia
maps to no lung opacity, according to the official catalog. Other states remain
excluded. Unanimity favors clearer cases and is not a representative final
benchmark. Reader identity and checkpoint training overlap remain unverified.

The official [NBIA public API](https://wiki.cancerimagingarchive.net/display/Public/NBIA%2BSearch%2BREST%2BAPI%2BGuide)
provides collection series inventory, byte sizes, patient/study membership and
SOP inventory. Use only MIDRC-RICORD-1C, its published CC BY-NC 4.0 license,
and DX/CR studies containing **exactly one image in the entire study**. Exclude
official withdrawals. Study-scope labels are not silently assigned to arbitrary
images from multi-image studies. Require nonempty patient identity internally;
take no more than one study per patient across both classes.

Traverse the annotation export's original study order and select the first 25
eligible positives and first 25 eligible negatives while excluding previously
selected patients. Do not sort by model score, visual quality, severity or ease
of download. Stop if either quota cannot be met; never impute an ambiguous
label. This deterministic ordering can introduce ordering bias, disclosed in
the receipt. No model output is consulted.

`prepare` writes and commits a complete protected plan BEFORE the first image
request. Bind source studies and API rows by integer offsets into hash-pinned
raw protected inputs. Get exactly one SOP per selected series. Manifests contain
opaque case IDs and indices only; raw API metadata is a separate protected
artifact, never a public log. Freeze plan, source and protocol hashes.

## Download and checks

`acquire` verifies the plan and all source/artifact hashes, then uses NBIA
`getSingleImage` with the bound series and SOP. No ZIP, bulk collection download,
external credential or viewer-only IDC proxy. Direct HTTPS requests have a
fixed host/endpoint allowlist, no proxy, cookies, authentication, redirects or
automatic retries. Count failed/partial response bytes against the global cap.
Limit to 50 requests/objects, 32 MiB per image, and 1 GiB combined plan metadata
plus image response bodies; preflight the published sizes before any image.
Do not download a replacement if a case fails. Preserve failures and the
original denominator in an immutable receipt; no existing run is overwritten.

Each body is saved under an opaque `case_NNN/image.dcm`. Require exact published
byte count and DICOM Part-10 prefix. Use pinned pydicom 3.0.2, header-only
`dcmread(stop_before_pixels=True, specific_tags=..., force=False)` to compare
study/series/SOP and patient membership against the protected API inventory.
Require single-frame DX/CR, grayscale MONOCHROME1/2, valid dimensions and pixel
storage fields. Reject lateral/unsupported view, but do not invent AP/PA for a
missing view. View, window/rescale/LUT and transfer syntax readiness are recorded
as sanitized metadata only. No pixel decoding, display, PNG conversion, model
imports, inference, threshold fit or GPU submission occurs.

SHA256 verifies local lineage; no published image checksum is claimed. Header
binding does not validate anatomy, diagnostic truth or the display transform.
Decompression, modality/VOI/presentation transforms and classifier preprocessing
must have a separate frozen protocol and approved Slurm job before evaluation.

All new directories use 2770, files 0660, project group ruishanl_1185 (or the
NFS-exported nobody ownership). Invented fixtures test quotas, exclusions,
patient deduplication, privacy, binding, byte limits, redirect denial and
failure/no-replacement behavior. No acquired patient data enters test output.
