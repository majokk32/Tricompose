# RICORD-1C 标注准备结果 / annotation readiness result

Follow-up: the separately approved bounded image acquisition has now completed
50/50 images (25+25, 50 unique patients), with no inference. See
`docs/ricord_image_acquisition_result.md` for the frozen cohort, actual bytes,
header checks and remaining DICOM display gate. The annotation-only operation
and its historical limitations documented below remain unchanged.

Status: the user-approved small annotation acquisition and aggregate audit
completed. Images and model scores are NOT available from this operation.
No GPU task, model call, new Slurm submission, training, threshold fitting or
historical selection change was made.

## What actually exists

The official JSON was acquired at 2,449,004 bytes, unchanged, from the fixed
TCIA annotation link under CC BY-NC 4.0. The transfer took approximately one
second including the public-catalog checks and aggregate parsing. Local SHA256
and observed-size checks pass; no published checksum was available/verified.

Protected original and acquisition receipt:

```text
artifacts/protected/tricompose_v1_2/reference_datasets/ricord_1c/annotations_12654973_001/
  annotations.json
  acquisition.json
  manifest.json
```

Manifest SHA256:
`e52a15d26d57747627b7e33385d71e94ed67ea4529716244792f637e8cb38a90`.

Protected group diagnostic:

```text
artifacts/protected/tricompose_v1_2/ricord_annotation_audits/reader_groups_12654973_002/
  audit.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`fd1dd438da60ceb527412a70efa3bbe9fb521b490aec6724eed6ddd5f9bdefe5`.

Both runs' artifact/source hashes, binding, arithmetic and private modes were
checked. Directories are 2770 and files 0660; NFS exposes the project group as
65534. The existing `reference_datasets` parent was tightened from 2775 to
2770; no access outside the project was granted. Existing-run invocations
were rejected before download/output writes. No prior annotation/run was
overwritten. These run IDs are opaque labels, not assertions about the current
Slurm allocation.

## Actual export structure

The export contains 1,000 studies, including two official withdrawals;
998 remain active. Its 48 label definitions form six label groups, with STUDY
scope. Three non-adjudication groups contain annotations for every active
study. Two other groups are sparsely populated; their role is not inferred.
The adjudication-marked group covers 143 active studies, not all 998.

The export does not contain a usable image/series inventory or patient-group
field in the inspected schema. Zero image metadata entries here does not mean
the collection has zero images. The [official TCIA catalog](https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/)
separately lists 998 examinations and 1,257 images. Label-to-image binding and
patient grouping remain unresolved.

The first pooled-group inventory flags differing labels rather than making a
majority vote. Its apparent conflicts are not automatically annotation errors:
different readers can give different interpretations of the same examination.

## Derived unanimity diagnostic (not official final gold)

The group-specific audit excludes the withdrawals and compares the three
comprehensively populated groups. It neither uses a scorer prediction nor
reconstructs official majority/adjudication. Participant-to-group identities
have not been independently verified.

| Classification across the three groups | Studies |
| --- | ---: |
| Unanimous typical appearance | 31 |
| Unanimous indeterminate appearance | 86 |
| Unanimous atypical appearance | 65 |
| Unanimous negative for pneumonia | 146 |
| Between-group disagreement | 597 |
| Missing/uninterpretable classification in one or more groups | 70 |
| Within-group conflicting classifications | 3 |
| Active denominator | 998 |

For the proposed **lung-opacity** diagnostic only, catalog-defined typical or
indeterminate appearance supports an opacity-positive reference, and negative
for pneumonia means no lung opacities. Atypical/disagreement/missingness remain
unknown or excluded, not implicit negatives.

| Proposed derived opacity reference | Studies |
| --- | ---: |
| Positive | 117 |
| Negative | 146 |
| Unknown or excluded | 735 |

These label counts could support a 25-positive + 25-negative small diagnostic.
No 50-case list has been selected or image downloaded. Unanimity deliberately
favors less-disputed examples and introduces selection bias. It is not a
representative test-set estimate, an independent-patient count, or validation
of general pneumonia diagnosis.

The [primary RICORD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7993245/)
describes three radiologist reads and majority/adjudication. This derived
unanimity subset does not claim to reproduce that adjudicated reference.

## Tests and limitations

Twenty-three new invented-fixture tests pass (12 acquisition, 11 group audit).
The full V1.2 suite also passes: 1,743 tests using the documented import paths.
An initial invocation omitted the real-validation/SynEHRgy import paths and
had four collection import errors; fixing the invocation required no old
module edits. Temporary fixtures are redirected into the workspace `.tmp`.
The reproducible command is:

```bash
TMPDIR=/project2/ruishanl_1185/inference_3mod/.tmp \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:TriCompose-v1.0/src:TriCompose-v1.1/src:experiments/synehrgy_v2/src:TriCompose-v1.2/real_validation:TriCompose-v1.0/eval/report_v1_1 \
python -m unittest discover -s TriCompose-v1.2/tests -q
```

Fixtures contain no real patient data; they test privacy, bounded acquisition,
unknown/conflict preservation, withdrawals and group availability. An initial
group-audit attempt failed closed because it confused interpretable diagnosis
count with read availability; this was corrected before a successful audit
receipt or any inference. Abstaining readers remain present.

Primary-metric eligibility stays false. Outstanding gates:

1. Separate approval for a small fixed image acquisition; no full approximately
   12 GB collection download is implied.
2. Bind study-level labels to eligible frontal images and obtain patient
   grouping information without exposing identifiers.
3. Freeze DICOM display/window/polarity and preprocessing provenance.
4. Document training-overlap uncertainty: RICORD is not listed in the current
   XRV DenseNet declared sources, but this does not prove zero overlap.
5. Show the complete GPU batch script/resources for explicit approval, then
   evaluate the unchanged frozen scorer and thresholds. Do not refit on this
   diagnostic or use it to retrospectively relabel old synthetic outputs.

This advances independent image-annotation availability. It does not yet
validate the scorer, locate a clinical generation error, establish repair
success or demonstrate an improvement over static selection.
