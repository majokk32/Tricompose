# RICORD-1C 50 例评分准备 / scoring preparation

Prepared 2026-10-05; **not submitted, no real pixel decoding or inference**.
The user's generic continuation authorizes preparing the implementation, not
submission before the complete batch script and resources have been shown.

## Implemented and checked

- `TriCompose-v1.2/real_validation/ricord_display.py`: metadata-grounded
  modality -> VOI -> single polarity transform -> uint8 protected PNG policy.
  No percentile fitting, dynamic image-minmax stretching, best-window search
  or clinical/anatomical editing. Missing/ambiguous metadata fails closed.
- `TriCompose-v1.2/real_validation/ricord_xrv50.py`: separate header-only
  preparation and guarded Slurm/CUDA-only scoring, fixed original denominator,
  no replacement, frozen checkpoint, exact-head/legacy-proxy separation,
  AUROC/AP, fixed-threshold confusion statistics, Wilson intervals and fixed-seed
  2,000-replicate class-stratified patient bootstrap.
- Tiny pylibjpeg 2.1.0 + pylibjpeg-libjpeg 2.4.0 wheels (approximately 3.4 MB)
  installed only in a new workspace target. The external XRV environment,
  checkpoint, original DICOMs, acquired cohort and old thresholds are unchanged.
- All 50 header-only presentation checks pass: 35 first-window paths and
  15 first-VOI-LUT paths; 24 explicit presentation shapes and 26 photometric
  fallbacks. No real PixelData was decoded or visually inspected.
- 21 new invented-fixture tests pass, including a newly constructed 2x2
  JPEG Lossless SV1 stream with known four pixels. No patient/image dataset
  supplies the codec test. The full V1.2 suite passes **1,783 tests**.
- The initial full-suite invocation loaded pydicom in older header tests before
  the JPEG plugin path was available, caching unavailable plugins (15 new-fixture
  errors). Setting both dependency paths at interpreter startup fixes the
  invocation; no consumed source or cohort was changed. Production script uses
  those explicit startup paths. Isolated tests and fresh-process preparation
  had already passed; actual scoring was never attempted.
- Missing explicit pixel/inference approval is rejected BEFORE input/model
  access. The batch script passes `bash -n`; no sbatch command was executed.

Frozen protected plan:

```text
artifacts/protected/tricompose_v1_2/real_validation/ricord_xrv_plans/display50_12666569_001/
  plan.json
  manifest.json
```

Manifest SHA256:
`3732ed1592bed8dea7aa479f278bd14ddfb3d8a4304356039ba84080651bd742`.

All plan artifact/source/dependency hashes, environment versions and private
permissions passed checks (2770 directories, 0660 files, project group exposed
as 65534). This plan consumes the new code/tests/protocol: **do not edit them**
in place for a future experiment. Record a separate unconsumed version instead.

Full test invocation (XRV interpreter is read-only):

```bash
TMPDIR=/project2/ruishanl_1185/inference_3mod/.tmp \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.tmp/ricord_header_dependencies_12666569_001:.tmp/ricord_pixel_dependencies_12666569_001:src:TriCompose-v1.0/src:TriCompose-v1.1/src:experiments/synehrgy_v2/src:TriCompose-v1.2/real_validation:TriCompose-v1.0/eval/report_v1_1 \
/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python \
  -m unittest discover -s TriCompose-v1.2/tests -q
```

## Proposed job (approval pending)

`TriCompose-v1.2/slurm/50_ricord_xrv50_v100.sbatch` contains the complete job.
Account ruishanl_1185, partition gpu, one node/task, **one V100, 2 CPUs,
12 GiB host RAM, 10-minute allocation cap**. Actual runtime is not measured;
the cap is not a runtime/queue-time guarantee. A current `noderes -f -g` check
showed idle/mixed V100 nodes; several A40 nodes were drained. Do not pin a node
or interpret free resources as a guaranteed start. Refresh before submission.

After explicit approval only: render the fixed 50 DICOMs and call the existing
frozen XRV once per eligible display, at most 50 classifier calls. No alternate
generator or seed. Source metadata/pixels, generated displays and case scores
remain protected. Native logs stay in the private per-job runtime; the Slurm
public stream contains sanitized status only.

Output will be a NEW run (not created by preparation):

```text
artifacts/protected/tricompose_v1_2/real_validation/ricord_xrv_pilots/ricord_xrv50_<job_id>/
  displays/case_NNN.png
  scores.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Main diagnostic is the exact Lung Opacity head, default threshold 0.5. The old
max(Lung Opacity, Infiltration) is a secondary legacy-proxy comparison, including
the old 0.642907085 weak-reference threshold as explicitly unvalidated transport.
Do not report this threshold against the exact head or call op-normalized scores
calibrated probabilities. No fitting, Brier/ECE or general pneumonia gold.
Derived unanimity, small/balanced/ordered selection bias, unresolved reader
identity/training overlap and clinically unverified display transforms remain
limitations. The engineering checks are NOT new clinical-performance results.
