# Frozen BioViL-T expert benchmark preparation — 2026-10-06

Status: **ready for explicit input/execution approval; not run**.

The image-path preflight initially found no exact file under seven standard
roots. The subsequent existing matched-data metadata index found **43 exact
local source images** with unchanged complete patient/study/DICOM suffixes.
This distinguishes a root-path mismatch from a genuinely absent image. There
were no permission-denied results in this scoped stat audit; it does not claim
availability of all images on the server.

The 43 images cover **44 reference/section anchors, 132 candidate-report pairs,
130 unique candidate texts and 34 MIMIC patient groups**. All 624 original
pairs remain in the plan; unavailable data are not replaced by another image
or zero score. No CheXpert/unresolved-source image is included. No image bytes
were hashed/opened, no pixels decoded, no model executed, no source copied,
no download performed and no new Slurm task submitted in preparation.

The source index check consumed the linkage CSV's path metadata internally,
not EHR diagnoses/labs/vitals or reports. Operational paths stay in a separate
protected resolver. Public artifacts contain only opaque IDs, hashes,
availability and aggregate counts.

## Sealed plan and pre-run audit

```text
artifacts/protected/tricompose_v1_2/
  radeval_image_linkage_runs/exact_index_12714150_001/
  radeval_image_benchmark_plans/biovil_cpu_12714150_001/
  radeval_image_benchmark_plan_audits/biovil_cpu_12714150_001/
```

Preparation manifest SHA256:
`6f41133e3066724ecfdbdd8d0fb425c193559c82ea7355e5380b584b6549df96`.

Independent audit SHA256:
`b2028ecd06871bdd2838e10368cc606bf6c2cd98a42550ac0676bb097f493ef3`.

The audit rechecks source/code pins, all 624 attempted pair availability rows,
129 independent source stat/identity checks, exact count/group/text joins,
protected modes and absence of raw source keys in the plan/manifest. **2,390
V1.2 tests pass**, including 25 new invented-fixture image-linkage/evaluation
tests. Script syntax and Git whitespace checks pass. None of this is clinical
qualification or pixel-computation authorization.

## Exact next computation

Entry point: `TriCompose-v1.2/slurm/60_radeval_biovil_existing_cpu.sh`.

- Actual existing Slurm allocation 12714150, 4 CPUs / 32 GiB; two worker
  threads, zero GPUs, 600-second process cap. No sbatch submission.
- Already installed frozen BioViL-T image/text weights and official vendor
  hashes verified. No training, new weights, threshold or score-weight fitting.
- Model inputs: the 43 linked MIMIC images and author candidate reports only.
  Ground-truth report and expert counts are not fed into BioViL-T.
- Expert-count correlation, whole-patient cluster intervals, fixed-anchor
  report selection versus uniform random choice; all three existing RadGraph
  scores evaluated on the identical available cohort.
- Clinical evaluation remains reference-count based, not new image-expert
  adjudication; training overlap and synthetic-domain transport unverified.
- Offline assets, blocked inference sockets, no external clinical API, no
  real image rendering, unchanged historic synthetic scores/choices.

See the [fixed protocol](radeval_image_benchmark_protocol.md). The complete
script and input/resource scope must be shown and approved before running. If
the current allocation expires, this script refuses rather than running on a
login node; a new Slurm submission needs a separately shown/approved batch.
