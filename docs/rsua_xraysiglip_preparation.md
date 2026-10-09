# Same-cohort XraySigLIP diagnostic — preparation record

Current status: the full script/resources were subsequently displayed and
explicitly approved. Job **12654662** completed on a debug P100 in 44 seconds.
See [the result](rsua_xraysiglip_result.md). The preparation-stage record below
is historical; do not resubmit without fresh explicit approval.

The [fixed protocol](rsua_xraysiglip_protocol.md) and
`TriCompose-v1.2/real_validation/rsua_xraysiglip.py` are complete. This is a new
frozen readout on the existing developmental public cohort, not new clinical
ground truth, an untouched final test, training or regeneration.

Metadata preparation used existing CPU allocation **12645021**, with no model
calls, image/weight bytes opened, downloads, API requests or new submission.

```text
artifacts/protected/tricompose_v1_2/real_validation/rsua_siglip_plans/plan50_12645021_001/
  plan.json
  manifest.json
```

Manifest SHA256:
`ca2fb6e62237b0c66cff988425eeaf3d51e3d09b029e5ddf002f34af0a90037f`.
**1,518 tests pass**, including 24 invented-vector/approval/atomic tests.
Independent metadata replay and **47 source/runtime/config hashes**, one plan
artifact hash, all 50 opaque input IDs/file statistics and project permissions
passed. Image/weight contents were not reopened by this audit. Full model
weight SHA256 is carried from the completed six-image GPU receipt and must
match in the approved GPU allocation, not just match file size/mtime.

The plan preserves the exact 50 existing converted PNGs (25 pneumonia / 25
paper-described normal proxy); no case reselection or clinical label enrichment.
Model-facing data contain only images and all 24 fixed generic polarity pairs
(eight heads x three families / 48 endpoints). There are no class labels or old
model scores in that plan. Cohort classes and XRV references are parsed only
after prediction fsync; prior BioViL metadata is read to bind unchanged images,
never used as a generation/scoring prompt or to choose cases/templates.

Maximum **52 joint forwards / attempts**, consisting of 50 primary images and
two fixed first-image technical repeats, zero retries. Class metrics evaluate
pneumonia only; the other seven heads have no invented references. Report each
template and the fixed mean, AUROC/AP, both polarity win rates, ties/availability,
template stability/coverage, repeat differences and unqualified cross-scorer
directions. No fitted clinical threshold, probability scores or winner updates.

Complete script:
`TriCompose-v1.2/slurm/48_rsua_xraysiglip50_flexible_gpu.sbatch`.
SHA256 `aff3993322183744ab9af2c29cd9f824963385079a1eda20fc768eb3c91965d5`.
Syntax check passed. Request one P100/V100/A40/A100/L40S-compatible GPU on
debug/gpu, two CPUs, **8 GiB host RAM**, **five-minute allocation cap**, no node
binding. Earlier same-model six-image execution used about 3.42 GiB maximum
resident host RAM and 2.603 GiB allocated GPU memory; this run remains sequential
batch-one. Queue wait is separate, and free resources/reservations can change.

Only after the complete script/resource display and new explicit approval:

```bash
sbatch TriCompose-v1.2/slurm/48_rsua_xraysiglip50_flexible_gpu.sbatch
```

Completed immutable output:
`artifacts/protected/tricompose_v1_2/real_validation/rsua_siglip_pilots/rsua_siglip50_12654662/`.
Numerical completion does not establish clinical validation or authorize repair.
