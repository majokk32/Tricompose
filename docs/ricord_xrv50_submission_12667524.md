# RICORD-1C frozen XRV diagnostic: submission 12667524

Submitted 2026-10-05 after the complete script/resource request was shown and
the user explicitly replied `go`. No resource, dataset, code or threshold
change was made between approval and submission.

- Job: 12667524; account ruishanl_1185; partition gpu.
- One node/task, one V100, two CPUs, 12G host memory, 00:10:00 allocation cap.
- Script: `TriCompose-v1.2/slurm/50_ricord_xrv50_v100.sbatch`.
- Exact approved/submitted script SHA256:
  `5a1b14172957e02e9e9e30d707b609004932133d277f37ef0de044bd7a870bb3`.
- Frozen display/scorer plan manifest SHA256:
  `3732ed1592bed8dea7aa479f278bd14ddfb3d8a4304356039ba84080651bd742`.
- Fixed image acquisition manifest SHA256:
  `de8e303594a3dba71da36185b4272d516b577a0bb5ed567720fdf517b6b95284`.
- Fixed source cohort: 50 images, 25 positive + 25 negative derived opacity
  references, 50 distinct patients; not general pneumonia gold.

Pre-submission source/artifact/dependency hashes, environment versions,
checkpoint, unchanged old threshold bundle and private directory modes passed.
The refreshed `noderes -f -g` listed idle/mixed V100s; drained A40s were not
requested. Available-looking resources do not guarantee scheduling priority.
At the first check, 2026-10-05 18:39:25 UTC, Slurm reported **PENDING / Priority**,
Elapsed 00:00:00. This is not a measured inference runtime or a failed job.
No resubmission, node pinning, cancellation or unrelated-job change occurred.

Expected NEW protected output (not yet a completion claim):

```text
artifacts/protected/tricompose_v1_2/real_validation/ricord_xrv_pilots/ricord_xrv50_12667524/
  displays/case_NNN.png
  scores.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Private native worker logs:
`artifacts/protected/tricompose_v1_2/job_runtime/ricord_xrv50_12667524/`.
Sanitized Slurm status logs:
`artifacts/protected/tricompose_v1_2/slurm_logs/tri_v12_ricord50_12667524.out`
and the matching `.err` file. Do not open patient pixels, source identifiers
or native logs directly into chat. Use aggregate-only validated summaries or
sanitized diagnostics. Submitted script/consumed plan/code remain immutable.

Evaluation reports exact Lung Opacity head separately from the secondary
legacy max(Lung Opacity, Infiltration) score/old weak-threshold transport.
No threshold fitting, probability calibration, training or generator calls.
Completion requires original-denominator accounting, artifact/source/dependency
hash checks and protected permission checks. Partial results remain partial;
no failed case is replaced and paper-primary eligibility remains false.

## Allocation confirmed

A subsequent Slurm check confirmed RUNNING on `d13-04`, with
`gres/gpu:v100=1`, two CPUs and 12G memory. At elapsed 00:01:25, a filesystem
metadata-only check counted 28 PNG files in the new temporary run. This is
not yet a committed result, a claim of 28 successful scores, or image-content
inspection. No submission/resource/cohort change was needed.

## Completion confirmed

Slurm subsequently reported COMPLETED, exit 0:0, elapsed 00:02:21 on `d13-04`.
All 50 images were scored, with zero failures. Saved aggregate statistics,
artifact/source/dependency hashes, lineage and protected permission checks
passed. Output manifest SHA256:
`dc14a2ba1aed3823f0752a73dda297e874f638b31644d2992227eb3f1cf697a9`.
Detailed aggregate-only results and limitations are in
`docs/ricord_xrv50_result_12667524.md`. The pending/running checks above are
historical observations, not current status. No resubmission was required.
