# Exact-opacity candidate sidecar: approved submission 12668204

Submitted 2026-10-05 after the complete actual batch script/resources were
displayed and the user explicitly replied **go**. Exactly one `sbatch` call,
job **12668204**; no duplicate submissions or unrelated job changes.

## Exact approved resources and hashes

- Script: `TriCompose-v1.2/slurm/51_cached_opacity240_v100.sbatch`.
- Script SHA256:
  `993f08b0825272a107715f775ae3ea478c92c3bce5fdadac4b76ba60318bee32`.
- Account **ruishanl_1185**, partition **gpu**, one node/task, **1 V100**,
  **2 CPUs**, **12G host memory**, time cap **00:10:00**.
- No node pinning, new generator, threshold fitting or checkpoint download.
- Slurm SubmitTime: **2026-10-05T12:20:02**, cluster local time.
- Submitted from existing CPU allocation **12666569**, node b05-05.

Fixed protected plan:
`artifacts/protected/tricompose_v1_2/candidate_opacity_plans/opacity_pool240_12666569_001/`.
Plan-manifest SHA256:
`650f129ce6b20e6e278bd937accfd348cef4e68328fcf793ab722caa55d7323a`.

Before submission, bash syntax, exact script/plan hashes, all **298 source
pins**, input artifact hashes, environment versions, complete 80/240/960
grid and private directory/file modes passed. `noderes -f -g` showed idle
and mixed V100 capacity. Availability is not a queue-time guarantee; the
initial observed scheduler status was **PENDING / Priority**.

## Frozen scope

Score all **240 existing synthetic CXR slots**, once per image slot, with the
same frozen XRV checkpoint as the completed RICORD diagnostic. Save exact
Lung Opacity and Infiltration scores separately. Exact head and same-call max
secondary readouts use the predeclared **0.5** threshold.

Append 16 `opacity_` columns to the unchanged 50-column, **960-row** candidate
CSV, keeping every original cell/order. Cached EHR opacity is unknown for all
80 anchors; report opacity is explicit for 268/960 slots. Missing evidence is
not negative/perfect agreement. No EHR relabeling, new selector, old winner
update, fault localization or clinical-accuracy claim is authorized.

Expected output, not a completion assertion:

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_runs/opacity_pool240_12668204/
  image_scores.json
  candidate_score_table.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Runtime/native diagnostics stay under the protected per-job directory. Public
logs contain only sanitized execution status. Patient source EHRs/reports/
images and benchmark case records are not read into chat. The job consumes
synthetic image files internally only; no real MIMIC target is used.

Completion must be checked using Slurm terminal status, original denominators,
source/output hashes, CSV preservation and aggregate-score recomputation.

## Completed and independently audited

Job **12668204** completed on **d13-04 / V100**, exit **0:0**, Slurm elapsed
**00:00:25**; worker runtime **20.750776 seconds**. **240/240 images scored,
240 calls, zero failures**, all **960 rows** retained. **48,000 original CSV
cells**, 300 source pins, output hashes, aggregate readouts and project-private
modes passed the independent check. No historical winner was changed.

Result-manifest SHA256:
`778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94`.
Full aggregate result and interpretation limits:
`docs/cached_opacity240_result_12668204.md`.
