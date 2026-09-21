# TriCompose workspace rules

These instructions apply to the entire `inference_3mod` workspace.

## CARC execution

- Never run training, model inference, `torchrun`, or other GPU/heavy computation on a login node.
- Heavy computation must run through Slurm.
- Before any `sbatch` submission, show the complete batch script and resource request to the user and obtain explicit approval.
- Lightweight source inspection, syntax checks, and filesystem metadata checks are allowed on the login node.

## Data privacy

- Never display, inspect, summarize, transmit, or place in public logs any raw
  patient-level MIMIC EHR row, identifier, source radiology report, source CXR,
  or real target artifact.
- Synthetic model outputs may be inspected and summarized only when the user
  explicitly requests it. This includes outputs from a real-anchor run, but it
  does not authorize opening the raw source inputs or real targets.
- A user-approved Slurm job may consume patient-level inputs internally for
  model computation. Its public output must contain only sanitized status,
  fixed tensor/image dimensions, runtime, memory use, and cryptographic hashes.
- Patient-derived inputs and outputs must live below `artifacts/protected/`.
  Access is restricted to the CARC project access group `ruishanl_1185`;
  membership in that group is managed by the CARC project owners. The current
  collaboration request explicitly confirms access for `yikeyang` and
  `xhuang90`.
  Protected directories must use group `ruishanl_1185` and mode `2770`;
  protected files must use group `ruishanl_1185` and mode `0660`. On the CARC
  NFS export, the server may expose this project-group ownership as `nobody` in
  file metadata; the project mount/traversal boundary and group mode remain the
  access boundary. Do not grant access outside this project. Group access does
  not authorize copying raw inputs or real targets into chat, public logs, Git,
  or external services.
- Any weekly report or comparison containing generated report text or rendered
  synthetic images must also remain below `artifacts/protected/` with the same
  project-group ownership and permissions. Do not copy such content into an
  ordinary README, Git history, public log, or chat response.
- Do not use patient identifiers in output filenames. Select records by an opaque integer index and keep the internal source key out of manifests and logs.
- Do not load a real target CXR or target report for the Phase-0 generation smoke test.

## Filesystem boundaries

- Only modify files below `/project2/ruishanl_1185/inference_3mod`.
- Treat every external model, environment, checkpoint, and dataset directory as read-only.
- Redirect caches, temporary files, bytecode, and framework outputs into this workspace.
- Refuse to overwrite an existing run directory; use a new opaque run ID instead.
