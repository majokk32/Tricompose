# TriCompose storage layout

TriCompose uses model names as the first directory below every runtime storage
root.  External repositories, environments, datasets, and checkpoints remain
read-only and are not included here.

Model-specific experiment code follows the same convention:

```text
experiments/medim/
experiments/roentgen_v2/
experiments/unidisc/
experiments/liquid/
```

The small EHR-to-radiology-text contract shared by multiple generators lives
in `src/tricompose/ehr_prompt_cxr/`, not in a combined model experiment.

## Protected artifacts

Patient-derived inputs and synthetic outputs are stored only below
`artifacts/protected/`. On CARC, protected directories use project-group mode
`2770` and protected files use mode `0660`, matching the workspace privacy
contract in `AGENTS.md`.

```text
artifacts/protected/
├── medim/
│   └── runs/<run_id>/
├── roentgen_v2/
│   ├── runs/<run_id>/
│   ├── evaluations/<run_id>/
│   ├── candidate_sweeps/<run_id>/
│   └── candidate_selections/<run_id>/
├── unidisc/
│   └── runs/<run_id>/
└── liquid/
    └── runs/<run_id>/
```

An existing run directory is never reused or overwritten.

## Runtime storage

```text
.cache/<model>/       downloaded or framework-managed cache
.tmp/<model>/         per-job temporary state
logs/slurm/<model>/   sanitized public Slurm stdout/stderr only
```

RoentGen-v2 evaluation, selector, and Python cache data remain grouped below
`.cache/roentgen_v2/` and `.tmp/roentgen_v2/` as named subdirectories.  Older
pipeline logs are classified under `ehrxdiff`, `llava_rad`, `unidisc`, and
`qwen2_5_vl` according to the model that produced the log.

Framework logs that could contain protected prompts are not written to
`logs/slurm/`; they stay inside the corresponding protected run directory.
