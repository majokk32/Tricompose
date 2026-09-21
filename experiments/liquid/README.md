# Liquid frozen EHR-prompt CXR experiment

This model-specific experiment implements:

```text
protected structured EHR facts
  -> shared deterministic radiology-style text
  -> frozen Liquid V1 7B text-to-image path
  -> synthetic 512 x 512 CXR candidate
```

Liquid is a general multimodal generator, not a direct structured-EHR CXR
model.  The generation path never reads a matched real CXR or source report.
Protected runs are stored under `artifacts/protected/liquid/runs/`; cache,
temporary files, and sanitized Slurm logs use `.cache/liquid/`, `.tmp/liquid/`,
and `logs/slurm/liquid/`.

The shared EHR serialization and staging contract live in
`src/tricompose/ehr_prompt_cxr/`.  Liquid-specific load-once batched runtime and
adapter code remain in this directory.

Slurm entry points:

1. `slurm/00_smoke2_debug_a40.sbatch`: two-case, batch-two A40 smoke test.
2. `slurm/01_generate50_gpu_a40.sbatch`: 50 cases with batch size four.

The two-case smoke run completed successfully with 512 x 512 outputs.  The
50-case job has not been submitted.
