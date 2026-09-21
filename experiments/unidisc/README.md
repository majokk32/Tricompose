# UniDisc frozen EHR-prompt CXR experiment

This model-specific experiment implements:

```text
protected structured EHR facts
  -> shared deterministic radiology-style text
  -> frozen UniDisc interleaved checkpoint
  -> synthetic 256 x 256 CXR candidate
```

UniDisc is a general multimodal generator, not a direct structured-EHR CXR
model.  The generation path never reads a matched real CXR or source report.
Protected runs are stored under `artifacts/protected/unidisc/runs/`; cache,
temporary files, and sanitized Slurm logs use `.cache/unidisc/`,
`.tmp/unidisc/`, and `logs/slurm/unidisc/`.

The shared EHR serialization and staging contract live in
`src/tricompose/ehr_prompt_cxr/`.  UniDisc-specific runtime and adapter code
remain in this directory.

Slurm entry points:

1. `slurm/00_smoke2_debug_a40.sbatch`: two-case A40 smoke test.
2. `slurm/01_generate50_debug_a40.sbatch`: same 50-case protected source.

The first smoke run reached checkpoint loading but encountered a VAST I/O
`OSError`; the adapter now permits one bounded I/O-only load retry.  No retry
has been submitted yet.
