# PromptEHR frozen smoke test

This adapter uses the official PromptEHR v0.0.6 source and the authors' public
MIMIC-III checkpoint. It does not train or fine-tune the model.

## Scientific scope

- Paper: *PromptEHR: Conditional Electronic Healthcare Records Generation with
  Prompt Learning*, EMNLP 2022.
- Authors: Zifeng Wang and Jimeng Sun, University of Illinois Urbana-Champaign.
- Architecture: BART-base with learned numerical and categorical demographic
  prompt embeddings.
- Public checkpoint schema: MIMIC-III longitudinal diagnosis, procedure, and
  medication codes. The downloadable checkpoint does not generate labs.
- Input contract in the released implementation: a structured seed patient.
  The generator keeps a random subset of seed events and imputes the remainder
  for the same number of visits. It is therefore not a pure BOS-only cold-start
  generator.

The smoke test selects ten prompts deterministically from the official public
synthetic demo cohort. No real MIMIC records are read. Every model parameter is
frozen before generation.

The dedicated compatibility environment uses the legacy generation branch kept
in the official v0.0.6 source. This avoids the incompatible generation helper
signature introduced after the checkpoint-era Transformers release. Exact
direct dependencies are recorded in `requirements-lock.txt`.

## Lightweight evaluation

The run reports structure validity, visit counts, modality coverage, exact
duplicates, event diversity, prompt overlap, and output novelty. These are
engineering smoke metrics only. They do not establish fidelity to either
MIMIC-III or MIMIC-IV, and no downstream evaluation is run.

Outputs remain under `artifacts/protected/promptehr/runs/<run_id>/` with private
permissions. Public Slurm logs contain only sanitized aggregate status.
