# LLaVA-Rad CXR-to-report experiment

This experiment runs the existing frozen LLaVA-Rad deployment over the same
immutable set of 50 Sana-generated synthetic CXRs used by the MAIRA-2 and
CXRMate-single report experiments.

## Scientific contract

- Input: one current synthetic frontal CXR.
- Output: a radiology `FINDINGS` section.
- Structured EHR, prior CXR, prior report, and real target artifacts are not
  loaded.
- The Vicuna-7B base, LLaVA-Rad LoRA/non-LoRA weights, and BioMedCLIP-CXR
  vision encoder are loaded read-only and kept frozen.
- The prompt matches the upstream report-generation wording:
  `Given the chest X-ray image, describe the findings in the image:`
- Decoding matches the upstream deterministic evaluation configuration:
  temperature 0, one beam, and at most 256 new tokens.

## Fixed source

- CXR model: `sana`
- Frozen CXR run: `cgb_sana50_fixed_20260804_002`
- Case IDs: opaque `case_000` through `case_049`

All generated text and manifests remain below
`artifacts/protected/llavarad_report/` with owner-only permissions.

## Slurm

`slurm/01_report50_debug_a40.sbatch` requests one A40, 8 CPUs, 64 GiB RAM,
and 15 minutes on the debug partition. It must not be submitted without the
user's explicit approval of the complete script and resource request.
