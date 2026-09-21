# CheXGenBench Sana frozen baseline

The unmodified official repository is checked out at workspace root:
`CheXGenBench/`, commit `cc7e91ebcee946836090acb4c399e4f3912280ac`.
Its patient-derived `MIMIC_Splits/` and image `assets/` are deliberately excluded
from the sparse checkout.

The frozen public model is pinned to
`raman07/CheXGenBench-Models-Sana-e20` revision
`6c8989d92e75fbcfc91a82c3570389293867ad6a`. The adapter preserves the official
Sana settings: FP16, 20 inference steps, guidance scale 4.5, and the pipeline's
1024x1024 output path.

Scientific signature:

`structured EHR -> deterministic radiology-style text -> frozen Sana -> CXR`

This is not direct structured-EHR-conditioned CXR generation. The Phase-0
adapter never loads a matched real CXR or source report. Protected prompts and
images are stored only under `artifacts/protected/chexgenbench_sana/runs/`.

The model repository does not currently declare an explicit weight license in
its model card. Treat the checkpoint as research-only pending clarification.

## Fixed 50-case pilot

The approved pilot is defined by the non-overwriting run ID
`cgb_sana50_fixed_20260804_002`. It takes the first 50 opaque cases from the
already fixed protected selection `medim50_20260803_001`. The prompt versions,
per-case seeds, official Sana revision, model-weight hashes, generation-record
hashes, and image hashes are written to the protected `frozen.json`. Later
evaluation must reject any mismatch; the run must never be regenerated in
place or selectively filtered after visual review.

Every neural component is inference-only: `eval()`, all parameters have
`requires_grad=False`, and generation uses `torch.inference_mode()`. The same
policy applies to RadDINO, BioViL-T, and Inception metric encoders.

## Official metric protocol

`official_metrics.py` preserves CheXGenBench's official metric encoders,
preprocessing, and definitions for FID, RadDINO FID, KID, RadDINO KID,
Inception Score, RadDINO PRDC, BioViL-T image-text alignment, and Frechet
Radiomics Distance. It fixes upstream command-line defects and replaces the
upstream FRD temporary image copies with in-memory feature extraction.

The matched 50 real CXRs are read only inside the approved evaluation job. They
are never copied, serialized, displayed, or named in public logs. Distribution
metrics compare fixed synthetic and real cohorts. Inception Score and BioViL-T
alignment are also reported separately for synthetic and real images. Because
the cohort contains only 50 cases, all results are pilot estimates and are not
directly comparable to CheXGenBench's full-test leaderboard.
