# Shared CXR baseline evaluation

This experiment applies a lightweight real-anchor protocol to frozen CXR
generators. It reuses the frozen TorchXRayVision disease checkpoint and disease
parameters from the prior MedIM and RoentGen-v2 evaluations.

The common metrics are XRV EHR positive support, five-label paired probability
MAE/correlation/threshold agreement/pseudo-AUROC, and a deterministic
image-statistic quality gate. BioViL, MS-SSIM, FID, age, and sex metrics are
intentionally omitted from this fast comparison. Matched real CXRs are read
only in the approved Slurm job and are never copied, serialized, displayed, or
logged.

Sana, PixArt, and RadEdit generation artifacts remain immutable. Evaluation
outputs are non-overwriting protected stages under each model's own
`evaluations/` folder.
