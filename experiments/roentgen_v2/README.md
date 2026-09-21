# RoentGen-v2 frozen adapter

This experiment implements the scientifically explicit path:

`protected structured EHR facts -> deterministic radiology-style prompt -> frozen RoentGen-v2 -> synthetic CXR`

It is not direct structured-EHR-conditioned CXR generation. It never loads a
real target CXR or real target report. Prompts and outputs remain under
`artifacts/protected/roentgen_v2/runs/`.

Each generated image has a machine-readable candidate record following
`schemas/roentgen_v2_generation_v1.schema.json`, including its opaque candidate
ID, model/prompt revisions, seed, artifact hash, runtime, and model-call cost.
The future candidate graph can consume this metadata without opening the image
or protected prompt.

## Audited upstream interface

- Official code commit: `039e34b1a40a1ad58c42522e0332a93bb3930886`
- Official Hugging Face revision: `c88d2bf041a448f505fc13dc009d6991be1c9b0f`
- Official defaults preserved: BF16, guidance scale 3, 75 steps, 512x512.

The sentence above refers to the released repository demo configuration. The
RoentGen-v2 paper's large-scale dataset protocol instead reports guidance scale
4.0 with 75 steps.

## Protected evaluation

`tricompose_roentgen_v2.evaluation` evaluates an existing 50-case real-anchor
run without copying or serializing the matched real CXRs. It reports protected
aggregate and opaque-case metrics for positive EHR-fact support with an XRV
real-CXR reference, the paper's five XRV diseases, sex and age-group alignment,
paired BioViL cosine and MS-SSIM, exploratory 50-sample FID, and a deterministic
image-statistic artifact gate. The FID and real-XRV pseudo-label AUROC are
explicitly marked exploratory rather than treated as clinical ground truth.

## Candidate composition pilot

`tricompose_roentgen_v2.candidate_sweep` generates a protected 10-case grid of
four seeds at guidance scales 3 and 4. Only cases with at least one positive,
radiographically observable EHR fact are included. The paired seeds are shared
between guidance scales, producing eight candidates per case.

`tricompose_roentgen_v2.candidate_selector` then performs reference-free
selection: it rejects candidates that meet deterministic hard image-failure
criteria when a non-failing alternative exists, maximizes frozen-XRV positive
EHR support, and uses candidate ID only as a deterministic tie-break. Every
candidate score and action is stored below `artifacts/protected/`; neither the
candidate generator nor selector loads a matched real CXR or source report.

All protected RoentGen artifacts share one root:

```text
artifacts/protected/roentgen_v2/
├── runs/
├── evaluations/
├── candidate_sweeps/
└── candidate_selections/
```
- The model repository is gated and totals about 6.53 GB. Access must first be
  accepted on Hugging Face.
- The runtime is local-only (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) so
  no protected prompt is transmitted externally.

## Prompt policy

The adapter includes positive radiographically observable diagnoses and
explicitly visible devices. It omits labs, vitals, race, and mechanical
ventilation when no visible device is specified. Absence of a positive EHR fact
does not get converted into a fabricated “normal CXR” statement.

Prompt version `roentgen_v2.ehr_to_radiology_prompt.pa_findings.v2` uses the PA
view token from the released checkpoint's training distribution and serializes
positive diagnoses as explicit, concise radiology-style findings. It preserves
the same cases and seeds as the earlier AP-prompt run so the comparison isolates
the prompt change. This remains a deterministic EHR-to-text adapter followed by
a text-conditioned CXR model, not direct structured-EHR-conditioned generation.

## Prepared Slurm stages

The scripts are intentionally not submitted automatically:

1. `slurm/00_setup_env_main.sbatch`: create the pinned inference environment.
2. `slurm/01_prefetch_model_main.sbatch`: download the gated audited snapshot.
3. `slurm/02_smoke2_debug_a40.sbatch`: generate two protected smoke images.
4. `slurm/03_generate50_gpu_a40.sbatch`: generate all 50 protected candidates.
5. `slurm/05_generate50_pa_findings_debug_a40.sbatch`: generate a new 50-case
   PA/radiology-findings prompt ablation without overwriting the earlier run.
6. `slurm/07_candidate_sweep_n10_cfg34_s4_debug_a40.sbatch`: generate the
   protected 10-case, 80-candidate CFG/seed sweep.
7. `slurm/08_select_n10_xrv_debug_a40.sbatch`: select one candidate per case
   using frozen XRV and a reference-free hard quality gate.
8. `slurm/09_eval_selected_vs_fixed_n10_debug_a40.sbatch`: materialize the
   fixed CFG3/seed0 and selected candidates as separate protected runs, then
   evaluate both against matched real anchors with an identical protocol.

Every `sbatch` still requires the user's explicit approval after the complete
script and resource request have been shown.
