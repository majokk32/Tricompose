# MeDiM EHR-prompt real-anchor experiment

This directory contains the isolated, no-training experiment for:

```text
real structured EHR
  -> deterministic radiology-style prompt
  -> frozen MeDiM
  -> generated CXR
```

The matched real CXR is an evaluation reference only. It is not loaded by the
generation job and is never used to construct the prompt.

## Privacy boundary

- Source code, configuration, and Slurm scripts live in this directory.
- All selected EHR facts, prompts, generated images, and case-level scores live
  under
  `artifacts/protected/medim/runs/<opaque_run_id>/`.
- Case filenames use only opaque sequential integers.
- External model, checkpoint, environment, and dataset directories are
  read-only.
- Jobs print only sanitized status, counts, dimensions, runtimes, and hashes.

## Stages

1. `prepare_cases`: select 50 distinct test patients and serialize EHR fields.
2. `medim_generate`: condition frozen MeDiM on each prompt, preserve all text
   tokens, and sample only the masked image-token region.
3. `consistency`: apply the same frozen TorchXRayVision labeler and the same
   positive-EHR support rule to the generated CXR and the matched real CXR.
   Real references are opened only in this evaluation stage.
4. `aggregate`: write protected cohort-level means, medians, operating-point
   support rates, paired generated-minus-real gaps, and bootstrap confidence
   intervals without exposing patient-level content in Slurm logs.
5. `official_image_metrics`: compute the MeDiM paper's image-generation metrics,
   FID and Inception Score, on the 50-image pilot. FID compares the generated
   cohort with the matched real cohort. IS is computed for both cohorts. Only
   aggregate results are written.

The primary score is the mean, over positive radiographically observable EHR
facts, of the maximum matching TorchXRayVision confidence. Missing or zero EHR
fields are `unknown`, not imaging negatives. Cases with no positive observable
fact are `not_applicable`. Support-device facts condition MeDiM but are excluded
from the primary score because this checkpoint does not provide device-specific
outputs.

This is a conditional real-anchor reconstruction experiment. A high score does
not establish prospective EHR-to-CXR prediction, and the matched real CXR is
never an input to MeDiM.

## Prompt serialization

The serializer uses ordinary Llama/MeDiM text tokens, not a new EHR tokenizer:

```text
The image is a radiograph of the chest, showing the thoracic cavity structures.
Portable frontal AP chest radiograph.
Clinical history: an <age-bin> <sex> patient.
Clinical indications include concern for <positive observable diagnoses>.
Known support devices include <positive device facts>.
Pre-imaging clinical context includes <binned pre-CXR measurements>.
```

It is deterministic, contains no LLM/API call, and never reads the matched CXR
or report. Zero and missing fields are omitted. Generation fails instead of
silently truncating if the prompt exceeds the configured token budget.

## Temporal interpretation

The dataset builder selects WBC, BNP, SpO2, and respiratory rate as the latest
available value at or before the CXR study time. Diagnosis and procedure flags,
however, are admission-level ICD aggregates and can include codes assigned
after the study. Accordingly, this run is an explicit-condition reconstruction
test, not a leakage-free forecasting experiment.

TorchXRayVision's `all` checkpoint was trained on a mixture that includes
MIMIC-CXR. It is suitable for this engineering/calibration check but must not be
presented as an independent final clinical evaluator.

The official MeDiM image metrics evaluate distributional image fidelity and
diversity, not EHR conditioning or clinical contradictions. The paper evaluates
the full MIMIC-CXR test set and the repository's paired FID config uses 4,096
samples. Results from this 50-image pilot are therefore deliberately marked as
small-sample estimates and are not directly comparable to the paper's reported
FID 16.60 and IS 2.87.

## Slurm entry points

- `slurm/01_prepare_cases.sbatch`: CPU-only protected case selection.
- `slurm/02_medim_generate_array.sbatch`: frozen MeDiM `mask_image_only`
  generation with the official 100 sampling steps. The 50 cases are split into
  five ten-case tasks, each capped at two hours on an A100-80GB GPU.
- `slurm/03_ehr_cxr_consistency.sbatch`: frozen TorchXRayVision scoring of both
  real and generated images, followed by protected aggregation.
- `slurm/04_official_image_metrics.sbatch`: P100 evaluation of 50 generated and
  50 matched real CXRs with PyTorch-FID-compatible legacy Inception features and
  the official MeDiM `is_score.py` protocol. The first invocation caches the
  approximately 91 MB FID Inception checkpoint below this workspace.

No Slurm job is submitted automatically by this repository.

## Official paired text-to-image diagnostic

`slurm/12_medim_official_paired_t2i2_a40.sbatch` is a focused two-condition
diagnostic. It composes the upstream `large_scale_train` and
`paired_standalone_fid_eval` configs and calls the upstream
`Diffusion.sample_for_fid` text-conditioned generation method:

1. a fully synthetic, in-domain radiology report;
2. an existing protected deterministic EHR prompt with the chest-radiograph
   facts reserialized into concise report-like clinical history, indication,
   and support-device sentences. Labs and vitals are deliberately omitted from
   this diagnostic because MeDiM is report-conditioned. The upstream
   `ValDataset` then adds its chest-radiograph prefix once.

Both conditions use the same seed, CFG 5, 256 text positions, 1,024 image
positions, and 1,024 effective image sampling updates. The neutral staged JPEG
only satisfies the paired-file dataset interface; all image positions are
masked and it is not an image condition.

The current upstream commit lacks `Diffusion.get_cond_dict` and
`Diffusion.get_vae`, and its tracked `update_batch` expects GPU-resident input
tensors. The workspace adapter supplies only those compatibility operations;
tokenization, sampling, CFG, and decoding remain in upstream code. It imports
the tracked `model.py`/`model_eval.py` path rather than the externally modified
`model_inf.py`/`model_eval_inf.py` files.

## Official mask-image-only adapter check

`configs/real_anchor_10_official.yaml` is a separate ten-case diagnostic run.
It reproduces the repository's previously run official report-to-image command:
`eval.task=mask_image_only`, 256 padded report/text positions are preserved,
1,024 image positions are masked, CFG is 5, and MaskGIT runs for 100 steps.
FID and Inception Score are separate downstream evaluations and are not part of
this generation adapter.

The run remains EHR-prompt-conditioned and never loads a source CXR or source
report. Its Slurm entry points are `05_prepare_official10.sbatch` and
`06_medim_official10_array.sbatch`.
