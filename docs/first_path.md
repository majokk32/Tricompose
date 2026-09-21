# First path: real-anchor longitudinal smoke test

## Scope and privacy contract

This path tests plumbing only:

```text
prepared real-derived longitudinal condition + previous real CXR
  -> EHRXDiff
  -> protected synthetic CXR
  -> UniDisc
  -> protected synthetic report
```

The EHRXDiff condition is accurately labeled
`previous_report_hash_plus_interval_ehr_hash_plus_previous_cxr`. The prepared
1536-dimensional table values were produced with a local hashing placeholder,
not a validated clinical text encoder. A successful job therefore proves that
the interfaces run; it does not establish clinical quality.

The adapters never load the real target CXR, target report, or patient-level
metadata CSV. The previous CXR is located internally within the prepared image
copy by filename. Third-party
stdout/stderr is discarded because the upstream code can print prompts,
generated text, and internal metadata. Public Slurm logs contain sanitized JSON
status only. Generated artifacts are created below `artifacts/protected/`.

## Read-only assets

| Component | Location | Size / role |
|---|---|---|
| EHRXDiff environment | `/home1/yikeyang/envs/ehrxdiff_env` | Python 3.10 environment |
| EHRXDiff checkpoint | `/project2/ruishanl_1185/EHRXDiff_baseline/local_data/ehrxdiff_outputs/train_logs/2026-06-05T00-39-27_plusreport_intervalehr_1k_conditioner_only_optfreeze_lr1e6_step200/checkpoints/last.ckpt` | about 6.28 GB |
| Cheff autoencoder | `/project2/ruishanl_1185/EHRXDiff_baseline/local_data/ehrxdiff_work/trained_models/cheff_autoencoder.pt` | about 291 MB |
| CLIP bootstrap | `/home1/yikeyang/.cache/clip/ViT-B-32.pt` | about 354 MB; explicitly pinned read-only to prevent downloads |
| Prepared test conditions | `.../tab/mimiciv-cxr-filtered_test_openai.h5` | opaque-index selection |
| Previous-CXR copy | `.../artifacts/intervalehr_1k_images/images` | input images; read only |
| UniDisc environment | `/project2/ruishanl_1185/Baselines/UniDisc_baseline/.venv` | Python 3.11 environment |
| UniDisc checkpoint | `/project2/ruishanl_1185/Baselines/UniDisc_baseline/ckpts/unidisc_interleaved/unidisc_interleaved.pt` | about 5.63 GB |
| UniDisc tokenizer | local read-only `NousResearch/Llama-2-7b-hf` snapshot under `/home1/yikeyang/.cache/huggingface/hub/` | tokenizer files only; no network access |
| UniDisc image tokenizer | `/project2/ruishanl_1185/Baselines/UniDisc_baseline/ckpts/vq_ds16_t2i.pt` | about 288 MB; resolved by upstream config |

No checkpoint or dataset is copied into this repository.

## Adapter contract

| Stage | Input | Protected output |
|---|---|---|
| 1: EHRXDiff | opaque split index; internal H5 condition and previous frontal CXR | `stage1_ehrxdiff/generated_cxr.png`, `manifest.json` |
| 2: UniDisc | Stage-1 protected PNG | `stage2_unidisc/generated_report.txt`, `manifest.json` |

Output filenames contain no source identifier. Existing stage directories are
never overwritten.

## Slurm resources

Both smoke jobs request:

```text
partition: debug
nodes/tasks: 1 / 1
GPU: 1 x A40 (48 GiB)
CPU: 4
host memory: 80 GiB
wall time: 1 hour
```

These are conservative first-run requests because both loaders may temporarily
hold a checkpoint and a loaded state dictionary in host memory. Peak GPU memory
is not yet measured; both jobs are expected to fit an A40, and each manifest
records the observed peak. Stage 1 uses 35 DDIM steps for smoke testing rather
than the upstream 100-step evaluation setting.

## Submission sequence

Do not run these commands until the user has reviewed and approved both batch
scripts. From the repository root, the intended dependency chain is:

```bash
stage1_job=$(sbatch --parsable \
  --export=ALL,TRICOMPOSE_RUN_ID=real_smoke_001,TRICOMPOSE_SAMPLE_INDEX=0 \
  slurm/01_ehrxdiff_real_anchor.sbatch)

sbatch --dependency="afterok:${stage1_job}" \
  --export=ALL,TRICOMPOSE_RUN_ID=real_smoke_001 \
  slurm/02_unidisc_cxr_to_report.sbatch
```

Use a new opaque `TRICOMPOSE_RUN_ID` for every attempt. Do not inspect the PNG,
report, or private manifests through this assistant. Job success can be checked
using the sanitized Slurm JSON line and file permissions.
