#!/bin/bash
#SBATCH --job-name=medim_native10
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/medim/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/medim"
OFFICIAL_REPO=/project2/ruishanl_1185/MeDiM/MeDiM_original_infer
PYTHON=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/python
ACCELERATE=/project2/ruishanl_1185/MeDiM/MeDiM_EHR/medim_ehr_env/bin/accelerate
SOURCE_RUN_ID="${1:?usage: sbatch 09_medim_official_native10_debug_a40.sbatch SOURCE_RUN_ID OUTPUT_RUN_ID}"
OUTPUT_RUN_ID="${2:?usage: sbatch 09_medim_official_native10_debug_a40.sbatch SOURCE_RUN_ID OUTPUT_RUN_ID}"
RUN_DIR="$WORKSPACE/artifacts/protected/medim/runs/$OUTPUT_RUN_ID"

TASK_KEY="${SLURM_JOB_ID}"
TASK_CACHE="$WORKSPACE/.cache/medim/$TASK_KEY"
TASK_TMP="$WORKSPACE/.tmp/medim/$TASK_KEY"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"

export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$OFFICIAL_REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TRANSFORMERS_CACHE="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export UNIDISC_DIR="$OFFICIAL_REPO"

"$PYTHON" -m medim_ehr_prompt.official_native_io stage \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  >"$TASK_TMP/stage_status.json"

OFFICIAL_LOG="$RUN_DIR/logs/official_inference.log"
cd "$RUN_DIR/work"

"$ACCELERATE" launch --num_processes 1 "$OFFICIAL_REPO/inference.py" \
  +experiments='[large_scale_train,eval_model]' \
  mode=eval \
  eval.task=mask_image_only \
  trainer.load_from_state_dict=/project2/ruishanl_1185/MeDiM/pretrained_models/medim_ckpts/model.safetensors \
  model.vqgan_config=/project2/ruishanl_1185/MeDiM/pretrained_models/chameleon/vqgan.yaml \
  model.vqgan_ckpt=/project2/ruishanl_1185/MeDiM/pretrained_models/chameleon/vqgan.ckpt \
  model.llama_ckpt=/project2/ruishanl_1185/MeDiM/pretrained_models/Llama-2-7b-hf \
  model.liquid_ckpt=/project2/ruishanl_1185/MeDiM/pretrained_models/Liquid_V1_7B \
  data.data_path_dir_train="$RUN_DIR/dataset/path" \
  data.data_path_dir_val="$RUN_DIR/dataset/path" \
  data.data_mimic_dir_train="$RUN_DIR/dataset/mimic" \
  data.data_mimic_dir_val="$RUN_DIR/dataset/mimic" \
  loader.eval_batch_size=1 \
  loader.batch_size=1 \
  trainer.limit_val_batches=10 \
  eval.num_sample_batches=10 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  eval.compute_standalone_mauve=false \
  eval.unconditional_fid=false \
  eval.txt_conditional_fid=false \
  eval.calculate_clip_score=false \
  trainer.compile=false \
  trainer.fsdp=false \
  trainer.output="$RUN_DIR/native_output" \
  sampling.steps=100 \
  hydra.run.dir="$RUN_DIR/hydra" \
  hydra.job.chdir=false \
  +output_dir="$RUN_DIR/work" \
  >"$OFFICIAL_LOG" 2>&1

chmod 600 "$OFFICIAL_LOG"

"$PYTHON" -m medim_ehr_prompt.official_native_io finalize \
  --output-run-id "$OUTPUT_RUN_ID" \
  >"$TASK_TMP/finalize_status.json"

GENERATED_COUNT=$(find "$RUN_DIR/cases" -type f -name generated_cxr.png -printf '.' | wc -c)
METADATA_COUNT=$(find "$RUN_DIR/cases" -type f -name generation.json -printf '.' | wc -c)
printf 'status=completed generated_count=%s metadata_count=%s\n' \
  "$GENERATED_COUNT" "$METADATA_COUNT"
