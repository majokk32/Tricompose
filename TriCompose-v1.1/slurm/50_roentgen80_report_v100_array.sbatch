#!/bin/bash
#SBATCH --job-name=tri_v11_rg80_rep
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0-2
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%A_%a.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%A_%a.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
REQUEST_RUN="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_requests/roentgen80_four_experts_20260813_7c91"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    MODEL_ID=maira2
    PYTHON="$WORKSPACE/maria-2/venv/bin/python"
    MODEL_DIR=/project2/ruishanl_1185/Baselines/hf_cache/hub/models--microsoft--maira-2/snapshots/795a2b1cd4a310624b4e3d14b5a23e41fd273deb
    OUTPUT_RUN_ID=roentgen80_maira2_20260813_a31e
    REQUIRED=("$PYTHON" "$MODEL_DIR" "$REQUEST_RUN/manifest.json")
    ;;
  1)
    MODEL_ID=llavarad
    PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/llavarad/bin/python
    EXTERNAL_ROOT=/project2/ruishanl_1185/SDP_for_VLM/SDP_for_VLM/llava-rad
    MODEL_DIR=/project2/ruishanl_1185/SDP_for_VLM/models/llava_rad
    MODEL_BASE=/project2/ruishanl_1185/SDP_for_VLM/models/vicuna-7b-v1.5
    BIOMEDBERT_DIR=/home1/yikeyang/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d
    OUTPUT_RUN_ID=roentgen80_llavarad_20260813_b472
    REQUIRED=("$PYTHON" "$EXTERNAL_ROOT" "$MODEL_DIR" "$MODEL_BASE" "$BIOMEDBERT_DIR" "$REQUEST_RUN/manifest.json")
    ;;
  2)
    MODEL_ID=chexagent2
    PYTHON="$WORKSPACE/CheXagent-2/venv/bin/python"
    MODEL_DIR="$WORKSPACE/CheXagent-2/checkpoints/chexagent-2-srrg-findings/9f7225fc382ddd1297ade1aa796da660237940bc"
    VISION_DIR="$WORKSPACE/CheXagent-2/checkpoints/chexagent2-xraysiglip/f0edbf5d90dba44edb7f4f96d8663537cb0749bf"
    OUTPUT_RUN_ID=roentgen80_chexagent2_fp32_20260813_c593
    REQUIRED=("$PYTHON" "$MODEL_DIR" "$VISION_DIR" "$REQUEST_RUN/manifest.json")
    ;;
  *)
    printf 'status=failed reason=invalid_array_index\n' >&2
    exit 1
    ;;
esac

for required in "${REQUIRED[@]}"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi

TASK_TAG="${MODEL_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1_1/$TASK_TAG"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1_1/$TASK_TAG"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 2770 "$TASK_CACHE" "$TASK_TMP"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE/xdg"
export HF_HOME="$TASK_CACHE/huggingface"
export HUGGINGFACE_HUB_CACHE="$TASK_CACHE/huggingface/hub"
export TRANSFORMERS_CACHE="$TASK_CACHE/huggingface/transformers"
export HF_MODULES_CACHE="$TASK_CACHE/huggingface/modules"
export TORCH_HOME="$TASK_CACHE/torch"
export MPLCONFIGDIR="$TASK_CACHE/matplotlib"
export CUDA_CACHE_PATH="$TASK_CACHE/cuda"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "$MODEL_ID" in
  maira2)
    "$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_report.py" \
      --request-run "$REQUEST_RUN" \
      --output-root "$OUTPUT_ROOT" \
      --output-run-id "$OUTPUT_RUN_ID" \
      --model-id "$MODEL_ID" \
      --model-dir "$MODEL_DIR"
    ;;
  llavarad)
    "$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_report.py" \
      --request-run "$REQUEST_RUN" \
      --output-root "$OUTPUT_ROOT" \
      --output-run-id "$OUTPUT_RUN_ID" \
      --model-id "$MODEL_ID" \
      --model-dir "$MODEL_DIR" \
      --model-base "$MODEL_BASE" \
      --biomedbert-dir "$BIOMEDBERT_DIR" \
      --external-root "$EXTERNAL_ROOT" \
      --runtime-dir "$TASK_TMP/model_runtime"
    ;;
  chexagent2)
    "$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_report.py" \
      --request-run "$REQUEST_RUN" \
      --output-root "$OUTPUT_ROOT" \
      --output-run-id "$OUTPUT_RUN_ID" \
      --model-id "$MODEL_ID" \
      --model-dir "$MODEL_DIR" \
      --vision-dir "$VISION_DIR" \
      --precision float32
    ;;
esac
