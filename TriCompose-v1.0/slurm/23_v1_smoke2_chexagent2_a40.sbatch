#!/bin/bash
#SBATCH --job-name=tri_v1_chexag_all
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/CheXagent-2/venv/bin/python"
MODEL_DIR="$WORKSPACE/CheXagent-2/checkpoints/chexagent-2-srrg-findings/9f7225fc382ddd1297ade1aa796da660237940bc"
VISION_DIR="$WORKSPACE/CheXagent-2/checkpoints/chexagent2-xraysiglip/f0edbf5d90dba44edb7f4f96d8663537cb0749bf"
CXR_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/cxr_candidates"
REPORT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/report_candidates"
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1/chexagent2_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1/chexagent2_${SLURM_JOB_ID}"

for required in "$PYTHON" "$MODEL_DIR" "$VISION_DIR" \
  "$CXR_ROOT/v1_smoke2_roentgen_v2_s2_20260806_001" \
  "$CXR_ROOT/v1_smoke2_sana_s2_20260806_001" \
  "$CXR_ROOT/v1_smoke2_pixart_s2_20260806_001"
do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE/xdg"
export HF_HOME="$TASK_CACHE/huggingface"
export HUGGINGFACE_HUB_CACHE="$TASK_CACHE/huggingface/hub"
export TRANSFORMERS_CACHE="$TASK_CACHE/huggingface/transformers"
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

"$PYTHON" "$WORKSPACE/tools/run_tricompose_v1_report.py" \
  --cxr-run "$CXR_ROOT/v1_smoke2_roentgen_v2_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_sana_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_pixart_s2_20260806_001" \
  --output-root "$REPORT_ROOT" \
  --output-run-id v1_smoke2_chexagent2_allcxr_20260806_001 \
  --model-id chexagent2 \
  --model-dir "$MODEL_DIR" \
  --vision-dir "$VISION_DIR"
