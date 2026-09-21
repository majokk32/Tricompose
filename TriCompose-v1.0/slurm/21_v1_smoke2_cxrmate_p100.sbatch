#!/bin/bash
#SBATCH --job-name=tri_v1_cxrmate_all
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
MODEL_DIR=/project2/ruishanl_1185/Baselines/hf_cache/hub/models--aehrc--cxrmate-single-tf/snapshots/84dfcba8125c9b9296bfc1ee317749788b6b59b4
CXR_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/cxr_candidates"
REPORT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/report_candidates"
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1/cxrmate_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1/cxrmate_${SLURM_JOB_ID}"

for required in "$PYTHON" "$MODEL_DIR" \
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
export HF_MODULES_CACHE="$TASK_CACHE/huggingface/modules"
export TORCH_HOME="$TASK_CACHE/torch"
export CUDA_CACHE_PATH="$TASK_CACHE/cuda"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

"$PYTHON" "$WORKSPACE/tools/run_tricompose_v1_report.py" \
  --cxr-run "$CXR_ROOT/v1_smoke2_roentgen_v2_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_sana_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_pixart_s2_20260806_001" \
  --output-root "$REPORT_ROOT" \
  --output-run-id v1_smoke2_cxrmate_allcxr_20260806_001 \
  --model-id cxrmate_single \
  --model-dir "$MODEL_DIR"
