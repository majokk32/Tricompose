#!/bin/bash
#SBATCH --job-name=tri_v11_r160_maira
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_1/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/maria-2/venv/bin/python"
MODEL_DIR=/project2/ruishanl_1185/Baselines/hf_cache/hub/models--microsoft--maira-2/snapshots/795a2b1cd4a310624b4e3d14b5a23e41fd273deb
REQUEST_RUN="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_requests/pool160_four_experts_20260813_e5c4"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates"
OUTPUT_RUN_ID=pool160_maira2_20260813_2f6a
TASK_CACHE="$WORKSPACE/.cache/tricompose_v1_1/maira2_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1_1/maira2_${SLURM_JOB_ID}"

for required in "$PYTHON" "$MODEL_DIR" "$REQUEST_RUN/manifest.json"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi

mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 2770 "$TASK_CACHE" "$TASK_TMP"
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

"$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_report.py" \
  --request-run "$REQUEST_RUN" \
  --output-root "$OUTPUT_ROOT" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --model-id maira2 \
  --model-dir "$MODEL_DIR"
