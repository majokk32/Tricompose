#!/bin/bash
#SBATCH --job-name=cgb_pixart50_eval
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_pixart/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_pixart/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PIXART_EXPERIMENT="$WORKSPACE/experiments/chexgenbench_pixart"
SANA_EXPERIMENT="$WORKSPACE/experiments/chexgenbench_sana"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
CONFIG="$PIXART_EXPERIMENT/configs/official_metrics_n50.yaml"
GENERATION_RUN_ID=cgb_pixart50_fixed_20260804_001
EVALUATION_RUN_ID=cgb_pixart50_official_metrics_20260804_001
CACHE_ROOT="$WORKSPACE/.cache/chexgenbench_sana/official_metrics"
TASK_CACHE="$WORKSPACE/.cache/chexgenbench_pixart/official_metrics/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/chexgenbench_pixart/metric_eval_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_metrics_environment\n' >&2
  exit 1
fi
if ! test -f "$CACHE_ROOT/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"; then
  printf 'status=failed reason=missing_frozen_inception_weights\n' >&2
  exit 1
fi
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
export PYTHONPATH="$WORKSPACE/src:$VENDOR:$SANA_EXPERIMENT/src:$PIXART_EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/torchinductor"
export MPLCONFIGDIR="$TASK_CACHE/matplotlib"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" -m tricompose_chexgenbench_pixart.official_metrics \
  --config "$CONFIG" \
  --generation-run-id "$GENERATION_RUN_ID" \
  --evaluation-run-id "$EVALUATION_RUN_ID"
