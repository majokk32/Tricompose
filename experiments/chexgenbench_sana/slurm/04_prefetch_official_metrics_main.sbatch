#!/bin/bash
#SBATCH --job-name=cgb_metrics_fetch
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_sana/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_sana/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/chexgenbench_sana"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
DESTINATION="$WORKSPACE/CheXGenBench/checkpoints/official-metrics"
STAGING="$WORKSPACE/.tmp/chexgenbench_sana/metric_models_${SLURM_JOB_ID}"
CACHE_ROOT="$WORKSPACE/.cache/chexgenbench_sana/official_metrics"
TASK_TMP="$WORKSPACE/.tmp/chexgenbench_sana/metric_prefetch_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_metrics_environment\n' >&2
  exit 1
fi
if test -e "$DESTINATION"; then
  printf 'status=failed reason=scorer_bundle_already_exists\n' >&2
  exit 1
fi
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$TASK_TMP"
chmod 700 "$CACHE_ROOT" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch" "$TASK_TMP"
export PYTHONPATH="$VENDOR:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m tricompose_chexgenbench_sana.metric_prefetch \
  --destination "$DESTINATION" \
  --staging "$STAGING"
