#!/bin/bash
#SBATCH --job-name=cgb_sana50_fixed
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_sana/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/chexgenbench_sana/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/chexgenbench_sana"
ROENTGEN_EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
PYTHON="$WORKSPACE/CheXGenBench/venv/bin/python"
REVISION=6c8989d92e75fbcfc91a82c3570389293867ad6a
MODEL_DIR="$WORKSPACE/CheXGenBench/checkpoints/sana-e20/$REVISION"
SOURCE_RUN_ID=medim50_20260803_001
OUTPUT_RUN_ID=cgb_sana50_fixed_20260804_002
TASK_CACHE="$WORKSPACE/.cache/chexgenbench_sana/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/chexgenbench_sana/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -d "$MODEL_DIR"; then
  printf 'status=failed reason=missing_model_snapshot\n' >&2
  exit 1
fi
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_TMP"
export PYTHONPATH="$WORKSPACE/src:$ROENTGEN_EXPERIMENT/src:$EXPERIMENT/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/torchinductor"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_VERBOSITY=error
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -m tricompose_chexgenbench_sana.adapter \
  --model-dir "$MODEL_DIR" \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --limit 50 \
  --batch-size 2
