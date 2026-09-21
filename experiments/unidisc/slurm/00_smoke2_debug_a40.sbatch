#!/bin/bash
#SBATCH --job-name=unidisc_ehr2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/unidisc/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/unidisc/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EXPERIMENT="$WORKSPACE/experiments/unidisc"
ROENTGEN_EXPERIMENT="$WORKSPACE/experiments/roentgen_v2"
UNIDISC_ROOT=/project2/ruishanl_1185/MeDiM/UniDisc_baseline
PYTHON="$UNIDISC_ROOT/.venv/bin/python"
SOURCE_RUN_ID=medim50_20260803_001
OUTPUT_RUN_ID="${1:?usage: sbatch 00_smoke2_debug_a40.sbatch OUTPUT_RUN_ID}"
TOKENIZER_CACHE_SOURCE=/home1/yikeyang/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf
TASK_CACHE="$WORKSPACE/.cache/unidisc/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/unidisc/job_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
if ! test -d "$UNIDISC_ROOT/ckpts/unidisc_interleaved"; then
  printf 'status=failed reason=missing_model_snapshot\n' >&2
  exit 1
fi
if ! test -d "$TOKENIZER_CACHE_SOURCE"; then
  printf 'status=failed reason=missing_tokenizer_snapshot\n' >&2
  exit 1
fi

mkdir -p "$TASK_CACHE/huggingface/hub" "$TASK_TMP"
chmod 700 "$TASK_CACHE" "$TASK_CACHE/huggingface" "$TASK_CACHE/huggingface/hub" "$TASK_TMP"
cp -a "$TOKENIZER_CACHE_SOURCE" "$TASK_CACHE/huggingface/hub/"
chmod -R u+rwX,go-rwx "$TASK_CACHE/huggingface/hub/models--NousResearch--Llama-2-7b-hf"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$ROENTGEN_EXPERIMENT/src:$UNIDISC_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export UNIDISC_DIR="$UNIDISC_ROOT"
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export HF_HUB_CACHE="$TASK_CACHE/huggingface/hub"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -m tricompose_unidisc.adapter \
  --model-root "$UNIDISC_ROOT" \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --limit 2
