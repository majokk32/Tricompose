#!/bin/bash
#SBATCH --job-name=rgv2_env
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/roentgen_v2/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/roentgen_v2/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
ENV_DIR="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1"
CACHE_DIR="$WORKSPACE/.cache/roentgen_v2/pip"
TMP_DIR="$WORKSPACE/.tmp/roentgen_v2/env_${SLURM_JOB_ID}"

if test -e "$ENV_DIR"; then
  printf 'status=failed reason=environment_already_exists\n' >&2
  exit 1
fi

mkdir -p "$CACHE_DIR" "$TMP_DIR" "$(dirname "$ENV_DIR")"
chmod 700 "$CACHE_DIR" "$TMP_DIR"
export PIP_CACHE_DIR="$CACHE_DIR"
export TMPDIR="$TMP_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 xformers \
  --index-url https://download.pytorch.org/whl/cu126
"$ENV_DIR/bin/python" -m pip install \
  accelerate==1.10.0 \
  diffusers==0.35.1 \
  transformers==4.55.4 \
  safetensors \
  huggingface_hub \
  Pillow

"$ENV_DIR/bin/python" -c \
  "import importlib.metadata as m; print('status=completed', 'torch='+m.version('torch'), 'diffusers='+m.version('diffusers'), 'transformers='+m.version('transformers'))"
