#!/bin/bash
#SBATCH --job-name=cgb_metrics_env
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
ENV_DIR="$WORKSPACE/CheXGenBench/metrics_venv"
CACHE_DIR="$WORKSPACE/.cache/chexgenbench_sana/official_metrics/pip"
TMP_DIR="$WORKSPACE/.tmp/chexgenbench_sana/metrics_env_${SLURM_JOB_ID}"

if test -e "$ENV_DIR"; then
  printf 'status=failed reason=environment_already_exists\n' >&2
  exit 1
fi
mkdir -p "$CACHE_DIR" "$TMP_DIR"
chmod 700 "$CACHE_DIR" "$TMP_DIR"
export PIP_CACHE_DIR="$CACHE_DIR"
export TMPDIR="$TMP_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install \
  pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
"$ENV_DIR/bin/python" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
"$ENV_DIR/bin/python" -m pip install numpy==1.26.4
"$ENV_DIR/bin/python" -m pip install \
  torchmetrics==1.7.1 \
  torch-fidelity==0.3.0 \
  prdc==0.2 \
  transformers==4.39.3 \
  'huggingface_hub[hf-xet]==0.30.1' \
  safetensors==0.5.3 \
  pandas==2.2.3 \
  scipy==1.15.2 \
  scikit-learn==1.6.1 \
  Pillow==11.1.0 \
  PyYAML==6.0.2 \
  termcolor==3.0.1 \
  tqdm==4.67.1 \
  timm==0.9.16 \
  pydicom==2.4.4 \
  scikit-image==0.23.1 \
  SimpleITK==2.3.1 \
  multiprocess==0.70.17 \
  matplotlib==3.10.1 \
  seaborn==0.13.2 \
  PyRadiomics==3.0.1

"$ENV_DIR/bin/python" -c \
  "import importlib.metadata as m; print('status=completed', 'torch='+m.version('torch'), 'torchvision='+m.version('torchvision'), 'torchmetrics='+m.version('torchmetrics'), 'transformers='+m.version('transformers'))"
