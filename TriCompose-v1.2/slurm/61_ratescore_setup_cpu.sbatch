#!/bin/bash
# Proposal only. No submission until complete display and explicit approval.
# Approval must cover downloads/environment installation AND BioLORD licensing.
#SBATCH --job-name=tri_ratescore_setup
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
[[ "${TRICOMPOSE_APPROVE_DOWNLOADS:-}" == 1 ]]
[[ "${TRICOMPOSE_CONFIRM_BIOLORD_LICENSE:-}" == 1 ]]
rg --quiet "/job_${SLURM_JOB_ID}/" /proc/self/cgroup
cd "$WORKSPACE"
unset PYTHONPATH PYTHONUSERBASE PIP_TARGET PIP_PREFIX PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST PIP_USER
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PIP_CONFIG_FILE=/dev/null PIP_KEYRING_PROVIDER=disabled NETRC=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export PIP_CACHE_DIR="$WORKSPACE/.cache/ratescore_setup_${SLURM_JOB_ID}"
export HF_HOME="$WORKSPACE/.cache/ratescore_hf_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$WORKSPACE/.cache/ratescore_xdg_${SLURM_JOB_ID}"
export TMPDIR="$WORKSPACE/.tmp/ratescore_setup_${SLURM_JOB_ID}"
RUNTIME_ENV="$WORKSPACE/runtime/venvs/ratescore-v12-${SLURM_JOB_ID}-001"
STATUS_RUN="$WORKSPACE/artifacts/protected/tricompose_v1_2/ratescore_setup_${SLURM_JOB_ID}_001"
[[ ! -e "$RUNTIME_ENV" && ! -L "$RUNTIME_ENV" && ! -e "$STATUS_RUN" && ! -L "$STATUS_RUN" ]]
mkdir -m 2770 "$STATUS_RUN"
exec >"$STATUS_RUN/setup.log" 2>&1
chmod 0660 "$STATUS_RUN/setup.log"
mkdir -m 2770 "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
runtime/venvs/radgraph-xl-v12-12714150-001/bin/python \
  TriCompose-v1.2/tools/prepare_ratescore_assets.py acquire \
  --run-id "assets_${SLURM_JOB_ID}_001" --download-approved --license-confirmed
runtime/venvs/radgraph-xl-v12-12714150-001/bin/python -m venv "$RUNTIME_ENV"
"$RUNTIME_ENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.6.0+cpu'
"$RUNTIME_ENV/bin/python" -m pip install --index-url https://pypi.org/simple \
  -r TriCompose-v1.2/configs/ratescore_runtime_v1.txt
"$RUNTIME_ENV/bin/python" -m pip check
# Dependency receipt only; no module/model import, report, image or inference.
"$RUNTIME_ENV/bin/python" -m pip freeze --all >"$STATUS_RUN/pip_freeze.txt"
chmod 0660 "$STATUS_RUN/pip_freeze.txt"
"$RUNTIME_ENV/bin/python" TriCompose-v1.2/tools/seal_ratescore_setup.py \
  --run-id "assets_${SLURM_JOB_ID}_001" --status-run "$STATUS_RUN" \
  --environment "$RUNTIME_ENV"
