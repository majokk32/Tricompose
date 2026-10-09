#!/bin/bash
# PREPARED ONLY: show this entire script and obtain approval before sbatch.
# Existing fully synthetic 4 CXR / 8 report pairs; frozen secondary scoring.
# No generation, training, downloads, selection changes or external API.
# <=4 image encodings +8 text encodings, zero retries; private trace checks.
#SBATCH --job-name=tri_paired_biovil
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
WORKSPACE=/project2/ruishanl_1185/inference_3mod
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_2"
RUN_ID="paired2_biovil_${SLURM_JOB_ID}"
TASK="$BASE/job_runtime/$RUN_ID"
OUTPUT="$BASE/paired_probe_endpoints"
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
PROGRAM="$WORKSPACE/TriCompose-v1.2/tools/score_paired_probes_v1.py"
PROGRAM_SHA=048b6c0668c5e9ce2ab0b646449cd2bbf05b3f0e39602d32ff96780ceb855b76
TESTS="$WORKSPACE/TriCompose-v1.2/tests/test_paired_probe_secondary_v1.py"
TESTS_SHA=2d41713abec46a0596aa0fda0aae3f1a2e62d690fc97a286b4ca51226770791c
cd "$WORKSPACE"
chmod 0660 "$BASE/slurm_logs/tri_paired_biovil_${SLURM_JOB_ID}.out" \
  "$BASE/slurm_logs/tri_paired_biovil_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
ACTUAL_SHA="$(sha256sum "$PROGRAM")"
ACTUAL_TESTS_SHA="$(sha256sum "$TESTS")"
if test "${ACTUAL_SHA%% *}" != "$PROGRAM_SHA" || test "${ACTUAL_TESTS_SHA%% *}" != "$TESTS_SHA"; then
  printf 'status=refused reason=worker_source_changed\n'
  exit 1
fi
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/cache" "$TASK/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor" MPLCONFIGDIR="$TASK/cache/matplotlib"
export TMPDIR="$TASK/tmp" TEMP="$TASK/tmp" TMP="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if ! "$PYTHON" "$PROGRAM" run \
  --plan-run "$BASE/paired_probe_plans/paired2_12784259_001" \
  --plan-manifest-sha256 ea4a1faf17b9d22a063bccd6d9482c4f598d017acb63920a59c2a0934ab0d37a \
  --source-run "$BASE/paired_probe_runs/paired2_12791443" \
  --source-manifest-sha256 e5979cddf6b2075ea5a3e5ddef2c282a6278e89493144714b3afd1fd4dac0713 \
  --audit-run "$BASE/paired_probe_audits/paired2_12791443_001" \
  --audit-manifest-sha256 548ee53017c7a46fe836098e2be5bc00dc2cc8a1b2d7c8a70aa171ffb35fd195 \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/controller.stdout.log" 2>"$TASK/controller.stderr.log"; then
  printf 'status=failed reason=see_protected_endpoint_receipts\n'
  exit 1
fi
printf 'status=completed_secondary_not_clinical\n'
