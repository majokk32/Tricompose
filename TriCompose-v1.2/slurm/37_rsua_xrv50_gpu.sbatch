#!/bin/bash
# PREPARED ONLY: show this complete script/resources and obtain approval.
# Public RSUA 25+25 pneumonia diagnostic; no training or generation.
# Do not refit thresholds, promote clinical validity, or change old choices.
#SBATCH --job-name=tri_v12_rsua50
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
PROGRAM="$WORKSPACE/TriCompose-v1.2/real_validation/rsua_pilot.py"
PROGRAM_SHA=c57276dbe9281e10b6602fe98b333ca6d5be48c6f7697b1f391fe6002e122266
COHORT="$ROOT/real_validation/rsua_pilot_cohorts/cohort50_12625457_001/cohort.json"
COHORT_SHA=ed33c7708ac737aa8e270da2219457def542186f02bcc085c7188c7a135779c1
WEIGHT_DIR="$WORKSPACE/artifacts/protected/tricompose_v1_1/evaluation/smoke48/scoring_12257177/cache/xrv"
WEIGHT_NAME=nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
THRESHOLDS="$ROOT/real_validation/real_xrv_thresholds_12549079/thresholds.json"
OUTPUT="$ROOT/real_validation/rsua_xrv_pilots"
RUN_ID="rsua_xrv50_${SLURM_JOB_ID}"
TASK="$ROOT/job_runtime/$RUN_ID"
cd "$WORKSPACE"
chmod 0660 "$ROOT/slurm_logs/tri_v12_rsua50_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_rsua50_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
printf '%s  %s\n' "$PROGRAM_SHA" "$PROGRAM" | sha256sum --check --status
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/cache" "$TASK/tmp"
mkdir -m 2770 "$TASK/cache/xdg" "$TASK/cache/huggingface" "$TASK/cache/torch" \
  "$TASK/cache/cuda" "$TASK/cache/matplotlib" "$TASK/cache/kernels"
export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/real_validation:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels"
export TMPDIR="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
RESULT=0
"$PYTHON" "$PROGRAM" evaluate \
  --cohort "$COHORT" --cohort-sha256 "$COHORT_SHA" \
  --cache-dir "$WEIGHT_DIR" --weight-filename "$WEIGHT_NAME" \
  --thresholds "$THRESHOLDS" --allow-rsua-pixels \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log" || RESULT=$?
# Only this new task's files; never follow links or change old run modes.
find "$TASK" -type d -exec chmod 2770 {} +
find "$TASK" -type f -exec chmod 0660 {} +
if test "$RESULT" -ne 0; then
  printf 'status=failed_closed private_diagnostics_retained\n'
  exit "$RESULT"
fi
printf 'status=completed source_choices_and_thresholds_unchanged\n'
