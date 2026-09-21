#!/bin/bash
#SBATCH --job-name=tri_v1_bank_final
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
STAGING_RUN="$WORKSPACE/artifacts/protected/tricompose_v1/staging_qwen2_pool10_v1aligned_20260806_001"
CXR_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/cxr_candidates"
REPORT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/report_candidates"
BANK_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/candidate_banks"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1/finalize_${SLURM_JOB_ID}"

if ! test -x "$PYTHON"; then
  printf 'status=failed reason=missing_environment\n' >&2
  exit 1
fi
mkdir -p "$TASK_TMP"
chmod 700 "$TASK_TMP"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TMPDIR="$TASK_TMP"

"$PYTHON" "$WORKSPACE/tools/finalize_tricompose_v1_candidate_bank.py" \
  --staging-run "$STAGING_RUN" \
  --cxr-run "$CXR_ROOT/v1_smoke2_roentgen_v2_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_sana_s2_20260806_001" \
  --cxr-run "$CXR_ROOT/v1_smoke2_pixart_s2_20260806_001" \
  --report-run "$REPORT_ROOT/v1_smoke2_maira2_allcxr_20260806_001" \
  --report-run "$REPORT_ROOT/v1_smoke2_cxrmate_allcxr_20260806_001" \
  --report-run "$REPORT_ROOT/v1_smoke2_llavarad_allcxr_20260806_001" \
  --report-run "$REPORT_ROOT/v1_smoke2_chexagent2_allcxr_20260806_001" \
  --output-root "$BANK_ROOT" \
  --run-id v1_smoke2_all_models_20260806_001
