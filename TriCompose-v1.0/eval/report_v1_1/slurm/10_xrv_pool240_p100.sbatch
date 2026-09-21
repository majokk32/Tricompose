#!/bin/bash
#SBATCH --job-name=tri_rpteval_xrv240
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_report_eval/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_report_eval/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EVAL_ROOT="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
SOURCE_XRV="$WORKSPACE/.cache/tricompose_v1/scoring/job_10871232/xrv"
XRV_WEIGHT=nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/evaluation/finding_labels"
OUTPUT_RUN_ID=xrv_pool240_uncalibrated_20260813_001
TASK_CACHE="$WORKSPACE/.cache/tricompose_report_eval/xrv_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_report_eval/xrv_${SLURM_JOB_ID}"
LOG_ROOT="$WORKSPACE/logs/slurm/tricompose_report_eval"

CXR_SANA="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates/pool80_chexgenbench_sana_s1_20260813_d0ce"
CXR_PIXART="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates/pool80_chexgenbench_pixart_s1_20260813_18b7"
CXR_ROENTGEN="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates/pool80_roentgen_v2_fp16_s1_20260813_0bd4"

for required in "$PYTHON" "$SOURCE_XRV/$XRV_WEIGHT" "$CXR_SANA/manifest.json" "$CXR_PIXART/manifest.json" "$CXR_ROENTGEN/manifest.json"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi

mkdir -p "$TASK_CACHE/xrv" "$TASK_TMP" "$LOG_ROOT"
chmod 2770 "$TASK_CACHE" "$TASK_CACHE/xrv" "$TASK_TMP" "$LOG_ROOT"
install -m 660 "$SOURCE_XRV/$XRV_WEIGHT" "$TASK_CACHE/xrv/$XRV_WEIGHT"

export PYTHONPATH="$EVAL_ROOT:$WORKSPACE/experiments/medim/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE/xdg"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$EVAL_ROOT/extract_cxr_labels_xrv.py" \
  --cxr-run "$CXR_SANA" \
  --cxr-run "$CXR_PIXART" \
  --cxr-run "$CXR_ROENTGEN" \
  --cache-dir "$TASK_CACHE/xrv" \
  --weight-filename "$XRV_WEIGHT" \
  --model-name densenet121-res224-all \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$OUTPUT_RUN_ID"
