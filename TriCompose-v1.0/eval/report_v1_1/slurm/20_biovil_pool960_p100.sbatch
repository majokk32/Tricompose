#!/bin/bash
#SBATCH --job-name=tri_rpteval_bv960
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_report_eval/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_report_eval/%x_%j.err

set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EVAL_ROOT="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
MODEL="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
OUTPUT_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_1/evaluation/secondary_scores"
OUTPUT_RUN_ID=biovil_report_cxr_pool960_20260813_001
TASK_CACHE="$WORKSPACE/.cache/tricompose_report_eval/biovil_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_report_eval/biovil_${SLURM_JOB_ID}"
LOG_ROOT="$WORKSPACE/logs/slurm/tricompose_report_eval"

CXR_SANA="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates/pool80_chexgenbench_sana_s1_20260813_d0ce"
CXR_PIXART="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates/pool80_chexgenbench_pixart_s1_20260813_18b7"
CXR_ROENTGEN="$WORKSPACE/artifacts/protected/tricompose_v1_1/cxr_candidates/pool80_roentgen_v2_fp16_s1_20260813_0bd4"
RPT_MAIRA_A="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/pool160_maira2_20260813_2f6a"
RPT_MAIRA_B="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/roentgen80_maira2_20260813_a31e"
RPT_CXRMATE_A="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/pool160_cxrmate_single_20260813_74d1"
RPT_CXRMATE_B="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/roentgen80_cxrmate_single_20260813_d6a4"
RPT_LLAVA_A="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/pool160_llavarad_20260813_aa39"
RPT_LLAVA_B="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/roentgen80_llavarad_20260813_b472"
RPT_CHEXA_A="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/pool160_chexagent2_fp32_20260813_51f8"
RPT_CHEXA_B="$WORKSPACE/artifacts/protected/tricompose_v1_1/report_candidates/roentgen80_chexagent2_fp32_20260813_c593"

for required in "$PYTHON" "$MODEL" "$VENDOR" "$CXR_SANA/manifest.json" "$CXR_PIXART/manifest.json" "$CXR_ROENTGEN/manifest.json" "$RPT_MAIRA_A/manifest.json" "$RPT_MAIRA_B/manifest.json" "$RPT_CXRMATE_A/manifest.json" "$RPT_CXRMATE_B/manifest.json" "$RPT_LLAVA_A/manifest.json" "$RPT_LLAVA_B/manifest.json" "$RPT_CHEXA_A/manifest.json" "$RPT_CHEXA_B/manifest.json"; do
  if ! test -e "$required"; then
    printf 'status=failed reason=missing_required_asset\n' >&2
    exit 1
  fi
done
if test -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"; then
  printf 'status=failed reason=output_run_exists\n' >&2
  exit 1
fi

mkdir -p "$TASK_CACHE" "$TASK_TMP" "$LOG_ROOT"
chmod 2770 "$TASK_CACHE" "$TASK_TMP" "$LOG_ROOT"
export PYTHONPATH="$EVAL_ROOT:$VENDOR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE/xdg"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$EVAL_ROOT/score_report_cxr_biovil.py" \
  --cxr-run "$CXR_SANA" --cxr-run "$CXR_PIXART" --cxr-run "$CXR_ROENTGEN" \
  --report-run "$RPT_MAIRA_A" --report-run "$RPT_MAIRA_B" \
  --report-run "$RPT_CXRMATE_A" --report-run "$RPT_CXRMATE_B" \
  --report-run "$RPT_LLAVA_A" --report-run "$RPT_LLAVA_B" \
  --report-run "$RPT_CHEXA_A" --report-run "$RPT_CHEXA_B" \
  --model-path "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$OUTPUT_RUN_ID"
