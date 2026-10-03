#!/bin/bash
# Prepared only. Do not submit without user approval.
# 48 CXR candidates x 4 frozen report experts = 192 reports.
#SBATCH --job-name=tri_v11_direct8_report
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-3%4
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/job_logs/%x_%A_%a.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_1/job_logs/%x_%A_%a.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PROTECTED="$WORKSPACE/artifacts/protected/tricompose_v1_1"
REQUEST_RUN="$PROTECTED/report_requests/directfacts8_four_experts_12262101_001"
OUTPUT_ROOT="$PROTECTED/report_candidates"
TASK_RUNTIME="$PROTECTED/report_runtime/direct8_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
test ! -e "$TASK_RUNTIME"
mkdir -p "$TASK_RUNTIME/cache" "$TASK_RUNTIME/tmp"
chmod 2770 "$TASK_RUNTIME" "$TASK_RUNTIME/cache" "$TASK_RUNTIME/tmp"

EXTRA_ARGS=()
case "$SLURM_ARRAY_TASK_ID" in
  0)
    MODEL_ID=maira2
    PYTHON="$WORKSPACE/maria-2/venv/bin/python"
    MODEL_DIR=/project2/ruishanl_1185/Baselines/hf_cache/hub/models--microsoft--maira-2/snapshots/795a2b1cd4a310624b4e3d14b5a23e41fd273deb
    ;;
  1)
    MODEL_ID=cxrmate_single
    PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
    MODEL_DIR=/project2/ruishanl_1185/Baselines/hf_cache/hub/models--aehrc--cxrmate-single-tf/snapshots/84dfcba8125c9b9296bfc1ee317749788b6b59b4
    ;;
  2)
    MODEL_ID=llavarad
    PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/llavarad/bin/python
    MODEL_DIR=/project2/ruishanl_1185/SDP_for_VLM/models/llava_rad
    EXTRA_ARGS=(
      --model-base /project2/ruishanl_1185/SDP_for_VLM/models/vicuna-7b-v1.5
      --biomedbert-dir /home1/yikeyang/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d
      --external-root /project2/ruishanl_1185/SDP_for_VLM/SDP_for_VLM/llava-rad
      --runtime-dir "$TASK_RUNTIME/tmp/model_runtime"
    )
    ;;
  3)
    MODEL_ID=chexagent2
    PYTHON="$WORKSPACE/CheXagent-2/venv/bin/python"
    MODEL_DIR="$WORKSPACE/CheXagent-2/checkpoints/chexagent-2-srrg-findings/9f7225fc382ddd1297ade1aa796da660237940bc"
    EXTRA_ARGS=(
      --vision-dir "$WORKSPACE/CheXagent-2/checkpoints/chexagent2-xraysiglip/f0edbf5d90dba44edb7f4f96d8663537cb0749bf"
      --precision float32
    )
    ;;
  *) exit 2 ;;
esac

OUTPUT_RUN_ID="directfacts8_${MODEL_ID}_${SLURM_ARRAY_JOB_ID}"
test -x "$PYTHON"
test -d "$MODEL_DIR"
test -r "$REQUEST_RUN/manifest.json"
test ! -e "$OUTPUT_ROOT/$OUTPUT_RUN_ID"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_RUNTIME/cache/xdg"
export HF_HOME="$TASK_RUNTIME/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK_RUNTIME/cache/torch" MPLCONFIGDIR="$TASK_RUNTIME/cache/matplotlib"
export CUDA_CACHE_PATH="$TASK_RUNTIME/cache/cuda" TMPDIR="$TASK_RUNTIME/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if "$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/run_report.py" \
  --request-run "$REQUEST_RUN" --output-root "$OUTPUT_ROOT" \
  --output-run-id "$OUTPUT_RUN_ID" --model-id "$MODEL_ID" \
  --model-dir "$MODEL_DIR" "${EXTRA_ARGS[@]}"; then
  printf 'status=completed model=%s\n' "$MODEL_ID"
else
  printf 'status=failed model=%s\n' "$MODEL_ID"
  exit 1
fi
