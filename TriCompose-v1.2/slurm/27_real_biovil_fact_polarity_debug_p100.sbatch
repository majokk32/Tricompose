#!/bin/bash
# Prepared only; show this full script/resources and obtain explicit approval.
# Real CXR pixels are internal inputs; no source report/EHR text is read.
#SBATCH --job-name=tri_v12_bvpol
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
CODE="$WORKSPACE/TriCompose-v1.2/real_validation"
MODEL="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
PREVIOUS="$ROOT/real_validation/real_biovil_valtest128_20260930_001/scores.json"
REFERENCE="$ROOT/real_validation/real_xrv_weak_valtest128_20261001_001/scores.json"
TASK="$ROOT/task_runtime/fact_polarity_${SLURM_JOB_ID}"
RUN="real_biovil_fact_polarity_${SLURM_JOB_ID}"
test "${ALLOW_REAL_IMAGE_POLARITY:-0}" = 1
test -n "${SLURM_JOB_ID:-}"
rg --quiet "/job_${SLURM_JOB_ID}/" /proc/self/cgroup
test ! -e "$TASK"
test ! -e "$ROOT/real_validation/$RUN"
test -x "$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
chmod 0660 "$ROOT/slurm_logs/tri_v12_bvpol_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_bvpol_${SLURM_JOB_ID}.err"
printf '%s  %s\n' \
  e0a3ff342866f076c40d0687cf79962ac1f87cb6afc516baecdb65f4fe0be379 "$CODE/biovil_fact_polarity.py" \
  5c1be96dfd9df8ff7bb549f9924cccac8d0b3ef371ff5d08a290881f56b012ce "$CODE/biovil_matched_pairs.py" \
  62a9a0ec455848a27ee2e46798053172796aa55a97cbef211aef17c45f43ef5d "$WORKSPACE/TriCompose-v1.0/eval/report_v1_1/score_report_cxr_biovil.py" \
  7be74bbf64e0c88f40e785209c478eadc98118e5ddd8ea94bb47e6ec6bd338d9 "$WORKSPACE/TriCompose-v1.0/eval/report_v1_1/contracts.py" \
  762009d2e41408b0ae43de6d49487162bd249f678981120bcaf77784d4a41c36 "$PREVIOUS" \
  cbcaa11fa41e6ede45dd3ece70f07d7fcaad1bd0af6bdfe5c45a762b72210219 "$REFERENCE" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp"
trap 'find "$TASK" -type d -exec chmod 2770 {} +; find "$TASK" -type f -exec chmod 0660 {} +' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export MPLCONFIGDIR="$TASK/cache/matplotlib" TMPDIR="$TASK/tmp"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels" TORCH_EXTENSIONS_DIR="$TASK/cache/extensions"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$CODE:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
"$WORKSPACE/CheXGenBench/metrics_venv/bin/python" "$CODE/biovil_fact_polarity.py" \
  --dataset-root /project2/ruishanl_1185/datasets/three_modalities/v2_labs_vitals \
  --previous-scores "$PREVIOUS" --reference-scores "$REFERENCE" \
  --model-path "$MODEL" --text-batch-size 16 --min-per-class 10 --replay-count 4 \
  --output-root "$ROOT/real_validation" --run-id "$RUN" \
  --allow-real-image-polarity >"$TASK/run.log" 2>&1
printf 'status=single_fact_polarity_finished_no_training_or_selection_change\n'
