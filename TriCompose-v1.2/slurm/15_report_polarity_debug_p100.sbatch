#!/bin/bash
# Prepared only. Show the entire script/resources and obtain approval first.
# Five synthetic reports, two fixed images; frozen metrics only, no generation.
#SBATCH --job-name=tri_v12_polarity
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
BANK="$ROOT/benchmarks/report_polarity_smoke2_12553939_001"
TASK="$ROOT/task_runtime/polarity_${SLURM_JOB_ID}"
OUT="$ROOT/verification_runs"
RUN="report_polarity_${SLURM_JOB_ID}"
TOOLS="$WORKSPACE/TriCompose-v1.2/benchmarks"
EVAL="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
PROGRAM="$TOOLS/score_report_polarity.py"
CB_PY="$WORKSPACE/cxrmate/venv/bin/python"
BV_PY="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
CB="$WORKSPACE/.cache/tricompose_report_eval/models/chexbert/chexbert.pth"
BERT="$WORKSPACE/.cache/tricompose_report_eval/models/bert-base-uncased"
BV="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"

test -x "$CB_PY"
test -x "$BV_PY"
test -r "$BERT/vocab.txt"
test ! -e "$TASK"
for suffix in chexbert biovil analysis; do
  test ! -e "$OUT/${RUN}_${suffix}"
done
printf '%s  %s\n' \
  58d6842b28727aa33f4e3d9b0495947df3eacead332f14168c6d08df88bad8a4 "$BANK/manifest.json" \
  0f57eaae0eef3ff7a67eff308b80f757a237b99b5115e5c39cd82e6f8bdcc9a7 "$PROGRAM" \
  64b6a1874e84ac15e5d925d2478eb701fd692d76591167f565a4b49e73267d91 "$TOOLS/build_report_polarity_benchmark.py" \
  5d131d8dc8253211f127f48d8ac61fc689373e64dc92b9ec00a794e2331eafe6 "$WORKSPACE/cxrmate/tools/chexbert.py" \
  62a9a0ec455848a27ee2e46798053172796aa55a97cbef211aef17c45f43ef5d "$EVAL/score_report_cxr_biovil.py" \
  6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1 "$CB" \
  b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77 "$BV/biovil_t_image_model_proj_size_128.pt" \
  6d86a8d760eaa09c9a55d57cc6f6bb01b0cbccb8b827fc775a79f37a8fbda76c "$BV/pytorch_model.bin" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
chmod 0660 "$ROOT/slurm_logs/tri_v12_polarity_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_polarity_${SLURM_JOB_ID}.err"
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

export PYTHONPATH="$WORKSPACE/cxrmate:$TOOLS:$EVAL"
"$CB_PY" "$PROGRAM" --mode chexbert --bank-run "$BANK" \
  --checkpoint "$CB" --bert-path "$BERT" --output-root "$OUT" \
  --run-id "${RUN}_chexbert" >"$TASK/logs/chexbert.log" 2>&1
printf 'status=polarity_chexbert_completed\n'
export PYTHONPATH="$TOOLS:$EVAL:$VENDOR"
"$BV_PY" "$PROGRAM" --mode biovil --bank-run "$BANK" \
  --model-path "$BV" --output-root "$OUT" --run-id "${RUN}_biovil" \
  >"$TASK/logs/biovil.log" 2>&1
printf 'status=polarity_biovil_completed\n'
"$BV_PY" "$PROGRAM" --mode analyze --bank-run "$BANK" \
  --chexbert-run "$OUT/${RUN}_chexbert" --biovil-run "$OUT/${RUN}_biovil" \
  --output-root "$OUT" --run-id "${RUN}_analysis" >"$TASK/logs/analysis.log" 2>&1
printf 'status=polarity_sensitivity_analysis_completed_no_clinical_localization\n'
