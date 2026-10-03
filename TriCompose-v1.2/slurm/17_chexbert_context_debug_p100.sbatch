#!/bin/bash
# Prepared only: explicit approval required before sbatch.
# Frozen development-only evaluator diagnosis; no generation or metric changes.
#SBATCH --job-name=tri_v12_cbctx
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
BANK="$ROOT/benchmarks/report_polarity_dev48_20261002_001"
CACHED="$ROOT/verification_runs/report_polarity_dev48_12560507_chexbert"
TASK="$ROOT/task_runtime/chexbert_context_${SLURM_JOB_ID}"
OUT="$ROOT/verification_runs"
RUN="chexbert_context_dev48_${SLURM_JOB_ID}"
TOOLS="$WORKSPACE/TriCompose-v1.2/benchmarks"
EVAL="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
PROGRAM="$TOOLS/diagnose_chexbert_polarity.py"
CB_PY="$WORKSPACE/cxrmate/venv/bin/python"
CB="$WORKSPACE/.cache/tricompose_report_eval/models/chexbert/chexbert.pth"
BERT="$WORKSPACE/.cache/tricompose_report_eval/models/bert-base-uncased"

test -x "$CB_PY"
test ! -e "$TASK"
test ! -e "$OUT/$RUN"
printf '%s  %s\n' \
  2bf3be71fbde99dce3d2a49ebdfef58d1c3f4dd96f9977d263c73482f3aca17b "$PROGRAM" \
  f22fa8d0071edbfffe9beb377f7edaf32c3eff2384e7da0019bb294e34990ad7 "$BANK/manifest.json" \
  5708bd946dc1e132b5db3a9fbb3423496fb4f7d0a5e8299c931c8e66a017e5dd "$CACHED/scores.json" \
  bd2749330be65a1445ac9fe383c244ee79b8e23e109de7332ee6a2602a67e402 "$TOOLS/prepare_expanded_polarity.py" \
  0f57eaae0eef3ff7a67eff308b80f757a237b99b5115e5c39cd82e6f8bdcc9a7 "$TOOLS/score_report_polarity.py" \
  7be74bbf64e0c88f40e785209c478eadc98118e5ddd8ea94bb47e6ec6bd338d9 "$EVAL/contracts.py" \
  3cb0ca4686a5cc766067568d2bd4e762db3e659c0798f6e32b1fcb78b729df6d "$EVAL/extract_report_labels_chexbert.py" \
  5d131d8dc8253211f127f48d8ac61fc689373e64dc92b9ec00a794e2331eafe6 "$WORKSPACE/cxrmate/tools/chexbert.py" \
  6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1 "$CB" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
chmod 0660 "$ROOT/slurm_logs/tri_v12_cbctx_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_cbctx_${SLURM_JOB_ID}.err"
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

"$CB_PY" "$PROGRAM" --bank-run "$BANK" --cached-run "$CACHED" \
  --checkpoint "$CB" --bert-path "$BERT" --output-root "$OUT" --run-id "$RUN" \
  >"$TASK/logs/diagnostic.log" 2>&1
printf 'status=development_chexbert_context_completed_no_metric_change_or_repair\n'
