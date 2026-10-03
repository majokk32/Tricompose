#!/bin/bash
# Prepared only: show this complete script and obtain approval before sbatch.
# Fictional language diagnostic; frozen model; no generation/metric changes.
#SBATCH --job-name=tri_v12_cbassert
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
BANK="$ROOT/benchmarks/authored_assertions_20261002_001"
TASK="$ROOT/task_runtime/authored_assertions_${SLURM_JOB_ID}"
OUT="$ROOT/verification_runs"
RUN="authored_assertions_chexbert_${SLURM_JOB_ID}"
TOOLS="$WORKSPACE/TriCompose-v1.2/benchmarks"
EVAL="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
CB_PY="$WORKSPACE/cxrmate/venv/bin/python"
CB="$WORKSPACE/.cache/tricompose_report_eval/models/chexbert/chexbert.pth"
BERT="$WORKSPACE/.cache/tricompose_report_eval/models/bert-base-uncased"

chmod 0660 "$ROOT/slurm_logs/tri_v12_cbassert_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_cbassert_${SLURM_JOB_ID}.err"
test -x "$CB_PY"
test ! -e "$TASK"
test ! -e "$OUT/$RUN"
test ! -e "$OUT/${RUN}_analysis"
printf '%s  %s\n' \
  824c96035872c902b078177baf90c1cfee3323ad281821dbd0641a7410385451 "$TOOLS/report_assertion_challenge.py" \
  055a0b5655311d8e064f9fb755c42aad8f3d41db3e3763a46c75c427b55f6b29 "$TOOLS/score_report_assertion_challenge.py" \
  5f76ca3d7f710f4fdf722f87f27afc6548c839f1c4d77e56a6502ec4e3583757 "$BANK/manifest.json" \
  74394e6bd5baa057c70cac42fe27d7ce49e3280f08f98ec929a74e07dfd38d4d "$BANK/resolver.jsonl" \
  7be74bbf64e0c88f40e785209c478eadc98118e5ddd8ea94bb47e6ec6bd338d9 "$EVAL/contracts.py" \
  3cb0ca4686a5cc766067568d2bd4e762db3e659c0798f6e32b1fcb78b729df6d "$EVAL/extract_report_labels_chexbert.py" \
  5d131d8dc8253211f127f48d8ac61fc689373e64dc92b9ec00a794e2331eafe6 "$WORKSPACE/cxrmate/tools/chexbert.py" \
  7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20 "$BERT/config.json" \
  07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3 "$BERT/vocab.txt" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
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

# Blind model phase: only invented text, opaque IDs and hashes; no answer key.
# Checkpoint hash is validated inside Slurm before loading the frozen model.
"$CB_PY" "$TOOLS/score_report_assertion_challenge.py" --bank-run "$BANK" \
  --checkpoint "$CB" --bert-path "$BERT" --output-root "$OUT" --run-id "$RUN" \
  >"$TASK/logs/prediction.log" 2>&1
# Separate CPU analysis reads authored answers only after predictions exist.
"$CB_PY" "$TOOLS/report_assertion_challenge.py" --mode analyze --bank-run "$BANK" \
  --score-run "$OUT/$RUN" --output-root "$OUT" --run-id "${RUN}_analysis" \
  >"$TASK/logs/analysis.log" 2>&1
printf 'status=authored_language_diagnostic_completed_no_primary_metric_or_selection_change\n'
