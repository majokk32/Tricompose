#!/bin/bash
# Prepared only: show the complete script and obtain approval before sbatch.
# New isolated library environment; official rules; fictional text only.
#SBATCH --job-name=tri_v12_context
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
BANK="$ROOT/benchmarks/authored_assertions_20261002_001"
BASELINE="$ROOT/verification_runs/authored_assertions_chexbert_12580901"
OUT="$ROOT/verification_runs"
RUN="authored_assertions_context_${SLURM_JOB_ID}"
TASK="$ROOT/task_runtime/report_context_${SLURM_JOB_ID}"
CTX_ENV="$WORKSPACE/runtime/venvs/report-context-v12-${SLURM_JOB_ID}"
TOOLS="$WORKSPACE/TriCompose-v1.2/benchmarks"
EVAL="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
BASE_PY="$WORKSPACE/cxrmate/venv/bin/python"
MEDSPACY_URL='https://files.pythonhosted.org/packages/91/82/822c73438eac61351361f3de7278e381c42df81740d2dca363adc736e74d/medspacy-1.3.1.tar.gz#sha256=976599d9446e98018b311951cfa926be5a59d0c140967b828fe98bc4bada40fe'

chmod 0660 "$ROOT/slurm_logs/tri_v12_context_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_context_${SLURM_JOB_ID}.err"
test -x "$BASE_PY"
test ! -e "$TASK"
test ! -e "$CTX_ENV"
for suffix in '' _analysis _gate _gate_analysis; do test ! -e "$OUT/${RUN}${suffix}"; done
printf '%s  %s\n' \
  00459f4c8c9faf009d60b56529991a690a53ac2b782e9ea1c21604c2789779cd "$WORKSPACE/TriCompose-v1.2/src/tricompose_v12/report_assertions.py" \
  475596019a21ff386529d358e502a716d8c57d48516391317812aeef6ad060db "$TOOLS/gate_report_assertion_predictions.py" \
  7beab7ff9efe1b854eb877a2728d2be71f0c45921821220d3c09ca99c54afcd3 "$TOOLS/score_report_assertions_context.py" \
  824c96035872c902b078177baf90c1cfee3323ad281821dbd0641a7410385451 "$TOOLS/report_assertion_challenge.py" \
  7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18 "$TOOLS/repair_cached_report_evidence.py" \
  7be74bbf64e0c88f40e785209c478eadc98118e5ddd8ea94bb47e6ec6bd338d9 "$EVAL/contracts.py" \
  3cb0ca4686a5cc766067568d2bd4e762db3e659c0798f6e32b1fcb78b729df6d "$EVAL/extract_report_labels_chexbert.py" \
  5f76ca3d7f710f4fdf722f87f27afc6548c839f1c4d77e56a6502ec4e3583757 "$BANK/manifest.json" \
  2fde45f8cef8993e7c6f932448c761557a61c4af8c258109e951ba0d36b348c1 "$BASELINE/manifest.json" \
  43f8b4dbb29062cd8ae8b9c44d9451b83ebc74355b1279c05844852a97c8f899 "$BASELINE/predictions.json" \
  | sha256sum --check --status
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
trap 'find "$TASK" -type d -exec chmod 2770 {} +; find "$TASK" -type f -exec chmod 0660 {} +' EXIT

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" PIP_CACHE_DIR="$TASK/cache/pip"
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL='' PIP_TRUSTED_HOST='' PIP_NO_INPUT=1
export HF_HOME="$TASK/cache/huggingface" TORCH_HOME="$TASK/cache/torch"
export MPLCONFIGDIR="$TASK/cache/matplotlib" TMPDIR="$TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TOOLS:$EVAL:$WORKSPACE/TriCompose-v1.2/src"

# Only the new environment is written. No pretrained weights are downloaded.
"$BASE_PY" -m venv --copies "$CTX_ENV" >"$TASK/logs/install.log" 2>&1
"$CTX_ENV/bin/python" -m pip --disable-pip-version-check install \
  --report "$TASK/install_report.json" "$MEDSPACY_URL" \
  'spacy==3.7.5' 'PyRuSH==1.0.12' 'numpy==1.26.4' \
  >>"$TASK/logs/install.log" 2>&1

# Blind official parser, then separate authored-key analysis.
"$CTX_ENV/bin/python" "$TOOLS/score_report_assertions_context.py" \
  --bank-run "$BANK" --output-root "$OUT" --run-id "$RUN" \
  >"$TASK/logs/parser.log" 2>&1
"$CTX_ENV/bin/python" "$TOOLS/report_assertion_challenge.py" --mode analyze \
  --bank-run "$BANK" --score-run "$OUT/$RUN" --output-root "$OUT" --run-id "${RUN}_analysis" \
  >"$TASK/logs/parser_analysis.log" 2>&1
# Existing CheXbert proposals can only be vetoed, not flipped or promoted.
"$CTX_ENV/bin/python" "$TOOLS/gate_report_assertion_predictions.py" --mode gate \
  --bank-run "$BANK" --score-run "$BASELINE" --context-run "$OUT/$RUN" \
  --output-root "$OUT" --run-id "${RUN}_gate" >"$TASK/logs/gate.log" 2>&1
"$CTX_ENV/bin/python" "$TOOLS/gate_report_assertion_predictions.py" --mode analyze \
  --bank-run "$BANK" --gate-run "$OUT/${RUN}_gate" \
  --output-root "$OUT" --run-id "${RUN}_gate_analysis" >"$TASK/logs/gate_analysis.log" 2>&1
printf 'status=official_context_diagnostic_completed_no_primary_selection_or_regeneration_change\n'
