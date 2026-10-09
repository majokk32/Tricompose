#!/bin/bash
# PREPARED ONLY: complete-script review and explicit approval before sbatch.
# Read four existing protected synthetic tokenizer traces; text-free readout.
# No GPU, model/tokenizer execution, real inputs, downloads or API calls.
#SBATCH --job-name=tri_paired_trace
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
WORKSPACE=/project2/ruishanl_1185/inference_3mod
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_2"
RUN_ID="paired_trace_${SLURM_JOB_ID}"
TASK="$BASE/job_runtime/$RUN_ID"
OUTPUT="$BASE/paired_conditioning_diagnostics"
PROGRAM="$WORKSPACE/TriCompose-v1.2/tools/check_paired_conditioning_v2.py"
PROGRAM_SHA=b4d3f66736dc3fba6b5a9010cdb5dfcfd44b7baee8e19bc3533a8379c526a830
TESTS="$WORKSPACE/TriCompose-v1.2/tests/test_paired_conditioning_v2.py"
TESTS_SHA=b1b5215b8bdc1d77f7eaf8923e986c47de69a0d99c1573250dc80f644e58af0a
CACHE_TOOL="$WORKSPACE/TriCompose-v1.2/tools/read_paired_endpoint_cache_v1.py"
CACHE_SHA=9c0478ba6e10eab10223e5897103810d0098d2b37246a08093596c0977193e39
INPUT_TOOL="$WORKSPACE/TriCompose-v1.2/tools/score_paired_probes_v1.py"
INPUT_SHA=048b6c0668c5e9ce2ab0b646449cd2bbf05b3f0e39602d32ff96780ceb855b76
cd "$WORKSPACE"
chmod 0660 "$BASE/slurm_logs/tri_paired_trace_${SLURM_JOB_ID}.out" \
  "$BASE/slurm_logs/tri_paired_trace_${SLURM_JOB_ID}.err"
if test -e "$TASK" || test -e "$OUTPUT/$RUN_ID"; then
  printf 'status=refused reason=existing_run\n'
  exit 1
fi
ACTUAL_PROGRAM="$(sha256sum "$PROGRAM")"
ACTUAL_TESTS="$(sha256sum "$TESTS")"
ACTUAL_CACHE="$(sha256sum "$CACHE_TOOL")"
ACTUAL_INPUT="$(sha256sum "$INPUT_TOOL")"
if test "${ACTUAL_PROGRAM%% *}" != "$PROGRAM_SHA" || \
   test "${ACTUAL_TESTS%% *}" != "$TESTS_SHA" || \
   test "${ACTUAL_CACHE%% *}" != "$CACHE_SHA" || \
   test "${ACTUAL_INPUT%% *}" != "$INPUT_SHA"; then
  printf 'status=refused reason=source_changed\n'
  exit 1
fi
mkdir -m 2770 "$TASK"
mkdir -m 2770 "$TASK/tmp" "$TASK/cache"
export TRICOMPOSE_SCOPE=approved_synthetic_trace_diagnostic
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="$TASK/tmp" TEMP="$TASK/tmp" TMP="$TASK/tmp"
export XDG_CACHE_HOME="$TASK/cache" HF_HOME="$TASK/cache/huggingface"
export TORCH_HOME="$TASK/cache/torch" MPLCONFIGDIR="$TASK/cache/matplotlib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if ! "$WORKSPACE/cxrmate/venv/bin/python" "$PROGRAM" \
  --output-root "$OUTPUT" --run-id "$RUN_ID" \
  >"$TASK/worker.stdout.log" 2>"$TASK/worker.stderr.log"; then
  printf 'status=failed reason=protected_trace_diagnostic\n'
  exit 1
fi
printf 'status=completed_trace_diagnostic_not_clinical\n'
