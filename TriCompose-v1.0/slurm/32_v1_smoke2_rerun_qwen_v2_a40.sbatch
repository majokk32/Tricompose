#!/bin/bash
#SBATCH --job-name=tri_v1_qwen_v2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1/%x_%j.err

set -euo pipefail
umask 077

WORKSPACE=/project2/ruishanl_1185/inference_3mod
V1_ROOT="$WORKSPACE/TriCompose-v1.0"
TOOLS="$WORKSPACE/tools"
BANK="$WORKSPACE/artifacts/protected/tricompose_v1/candidate_banks/v1_smoke2_all_models_20260806_001"
SCORE_ROOT="$WORKSPACE/artifacts/protected/tricompose_v1/scores"
POLICY="$V1_ROOT/configs/policy_v1.json"

XRV_RUN="$SCORE_ROOT/v1_smoke2_xrv_20260806_001"
BIOVIL_RUN="$SCORE_ROOT/v1_smoke2_biovil_20260806_001"
QWENVL_RUN_ID=v1_smoke2_qwenvl_v2_20260806_001
SELECTION_RUN_ID=v1_smoke2_static_selection_v2_20260806_001

QWENVL_PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
BASE_PYTHON=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/bin/python
QWENVL_MODEL=/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct

TASK_CACHE="$WORKSPACE/.cache/tricompose_v1/qwen_v2/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1/qwen_v2/job_${SLURM_JOB_ID}"
mkdir -p "$TASK_CACHE" "$TASK_TMP" "$WORKSPACE/logs/slurm/tricompose_v1"
chmod 700 "$TASK_CACHE" "$TASK_TMP" "$WORKSPACE/logs/slurm/tricompose_v1"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/torchinductor"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for executable in "$QWENVL_PYTHON" "$BASE_PYTHON"; do
  if ! test -x "$executable"; then
    printf 'status=failed reason=missing_python_environment\n' >&2
    exit 1
  fi
done
for directory in "$BANK" "$XRV_RUN" "$BIOVIL_RUN" "$QWENVL_MODEL"; do
  if ! test -d "$directory"; then
    printf 'status=failed reason=missing_frozen_input\n' >&2
    exit 1
  fi
done
for run_id in "$QWENVL_RUN_ID" "$SELECTION_RUN_ID"; do
  if test -e "$SCORE_ROOT/$run_id"; then
    printf 'status=failed reason=score_run_already_exists\n' >&2
    exit 1
  fi
done

PYTHONPATH="$WORKSPACE/src:$V1_ROOT/src" \
  "$QWENVL_PYTHON" "$TOOLS/score_tricompose_v1_qwenvl.py" \
  --candidate-bank "$BANK" \
  --model-path "$QWENVL_MODEL" \
  --output-root "$SCORE_ROOT" \
  --run-id "$QWENVL_RUN_ID" \
  --max-new-tokens 256 \
  --min-pixels 200704 \
  --max-pixels 401408

PYTHONPATH="$WORKSPACE/src:$V1_ROOT/src" \
  "$BASE_PYTHON" "$TOOLS/aggregate_tricompose_v1_scores.py" \
  --candidate-bank "$BANK" \
  --xrv-run "$XRV_RUN" \
  --biovil-run "$BIOVIL_RUN" \
  --qwenvl-run "$SCORE_ROOT/$QWENVL_RUN_ID" \
  --policy "$POLICY" \
  --output-root "$SCORE_ROOT" \
  --run-id "$SELECTION_RUN_ID"

printf 'stage=tricompose_v1_qwen_v2_and_selection status=completed\n'
