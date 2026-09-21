#!/bin/bash
#SBATCH --job-name=tri_v1_score
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:55:00
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

XRV_RUN=v1_smoke2_xrv_20260806_001
BIOVIL_RUN=v1_smoke2_biovil_20260806_001
QWENVL_RUN=v1_smoke2_qwenvl_20260806_001
SELECTION_RUN=v1_smoke2_static_selection_20260806_001

XRV_PYTHON=/project2/ruishanl_1185/yikeyang_medim_ehr_joint/work/.venv/bin/python
BIOVIL_PYTHON="$WORKSPACE/CheXGenBench/metrics_venv/bin/python"
QWENVL_PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
BASE_PYTHON=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/bin/python

XRV_SOURCE_ROOT=/home1/yikeyang/.torchxrayvision/models_data
XRV_WEIGHT=nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
BIOVIL_MODEL="$WORKSPACE/CheXGenBench/checkpoints/official-metrics/biovil-t/301f526e823b805d3fe712d0cf06a66f042789c5"
BIOVIL_VENDOR="$WORKSPACE/runtime/vendor/hi_ml_multimodal_0_2_2"
QWENVL_MODEL=/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct

TASK_CACHE="$WORKSPACE/.cache/tricompose_v1/scoring/job_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v1/scoring/job_${SLURM_JOB_ID}"
mkdir -p "$TASK_CACHE" "$TASK_TMP" "$WORKSPACE/logs/slurm/tricompose_v1"
chmod 700 "$TASK_CACHE" "$TASK_TMP" "$WORKSPACE/logs/slurm/tricompose_v1"
XRV_CACHE="$TASK_CACHE/xrv"
mkdir -p "$XRV_CACHE"
chmod 700 "$XRV_CACHE"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/torchinductor"
export MPLCONFIGDIR="$TASK_CACHE/matplotlib"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for executable in "$XRV_PYTHON" "$BIOVIL_PYTHON" "$QWENVL_PYTHON" "$BASE_PYTHON"; do
  if ! test -x "$executable"; then
    printf 'status=failed reason=missing_python_environment\n' >&2
    exit 1
  fi
done
for directory in "$BANK" "$XRV_SOURCE_ROOT" "$BIOVIL_MODEL" "$BIOVIL_VENDOR" "$QWENVL_MODEL"; do
  if ! test -d "$directory"; then
    printf 'status=failed reason=missing_frozen_input\n' >&2
    exit 1
  fi
done
if ! test -f "$XRV_SOURCE_ROOT/$XRV_WEIGHT"; then
  printf 'status=failed reason=missing_frozen_xrv_weight\n' >&2
  exit 1
fi
install -m 600 "$XRV_SOURCE_ROOT/$XRV_WEIGHT" "$XRV_CACHE/$XRV_WEIGHT"
for run_id in "$XRV_RUN" "$BIOVIL_RUN" "$QWENVL_RUN" "$SELECTION_RUN"; do
  if test -e "$SCORE_ROOT/$run_id"; then
    printf 'status=failed reason=score_run_already_exists\n' >&2
    exit 1
  fi
done

PYTHONPATH="$WORKSPACE/src:$V1_ROOT/src:$WORKSPACE/experiments/medim/src" \
  "$XRV_PYTHON" "$TOOLS/score_tricompose_v1_xrv.py" \
  --candidate-bank "$BANK" \
  --output-root "$SCORE_ROOT" \
  --run-id "$XRV_RUN" \
  --cache-dir "$XRV_CACHE"

PYTHONPATH="$WORKSPACE/src:$V1_ROOT/src:$BIOVIL_VENDOR" \
  "$BIOVIL_PYTHON" "$TOOLS/score_tricompose_v1_biovil.py" \
  --candidate-bank "$BANK" \
  --model-path "$BIOVIL_MODEL" \
  --output-root "$SCORE_ROOT" \
  --run-id "$BIOVIL_RUN"

PYTHONPATH="$WORKSPACE/src:$V1_ROOT/src" \
  "$QWENVL_PYTHON" "$TOOLS/score_tricompose_v1_qwenvl.py" \
  --candidate-bank "$BANK" \
  --model-path "$QWENVL_MODEL" \
  --output-root "$SCORE_ROOT" \
  --run-id "$QWENVL_RUN" \
  --max-new-tokens 384 \
  --min-pixels 200704 \
  --max-pixels 401408

PYTHONPATH="$WORKSPACE/src:$V1_ROOT/src" \
  "$BASE_PYTHON" "$TOOLS/aggregate_tricompose_v1_scores.py" \
  --candidate-bank "$BANK" \
  --xrv-run "$SCORE_ROOT/$XRV_RUN" \
  --biovil-run "$SCORE_ROOT/$BIOVIL_RUN" \
  --qwenvl-run "$SCORE_ROOT/$QWENVL_RUN" \
  --policy "$POLICY" \
  --output-root "$SCORE_ROOT" \
  --run-id "$SELECTION_RUN"

printf 'stage=tricompose_v1_score_all status=completed\n'
