#!/bin/bash
#SBATCH --job-name=tri_v12_ehr100
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --constraint=p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_2/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/logs/slurm/tricompose_v1_2/%x_%j.err

set -euo pipefail
umask 007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/runtime/venvs/roentgen-v2-cu126-v1/bin/python"
MODEL_DIR="$WORKSPACE/runtime/models/synehrgy_v2/qwen2-40bins"
EXPERIMENT="$WORKSPACE/experiments/synehrgy_v2"
POOL_ID=v12_explicit_ehr_pool100_20261001_001
SCREEN_ID=v12_explicit_ehr_screen_20261001_001
STAGE_ID=v12_explicit_ehr_two_20261001_001
POOL="$WORKSPACE/artifacts/protected/tricompose_v1_2/ehr_pools/$POOL_ID"
SCREEN="$WORKSPACE/artifacts/protected/tricompose_v1_2/ehr_cohorts/$SCREEN_ID"
STAGE_V1="$WORKSPACE/artifacts/protected/tricompose_v1/$STAGE_ID"
STAGE_V11="$WORKSPACE/artifacts/protected/tricompose_v1_1/staging/$STAGE_ID"
TASK_CACHE="$WORKSPACE/.cache/tricompose_v12/ehr_${SLURM_JOB_ID}"
TASK_TMP="$WORKSPACE/.tmp/tricompose_v12/ehr_${SLURM_JOB_ID}"

test -x "$PYTHON" || { echo 'status=failed reason=missing_environment' >&2; exit 1; }
test -d "$MODEL_DIR" || { echo 'status=failed reason=missing_checkpoint' >&2; exit 1; }
for target in "$POOL" "$SCREEN" "$STAGE_V1" "$STAGE_V11"; do
  test ! -e "$target" || { echo 'status=failed reason=run_exists' >&2; exit 1; }
done

restore_permissions() {
  set +e
  chmod 2770 "$WORKSPACE/artifacts/protected"
  for target in "$POOL" "$SCREEN" "$STAGE_V1" "$STAGE_V11"; do
    if test -d "$target"; then
      find -P "$target" -type d -exec chmod 2770 {} +
      find -P "$target" -type f -exec chmod 0660 {} +
    fi
  done
}
trap restore_permissions EXIT

chmod 2770 "$WORKSPACE/artifacts/protected"
mkdir -p "$TASK_CACHE" "$TASK_TMP"
chmod 2770 "$TASK_CACHE" "$TASK_TMP"
export PYTHONPATH="$WORKSPACE/src:$EXPERIMENT/src:$WORKSPACE/TriCompose-v1.0/src:$WORKSPACE/TriCompose-v1.1/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK_CACHE"
export HF_HOME="$TASK_CACHE/huggingface"
export TORCH_HOME="$TASK_CACHE/torch"
export TMPDIR="$TASK_TMP"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# Frozen, unconditional synthetic EHR generation. All 100 outputs are retained.
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/benchmarks/generate_unconditional_ehr_pool.py" \
  --model-dir "$MODEL_DIR" \
  --run-id "$POOL_ID" \
  --count 100 \
  --seed 5200 \
  --temperature 1.0 \
  --top-k 50

# Predeclared CPU-only eligibility screen; abort if fewer than two qualify.
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/benchmarks/select_explicit_ehr_cohort.py" \
  --source-run "$POOL" \
  --expected-source-run-id "$POOL_ID" \
  --output-run-id "$SCREEN_ID" \
  --expected-count 100 \
  --expected-base-seed 5200 \
  --target 2

"$PYTHON" "$WORKSPACE/tools/stage_tricompose_v1.py" \
  --source-root "$POOL" \
  --case-ids-file "$SCREEN/selected_case_ids.txt" \
  --output-root "$WORKSPACE/artifacts/protected/tricompose_v1" \
  --run-id "$STAGE_ID" \
  --prompt-models roentgen_v2,chexgenbench_sana,chexgenbench_pixart \
  --validate

"$PYTHON" "$WORKSPACE/TriCompose-v1.1/tools/stage_existing_v1_ehrs.py" \
  --source-v1-run "$STAGE_V1" \
  --output-root "$WORKSPACE/artifacts/protected/tricompose_v1_1/staging" \
  --run-id "$STAGE_ID"

echo 'status=complete phase=synthetic_ehr_and_prompt_staging cases=2'
