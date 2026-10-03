#!/bin/bash
# Show this complete script/resources and obtain approval before sbatch.
# Secondary finding extraction only; no generation, training, reranking or repair.
#SBATCH --job-name=tri_v12_qwen_sep
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
MODEL=/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_1"
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
SOURCE="$BASE/evaluation/v12_explicit_two_12538545_001/eight_head_rescore_12550488"
TOOLS="$WORKSPACE/TriCompose-v1.2/benchmarks"
RUNS="$ROOT/verification_runs"
TASKS="$ROOT/task_runtime"
TASK="$TASKS/qwen_separated_${SLURM_JOB_ID}"
RUN_ID="qwen_separated_${SLURM_JOB_ID}"
CROSS_HASH=7c859e3ad3345b6128b8a6dbc88bfe5a4261dd634bbca3f37b11e670260ff586

test -x "$PYTHON"
test -r "$MODEL/model.safetensors"
printf '%s  %s\n' \
  0e5cb1cd578ecef6df8e2ef7eab89c93dcfd3d905ca514b143ed4eb8b8275a7c "$MODEL/config.json" \
  3ce5299c53ec2a6713fff97be266b35c44fdf3e3c7ce2298e9a6af0fb3ed75de "$SOURCE/selection/manifest.json" \
  "$CROSS_HASH" "$SOURCE/crossmodal/crossmodal_details.json" \
  24559478390018f17bd08cb20b5b0f3488ed0a27b8c431ab9a5930e225d1c2e1 "$BASE/cxr_candidates/v12_explicit_two_roentgen_v2_s2_12533465/manifest.json" \
  3facb7ecd2aea8e026b47fdea9f729d96643387ea518d2764344c6a7ffad9e99 "$BASE/cxr_candidates/v12_explicit_two_chexgenbench_sana_s2_12533465/manifest.json" \
  8497a4aeeee44d12865264dfe2840870622777d2a36c78b2c29edbf8cac076ca "$BASE/cxr_candidates/v12_explicit_two_chexgenbench_pixart_s2_12533465/manifest.json" \
  f8f8636d5b5fd9de9367c54f80a94039697997088fc7d685fcaf46a5a4f6d819 "$BASE/report_candidates/v12_explicit_two_maira2_12538545/manifest.json" \
  84cb2e4d61edd038cdafcc6c533914506290433f833bff1a4366f0381585105e "$BASE/report_candidates/v12_explicit_two_cxrmate_single_12538545/manifest.json" \
  603b5f6a2108d41d6c44659d31d83bb9d665d2373478fbae79e3fe4cea7e7504 "$BASE/report_candidates/v12_explicit_two_llavarad_12538545/manifest.json" \
  ee28dafdab6533f50a29ffa2c6d0d1a26faa2543263cb14be5b09b5b6a516408 "$BASE/report_candidates/v12_explicit_two_chexagent2_12538545/manifest.json" \
  | sha256sum --check --status
test ! -e "$TASK"
test ! -e "$RUNS/$RUN_ID"
test ! -e "$RUNS/${RUN_ID}_analysis"
mkdir -p -m 2770 "$RUNS" "$TASKS"
chmod 2770 "$RUNS" "$TASKS"
mkdir -m 2770 "$TASK" "$TASK/cache" "$TASK/tmp" "$TASK/logs"
chmod 0660 "$ROOT/slurm_logs/tri_v12_qwen_sep_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_qwen_sep_${SLURM_JOB_ID}.err"

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/TriCompose-v1.0/eval/report_v1_1:$TOOLS"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$TASK/cache/xdg" HF_HOME="$TASK/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$TASK/cache/torch" CUDA_CACHE_PATH="$TASK/cache/cuda"
export TRITON_CACHE_DIR="$TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$TASK/cache/kernels" MPLCONFIGDIR="$TASK/cache/matplotlib"
export TMPDIR="$TASK/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

CXR_ARGS=()
for model in roentgen_v2 chexgenbench_sana chexgenbench_pixart; do
  CXR_ARGS+=(--cxr-run "$BASE/cxr_candidates/v12_explicit_two_${model}_s2_12533465")
done
REPORT_ARGS=()
for model in maira2 cxrmate_single llavarad chexagent2; do
  REPORT_ARGS+=(--report-run "$BASE/report_candidates/v12_explicit_two_${model}_12538545")
done
"$PYTHON" "$TOOLS/verify_candidate_findings_qwen.py" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" --model-path "$MODEL" \
  --expected-images 12 --expected-reports 48 --max-new-tokens 384 \
  --output-root "$RUNS" --run-id "$RUN_ID" >"$TASK/logs/verify.log" 2>&1
printf 'status=separated_secondary_verification_completed\n'
"$PYTHON" "$TOOLS/analyze_qwen_evidence.py" --review-run "$RUNS/$RUN_ID" \
  --selection-run "$SOURCE/selection" \
  --crossmodal-details "$SOURCE/crossmodal/crossmodal_details.json" \
  --crossmodal-sha256 "$CROSS_HASH" --output-root "$RUNS" \
  --run-id "${RUN_ID}_analysis" >"$TASK/logs/analysis.log" 2>&1
find "$TASK" -type d -exec chmod 2770 {} +
find "$TASK" -type f -exec chmod 0660 {} +
printf 'status=secondary_evidence_analysis_completed_no_selection_change\n'
