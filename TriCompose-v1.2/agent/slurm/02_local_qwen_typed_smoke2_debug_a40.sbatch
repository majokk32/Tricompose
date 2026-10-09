#!/bin/bash
# PROPOSED ONLY: obtain explicit approval after showing this entire script.
# Typed-content fix only. Frozen Qwen; numeric cache, NOT new generation.
#SBATCH --job-name=tri_llm_typed2
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON=/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python
MODEL=/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct
AGENT="$WORKSPACE/TriCompose-v1.2/agent"
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
RUN_ID="llm_typed_smoke2_${SLURM_JOB_ID}"
LLM_TASK="$ROOT/llm_agent_runs/task_${SLURM_JOB_ID}"

test -x "$PYTHON"
test -r "$MODEL/model.safetensors"
for directory in "$ROOT" "$ROOT/slurm_logs" "$ROOT/llm_agent_runs"; do
  test "$(stat -c %a "$directory")" = 2770
  project_gid=$(stat -c %g "$directory")
  test "$project_gid" = 96293 || test "$project_gid" = 65534
done
printf '%s  %s\n' \
  5c0e68865cb4a907fa21ba1c1eaa4deef73ed053973f5ef37dce632491db8100 "$AGENT/run_local_qwen_v2.py" \
  e6246a2f057f47e0df8f2849170fbddcadb32ba1afe231e1d83ef3009b11c09a "$AGENT/tricompose_llm/local_qwen_v2.py" \
  8a58ba4b30619ae7fffce9bb394f9ab94170da5eb92d103a56e3af355ebf156f "$AGENT/run.py" \
  3132f0519501f3db50b6a50f284ed14ec8185831d95075c682a8351044c90b3f "$AGENT/tricompose_llm/__init__.py" \
  d0aee1ac579a085de7734af9dbd559960b8ff1a1ec589ead7ef71d7bdeccf9ea "$AGENT/tricompose_llm/contracts.py" \
  dff49516fa6d5d7d78db19af150392ed628a2439cecaad22bf51a1693d6d2209 "$AGENT/tricompose_llm/controller.py" \
  c322d0c1d91ef2da3a30e75e78be195559a5d5ad5b36d05ca895a7c6c43acab5 "$AGENT/tricompose_llm/demo.py" \
  227d5bd5c0a67a878fb3cd819f4466e3aac5b5267811f4801e6839139b403bb2 "$AGENT/tricompose_llm/local_qwen.py" \
  157feb416b81be9cc381bc3e617cfe02e95aed450eda5d32bd207ff825588951 "$AGENT/tricompose_llm/planner.py" \
  | sha256sum --check --status
test ! -e "$LLM_TASK"
test ! -e "$ROOT/llm_agent_runs/$RUN_ID"
mkdir -m 2770 "$LLM_TASK" "$LLM_TASK/cache" "$LLM_TASK/tmp"
chmod 0660 "$ROOT/slurm_logs/tri_llm_typed2_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_llm_typed2_${SLURM_JOB_ID}.err"
finish_permissions() {
  find "$LLM_TASK" -type d -exec chmod 2770 {} +
  find "$LLM_TASK" -type f -exec chmod 0660 {} +
}
trap finish_permissions EXIT

unset TRICOMPOSE_LLM_API_KEY OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$LLM_TASK/cache/xdg" HF_HOME="$LLM_TASK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$LLM_TASK/cache/torch" CUDA_CACHE_PATH="$LLM_TASK/cache/cuda"
export TRITON_CACHE_DIR="$LLM_TASK/cache/triton" TORCHINDUCTOR_CACHE_DIR="$LLM_TASK/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$LLM_TASK/cache/kernels" MPLCONFIGDIR="$LLM_TASK/cache/matplotlib"
export TMPDIR="$LLM_TASK/tmp" TMP="$LLM_TASK/tmp" TEMP="$LLM_TASK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

"$PYTHON" "$AGENT/run_local_qwen_v2.py" \
  --mode cache-replay --planner local-qwen --model-path "$MODEL" \
  --case-count 2 --budget-units 16 --max-steps 4 \
  --output-root "$ROOT/llm_agent_runs" --run-id "$RUN_ID" \
  >"$LLM_TASK/driver.log" 2>&1
printf 'status=llm_typed_cached_routing_completed_no_generator_calls\n'
