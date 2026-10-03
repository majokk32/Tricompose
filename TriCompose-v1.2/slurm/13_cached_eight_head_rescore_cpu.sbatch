#!/bin/bash
# Show the complete script/resources and obtain approval before sbatch.
# Cached diagnostic scores only: no inference, generation or training.
#SBATCH --job-name=tri_v12_rescore48
#SBATCH --account=ruishanl_1185
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --chdir=/project2/ruishanl_1185/inference_3mod
#SBATCH --output=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.out
#SBATCH --error=/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/slurm_logs/%x_%j.err

set -euo pipefail
umask 0007
WORKSPACE=/project2/ruishanl_1185/inference_3mod
PYTHON="$WORKSPACE/cxrmate/venv/bin/python"
EVAL="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
ROOT="$WORKSPACE/artifacts/protected/tricompose_v1_2"
BASE="$WORKSPACE/artifacts/protected/tricompose_v1_1"
PILOT="$BASE/evaluation/v12_explicit_two_12538545_001"
OLD="$PILOT/scoring_12538856"
CAL="$ROOT/real_validation/real_xrv_thresholds_12549079"
IMAGE_SUMMARY="$ROOT/real_validation/real_xrv_reference_12549079/summary.json"
STAGING="$BASE/staging/v12_explicit_ehr_two_20261001_001"
OUT="$PILOT/eight_head_rescore_${SLURM_JOB_ID}"

printf '%s  %s\n' \
  e44d7fa93a52eed2a0a1d1fa6470a0a5dfdab8282b550886eae74bb94a02a19c "$PILOT/registry/manifest.json" \
  bee3a47aee23b738c67a2b0443bebaab3db77e1c52807fcdc30e56697cb42a5c "$CAL/cxr_finding_labels.json" \
  5f7a8cc6ad9b480cbf532dc3a5b5d555c3e92f69782fa52351cd12fe2df848cc "$CAL/evaluation_summary.json" \
  fcd9cd27c788f0b5c2c187fc5325b3f3e8449fe0df4c8d1759c07965b2ea176c "$IMAGE_SUMMARY" \
  | sha256sum --check --status
test ! -e "$OUT"
mkdir -m 2770 "$OUT"
mkdir -m 2770 "$OUT/logs" "$OUT/cache" "$OUT/tmp"
chmod 0660 "$ROOT/slurm_logs/tri_v12_rescore48_${SLURM_JOB_ID}.out" \
  "$ROOT/slurm_logs/tri_v12_rescore48_${SLURM_JOB_ID}.err"

export PYTHONPATH="$WORKSPACE/TriCompose-v1.2/real_validation:$EVAL"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$OUT/cache/xdg" HF_HOME="$OUT/cache/huggingface" TMPDIR="$OUT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CXR_ARGS=()
for model in roentgen_v2 chexgenbench_sana chexgenbench_pixart; do
  CXR_ARGS+=(--cxr-run "$BASE/cxr_candidates/v12_explicit_two_${model}_s2_12533465")
done
REPORT_ARGS=()
for model in maira2 cxrmate_single llavarad chexagent2; do
  REPORT_ARGS+=(--report-run "$BASE/report_candidates/v12_explicit_two_${model}_12538545")
done

"$PYTHON" "$EVAL/aggregate_crossmodal.py" --staging-run "$STAGING" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" --cxr-labels "$CAL/cxr_finding_labels.json" \
  --report-labels "$OLD/chexbert/report_finding_labels.json" --biovil-scores "$OLD/biovil/biovil_scores.json" \
  --output-root "$OUT" --run-id crossmodal >"$OUT/logs/crossmodal.log" 2>&1
"$PYTHON" "$EVAL/aggregate_ehr_edges.py" --staging-run "$STAGING" \
  "${CXR_ARGS[@]}" "${REPORT_ARGS[@]}" --cxr-labels "$CAL/cxr_finding_labels.json" \
  --report-labels "$OLD/chexbert/report_finding_labels.json" \
  --output-root "$OUT" --run-id ehr_edges >"$OUT/logs/ehr_edges.log" 2>&1
"$PYTHON" "$EVAL/finalize_unified_score_table.py" --score-table-run "$PILOT/registry" \
  --crossmodal-run "$OUT/crossmodal" --policy-config "$EVAL/static_score_policy_v1_1.json" \
  --output-root "$OUT" --run-id scored >"$OUT/logs/scored.log" 2>&1
"$PYTHON" "$EVAL/select_edge_specific_candidates.py" --score-table-run "$OUT/scored" \
  --ehr-edge-run "$OUT/ehr_edges" --policy-config "$EVAL/lexicographic_selection_policy_smoke48.json" \
  --output-root "$OUT" --run-id selection >"$OUT/logs/selection.log" 2>&1
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/benchmarks/analyze_candidate_scorers.py" \
  --selection-run "$OUT/selection" --crossmodal-details "$OUT/crossmodal/crossmodal_details.json" \
  --report-structure "$PILOT/structure/report_unimodal_details.json" \
  --output-root "$OUT" --run-id audit >"$OUT/logs/audit.log" 2>&1
"$PYTHON" "$WORKSPACE/TriCompose-v1.2/benchmarks/annotate_scoring_scope.py" \
  --selection-run "$OUT/selection" --image-summary "$IMAGE_SUMMARY" \
  --threshold-summary "$CAL/evaluation_summary.json" \
  --output-root "$OUT" --run-id scoped >"$OUT/logs/scoped.log" 2>&1
printf 'status=cached_diagnostic_rescore_completed\n'
