#!/bin/bash
set -euo pipefail
umask 0007

WORKSPACE=/project2/ruishanl_1185/inference_3mod
EVAL_ROOT="$WORKSPACE/TriCompose-v1.0/eval/report_v1_1"
PROTECTED="$WORKSPACE/artifacts/protected/tricompose_v1_1"
PYTHON=python

STAGING="$PROTECTED/staging/bridge_gpt2_pool80_v11_20260813_003"
CANDIDATE_REGISTRY="$PROTECTED/evaluation/unified_score_tables/score_table_pool960_skeleton_20260813_001"
XRV_LABELS="$PROTECTED/evaluation/finding_labels/xrv_pool240_uncalibrated_20260813_001/cxr_finding_labels.json"
CHEXBERT_LABELS="$PROTECTED/evaluation/finding_labels/chexbert_report_pool960_20260813_001/report_finding_labels.json"

CXR_SANA="$PROTECTED/cxr_candidates/pool80_chexgenbench_sana_s1_20260813_d0ce"
CXR_PIXART="$PROTECTED/cxr_candidates/pool80_chexgenbench_pixart_s1_20260813_18b7"
CXR_ROENTGEN="$PROTECTED/cxr_candidates/pool80_roentgen_v2_fp16_s1_20260813_0bd4"
RPT_MAIRA_A="$PROTECTED/report_candidates/pool160_maira2_20260813_2f6a"
RPT_MAIRA_B="$PROTECTED/report_candidates/roentgen80_maira2_20260813_a31e"
RPT_CXRMATE_A="$PROTECTED/report_candidates/pool160_cxrmate_single_20260813_74d1"
RPT_CXRMATE_B="$PROTECTED/report_candidates/roentgen80_cxrmate_single_20260813_d6a4"
RPT_LLAVA_A="$PROTECTED/report_candidates/pool160_llavarad_20260813_aa39"
RPT_LLAVA_B="$PROTECTED/report_candidates/roentgen80_llavarad_20260813_b472"
RPT_CHEXA_A="$PROTECTED/report_candidates/pool160_chexagent2_fp32_20260813_51f8"
RPT_CHEXA_B="$PROTECTED/report_candidates/roentgen80_chexagent2_fp32_20260813_c593"

CROSSMODAL_ROOT="$PROTECTED/evaluation/crossmodal"
CROSSMODAL_RUN_ID=crossmodal_primary_pool960_20260813_001
SCORE_ROOT="$PROTECTED/evaluation/unified_score_tables"
SCORE_RUN_ID=score_table_primary_pool960_20260813_001
BASELINE_ROOT="$PROTECTED/evaluation/selection_baselines"
BASELINE_RUN_ID=baselines_primary_pool960_20260813_001

for required in "$XRV_LABELS" "$CHEXBERT_LABELS" "$CANDIDATE_REGISTRY/manifest.json"; do
  if ! test -f "$required"; then
    printf 'status=failed reason=missing_primary_evidence\n' >&2
    exit 1
  fi
done

export PYTHONPATH="$EVAL_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

"$PYTHON" "$EVAL_ROOT/aggregate_crossmodal.py" \
  --staging-run "$STAGING" \
  --cxr-run "$CXR_SANA" --cxr-run "$CXR_PIXART" --cxr-run "$CXR_ROENTGEN" \
  --report-run "$RPT_MAIRA_A" --report-run "$RPT_MAIRA_B" \
  --report-run "$RPT_CXRMATE_A" --report-run "$RPT_CXRMATE_B" \
  --report-run "$RPT_LLAVA_A" --report-run "$RPT_LLAVA_B" \
  --report-run "$RPT_CHEXA_A" --report-run "$RPT_CHEXA_B" \
  --report-labels "$CHEXBERT_LABELS" \
  --cxr-labels "$XRV_LABELS" \
  --output-root "$CROSSMODAL_ROOT" \
  --run-id "$CROSSMODAL_RUN_ID"

"$PYTHON" "$EVAL_ROOT/finalize_unified_score_table.py" \
  --score-table-run "$CANDIDATE_REGISTRY" \
  --crossmodal-run "$CROSSMODAL_ROOT/$CROSSMODAL_RUN_ID" \
  --policy-config "$EVAL_ROOT/static_score_policy_v1_1.json" \
  --output-root "$SCORE_ROOT" \
  --run-id "$SCORE_RUN_ID"

"$PYTHON" "$EVAL_ROOT/run_selection_baselines.py" \
  --score-table-run "$SCORE_ROOT/$SCORE_RUN_ID" \
  --random-repetitions 100 \
  --output-root "$BASELINE_ROOT" \
  --run-id "$BASELINE_RUN_ID"
