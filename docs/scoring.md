# Report scoring and CXR-report verification

## Purpose

This layer turns frozen-model outputs into versioned score bundles that a
future TriCompose candidate graph or router can consume without knowing the
implementation details of each evaluator.

Every producer writes `tricompose.score_bundle.v1`, defined in
`schemas/score_bundle_v1.schema.json`. A record contains:

- metric name;
- scope (`report_reference`, `report_pair`, or `cxr_report`);
- opaque candidate IDs;
- numeric value;
- whether higher is better.

Raw reports, rendered images, and verifier rationales remain protected. Public
logs contain status, hashes, runtime, dimensions, memory, and call counts only.

## Report metrics

`tricompose.scoring.report_metrics` implements:

- BLEU-1;
- BLEU-2;
- BLEU-3;
- ROUGE-L;
- METEOR.

Phase-0 uses dependency-free, versioned definitions:

- lowercase ASCII alphanumeric tokenization;
- sentence BLEU-N with uniform weights and fixed epsilon smoothing;
- token LCS ROUGE-L F1;
- exact-token METEOR with fragmentation penalty, without stemming or WordNet
  synonym expansion.

For a paired ground-truth report, use `--comparison-role reference`. That mode
is reference-based accuracy evaluation and is not run in the current
real-anchor smoke test because loading the real target report is out of scope.

For two generated reports, use `--comparison-role peer`. The output contains
both directional scores and their symmetric mean. These are candidate
agreement features only: they cannot identify which report is correct.

Current protected peer comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  -m tricompose.scoring.report_metrics \
  --run-id real_smoke_003 \
  --candidate-a artifacts/protected/real_smoke_003/stage2_unidisc/generated_report.txt \
  --candidate-a-id unidisc \
  --candidate-b artifacts/protected/real_smoke_003/stage2_llavarad/generated_report.txt \
  --candidate-b-id llavarad \
  --comparison-role peer \
  --output-dir artifacts/protected/real_smoke_003/verification/report_pair_v1
```

## Qwen-VL CXR-report score

`tricompose.verifiers.qwenvl_cxr_report` loads one frozen Qwen-VL model, one
protected synthetic CXR, and one or more protected report candidates. It asks
the model for a deterministic JSON score in `[0, 1]` based on findings,
absence statements, laterality, support devices, and contradictions.

The available local checkpoint is `Qwen2.5-VL-7B-Instruct`, not the older
Qwen2-VL checkpoint. The verifier supports either `qwen2_vl` or `qwen2_5_vl`
model types when a complete local checkpoint is supplied.

Important limitation: Qwen-VL has no native contrastive image-report matching
head. This value is a generated VLM judge score, not a calibrated probability
or embedding similarity. Before paper-scale use, calibrate its prompt and
thresholds on:

- real matched pairs;
- shuffled pairs;
- disease-conflicting hard negatives;
- laterality, negation, severity, and device-location perturbations.

The current Slurm job scores UniDisc and LLaVA-Rad in one model load and writes:

```text
artifacts/protected/<run_id>/verification/qwen25vl_v1/qwen_vl_scores.json
```

The protected bundle includes candidate scores and short judge rationales.
Public Slurm logs do not include those values or rationales.

## Agent integration

The score-feature contract is in
`configs/scoring/report_selection_v1.json`. The executable Phase-0 report
selection policy is in
`configs/agent/report_selector_phase0_v1.json`, and its output follows
`schemas/agent_decision_v1.schema.json`.

- Qwen score is a candidate-level CXR-report feature.
- BLEU/ROUGE-L/METEOR in peer mode are pair-level disagreement features.
- Reference-mode report metrics are evaluation-only features.
- The current deterministic policy emits `select` only when both its minimum
  Qwen score and top-two margin pass; otherwise it emits `verify_more`.
- A `select` result is explicitly `provisional`, because Qwen calibration and
  multi-case validation are not complete.
- The decision builder verifies report hashes against the reports actually
  scored by Qwen, preventing stale-score selection.

Run the lightweight selector after the protected Qwen bundle exists:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  -m tricompose.agent.report_selector \
  --run-id real_smoke_003 \
  --policy-config configs/agent/report_selector_phase0_v1.json \
  --qwen-score-bundle artifacts/protected/real_smoke_003/verification/qwen25vl_v1/qwen_vl_scores.json \
  --peer-score-bundle artifacts/protected/real_smoke_003/verification/report_pair_v1/report_metrics.json \
  --candidate unidisc unidisc artifacts/protected/real_smoke_003/stage2_unidisc/generated_report.txt \
  --candidate llavarad llavarad artifacts/protected/real_smoke_003/stage2_llavarad/generated_report.txt \
  --output-dir artifacts/protected/real_smoke_003/agent/report_selection_v1
```

This command performs no model inference and can run on the login node. Its
protected output is `agent_decision.json`; stdout contains only status,
artifact filename, and hash.

The candidate graph should index score records by:

```text
(run_id, metric, scope, candidate_ids)
```

It should retain verifier version, prompt version, model-call count, runtime,
peak GPU memory, thresholds, margin, and the emitted action for cost-aware
routing. The decision schema already reserves the action vocabulary
`select`, `verify_more`, `regenerate_report`, and `stop`; the Phase-0 policy
currently emits only the first two.
