# Expert error-type diagnostic — 2026-10-06

## What this adds / 本次新增

Following the [completed image-grounded benchmark](radeval_biovil_result_12714150_001.md),
this **post-hoc numerical diagnostic** separates the seven released significant-
error types. It consumes the sealed derived table only: no model inference,
raw report/EHR/image access, new label creation, fitting, regeneration or winner
change. It is not a replacement primary endpoint or independent confirmation.

本次发现：一个全局图文相似度分数可能减少“遗漏”，但不能同样可靠地
控制错误发现、位置和严重程度。不能把这些不同风险压成一个分数后直接
驱动修复。这里的结果是评分器诊断，不是生成模型优劣或已完成修复的证明。

## Fixed denominators and definitions

- All 624 original pairs retained; all four scores available on the same
  132 pairs / 44 three-report comparison anchors / 34 patient groups.
  The 492 unavailable-image pairs remain missing, not zero-filled.
- Seven types are the authors' annotation categories, not new CheXpert labels,
  clinician-created rules or disease prevalence estimates. Error counts compare
  a candidate report with a reference, not independently adjudicated image truth.
- For each type, strict pairwise accuracy asks: **among reports for the same
  anchor whose error counts differ, does the score rank the less erroneous
  report higher?** Equal error counts are excluded; score ties contribute 0.5.
  Its 50% chance reference is not a disease sensitivity/specificity measure.
- Mean error difference is original metric-max choice minus uniform random
  choice, using the unchanged three reports. Negative is favorable. Category
  components sum to the earlier total-error difference; no new model/seed is
  selected. Intervals: 1,000 whole-patient-cluster resamples, seed 0.
- All categories are reported, including sparse ones. Intervals are exploratory,
  not multiplicity-adjusted. No clinical threshold, rule or component is promoted.

## BioViL-T: different error types behave differently

| Significant-error category | Reports with this error / 132 | Strict report comparisons | Pairwise accuracy | Chosen-minus-random mean errors | 95% cluster interval |
| --- | ---: | ---: | ---: | ---: | --- |
| False prediction / 错误发现 | 97 | 92 | 44.57% | +0.0303 | [-0.3137, 0.2501] |
| Omission / 遗漏发现 | 96 | 82 | 68.29% | -0.3030 | [-0.4940, -0.0196] |
| Incorrect location / 位置错误 | 52 | 57 | 40.35% | +0.1288 | [-0.0392, 0.2745] |
| Incorrect severity / 严重程度错误 | 32 | 41 | 39.02% | +0.0227 | [-0.1078, 0.1212] |
| Unsupported comparison / 无依据的比较 | 26 | 40 | 60.00% | -0.1288 | [-0.3039, 0.0227] |
| Omitted change / 遗漏变化 | 46 | 46 | 46.74% | -0.0152 | [-0.1288, 0.1288] |
| Inarticulate report / 表达问题 | 7 | 14 | 35.71% | +0.0152 | [-0.0490, 0.0556] |

The favorable omission contribution (-0.3030) does not guarantee control of
false findings or details. Other components partially cancel it, reproducing
the original **-0.2500** total mean difference whose interval crosses zero.
This describes the observed trade-off; it does not establish that length,
image preprocessing, training overlap or any other factor caused it.

## Same-cohort comparison: keep every frozen metric

These are strict pairwise accuracies for the identical categories/denominators
above. RadGraph scores require the reference report; BioViL-T does not.

| Error type | BioViL-T | RadGraph entity | RadGraph relation-presence | RadGraph full-relation |
| --- | ---: | ---: | ---: | ---: |
| False prediction | 44.57% | 44.02% | 42.39% | 40.76% |
| Omission | 68.29% | 73.17% | 70.73% | 67.07% |
| Incorrect location | 40.35% | 29.82% | 23.68% | 30.70% |
| Incorrect severity | 39.02% | 36.59% | 39.02% | 42.68% |
| Unsupported comparison | 60.00% | 42.50% | 45.00% | 48.75% |
| Omitted change | 46.74% | 36.96% | 41.30% | 48.91% |
| Inarticulate report | 35.71% | 46.43% | 39.29% | 32.14% |

Do not select the largest entry in each row to build a reward: that would tune
on the inspected benchmark and does not create a reference-free deployment
metric. Comparisons overlap and are clustered; sparse rows are especially
unstable. Below-chance observed ranking is not proof of systematic inversion
or a reason to reverse the score after seeing these outcomes.

## Development implication

The next score contract should retain **separate support, contradiction,
coverage and unavailable evidence**, instead of treating global similarity as
factual truth. Location/severity/temporal checks need evidence appropriate to
their scope; a single image cannot establish change without a valid prior.
Independent calibration/validation is still required before any such check
can localize a faulty modality or authorize repair. A finding not measured by
the verifier remains unknown, not negative or implicitly accepted.

This experiment does not measure EHR edges, negation-specific detection,
laterality-specific accuracy, synthetic-domain transfer, actual regeneration
benefit or CVPR-level fault localization. Preserve fixed EHRs, existing
generation outputs and old winners. No new inference is authorized by this
diagnostic or document. See the [frozen diagnostic protocol](radeval_error_type_diagnostic_protocol.md).

## Outputs and reproducibility

```text
artifacts/protected/tricompose_v1_2/
  radeval_error_type_diagnostics/error_types_12714150_001/
    frozen_plan.json
    diagnostic.json
    summary.json
    manifest.json
  radeval_error_type_diagnostic_audits/error_types_12714150_001/audit.json
```

The numerical replay completed in **0.693 seconds** in existing CPU allocation
12714150. It preserves all 624 rows, computes 28 metric/category diagnostics,
passes deterministic reverse-order replay, and independently checks that all
12 category-component sums reproduce original total-error choice results.

- Diagnostic manifest SHA256:
  `1a3f64824bc43185206dae55ec643cbf8cd2eea460c30c9c06015eba42b8c786`.
- Independent audit SHA256:
  `7fe9ddb9cf249174c839a695ff53c6b25d7f5379d3754ce29d9da9178a1c8b07`.
- Audit passed: 15 code/input/output hashes, 28 category denominator replays,
  28 independently recomputed correlations, 1,232 analytical choices,
  28 independent patient-cluster intervals and protected permissions.
- Fifteen new invented-fixture tests cover missingness, zero-information types,
  score ties, patient clustering, exact schemas, determinism and no promotion.
  Full V1.2 suite: **2,405 tests pass** in **15.165 seconds**. Git whitespace
  checks pass and both protected result tables are ignored by the project Git
  wrapper. These are engineering checks, not new clinical annotations.
