# Frozen BioViL-T expert benchmark result — 2026-10-06

## Conclusion / 主要结论

The approved image-grounded benchmark completed successfully, but **BioViL-T
global cosine is not qualified as the primary clinical selection or repair
judge**. Its association with fewer expert significant errors is weak and
negative on this subset; the interval includes zero. Its within-anchor choice
benefit is small and uncertain. Preserve both results without changing score
direction, fitting weights, or selecting favorable cases.

已实际完成真实图像与候选报告的评分。工程运行成功不等于评分器可靠：
BioViL-T 的全局图文相似度不能据此作为自动查错、定位错误模态或重新生成
的主判据。它暂时保留为辅助匹配信号，历史评分和择优结果均未改变。

## Fixed scope and execution

This executes the previously sealed
[protocol](radeval_image_benchmark_protocol.md) and
[preparation](radeval_biovil_preparation_12714150_001.md). Their preparation-only
status text is retained as historical, hashed evidence, not edited after the run.

- Existing approved CPU Slurm allocation **12714150**, two worker threads,
  zero GPUs, zero new sbatch submissions. Frozen/offline BioViL-T, no training,
  package/model download, threshold fitting or external clinical API.
- **43/43** locally linked source images encoded successfully;
  **130/130** distinct candidate texts encoded successfully;
  **132/132** available image-report pairs scored successfully.
- Those pairs form **44** source/section/reference comparison anchors, with
  three candidates each, and **34** recognizable MIMIC patient groups.
  One image can contribute more than one section anchor; 44 is not 44 patients.
- All **624** original pair rows remain. **492** pairs have unavailable local
  source images and retain null scores, not zeros. Available coverage is
  **21.15%**. The subset was fixed by exact local image availability before
  scoring, not by diagnoses, expert errors or visual quality.
- BioViL-T sees only the image and candidate report. The reference report and
  released expert counts are not fed into either encoder. RadGraph comparison
  scores are reused from the earlier reference-based benchmark.
- Official global embeddings, resize 512 / center crop 448, L2 normalization,
  128 dimensions; no silent text truncation or preprocessing optimization.
- Total worker runtime **42.579 seconds**, peak process RSS **1.516 GiB**.
  No real image rendering/copying, source EHR reads, generation, or winner edits.

## Primary: agreement with expert error burden

Spearman rho compares a higher-is-better score against **negative** significant
error count. Positive rho would mean higher scores accompany fewer errors;
negative rho means the opposite trend. All four metrics below use the **same
132 pairs**, not BioViL-T's subset versus RadGraph's full 624-pair benchmark.
Intervals use 1,000 whole-patient-cluster bootstrap resamples, seed 0.

| Score | Significant-error rho | 95% cluster interval | Kendall tau-b | All-error rho |
| --- | ---: | --- | ---: | ---: |
| BioViL-T raw cosine | -0.1510 | [-0.3198, 0.1098] | -0.0945 | -0.1724 |
| RadGraph entity F1 | 0.1372 | [-0.0885, 0.2958] | 0.0943 | 0.1166 |
| RadGraph relation-presence F1 | 0.1555 | [-0.0687, 0.3224] | 0.1059 | 0.1217 |
| RadGraph full-relation F1 | 0.2400 | [0.0509, 0.3992] | 0.1676 | 0.2201 |

BioViL-T is the predeclared primary verifier; its interval crosses zero.
Full-relation RadGraph has a positive association on this availability subset,
but it requires a reference report unavailable during fully synthetic inference.
The four-score comparison is not a multiplicity-adjusted discovery or proof
that this component can select the best report for a fixed image. Its earlier
full-pool result remains unchanged; different cohort estimates must not be
substituted for each other.

All-error rho intervals respectively are [-0.3581, 0.1190],
[-0.1073, 0.2832], [-0.1108, 0.2998], and [0.0276, 0.3888].
Raw cosine is not a calibrated probability or clinical accuracy percentage.

## Secondary: choose among three reports for the same anchor

This endpoint was declared before the image run, following the earlier
report-only post-hoc diagnostic. It is an offline choice diagnostic, not newly
deployed selection. All three candidate scores and error counts must be
available. Maximum-score ties use uniform expected choice, not a favorable
candidate slot. All 44 available anchors are complete for both error targets.

Uniform random choice averages **4.8182 significant errors** per anchor;
the minimum-expert-error oracle averages **3.2045**. Lower is better.

| Score used to choose | Mean significant errors | Difference vs random | 95% cluster interval of difference | Strict pairwise accuracy |
| --- | ---: | ---: | --- | ---: |
| BioViL-T raw cosine | 4.5682 | -0.2500 | [-0.7647, 0.1062] | 53.30% |
| RadGraph entity F1 | 4.6250 | -0.1932 | [-0.5742, 0.3971] | 50.47% |
| RadGraph relation-presence F1 | 4.7727 | -0.0455 | [-0.4697, 0.6031] | 46.23% |
| RadGraph full-relation F1 | 4.8182 | 0.0000 | [-0.4137, 0.5982] | 45.28% |

Pairwise accuracy uses 106 comparisons with unequal significant-error counts;
score ties contribute expected 0.5 correctness. It is not 132-pair clinical
accuracy or fault-localization accuracy. All four selection-difference
intervals include zero. A small favorable mean is not a demonstrated gain.

The secondary all-error target has random mean **5.3258** and oracle **3.7273**.
Score-chosen means in the same order are **5.0909 / 5.1932 / 5.2500 / 5.3182**;
all four difference intervals also include zero. Full exact statistics and
tie counts remain in the protected evaluation.

整体跨病例相关性与同病例择优是两个问题。这里 BioViL-T 的整体相关性
偏负，但固定病例内平均错误数略降；后者的区间跨零，不能据此声称有效。
RadGraph full-relation 的整体相关性较好，也没有转化成固定病例择优收益。

## Evidence limits and development decision

The released experts counted report errors against a reference report. This
is **not new radiologist adjudication of the image**, independent image truth,
or an EHR-consistency benchmark. Some reference claims may not be supported
by a single image. Training-set overlap is unverified; this is a small,
availability-selected development subset, not an untouched final test.
Synthetic-domain transfer, error localization and targeted repair are unproven.

Keep BioViL-T secondary; do not invert its sign after this result, fit a
combined reward, tune thresholds on these outcomes, or turn generated-report
consensus into truth. Keep reference-based RadGraph useful for structural
comparison without pretending a generated reference is independent evidence.
The next evidence-producing step should test **finding-level contradiction
and abstention** under a separately frozen protocol and an independently
supported reference. Do not launch another generator optimization merely
to improve these verifier scores. No new computation is authorized by this
result document; any new sbatch still requires complete-script approval.

## Artifacts and independent replay

All data-bearing paths below are workspace-relative and protected:

```text
artifacts/protected/tricompose_v1_2/
  radeval_image_benchmark_runs/biovil_cpu_12714150_001/
    paired_score_table.json
    evaluation.json
    summary.json
    manifest.json
  radeval_image_benchmark_audits/biovil_cpu_12714150_001/audit.json
```

`paired_score_table.json` retains all 624 opaque rows, four score cells,
released error counts and explicit missing statuses. Native embeddings, source
text, image resolver and worker logs remain protected internal artifacts;
do not publish/print them or real source inputs.

- Run manifest SHA256:
  `8e55b223ef22c5da61c20aeb0c53f818ab6c1736eeb2c95ee85b39803ea0c18a`.
- Independent audit SHA256:
  `f9af2ec5a0a065e4fc92e174defc8fba474898ee26552a5ca053d98a01e1d1b1`.
- Audit passed: 24 code/source/output hash checks, 43 approved image byte
  hashes, 173 finite normalized vectors, all 624 pair rows, 352 analytical
  choice records, 16 independent correlation/selection/bootstrap replays,
  and nine protected permission checks. Zero new model calls in the audit;
  no real image rendering, reference-report model input or changed winners.
- Full V1.2 suite: **2,390 tests passed** in **15.097 seconds**. These are
  engineering tests, not additional clinical evidence.
