# Online report escalation: completed two-case engineering smoke

## 执行结果 / Execution

Approved Slurm job **12657042** completed on V100 node d14-15 with exit **0:0**.
Allocation: one V100, four CPUs, 48G host RAM, thirty-minute upper cap.
Measured Slurm elapsed: **00:09:16**. Controller runtime from after initial
asset checks through generation, controls and endpoint: **417.261 seconds**.
Neither number is isolated GPU kernel time; both include startup/I/O, and
the job also performs frozen-asset checks and a metadata audit.

Kept the same two already-inspected DEVELOPMENT EHRs and all original prompts.
New generation: **2 RoentGen-v2 images, 2 CXRMate-single reports and 2
CheXagent-2 reports**. The original EHR/source inputs, historical output
selections, thresholds and sealed code/protocols were not modified. No training,
finetuning, external API, download, EHR replacement or original target read.

All **12 primary attempts completed, zero failures**:
2 CXR generation + 2 XRV + 4 report generation + 4 CheXbert.
Secondary BioViL-T measured all four completed pairs, with **2 image and
4 text encoder calls**, zero unavailable reports. It followed sealed choices
and was not used to route or select. All models remain frozen.

## 方法比较 / Method comparison

| Method | Selected EHR triples | Charged primary prefix attempts | Mean BioViL raw cosine |
| --- | ---: | ---: | ---: |
| Fixed initial CXRMate path | 2 | 8 | 0.4671735764 |
| Online report escalation | 2 | 8 | 0.4671735764 |
| Always-second static, same fresh-output veto | 2 | 12 | 0.4671735764 |

**Both online cases stopped unresolved for direct EHR/image proxy opposition**.
No second report was requested by the online policy. Online selections were
sealed first; both additional CheXagent reports were subsequently acquired
as paid, static-only shadow controls. They did not enter online observations.
Both static proposals failed the frozen output-change predicate and fell
back to the initial reference. **All three methods select the same output
for both EHRs; zero accepted report changes.**

Thus the policy's actually observed prefix count is eight versus twelve for
the static control, but eight is also the fixed baseline count. The collection
still paid all twelve plus secondary encodings. **No measured actual GPU saving,
generation-quality improvement or advantage over the fixed baseline is
demonstrated.** Output retention is not clinical acceptance; the online status
is explicitly unresolved, not successful repair.

## Three-edge selected-output diagnostics / 三条边的诊断读数

All methods share these same selected-output aggregates. These are counts from
the fixed eight-enabled-head XRV/CheXbert proxy profile, not clinical truth or
calibrated probability scores. Six disabled heads remain unknown; unavailable
comparisons do not become negatives.

| Edge | Known reference facts | Comparable | Supported | Proxy oppositions |
| --- | ---: | ---: | ---: | ---: |
| EHR–CXR | 2 | 2 | 0 | 2 |
| EHR–Report | 2 | 1 | 0 | 1 |
| CXR–Report | 16 | 8 | 8 | 0 |

这些读数说明评分器看到了 EHR–图像冲突，换同一张图的报告没有消除它。
但不能据此断言图像在临床上一定错了：还需要区分固定 EHR/facts/prompt
桥接、图像条件保真度和冻结评分器自身误判。BioViL 是未校准图文余弦，
0.4672 不是准确率或临床质量评分。这两条已看过的发展集病例不能支撑
泛化、显著性或论文级错误定位/修复结论。

## Audit and artifacts / 审计与文件

The in-job metadata audit passed, reconstructing **four completed receipts**,
frozen-scorer label/artifact/parent dependencies, online/static decisions,
prefix/continuation charge accounting and sealing chronology. No report bodies
or image pixels were reopened by the audit. Metadata validation is not clinical
acceptance. A subsequent CPU metadata check verified root artifact hashes,
audit binding and group/private modes for **312 run entries, three audit entries
and ten runtime entries**; zero new model calls were incurred by that check.

Protected output:

```text
artifacts/protected/tricompose_v1_2/online_report_runs/online2_12657042/
  cases/<opaque_case>/inputs/             # unchanged canonical EHR/facts/prompt
  cases/<opaque_case>/operations/         # generated images, reports and labels
  cases/<opaque_case>/online_selection.json
  selection.json
  score_rows.json
  completed_triplets.json                 # generated artifact paths and hashes
  execution_summary.json
  endpoint_sidecar.json
  method_comparison.json
  method_comparison.csv                   # three-method raw score/cost table
  summary.json
  controller.journal.jsonl
  manifest.json
artifacts/protected/tricompose_v1_2/online_report_audits/online2_12657042/
  audit.json
  manifest.json
```

Run manifest SHA256:
`aaf6da59ce052de0722e2e1c47e4813f8ad773a95153442803816bff79f8c2fb`.
Audit manifest SHA256:
`0fce53d6d3faff1d74f8b7f14d2c2d2d93ef0c93e5c06b586a4b08b0387270af`.

Generated clinical content remains protected, directories 2770/files 0660;
none is copied into this note, ordinary README, chat or Git. Completed audited
runs are not amended or overwritten. Preparation/source-policy/software tests
are documented separately; this execution does not alter the frozen protocol.

## What this supports / 下一步边界

Supported: end-to-end fresh generation, rule-based stopping before unseen
controls, honest charged-call accounting and provenance-correct scoring/audit.
Not supported: clinically localized image fault, successful regeneration,
method superiority, actual GPU saving or a completed scientific benchmark.

Before expanding or adding another seed, investigate the two EHR/image proxy
conflicts using existing frozen provenance/metadata and the scorer's prior
validation evidence. Do not resolve them by replacing/enriching EHRs, relabeling
unknowns, tuning thresholds on these cases, selecting by secondary cosine or
claiming report-only regeneration repairs image fidelity. Any additional GPU
execution requires another complete script/resource display and approval.
