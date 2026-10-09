# Scope-guarded stopping / 按证据范围停止：缓存对照结果

Completed under existing CPU Slurm allocation **12654973**, following the
[separately frozen development protocol](scope_guarded_stopping_protocol.md).
This is an already inspected DEVELOPMENT bank, not an independent test.
No new model/GPU/API call, download, Slurm submission, training, source
EHR/report body or image/weight read. All old data, scores and winners remain.

## What changed / 本轮只改了什么

Keep the original initial Sana/MAIRA-2 path, model ordering, proxy-stop test,
candidate ranking and invocation accounting. Veto an image branch unless
cached metadata shows a basic-invalid image or an explicit directly evidenced
EHR finding opposed by the current XRV proxy state. Missing/unknown/uncertain
evidence and correlated report votes do not authorize an image branch.

Without that basis, try remaining report experts on the current image, then
stop **unresolved** and return the best eligible observed candidate. An
allowed branch means cached proxy exploration, not a confirmed image fault,
clinical repair, or permission to execute new inference. No numeric threshold
or new score weight was fitted; BioViL was attached only after selection.

All **80 fixed synthetic EHRs / 240 image slots / 960 cached triples** remain.
Five fixed caps, 4/8/12/20/30, produce **400 guarded cache trials**, not new
patients or model calls. References include all 3,200 original trials, 10,000
score-free-final seed trials and 400 fixed-image `report_only_static` trials.
Random seed settings are averaged within EHR before cohort comparisons.

## Full-cap comparison / 上限 30 的完整比较

Equal caps do not imply equal simulated expenditure. BioViL-T is a secondary
global image–text retrieval readout, not clinical correctness. Support,
opposition and coverage use raw fourteen-head XRV/report-label proxies and
known-reference denominators; they are not qualified clinical finding truth.

| Method | Mean simulated calls | BioViL mean, 80/80 EHRs | CXR–Report support / known | Opposition / known | Comparable coverage / known |
|---|---:|---:|---:|---:|---:|
| Fixed Sana + MAIRA-2 | 4.000 | 0.6186 | 0.0865 | 0.0042 | 0.0906 |
| Random acquisition + score-free final | 30.000 | 0.5082 | 0.1551 | 0.1035 | 0.2586 |
| Old random acquisition + scored final | 30.000 | 0.6122 | 0.2365 | 0.0000 | 0.2365 |
| Static all-image reranking | 30.000 | 0.6122 | 0.2365 | 0.0000 | 0.2365 |
| Original targeted heuristic | 28.825 | 0.6075 | 0.2271 | 0.0000 | 0.2271 |
| Fixed-image report-only static | 10.000 | 0.6388 | 0.1375 | 0.0042 | 0.1417 |
| Scope-guarded targeted | 10.050 | 0.6397 | 0.1375 | 0.0042 | 0.1417 |

**中文结论：缺乏直接图像冲突证据时停止，避免了旧规则的大量换图探索；但
新规则与固定图、只换报告的基线在 79/80 个 EHR 上选择完全相同的 triple。
尚未证明明显的自适应优势，更没有证明临床错误定位或修复成功。**

Versus full-cap all-image static, the guarded mean BioViL delta is **+0.0275**
(23 EHRs higher / 44 tied / 13 lower), with **19.95 fewer simulated calls**.
But proxy support/coverage are lower, 0.1375/0.1417 versus 0.2365/0.2365,
and proxy opposition is nonzero. This is a trade-off, not across-metric
dominance or measured GPU savings.

The essential fixed-image control nearly matches the new policy: BioViL
delta **+0.0009179**, 1 higher / 79 tied / 0 lower, with **0.05 more simulated
calls**. Its full-cohort CXR–Report support/opposition/coverage are identical.
Do not attribute the entire full-static contrast to an innovative agent.

## Every cap / 不隐藏其他预算

| Cap | Guarded mean simulated calls | Guarded BioViL | Fixed-image report-only BioViL | All-image static BioViL | Original targeted BioViL |
|---|---:|---:|---:|---:|---:|
| 4 | 4.000 | 0.6186 | 0.6186 | 0.6186 | 0.6186 |
| 8 | 8.000 | 0.6313 | 0.6304 | 0.6304 | 0.6313 |
| 12 | 10.025 | 0.6397 | 0.6388 | 0.6388 | 0.6397 |
| 20 | 10.050 | 0.6397 | 0.6388 | 0.6702 | 0.6655 |
| 30 | 10.050 | 0.6397 | 0.6388 | 0.6122 | 0.6075 |

At cap 20, both original all-image methods have a higher alternate BioViL
mean. More candidate search is not always harmful or always helpful; no best
cap was chosen or operating point retuned after inspecting these results.
The complete CSV retains all seven methods, five caps and three EHR scopes.

## Stop outcomes and EHR missingness / 停止不等于成功

At cap 30:

- **72 EHRs:** stop unresolved, no direct cached EHR–image reference.
- **3 EHRs:** stop unresolved, direct EHR evidence but no comparable image
  proxy state on the final observed image.
- **5 EHRs:** stop unresolved, no direct EHR–image opposition on the final
  observed image; this does not establish complete agreement.
- **79 EHRs** explore one image; **1 EHR** explores two images. All 80 final
  outcomes remain unresolved; zero proxy-satisfied stops/clinical acceptances.
- There are **80 veto decisions** at full cap. Decisions are not patients,
  confirmed errors, rejected triples or clinical repair events.

The direct-EHR scope remains **8/80**, with **72/80 unavailable** for the
two direct EHR edges. Those 72 EHRs are retained, not relabeled normal,
discarded, enriched or replaced. Their absent edge scores remain NA.
This direct-fact split is distinct from the historical 15/65 prompt tiers.

Among the eight available EHRs, EHR–CXR support/known is 0.625 for guarded
versus 0.500 for fixed-image report-only; EHR–Report support/known stays 0.500.
Guarded's BioViL subgroup mean is 0.6141, versus all-image static's 0.6723.
Among the 72 unavailable EHRs, guarded and fixed-image report-only have
identical selected triples/readouts. The full-cohort mean does not establish
that clinically important or directly conditioned cases improve.

## Interpretation and next boundary / 对项目的意义

This establishes an implemented **evidence-scope action veto and explicit
abstention**, with reproducible cost/lineage accounting. It does not establish
the professor's natural-error localization and targeted regeneration claim.
Repeated reports share image evidence; XRV/CheXbert labels and global BioViL
do not independently adjudicate which modality is clinically erroneous.

Do not add another threshold or large model solely to raise this inspected
bank's number. Before further GPU execution, freeze a separate prospective
policy/data protocol and test incremental value over fixed-image report-only
static, not only exhaustive all-image static. Keep EHR fixed, retain unknowns,
report partial coverage and unresolved outcomes, and measure actual calls,
failures and runtime. Any clinical fault/repair claim requires appropriately
qualified reference evidence; no human labels or physician feedback are
invented. Any new Slurm job still needs a complete script/resource request
and explicit approval.

## Output and verification / 输出与核验

```text
artifacts/protected/tricompose_v1_2/automatic_replays/scope_guarded_stop_12654973_001/
  guarded_outcomes.jsonl
  frozen_policy.json
  case_means.csv
  method_comparison.csv
  paired_case_comparison.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`3d4553641af5408d639e857b3f20d830761458c1c08ceff9064dc6cb3d549360`.
Construction before serialization took **2.194871 CPU seconds**, not model
runtime. Private directories/files use 2770/0660 inside the CARC project.

**24 new invented-fixture tests pass; the full V1.2 suite passes 1,587 tests
in 9.252 seconds.** Post-run checks verify all 25 bounded consumed source
hashes, seven artifacts, exact old 3,200-trial replay, all 400 guarded traces,
fixed EHR hashes/states/provenance, every actual cached image-branch basis,
observed-only selection and original invocation receipts. All 2,800 case
means, 105 method/subgroup rows, 90 paired contrasts and the bilingual output
replay exactly. An independent integer/count implementation reproduces seed
averaging, all three edges' support/opposition/coverage and paired BioViL
arithmetic. Source hashes and old full-cap random/static choices remain
unchanged. Sealed worker/tests/protocol/output are not edited after execution.
These engineering checks are not clinical validation.
