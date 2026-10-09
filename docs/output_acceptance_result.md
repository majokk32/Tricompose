# Observed-only output acceptance / 输出验收层结果

Completed under existing CPU Slurm allocation **12654973**, following the
[separately frozen protocol](output_acceptance_protocol.md). This is an
already inspected DEVELOPMENT cache, not independent clinical validation.
No new model/GPU/API call, download, Slurm submission, training, EHR/report
body, image or weight read. All old EHRs, scores, winners and outputs remain.

## What is implemented / 接好了什么

`TriCompose-v1.2/tools/apply_output_acceptance.py` provides the reusable
`assess_output(baseline, proposed, observed_candidates)` interface and a
CPU-Slurm guarded replay CLI. The gate sees only hash-bound **observed**
candidate metadata, with alternate scores and old rank/winner flags stripped.
It is connected to cached final selection, **not installed into live GPU
generation**. Fresh eight-head worker receipts need a separately validated
adapter; they must not be silently cast into this historical fourteen-head
profile.

The layer distinguishes:

- unchanged output, still unverified;
- strict proxy-fact-preserving report/image change, still unverified;
- vetoed proposal with the original fixed candidate retained, unresolved;
- missing proposal or no metadata-eligible output, separately unresolved.

It reuses immutable per-finding preservation predicates: no lost supported
or comparable fact IDs, no new explicit opposition, no lower available cached
structure, and strict finding-evidence gain. Unknown/uncertain cannot conceal
a contradiction. Changed images must improve the fixed-EHR image edge against
both the original reference and the best eligible **already observed** report
on the initial image, with a valid cached image-branch basis. An unobserved
full-bank winner is never used as a free reference.

If the parent's proposal fails, the layer does not rerank other alternatives:
retain the metadata-eligible initial fixed result, otherwise return null.
Fallback is not certified correctness. Tampered/changed EHR or inconsistent
shared image/report evidence is a contract error, not a clinical failure.
Baseline proxy labels may themselves be wrong; preservation is deliberately
conservative, not a guarantee of medical safety.

## Complete experiment / 完整对照范围

All **80 fixed EHRs/960 cached triples**, caps 4/8/12/20/30 and the **8 direct-
EHR / 72 unavailable** scopes remain. Apply the same gate to two unchanged
parent policies, `scope_guarded_targeted` and `report_only_static`: **800 cached
decisions**, not 800 patients or model calls. Acquisition traces, stop reasons
and already incurred model charges are unchanged, including vetoed proposals.
The wrapper cannot refund old generation or claim prospective GPU savings.

All seven old controls are retained alongside the two gated versions, with
random replicates averaged within EHR. Selection decisions were written and
hashed before full-bank BioViL/raw-edge attachment. No endpoint affects a
gate, fallback, reference or final candidate.

## Full-cap result / 上限 30

| Route | Mean simulated calls | BioViL | CXR–Report support / known | Opposition / known | Comparable coverage / known |
|---|---:|---:|---:|---:|---:|
| Fixed Sana + MAIRA-2 | 4.000 | 0.6186 | 0.0865 | 0.0042 | 0.0906 |
| Original scope-guarded | 10.050 | 0.6397 | 0.1375 | 0.0042 | 0.1417 |
| Original report-only static | 10.000 | 0.6388 | 0.1375 | 0.0042 | 0.1417 |
| Gated scope-guarded | 10.050 | 0.6280 | 0.1302 | 0.0042 | 0.1344 |
| Gated report-only static | 10.000 | 0.6280 | 0.1302 | 0.0042 | 0.1344 |
| Full-bank static | 30.000 | 0.6122 | 0.2365 | 0.0000 | 0.2365 |

Both gated routes choose **identical outputs for 80/80 EHRs at EVERY cap**, not
only cap 30. The scoped route retains slightly more simulated expenditure at
larger caps. This does not demonstrate an adaptive/agent advantage over the
simple gated report-only control.

**中文结论：验收层成功接入缓存选择，保留了 13 个符合代理事实保护条件的报告
改动，并单独记录了 4 个否决后的参考保留结果。它没有发现新的有效换图收益。
这版最清楚的基线是有事实保护条件的报告择优，而不是已验证的三模态修复 Agent。**

## Status, not clinical success / 状态必须分开

| Full-cap status | Gated scope-guarded | Gated report-only static |
|---|---:|---:|
| Unchanged, unverified | 63 | 64 |
| Proxy-preserving report change, unverified | 13 | 13 |
| Vetoed proposal, fixed result retained, unresolved | 4 | 3 |
| Proxy-preserving image change | 0 | 0 |
| Null output | 0 | 0 |
| Clinical acceptance/confirmed repair | 0 | 0 |

The policies have different parent histories: their one previous image-change
case now resolves to the same fixed candidate. Thus identical final triples
do not imply identical stop histories/status counts or compute spent.
All 80 outputs remain clinically unqualified. The two EHR edges have only
eight available EHR denominators; support/known is 0.500 on each at full cap.
The other 72 EHR edges remain NA, not normal or contradicted.

## Retain the trade-off / 不隐藏指标下降

The gated scope mean BioViL **0.6280** is below its parent's **0.6397** at
exactly the same charged acquisition. Support/coverage also fall slightly
because some parent changes are vetoed. This is a predeclared constraint
trade-off, not an across-metric improvement or a new clinical efficacy result.
Against fixed, the mean BioViL is +0.00936 and raw support/coverage increase,
but the result uses more simulated calls and remains proxy-only.

| Cap | Gated BioViL, both routes | Gated scope mean calls | Gated report-only mean calls | Identical selected outputs |
|---|---:|---:|---:|---:|
| 4 | 0.6186 | 4.000 | 4.000 | 80/80 |
| 8 | 0.6302 | 8.000 | 8.000 | 80/80 |
| 12 | 0.6280 | 10.025 | 10.000 | 80/80 |
| 20 | 0.6280 | 10.050 | 10.000 | 80/80 |
| 30 | 0.6280 | 10.050 | 10.000 | 80/80 |

Do not tune a cap, threshold, preservation condition or secondary endpoint
on these inspected outcomes and call it independent confirmation. BioViL
global retrieval, fourteen-head label support and structure are distinct,
uncalibrated readouts; none is clinical truth. Full-bank static's higher
support/lower opposition is not full clinical consistency at partial coverage.

## Next boundary / 下一步的接口边界

The interface is ready for observed-only metadata integration and tested
fallback/cost handling. The next live test must freeze its fresh-receipt
adapter, policy and independent evaluation boundaries before generation;
compare with the same gated report-only baseline and record real calls,
failures and time. Cached equivalence and a passing proxy gate do not justify
clinical error-localization or repair claims. Do not replace EHRs, manufacture
clinical labels/human feedback or relax gates to make a result look better.
Any new GPU/Slurm submission requires its complete script/resources and
subsequent explicit user approval. No such job is submitted by this result.

## Output and verification / 输出和核验

```text
artifacts/protected/tricompose_v1_2/output_acceptance/output_gate_12654973_001/
  frozen_policy.json
  selection_plan.json
  gated_outcomes.jsonl
  selected_references.jsonl
  case_means.csv
  method_comparison.csv
  paired_case_comparison.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

`selected_references.jsonl` points to already generated candidates by opaque
IDs/hashes; it does not copy or newly generate EHRs, images or report text.
`case_means.csv` contains all **3,600** case/method/cap rows; the comparison
has **135** method/cap/scope rows and **75** predeclared paired contrasts.
Full decision comparisons and per-finding veto reasons remain protected.

Manifest SHA256:
`211912fa0cf4e38f5eaf146f57050985b7546497639933be7e6d66ec2ae9d545`.
Construction before final serialization took **3.864852 CPU seconds**, not
GPU/model runtime. Directories/files use project-private 2770/0660 modes.

**32 new invented-fixture tests pass; full V1.2 suite passes 1,646 tests in
11.216 seconds.** The metadata audit checks **42 source hashes, nine artifacts,
800 exact gated replays, 800 unchanged parent cost receipts and 800 selected
reference bindings**. A separate raw-state set implementation reproduces
all gate decisions and the four observed-only cross-image reference checks.
All original 3,200 and guarded 400 trial histories, fixed-EHR/image/report
invariants, pre-endpoint decision hash, tables and bilingual report replay.
Independent integer/within-EHR arithmetic reproduces all edge means,
coverage/missingness and paired endpoint contrasts; all-cap 80/80 equality
is checked. Original sources/results remain unchanged. Consumed code,
tests/protocols and protected output are immutable after execution. These
engineering checks are not medical review or paper-grade clinical validation.
