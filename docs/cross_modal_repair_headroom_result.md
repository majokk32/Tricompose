# Cross-modal repair feasibility / 换报告和换图的代理修复空间

Completed in existing CPU Slurm allocation **12654973**, following the
[separately frozen protocol](cross_modal_repair_headroom_protocol.md).
This is the already inspected DEVELOPMENT cache, not clinical truth,
independent validation or a new selection policy. No new model/GPU/API call,
download, Slurm submission, training, source EHR/report body, image or weight
read. All old EHRs, labels, scores, winners and protected outputs remain.

## Question / 要回答什么

Does an existing alternative improve explicit proxy facts without dropping
old comparable/supported facts or introducing a new contradiction? Compare
same-image report alternatives with changed-image/report alternatives while
keeping the EHR fixed. This complements the earlier score/call-count table;
it is not another scalar score or an oracle-selected output.

For each of **80 fixed EHRs**, retain the original Sana seed-0 / MAIRA-2
candidate and the immutable fixed-image report-only static winner. Enumerate
all three same-image reports and eight other-image/report alternatives:
**880 comparisons = 240 report alternatives + 640 cross-image alternatives**.
These are repeated observations of 80 EHRs, not 880 patients or GPU calls.

Same-image opportunities reuse the previously frozen report-headroom rule.
Cross-image candidates must strictly gain fixed-EHR image support or remove
an explicit fixed-EHR image opposition, while preserving comparable finding
IDs on every edge, existing direct-EHR support and positive image–report
support; introduce no new opposition and do not worsen available cached
report structure. Each must pass against **both** references. A conflict
disappearing into unknown/uncertain is not a repair. A higher structure score,
more image/report negative agreement or higher BioViL alone does not qualify.

This uses raw historical fourteen-head labels, not global No-Finding
expansion, the separate fresh eight-head profile, calibrated clinical states,
or independent evidence that a particular modality is wrong. Preserving a
possibly mistaken baseline image proxy is deliberately conservative and can
reject a genuinely better image; these gates do not establish clinical safety.

## Complete finite-cache results / 全部候选的检查结果

| Diagnostic | Result |
|---|---:|
| Fixed EHR denominator | 80 |
| EHRs with cached direct radiographic facts | 8 |
| EHRs without such direct facts, retained | 72 |
| EHRs with strict same-image report opportunities | 13/80 |
| Strict same-image report alternative pairs | 21/240 |
| EHRs with cross-image gain passing both references | 0/80 |
| Cross-image alternatives passing both references | 0/640 |
| Cross-image opportunities permitted by current scope basis | 0 |
| Finite-cache category: report-only opportunity | 13 |
| Finite-cache category: neither opportunity | 67 |

**中文结论：在当前有限候选池与严格保护条件下，可检验的增量主要来自换报告。
尚未找到符合条件的换图候选，不能把循环生成或相似度增加描述为定位并修复
错误图像。0/640 不是“图像模型不能用”或“任何新 seed 都无法修复”的证明。**

The initial branch basis is one direct EHR–XRV opposition, three missing
comparable image states, four no-direct-opposition cases and 72 no-direct-EHR
references. Several agreeing reports do not create an independent EHR/image
anchor. Unknown/uncertain and the 72 unavailable EHR edges stay unavailable;
no EHR enrichment, replacement or difficult-case exclusion occurred.

## Audit the actual guarded selections / 现有策略真的改好了什么

The prior cap-30 guarded result is not changed. This diagnostic checks its
selected candidate against the original fixed baseline and, for image changes,
also against the same-image static reference.

| Selected transition | EHRs | Passes proxy fact-preservation improvement |
|---|---:|---:|
| Unchanged | 63 | Not a repair attempt |
| Same-image report change | 16 | 13 |
| Image-and-report change | 1 | 0 |
| Total changed outputs | 17 | 13 pass / 4 do not pass |

Of the four non-passing changed outputs, three are report changes: one loses
old supported/comparable finding IDs, one lowers cached report structure, and
one has no strict finding-evidence gain. The one image change loses required
finding IDs against both references. These reasons are not independently
verified clinical errors; a non-passing alternative is not automatically worse
clinically. None of these four erased an old opposition solely into missing
evidence, although that failure mode is explicitly tested and forbidden.

Importantly, **three of the four non-passing changes have higher BioViL**.
Conversely, of 13 passing changes, only **five** have higher BioViL and **eight**
have lower BioViL. All comparisons are against their own unchanged fixed
baseline. Global similarity and per-finding preservation answer different
questions; neither becomes independent clinical truth or validates the other.
Do not retrofit the predicate to reward a preferred endpoint.

## Conditional endpoint readout / 不能冒充全队列收益的辅助读数

The complete label-only plan was written and hashed before endpoint attachment.
No endpoint chooses an alternative or changes its admissibility. For all 21
qualifying report alternatives, average gaps within EHR, then across the 13
opportunity-bearing EHRs: conditional BioViL gap **+0.0507731**. All required
endpoints are available for those 13 EHRs.

This is **not** a selected-output effect, a mean over all 80 EHRs or a
prospective regeneration benefit. It averages all qualifying alternatives,
not a best-cosine model. The cross-image conditional mean is **NA**, not zero,
because no cross-image candidate qualifies. Full per-pair endpoint values,
including failures, remain protected.

## Next boundary / 接下来应该接什么

The next reusable component is an explicit **output acceptance layer** that
records unchanged, proxy-fact-preserving change and unresolved/non-passing
change separately. It should preserve the original baseline/winner and not
call a score increase clinical repair. This result is descriptive: that new
layer is not installed or retrospectively applied to existing winners here.

Any new policy needs its own frozen protocol and comparison, retaining
fixed-image report-only static as a serious baseline. A prospective test must
charge actual generators, verifiers, failures, loading and runtime separately;
these cached opportunities do not establish GPU savings or that a fresh call
will succeed. Clinical localization still needs appropriately qualified
independent reference evidence. Do not invent human feedback, enrich EHRs or
weaken preservation after seeing this zero-cross-image result. Future GPU
execution needs the complete batch script/resources and explicit approval.

## Output and verification / 文件与核验

```text
artifacts/protected/tricompose_v1_2/repair_headroom/cross_modal_headroom_12654973_001/
  label_plan.json
  pair_opportunities.jsonl
  case_opportunities.jsonl
  case_table.csv
  guarded_selection_audit.jsonl
  conditional_readouts.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`9f179fa5597a88acd9aa66f5a2d77078ed68bce781468f9ef812cd98837382a3`.
Construction before final serialization took **2.280905 CPU seconds**, not
GPU/model runtime. Project-private directories/files are 2770/0660.

**27 new invented-fixture tests pass; full V1.2 suite passes 1,614 tests in
11.500 seconds.** Metadata audit verifies **38 consumed source hashes, eight
artifacts, all 880 pairs, every 80-case denominator and selected transition**.
Old 3,200-trial and 400-guarded-trial histories replay exactly; label-plan hash,
case CSV, endpoints, conditional aggregation and bilingual report reconstruct
exactly. A separate raw-state set implementation reproduces **1,520 individual
gate comparisons** (240 report comparisons plus 640 × two cross-image
references), including finding-ID preservation rather than count equality.
All source hashes remain unchanged. Sealed worker/tests/protocol/output are
not edited after execution. These engineering checks are not clinical review.
