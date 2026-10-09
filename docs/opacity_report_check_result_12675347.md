# Opacity report-assertion check: completed, not qualified for repair

## 完成情况 / Completion

The approved frozen text-only Qwen job **12675347** completed on V100
`d14-10`, exit `0:0`, Slurm elapsed **3 minutes 51 seconds**. Worker elapsed
before serialization was **76.513519 seconds**; peak allocated CUDA tensor
memory including model load **15.624 GiB**, not total device memory. All **24
planned invocations** were attempted: 16 primary texts, two fixed replays and
six wholly authored controls. No retries, replacement, new image/report/EHR
generation, training, selection or change to old scores/winners.

New protected output:

```text
artifacts/protected/tricompose_v1_2/report_opacity_check_runs/opacity_text16_12675347/
  report_evidence.json
  raw_responses.json
  replay_checks.json
  authored_control_readout.json
  candidate_text_readout.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Result manifest SHA256:
`608437033da02dcf7fcf1922644e1c67f79a7e15a06defd1fc37cc0b11392a29`.
Raw source/response/quote text remains protected and is not reproduced here.

## Independent audit

Post-run audit ran inside existing CPU Slurm **12666569**, with no model or
image inference. Independently implemented JSON/quote/offset/state decoding
reparsed **all 24** raw responses against the approved synthetic report texts
or authored controls, without displaying their bodies. Verified:

- **667** immutable source pins and seven result artifacts.
- All **16** distinct report hashes and **36** candidate contexts; deterministic
  representatives and every alias retain the original fixed EHR/image lineage.
- All **972** original named context cells unchanged; seven diagnostic fields
  appended. No new total score or selected triple.
- **18** accepted primary evidence quotes, verbatim unique locations, Unicode
  offsets and hashes; this verifies traceability, not semantic correctness.
- Complete/failed denominators, actual call counters, six control outcomes,
  two replay outputs, permissions **2770/0660** and Git exclusion.
- Both fixed replays match their primary state and exact response SHA256.
- Full V1.2 regression: **1,958 tests passed in 13.270 seconds**.
  `git diff --check` passed.

Independent audit script: `.tmp/audit_opacity_report_text16_12675347.py`.
This script is Git-ignored and does not emit report/response bodies.

## Primary-text contract outcomes

| Outcome | Distinct texts / 16 |
| --- | ---: |
| Complete positive proposal | 3 |
| Complete negative proposal | 7 |
| Complete uncertain proposal | 3 |
| Complete unknown proposal | 0 |
| Failed-unavailable | 3 |

Three failures remain unavailable, never counted as completed unknown:
**two quotes not present in their source**, **one ambiguously repeated quote**.
No primary response reached the 512-token generation cap. These are contract
failures, not confirmed clinically incorrect source reports. No substring
rescue, correction or section preference was applied.

## Authored language controls: important failure signal

All six controls completed the quote contract; only **3/6** proposed states
matched their authored expectations. These tiny investigator-written controls
are not independent clinical gold or an estimate of report-population accuracy.

| Control type | Authored expectation | Proposed state | Match |
| --- | --- | --- | --- |
| Explicit presence | positive | positive | yes |
| Explicit absence | negative | negative | yes |
| Possible finding | uncertain | positive | no |
| Finding not mentioned | unknown | negative | no |
| Opposing assertions | uncertain | uncertain | yes |
| Qualified absence | uncertain | negative | no |

**结论：这版核验器还不能用于自动临床判错或重生成触发。** 它在 uncertainty、
unknown 和 qualified absence 上暴露了问题；引用真实存在并不等于引用的含义
被正确分类。不能用代码测试通过或两次输出一致代替评分器的语义有效性。

## Reattached context readout, not a repair result

The subset was selected because cached CheXbert opacity proposals opposed two
agreeing image proxies. It is a conflict-enriched post-hoc development subset,
not a representative or untouched evaluation cohort.

| New text proposal vs cached CheXbert proposal | Contexts / 36 |
| --- | ---: |
| Same explicit state | 11 |
| Opposite explicit state | 13 |
| Uncertain / not comparable | 8 |
| Failed response | 4 |

Because the two image proxies opposed the original cached report proposal,
the same new readout has **13** contexts supporting their image-proxy state,
**11** opposing it, **8** not comparable and **4** unavailable. These are **36
correlated candidate slots**, not 36 independent reports or patients. All
underlying reports and images are unchanged. A numerical reduction from the
original 36 apparent oppositions does **not** establish quality improvement:
the extractor changed, 12 contexts lack comparable new evidence, and the
authored-control check failed. Neither extractor nor agreeing image proxies
are established truth. EHR opacity evidence remains unknown in this scope.

## Runtime caveat and next step

Actual recorded dtype is **`torch.bfloat16`**, not the earlier anticipated
V100 fp16 fallback. The unchanged loader calls `torch.cuda.is_bf16_supported()`
without an argument; the installed Torch 2.9.1 implementation defaults to
`including_emulation=True` and can accept BF16 tensor creation below its
native-support branch. This explains why an fp16 fallback was not guaranteed.
It does **not** establish that dtype caused the observed semantic failures;
there is no controlled dtype comparison. Do not patch the consumed loader,
worker, prompt, tests or plan, nor silently rerun this result.

The defensible next step is a **new versioned assertion-extraction diagnostic**:
separate locating evidence from determining its polarity, preserve negation,
uncertainty and omitted findings, and qualify it on a broader predeclared
authored language suite. Keep this failed diagnostic and all denominators.
Only after the assertion evidence is qualified should a new frozen conflict
trigger and independent repair-evaluation endpoint be proposed. New GPU work
requires another complete script/resource presentation and explicit approval.

下一步优先修核验器，不是修改 EHR、重生成胸片或把报告直接判错；当前仍没有
经过验证的错误模态定位、临床修复效果或新最佳 triple。
