# Fresh action effects / 最近几类动作到底改变了什么

2026-10-09. Completed CPU-only join of two already audited synthetic execution
jobs, in existing CPU allocation12851223. No new model/scorer/API call, GPU
submission, prompt/EHR change, threshold fitting or winner replacement.
Protocol: `docs/fresh_action_effects_protocol.md`.

## Joined before/after table / 放到一起看，而不是另造一个分数

Each support entry is supported / known reference facts under the unchanged
diagnostic XRV/CheXbert profile. It is NOT clinical accuracy. Positive and
negative support, opposition and missingness remain separate in the CSV.

| Synthetic case | Observed action | EHR–CXR support | EHR–Report support | CXR–Report support | Original proxy gate |
|---|---|---|---|---|---|
| case_009 | Same image, CXRMate-single→LLaVA-Rad | 0/1→0/1 | 0/1→0/1 | 4/8→3/8 | Not passed |
| case_018 | Same image, MAIRA-2→LLaVA-Rad | 0/1→0/1 | 1/1→0/1 | 2/8→2/8 | Not passed |
| case_009 | RoentGen-v2 seed0→seed2, CXRMate-single fixed | 0/1→0/1 | 0/1→0/1 | 0/8→3/8 | Not passed |
| case_018 | RoentGen-v2 seed0→seed2, CXRMate-single fixed | 0/1→0/1 | 0/1→0/1 | 0/8→5/8 | Not passed |
| case_009 | RoentGen-v2 seed0→Sana seed1, CXRMate-single fixed | 0/1→0/1 | 0/1→0/1 | 0/8→3/8 | Not passed |
| case_018 | RoentGen-v2 seed0→Sana seed1, CXRMate-single fixed | 0/1→0/1 | 0/1→0/1 | 0/8→3/8 | Not passed |

The report-only and image actions use DIFFERENT baseline images and, for
report switches, different baseline experts. These are six paired observations
of TWO EHRs, not six independent patients or a randomized action-type ranking.
The Sana contrast changes model, original prompt renderer, native settings
and seed, not just architecture. No branch was removed because it failed.

## Interpretation / 目前能得出什么

- Report-only switches introduce an EHR–Report proxy opposition in both cases;
  one also loses its previously supported EHR fact. An unchanged image means
  a report-only action cannot resolve its EHR–XRV opposition.
- New RoentGen support is negative-label agreement only. The larger CXR–Report
  count does not resolve the fixed EHR–image proxy opposition.
- Sana gains two positive image–report supports per case, but also one/two new
  image–report oppositions. These positive gains do not establish support of
  the particular EHR constraint, which remains unsupported by the XRV label.
- All six original proxy gates fail. This is neither a 0% clinical repair-rate
  estimate nor proof that the images are faulty, the prompts are ignored, or
  every future action will fail. Existing independent observer diagnostics
  have not qualified a clinical image-fault judge.

中文：**目前不是“换了模型就修好”，也不是“生成器全都不行”。动作确实改变了
一些图文标签，但没有改善这两个固定 EHR 的目标图像约束。仅凭这些代理分数，
仍然不能区分条件生成不足与评分证据不足。** 原 EHR、prompt、评分器和选优结果
均保持不变，不把 unknown 当阴性，也不为了接受新结果而放宽门槛。

The next useful step is a separately approved conditioning/interface diagnostic,
not another scorer update or blind report retry. A small invented-text positive
control can check model response independently of the unchanged EHR extraction
path, but must remain a separate engineering experiment. It cannot be reported
as a repaired EHR-conditioned triple or independent clinical validation. A
stronger numeric-only API planner does not, by itself, resolve observer bias.

## Cost and verification / 实际调用和文件

Two named source jobs charged **20 downstream attempts**: report switches4,
RoentGen image chains8, Sana image chains8. Historical acquisition, observers
and Qwen planning remain sunk costs and are NOT included in this subtotal.
Shared Qwen/rule requests were executed once, not twice. Source worker wall
intervals are131.628s and212.143s, including startup/I/O; not GPU time, per-model
throughput or measured compute savings. This CPU join makes zero new calls.

Authenticated both run manifests and their postflights; replayed existing gates,
numeric receipt arithmetic, fixed EHR/fact hashes, six pairs/four image slots
and charged counts. Output artifact hashes, CSV row count, project permissions
and Git exclusion passed. Fourteen new invented-fixture tests and all **3,988
V1.2 tests** passed (full suite31.262s). Model inference, anatomy, tokenizer
content and report-structure correctness were not independently recomputed.

Protected output:

```text
artifacts/protected/tricompose_v1_2/fresh_action_effects/action_effects_12851223_001/
  action_effects.csv
  action_effects.json
  summary.json
  manifest.json
```

Manifest SHA256:
`98368c9c062dd8194300b2bca574d249f5def94c28a72166150ae46297c35294`.
The detailed table contains before/after support split by polarity, coverage,
opposition, missing comparisons, deltas, rejection codes, hashes and charges.
No report prose, rendered images, raw MIMIC data or credentials enter this
aggregate document or Git. No additional GPU job is approved.
