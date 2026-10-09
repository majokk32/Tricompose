# Mechanical image-validity guard: implemented and checked

Following the [completed no-information controls](image_abstention_control_result.md),
the [separately frozen guard contract](image_validity_guard_protocol.md) is now
implemented. This is a deterministic engineering safeguard before clinical
scoring, not a learned quality metric, anatomical validator or clinical benchmark.

## Fixed-input result / 固定输入结果

Exactly ten existing protected PNGs were inspected in CPU Slurm allocation
**12645021**: six original synthetic CXRs and four saved uniform controls.
No new model call, GPU job, external API, download or Slurm submission occurred.
No EHR/report body or raw real image/target was opened.

| Input | Unique images | Basic pass, not clinical | Mechanically blocked | Cached finding states retained |
|---|---:|---:|---:|---:|
| Original synthetic images | 6 | 6 | 0 | 48 |
| Uniform black/white controls | 4 | 0 | 4 | 0 |

The four uniform images are `artifact_invalid / spatially_uniform`. Their
**32** finding readouts are unavailable in the guarded view, including the
**four unsupported negative assertions** observed in the preceding diagnostic.
The complete original verifier records remain unchanged. Blocking an input
means null readouts, not negative labels and not a fabricated all-unknown model
response. Eighteen logical input slots deduplicate to these ten images; they
are not eighteen independent clinical cases.

四张纯黑/纯白图全部被拦截，对应的 32 个 finding 读数不进入可比较结果；
原始模型记录仍保留。六张原图仅通过基础检查，原来的 48 个读数未改。
这不证明原图解剖正确，也不解决 XRV 与 Qwen 的临床判断分歧。

## Candidate-table handoff / 候选表

The original 960-row candidate CSV now has an append-only `imageguard_` sidecar.
Exact original CXR candidate ID, encoded image hash and EHR lineage are required
for the join. **24** triple slots receive `basic_pass_not_clinical`; **936** remain
`not_checked`, with null basic-comparison permission. The six checked images
have four report candidates each, so 24 slots are not 24 inspected images.
Uniform controls never enter the patient candidate bank.

All **94,080 original candidate cells** and row order are preserved. No scores,
winners, EHRs, report records or request execution histories were changed.
Clinical acceptance, primary-metric eligibility and regeneration authorization
remain false; zero clinical requests are resolved.

## Interface and limits / 接口与边界

- `TriCompose-v1.2/tools/image_validity_guard.py`: bounded PNG inspection,
  hash-bound receipts, cached guarded views and the `guarded_invoke` pre-call hook.
- `TriCompose-v1.2/tools/evaluate_image_validity_guard.py`: fixed-input CPU
  evaluation and lossless candidate-table attachment.
- `TriCompose-v1.2/tests/test_image_validity_guard.py`: 31 new tests using
  invented buffers, metadata and mock decoders, not patient inputs.

`guarded_invoke` checks the input before invoking a callback and supplies the
same checked in-memory RGB image. Invalid/unavailable inputs do not reach that
callback. The callback still requires its own approved Slurm execution,
privacy protections, frozen-model contract and actual call/time ledger.
This hook is available but was **not executed with a model** in this evaluation;
historical sealed GPU workers were **not replaced**. Actual compute savings
remain null, not an inferred benefit from cached results.

Supported native PNG modes are L and RGB. Uniformity means every native channel
is exactly spatially constant. Corrupt PNG decoding is invalid; unsupported
formats/modes, bounds, changed hashes or unreadable inputs are unavailable.
16-bit/alpha/palette inputs fail closed instead of being silently converted
and treated as blank. A noise image, wrong anatomy or single-pixel variation
can still pass. No anatomy, crop, realism or disease threshold was introduced.

下一次新的评分 worker 可以接入这个接口，先检查图片再调用评分模型。
本轮只是接口与缓存结果验证，没有修改已冻结的历史 worker，也没有重新评分
或重新生成患者。护栏不能替代医学图像质量评价或独立临床证据。

## Audit and protected output / 审计与输出

Output, relative to `artifacts/protected/tricompose_v1_2/`:
`image_validity_guards/pngguard_scope2_12645021_001/`.

Start with `candidate_score_table.csv` and `summary.json`. Per-input guard
receipts, preserved verifier records, logical-slot bindings and the bilingual
result note are alongside them. Directories use 2770 and files 0660 with the
project-only group boundary; no existing run was overwritten.

Manifest SHA256:
`0c9654d2147b7268890527c5db5f85f0c455217f501f17484ddf23365983ccba`.

The full suite passed **1,362 tests**. Independent checks passed **177 workspace
source bindings / four external decoder hashes / six artifact hashes**, native
PNG extrema and exact encoded/normalized pixel bindings for all ten inputs,
unchanged original verifier records, all 94,080 source CSV cells, exact joins,
unchanged parent manifests and private permissions. Pillow 12.0.0 was reused
from an existing read-only environment for bounded CPU decoding only.
Recorded inspection/analysis runtime was 0.172679 seconds, excluding metadata
loading and hash preflight; it is not the whole allocation runtime.

This is inspected development data, not a held-out clinical test. Models remain
frozen. No clinical accuracy, modality-error localization, repair success or
prospective inference savings is established.
