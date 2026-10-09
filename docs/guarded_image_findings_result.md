# Prospective guarded image verifier: completed

The [separately frozen integration protocol](guarded_image_findings_protocol.md)
and [preparation/submission receipt](guarded_image_findings_preparation.md)
remain traceable. The complete script/resources were displayed and explicitly
approved before one submission of the unchanged script as **12651080**. No
replacement job, prompt, checkpoint, resource or guard-threshold change occurred.

## Actual execution / 实际运行

Server timestamps: submitted **2026-10-04 16:12:21**, started **16:13:54**,
completed **16:15:00**, exit **0:0**. Queue wait **1m33s**; allocation elapsed
**1m06s**, on **NVIDIA A100-PCIE-40GB, b01-20**, two CPUs and 32 GiB RAM.
The ten-minute cap was a maximum execution request, not measured inference
time or a queue-time guarantee. No requeue/restart was recorded.

The worker recorded **29.0984 seconds excluding preflight**, including lazy
model loading, guard/inference and analysis; this is not full allocation time.
Peak allocated VRAM was **15.642 GiB**. Input tokens **517–677**, output tokens
**84–86**. Model, prompt, seed, pixel limits, 384-token cap and zero-retry policy
were unchanged. All model weights stayed frozen.

| Input arm | Logical slots | Unique images | Actual model calls | Blocked before call | Complete responses |
|---|---:|---:|---:|---:|---:|
| Original synthetic images | 6 | 6 | 6 | 0 | 6 |
| Saved uniform controls | 12 | 4 | 0 | 4 | 0 |
| Total | 18 | 10 | 6 | 4 | 6 |

All four uniform frames were mechanically invalid / spatially uniform. Each
retains a record with null states/tokens/response hash; none reached the model
callback. These are not all-unknown model responses or disease negatives.
The six originals passed only basic validity, each with one complete named
eight-finding response, zero failures, zero retries and zero token-cap failures.
Controls never entered a CXR generator or the patient candidate bank.

The guard recomputed its decision from native pixels and passed the exact
checked in-memory RGB image to the lazy callback. Model requests contained
only pixels and the unchanged image prompt, not EHR/report bodies, scores,
IDs, arm names or expected states. New predictions were persisted and fsynced
before cached baseline states were parsed for repeatability comparison.

## Repeatability, not accuracy / 复跑一致性，不是准确率

The six originals have **46/48** matching finding states versus the completed
unguarded control run. Two changes were observed:

| Finding | Old state | Guarded-run state | Changed slots |
|---|---|---|---:|
| Consolidation | positive | uncertain | 1 |
| Pneumonia | positive | uncertain | 1 |

These are two image/finding readout slots, not two independently adjudicated
patient errors. Both transitions withdraw certainty instead of reversing
polarity. They must not become hard negatives, clinical contradictions or
automatic regeneration triggers. `uncertain` remains distinct from negative,
unknown and unavailable.

The old run used an A40 and this run an A100, but the cause of the two changes
has not been established. Do not attribute them to hardware, the guard or
image quality without evidence. Same-checkpoint repeatability is not clinical
truth; 46/48 is not a clinical accuracy estimate. No extra rerun or post-result
threshold/prompt adjustment was performed.

本轮证明调用前护栏确实能拦截这四张无信息图；不能证明其余六张图的解剖或
临床内容正确。两处“阳性 → 不确定”只记作复跑差异，不判定错误模态、不修图。

## Cost and scientific limits / 成本与结论边界

The same fixed ten unique inputs previously consumed ten calls; this guarded
run consumed six. **Four avoided calls** are specific to these known mechanical
controls, not cohort-level clinical compute savings. The unguarded allocation
was 54 seconds and the guarded allocation 66 seconds on different GPUs; this
is not a controlled runtime comparison and establishes no GPU-time speedup.

Basic PNG validity is not anatomy/crop/realism validation. A nonuniform noise
or wrong-anatomy image can pass. No independent clinical truth, error
localization, repair success, clinical metric improvement or primary-metric
eligibility is established. Clinical compute savings remain null and automatic
regeneration remains unauthorized. This is inspected development data, not a
held-out clinical benchmark.

The prospective worker is now tested, but older sealed workers were not
replaced. Original EHRs, report records, scores, winners, request histories and
960-row candidate tables are unchanged. This run did not attach its fresh
states to the full-bank score table. A subsequent append-only handoff can keep
the old and fresh states side by side with exact image ID/hash joins and
explicit uncertainty/coverage; it must not replace clinical judgments or
silently extend these six-image checks to the other 234 images.

## Protected artifacts and audit / 输出与审计

Output, relative to `artifacts/protected/tricompose_v1_2/`:
`guarded_image_runs/guardhook_scope2_12651080/`.

Files: ten prediction/guard records, 18 logical-slot bindings, quote-free
repeatability metadata, summary, bilingual result note and manifest. Directories
2770/files 0660, project-group boundary only; no existing run was overwritten.
No raw real EHR/report/image/target was opened, and no training, download or
external API was used. No follow-up GPU job was submitted.

Run manifest SHA256:
`850f90f5b4624bc15d26aedcdc50585238805a1cd3c28c3b276055ba3d046149`.

Approved script SHA256:
`53f08e84857dcd7b4107531f41785fd1ded0f6891ae1d400bfe8ebc42b45d0fa`.

Independent checks passed **172 workspace source bindings / four decoder
hashes / five artifact hashes**, exact ten-input and original/control scope,
call/null/eight-state contracts, guard receipt hashes, token and budget
accounting, 46/48 repeatability arithmetic and the two transitions, unchanged
sources and project-only permissions. The pre-run suite passed **1,385 tests**,
including 23 new invented/mock integration tests. Historical plan and output
files are sealed and were not edited during documentation.
