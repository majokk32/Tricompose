# No-information image controls: preparation receipt

Status update: after complete script/resource display and explicit approval,
the unchanged experiment completed as **12650073** on A40 b04-10 in 54 seconds,
exit 0:0. See [the completed result](image_abstention_control_result.md).
The preparation record below describes what was verified before submission.

The [frozen protocol](image_abstention_control_protocol.md) adds a mechanical
safety diagnostic rather than repeating old swap experiments. Existing image
scorer disagreements cannot currently decide which modality is wrong.

Preparation completed inside CPU Slurm allocation **12645021** with **zero
model/GPU/API calls and zero new submissions**. It consumed hash-bound source
metadata and code, no image pixels/bytes, EHR/report bodies or full weights.
The cached reference predictions were hashed but their states were not parsed
to select cases. Original generations, scores, winners and EHRs remain unchanged.

## Fixed experiment

The same six source images have three logical arms each: original rerun,
same-sized uniform black, same-sized uniform white (**18 logical slots**).
The unchanged image-only frozen model sees only pixels and its old generic
eight-finding prompt. Exact normalized RGB buffers including dimensions are
deduplicated at execution, so actual calls may be below the hard **18-call**
cap. Repeated uniform controls are not independent patients. No retries;
384 generated tokens, seed zero, greedy mode.

Report unknown and uncertain separately. Positive AND negative assertions on
no-information controls are unsupported; parser failures are unavailable,
not successful abstentions. Keep unique and logical counts, full availability
denominators and missing-result bounds. Original readout agreement with the
cached run tests repeatability only, not clinical accuracy. No new ranking,
calibration, clinical fault label or regeneration permission follows from this
experiment. Blank frames must never enter generation as a previous CXR or as
replacement patient artifacts.

## Plan, tests and approval boundary

Protected plan:
`artifacts/protected/tricompose_v1_2/image_control_plans/noinfo_scope2_12645021_001/`.

Manifest SHA256:
`60dda9d2f15ace54c3e121fab3adba1285ed47a4b49ba3b3ec92fac945c7c217`.

Independent **154-source/one-artifact** hash, exact plan-rebuild, fixed slot
inventory, unchanged prompt/budgets and project-only permission checks passed.
The full suite has **1,331 passing tests**, including 25 new invented buffer/
mock tests; no actual protected images were decoded in those tests.

Script: `TriCompose-v1.2/slurm/45_image_abstention_controls_flexible_gpu.sbatch`.
Resource request: gpu partition; one A40/A100/L40S-compatible GPU with >=24 GiB,
two CPUs, 32 GiB RAM, ten-minute execution cap. Caches/temp/logs protected;
offline frozen local checkpoint/environment. The cap does not bound queue wait.
The latest read-only node snapshot lists free A40 capacity as mixed/plnd rather
than guaranteeing immediate availability; drained nodes are not usable.

At preparation, no submission or observed safety/repeatability result existed.
Subsequent complete-script display and explicit approval authorized exactly
one submission. Actual protected output:
`tricompose_v1_2/image_control_runs/noinfo_scope2_12650073/`.
Any further submission still requires its own complete-script/resource display
and explicit approval; this receipt does not authorize another run.
