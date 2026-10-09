# Prospective guarded verifier: preparation and approved execution receipt

After the complete script/resources were displayed, the user's explicit
approval was received. The unchanged script was submitted exactly once as
**12651080**, at **2026-10-04 16:12:21 server time**. Initial scheduler status:
**PENDING (Priority)**, no node assigned and no inference started at that check.
It subsequently completed **16:13:54–16:15:00**, exit **0:0**, on A100 b01-20.
See the [completed result](guarded_image_findings_result.md) for actual calls,
guard outcomes and repeatability limits. No resource change or replacement
submission occurred.

The [integration protocol](guarded_image_findings_protocol.md) is implemented
in `TriCompose-v1.2/tools/verify_guarded_image_findings.py`. It calls the already
sealed mechanical guard before a lazy frozen-model callback. No historical
worker, clinical prompt, checkpoint, score, winner, EHR or request history was
edited. This is an engineering integration check, not clinical repair.

## Frozen preparation

Protected plan, relative to `artifacts/protected/tricompose_v1_2/`:
`guarded_image_plans/guardhook_scope2_12645021_001/`.

Manifest SHA256:
`2864941d29f60e890cdd51e88a7126986027b495345d4cf8a0ff23bdc2b77959`.

Prepared in existing CPU Slurm allocation **12645021**. Exactly **ten** unique
already-existing inputs, **18** logical slots: six original synthetic images
and four saved uniform controls. Original images span the same fixed two EHR
cases and three generators. No image bytes/pixels, full weights, EHR/report
bodies or real targets were opened during preparation. Baseline prediction
bytes were hash-bound but states were not parsed. Zero inference/API/model
calls, downloads or new submissions occurred.

Independent checks passed **159 workspace source bindings / four frozen
decoder hashes / one plan artifact**, exact six-original/four-control inventory,
deduplication and logical-slot coverage, source-file metadata, fixed prompt/model
settings, unchanged sources and project-only permissions. Directories 2770,
files 0660; no overwrite. The full suite passed **1,385 tests**, including
**23** new invented/mock tests. This preparation made no prospective GPU call;
the later approved execution is reported separately.

## Approved execution request

Complete script:
`TriCompose-v1.2/slurm/46_guarded_image_findings_flexible_gpu.sbatch`.

Script SHA256:
`53f08e84857dcd7b4107531f41785fd1ded0f6891ae1d400bfe8ebc42b45d0fa`.

Resource request: project account ruishanl_1185, gpu partition, one GPU with
A40/A100/L40S feature (worker requires at least 24 GiB), one node/task, two CPUs,
32 GiB RAM and ten-minute execution cap. No hard node binding. Queue time is
not included in that cap; the pre-submission `noderes -f -g` listed free high-memory cards
on drained nodes, so immediate execution cannot be promised. Smaller P100/V100
requests are not silently substituted for the unchanged worker contract.

Maximum ten unique model calls, zero retries, unchanged eight-head prompt,
greedy seed zero, 384 output tokens and original pixel limits. The worker
recomputes the guard from native pixels, not prior cached decisions. Blocked
inputs remain null and never invoke the callback. The expected six-pass/four-
blocked pattern is a known development expectation, not clinical evidence or
a model input. The completed execution recorded six actual calls and four
blocked-before-call inputs; this is mechanical integration, not clinical truth.

Predictions must be saved and fsynced before old baseline states are parsed.
Original state matches indicate repeatability, not truth. Any reduction versus
the previous ten-call run is specific to these mechanical controls, not
cohort-level efficiency, clinical quality or GPU-runtime speedup. Valid PNG
variation does not establish valid anatomy.

Output was committed atomically under `guarded_image_runs/guardhook_scope2_12651080/`.
Protected offline caches/tmp/logs are per-job. Existing candidate tables are
not rewritten or reranked; no new generation, clinical resolution or automatic
regeneration is authorized. The script/resources were displayed and separately
approved before the submission recorded above. Any further submission needs
a new complete-script/resource display and explicit approval. Preparation and
submission do not establish that GPU execution completed or passed.
