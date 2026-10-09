# Prospective guarded image-only verifier integration

Status: separate fixed engineering integration check; GPU execution requires
displaying its complete script/resources and subsequent explicit approval.
Historical workers, plans, scores, winners and EHRs remain immutable.

## Inputs and unchanged model contract

Use exactly the six original synthetic images and four already saved uniform
controls from completed run 12650073, with its 18 logical slots. Authenticate
the original image plan, completed control manifest and encoded/normalized
pixel bindings. No new controls, case selection, anatomy threshold, prompt,
checkpoint or clinical scoring rule is introduced. Controls never enter the
patient candidate bank or a CXR generator.

Prepare using metadata only in an actual CPU Slurm allocation. Do not decode
images, load weights or parse old prediction states. Bind the already frozen
guard implementation and Pillow decoder files from its completed CPU manifest.
Hash the baseline prediction bytes without interpreting their states. The
new plan pins the new worker/tests/protocol and transitive existing program
sources. Record the approved batch script separately at execution, avoiding
a circular plan-manifest/script-hash dependency.

On separately approved GPU execution, call `guarded_invoke` for each unique
input in normalized-pixel-hash order. Recompute the native guard; cached guard
decisions and old prediction states cannot select which input reaches inference.
Check normalized identity/dimensions on the exact in-memory image before model
loading or inference. Guard-blocked inputs receive null states and zero model
calls. A lazy callback loads the unchanged frozen Qwen model only when an input
passes the guard. Requests use the old eight-finding image-only prompt, greedy
seed zero, 384 output tokens, no retries and unchanged pixel limits. No EHR,
report, scores, candidate IDs, arm names or expected states enter requests.

Maximum budget: ten unique model calls, zero retries; the known fixed controls
are expected to block four calls, but that expectation is not an input to the
guard/model. Allocate one A40/A100/L40S GPU with at least 24 GiB VRAM, two CPUs,
32 GiB RAM and a ten-minute execution cap. Do not relax to smaller/unsupported
GPUs or resubmit without a new complete-script/resource display and approval.

## Accounting and comparison

Keep all ten unique input records and all 18 logical slots. Distinguish
blocked_before_model_call, complete and failed_unavailable; blocked/failed
states are null, not invented negatives or successful all-unknown responses.
Every attempted model inference consumes one call even if decoding fails;
an inference/runtime exception fails the run closed without a retry. Persist
and fsync the new predictions before parsing old baseline states.

Then compare complete unblocked states with exactly matching old observations.
Original repeatability is not clinical accuracy. Report guard counts, actual
model calls, blocked-before-call inputs, unavailable responses, token counts,
peak allocated VRAM and measured runtime. Compare call count with the recorded
ten-call unguarded run on these same inputs only. Fewer calls here are a
mechanical control-specific observation, not clinical efficiency/quality,
cohort-wide savings or GPU-runtime speedup; separate measured times/allocations.

The guard only checks bounded PNG decoding, supported native L/RGB and exact
spatial uniformity. Basic pass is not valid anatomy, clinical truth, error
localization or permission to regenerate. Do not rank or rewrite the 960-row
candidate bank in this integration check. Clinical acceptance, primary-metric
eligibility, selection changes and regeneration authorization remain false.

## Output and tests

New atomic protected `guarded_image_runs/<run_id>/`: ten predictions with guard
receipts, logical-slot bindings, quote-free repeatability metadata, summary,
bilingual result note and hash manifest. Pin consumed metadata/program/image
bytes, decoder files and checkpoint; recheck after inference. Directories 2770,
files 0660, project group only, no overwrite. Framework logs/caches/tmp stay
protected and offline; public Slurm logs contain sanitized status only.

Tests use invented metadata and mock guards/callbacks: no blocked inference,
exact image identity before callback, call budget, failure/null states,
deduplication, untouched inputs, actual Slurm/approval before allocation/import,
atomic cleanup/refused overwrite, baseline parsing after fsync and seal checks.
No real patient EHR/report/image/target or external API is used.
