# Mechanical PNG guard before image clinical scoring

Status: separately frozen engineering safety contract, following the completed
no-information diagnostic. Not a learned image-quality scorer or clinical test.
The old V1.1 constant-grayscale validity check supplies the basic principle;
this guard adds explicit fail-closed semantics and an optional pre-call hook.

## Decision scope

Only hash-bound protected synthetic single-frame PNGs are handled. File content
is bounded to 64 MiB; image sides to 64–4096 pixels, inherited from the current
verifier's bounded input contract, not tuned clinical quality thresholds.
Supported native modes are L and RGB. Unsupported formats, modes (including
16-bit/palette/alpha images), frame counts, sizes, unreadable/missing inputs or
changed hashes are verification_unavailable, never clinically false images.
Decode failures of a bounded, hash-bound PNG are artifact_invalid.

After native decoding, each channel must have an exact min/max range. If every
channel is constant, mark artifact_invalid / spatially_uniform. This includes
uniform black, white, gray and colored frames; it does not confuse different
constant RGB channel values with spatial variation. Do not convert a 16-bit
image to 8-bit and misclassify lost dynamic range as blank. No near-uniform,
entropy, anatomy, crop, realism or disease threshold is added.

Everything else is basic_pass_not_clinical. A gradient, noise image, wrong
anatomy or single-pixel variation can pass: that is a stated limitation, not
clinical acceptance. Unsupported/unavailable images stay blocked without
inventing patient findings or automatic regeneration permission.

## Public hook and cached view

`TriCompose-v1.2/tools/image_validity_guard.py` provides a bounded PNG inspector
and `guarded_invoke`: inspect before a callback and invoke it only on basic
pass, using the exact in-memory normalized RGB image checked by the guard.
Block invalid/unavailable inputs before callback invocation; return null
readouts, not negative or all-unknown labels. The callback keeps its own Slurm,
approval, frozen-model, privacy and call/time-ledger obligations; the guard
never grants inference/API authorization. No callback/model is run in this
engineering evaluation. No already frozen GPU worker is replaced or edited.

A cached view preserves each original verifier record exactly. If an image
is blocked, its guarded readout is null; otherwise preserve the complete named
four-state vector or failed/null contract. Source predictions are not corrected
or overwritten, and no blocked observation is silently dropped. A checksum is
lineage evidence, not independent clinical truth.

## Fixed CPU evaluation and join

Worker: `TriCompose-v1.2/tools/evaluate_image_validity_guard.py`.
Tests: `TriCompose-v1.2/tests/test_image_validity_guard.py`, invented metadata/
mock decoders only. In the existing CPU Slurm allocation, inspect exactly the
six original synthetic images and four saved uniform controls from run
12650073. Authenticate the original fixed image plan/synthetic provenance,
the completed control manifest and all consumed artifacts. Bind guards to
encoded source SHA256 and normalized-pixel hashes; do not select by old labels.
Pillow is imported only for this bounded decoding, no torch/model/weights/API.

Attach `imageguard_` availability only to the immutable 960-row candidate CSV
by exact original CXR candidate ID AND image hash. Four control frames must
never enter the patient candidate bank. The 936 uninspected slots remain
not_checked/null, not invalid or zero errors. Preserve every source cell/order,
score, winner, EHR and all request histories. Do not expand checks by hash alone.

Report unique and logical input counts, pass/invalid/unavailable reasons,
blocked readout slots and retained cached counts. These are mechanical checks
on inspected development data, not clinical accuracy, false-repair rate,
improved consistency or actual model-call savings. No new generation, GPU job,
training, download, external service, EHR/report body or real target is used.

Use a fresh atomic protected `image_validity_guards/` run, dirs 2770/files 0660,
project group only. Recheck all consumed source/program hashes before commit.
Seal versions and Pillow decoder metadata. Output guard/cached-view/input
metadata, original candidate CSV plus guard sidecars, summary, bilingual note
and manifest. This new hook is available for future explicitly approved
execution; it is not silently installed into historical jobs or rankings.
