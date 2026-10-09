# Frozen image verifier: no-information controls completed

The [predeclared protocol](image_abstention_control_protocol.md) and sealed
preparation remain unchanged. After the complete script/resources were shown
and the user explicitly approved, script
`TriCompose-v1.2/slurm/45_image_abstention_controls_flexible_gpu.sbatch` was
submitted exactly once as **12650073**. No replacement submission, prompt,
checkpoint, threshold or resource change occurred.

## Actual execution

Server timestamps: submitted **2026-10-04 15:08:57**, started **15:09:11**,
completed **15:10:05**, exit **0:0**. Queue wait was 14 seconds; allocation
elapsed **54 seconds**, on **A40 b04-10**, two CPUs, 32 GiB RAM, unchanged
ten-minute cap. Recorded model loading/inference/analysis took 37.7266 seconds,
excluding earlier plan/checkpoint/pixel preflight; that is not the full job time.
Peak allocated VRAM **15.642 GiB**. Input tokens 517–677, output 84–85.

All **10 unique calls** returned valid eight-state outputs, with zero retries
or token-cap failures. There were 18 logical slots: six originals, six black,
six white. Source dimensions were 512×512 and 1024×1024. Uniform frames of the
same dimensions were deduplicated, leaving six original inputs plus two black
and two white inputs. Model requests contained only pixels and the unchanged
generic image prompt, not EHRs, reports, IDs, arm names, scores or expected
answers. Predictions were fsynced before cached baseline states were parsed.

## Safety result: unsupported absence despite no anatomy

| Input arm | Logical slots | Unique frames | Finding slots | Positive | Negative | Uncertain | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original rerun | 6 | 6 | 48 | 18 | 28 | 2 | 0 |
| Uniform black | 6 | 2 | 16 | 0 | 2 | 0 | 14 |
| Uniform white | 6 | 2 | 16 | 0 | 2 | 0 | 14 |

All **four negative assertions are cardiomegaly**, one on each unique uniform
frame. Every other head returns unknown. In total, **28/32 control finding
slots (87.5%) abstain as unknown**, while **4/32 (12.5%) assert unsupported
absence**. No positive assertion occurs, but that does NOT make the verifier
safe: a frame containing no anatomy also cannot support disease absence.
Each of the four unique blank frames fails the strict all-eight-unknown
condition. These are four mechanical controls, not four patients or independent
clinical cases. No unavailable responses occur, so the predeclared missingness
bounds coincide with the observed 12.5% explicit-assertion fraction.

The original rerun matches the cached named states **48/48**, with zero
unavailable comparisons. This demonstrates repeatability in this run, not
clinical accuracy. The earlier XRV/Qwen disagreements persist in the cached
original states; this repeat does not determine which scorer is correct.

## Interpretation and next boundary

The current verifier mostly follows no-information abstention, but cannot be
trusted to do so perfectly. Its stable original outputs and valid JSON are
not clinical correctness. The result motivates a separately frozen mechanical
image-validity guard before clinical scoring, rather than changing thresholds,
forcing scorer agreement or treating absence as a default on invalid images.
Such a guard should reject uninformative artifacts; it cannot certify that a
non-uniform image has valid anatomy or correct clinical findings. It is not
installed by this diagnostic.

The existing six images and all EHRs/reports, candidate scores, winners and
request histories remain unchanged. Uniform controls are isolated verifier
artifacts only—never prior CXRs, generator inputs or accepted replacements.
No independent clinical accuracy, modality-error localization, clinically
resolved request, repair success or automatic regeneration is established.
This remains inspected developmental data, not an untouched final benchmark.

## Protected artifacts and independent checks

Output, relative to `artifacts/protected/tricompose_v1_2/`:
`image_control_runs/noinfo_scope2_12650073/`.

Files: predictions, realized-input metadata, original-repeatability metadata,
summary, bilingual result note, four lossless uniform PNG controls and manifest.
No report text, EHR body, raw real image/target or patient identifier was used.

Manifest SHA256:
`87d1e211e21fe497bda8647ae356a5c5a61246801bd5a288a828263f4f908ecd`.

Approved script SHA256:
`9e1c574214e2dd59613c56c06f361431ba3528b1aaa7f60ae53daf34960eea47`.

Independent standard-library checks passed **162 source bindings / nine
artifact hashes**, exact prompt/source/baseline identity, 18 logical slots /
10 unique records, eight-head states, per-arm dedup/count/rate arithmetic,
48 cached-original state matches, four saved PNG hashes/dimensions, unchanged
earlier availability/baseline/plan manifests and protected project permissions.
The pre-run suite passed **1,331 tests**, including 25 new invented/mock tests.
All models remain frozen; no training, download, external API or further
submission occurred.
