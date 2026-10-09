# Fixed-subset image-only verifier: completed

The [frozen protocol](cached_image_findings_protocol.md) remains unchanged.
This note records an actual frozen-model run, not a new clinical score or a
claim that the image/report errors are localized.

## Execution

After complete script/resource display and explicit approval, the exact
`TriCompose-v1.2/slurm/44_cached_image_findings_flexible_gpu.sbatch` was
submitted once as **12649136**, at 2026-10-04 14:27:33 server time.
It ran on **A40 b04-10**, two CPUs, 32 GiB RAM, with the unchanged ten-minute
cap: start 14:27:42, end 14:28:46, allocation **1m04s**, exit **0:0**.
Queue wait was nine seconds; no alternative submission or resource change.

Exactly six greedy image-only calls completed, zero retries/token-cap failures.
All six responses satisfy the original eight-named-state contract. Peak allocated
VRAM was **15.642 GiB**. Recorded loading/inference/comparison runtime was
33.42 seconds; this excludes earlier plan/full-checkpoint preflight and is not
the full allocation time. Input token lengths were 517–677, output 84–85.

The model received only each current synthetic CXR and the unchanged generic
image prompt. No EHR, report, old classifier label/score, model/candidate name,
winner or answer key was sent to it. Outputs export state/response hashes, not
raw response text. Predictions were fsynced before the comparison artifact was
parsed. No new image/report/EHR was generated; all existing models remain frozen.

## Unique image/finding comparison

The inventory is six images x 14 findings = 84 rows before report repetition.
Eight image heads give 48 supported checks; 36 other checks remain outside
scope. Raw historical XRV states are uncalibrated operational proxies, not
reference labels. The table measures model disagreement, not accuracy.

| Finding, six images each | Explicit agreement | Explicit opposition | Uncertain comparison |
|---|---:|---:|---:|
| Atelectasis | 3 | 3 | 0 |
| Cardiomegaly | 3 | 3 | 0 |
| Consolidation | 3 | 3 | 0 |
| Edema | 5 | 1 | 0 |
| Lung opacity | 4 | 2 | 0 |
| Pleural effusion | 4 | 2 | 0 |
| Pneumonia | 0 | 4 | 2 |
| Pneumothorax | 3 | 3 | 0 |
| Total supported checks | 25 | 21 | 2 |

Eleven oppositions are XRV-positive/Qwen-negative and ten the reverse; the
disagreement is not purely one classifier being more conservative. In particular,
XRV calls all six pneumonia checks negative, while Qwen returns four positive
and two uncertain. This does not establish which evaluator is correct or that
a generated image is clinically wrong. No threshold/template is retuned to
make these models agree; uncertain is not counted as a hard polarity conflict.

## Source-scope-retained report comparisons

All 336 original candidate/finding gate rows remain intact and in original order.
Only 17 report assertions were retained by the earlier unchanged source-scope
gate. Their relations with the new Qwen image readout are:

| Availability/relation | Candidate/finding occurrences |
|---|---:|
| Retained report and Qwen image explicitly agree | 17 |
| Within image heads but report assertion not retained | 175 |
| Outside the eight image heads | 144 |
| Total original rows | 336 |

The 17 matching occurrences comprise 16 negatives and one positive and cover
only **10 unique image/finding pairs in five image slots**. They are not 17
patients, 17 images or independent clinical confirmations. Among them:

| XRV / Qwen image / retained report states | Occurrences |
|---|---:|
| Negative / negative / negative | 7 |
| Positive / negative / negative | 9 |
| Positive / positive / positive | 1 |

Nine retained report assertions therefore remain opposed by XRV, even though
Qwen's image and report-derived states agree. The same Qwen checkpoint used
for image/text extraction plus a same-report literal guard is not an independent
clinical majority. No verified CXR truth, EHR edge, modality fault, candidate
acceptance or clinically resolved evidence request is established. The subset
is previously inspected development data, not an untouched final test.

## Outputs, audit and limits

Protected run, relative to `artifacts/protected/tricompose_v1_2/`:
`verification_runs/image_only_scope2_12649136/`.
Files: `predictions.json`, `image_report_comparison.jsonl`, `summary.json`,
`RESULTS_CN_EN.md`, `manifest.json`.

Manifest SHA256:
`07aef0cf22a59ecf6441a07601afca3989c95dd5a5b608be5c4ee05a2bce17e1`.

Independent checks verified 156 source bindings, four artifact hashes, all
six image/result hashes and source order, unchanged prompt hash, eight-state
schemas, token/call accounting, 336 preserved original gate rows, 84 unique
image/finding denominator entries, exact agreement/opposition/uncertainty
arithmetic, unchanged historical full-bank/gate/availability manifests and
project-only permissions (directories 2770, files 0660). No raw real patient
EHR/report/image or real target was consumed. The pre-run full suite passed
**1,279 tests**, including 25 new invented/mock fixtures.

All EHRs, original generations, score tables, winners and request histories stay
immutable. Primary-metric eligibility and automatic regeneration authorization
remain false; clinical accuracy stays unavailable. This is usable image-only
secondary evidence with explicit disagreement, not permission to pick a model
as truth or force regeneration until its score improves.
