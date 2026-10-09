# Frozen V2 authored-language result (job 12646689)

Status: completed, independently replayed metadata/metric audit. Development
language diagnostic only, not clinical accuracy or permission to regenerate.
The original [protocol](authored_report_span_protocol.md) and its sealed plan
are unchanged; this separate result note does not rewrite them.

## Execution

After full script/resource presentation and explicit user approval, the exact
`TriCompose-v1.2/slurm/43_authored_report_spans_flexible_gpu.sbatch` was submitted
once at 2026-10-04 13:02:09 server time. Slurm allocated one A40 on `b05-09`,
two CPUs and 32 GiB host RAM; the unchanged 10-minute cap was not a queue-time
promise. Execution ran 13:04:39–13:08:59, **4m20s**, exit **0:0**. Queue time was
2m30s. Frozen BF16 Qwen made exactly 56 calls, no retries, no token-cap failures;
output lengths were 113–141 tokens. Peak allocated GPU memory: **15.628 GiB**.
No real MIMIC row/report/image, new checkpoint, external API, training or
generation/repair was used.

## Same-input authored comparison

All 56 existing invented texts and all 80 designated checks are retained.
Expected states are the existing same-author language policy, not independent
clinician/image/span ground truth. No case or prompt was changed after seeing
results. Reference/baseline parsing occurred after predictions were fsynced.

| Diagnostic on 80 designated checks | Qwen span V2 | Frozen CheXbert |
|---|---:|---:|
| Four-state matches | 50/80 (62.50%) | 48/80 (60.00%) |
| Four-state macro F1 | 0.5930 | 0.6009 |
| Explicit positive/negative flips | 8 | 7 |
| Determinate output on uncertain/unknown reference | 18 | 13 |
| Uncertain-reference recovery | 2/20 | 8/20 |
| Unmentioned-reference recovery | 19/23 | 14/23 |

Qwen's designated matrix (reference → prediction):

- Positive: all 15 remain positive.
- Negative: 14 remain negative; eight become positive.
- Uncertain: two remain uncertain; 18 become positive.
- Unknown: 19 remain unknown; four become uncertain.

All 56 Qwen responses satisfy the source-ID/schema contract; four are
all-unknown. Reference integrity is not semantic correctness. In the authored
modal-may, cannot-be-excluded, cannot-rule-out and qualified-absence families,
the designated reference recovery is zero out of four for each. The uncertainty
problem is not a JSON parse failure or omitted model call.

Paired designated checks: 37 both correct, 13 Qwen-only correct, 11 CheXbert-only
correct, 19 both wrong. A two-check aggregate match gain is not evidence of
general clinical superiority; macro F1 and uncertainty recovery are worse here.
The 224 full-vector secondary checks have 194 matches (86.61%), but most of that
denominator is unmentioned findings. Do not substitute it for 50/80 or claim
clinical improvement from unknown agreement.

## Output and audit

Protected run, relative to `artifacts/protected/tricompose_v1_2/`:
`verification_runs/authored_span_v2_12646689/`.
Files: `predictions.json`, `details.json`, `summary.json`, `RESULTS_CN_EN.md`,
`manifest.json`. No report text, quotes, model responses or real reference rows
are exported. Directories are 2770 and files 0660 with project-only group access.

Manifest SHA256:
`2d104e65a350bdca9970d8e4b080e0b5a9c39b93a3572a4f3a30897e94fb82eb`.
Independent checks covered 184 source bindings, four result artifacts, all
56 input/prompt hashes, source span IDs/offsets/hashes, four-state derivation,
80 designated and 224 secondary metrics, exact family/finding/baseline results,
Markdown bytes, token/call accounting and protected permissions. The historical
full bank and prior report-verifier progress manifest remain unchanged.
The pre-run full suite passed **1,216 tests**, including 15 new invented/mock
fixtures. No code, annotation policy, score, threshold or winner was revised.

## Consequence for TriCompose

Keep Qwen as an unqualified secondary report-extraction readout. A valid source
ID cannot establish that its polarity is correct. These results do not authorize
using its positive predictions as image facts or automatic regeneration triggers.
Nor do they measure whether a generated CXR or report is clinically wrong.
Any future interface improvement must retain this failed diagnostic unchanged,
freeze a separate protocol and be tested on other fixed inputs; no retrospective
label repair or favorable-family filtering. New GPU work needs separate script/
resource approval, and independent clinical qualification remains unavailable.
