# Same-cohort frozen semantic report check

Status: after complete script/resource display and explicit user approval,
job **12639326 completed** on 2026-10-04 using one **L40S**, with exit code
0:0 and total batch elapsed **1 minute 56 seconds**. It ran 01:12:21–01:14:17
server time, earlier than the initial scheduler estimate. This remains a
secondary diagnostic, not a new primary consistency metric or a repair result.

## Fixed inputs and unchanged model

The preceding literal-scope smoke selected the first two sorted opaque EHR
case IDs from the historical full bank, keeping every three-image/four-report
combination. Reuse those exact **24 candidate slots / 22 distinct report text
hashes**. Do not choose a different cohort after seeing rule coverage, scores,
findings or winners. These two cases have no explicit cached EHR comparison
facts; that is not evidence that the synthetic EHRs are invalid.

Preparation validates the sealed parent plan, original metadata, source code
and literal-scope result. It does not read report bodies, image pixels, EHRs,
real targets or checkpoint weights. Small checkpoint configurations/tokenizer
files are hashed; the audited weight SHA and filesystem stat are pinned. The
GPU worker verifies all asset hashes before loading the frozen model.

Reuse Qwen2.5-VL-7B-Instruct, the existing `verify_report_evidence_qwen.py`
prompt, `request_messages`, strict `decode_evidence`, existing model loader and
greedy inference function unchanged. Four heads: cardiomegaly, consolidation,
pleural effusion and pneumothorax. Seed 0, `do_sample=false`, 512 output-token
cap, no prompt retuning, quote normalization, retries or model training.

Each request sees only one exact report text. It never sees EHR, image, model
name, cached labels, scores, answer key or prior ranking. Source whitespace,
newlines and Unicode remain unchanged. Every quoted assertion must align to
one unique contiguous source span. Negation, uncertainty and conflicting
assertions are retained rather than resolving them in favor of a label.

## Receipts and comparison

One inference per distinct report hash, at most 22 calls. Shared texts reuse
one receipt but keep all 24 candidate slots. A malformed or truncated response,
unmatched/ambiguous quote or unavailable input remains unavailable/unknown.
No success-conditioned retry and no report/EHR replacement is permitted.

The GPU stage atomically seals `evidence.json`, `raw_responses.json`,
`summary.json` and `manifest.json`. Quotes and model responses stay protected.
The separate analysis stage verifies those receipts and reparses each response
against exact source bytes before producing a quote-free
`assertion_comparison.jsonl`, `summary.json`, `RESULTS_CN_EN.md` and manifest.

All 336 candidate/finding rows are preserved. Compare four supported heads
side-by-side with old CheXbert states and frozen literal-scope states. Ten
unsupported heads stay outside scope, not negative. Report candidate-slot and
distinct-text denominators separately. Unknown/uncertain/unavailable states
are not binary agreements. Preserve original scores, ranking and winners.

Exact quoting proves traceability, **not semantic correctness**. Qwen can
misinterpret a quote or omit an assertion. Extractor disagreement does not
identify the faulty modality; less disagreement after abstention does not
mean clinical improvement. Clinical accuracy, primary metric eligibility and
fault-localization accuracy remain unavailable/unqualified. No EHR/CXR edge
is rescored and no regeneration is authorized by this task.

## Paths and execution

Worker: `TriCompose-v1.2/benchmarks/verify_legacy_report_semantics.py`.
Comparison contract: `TriCompose-v1.2/src/tricompose_v12/legacy_report_semantics.py`.
Tests: `TriCompose-v1.2/tests/test_legacy_report_semantics.py` (invented fixtures).

Validation: all 18 new tests pass, and the complete V1.2 suite passes **1,114
tests** in 10.647 seconds using the documented multi-package import paths.
The initial incomplete import-path invocation had four import errors; rerunning
with the repository's existing paths resolved them without source changes.
Batch shell syntax, new plan/source fingerprints (157 source entries), parent
output hashes, protected permissions and overwrite refusal were independently
checked. No report body or GPU model was opened in these preparation checks.

All artifact paths below are relative to
`artifacts/protected/tricompose_v1_2/`:

- Parent plan: `legacy_report_scope_plans/scope2_plan_12632006_001/`.
- Parent scope result: `legacy_report_scope_runs/scope2_checked_12632006_001/`.
- New plan: `legacy_report_semantic_plans/semantic_scope2_12632006_001/`.
- New plan manifest SHA256:
  `2c76baa73aea4b0af0bd4745ff164387727107a627a519524122c1ce84a99d2c`.
- Proposed receipts: `verification_runs/qwen_scope2_<job_id>/`.
- Proposed comparison: `verification_runs/qwen_scope2_<job_id>_analysis/`.

The proposed script is
`TriCompose-v1.2/slurm/39_legacy_report_semantics_flexible_gpu.sbatch`.
One GPU, two CPUs, 32 GiB host RAM, 20-minute time cap, `gpu` partition,
eligible V100/A40/A100/L40S nodes; the worker retains the existing >=24 GiB
VRAM guard. No specific node or A40 queue is forced. Drained nodes cannot run
the task; current free P100s are not compatible with this memory requirement.
The submission-time resource refresh showed one non-drained A40 with a free
GPU, but scheduler priority still left this job pending; availability is not
a start-time guarantee. Twenty minutes is a requested upper limit, not a predicted inference duration
or queue-start guarantee. No new downloads, external APIs or changes to
external environments/checkpoints.

Show the **complete** batch script and resource request and obtain explicit
approval before any submission. Outputs remain non-overwriting and atomic,
with protected directories 2770/files 0660 within the project-group boundary.

## Completed result and independent cached audit

The model made exactly **22 calls**, without retries. **10 responses passed**
the strict schema/exact-quote contract; **12 were unavailable**: nine
quote/source mismatches, two polarity-key schema mismatches and one ambiguous
quote location. These are verifier-interface failures, not twelve confirmed
bad reports. Peak allocated GPU memory was 15.604 GiB (BF16, PyTorch
2.9.1+cu128). This observation does not authorize lowering the memory guard.

Candidate-level counts, 24 slots per head:

| Finding | Positive | Negative | Unknown | Same explicit CheXbert/Qwen state | Opposed explicit state | Not comparable |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 7 | 5 | 12 | 7 | 2 | 15 |
| Consolidation | 2 | 10 | 12 | 1 | 0 | 23 |
| Pleural effusion | 2 | 10 | 12 | 7 | 0 | 17 |
| Pneumothorax | 0 | 10 | 14 | 5 | 0 | 19 |

Of 96 supported candidate/finding checks: 11 positive, 35 negative, 50 unknown;
22 same explicit states, two opposed states and 72 not comparable. All 336
fourteen-head rows remain present; 240 unsupported rows are outside scope.
These two EHRs still have no explicit cached EHR comparison facts. More
extracted assertions are **not improved clinical correctness**. No EHR/CXR
edge, score, winner or generated artifact changed.

Receipts: `verification_runs/qwen_scope2_12639326/`; manifest SHA256
`d543ac0287b40bc1596944f348a1780b76c742b60a5cdc7decaddc8d2352dca4`.
Comparison: `verification_runs/qwen_scope2_12639326_analysis/`; manifest SHA256
`86052b941f55749a16eed660c7bf33e2d086ce43741e4208ef82d67d7741bc6e`.
Source/output hash, count, lineage and protected-permission checks passed.

A separate **CPU-only, zero-new-call** diagnostic reused the existing frozen
Unicode-whitespace aligner on all 22 cached responses. It recovered only one:
diagnostic availability 10→11/22. Eight source mismatches remain after
whitespace folding, plus two schema failures and one ambiguous location. No
word changes, paraphrase matching, guessed polarity keys or preferred source
location were used. Original strict receipts/comparison remain unchanged.

Audit: `legacy_semantic_receipt_audits/receipt_scope2_12632006_001/`, with
`audit.json`, `summary.json`, `RESULTS_CN_EN.md`; manifest SHA256
`9295833b40cb5d65e1029e3d5ad43fb7b82a780b7ba227458202075dbe5b4cce`.
Code: `TriCompose-v1.2/tools/audit_legacy_semantic_receipts.py`. Nine new
invented-fixture tests pass; full V1.2 suite: **1,123 tests, 10.511 seconds**.
Offset/hash reconstruction, sealed receipts, permissions and overwrite
refusal were verified. Audit artifacts contain no quotes/model responses.

Next prepare the [source-span selection interface](report_span_selection_plan.md)
as a separately versioned development protocol, not a retrospective contract
relaxation or a qualified primary ranking metric.
