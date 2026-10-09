# Source-span selection: V2 completed; clinical qualification still pending

The frozen same-cohort review finished, but only 10/22 responses passed the
exact-quote contract. Existing whitespace alignment recovered one more. Eight
quotes still do not match source, two responses have incorrect polarity keys,
and one quote is ambiguous. Retain these failures; do not tune a new synonym
list or silently convert paraphrases into original evidence.

## Proposed interface

1. Keep the same 24 slots / 22 exact texts, hashes and fixed EHRs. No case or
   report replacement based on failure, score, disease or winner.
2. Deterministically enumerate original text spans with opaque IDs, character
   offsets and hashes. Declare splitting/bounds and test on invented fixtures
   before additional input access. Preserve sentence/line context, qualifiers
   and repeated occurrences. Do not rewrite clinical text.
3. Let the frozen model select existing IDs for four-head positive/negative/
   uncertain assertions, rather than type new quotes. Include report context;
   never EHR, image, generator identity, prior labels, scores or answer key.
4. Decode only declared keys, polarity arrays and existing IDs. Invalid IDs,
   missing keys, duplicate/conflicting assignments, token truncation and input
   bounds fail closed. No nearest-ID or paraphrase fallback. Restore evidence
   by looking up original spans; every repeated occurrence has a distinct ID.
5. Seal separately from the exact-quote baseline. Compare availability,
   polarity coverage, runtime and calls at candidate and distinct-text levels.
   No primary score update, reranking or regeneration is implied.

Valid source IDs prove that an evidence reference exists, **not that it supports
the assigned finding/polarity**. A span may describe a different disease,
history, qualified absence or section conflict. Source grounding is not image
factuality, EHR fidelity or modality-error localization.

## Gates before a new GPU pilot

- Invented tests: negation, uncertainty, qualified absence, historical/current
  statements, section conflict, shared disease lists and Unicode; deterministic
  exact hash/offset reconstruction, including repeated sentences.
- Unknown/failed/outside-scope are not negative or binary agreement. Keep all
  fourteen old heads, ten explicitly outside this interface's coverage.
- Preserve source EHRs, texts, strict receipts, scores and winners. This sample
  is exploratory development, not final testing or clinical qualification.
- Any clinical qualification needs independent held-out evidence, not a prompt
  selected after observing these 24 reports. Do not invent human labels or
  calibration when none is available.
- No training, downloads, external API or login-node inference.
- Show a complete new Slurm script/resource request and obtain explicit
  approval before new model calls. Previous approval covered the completed
  exact-quote experiment only.

## Implementation and sealed staging

Status: **V1 job 12639717 completed**, exit code 0:0, on A100 in 1m20s
(2026-10-04 02:03:00–02:04:20 server time). Its strict interface availability
is only **2/22**. After separate full script/resource display and explicit
approval, V2 job **12645404 completed** on A100 in 1m56s, exit 0:0. This is an engineering reliability step, not claimed
as TriCompose's paper innovation, validated scorer or clinically successful repair.

Source contract: `TriCompose-v1.2/interfaces/report_span_selection.py`.
Worker: `TriCompose-v1.2/tools/verify_report_span_selection.py`.
The separate `interfaces/` and `tools/` locations preserve the old experiment's
frozen benchmark/contract source inventory; no sealed old worker was edited.

Generic splitting was declared/tested before staging: line breaks and
non-numeric sentence/semicolon boundaries, at most 64 spans, 2,048 characters
per span and 8,192 per report. Every non-whitespace source character is covered
in original order. Full original context is supplied alongside numbered source
spans. No medical synonym list or label-dependent segmentation is used.
Repeated source sentences receive distinct location IDs. Limits refuse input
without truncation or replacing a case. A valid source reference intentionally
does NOT pass a semantic correctness check; one test explicitly demonstrates
that a wrong finding assignment can be syntactically valid.

All **32 new invented/mock tests** pass; complete V1.2 suite: **1,155 tests,
10.658 seconds**. Tests check context/negation/uncertainty preservation and
decoder behavior, not model clinical accuracy. CUDA/VRAM guards are mocked;
no model is loaded in tests.

Staging used existing CPU Slurm allocation 12632006 and read only the same
24 synthetic file slots, producing **22 ready requests, zero rejected**, with
3–14 spans per report. No raw patient input, EHR body, image, new Slurm/model
call, external API or download. Old scores and winners remain sealed.

Protected plan:
`artifacts/protected/tricompose_v1_2/report_span_plans/span_scope2_12632006_001/`.
Files: `plan.json`, `requests.jsonl`, `summary.json`, `manifest.json`.
Manifest SHA256:
`bc650307d1713f3f572cf3593a8965d7ccb67c2f702a3bba273e4e7112dc93ea`.
Exact bytes, offsets, hashes, deterministic request reconstruction, 187 source
entries, protected modes/group and overwrite refusal were independently checked.
`requests.jsonl` includes synthetic source context and remains protected.
`input_text_sha256` hashes the actual single user content supplied to the
tokenizer BEFORE the existing frozen chat-template wrapper, not an intermediate
renderer or a claim that template special tokens are part of that content.

Approved/submitted script: `TriCompose-v1.2/slurm/40_report_span_selection_flexible_gpu.sbatch`.
One GPU with >=24 GiB VRAM (V100/A40/A100/L40S), 2 CPUs, 32G host RAM, 20-minute
upper limit, at most 22 greedy model calls, seed 0, 512 output tokens, no retries.
Use the same checkpoint/loader as the completed exact-quote experiment; freeze
weights and assets. Requests see no baseline labels, EHR, image or scores.
The subsequent separate analysis verifies/reparses receipts and compares old
exact-quote and new span-interface availability and states, without ranking.
Same-checkpoint outputs are correlated, not two independent evaluator votes.

Completed receipts: `verification_runs/span_scope2_12639717/`; comparison:
`verification_runs/span_scope2_12639717_analysis/`, relative to the protected
V1.2 root. Fresh job-based IDs refuse overwriting. Display the entire script
and resources and obtain explicit approval before any further submission.

## V1 result, failure diagnosis and frozen V2 clarification

V1 made exactly 22 calls, no retries. Two responses passed; twenty failed
`polarity_inventory_mismatch`. All twenty returned each finding as an **array
of strings** rather than the required object with three polarity arrays. All
four heads fail this shape in each of those twenty responses: 80 finding values
are arrays, eight values in the two complete responses are valid objects. No
response hit the token cap. Array lengths are 0 (37 heads), 1 (39), 2 (three),
3 (one). Polarity was not encoded in these flat arrays and cannot be inferred
from the IDs themselves. Do not assign positive/negative from other scorers.

The V1 wording `Each value has ... keys. Each value is an array ...` is
ambiguous about hierarchy. This is a prompt-interface defect; the failed
responses do not establish bad synthetic CXR/report quality. A causal effect
of corrected wording remains a hypothesis until a separate run completes.

V1 has only eight explicit candidate/finding states (two positive/six negative)
and 88 unknown within its 96 supported checks. CheXbert/V1 comparison has five
same explicit states, zero opposed states and 91 not comparable. This lower
opposition is missing evidence/abstention, not clinical improvement. All 336
historical rows are retained, including 240 outside scope; EHR comparison facts
remain unavailable for these two cases. Old scores and winners did not change.

Result manifest: `6a0a56a76c04b2f60987cba98b8ef1b82bb999767dd3e98cb9f2ad345319b4d9`.
Comparison manifest: `fe9e44045d3bbd9889bfc39fcc7432c0a234f524cfa16dde899d4b64ed9a78f8`.
All 22 responses were independently reparsed against sealed requests; 336
candidate rows, offsets, source/output hashes and protected permissions passed.
Peak allocated VRAM 15.693 GiB, BF16, PyTorch 2.9.1+cu128.

V2 (`report-source-span-selection-v2-explicit-json-object`) changes **only**
the format instruction: explicitly say each finding's value is an OBJECT,
only its polarity-key values are arrays, retain all twelve keys, and provide
the complete four-object/twelve-array JSON shape. Placeholder empty arrays
are not clinical answers. Source texts, order, segmentation, source IDs,
decoder, limits, non-format clinical instructions, checkpoint/loader/seed and
token budget remain unchanged. Flat old responses still fail; no retrospective
normalization, polarity reconstruction or overwrite is allowed.

Code: `interfaces/report_span_selection_v2.py` reuses the frozen V1 inventory
and decoder; `tools/verify_report_span_selection_v2.py` configures a fresh private
instance of the unchanged V1 worker. It does not edit V1 on disk. Source hashes
pin both reused and new code. Eleven new invented/mock tests pass; full suite
**1,166 tests, 8.670 seconds**. Tests prove structural preservation/guards, not
model correctness. The analysis will count empty/all-unknown responses as well
as technically complete responses, so copying an empty template is not clinical
success. Same-checkpoint outputs remain correlated, not independent votes.

V2 staging in existing CPU allocation 12645021 produces the same **22 ready
requests / 24 slots**, identical original text and inventories, with 22 changed
prompt hashes. Preparation made no new model/Slurm/API calls. Plan:
`report_span_v2_plans/span_v2_scope2_12645021_001/`, manifest SHA256
`576483c1b4f544b7fd99f5b6e7aca94d8c22c83694294fd41cd3ab9d1aa74afc`.
All 189 source entries, original V1 manifests, permissions and exact requests
were checked. This is development on the observed failures, not held-out
qualification or clinical calibration.

Approved/submitted script: `slurm/41_report_span_v2_flexible_gpu.sbatch`; one eligible
>=24 GiB GPU, 2 CPUs, 32G host RAM, 20-minute cap, up to 22 greedy calls, 512
tokens, no retries. Every further submission requires a separate complete
script/resource display and explicit approval.

## V2 completed result and independent receipt audit

Job **12645404** completed on NVIDIA A100-PCIE-40GB in **1m56s**, exit 0:0
(2026-10-04 11:39:06–11:41:02 server time). Exactly 22 greedy calls, zero
retries, no token-cap failures; BF16, PyTorch 2.9.1+cu128, peak allocated VRAM
**15.744 GiB**. No new report/image/EHR generation, external API or training.

| Same-cohort interface | Strictly complete distinct responses |
|---|---:|
| Exact-quote baseline | 10/22 |
| Span V1 | 2/22 |
| Span V2, clarified hierarchy | 22/22 |

Of the 22 complete V2 responses, **19 select at least one source span** and
**three have all four heads unknown**. The latter are not counted as clinical
successes or automatically labeled erroneous: independent semantic evidence
is still missing. V1-to-V2 transitions are two complete-to-complete and twenty
unavailable-to-complete. No flat V1 response was retroactively repaired.

Across 24 candidate slots × four heads, V2 has **18 positive, 40 negative and
38 unknown** states. Across 22 distinct texts × four heads, counts are 17/40/31.
CheXbert/V2 comparisons have **41 same explicit states, one opposed explicit
state and 54 not comparable**. These are correlated extractor comparisons,
not clinical truth, independent votes or a measured generation-quality gain.
Unknown is not a negative. All **336** historical rows remain, including
**240** outside scope. The two fixed EHRs still lack explicit cached EHR-edge
reference findings; no EHR-edge score, old score or winner was changed.

Protected receipts: `verification_runs/span_v2_scope2_12645404/`, manifest
`0521476dddf9e86a8e61732bb3872df08d68e11274388d4b4fb0b0b8cc59c332`.
Protected comparison/report:
`verification_runs/span_v2_scope2_12645404_analysis/`, manifest
`4defd00b806b3187bc9657521bfae22b6b63e55e7d0b86812e4846344e67121b`.
All 22 sealed responses were independently reparsed; all three analysis
artifacts were reproduced exactly. Source/artifact hashes, unchanged parent
manifests, 336-row counts and protected group/modes passed. No report quotes
or original source text are copied to this document.

Next gate is independent assessment of finding/polarity, negation, temporal
scope and qualified absence on separately fixed held-out evidence. Valid IDs
and JSON cannot establish those properties. Do not use this same-cohort
format improvement as primary ranking, confirmed fault localization or repair
benefit, or expand to the full bank as though clinical qualification had passed.
