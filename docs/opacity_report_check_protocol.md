# Opacity report-assertion check: frozen-model development diagnostic

Prepared scope only. New GPU inference needs full script/resource presentation
and explicit subsequent approval. No automatic repair, reranking, training,
new clinical gold, source editing, or image/EHR input to the verifier.

## Bound subset and unchanged denominators

Use the completed metadata conflict registry manifest
`87674da73de24b840f4baf08a1cd91fcd98e6d5f0d687026ab47ed3ec6b0600d`.
Its **16 unique report-byte hashes / 36 candidate contexts** are all retained;
no further selection by text, score, readability or model. Other bank members
remain outside this bounded investigation, not failed/removed from the 960-row
bank. Resolve every alias through hash-bound frozen V1.1 report-run/candidate
metadata and fixed EHR/fact/image lineage. Select an input file deterministically
among exact-byte-identical aliases. CPU preparation hashes report files but
does not decode text, open images or run a model.

The diagnostic asks **what a text asserts**, not whether it describes the CXR
correctly. Qwen2.5-VL-7B is the existing locally pinned frozen checkpoint;
one text-only task per exact report hash, with source contexts reattached later.
Neither scores, cached labels, case/candidate/model IDs, image data/paths nor
EHR/history enter model messages. Investigator plan/manifest are not access-
isolated from the project group; this is model-input separation, not perfect
blinding or independent clinical adjudication.

## New one-finding quote prompt and contract

Freeze the prompt before any source text inspection/model run. Reuse the
previous frozen Qwen loader/inference and exact-quote contract design, but
do **not** modify the consumed four/eight-finding sources or pretend that the
new one-finding prompt has prior validation.

Request `lung_opacity` only, with positive/negative/uncertain arrays of at most
two contiguous source quotes each, 3..256 Unicode characters. Quotes retain
negation/uncertainty qualifiers. Do not infer a finding just from a named
different disease or expand generic no-acute-disease into all negatives.
Qualified absence is not global absence. The LLM interprets relevant assertion
semantics; exact quote matching checks traceability, **not semantic correctness**.
No new regex clinical classifier, synonym mapping, threshold or scorer weight
is fitted. Scope may differ from CheXbert; differences are not certified errors.

Require exactly the finding/polarity inventory, no duplicate keys/quotes,
verbatim uniquely located quotes with offsets/hash. Missing assertion -> unknown;
any uncertainty or opposed positive/negative quotes -> uncertain. All three
arrays empty is a completed unknown proposal, not an extraction failure.
Malformed/truncated/ambiguous/unbound evidence -> failed-unavailable with
unknown placeholder; never count that as a matching clinical unknown.
No retries, paraphrase correction, substring rescue, section preference,
favorable prompt/seed selection or label flipping.

Keep every report including nonempty/bounds/UTF-8/token/model failures. Character
bound 8192; no silent text shortening. Each text has one primary invocation
unless unavailable before inference. Repeat original positions 0 and 1 once
as replay diagnostics without replacing primary outputs. Add **six wholly
invented controls** for explicit positive/negative/uncertain/unknown, opposing
assertions and qualified absence. Controls share the same prompt/config, but
their expected states are never in model input and are not clinical gold.
Full completion expects **24 invocations**: 16 primary, two replay, six authored
controls. A failed replay does not erase the primary. Report actual attempted
counts and every unavailable denominator; do not invent 24 successful calls.

## Frozen runtime and outputs

Existing read-only environment:
`/project2/ruishanl_1185/SDP_for_VLM/envs/attack/bin/python`.
Existing read-only model:
`/project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct`.
Weight SHA256:
`26fa644c8e61aa8185da5055b61aa28d6897fee102c2f43173acab657c4257a1`.
Bind config/tokenizer/processor/template files, package versions, loader/helper
source closure, source report metadata/hash and new worker/tests/protocol.
Offline local-files-only loading, eval/requires_grad false/inference_mode,
greedy decoding, seed 0, max-new-tokens 512, no images or API. Select fp16/bf16
with the unchanged frozen loader; report actual dtype/GPU. GPU must expose at
least 24 GiB; no quantization, new installation or downloading.

Atomic new protected run, no overwrite, group-only 2770/0660. Save quote evidence
and raw model responses privately, not in README/Git/chat/public logs.
Public output only sanitized status/runtime/memory/hash.

Outputs: `report_evidence.json`, `raw_responses.json`, `replay_checks.json`,
`authored_control_readout.json`, `candidate_text_readout.csv` (all 36 contexts),
`summary.json`, `RESULTS_CN_EN.md`, `manifest.json`. Preserve old CheXbert states,
the original dependency patterns and every fixed EHR/image/report hash. New
four-state proposal relation distinguishes support/opposition, unavailable
response and completed non-comparable evidence. No new total score/winner.

Post-run independently reparse all responses against the approved synthetic
source text **inside Slurm**, verify quotes/offsets, fixed state reducer, costs,
all 16/36 joins, replay deltas, authored-control denominators, hashes/modes and
unchanged historical sources. Exact agreement of two extractors remains a
correlated text proposal, not reference accuracy or fault attribution. Full-
text linguistic/extraction checks do not validate EHR/CXR clinical content.

Worker: `TriCompose-v1.2/tools/check_opacity_report_assertions.py`.
Tests: `TriCompose-v1.2/tests/test_opacity_report_check.py`.
Consumed code/plan/protocol and existing runs must remain immutable.
