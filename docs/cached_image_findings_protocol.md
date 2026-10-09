# Fixed six-image, image-only frozen verifier follow-up

Status: protocol for preparation; no GPU submission authorized by this file.
The current fixed two-case subset contains all three historical CXR model paths
and four reports per image. This follow-up adds image-only evidence rather than
repeating report extraction or treating source-scope commits as clinical truth.

## Fixed inputs and unchanged interface

Use exactly the same two opaque cases / six seed-zero CXR slots / 24 reports
as `legacy_report_scope_plans/scope2_plan_12632006_001/`. No selection by score,
assertion coverage, pathology, apparent image quality or previously chosen winner.
Keep all EHRs and original image/report artifacts fixed.

Reuse `verify_candidate_findings_qwen.request_messages('image', image=...)`, its
eight named findings, original prompt, strict decoder, infer helper and existing
frozen Qwen2.5-VL loader/checkpoint. The four-state policy remains positive,
negative, uncertain, unknown. Failed decoding/token caps yield unavailable/null
states, not a successful all-unknown response. No new label vocabulary, prompt,
threshold, temperature, scoring weight, training or model download.

Each request contains only its current synthetic image and the generic image
prompt. Never send an EHR, report, model/candidate ID, old label, score, winner,
intervention answer or provenance path to the model. Image ordering is bound
by existing image hashes and opaque IDs, not outcomes. Six calls maximum,
384 generated-token cap, greedy seed zero, zero retries; each image assessed
once and its result reused for the four report comparisons.

## Metadata-only preparation

Authenticate the pinned scope plan, original full-bank manifest, existing model
plan and scope-gate manifest. Resolve selected CXR metadata and original request
metadata without opening image bytes, EHR bodies, prompts or reports. Check
synthetic SynEHRgy staging provenance, text-only generation inputs, fixed hashes,
models, seed and complete 2 x 3 x 4 inventory. Reject missing/changed artifacts,
real-source provenance and unsupported models, rather than replacing a case.
Checkpoint stats and small configuration/tokenizer hashes may be checked here;
full weight and image hashing only happen inside approved GPU execution.

The historical teammate's original V1 staging header is under `TriCompose-v1.0`
inside the workspace, outside protected/. Read only its pinned, bounded
`run_manifest.json` to confirm a known SynEHRgy generator; do not load or copy
any EHR body, source case, prompt or cohort there. All image inputs and new
plans/results remain protected; original teammate files are never modified.

Freeze a new protected `image_only_plans/` run with model hashes/stats, unchanged
prompt hash, exact six-image inventory, budgets and program/test/protocol hashes.
Source dependencies and existing plans remain read-only.

## GPU execution and separate comparison

New execution requires complete script/resources shown and explicit approval.
Use one A40/A100/L40S-compatible GPU with at least 24 GiB, two CPUs, 32 GiB RAM
and a ten-minute execution cap (queue wait separate). Offline assets only;
all caches/temp/logs under a new protected task directory. No external API.

Verify frozen weights and exact image hashes before use. Disable model gradients
and keep eval/inference mode. Preserve raw responses only as hashes, states,
token/call counters and runtime/memory; no raw response text is exported.
Write/fsync/hash-freeze predictions before parsing the report-gate comparison
artifact. Source paths, IDs and state rows remain in protected artifacts only.

Compare XRV/Qwen at unique image/finding level (6 x 14 inventory, eight supported
heads = 48 checks). Separately compare Qwen image states with source-scope-retained
report assertions across all 336 candidate/finding rows. Preserve every original
gate cell and append `imagecheck_` fields. Withheld report assertions, missing
image evidence, unsupported findings and uncertainty remain explicit rather
than producing agreement. Same-image report repetition does not multiply the
independent image count. Report all availability/decision denominators, polarity
opposition and support descriptively, without selecting the best subgroup.

## Interpretation and invariants

This is a previously inspected developmental synthetic subset, not an untouched
test or independently annotated image benchmark. Distinct models are not
independent clinical votes. Agreement/opposition cannot establish which modality
is faulty; no benchmark accuracy is transferred to a candidate confidence.
Image-only evidence does not validate an EHR edge, and all original scores,
rankings, winners, EHRs and request histories remain unchanged. No clinically
resolved request, primary score update or automatic regeneration is authorized.

Worker: `TriCompose-v1.2/tools/verify_cached_image_findings.py`.
Invented/mock tests: `TriCompose-v1.2/tests/test_cached_image_findings.py`.
Fresh atomic protected runs only, files 0660/directories 2770, project access
group only. Every new submission requires its own approval; a prepared plan is
not a running or completed verification experiment.
