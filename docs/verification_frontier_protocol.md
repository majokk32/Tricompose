# Verification frontier after prospective image readouts

This is a deterministic **planning-only** metadata handoff, not a clinical
selector, learned router, localization benchmark, or regeneration loop.
Use the existing CPU Slurm allocation. No model/API/download/new submission.
Do not read EHR/report/image bodies, source references, response text, weights,
credentials, or recursively follow parent manifest source paths.

## Frozen inputs and lossless history

Authenticate the named artifacts of the completed `liveimage_pool960_12645021_001`
manifest (SHA256 `0120146adc043b482f51ef515d70ad3dd1b78132c3b989b22e7bd918b3c68dcb`)
and original image-availability request manifest (SHA256
`d933d9ba37181e2eed31f3f92fd8831d514a313fc225c6a6bb6f4260ad08f1bf`).
Read the 960-row candidate CSV, 13,440-row fact JSONL, 84-row unique image/finding
JSONL, aggregate summary, and 2,072-row original request JSONL only. Bound
metadata reads to 64 MiB. Check exact candidate/case/report IDs and all four
artifact hashes; keep each EHR fixed and all 14 findings per candidate.
Readouts for the same image/finding must agree across its report consumers;
an equal artifact hash alone must never extend a checked candidate's scope.
Keep all original candidate and request cells, row order, execution statuses,
reason codes, scores and missingness. Append `frontier_` fields only.

## Operational priority, not a clinical score

Tiers are a transparent engineering convention; lower numbers mean earlier
investigation, **not** higher clinical severity, error probability, score,
optimal compute allocation, or permission to execute:

| Tier | Requirement |
|---|---|
| 0 | Obtain independent evidence for a changed same-image readout |
| 1 | Obtain independent evidence for explicit XRV/Qwen polarity disagreement |
| 2 | Resolve unavailable/unsupported verifier or unretained report scope |
| 3 | Perform an initial check for an unchecked exact artifact slot |
| 4 | Cached readout exists, but independent clinical qualification is missing |
| 5 | Assess fixed EHR observability/conditioning scope without inventing facts |

Changed readouts take precedence over scorer disagreement. Unknown/uncertain
cannot be hard polarity opposition. A missing/unretained report assertion is
not a negative finding. Stable explicit agreement is still unqualified, not
acceptance. The same Qwen checkpoint reading image and text is correlated
evidence; four reports per image are not four independent image votes.
Unobserved EHR facts do not authorize replacing EHR or adding disease/device.
No prior public-data AUROC or swap sensitivity is transferred to these slots.

Append availability to each original request without marking it executed or
clinically resolved. Match request dependencies and exact consumer IDs.
For image-report checks require both exact image and retained exact report
assertion; do not promote a shared image readout to report factuality.

Separately create supplemental logical requests for each checked image/finding
with a changed readout or explicit current scorer opposition. Deduplicate by
fixed case, exact CXR candidate ID, CXR SHA256 and finding, binding to the fresh
source manifest. Keep these separate from original request history even if an
original request touches the same fact. They request **independent evidence**,
not a same-model retry or a replacement image. IDs and ordering are deterministic.
All original report consumers are linked, but counted separately from unique
image/finding requests. An independent qualified executor is not supplied here.

## Hard limits and outputs

All model-execution, clinical-selection, clinical-resolution, regeneration and
primary-metric flags remain false; clinical scores and cost estimates remain
null. No call budget has been approved for this frontier. Logical requests,
candidate links, cached calls, and new actual model calls must be counted
separately. Candidate actions explain what evidence is missing, not which
modality is clinically faulty or which triple is best.

Candidate actions must also surface missing initial exact-image/report checks
even when the old heuristic did not create a corresponding request. An empty
next-request-ID list with an initial-check requirement means a request plan is
still missing, not that a candidate passed or that execution is authorized.

Worker: `TriCompose-v1.2/tools/build_verification_frontier.py`; tests use only
invented metadata. Write a fresh atomic protected run under
`verification_frontiers/<run_id>/`, refuse overwrite, directories 2770/files
0660 with project group. Emit `candidate_action_table.csv`,
`evidence_request_frontier.jsonl`, `supplemental_image_evidence_requests.jsonl`,
`summary.json`, a bilingual note, and a sealed source/artifact manifest.
Recheck all consumed hashes and preserved cells before committing. Tests cover
state safety, exact lineage, no hash-only spread, shared-image deduplication,
unchanged histories, ordering/determinism, no false authorization, and atomic
failure cleanup. This development frontier cannot establish clinical accuracy.
