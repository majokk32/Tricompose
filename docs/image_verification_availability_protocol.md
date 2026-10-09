# Cached image evidence → immutable full-bank availability

This CPU-only handoff consumes the completed six-image check, not a new model
call. Preserve all existing scores, EHR anchors, findings, winners and request
execution histories. Append diagnostic `imageverify_` fields only; do not turn
agreement into clinical truth, a calibrated score, ranking or repair permission.

## Pinned, quote-free inputs

Under protected `tricompose_v1_2/`, consume only:

- `report_verification_availability/reportgate_pool960_12645021_001/`:
  original annotated candidate CSV, fact JSONL and request JSONL.
- `verification_runs/image_only_scope2_12649136/`: predictions JSON,
  image/report comparison JSONL and aggregate summary.
- `candidate_reliability_overlays/reliability_pool960_12632006_001/`:
  correlated report dependency groups, for the unchanged reliability validator.

Authenticate each fixed manifest and consumed artifact. Never recursively walk
their source paths, read report/EHR/image bodies, raw responses, reference keys,
real annotations, checkpoints or credentials. Recheck consumed hashes before
atomic commit. Run in the existing CPU Slurm allocation, with no new submission.

## Exact joins and counts

Image evidence requires exact CXR candidate identity AND SHA256. One image
readout may be reused across its linked reports; those are correlated uses,
not additional images or model calls. Report-relation evidence additionally
requires exact triple/finding/report identity, both artifact hashes and the
unchanged retained report assertion. A matching text or image hash alone must
not extend the checked scope. Recompute relations from unchanged four states
and reject inconsistent cached comparisons or gate metadata.

The fixed pool remains 80 EHRs / 240 CXR slots / 960 triples / 13,440 finding
rows / 2,072 logical requests. The image check covers six images / 24 triples /
336 rows; eight supported heads yield 48 unique image/finding checks, not 192
independent checks after report repetition. Explicitly preserve unsupported
heads, failed response/null, unknown, uncertain, unretained report assertions
and unchecked candidates. Unchecked decision counts are null, not zero errors.

Only image-finding and image/report-relation requests receive image-side
availability. Report-assertion, EHR observability and EHR conditioning requests
remain outside this handoff's remit. Preserve all consumer IDs/dependencies,
partial coverage and original execution flags; technical availability is not
clinical resolution. Same-Qwen image/text agreement and correlated report votes
cannot arbitrate XRV disagreements or identify a faulty modality.

## Acceptance and outputs

Worker: `TriCompose-v1.2/tools/attach_image_verification_availability.py`.
Tests: `TriCompose-v1.2/tests/test_image_verification_availability.py`, invented
metadata only. No training, API, inference, new threshold/prompt/rule, primary
metric update, winner change or regeneration.

Create a fresh atomic protected `image_verification_availability/` run, dirs
2770/files 0660/project group only. Outputs: candidate CSV, fact availability
JSONL, request availability JSONL, unique image/finding JSONL, summary,
bilingual result note and hash manifest. Verify exact old cells/order, all
source bindings, shared-image denominators, fixed scope and null semantics.
Show available and unavailable denominators without claiming model accuracy.
