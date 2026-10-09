# Prospective guarded readouts → immutable full-bank table

This CPU-only handoff attaches the completed guarded verifier run, not a new
model call. Keep all old scores, winner/EHR anchors, gate states and histories.
Append `liveimage_` diagnostics only; new readouts are not calibrated clinical
scores, independent truth, error localization or regeneration permission.

## Fixed metadata inputs

Authenticate fixed manifests/artifact hashes for: the 960-row mechanical-guard
CSV; the earlier full-bank image availability fact JSONL; the original six-image
plan; guarded run 12651080 predictions/logical slots/summary; and completed
unguarded control run 12650073 predictions. Read only those bounded quote-free
artifacts. Do not follow manifests recursively into source paths, read EHR/
report/image bodies, response text, real references, weights or credentials.
No image byte/hash decoding is performed by this handoff.

Validate the ten unique guarded inputs / 18 logical slots, six original images
and four isolated controls, unchanged frozen image-only prompt, named four-state
contracts, guard receipts and actual six-call/four-blocked accounting. Match
original CXR candidate ID AND encoded SHA256, then case/EHR/facts hashes from
the original plan. Matching pixel/image/report hashes alone cannot extend scope.
Uniform controls never join to patient candidates. Keep original and new
observations in their separate source runs; do not overwrite either.

## Append-only semantics

Keep all 960 candidate rows and 13,440 fact rows in order. Every fact must still
match its candidate/report identity and artifact hashes. Only 24 triple slots
receive the fresh six-image readout. The remaining 936 candidates remain
not_checked with null states/counts, not zero contradictions or negative labels.
All 14 fact inventory rows remain; the new verifier supports eight heads.
Unsupported heads are outside scope with null readout/repeatability. Failed or
guard-blocked readouts are unavailable/null, not unknown or negative. Explicit
unknown and uncertain states remain those states.

Per checked image/finding, show old cached/unguarded state, new guarded state,
same/changed readout flag and source hashes. Same unknown/uncertain states may
match as technical readouts, not clinical agreement. Unknown and uncertain
states cannot produce a hard polarity contradiction. Use the unchanged
four-state relation function for new XRV/Qwen relations and exact retained
report assertions; missing/unretained report facts are not negative findings.
Report relations reuse correlated same-image reports/same Qwen image-text
checkpoint, not independent votes or clinical accuracy.

The deduplicated image/finding sidecar has six × 14 = 84 rows, of which 48 are
supported heads and 36 outside scope. Summarize two changed readout slots and
46/48 matches uniquely; repeated uses across four reports are separate logical
occurrences. Show changed candidate counts separately. Do not choose a new
winner, suppress old findings, update requests or trigger repairs. EHR edge
scores and their missingness are untouched.

## Execution, tests and output

Worker: `TriCompose-v1.2/tools/attach_guarded_image_availability.py`.
Tests: `TriCompose-v1.2/tests/test_guarded_image_availability.py`, invented
metadata only. Use the existing CPU Slurm allocation, no new Slurm submission,
GPU/inference/API/download/training or threshold/prompt modification.

Use a fresh atomic protected `guarded_image_availability/<run_id>/`, project
group only, directories 2770/files 0660, refuse overwrite. Output candidate CSV,
fact JSONL, deduplicated image/finding JSONL, summary, bilingual note and sealed
hash manifest. Recheck consumed metadata/program hashes and all original cells
before commit. Tests cover exact joins/no hash-only spread, excluded controls,
state/receipt/accounting failures, unknown-safe relations, changed readouts,
complete 14-head inventory, source preservation and atomic guards/cleanup.
Clinical flags remain false; zero clinically resolved requests and zero new
model calls. This is development evidence availability, not a new benchmark.
