# Fixed-image report expert control

This is an offline, training-free diagnostic after the successful live frozen
worker run. It is not adaptive regeneration, error localization, or a validated
clinical selector. The original outputs and selection decisions stay immutable.

## Fixed inputs and added work

Keep the two original synthetic EHR anchors, their final prompts, and all four
fresh synthetic images: two image generators per EHR. Keep the four existing
CXRMate-single reports and their fresh XRV/CheXbert evidence. Add one MAIRA-2
report for each exact image using the unchanged official current-frontal-only
adapter. Neither report expert receives EHR content, a prior image/report, an
indication, or a real target. CXRMate-single is not CXRMate-ED.

This creates eight graph pairs, not eight independent patients. Both fixed EHR
anchors have zero cached direct radiographic constraints. Their EHR-edge rates
remain **NA**, not zero or success. This experiment therefore cannot establish
three-modal agreement or EHR-to-image fidelity.

## Predeclared proxy selection

`src/tricompose_v12/fixed_image_reports.py` contains the exact policy. For each
fixed image, minimize this lexicographic key:

1. Explicit XRV–CheXbert opposition count + missing positive XRV finding count.
2. Negative of supported positive XRV findings.
3. Prefer the original CXRMate-single path on a tie.
4. Opaque report candidate ID.

Missing positive means the report labeler returned unknown or uncertain for an
explicit positive image proxy label. This prevents omission from improving the
first term relative to an explicit denial. Unknown/uncertain never become
negative. Negative agreement is reported but is not a selection reward. There
is no learned weight, new scorer, probability calibration or label refitting.
Frozen XRV head masks and thresholds remain unchanged and pending independent
review. Labels and report assertions remain unverified proxies; missingness can
be a labeler error rather than a report omission. The selector cannot confirm
which modality is wrong, and correlated report experts are not independent votes.

## Independent secondary measurement

Write `score_rows.json` and `selection.json` before any BioViL call. Bind their
hashes into the independent endpoint request. Reuse the existing frozen BioViL-T
worker's full-report policy: no silent truncation, empty/overlength/special-token
reports have NA plus a reason. Score all eight image/report pairs. The endpoint
is raw, uncalibrated cosine, not clinical correctness or a probability. Choices
are never changed after seeing these values.

Compare selected versus fixed-baseline cosine on each unchanged image, average
the two images within each EHR, then average the two EHR-level deltas. If any
required pair is unavailable, keep that EHR's delta and the complete-cohort mean
NA. Do not convert missing values to zero or silently exclude cases. Two EHRs
are an engineering check, not a significance or paper-grade held-out experiment.

## Provenance, execution, and outputs

`benchmarks/prepare_fixed_image_reports.py` runs inside an existing CPU Slurm
allocation. It authenticates the completed source run/audit, pins original
artifact hashes and source code, strongly hashes the MAIRA checkpoint, and
prepares four image-only report requests. It instantiates no runtime/model.

`benchmarks/run_fixed_image_reports.py` refuses non-GPU Slurm execution before
creating outputs. MAIRA, CheXbert, and BioViL run sequentially in their existing
local environments, offline, with private stdout/stderr, bounded timeouts and
owned-process-group termination. All outputs live under `artifacts/protected/`;
only the newly created run's modes are normalized to project-private access.
Existing runs cannot be overwritten. Interrupted runs retain reservations and
artifacts for audit, without automatic resume.

Cost accounting is explicitly separate from the original single-case router
ledger: source cost is retained as 16 charged single-sample calls. New work is
four report generation samples, four CheXbert scored samples, and at most four
image/eight text BioViL encoder calls. CheXbert batches four samples in one
forward pass; scored-sample units are not forward-pass counts. Each child has a
durably flushed pre-spawn reservation; process success and artifact validation
are separate events. Worker wall time includes loading/I/O and is not GPU-kernel
time. No inference-cost-saving claim is made.

The new run contains `score_table.csv`, `score_rows.json`, `selection.json`,
`secondary.json`, `comparison.json`, `cost_journal.jsonl`, and a hash-bound
`manifest.json`, plus protected report/scorer artifacts. Report text and images
are never copied into documentation, chat, Git, or public logs.

## Prepared state, 2026-10-03

CPU preflight passed in 43.230 seconds, with **zero new model calls**. The
immutable plan is under `artifacts/protected/tricompose_v1_2/`
`fixed_image_report_plans/fixed_reports4_12605930_001/`.
The complete V1.2 synthetic-only test suite passes **722 tests**. This does not
claim the new GPU run has executed or its reports are clinically correct.

`slurm/30_fixed_image_reports_debug_a40.sbatch` is prepared only: debug, one A40,
two CPU cores, 48 GiB host RAM, twenty-minute upper limit. A40 is requested
because the unchanged MAIRA adapter uses approximately 27.5 GB float32 weights;
the previously used 16 GB P100 cannot hold that implementation. Queue snapshot
availability is not a scheduling guarantee. Submission requires explicit user
approval after the full script is shown.

On the subsequent resource check, debug's A40 node was drained; the gpu
partition had an A40 resource snapshot instead. Script
`31_fixed_image_reports_gpu_a40.sbatch` preserves the frozen plan and resource
sizes, changes the partition to `gpu`, and restricts the two newly created
Slurm log files to mode `0660`. The earlier debug script remains unchanged.
After the full updated script/resource request was shown and the user explicitly
approved proceeding, script 31 was submitted as **job 12622258**. The first
status check was `PENDING (Priority)`, not execution or new scoring success.
Availability is not an immediate-start guarantee. No report/endpoint quality
improvement is claimed from submission; completed artifacts require validation.
