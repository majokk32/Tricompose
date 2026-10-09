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

## Prepared post-run audit

`TriCompose-v1.2/audits/audit_fixed_image_reports.py` runs inside a CPU Slurm
allocation after the GPU run completes. It rehashes frozen checkpoint bytes and
source artifacts; recomputes original image receipts, all eight raw proxy rows,
pre-endpoint choices, paired endpoint comparisons and CSV; checks complete
encoder-count/missingness accounting, ordered pre-spawn reservations and private
modes. It does not open report bodies/image pixels, retokenize report content,
rerun embeddings, or independently validate clinical truth. Its new protected
audit run records a hash-bound execution check with zero additional model calls.

The original immutable GPU plan/source files are unchanged while its job is
pending. Adding this separate auditor does not change routing, thresholds,
costs, candidates or inference parameters. Audit readiness is not a completed
post-run validation or a new scientific result.

The complete suite, including the new journal/endpoint audit tests, passes
734 synthetic-only tests. Source files and original generation parameters
pinned by the pending GPU plan remain unchanged.

## Bounded existing-allocation observer

`audits/watch_fixed_image_reports.py` observes only approved GPU job 12622258.
It is running as a one-CPU, 1G step within the existing CPU allocation 12621834,
not a new `sbatch` job/GPU request. It polls accounting once per minute for at
most three hours, requires both `COMPLETED` state and a published manifest, and
then invokes the metadata auditor once with a five-minute audit timeout. It
cannot submit/cancel jobs, retry models, resume failed runs or change scores.
Terminal failure, missing publication or deadline expiry does not become
clinical failure/acceptance. This depends on the existing allocation remaining
alive and is not a guaranteed notification service.

Its private status directory is
`artifacts/protected/tricompose_v1_2/fixed_image_report_watch/`
`observer_12621834_12622258_001/`. The eventual audit, if completed, is under
`fixed_image_report_audits/fixed_reports4_audit_12622258_001/` in the same
protected V1.2 root. No report bodies or image pixels enter observer logs. The
initial CPU step could not find the relative `env` executable and exited before
the observer started; the retry used absolute executable paths and successfully
started the observer, with no model calls or original input changes.

All 740 synthetic-only tests pass, including mocked-clock deadline, terminal
failure and no-premature-audit tests. The GPU job is still pending, not a
completed result. Slurm's initial estimated 12:50 start is **PDT local time**,
equivalent to 19:50 UTC; estimates can change and are not completion guarantees.

## Completed run and observer, 2026-10-03

The pending statements above are retained chronological preparation snapshots.
Approved GPU job **12622258 completed**, exit `0:0`, elapsed **00:02:13**.
The run `fixed_image_report_runs/fixed_reports4_12622258/` under the protected
V1.2 root contains all four new MAIRA reports and all eight endpoint scores.
The frozen pre-endpoint selector retains CXRMate-single on all four images;
its EHR-averaged paired BioViL gain is zero. No EHR/image or original winner
changed, and unavailable direct EHR-edge evidence remains NA.

The CPU observer completed and its hash-bound audit passed recomputation of
proxy rows, selection, endpoint pairing, CSV, durable cost journal and protected
modes, without report-body/image-pixel review or new model calls. The manifest
still explicitly denies clinical acceptance, repair and independent clinical
truth. Allocation duration is not measured GPU-kernel time.

The separate [score coverage diagnostic](score_coverage_diagnostic_protocol.md)
now analyzes this fresh control and the complete historical candidate bank in
separate profiles. It preserves the zero-gain result and every original choice;
no threshold or policy is tuned to BioViL. Detailed diagnostic tables stay in
the new protected `score_coverage_diagnostics/score_coverage_12621834_001/` run.
