# Bounded actual regeneration: two-case engineering trial

Status: approved GPU job **12632531 completed**, primary generation/verification
and metadata audit passed; no retry passed the replacement gate. Secondary
BioViL failed with an import error; separate recovery is prepared, not submitted.
This is a new, preregistered exploratory control, not a modification of the
completed full-pool selector or a validated clinical localization method.

## Fixed question and scope

Can one actually new RoentGen image and its new CXRMate report improve fixed
EHR-conditioned proxy evidence without losing existing support/coverage or
worsening report quality? Compare the old fixed RoentGen/CXRMate triple,
the old same-image four-expert static result, and the bounded retry result.
The static method has a different historical budget: this is NOT an equal-cost
efficacy or end-to-end compute-saving comparison. A subsequent untouched cohort
and equal-budget non-targeted/static controls are still required.

The completed source is the immutable `pool80_fresh_12631194` full-pool control.
EHR, canonical facts, final prompt text, checkpoints, preprocessing and scorer
thresholds stay fixed. The image seed alone changes from 0 to 1. RoentGen reads
EHR-derived radiology-style text, NOT structured EHR natively. The previous or
final report is never passed to the image generator. CXRMate-single receives
only the newly generated image, NOT structured EHR or old reports.

Choose the first two sorted opaque EHR anchors with explicit facts covered by
an enabled XRV head. This selection uses only EHR evidence and scorer-head
availability, not baseline image/report scores or BioViL values. Keep the
original 80-case denominator and disclose the direct-fact and enabled-head
strata. The other 78 EHRs are unchanged, not rejected or silently replaced.
This stratum trial is NOT a representative 80-case efficacy estimate.

## Decision and retry boundary

1. Authenticate the completed source, freeze both fixed/static references.
2. Direct EHR versus XRV explicit opposition triggers one `regenerate_cxr`
   action, explicitly labelled **heuristic**. It is not clinical fault
   attribution: the EHR extraction and classifier can themselves be wrong.
3. Missing comparable evidence -> abstain; no opposition -> stop.
   Several reports viewing one image are not independent localization votes.
4. A triggered case runs exactly one possible chain:

   ```text
   unchanged EHR-derived final prompt + RoentGen seed 1
      -> new CXR -> frozen XRV
      -> official frozen CXRMate-single -> frozen CheXbert
   ```

   The new report is a downstream refresh after image replacement; this is
   not a pure report-only repair or seed-based report diversity experiment.
5. A new triple must strictly improve fixed EHR-CXR support/opposition against
   BOTH the fixed and static references, preserve all previously comparable
   finding IDs and direct-EHR supports, preserve prior positive CXR-report
   supports, introduce no new opposition on any edge, and not worsen existing
   report structure/repetition/temporal-risk checks. Negative agreement alone
   is not an improvement. Unknown/uncertain cannot hide a conflict.
6. If the chain fails, is identical, or cannot pass all gates, retain the old
   static result and record unresolved/NA. Do not lower thresholds after seeing
   outcomes. Retaining the old result is not evidence of clinical correctness.
7. Save new selection separately; never overwrite the old winner or source run.

Baseline image findings are conservative proxy-preservation constraints, NOT
real ground truth. Strict gate passage is an exploratory surrogate outcome.
Changing image labels also changes the report's image reference, so improvements
in image-report agreement alone do not prove better medical fidelity.

## Hard budgets and receipts

Per triggered EHR, at most FOUR generation/verification attempts: one image
generator, XRV, one report generator, CheXbert. No operational retries or extra
sampling. Each subprocess has a 120-second limit; failures/timeouts are charged
before spawning and retained in the hash-chained fsynced case journal. A failed
phase prevents dependent phases. Interrupted runs are not automatically resumed.

After sealing the selection, run frozen full-report BioViL-T on the union of
old fixed, old static, and new candidates: at most six unique pairs, four image
encodings and six text encodings. This endpoint is never routing evidence.
These encodings, model loading, failed endpoint attempts and wall time are
recorded SEPARATELY and are not free or included in the four-call chain limit.
Historical generation/scoring costs remain shared sunk costs, not zero.
Overlength or failed secondary results stay NA, never become zero.

The GPU controller reuses the existing official frozen adapters unchanged.
Metadata audit replays cost journals, reconstructs fresh receipts and decisions,
checks source/input hashes, CSV values, endpoint isolation and protected modes;
it does not open report bodies/pixels and cannot validate clinical anatomy.

## Code and outputs

- `TriCompose-v1.2/src/tricompose_v12/bounded_regeneration.py`: pure route,
  seed-only request clone, cross-image preservation gate and decision.
- `TriCompose-v1.2/benchmarks/prepare_bounded_regeneration.py`: CPU Slurm
  source/model/input preflight; no factory calls or clinical body parsing.
- `TriCompose-v1.2/benchmarks/run_bounded_regeneration.py`: approved GPU
  controller, real workers, durable budget, sealed choice then secondary scores.
- `TriCompose-v1.2/audits/audit_bounded_regeneration.py`: CPU metadata audit.

All generated data, score tables, per-case comparisons, receipts and native
logs remain under `artifacts/protected/tricompose_v1_2/bounded_regeneration_*`.
No training, external API, real-target reading, checkpoint download, original
EHR editing, threshold fitting, weight tuning, or public clinical text/images.

Inspect current `noderes -f -g` before preparing a compatible GPU request.
The exact complete script/resources must be displayed and receive explicit
approval before sbatch. A source implementation or CPU preflight is not a
completed GPU experiment or permission to submit it.

## Completed CPU preparation and approved GPU submission

CPU preflight completed in the existing allocation `12625457` in 65.933 seconds.
The sealed plan is
`artifacts/protected/tricompose_v1_2/bounded_regeneration_plans/retry2_12625457_001/`;
its manifest SHA256 is
`5a4e3bfbfd7e87879e14ea545932e3fef6331f3440bb4ef2973e52355fec8542`.
The full cohort has 80 EHRs, eight with explicit direct facts, five with facts
covered by enabled classifier heads. EHR-only stratification selects opaque
source ordinals **5 and 11**. Both fixed RoentGen references subsequently trigger
the heuristic retry; that outcome did not choose the cohort.

An independent execution-loader CPU check completed in 30.559 seconds with zero
factory calls, zero model calls, no clinical bodies/pixels, and all frozen
source/checkpoint/input pins verified. Its protected receipt is
`bounded_regeneration_audits/retry2_preflight_12625457_001/`, manifest SHA256
`afe892ecea08c3cdb9f362cf5ebaa2e61c6f7a311bc420b00aa3a26cc0eeb29d`.
Project-private plan modes/group passed. The full V1.2 suite has **951 passing
synthetic-only tests**, 14.192 seconds in that CPU allocation. Syntax and batch
shell checks passed.

Fully displayed and subsequently explicitly approved script:
`TriCompose-v1.2/slurm/35_bounded_regeneration_debug_p100.sbatch`.
Script SHA256:
`3719f74c44f789f64a08e92b2df2021bae1c50a49ad9d53cda28dff69a75b39c`.
Resources: debug/P100 ×1, two CPUs, 24G host RAM, twenty-minute wall-time upper
limit, no array. Current resource checks showed idle debug P100s and drained
A40/A100 nodes; this is a time-specific snapshot, not a start-time guarantee.
The script hashes the separately pinned metadata auditor before execution.
After explicit approval, the unchanged script was submitted as **job 12632531**.
It started immediately on debug node `e23-02` with one P100. The submission
receipt is `bounded_regeneration_audits/submission_12632531/`, manifest SHA256
`141b377d5b8a5bef877067c4b4c973e99949097729156e808a1565eca5ec376e`.
The completed output is `bounded_regeneration_runs/retry2_12632531/`, manifest
SHA256 `eb8099458d895da46281f1955c8e73f19082b61b0759bb8064bcb79ff922c32d`.
Slurm elapsed **4m48s**, exit `0:0`; controller wall time 232.978 seconds.
Both new chains completed: two images, two reports, eight charged primary
generator/verifier attempts, zero primary failures. Automatic metadata audit
passed, manifest SHA256
`58def50df7b72bc1d2cd6fc99bd4dd411f10241cc3633625d2c1e98e846a53f5`.

### Actual negative result

Neither retry passes the gate: **0/2 accepted replacements**. Both original
static results are retained and all historical winners remain unchanged.

| Aggregate over two fixed EHRs | Fixed RoentGen/CXRMate | Historical static | New retry candidates | Retained selection |
|---|---:|---:|---:|---:|
| EHR-CXR support / known | 0/2 | 0/2 | 0/2 | 0/2 |
| EHR-CXR proxy opposition | 2 | 2 | 2 | 2 |
| EHR-report comparable / known | 0/2 | 0/2 | 0/2 | 0/2 |
| CXR-report positive support | 0 | 1 | 0 | 1 |
| CXR-report negative support | 0 | 0 | 8 | 0 |
| CXR-report comparable / 16 known image references | 0/16 | 1/16 | 8/16 | 1/16 |
| Unsupported temporal language flags | 2 | 1 | 2 | 1 |

New negative agreement does not fix either direct-EHR/image opposition.
The first retry slightly increases repeated-4gram ratio; the second loses old
positive image/report support and increases temporal-risk language relative to
its static reference. Gates and thresholds were NOT relaxed after seeing this.
No clinical fault-localization accuracy, repair efficacy, or equal-cost
superiority has been demonstrated by this two-case control.

Immutable protected bilingual summary:
`bounded_regeneration_summaries/retry2_12632531/RESULTS_CN_EN.md`, manifest SHA256
`8f35bcaffc4280decea0d79ded7a12fd94eb69320ec18a4953463ccec1e03669`.
No source report body or image was opened for CPU reporting; summary/table
hashes and project-private modes passed checks.

### Missing secondary endpoint and completed recovery

The original BioViL attempt failed after 7.651 seconds with a sanitized
`ModuleNotFoundError`. The new direct CLI invocation omitted the pinned local
`health_multimodal` vendor bootstrap used by the previously successful workers.
This is an invocation defect, not evidence of generation quality. The failed
attempt is retained/charged; its original NA values are never overwritten.

`audits/complete_bounded_regeneration_endpoint.py` restores the same existing
vendor path in a separately pinned secondary-only worker. It authenticates
the completed source and audit, encodes only the five original requested pairs
(at most four images/six texts), and cannot regenerate EHR/CXR/report, change
thresholds or update the sealed choice. A separate result sidecar will hold
recovered scores; the historical failed output remains immutable.

CPU preparation passed in 34.676 seconds with no factories or model calls.
Plan: `bounded_regeneration_endpoint_plans/retry2_endpoint_12625457_001/`,
manifest SHA256 `17a0e2de31bcf003723411ec32613a4cc1285d46be428cfd80c54d6d26610f6d`.
The complete suite now has **963 passing synthetic-only tests** (12.820s).
Fully displayed, explicitly approved and submitted unchanged script:
`TriCompose-v1.2/slurm/36_bounded_retry_secondary_debug_p100.sbatch`, SHA256
`634491f348e3342737b9b5dda5856fd54f6defec5699ac2cbc763941d8df335b`.
Resources: debug/P100 ×1, two CPUs, 8G host RAM, five-minute upper limit.
**Job 12633003 completed in 24 seconds**, exit `0:0`, on `e23-02` without waiting.
No new generation, primary rescoring, choice change or threshold update occurred.
Recovery wall time including loading/I/O was 16.356 seconds; peak VRAM 0.583 GiB.
All five requested pairs are available, with four image and five text encodings.
The original 7.651-second failed attempt remains separately charged.

| Two-case method | Full-report BioViL raw cosine mean | EHR-CXR support / known | EHR-CXR proxy opposition |
|---|---:|---:|---:|
| Fixed RoentGen/CXRMate | -0.0909 | 0/2 | 2 |
| Historical static | 0.0673 | 0/2 | 2 |
| New retry candidates | 0.7446 | 0/2 | 2 |
| Retained bounded selection | 0.0673 | 0/2 | 2 |

The secondary image/report gain does not cure the fixed-EHR proxy opposition.
Neither candidate passes the original primary gate; no post-hoc change to
selection, thresholds or weights was made. These are uncalibrated raw cosines,
not probabilities or proof of clinical correctness. Two development cases do
not establish clinical repair efficacy or fault-localization accuracy.

New, immutable output:
`bounded_regeneration_endpoints/retry2_endpoint_12633003/`, including the
five-row `score_table.csv` and `case_comparison.csv`.
Independent CPU metadata audit passed in 5.090 seconds:
`bounded_regeneration_audits/retry2_endpoint_audit_12633003/`, manifest SHA256
`da0a9843b636d187b291f05069d887cf5a131ce9c6bed808c581b383139874df`.
The audit authenticated source/checkpoint/input hashes, unchanged sealed choice,
all score identities/counters, CSV recomputation and project-private modes;
it opened no clinical bodies or image pixels and made no model calls.
All 12 recovery/summary synthetic-metadata regressions were rerun successfully.

New protected bilingual handoff:
`bounded_regeneration_summaries/retry2_with_secondary_12633003/RESULTS_CN_EN.md`,
manifest SHA256
`a7b6b47013f9bce8137de6bc24b9f9b2cfddfacc62bed2bbd3cbf610009a4caa`.
Its artifact hashes and protected permissions passed verification. The original
trial and original NA summary were not modified. Any additional GPU job still
requires complete script/resource display and explicit approval.
