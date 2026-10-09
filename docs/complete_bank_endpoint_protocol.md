# Complete historical candidate-bank endpoint coverage

This is a development-only extension of cached evaluation, not a new selector,
held-out clinical benchmark, targeted regeneration or human-adjudicated result.
All eighty fixed synthetic EHRs, 240 images and 960 report candidates stay fixed.
The historical fourteen-field profile is not mixed with fresh eight-head scores.

## Frozen request

`complete_endpoint_inventory.py` requests the exact set difference between the
entire hash-authenticated inventory and the old endpoint inventory. Endpoint
values, clinical labels, model names and winners do not choose the missing set.
Old NA results remain NA and are not retried. No case/candidate is discarded.

The already scored selected union covers 596 pairs. The new request contains
the remaining **364 pairs**, with at most **216 image encoder samples** and
**364 text encoder samples**. This supplies whole-inventory measurement rather
than an endpoint-selected subset. It does not permit endpoint-based reranking,
retrospective model-priority tuning or a claim that cosine is clinical accuracy.
Empty/overlength/special-token reports still become NA with a reason, not zero
or silently truncated text. Nominal inventory completion does not guarantee
every endpoint is numerically available.

## CPU preflight and GPU execution boundaries

`benchmarks/complete_bank_endpoint.py prepare` completed inside existing CPU
allocation 12621834 in 15.308925 seconds, with zero model calls. It verified
the earlier diagnostic source/result hashes, every synthetic candidate/parent
artifact hash, all three image/eight report runs, and the unchanged local
BioViL-T weights. The immutable request is under the protected V1.2 root:
`score_coverage_diagnostics/full_bank_endpoint_request_12621834_001/`.

The request manifest SHA256 is
`a856df5c3288f6e5bb48907a297c50078974d6ff046ae9545abf05422dee404f`.
No report bodies/image pixels were opened during preparation. Files are 0660,
directories 2770 within the project/NFS group boundary. Existing runs cannot
be overwritten; execution refuses an existing target before loading a model.

The separate score mode checks frozen plan/source hashes before and after
reusing the existing frozen, full-report BioViL-T worker. Output is a new
protected score run; old scores/choices remain unchanged. Evaluator runtime,
encoder sample counts and peak allocated GPU memory are additional evaluation
cost, not saved generation cost or GPU-kernel time. Failures are not retried.

`slurm/32_complete_bank_biovil_gpu_p100.sbatch` is **prepared, not submitted**:
normal gpu partition, one P100, two CPUs, 8G host RAM, ten-minute upper limit.
The latest resource snapshot showed non-drained P100 slots on the gpu partition;
availability is not an immediate-start promise. It instantiates no generator,
downloads nothing, trains nothing and changes no ranking/scoring threshold.
Submission requires the full script/resource request and explicit user approval.

The full V1.2 software suite passes **773 invented-fixture tests**, including
exact inventory difference, score-independent missingness, NA/no-retry semantics,
lineage rejection, allocation guards and refusal to overwrite before inference.
These are engineering checks, not completed new GPU metrics or clinical truth.

## Approved execution and completed merge, 2026-10-03

The prepared-only paragraphs above are historical snapshots. After the full
script/resources were displayed and the user explicitly approved, script 32
was submitted as **12624822**. It ran on **e21-03/P100**, completed exit `0:0`
in **47 seconds** (program 43.38084 seconds). It scored all 364 missing pairs,
using 216 image and 364 text encoder samples, no unavailable reports and
0.611 GiB peak allocated GPU memory. These are evaluation costs, not model
generation savings or GPU-kernel timing. No new EHR/CXR/report was generated.

`benchmarks/merge_complete_bank_endpoint.py` and
`src/tricompose_v12/full_bank_secondary.py` then authenticated the frozen
request, exact set difference, scorer provenance, every old/new endpoint pair,
encoder counts and NA semantics. The new protected merge is
`complete_bank_endpoints/complete_bank_secondary_12624822_001/` under the V1.2
protected root. It contains:

- `candidate_score_table.csv`: all 960 candidates, raw three-edge readouts and
  full-bank BioViL measurements; unavailable EHR-edge rates remain NA.
- `same_image_ties.csv`: all 1,440 unordered report pairs on the 240 fixed image
  slots, retaining all eighty EHR cases and their correlations.
- `RESULTS_CN_EN.md` and `summary.json`: selected-union versus full-inventory
  coverage diagnosis, not a new model ranking or clinical acceptance.
- `scores.json` and `manifest.json`: lossless old/new records and source/result
  hashes. The old 596 records are retained exactly; all 960 cosine values are
  available in this run.

The source primary-ranking prefix is tied in 79 report pairs. Previously only
25 tied pairs from 21 EHRs had both endpoints; now all 79 from 49 EHRs do.
The within-EHR-then-cohort mean absolute BioViL gap on ties is 0.4441033.
It is an absolute disagreement magnitude, **not** an improvement, clinical
accuracy or independent clinical truth. The bank remains already inspected
development data. All 72 cases lacking direct comparable cached EHR findings
are retained; this is not the different historical prompt-conditioning tier.

The initial CPU merge attempt refused equivalent relative/absolute source-path
spellings before creating output. Only the new merger was corrected to compare
resolved workspace-bound paths together with identical source hashes. No plan,
model, original score, generation setting or GPU task was rerun. Regression
tests cover equivalent spellings, changed paths and outside-workspace paths.

Exact record/table/report recomputation, all source/result/checkpoint hashes,
old-value retention and project-private permissions passed. The full V1.2
software suite now passes **786 invented-fixture tests**. The selected outputs,
policies, thresholds and clinical claims are unchanged.
