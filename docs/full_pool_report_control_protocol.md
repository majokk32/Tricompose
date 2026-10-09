# Full fixed-pool report benchmark / 全固定池报告对照

Status: **completed and metadata-audited**, approved job 12631194 on 2026-10-03.
Slurm elapsed time was 87 seconds; controller wall time 64.740 seconds and
CPU audit 14.629 seconds. XRV scored all 240 images; CheXbert scored all 960
reports. Peak allocated VRAM was 0.045 GiB for XRV and 0.545 GiB for CheXbert.
Every new Slurm submission needs the exact complete script/resources shown
first and explicit approval afterwards. No generators, training, API or
downloads are part of this protocol.

## Fixed inventory and provenance

Retain all 80 original fully synthetic EHRs, including cases without directly
comparable radiographic findings. Use their original Sana/PixArt/RoentGen-v2
seed-0 CXRs (240 images) and original CXRMate-single/MAIRA-2/LLaVA-Rad/
CheXagent-2 reports (960 texts). No new EHR, image, prompt, report or seed is
generated, no case screening is done, and original winners are not changed.
All four report experts here are CXR-only. CXRMate-single is not CXRMate-ED;
text-conditioned CXR generation is not direct structured-EHR-to-CXR.

Staging reads metadata and hashes only. All source request manifests and
requests, canonical synthetic EHRs/facts/final prompts, image/report bytes,
source code, checkpoints and scorer configuration are hash-bound without
opening any clinical body or image pixels. Direct EHR states/categories come
from the authenticated cached evidence interface; this is not a new raw-EHR
review. Legacy image/report label states are not reused as fresh labels.

Immutable prepared plan:
`artifacts/protected/tricompose_v1_2/full_pool_report_plans/pool80_fresh_12621834_001/`.
Plan manifest SHA256:
`ffa4691cb2df67ba6d7c8e8a7bbd235fff1b592d82620837f6b26e1011ae65ca`.
CPU preflight took 17.064 seconds in existing allocation 12621834, with no
model factories instantiated or new inference calls. Initial implementation
and all previous regressions: 917 tests passed in 12.512 seconds.

The separate metadata preflight audit passed for all 80/240/960 records and
found **8 EHRs with direct comparable radiographic states and 72 without**.
This is a different criterion from prompt/context conditioning coverage; do
not relabel the 72 as invalid EHRs or silently exclude them. Audit directory:
`artifacts/protected/tricompose_v1_2/full_pool_report_audits/pool80_preflight_12621834_001/`.
Audit manifest SHA256:
`e64d4f42fff34348c5c81af729deac5a54b1b042bff82f417b1718ebe859773e`.
The reviewed-script SHA256 is:
`a1135800b4d8f8f12864b4be8559617922655f14220156cb504397edf27f698e`.
This audit read no endpoint values, report bodies or image pixels, and made no
new inference calls. It does not clear a clinical-validity gate.

## Fresh scoring, unchanged report-choice rule

Re-run frozen XRV on all 240 images with exactly the same operating-point-
normalized score space, preprocessing, checkpoint, eight-enabled-head mask and
already-frozen threshold bundle as the two-case control. Re-run frozen
CheXbert on all 960 reports, requesting batches of 16. Its legacy `model_calls`
field counts samples, not physical forward invocations. Never interpret XRV
scores as calibrated disease probabilities or fit new thresholds on this pool.
Thresholds remain pending independent review, not a clinical truth source.

Recompute synthetic report structure internally in the approved GPU controller;
public logs must not expose generated report bodies. All three raw edges use
the same fresh receipt profile and positive/negative/uncertain/unknown states.
Unknown and uncertain are not negatives. Weak context is not a direct EHR
finding, and lack of an EHR fact is not a contradictory negative observation.

On each SAME image, use CXRMate as the baseline. Test MAIRA-2, LLaVA-Rad and
CheXagent-2 in the unchanged predeclared order. Preserve supported finding IDs
and all baseline comparable IDs; do not introduce explicit proxy opposition
or silence existing conflict into unknown. Require strict evidence-set
improvement, nonduplicate text, the existing per-model section contract, and
no worse generic/temporal/repetition flags. This reuses the two-case gate
without relaxing it. If no alternative passes, keep the baseline unresolved.
This is static same-image report selection, NOT error localization, online
targeted regeneration, report editing or an adaptive agent.

## Secondary measurement and statistics

Seal choices and their SHA256 BEFORE loading the previously completed full-
report BioViL-T score cache. Preserve all 960 original pair IDs, hashes,
scores, explicit NA/status reasons and checkpoint/text-policy provenance.
No new BioViL encoding is required. Historical generation and encoder costs
remain sunk costs, not free inference or an adaptive compute-saving result.

Report all 12 fixed CXR/report combinations plus the same-image first-eligible
selector for each CXR generator (15 comparison rows). Keep three images per
EHR; do not choose a CXR or filter an EHR because its secondary score is poor.
Report raw known/comparable/positive-support/negative-support/opposition/
missing counts and explicit rates/NA, section success, temporal flags and raw
BioViL cosine. A low opposition count can reflect missing descriptions or
negative-only findings; it is not clinical factuality or completeness.

Report selected-minus-CXRMate secondary changes per image and per EHR. Compute
paired exploratory bootstrap intervals with 2,000 draws, seed 0, clustering
on the 80 synthetic EHR cases. For the pooled outcome, average exactly three
fixed images within each EHR BEFORE resampling EHRs. If a required endpoint
is NA, its pooled case outcome is NA rather than a two-image cherry-picked
mean. Images, reports and candidate pairs are NOT independent patients.

This pool and its BioViL values have already been inspected in earlier
diagnostics. Sealing this new selection prevents endpoint-based choice in this
implementation but does NOT turn the old pool into an untouched holdout.
Bootstrap intervals describe exploratory proxy differences, not clinical
efficacy. EHR-edge denominators and no-direct-evidence cases must be reported;
no conclusion of valid full triples or correct faulty-modality attribution is
licensed by this benchmark.

## Runtime, outputs and audit

Reviewed and approved script: `TriCompose-v1.2/slurm/34_full_pool_fresh_scores_debug_p100.sbatch`.
One debug P100, two CPUs, 24GB host RAM, 20-minute wall cap; each owned scorer
process has a 420-second timeout. It performs XRV/CheXbert verification only,
then a metadata-only CPU audit inside the approved allocation. Offline
workspace caches/temp/logs, fsynced reservations, failure retention, no
automatic resume and non-overwriting opaque runs are required.

Completed result root:
`artifacts/protected/tricompose_v1_2/full_pool_report_runs/pool80_fresh_12631194/`.
Output includes `score_table.csv` (960 rows), `model_comparison.csv` (15 rows),
`case_outcomes.csv`, raw receipts, structure metadata, `selection.json`,
`comparison.json`, summary and actual scorer wall-time/memory journals.

The CPU auditor authenticates source/checkpoint/result hashes and project-
private modes, recomputes all receipts, strict choices, raw CSV values/NA,
cached-secondary preservation and case-level bootstrap arithmetic. It does
not reopen report bodies or pixels; structure flags are authenticated outputs,
not independently re-executed judgments. Results/audits remain group-private
below `artifacts/protected/`, modes 2770 for directories and 0660 for files.

Implementation: `src/tricompose_v12/full_pool_report_control.py`,
`benchmarks/prepare_full_pool_report_control.py`,
`benchmarks/run_full_pool_report_control.py`,
`audits/audit_full_pool_report_control.py`, and synthetic-only tests.

## Completed results: mixed secondary outcomes, not general superiority

Of 720 comparisons, 96 alternatives pass the unchanged gate, allowing report
switches on 70/240 images and at least one switch for 63/80 EHR cases. The other
170 baselines stay unresolved. Selected expert counts are CXRMate 170,
LLaVA-Rad 43, CheXagent-2 17 and MAIRA-2 10. No images, EHRs or old winners changed.

Across all three image paths, compared with fixed CXRMate, image/report label
supports rise **554 to 640**, including positive supports 79 to 97 and negative
supports 475 to 543. Proxy opposition falls 114 to 107, and comparable finding
exposures rise 668 to 747 out of 1,920. These improvements use the gate's own
scorers and are partly guaranteed by its construction, not an independent
measure of clinical correctness. EHR/report direct supports rise 4 to 6 out
of 24 repeated direct-fact/path exposures; opposition stays at two. EHR/image
scores stay unchanged because no image is regenerated. Most cases still lack
direct EHR constraints, and location/severity/device correctness is not proven.

| Fixed CXR generator | CXRMate raw BioViL mean | Selected mean | Paired mean change | 95% exploratory case-bootstrap interval |
| --- | ---: | ---: | ---: | --- |
| Sana | 0.6258 | 0.6158 | -0.0101 | [-0.0428, 0.0220] |
| PixArt | 0.6088 | 0.6540 | +0.0452 | [0.0114, 0.0719] |
| RoentGen-v2 | -0.0867 | 0.1354 | +0.2222 | [0.1457, 0.3071] |
| Three fixed images averaged within each EHR | 0.3826 | 0.4684 | +0.0858 | [0.0569, 0.1163] |

The aggregate secondary gain is concentrated on a particularly low historical
RoentGen/CXRMate baseline; Sana's secondary score decreases. Across all fixed
paths, fixed MAIRA-2/LLaVA-Rad/CheXagent-2 have mean cosines 0.5655/0.5918/0.4998,
all above the selector's 0.4684. Fixed MAIRA-2 and CheXagent-2 also have higher
direct EHR/report label supports (nine vs six). Therefore this does **not**
show superiority to the best fixed expert, complete triple consistency,
natural fault localization or online cost savings. Do not use the historical
pool's exploratory intervals as an untouched final-test claim.

The automatic CPU audit recomputed all 240 image receipts, 960 completed
receipts, sealed choices, original secondary-cache preservation, raw CSV/NA,
model totals and EHR-case bootstrap arithmetic. It checked source/checkpoint/
artifact hashes, bounded journals and project-private modes, without report
body or pixel reads. Source result manifest SHA256:
`f29d158088eb352426397b1c16458a2c71c1f9963e63ef5e5f92deaa044b2675`.
Audit:
`artifacts/protected/tricompose_v1_2/full_pool_report_audits/pool80_fresh_12631194/`.
Audit manifest SHA256:
`a8b378dbc0c20faab44f03aaec1cd52bb56bfdfd3f58160cbe53608df5819c0b`.

A separate immutable bilingual handoff (no embedded report text/images) is:
`artifacts/protected/tricompose_v1_2/full_pool_report_summaries/pool80_fresh_12631194/RESULTS_CN_EN.md`.
It includes all fixed experts, negative deltas, raw positive/negative support
counts, direct-EHR strata and limitations; the original sealed output is not
edited. Next: a separately declared bounded real-regeneration comparison
against this static reference, not retuning this gate to observed endpoints.
