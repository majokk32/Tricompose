# Same-image four-expert control with a compatible fresh profile

Status: completed and metadata-audited on 2026-10-03. After explicit approval
of the complete script, job 12630490 ran on debug P100 (two CPUs, 24GB host RAM,
ten-minute cap). Slurm elapsed time was 33 seconds; controller wall time was
31.007 seconds. Any further submission still requires the complete exact
script/resource request followed by explicit approval.

The native n-best run (12629940) produced different texts but identical
CheXbert finding vectors per image. No alternative passed the frozen gate.
The next diagnostic asks whether DIFFERENT existing report experts provide
finding-level alternatives on the SAME two images. It does not expand the
cohort, change EHR/prompts/images, or tune against previous BioViL scores.

## Scope and computation

Keep the original opaque cohort ordinals 0/1 and original Sana seed-0 images.
Use exactly one existing report per image from each frozen expert:
CXRMate-single, MAIRA-2, LLaVA-Rad, CheXagent-2 (eight reports total).
CXRMate-single remains CXR-only, not CXRMate-ED or an EHR+CXR expert.
All reports were already generated in the authenticated original synthetic
bank; do not load or run any report-generating model again. Metadata is copied
to a new private subset; text paths and byte hashes remain those of originals.
The original CXRMate report hashes equal both fresh n-best top-one hashes.

Reuse ONLY the exact image-label bundle from job 12629940 on the identical
image bytes, classifier checkpoint, preprocessing and frozen thresholds.
Its eight enabled heads and four-state semantics are validated with source-bound
image receipts. Legacy uncalibrated 14-head XRV is never substituted. Historical
CheXbert edge scores/states are NOT reused: re-run frozen CheXbert on all eight
reports, then compute all three edges under one fresh compatible receipt profile.

Image-score reuse needs no new classifier call; it does not erase its original
cost. This job newly scores eight report samples (one batch of eight), and
BioViL encodes at most two images/eight texts. Record actual stage startup/I/O
wall time and allocated peak memory. Legacy `model_calls` counts report samples,
not physical batched invocations. All report/CXR/EHR generation costs are
historical sunk costs, not free inference or savings. No new inference-time
targeted regeneration/adaptive-loop experiment is claimed.

## Separately declared cross-expert policy

Retain exact image-positive supports, direct EHR supports and all previously
comparable finding IDs. No new explicit proxy opposition; no silencing an
old conflict into unknown/uncertain; require a strict evidence-set improvement.
Unknown/uncertain image findings cannot become negatives or clinical truth.

Structure contracts are MODEL-SPECIFIC, reusing existing evaluator definitions:
CXRMate requires findings plus impression; the other three require findings
only. A findings-only expert is not automatically worse because it lacks an
impression heading. All candidates must pass their own section contract and
be nonempty. Common genericity/unsupported-temporal flags and sentence/four-gram
repetition may not worsen relative to the baseline. These are proxies, not a
clinical factuality score.

This is a new cross-expert control, NOT a relaxation of the published n-best
predicate, whose original source, outputs and abstentions stay unchanged.
The predeclared expert priority is CXRMate, MAIRA-2, LLaVA-Rad, CheXagent-2.
Choose the first eligible nonduplicate alternative; if none passes, retain
CXRMate unresolved. No BioViL values, source winner flags or per-case endpoint
oracle are available to this decision. Seal the complete selection BEFORE
new BioViL evaluation, and report every expert and any negative selected delta.

Two EHRs here have no direct radiographic constraints; their EHR-edge rates
remain NA. The six beam candidates, eight expert reports, and old 80-EHR bank
are not independent patients or untouched evaluation sets. This small control
cannot establish full three-modal fidelity, natural error localization,
clinical repair, diversity preservation or GPU savings. Cached reports on one
image are correlated; consensus is not proof the image is wrong.

## Runtime and outputs

- CPU staging consumes synthetic metadata, hashes and cached image labels only;
  no report-body/image-pixel/model inspection.
- The approved GPU controller internally consumes synthetic report text for
  existing structure checks, CheXbert and secondary BioViL. Public output has
  sanitized status, wall time and hashes only.
- Preserve fsynced reservations, bounded owned worker groups, immutable private
  outputs, failure retention and no automatic retries/overwrite.
- Save eight-row `score_table.csv`, four-path `model_comparison.csv`, raw receipts,
  model-specific structure metadata, selection, secondary measurements and costs.
- All results/plans are under `artifacts/protected/tricompose_v1_2/`, project
  group only. No real patient inputs/targets or external API are used.

Implementation: `report_expert_control.py`,
`benchmarks/prepare_report_expert_control.py`,
`benchmarks/run_report_expert_control.py`, and invented-state tests.

## Completed diagnostic, not a repair success

All eight secondary endpoints were available. The fresh CheXbert scored eight
report samples; BioViL-T encoded two images/eight texts. Peak allocated VRAM
was 0.448 GiB for CheXbert and 0.579 GiB for BioViL. No image/report generation
or new XRV calls occurred. Historical generation/classifier costs remain sunk
costs, not zero-cost generation or claimed compute savings.

| Frozen report expert | Comparable / image reference | Negative supports | Positive supports | Explicit proxy oppositions | Mean raw BioViL-T cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| CXRMate-single | 6 / 16 | 6 | 0 | 0 | 0.5460 |
| MAIRA-2 | 1 / 16 | 0 | 0 | 1 | 0.0747 |
| LLaVA-Rad | 8 / 16 | 6 | 0 | 2 | 0.1709 |
| CheXagent-2 | 6 / 16 | 3 | 0 | 3 | 0.5009 |

The denominator 16 is eight enabled classifier findings on each of two fixed
images. These are frozen proxy reference states, NOT real-target facts or
clinical truth; all explicit image references happen to be negative here.
Each expert passed its existing model-specific section contract on both
reports; unsupported temporal-language flags were zero for all eight reports.
Neither section success nor the raw cosine establishes clinical accuracy.

Zero of six cross-expert alternatives passed the preregistered evidence-set
gate. MAIRA-2 lost comparable finding IDs; other alternatives introduced proxy
oppositions and some also lost comparable IDs. Both CXRMate baselines were
retained **unresolved**, not declared repaired or clinically optimal. The
selected-minus-baseline BioViL delta is zero because no report was replaced.
Higher coverage alone would not be a valid replacement rule: LLaVA-Rad adds
comparisons here, but also introduces opposition.

Both fixed EHRs still have zero directly comparable radiographic facts.
EHR-CXR and EHR-report rates stay NA, not zero and not perfect agreement. This
two-case test cannot establish general report-model superiority or successful
three-modal composition. Do not tune thresholds, relax preservation, insert
EHR facts or pick easier cases to turn this abstention into an apparent gain.

Immutable output:
`artifacts/protected/tricompose_v1_2/report_expert_runs/experts2_12630490/`.
Per-candidate values: `score_table.csv`; aggregated paths: `model_comparison.csv`;
decisions: `selection.json`; secondary changes: `comparison.json`.

Independent CPU metadata audit recomputed all eight receipts, raw edge/NA CSV
cells, model aggregates, choices and endpoint arithmetic, and checked source,
checkpoint, artifact hashes, bounds/journals and project-private permissions.
It did not open report bodies or image pixels; authenticated structure flags
were checked, not re-executed. Audit output:
`artifacts/protected/tricompose_v1_2/report_expert_audits/experts2_result_audit_12630490/`.
Audit manifest SHA256:
`0fd5af9e577154b682bc2614845bab917a3300e873164843b0e88043ab9aff86`.
Auditor: `TriCompose-v1.2/audits/audit_report_expert_control.py`.
