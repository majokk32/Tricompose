# Cross-modal repair feasibility / 换报告与换图的代理修复空间

This is a frozen, descriptive DEVELOPMENT diagnostic on the already inspected
80-EHR/960-triple bank. It is not a new router, score, clinical localization
benchmark, oracle clinical accuracy, actual regeneration or execution approval.
No training, threshold fitting, new cases, best-budget choice or BioViL tuning.
Run only inside the existing CPU Slurm allocation, using bounded authenticated
synthetic cached labels/metadata. No EHR/report body, image, weight or real
patient input read; no model/API/download/Slurm submission.

## Fixed references and complete inventory

For every EHR retain the original Sana seed-0 / MAIRA-2 fixed candidate and
the immutable cap-30 fixed-image `report_only_static` winner. Keep the fixed
EHR hashes, four-state facts, source categories and scorer profile. Enumerate
all three same-image report alternatives and all eight other-image/report
alternatives: 880 directed candidate comparisons, not 880 patients. Keep all
80 EHRs and unavailable branches; do not choose easy cases.

Same-image opportunities reuse the existing immutable report headroom
predicate: preserve supported image-positive and direct-EHR finding IDs,
all previously comparable finding IDs, introduce no new opposition, do not
worsen available cached report structure, and strictly gain finding evidence.
Opposition disappearing into unknown/uncertain is not a repair.

## Cross-image predicate, not a changed clinical reference

For each other-image candidate compare it against BOTH fixed and report-only
references. Use raw four-state fact sets, not historical global No-Finding
expansion or aggregate count compensation:

1. Same case, EHR/fact hashes, states and source categories; a different CXR
   identity AND byte hash. No new EHR or invented facts.
2. Artifact and basic-image metadata pass, and both existing report-structure
   values are available with no decrease. This is not an anatomy/factuality
   guarantee. No new claims about uncached temporal/severity/device checks.
3. Preserve all old comparable finding IDs on each of the three edges;
   preserve all direct-EHR supports on both EHR edges and old positive
   image–report supports. A changed image alone cannot earn credit by changing
   its own reference. Baseline image labels are conservative proxy constraints,
   not clinical truth or a known desired target.
4. No new explicit opposition on any edge. Lower total opposition cannot pay
   for a newly conflicting different finding.
5. Strictly improve the fixed EHR–image edge: gain direct support or remove
   an old explicit opposition while preserving its comparison. More negative
   image–report agreement, higher cosine/structure or changed labels alone
   does not qualify. Unknown/uncertain/missing evidence stays unavailable.

Report this predicate separately from the current scope policy's permission
to explore another cached image. That policy requires a basic-invalid-image
flag or an explicit directly evidenced EHR–XRV opposition. A hypothetical
gain without that branch basis is NOT an approved targeted action. Even a
gain with a basis is not a confirmed error or permission for GPU execution.

Retain both-reference failure reasons and gained/lost/comparable/opposition
finding-ID sets for EVERY alternative. Distinguish same-image only, cross-image
only, both and no opportunity **within this finite cache/predicate**, and
no-direct-EHR versus missing-image-evidence strata. No opportunity does not
mean global optimality, an impossible repair or a clinically correct baseline.

## Current selected outputs and secondary endpoints

Audit all 80 cap-30 scope-guarded selections against their fixed reference:
unchanged, same-image report change, or image-and-report change. Apply the
appropriate preservation predicate; cross-image changes must beat both fixed
and report-only references. Record passage/failure, not clinical repair success.
Do not change any original winner or select a new oracle output.

Write/hash the complete label-only plan before attaching full-bank BioViL
readouts. Candidate plans and admissible alternative IDs never receive BioViL.
Attach every baseline/alternative readout afterward, including failed pairs.
For qualifying alternatives only, average ALL alternative gaps within EHR
and then across opportunity-bearing EHRs. Label the result conditional,
not an actual selected-output effect/full-cohort benefit. Missing required
endpoints stay NA; no opportunities gives NA, not zero. Do not choose the
best endpoint/model or treat 880 correlated comparisons as independent patients.
Keep observed guarded-selection deltas separate from opportunity averages.

## Implementation, verification and output

New tool: `TriCompose-v1.2/tools/audit_cross_modal_repair_headroom.py`.
New invented-fixture tests: `TriCompose-v1.2/tests/test_cross_modal_repair_headroom.py`.
Do not edit consumed old workers, source modules, tests, protocols or outputs.
Authenticate all old reference manifests, 3,200 old exact replays, complete
candidate lineage and the 400 guarded trials. Recheck all consumed hashes
before publishing a fresh atomic run under protected `repair_headroom/`.
Directories 2770/files 0660, project group only; refuse overwrite.

Outputs: label plan, all measured pair rows, per-EHR feasibility table,
guarded-selection audit, conditional summaries, bilingual report and manifest.
Engineering checks cover silence, unequal finding IDs with equal counts,
reference drift, both-reference veto, missingness, gate/quality failures,
endpoint isolation, deterministic complete denominators and no execution.
Clinical accuracy/localization/repair success and primary-metric eligibility
remain false/NA. Any prospective GPU experiment still requires its full batch
script/resource request and subsequent explicit approval.
