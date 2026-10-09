# Observed-only output acceptance / 仅已观察候选的输出验收

This is a separately frozen **exploratory DEVELOPMENT** post-selection veto,
not a clinically calibrated acceptance gate, new inference controller or
clinical repair benchmark. The 80-EHR/960-triple bank and preceding diagnostic
outcomes were already inspected. No threshold/weight fitting, new cases,
training, endpoint tuning, GPU/model/API call, download or Slurm submission.
Use cached synthetic metadata inside the existing CPU Slurm allocation only.
Never open EHR/report bodies, image/weight bytes or raw real patient inputs.

## Exact policy and observation boundary

Apply the SAME wrapper to all five caps 4/8/12/20/30 for the original
`scope_guarded_targeted` and `report_only_static` trials: 800 new cached
decisions, not new patients or model calls. Preserve each parent acquisition,
action trace, stop reason and costs. This is a final-selection ablation,
not cost saved by changing earlier actions.

For one parent trial:

1. The first observed candidate is the original fixed reference. Only that
   candidate, the parent's proposed winner and its observed candidate list
   may enter the gate. Validate fixed EHR hashes, four-state facts/source
   categories and same-image classifier identity/states. Strip alternate
   scores, old winner/rank flags and non-controller fields before the gate.
2. If the proposed result is unchanged and metadata-eligible, retain it as
   `unchanged_unverified`, NOT a repaired or clinically accepted result.
3. A same-image report change must pass the immutable strict report-headroom
   predicate: preserve old supported image-positive/direct-EHR facts and
   comparisons, no new opposition or lower cached structure, strict finding
   evidence gain. Unknown/uncertain silence cannot count as a repair.
4. For a changed image, select the report-only reference using the unchanged
   historical key among **already observed, eligible reports on the initial
   image**. Compare against it AND the fixed reference using the frozen
   cross-image preservation predicate and require the explicit cached branch
   basis. An unobserved full-bank/static winner cannot serve as this reference.
   This differs from the previous offline oracle-feasibility diagnostic.
5. Passing means `proxy_preserving_*_change_unverified`. It is not independent
   clinical truth, natural-fault localization, regeneration success or GPU
   execution authorization. Baseline image labels can be mistaken; preserving
   them is a conservative surrogate constraint, not guaranteed medical safety.
6. If the proposal fails, retain the metadata-eligible INITIAL fixed result
   with `unresolved_proposal_veto_fixed_retained`. Do not search another
   observed candidate, change a seed/model or pick a better full-bank output.
   An absent proposal is separately unresolved. If the initial result itself
   fails artifact/basic-image metadata, return no eligible output, not a
   falsely accepted invalid baseline. Tampered/cross-case/changed EHR evidence
   is a contract error, not clinical failure or an allowed fallback.

Keep all EHRs, missingness and budgets. Fallback is not certified correctness;
unknown is not negative. Same-image expert votes are not independent evidence.
Use raw fourteen-head cached facts, not global No-Finding-expanded negatives,
fresh eight-head thresholds or newly manufactured device/view/severity labels.
Keep the parent's already incurred model charges on every veto. CPU wrapper
time is separate and must not become a GPU-savings claim.

## Fixed controls and measurement

Retain original fixed, random-acquisition/scored-final, true score-free final,
all-image static, original targeted, report-only static and scope-guarded
outcomes. Add the two gated versions. Average random seeds within EHR before
cohort statistics; report all budgets and direct-EHR 8/no-direct-EHR 72 scopes.
The two gated versions must be compared to expose equivalence with report-only
selection. No assertion that a new wrapper constitutes an adaptive advantage.

Freeze/write/hash all output decisions before attaching full-bank BioViL
and raw edge readouts. Report selected/proposed IDs, exact reference IDs,
fact-ID gains/losses/oppositions, unresolved reasons, all edge coverage,
structure availability, alternate endpoint and simulated charges separately.
No endpoint enters the gate; no best-budget or best-seed selection.
Compare each gated version with its parent at exactly equal charged
acquisition. Compare gated scope versus gated report-only, fixed and full
static with expenditure differences disclosed. No p-values or clinical
accuracy/repair success; missing readouts remain NA, not zero.

## Implementation and safety

New callable/CLI: `TriCompose-v1.2/tools/apply_output_acceptance.py`.
New invented-fixture tests: `TriCompose-v1.2/tests/test_output_acceptance.py`.
Reuse immutable headroom helpers; do not edit consumed modules, old tests,
protocols or outputs. Authenticate the headroom manifest/helper fingerprints,
all original 3,200 replays, 400 guarded replays and 400 report-only controls.
Test observed-only references, no unseen winner/endpoint leakage, veto cost
retention, all fixed-EHR/image invariants, null/invalid fallback, strict
fact-preservation, missingness, deterministic decisions and no model execution.

Publish a fresh atomic protected `output_acceptance/` run containing frozen
policy, sealed selection decisions, 800 gated outcome metadata rows, selected
reference records, unified case/score tables, method/subgroup comparisons,
paired contrasts, summary, bilingual report and manifest. Refuse overwrite;
project-private directories 2770/files 0660. Recheck consumed hashes before
commit. Original winners remain immutable; no image/report/EHR is copied or
regenerated. This layer is wired into cached replay only, not installed into
new GPU generation jobs. Further GPU work requires its complete script and
resource request followed by explicit user approval.
