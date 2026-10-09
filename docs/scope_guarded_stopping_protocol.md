# Scope-guarded stopping: a cached action-veto experiment

This is a separately frozen **exploratory DEVELOPMENT** policy comparison on
the already inspected 80-EHR/960-triple bank. It is not an untouched test,
clinical error localization, actual regeneration or a deployment authorization.
Keep all EHRs, four-state facts, checkpoints, old scores/winners and model order.
No training, threshold fitting, best-seed/budget selection or BioViL-driven
policy tuning. All computation uses cached metadata inside an existing CPU
Slurm allocation; no source EHR/report body, image, weight, model/API call,
download or Slurm submission.

## One changed mechanism

Retain the original targeted heuristic, original initial Sana/MAIRA-2 path,
report/image order, candidate ranking, cost accounting and proxy-stop test.
Only **veto an unsupported proposed image branch**:

1. A cached basic-invalid-image flag permits exploration of another CXR.
   This is a mechanical flag, not an anatomy/clinical-quality diagnosis.
2. Otherwise require at least one directly evidenced explicit EHR finding
   opposed by the current image's explicit XRV proxy state. Unknown/uncertain,
   weak context, missing comparison and correlated report votes cannot meet
   this condition. Retain evidence IDs and source categories.
3. If that condition is absent, try any remaining reports on the same image.
4. If those reports are exhausted, stop with an explicit unresolved reason:
   no direct EHR anchor, no comparable image evidence, or no directly evidenced
   EHR–image opposition. Return the best eligible **observed** candidate using
   the unchanged historical lexicographic key, not an unseen/full-bank winner.
5. Preserve proxy-satisfied, budget-exhausted and inventory-exhausted statuses
   separately. Even proxy satisfaction never grants clinical acceptance.

An allowed image branch is **proxy-driven exploration in this cache test**,
not proof that the image was erroneous. Absence of an anchor prevents this
policy from making a targeted image-repair claim; it does not prove that the
image is normal/correct or that arbitrary exploratory sampling is useless.
This conservative choice may lose quality and must be tested, not assumed.

The original global No-Finding source-score adjustment stays explicitly legacy;
it is not converted into raw finding labels. Routing uses the original raw
facts. No calibration profile is transferred or hidden new threshold introduced.
Same-image reports remain correlated and cannot independently condemn a CXR.

## Information, lineage and costs

Pass the controller only the current observed candidate's fixed lineage, raw
four-state facts, original routing/ranking inputs and artifact gates. Strip
secondary scores and winner/rank fields. No BioViL endpoint or unseen score
may choose an action, stop or final candidate. EHR facts/hashes stay invariant
across every observation; classifier states/hashes stay invariant while only
the report changes. Reserve two simulated calls per new report and another
two per new image, sharing image work; check the cap before observation.

Use all original caps **4/8/12/20/30**, with one deterministic guarded trial
per EHR/cap (400 new cache trials). Charge the same generator/XRV/CheXbert
invocation units as before. These are not actual GPU calls or runtime savings.
Budget refusal must not inspect a new candidate's scores or create a partial
charge. A returned result remains clinically unqualified, including when the
EHR has no direct comparable finding. Do not drop those 72 EHRs.

## Required controls and readouts

Reuse unchanged fixed, old random-acquisition/scored-final, static rerank and
targeted outcomes. Reuse all 10,000 true score-free final-choice controls and
the old fixed-image `report_only_static` control. The latter is essential:
if guarded stopping merely reproduces same-image reranking, do not call it a
new adaptive advantage. Full-budget old random/static equality is not an
independent replication.

Authenticate the original synthetic cached-state sources, old trial traces and
fixed-image control by manifests/hashes; reproduce the old 3,200 trials exactly.
Choose new guarded IDs before attaching full-bank BioViL/raw-edge readouts.
Report all budgets, actual simulated expenditure, image/report observations,
stop/veto reasons, proxy support/opposition **and coverage**, and cached direct
EHR 8/no-direct-EHR 72 subgroups. Average all random replicates within EHR;
pair same EHRs and preserve missingness. Equal caps need not imply equal work.
Do not pool historical fourteen-head and fresh eight-head profiles.

No p-values, clinical accuracy, confirmed fault/repair success, patient-level
independence or prospective savings claim. A gain on the alternate automatic
readout is descriptive compatibility, not clinical truth. A failed/no-gain
result must remain visible; do not retune after this run or call it independent
confirmation. Further GPU work requires complete-script/resources and approval.

## Implementation and protected output

Worker: `TriCompose-v1.2/tools/scope_guarded_stopping.py`.
Tests: `TriCompose-v1.2/tests/test_scope_guarded_stopping.py`.

Fresh atomic `automatic_replays/scope_guarded_stop_<opaque_run_id>/` contains
guarded outcomes/action decisions, frozen-policy description, case means,
method/subgroup tables, paired comparisons, summary and bilingual report.
Seal all consumed metadata/code/tests/this protocol; recheck hashes before
commit; refuse overwrite. Directories 2770/files 0660 within the CARC project
boundary. No old source/tests/protocol, result or winner is edited. Fixture
and arithmetic checks are engineering evidence, not clinical validation.
