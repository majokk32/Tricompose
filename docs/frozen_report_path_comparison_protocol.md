# Frozen same-image report comparison / 冻结同图报告路径对照

## Scope

This CPU-only descriptive development comparison uses the entire existing
80-synthetic-EHR, 240-CXR, 960-report bank and its complete frozen BioViL-T
endpoints. It adds no generator calls, training, API use or Slurm submission.
Run only within an existing CPU Slurm allocation. Read cached synthetic finding
states, schemas, hashes and lineage; do not open report bodies/image pixels,
raw patient inputs or real targets. Write new atomic, non-overwriting runs under
`artifacts/protected/tricompose_v1_2/report_path_comparisons/` (2770/0660,
CARC project group, including the documented NFS ownership representation).

## Fixed choices, before endpoint values

Each of the 80 EHR anchors has exactly three registered CXR slots (Sana,
PixArt, RoentGen-v2, each seed 0) and four report experts (MAIRA-2,
CXRMate-single, LLaVA-Rad, CheXagent-2). CXRMate-single is image-only;
it must not be renamed CXRMate-ED or described as EHR+CXR report generation.
This bank uses EHR-derived text for CXR generation, not proven direct structured
EHR-to-CXR generation. No EHR, prompt, image, report or original winner changes.

Compare ALL four fixed report experts and a same-image projection of the
unchanged V1.1 `automatic_replay.candidate_key`:

1. Source artifact gate count; only gate-zero candidates are eligible for static choice.
2. Source total proxy contradiction count, ascending.
3. Source EHR direct-support count, descending.
4. Source report–CXR support recall, descending (missing sorts last).
5. Source report structure score, descending (missing sorts last).
6. Cached known runtime, ascending (missing sorts last).
7. Opaque candidate ID as deterministic final tie-breaker.

Existing global No-Finding EHR–report adjustments remain SOURCE metrics;
separate raw-state readouts do not flip unknown to negative. The Sana projection
must reproduce all 80 archived report-only-static choices at maximum budget,
including image/EHR hashes. The new PixArt/RoentGen projections are controls,
not replacements for the original joint image/report winners.

Seal `choices.json` before reading the complete endpoint cosine values.
The pure choice function accepts no endpoint input. Attach authenticated full
960-pair endpoint records only afterward and recheck the exact sealed choices.
No endpoint oracle, post-result threshold changes, new policy or learned router.

## Denominators and missingness

Keep the historical 14-field uncalibrated XRV/CheXbert profile separate from
the fresh eight-enabled-head smoke test. Positive, negative, uncertain and
unknown are distinct; only positive/negative enter explicit comparisons.
For each selected path retain raw counts for EHR–CXR, EHR–report and
CXR–report: known reference, comparable, positive/negative support, opposition.
Show CXR-positive and CXR-negative support separately, plus opposition and
coverage over explicit image-reference facts. Unknown/missing is not agreement.
No denominator means NA; unavailable selected reports leave rates NA.

Aggregate BioViL by averaging the three images WITHIN each EHR and then
averaging EHRs. Display separate CXR-generator controls and direct/no-direct
cached-EHR-evidence strata. Only eight EHRs have directly comparable cached
EHR facts; the other 72 remain present with NA direct-EHR rates. This is not
the previous prompt-conditioning 15/65 split. Fact rates pool explicit fact
comparisons and retain counts, rather than claiming per-patient accuracy.

All four baseline contrasts must use identical EHR/image inventories and
hashes. Paired wins/ties/losses are per EHR, not 240 independent patients.
Any missing required cosine makes that EHR mean NA and the full-cohort mean
or delta NA; separately labeled available-case means never replace the full
denominator. No significance test, clinical acceptance or error attribution.

Simulated calls per image: fixed=4 (CXR+XRV+report+CheXbert), static=10
(CXR+XRV+four reports+four CheXbert). Incremental report/scorer calls for an
already fixed image are 2 versus 8. Shared EHR costs are sunk; secondary
evaluation is not claimed free. No actual GPU runtime savings are measured.

## Outputs and checks

New run contains sealed choices; 1,200 selected-path rows/score-table entries
(five paths × 240 fixed images); method and same-image paired CSVs, per-EHR
paired deltas, selection frequencies, bilingual report, summary and hash-bound
manifest. Check input hashes before/after computation, same-image and fixed-EHR
invariants, archived Sana equivalence, exact score-table recomputation and
private output modes/group. Never infer clinical model rank from selection
frequency, clinical accuracy from cosine, or repair success from label agreement.

This already inspected bank is a DEVELOPMENT control, not held-out validation.
Report negative results as well as positive results; do not tune the source key
to the new endpoint table. An independent prospective/held-out comparison is
still required before making efficacy or clinical-error-localization claims.
