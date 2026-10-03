# TriCompose V1.1 report evaluation

This directory implements the report branch of the project evaluation matrix.
It is isolated from the teammate-owned EHR and CXR evaluation directories.

Despite its legacy location under `TriCompose-v1.0/eval/`, this is the V1.1
evaluation implementation. It remains in place to avoid breaking protected run
scripts and historical paths. New version boundaries and completion gates are
defined in [`../../../docs/version_roadmap.md`](../../../docs/version_roadmap.md).

## Scope

The evaluation has three separately reported components:

1. Report unimodal quality.
2. EHR-report consistency.
3. Report-CXR consistency.

The current 80-case cohort is fully synthetic and has no real reference
report. Therefore BLEU, ROUGE-L, METEOR, BERTScore, CheXbert-F1,
RadGraph-F1, and RadCliQ are not valid unimodal factuality metrics for this
cohort. They remain reserved for a separately declared real-anchor run.

## Report unimodal evaluation

`evaluate_report_structure.py` computes, without model inference:

- empty report rate;
- findings and impression section presence/completeness;
- model-specific section-contract pass rate;
- conservative generic-report rate;
- unsupported prior/no-change language rate;
- repeated-sentence and repeated-4-gram statistics;
- exact normalized-template diversity;
- length and sentence-count distributions.

The evaluator reads protected synthetic reports but never writes their text to
the evaluation bundle. Results are written atomically below
`artifacts/protected/` and existing run directories are never overwritten.

## Cross-modal evaluation contract

Both cross-modal edges use the same 14-finding order:

```text
atelectasis, cardiomegaly, consolidation, edema,
enlarged_cardiomediastinum, fracture, lung_lesion, lung_opacity,
pleural_effusion, pleural_other, pneumonia, pneumothorax,
support_devices, no_finding
```

Every state is one of:

```text
positive | negative | uncertain | unknown
```

`unknown` is excluded from agreement/contradiction denominators and is never
converted into a negative label.

The planned primary evidence is:

- report findings: frozen CheXbert/CheXpert-style labeler;
- CXR findings: frozen CXR classifier with recorded calibration thresholds;
- EHR findings: direct evidence-grounded facts only;
- EHR-report: support, explicit contradiction, coverage, macro/micro F1 and
  Cohen's kappa on comparable states;
- report-CXR: label agreement, per-finding precision/recall, hallucinated and
  omitted finding rates.

Qwen2.5-VL and BioViL-T are secondary evidence. Raw cosine or VLM scores must
not be presented as calibrated clinical probabilities. Report-to-CXR cycle
consistency remains optional and separate.

## Current dependency audit

Available locally:

- frozen XRV DenseNet weight;
- frozen BioViL-T checkpoint;
- frozen CheXbert checkpoint and its local BERT tokenizer/model assets;
- existing read-only Qwen2.5-VL deployment.

Not currently found:

- RadGraph checkpoint;
- RadCliQ implementation/checkpoint.

Audit on 2026-09-22: XRV evidence covers 240/240 historical CXRs and CheXbert
960/960 historical reports. Neither BioViL-T nor Qwen evidence has been merged.
Report-labeler availability does not by itself satisfy the calibration gate.

## Unified candidate table and non-agent baselines

`build_unified_score_table.py` is the legacy-named candidate-registry builder;
its output is not a score table until frozen evidence has been merged. It
materializes one row for every exact EHR-CXR-report lineage. The current
registered grid is 80 cases x 3 CXR models x 4 report models = 960 rows.
Pending evidence is `null`, not zero.

`aggregate_crossmodal.py` joins hash-bound frozen labels and computes all three
finding edges:

- EHR-CXR;
- EHR-report;
- report-CXR.

It also records report-expert disagreement per CXR and CXR-candidate
disagreement per EHR. `finalize_unified_score_table.py` merges those records
back into the skeleton under the explicit policy in
`static_score_policy_v1_1.json`. The static score uses the three finding edges
and reference-free report structure only. Qwen2.5-VL, BioViL, and generation
cost stay in separate columns and are not silently mixed into the total.

If an EHR contains no direct comparable radiographic fact, its EHR-CXR and
EHR-report edges are marked `not_applicable_no_comparable_ehr_facts`. Such an
edge has no numeric score and remaining weights are renormalized. This is not
the same as pending evidence.

`run_selection_baselines.py` implements three deterministic, training-free
baselines:

1. all 12 fixed model paths, reported separately;
2. reproducible random selection;
3. exhaustive static reranking of all 12 candidates per case.

The same-cohort best fixed path is descriptive only. A paper claim requires a
validation split for choosing a fixed path and a disjoint test split for final
comparison. Exhaustive static reranking costs 3 CXR plus 12 report calls per
case; it is not an Agent and does not regenerate a targeted modality.

With default 0.5 XRV thresholds, results are marked diagnostic rather than
paper-primary. A protected threshold bundle calibrated on a matched real
validation set is required for paper-primary selection.

## Edge-specific EHR evaluation

`aggregate_ehr_edges.py` supersedes use of one symmetric CheXpert-14 contract
for all three edges. It leaves the CXR-report evaluator unchanged and defines:

- EHR-CXR direct observable consistency on explicit EHR findings;
- disease-specific results for edema, pleural effusion, cardiomegaly,
  pneumonia, pneumothorax, and support devices;
- diagnostic AUROC/AUPRC/Brier/ECE only where an explicit binary EHR reference
  and a CXR classifier probability both exist;
- EHR-report direct finding agreement and hard contradiction;
- diagnosis/medication/lab/vital source coverage without treating an omitted
  EHR fact as a report error;
- CHF and loop-diuretic rules as weak support/incompatibility evidence, never
  hard labels;
- temporal and LLM scores as explicitly unavailable or secondary when they
  have not been run.

This edge-specific output does not produce a mixed total score or candidate
ranking. It is intended to support stratified reporting before any router is
implemented.

## Edge-specific static selection

`select_edge_specific_candidates.py` joins the revised EHR-edge evidence with
the valid report-CXR edge and the existing unimodal validity registry. It
creates the complete 960-row score table and selects one of 12 triples per
case with the explicit policy in
`lexicographic_selection_policy_v1_1.json`.

The selector is lexicographic: validity gates, total hard contradictions,
direct support on the two EHR edges, report-CXR support, report structure
quality, known runtime, and finally a deterministic candidate ID. Keeping
EHR-edge support separate prevents the denser report-CXR label vector from
implicitly dominating sparse EHR evidence. Weak EHR priors never become hard
contradictions. A 0-100 clinical-balance score is included for diagnostic
reporting but is explicitly not used for selection.

The output compares all 12 fixed paths, one predeclared operational fixed
path, the same-cohort descriptive best fixed path, and exhaustive static
reranking. This is still a high-cost non-agent baseline and remains diagnostic
until the frozen CXR classifier thresholds are calibrated on a disjoint
matched validation set.

## September evaluation corrections and preparation

- XRV `DenseNet.forward()` applies `sigmoid` then `op_norm` using published
  operating points. Our legacy `finding_probabilities` JSON field therefore
  stores normalized classifier scores, not calibrated disease probabilities.
  The field is retained for compatibility; new bundles explicitly declare
  `score_semantics: xrv_op_norm_0_1` and `probability_semantics: false`.
- Brier/ECE are unavailable for these scores. AUROC and average precision can
  use ranking scores where both reference classes exist. Average precision now
  groups tied scores; it no longer ranks tied positives first. EHR weak labels
  are still not image ground truth, regardless of score calibration.
- `xrv_calibration.py` implements deterministic per-finding operating-point
  selection with explicit unknown exclusion, minimum positive/negative counts,
  patient-disjoint split checks, checkpoint/preprocessing/mapping fingerprints,
  and disabled-finding masks. This fits thresholds, not generator weights or
  probability calibration. It does not establish independent clinical validity.
- `fit_xrv_thresholds.py` consumes an already prepared protected calibration
  bundle only inside Slurm. No matched-real bundle has been prepared or fitted
  by the September work. Its schema is
  `tricompose-xrv-calibration-input-v1`; required top-level fields are
  `provenance`, `heldout_group_hashes`, and `records`.
  Each record has protected `sample_sha256`, `patient_group_sha256`, and full
  14-key `scores` / `reference_states` maps. Unavailable scores are null.
  Provenance fields and reference-quality categories are validated by
  `validate_provenance()`. Use protected keyed group hashes, not raw patient IDs.
- Threshold bundles remain `primary_metric_eligible: false` pending independent
  calibration-set/reference-quality and hard-negative review. Legacy bundles
  cannot claim paper eligibility solely through a status string. Unsupported
  or underpowered findings stay unknown rather than defaulting to negative.
- `paired_case_bootstrap.py` provides paired, patient-group resampling of
  case-level outcomes for future held-out comparisons. It rejects duplicate
  case rows, preserves missing-pair counts, and does not interpret 960
  candidate rows as 960 independent patients. No paper CI has been produced.

The historical 240-CXR scalar metrics were recomputed into a new companion
bundle, without model inference or overwriting old evidence:

```text
artifacts/protected/tricompose_v1_1/evaluation/metric_corrections/
  xrv_score_semantics_20260922_001/
```

This is an arithmetic/interpretation correction, not a claim of improved
generation or calibrated performance. Existing selection outputs are unchanged.

## Explicit two-case, two-seed evaluation (September 22)

The latest completed generation comprises 2 fixed EHRs, 12 CXR candidates
(3 models × 2 seeds per case), and 48 reports (4 experts per image). These are
new September outputs, not relabeled August artifacts.

`candidate_grid.py` validates every case/model/seed/report combination and
rejects duplicate combinations, missing slots, and cross-case/seed parent reuse.
`build_unified_score_table.py --cohort-contract smoke48_grid.json` explicitly
selects this cohort; without a contract the historical 80-case/single-seed
requirement remains. Finalization and edge-specific selection preserve the
declared grid rather than enforcing 960 rows unconditionally.

The following protected CPU outputs are complete:

```text
artifacts/protected/tricompose_v1_1/evaluation/smoke48/
  validity_12256939_001/cxr_basic_validity.json
  structure_12256939_001/report_unimodal_details.json
  structure_12256939_001/report_unimodal_summary.md
  registry_12256939_001/score_table.jsonl
```

The 48-row registry is NOT yet clinically scored and is NOT selection-ready.
The bounded image audit checks readable/nonconstant PNGs only, not anatomy,
crop quality, realism, or conditioning. Report structure uses the existing
reference-free evaluator and each model's section contract. Short text and
regex-flagged temporal language are review flags, not clinical adjudications.

`slurm/30_smoke48_score_select_v100.sbatch` was explicitly approved and
submitted as job **12257177**, which completed with exit 0 in **67 seconds**.
It requested 1 V100, 4 CPUs, 24 GiB RAM and a 15-minute limit. Frozen XRV
completed 12 CXRs, CheXbert 48 reports, and secondary BioViL-T 48 pairs.
Aggregation and selection completed without regeneration. All 48 scored rows
and two selections were verified, including artifact hashes and protected modes.
Outputs are under `evaluation/smoke48/scoring_12257177/`:

```text
xrv/                  chexbert/             biovil/
crossmodal/           ehr_edges/            scored/
selection/candidate_score_table.csv
selection/selected_triples.jsonl
selection/selection_results.json
selection/selection_summary.md
```

Selection uses `lexicographic_selection_policy_smoke48.json`. The operational
fixed baseline is Sana + seed 0 + MAIRA-2, declared before clinical scoring.
All 24 model/seed paths are reported separately; a fixed path never chooses
its seed by score. Exhaustive selection uses 6 CXR + 24 report calls per case,
not the historical 3 + 12. These counts exclude EHR generation and evaluator
calls. Selected-path runtime is not the cost of generating the full bank.
All-invalid cases are rejected rather than exported as acceptable winners.

Both current cases have context-only EHR conditioning. EHR direct-comparison
scores may be NA; missing facts never become negatives or perfect agreement.
BioViL raw cosine is retained as secondary evidence and excluded from ranking.
XRV labels are uncalibrated operating-point-normalized classifier outputs, not
probabilities or image ground truth. Apparent improvements on the selection
scorer are diagnostic and are not independent validation. Two EHR cases do
not support a cohort-level clinical superiority claim or a paper confidence
interval. No V1.2 router/repair implementation was added.

### Fixed-versus-selected review

The protected review is at:

```text
artifacts/protected/tricompose_v1_1/evaluation/smoke48/
  review_12257177_001/comparison.md
  review_12257177_001/candidate_review.csv
  review_12257177_001/images/
```

This post-hoc synthetic-output review leaves the original selection immutable.
It decomposes support into positive agreement, negative agreement, explicit
contradiction and classifier-positive omission, and includes same-image
different-report controls. On the two selected triples, positive support is
1 and negative support is 9; the declared fixed path has positive support 2
and negative support 0. Both have zero explicit contradiction signals. Thus
the earlier 8.3%→41.7% coverage increase is predominantly a change in negative
coverage under an uncalibrated labeler, not evidence of clinical improvement.
The report also records the BioViL disagreement and keeps it secondary.

### Positive-focused policy ablation

An offline policy ablation (no GPU and no new generation) is at:

```text
artifacts/protected/tricompose_v1_1/evaluation/smoke48/
  positive_focused_12257177_001/positive_focused_summary.md
```

It orders candidates by validity, contradiction count, positive CXR-report
support, structure quality, runtime and ID; negative agreement is retained for
reporting but is not a reward. On the two-case smoke, fixed path positive
support is 2/24 (mean positive recall 0.0833), while this ablation selects
4/24 (mean positive recall 0.1667), with zero contradictions in both. This is
an exploratory scorer-sensitivity result on two EHRs, not a validated clinical
improvement. The original selector remains unchanged.

The diagnostic selection picked RoentGen-v2 seed 1 + CheXagent-2 for case_000,
and Sana seed 1 + CheXagent-2 for case_003. Fixed-path report-CXR support/coverage
was 2/24 (8.33%); selected-path support/coverage was 10/24 (41.67%). Both had
zero explicit contradiction signals. This is coverage under the same
uncalibrated selector, not clinical accuracy or an independent improvement claim;
14/24 classifier-reference facts remained non-comparable after selection.
Both direct EHR edges correctly remain NA for all 48 candidate rows.
