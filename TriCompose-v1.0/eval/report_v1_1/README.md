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
- existing read-only Qwen2.5-VL deployment.

Not currently found:

- a local CheXbert checkpoint;
- RadGraph checkpoint;
- RadCliQ implementation/checkpoint.

The cross-modal primary table must not be finalized until the report labeler
dependency is satisfied or explicitly reported as unavailable.

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
