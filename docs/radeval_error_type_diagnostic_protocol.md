# RadEvalExpert error-type diagnostic: derived output only

Status: fixed post-hoc diagnostic, following the completed BioViL-T benchmark.
This is not a replacement primary endpoint, independent confirmation, a new
verifier, or authorization for another inference/generation job.

Consume only the audited `paired_score_table.json` and `evaluation.json` under
`artifacts/protected/tricompose_v1_2/radeval_image_benchmark_runs/biovil_cpu_12714150_001/`.
No source CSV/report/EHR/image, native embedding, model, checkpoint, network
service or new Slurm submission is needed. Run the lightweight numerical replay
inside existing CPU allocation 12714150. Refuse to overwrite the diagnostic run.

Retain all 624 attempted pairs, 492 unavailable-score rows, all four original
metrics and all seven released **significant-error categories**:
false prediction, omission, incorrect location, incorrect severity, unsupported
comparison, omitted change, and inarticulate report. These are author annotation
types, not new disease labels or hand-authored clinical rules. Counts compare
a report to the released reference, not newly adjudicated image truth.

For each category/metric report paired/error-positive counts, pooled score vs
negative-count correlation, fixed-anchor strict-pair ranking (score ties = 0.5),
and the category's contribution to original metric-max-choice error burden
versus uniform random choice. All three candidates must be complete for a
choice anchor. Keep missing counts/scores null; zero errors in every report
means no discriminating pairs, not perfect accuracy. Include all categories,
including sparse/constant ones; do not select a favorable category or section.

Use 1,000 whole-source-patient-cluster bootstrap resamples, seed 0, for the
mean selection-error difference. These intervals are exploratory and not
multiplicity-adjusted. Sparse strict comparisons cannot qualify a repair rule.
Verify deterministic reverse-order replay and that category mean components
sum to the original total-error selection results on the identical complete
cohort. Seal code/input hashes and protected output permissions.

This diagnostic can identify *candidate weaknesses* of a global metric, not
their causal mechanism. It cannot establish negation/laterality sensitivity,
clinical error localization or image-grounded factual truth without additional
independent evidence. Do not tune weights, flip score direction, replace EHR,
change historic winners or regenerate modalities from these outcomes.
