# Ranking and invariant-switch controls / 排序与固定证据对照

These two controls reuse the already inspected 80-EHR development cohort and
its frozen candidate/evidence caches. They are not independently preregistered
confirmatory tests. Freeze
`TriCompose-v1.2/configs/ranking_switch_control_pool80_v1.json` before execution.
Do not change any old winner, rule, threshold, report/image model or EHR.

## 1. Runtime tie-break ablation

For every existing fixed-image control/baseline trial, reuse EXACTLY the original
observed slots, model-call ledger, action trace, stop reason and fixed EHR.
Remove only the runtime component of the lexicographic key. Keep the original
artifact, opposition, direct-EHR support, report/classifier support and structure
prefix, followed by the same deterministic candidate-ID tie-break.

Do not execute a new policy or reveal an unobserved candidate. If the original
policy stops with `stop_proxy_satisfied`, retain its current candidate rather
than reranking a stopped history. Thus this isolates final ranking on a frozen
observation history, NOT the effect of changing runtime preference on future
actions. A changed winner must share the exact earlier quality prefix with the
old winner. Candidate-ID ordering is deterministic, not clinical evidence or a
new score. Runtime and alternate cosine stay separate readouts; missing runtime
is not zero. Retuning weights/model priorities on this cohort is prohibited.

Attach existing BioViL scores after selection only. A new choice outside the
scored union stays NA; record its hash-bound pair for optional later scoring.
Report paired means on identical available cases, their missing counts, and
full-cohort choice/cost effects separately. No substitute winner or imputation.

## 2. Image switching against invariant EHR evidence

Compare original joint static/targeted outcomes with the original fixed path at
all caps. Hold the cached EHR hashes AND four-state reference vectors fixed.
Distinguish image changes by artifact hashes, not model names. Record unchanged
explicit EHR constraint count, image/report support/opposition/missing transitions,
and all-three support. Unknown/uncertain never become negative or constraints;
weak medication/lab context is not promoted to a hard radiographic label.

Decompose image/report agreement gains into findings with explicit EHR constraints
and those without them, plus positive/negative support. New image/classifier
labels can explain gains in pairwise agreement without an improvement in the
unchanged EHR reference. This is descriptive evidence accounting, not a new gate:
no-direct-EHR is unavailable evidence, not a failed/normal patient, and agreement
alone does not establish natural clinical error or prove repaired generation.

## Execution, claims and outputs

CPU cached computation requires an existing Slurm allocation before protected
reads. Never open raw patient inputs, report bodies or image pixels. No inference,
GPU, training, API, downloads or new submission are included. Write a NEW atomic
protected run containing source hashes, frozen control, ranking/transition rows,
method/subgroup/budget tables, pending endpoint pair inventory and bilingual
interpretation. Keep costs simulated, not actual GPU savings. Confirmatory claims
need a separately frozen cohort and independent evidence; any missing endpoint
GPU scoring requires its own shown complete script/resources and explicit approval.
