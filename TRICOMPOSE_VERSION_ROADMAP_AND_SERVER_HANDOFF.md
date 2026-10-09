# TriCompose Version Roadmap and Server-Agent Handoff

**Document purpose:** give the server-side coding agent a precise, versioned implementation plan.  
**Project root on CARC:** `/project2/ruishanl_1185/inference_3mod`  
**Last updated:** 2026-09-21

---

## 1. Project goal

TriCompose generates a complete synthetic patient triple using frozen existing models:

```text
Synthetic structured EHR
    -> EHR-derived radiology facts/prompt
    -> multiple synthetic CXR candidates
    -> multiple synthetic report candidates
    -> cross-modal verification and selection/repair
    -> one final (EHR, CXR, Report) triple
```

The contribution is **not** merely connecting several models. The intended claim is:

> Training-free inference-time composition and cross-modal verification can produce more consistent synthetic EHR–CXR–Report triples than any fixed generation path, while controlling inference cost.

All generation models remain frozen. A vague LLM agent is not required. The main method should be deterministic, auditable, and reproducible.

---

## 2. Non-negotiable scientific rules

1. Once a synthetic EHR is selected for a case, keep it fixed. Do not replace a difficult EHR merely because downstream generation fails.
2. Preserve complete lineage: EHR, facts, prompt, model revision, seed, parent IDs, hashes, runtime, and cost.
3. Use four finding states:

   ```text
   positive / negative / uncertain / unknown
   ```

4. `unknown` is neither negative nor agreement.
5. Record EHR evidence strength:

   ```text
   direct / compatible / risk_only
   ```

6. Only explicit `direct` positive/negative evidence can create a strong EHR contradiction. For example, heart failure or diuretic use does not logically guarantee current pulmonary edema.
7. Separate **real-anchor evaluation** from **fully synthetic evaluation**.
8. Do not use reference-based metrics such as GREEN, standard RadGraph-F1, RadCliQ, RadFact, or VLScore as reference-free selection scores when no real report is available.
9. Separate case-level selection metrics from cohort-level metrics. FID/KID/PRDC cannot rank a single image.
10. Keep selection evaluators separate from final held-out evaluators whenever possible to reduce evaluator overfitting.
11. Never overwrite old candidate banks or results. Every run must have a new run ID and immutable manifest.

---

## 3. Version overview

| Version | Purpose | Main output | Status |
|---|---|---|---|
| V1.0 | Frozen-model generation plus static best-of-N | One selected triple from an existing candidate bank | Completed as an engineering smoke test |
| V1.1 | Evaluation foundation and prompt de-collapse | Six-dimensional, finding-level evaluation with explainable JSON | Next implementation target |
| V1.2 | Error localization and targeted regeneration | Deterministic self-correction under a fixed call budget | Implement only after V1.1 is validated |
| V1.3 | Paper-grade evaluation and scale-up | Independent evaluation, statistics, human review, full baselines | Final experimental version |
| V2.0, optional | More general budgeted inference-time composition | Learned or optimized policy and broader generalization | Only if V1.2 shows clear benefit |

The version numbers describe research capability, not only code changes.

Earlier planning notes called the entire error-localization idea “V1.1.” This roadmap deliberately splits that work into **V1.1 evaluation foundation** and **V1.2 targeted repair**, because repair cannot be validated before the evidence contract and scorers are trustworthy. Do not rename or overwrite historical artifacts that already contain `v1_1`; use explicit new evaluator/repair version fields to avoid ambiguity.

---

## 4. V1.0 — Static best-of-N composition

### 4.1 What V1.0 accomplished

V1.0 established the complete frozen generation and lineage pipeline:

```text
Synthetic EHR
  -> deterministic radiology prompt
  -> RoentGen-v2 / Sana / PixArt configurations from CheXGenBench
  -> MAIRA-2 / CXRMate-single / LLaVA-Rad / CheXagent-2
  -> frozen scorers and deterministic rules
  -> static ranking
  -> final atomic triple export
```

The original smoke bank contained:

- 2 fixed synthetic EHRs;
- 3 CXR models × 2 seeds = 12 CXR candidates;
- 4 report models per CXR = 48 report/triple candidates;
- a predetermined fixed baseline: RoentGen-v2 seed 0 → MAIRA-2;
- a scored-best selection: Sana seed 1 → CXRMate-single for the retained case.

V1.0 also implemented hash-bound lineage and prevented partial or cross-case export.

Naming and dependency boundaries:

- CheXGenBench is the evaluation/generation framework; Sana and PixArt are the actual CXR generator configurations.
- All current report generators are CXR-conditioned in this V1 bank.
- `CXRMate-single` is CXR-only; it is not `CXRMate-ED` and does not provide an independent EHR+CXR report path.
- The CXR adapters receive EHR-derived radiology text, not native structured EHR.

### 4.2 What V1.0 did **not** prove

- The two EHRs collapsed to one CHF-only prompt, so there was only one unique downstream case.
- XRV, BioViL-T, Qwen scalar, and lexical agreement were uncalibrated engineering evidence.
- Qwen's positional 14-character finding output was incomplete for all reports, so finding-level localization was unavailable.
- Nine of 24 combinations for each EHR lacked complete EHR–report evidence; V1.1 must expose coverage so incomplete evidence does not create hidden selection bias.
- The selected report still hallucinated multiple views.
- Therefore V1.0 proved pipeline mechanics, not cohort-level superiority or clinical calibration.

### 4.3 Why V1.0 should remain frozen

V1.0 is the historical baseline. Do not silently rewrite its scores or outputs. New evaluation code may re-evaluate the immutable V1.0 bank under a new version, but results must be stored separately with the evaluator version recorded.

---

## 5. V1.1 — Evaluation foundation and prompt de-collapse

### 5.1 Motivation

Targeted repair cannot be trusted until the system can answer three questions reliably:

1. What fact is expressed by each modality?
2. Is a pair supported, contradicted, or simply not observable?
3. Is a score a case-level selection score or a cohort-level evaluation score?

V1.1 therefore builds the measurement layer first. It is intentionally not yet a dynamic agent.

### 5.2 V1.1 scope

#### A. Repair the EHR-to-prompt bottleneck

- Expand the deterministic mapping from EHR to radiology-relevant facts.
- Preserve explicit diagnoses, devices, image-relevant medications/context, and clinical indication when supported.
- Do not invent laterality, location, severity, or a negative finding.
- Keep both the canonical fact representation and model-specific prompt text.
- Add prompt/fact fingerprints and duplicate detection.
- Select a fixed cohort before viewing generated quality; require distinct prompt hashes and meaningful finding combinations.

#### B. Replace positional finding strings with named evidence

Use a versioned structure similar to:

```json
{
  "ontology_version": "tricompose_findings_v1.1",
  "findings": {
    "cardiomegaly": {
      "state": "positive",
      "evidence_strength": "direct",
      "source": "diagnosis",
      "provenance": ["original code or source span"],
      "laterality": "unknown",
      "severity": "unknown",
      "time": "current"
    }
  }
}
```

Initial ontology: CheXpert-style findings plus support devices. Laterality, severity, location, and temporal status are optional attributes and are scored only when observed.

#### C. Implement the six-dimensional evaluation vector

```text
1. EHR quality
2. CXR quality
3. Report quality
4. EHR–CXR consistency
5. CXR–Report consistency
6. EHR–Report consistency
```

Minimum V1.1 metrics:

| Dimension | V1.1 metric |
|---|---|
| EHR quality | schema validity, impossible-value/clinical-rule violations, fact/provenance completeness |
| CXR quality | resolution/blank/crop/non-degeneracy gates; retain XRV probabilities |
| Report quality | empty/generic/repetition checks; unsupported view/prior/temporal/measurement flags |
| EHR–CXR | direct-known-fact support, explicit contradiction, coverage |
| CXR–Report | calibrated XRV vs named report findings: support, contradiction, coverage; BioViL-T secondary |
| EHR–Report | provenance-aware direct-fact support, contradiction, coverage |

Each edge must save per-finding evidence, not only a scalar.

#### D. Add basic cohort evaluation

- EHR: demographic/clinical marginals; JSD for categorical and Wasserstein distance for continuous variables.
- CXR: RadDINO-FID/KID on a cohort, if the public checkpoint is available and stable.
- Report with a real reference: CheXbert-F1 and RadGraph-F1.
- Lexical metrics may be reported for comparability but are never the main clinical result.

### 5.3 V1.1 comparison

Run both methods on the same fixed cases and candidate bank:

1. predetermined fixed path;
2. static best-of-N using the V1.1 explainable selection stack.

Do not introduce targeted regeneration yet. This isolates whether the new evaluation can rank existing candidates sensibly.

### 5.4 V1.1 required artifacts

For every modality and edge, save:

```text
metric_version
ontology_version
evaluator_name and checkpoint/revision
raw probabilities or extracted facts
thresholds and calibration source
support
contradiction
coverage
per-finding decision and provenance
runtime and GPU information
```

Produce:

```text
evaluations/v1_1/<run_id>/
  manifest.json
  per_case/<case_id>.json
  cohort_summary.json
  report.md
```

The precise path may follow the existing repository convention, but it must be versioned and immutable.

### 5.5 V1.1 done criteria

V1.1 is complete only when:

- distinct source EHRs no longer silently collapse without being flagged;
- named finding output has no positional ambiguity;
- `unknown` is preserved through all three modalities;
- all three cross-modal edges return support, contradiction, and coverage;
- every scalar can be traced to per-finding evidence;
- the fixed and static-reranking baselines can be compared on the same cohort;
- unit tests cover unknown handling, direct vs weak evidence, hash lineage, and duplicate detection.

### 5.6 Explicit V1.1 non-goals

- no LLM router;
- no targeted regeneration;
- no claim that consistency equals clinical correctness;
- no full TSTR/privacy/radiologist study yet;
- no tuning thresholds on the final test set.

---

## 6. V1.2 — Error localization and targeted regeneration

### 6.1 Motivation

Static reranking can select only among candidates already generated. V1.2 tests the smaller methodological idea proposed by the advisor:

> Use redundant generation paths to localize whether the CXR or report is likely wrong, then regenerate only that modality.

This must remain a deterministic, training-free self-correction framework in the first implementation.

### 6.2 Evidence-independence prerequisite

The current V1 bank does **not** contain two independent report paths: MAIRA-2, CXRMate-single, LLaVA-Rad, and CheXagent-2 all consume the CXR, and `CXRMate-single` does not consume the EHR. Consequently, agreement among current reports can reflect shared dependence on the same CXR and cannot by itself prove that the CXR is wrong.

Before making a strong modality-localization claim, do one of the following:

1. integrate and validate a genuine EHR+CXR→Report path such as CXRMate-ED; or
2. explicitly call localization a heuristic and establish its accuracy using controlled error injection and independent human/held-out evaluation.

Do not represent `CXRMate-single` as `CXRMate-ED` in code, tables, or the paper.

### 6.3 Calibration and controlled corruption

Before enabling repair, create a held-out calibration bank containing:

- true matched triples;
- random patient swaps;
- same-disease hard-negative swaps;
- wrong-CXR replacement;
- wrong-report replacement;
- controlled presence/absence, negation, laterality, severity, device, and temporal perturbations.

For every edge evaluator, report AUROC/AUPRC, pairwise ranking accuracy, error-type sensitivity, coverage, and calibration. Freeze thresholds after validation.

### 6.4 Deterministic localization rules

Use EHR evidence, CXR evidence, and at least two report paths where available.

```text
EHR + multiple reports agree, CXR disagrees
    -> regenerate_cxr

EHR + CXR agree, one report disagrees
    -> regenerate_report or switch_report_model

Reports contradict one another
    -> keep the report better supported jointly by EHR and CXR;
       otherwise verify_more

Evidence is insufficient or all modalities conflict
    -> verify_more, then reject if the budget is exhausted
```

Allowed actions:

```text
select
regenerate_cxr
regenerate_report
switch_report_model
verify_more
reject
```

The reason for every action must include the findings and edges that triggered it.

### 6.5 Budgeted targeted repair

- Keep the EHR fixed.
- Start with a small, predefined set of CXR/report calls.
- Only call another model/seed when the localization rule identifies a repairable conflict.
- Enforce a maximum CXR budget, report budget, and total GPU-time/call budget.
- Stop when all hard gates pass and no strong contradiction remains, or reject at budget exhaustion.

Budget values must be configuration, not hard-coded into scoring logic.

### 6.6 V1.2 baselines

1. Fixed single path.
2. Random candidate/path selection.
3. Static best-of-N reranking.
4. Generate-all exhaustive selection/oracle analysis.
5. V1.2 localization + targeted repair.

The central comparison is quality versus compute, not consistency alone.

### 6.7 V1.2 main metrics

- error-localization accuracy;
- EHR–CXR, CXR–Report, and EHR–Report support/contradiction/coverage;
- any-strong-contradiction rate;
- minimum-edge consistency;
- repair success rate;
- false-repair rate;
- reject rate;
- average model calls and GPU time;
- CXR/report quality and diversity before versus after repair.

### 6.8 V1.2 done criteria

- controlled corruptions can be localized above simple/random baselines;
- repair decisions are reproducible from saved evidence;
- targeted repair reduces contradictions without materially degrading single-modality quality/diversity;
- its quality–compute point is meaningfully better than fixed paths;
- comparison with static reranking and exhaustive generation is complete.

If static reranking performs equally well at similar or lower cost, report that result and do not overclaim the repair framework.

---

## 7. V1.3 — Paper-grade evaluation and scale-up

### 7.1 Purpose

V1.3 turns a working method into defensible experimental evidence. It should not change the central algorithm unless V1.2 exposes a documented failure.

### 7.2 Full evaluation

#### EHR

- schema and clinical validity;
- marginal, dependency, and temporal fidelity;
- TSTR/TRTS or selected downstream utility;
- privacy/memorization analysis when making a shareable-synthetic-data claim.

#### CXR

- RadDINO-FID/KID/PRDC;
- pathology-stratified results;
- artifact and duplicate rates;
- downstream utility;
- blinded radiologist realism/clinical-utility review.

#### Report

- real-anchor: GREEN + RadGraph-F1 + CheXbert-F1;
- optionally RadFact, VLScore, RadCliQ, or SPEC-CXR subject to stable implementation and budget;
- fully synthetic: structural quality plus cross-modal factuality, not reference-based metrics.

#### Cross-modal and statistical evaluation

- all three pairwise edges, per finding and subgroup;
- cross-path disagreement/uncertainty;
- rare and abnormal subgroups;
- patient-level bootstrap 95% confidence intervals;
- independent held-out evaluator and blinded human review;
- complete model-call/GPU-time accounting and Pareto curves.

### 7.3 V1.3 experimental split

Maintain patient-level separation among:

```text
development/debug
calibration/threshold selection
final held-out test
human-review subset
```

No threshold, weight, or stopping rule may be adjusted after examining held-out test outcomes.

### 7.4 V1.3 paper claim

The strongest supportable claim would be:

> Under a fixed inference budget, TriCompose uses interpretable cross-path evidence to identify and selectively repair inconsistent downstream modalities, producing more coherent synthetic EHR–CXR–Report triples than fixed pipelines and static selection while preserving modality quality and diversity.

If error localization is not reliably validated, fall back to the narrower static-composition claim rather than presenting an unsupported self-correction claim.

---

## 8. Optional V2.0 — General budgeted inference-time composition

Only start V2.0 if V1.2/V1.3 show all of the following:

- meaningful case-level complementarity across models;
- calibrated consistency scores correlate with independent fidelity/human judgments;
- adaptive calls save compute relative to exhaustive generation;
- deterministic repair is useful but leaves a clear optimization opportunity.

Possible extensions:

- learned cost-aware router or imitation learning from an exhaustive oracle;
- uncertainty-aware stopping;
- generalization to unseen generator combinations;
- a broader multimodal composition formulation beyond EHR/CXR/report.

Do not begin with reinforcement learning. A learned policy is unnecessary unless deterministic V1.2 establishes that adaptive action selection is valuable.

---

## 9. Immediate execution order

The server agent should work in this order:

1. Read this document and audit the current repository without modifying anything.
2. Locate the immutable V1.0 candidate bank, manifests, scorers, prompt bridge, tests, and existing run conventions.
3. Produce an implementation-gap matrix for V1.1.
4. Write/update tests for the evidence contract before changing scoring behavior.
5. Implement the named finding schema and migrations/adapters without rewriting V1.0 artifacts.
6. Implement prompt duplicate detection and improve the EHR-to-fact bridge conservatively.
7. Implement the three pairwise edge evaluators with support/contradiction/coverage.
8. Re-evaluate the frozen V1.0 bank as a regression test under evaluator version V1.1.
9. Prepare, but do not automatically submit, a fixed-cohort smoke run.
10. After review, run V1.1 on a small cohort, then scale only if artifacts and metrics pass validation.

Do not start V1.2 targeted regeneration until the user explicitly approves the V1.1 evaluation outputs.

Repository-context warning:

- The local laptop snapshot contains handoff artifacts but not the complete server source tree.
- The server agent must audit the actual code, configs, and tests under `/project2/ruishanl_1185/inference_3mod` before claiming that an item is implemented.
- `research_idea_tri_modal_cycle_verifier_zh.md` describes an older EHRXDiff→Report→EHR cycle proposal. It is historical background, not the current TriCompose V1.1 implementation specification.

---

## 10. First message to send to the server agent

Copy this prompt after uploading the file:

```text
Read TRICOMPOSE_VERSION_ROADMAP_AND_SERVER_HANDOFF.md completely.

First perform a strictly read-only audit of the current repository and compare it with
the document. In your response, provide:

1. the actual current V1.0 implementation state, with exact file paths;
2. a V1.1 gap matrix: required / already present / missing / semantically incorrect;
3. any conflict between this roadmap and the repository's real contracts or artifacts;
4. the smallest ordered implementation plan for V1.1;
5. the tests and output artifacts you will use as completion evidence.

Important constraints:
- do not modify files yet;
- do not download models;
- do not submit Slurm jobs;
- do not overwrite any candidate bank or prior output;
- preserve unknown as unknown;
- distinguish direct, compatible, and risk_only EHR evidence;
- distinguish real-anchor metrics from fully synthetic metrics;
- do not start targeted regeneration or an LLM agent in V1.1.
```

After the audit is reviewed, authorize implementation with:

```text
Implement only the approved V1.1 evaluation foundation. Work test-first, preserve all
V1.0 artifacts, version every schema/evaluator/output, and stop before any GPU/Slurm run.
Return changed files, test results, remaining blockers, and the exact proposed smoke-run
command for review.
```

---

## 11. Decision gates

Before moving to the next version, answer these questions explicitly:

### V1.1 → V1.2

- Do the three edge evaluators distinguish matched pairs from hard negatives?
- Does per-finding evidence agree reasonably with manual review?
- Has prompt collapse been fixed or at least reliably detected?
- Is static reranking better than the predetermined fixed path on more than one unique case?

### V1.2 → V1.3

- Can the method localize wrong CXR versus wrong report on controlled corruptions?
- Does targeted repair outperform static reranking under a comparable budget?
- Does it reduce contradictions without collapsing diversity or image/report quality?

### Continue to V2.0

- Is there genuine model complementarity?
- Is there a useful quality–compute frontier?
- Would a learned policy address a demonstrated limitation rather than add unnecessary complexity?

If the answer to a gate is no, keep the narrower validated contribution instead of forcing the next version.
