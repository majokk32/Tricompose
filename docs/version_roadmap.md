# TriCompose version roadmap

Last updated: 2026-09-21

[中文版](version_roadmap_zh.md)

This roadmap separates repository implementation from validated research
claims. A source file or Slurm script being present means that an implementation
exists; it does not prove that a protected CARC run completed or that a clinical
claim has been validated.

## Objective

TriCompose uses frozen existing models to construct a fully synthetic triple:

```text
Synthetic structured EHR
  -> evidence-grounded radiology facts and model-specific text
  -> synthetic CXR candidates
  -> synthetic report candidates
  -> cross-modal verification
  -> one selected or repaired (EHR, CXR, Report) triple
```

The contribution is inference-time composition and verification, not training
another large three-modal generator. The main method should be deterministic,
auditable, and reproducible before any learned router is considered.

## Shared scientific rules

1. Fix each synthetic EHR before downstream generation. Do not replace a
   difficult patient merely because its CXR or report is poor.
2. Preserve lineage and hashes from EHR through facts, prompt, CXR, report,
   scores, model revision, seed, runtime, and cost.
3. Use `positive`, `negative`, `uncertain`, and `unknown` finding states.
   `unknown` is neither negative nor agreement.
4. Separate direct radiographic evidence from compatible or risk-only context.
   Heart failure or loop-diuretic use does not require current edema.
5. Separate real-anchor evaluation from fully synthetic evaluation.
6. Reference-based report metrics cannot become reference-free selection
   metrics when no real report is available.
7. FID, KID, and PRDC are cohort metrics and cannot rank one CXR candidate.
8. Do not overwrite protected runs or historical baselines.
9. Keep selection evidence separate from independent final evaluation whenever
   possible.

## Version map

| Version | Research role | Repository status |
|---|---|---|
| Historical Phase-0 | Real-anchor interface and privacy smoke tests | Retained for provenance; not the current fully synthetic claim |
| V1.0 | Frozen generation graph plus static best-of-N | Implemented engineering baseline; scoring remains uncalibrated |
| V1.1 | Prompt de-collapse and explainable evaluation foundation | Substantially implemented; calibration and paper-primary validation remain |
| V1.2 | Error localization and targeted regeneration | Planned; not implemented as a validated method |
| V1.3 | Paper-grade scale-up and independent evaluation | Planned |
| V2.0, optional | General cost-aware or learned composition policy | Start only if V1.2 demonstrates a real need |

Earlier notes used “V1.1” for the complete targeted-repair idea. The repository
already uses `TriCompose-v1.1` for prompt repair and evaluation, so this roadmap
assigns targeted repair to V1.2. Historical run names must not be renamed.

## V1.0: static composition baseline

### Implemented

V1.0 provides:

- SynEHRgy-based synthetic EHR candidates;
- deterministic EHR-to-radiology-fact and text conditioning;
- RoentGen-v2, Sana, and PixArt CXR branches;
- MAIRA-2, CXRMate-single, LLaVA-Rad, and CheXagent-2 report branches;
- hash-bound candidate graph and atomic triple export;
- fixed-path and exhaustive static best-of-N selection;
- XRV, BioViL-T, Qwen2.5-VL, and deterministic quality evidence.

The CXR generators consume model-specific radiology text, not native structured
EHR. `CXRMate-single` is CXR-only and must not be described as CXRMate-ED.

### Validated scope and limitations

The V1 smoke established plumbing, lineage, and static selection. It did not
establish clinical calibration or cohort-level superiority. Two different EHRs
collapsed to the same CHF-only prompt, Qwen's positional finding vectors were
incomplete, and the retained winner still contained a view hallucination. The
V1.0 artifacts remain immutable as the historical baseline.

## V1.1: prompt repair and evaluation foundation

### Why this version exists

Targeted regeneration is not meaningful until every modality has explicit
facts and every edge can distinguish support, contradiction, and unavailable
evidence. V1.1 fixes conditioning collapse and makes static evaluation
interpretable before adaptive actions are introduced.

### Implemented in the repository

`TriCompose-v1.1/` contains:

- V1.1 EHR-fact and prompt-manifest schemas;
- conservative direct-fact and longitudinal-context extraction;
- model-specific RoentGen-v2, Sana, and PixArt prompt rendering;
- prompt token-budget validation and duplicate fingerprints;
- hash-bound CXR/report request and candidate contracts;
- non-overwriting execution adapters and Slurm entry points;
- tests for prompt semantics, staging, contracts, and cleanup.

The protected 80-EHR bridge run documented in
`TriCompose-v1.1/README.md` preserved all EHR hashes and increased unique
shared clinical intents from 7 V1.0 prompt variants to 49 grounded V1.1
intents. Twenty cases honestly remained neutral rather than receiving random
or unsupported text.

The legacy-located but V1.1-specific directory
`TriCompose-v1.0/eval/report_v1_1/` implements:

- reference-free report-structure checks;
- named 14-finding states with unknown-safe semantics;
- frozen XRV and CheXbert extraction entry points;
- EHR-CXR, EHR-report, and report-CXR support, contradiction, and coverage;
- direct EHR evidence separated from weak CHF/diuretic priors;
- BioViL-T and Qwen2.5-VL as secondary evidence;
- a 960-lineage registry for 80 cases × 3 CXR paths × 4 report paths;
- fixed-path, random, and exhaustive static-reranking baselines;
- edge-specific lexicographic selection with validity and contradiction gates.

Pending evidence remains `null` and is distinct from
`not_applicable_no_comparable_ehr_facts`. Diagnostic convenience scores are
not used in the lexicographic selection order.

### Remaining before V1.1 is scientifically complete

1. Verify protected pool status on CARC; protected generated artifacts are not
   stored in Git.
2. Confirm a stable CheXbert checkpoint and record its exact revision.
3. Calibrate per-finding CXR thresholds on a disjoint matched real validation
   set. Default 0.5 thresholds are diagnostic only.
4. Build random-swap, same-disease hard-negative, and minimal
   fact-perturbation calibration sets.
5. Test whether repaired prompts change CXR clinical content and diversity;
   prompt uniqueness alone is insufficient.
6. Compare the predeclared fixed path, random selection, all fixed paths, and
   exhaustive static reranking on the same fixed cohort.
7. Keep real-anchor metrics separate from fully synthetic selection.
8. Produce a protected report with per-edge coverage, support,
   contradictions, quality, model calls, runtime, and rejection.

### V1.1 completion gate

V1.1 is complete when prompt collapse is fixed or detected, each edge emits
traceable per-finding evidence, unknown and weak evidence never become hard
negatives, thresholds are frozen before final testing, and fixed/static
baselines are compared across multiple unique clinical intents.

V1.1 does not include targeted regeneration or an LLM router.

## V1.2: error localization and targeted regeneration

### Research question

Can redundant evidence identify whether a CXR or report is more likely wrong,
and can regenerating only that modality improve consistency at a lower cost
than exhaustive generation?

### Evidence-independence requirement

All four report generators in the current aligned pool depend on the CXR.
Agreement among them can reflect shared dependence on the same image and does
not independently prove that the image is wrong. Before making a strong
modality-localization claim, either:

1. deploy a genuine EHR+CXR-to-report path such as CXRMate-ED; or
2. label localization as a heuristic and validate it with controlled error
   injection plus independent held-out or human evaluation.

Never represent `CXRMate-single` as CXRMate-ED.

### Planned implementation

1. Build random-swap, same-disease, wrong-CXR, wrong-report, and minimal
   presence/negation/laterality/severity/device/temporal corruptions.
2. Freeze localization thresholds on validation data.
3. Implement deterministic actions:

   ```text
   select
   regenerate_cxr
   regenerate_report
   switch_report_model
   verify_more
   reject
   ```

4. Keep the EHR fixed and enforce configurable call-count and GPU-time budgets.
5. Record the findings and edges that caused every action.

Compare fixed, random, exhaustive static reranking, generate-all oracle
analysis, and targeted repair. Report localization accuracy, abstention,
false-repair rate, repair success, all three edges, quality/diversity,
rejection, calls, and GPU time.

Proceed only if targeted repair improves the quality-compute trade-off. If
static reranking performs equally well at comparable or lower cost, retain the
narrower static-composition contribution.

## V1.3: paper-grade evaluation and scale-up

V1.3 freezes the method and adds independent evidence:

- EHR distribution, dependency, temporal fidelity, utility, and relevant
  privacy evaluation;
- RadDINO-FID/KID/PRDC, pathology-stratified CXR results, artifact/duplicate
  rates, and downstream utility;
- real-anchor GREEN, RadGraph-F1, and CheXbert-F1, with optional stable
  RadFact, VLScore, RadCliQ, or SPEC-CXR;
- fully synthetic structural quality plus reference-free cross-modal facts;
- rare/abnormal subgroups and patient-level bootstrap confidence intervals;
- blinded radiologist review;
- selection evaluators separated from held-out evaluators;
- complete quality-compute Pareto curves.

Use patient-level development, calibration, final-test, and human-review
splits. Do not change thresholds or stopping rules after seeing final-test
outcomes.

## Optional V2.0

Consider a learned cost-aware router only if V1.2 demonstrates meaningful
model complementarity, evaluator correlation with independent judgment,
adaptive compute savings, and a limitation deterministic routing cannot solve.
Do not begin with reinforcement learning; a simple oracle-imitation or
budgeted policy is the first justified learned extension.

## Immediate server execution order

1. Audit protected V1.1 runs and manifests without printing raw
   patient-derived source data.
2. Run lightweight unit-test groups with their documented `PYTHONPATH`s.
3. Confirm which 80-case CXR/report jobs actually completed.
4. Confirm XRV, CheXbert, BioViL-T, and Qwen evidence coverage and revisions.
5. Finish calibrated V1.1 static evaluation before editing V1.2 code.
6. Show every Slurm request for user review; do not submit without approval.
7. After V1.1 review, build the corruption benchmark before targeted repair.

### Lightweight test groups

Run these separately so each package receives the intended import path:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:experiments/roentgen_v2/src \
python -m unittest discover -s tests -v

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:TriCompose-v1.0/src \
python -m unittest discover -s TriCompose-v1.0/tests -v

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:TriCompose-v1.0/src:TriCompose-v1.1/src \
python -m unittest discover -s TriCompose-v1.1/tests -v

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=TriCompose-v1.0/eval/report_v1_1 \
python -m unittest discover \
  -s TriCompose-v1.0/eval/report_v1_1 -p 'test_*.py' -v
```

The V1.1 permission tests require a POSIX filesystem that preserves the setgid
directory bit. The OpenAI-compatible policy round-trip test requires loopback
socket binding. Report environment restrictions separately from code failures.

## Prompt for the server agent

```text
Read docs/version_roadmap.md, TriCompose-v1.1/README.md, and
TriCompose-v1.0/eval/report_v1_1/README.md completely.

Perform a strictly read-only audit first. Report:
1. which V1.1 protected runs actually exist and completed;
2. candidate counts, unique case IDs, model paths, hashes, and evidence coverage;
3. which evaluator checkpoints and threshold bundles are available;
4. the gap between current artifacts and the V1.1 completion gate;
5. the smallest ordered implementation or evaluation plan, with exact paths;
6. proposed tests and Slurm commands, but do not submit them.

Do not print raw EHR rows, real reports, real images, patient identifiers, or
protected synthetic report/image contents. Do not overwrite prior runs. Keep
unknown as unknown; weak context cannot create a hard contradiction. Do not
start V1.2 targeted regeneration until V1.1 is reviewed and approved.
```

## Decision gates

### V1.1 to V1.2

- Do edge scores distinguish matched data from hard negatives?
- Does per-finding evidence agree with manual review?
- Do repaired prompts affect CXR content without collapse?
- Does static reranking beat fixed/random baselines across unique cases?

### V1.2 to V1.3

- Can controlled wrong-CXR and wrong-report cases be localized?
- Does targeted repair improve over static reranking at comparable cost?
- Are quality and diversity preserved?

### V1.3 to V2.0

- Is complementarity real and stable?
- Is there a meaningful quality-compute frontier?
- Would a learned policy solve an observed limitation?

If a gate fails, keep the narrower validated contribution instead of forcing
the next version.
