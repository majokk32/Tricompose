# TriCompose V1.1

TriCompose V1.1 fixes the conditioning collapse observed in the immutable V1.0
Pool-100 baseline. It does not modify or regenerate the existing synthetic EHR
records. Instead, it rebuilds the deterministic bridge:

```text
existing canonical synthetic EHR
  -> direct radiographic facts (V1 semantics, latest visit)
  + evidence-grounded clinical context (longitudinal diagnoses)
  -> one shared clinical intent
  -> model-specific RoentGen-v2 / Sana / PixArt prompt surfaces
```

The distinction between direct findings and context is strict. For example,
documented COPD or chronic kidney disease may appear under `Clinical context`,
but it is never converted into emphysema, edema, or another positive image
finding. Unknown findings remain unknown. Case IDs, random nonces, raw tokens,
and unsupported view/laterality/location/severity are not used to create prompt
diversity.

V1.0 code and artifacts remain unchanged and serve as the baseline. V1.1 uses
new schemas, package names, run IDs, and output directories.

The repository-wide version boundaries and server handoff are documented in
[`../docs/version_roadmap.md`](../docs/version_roadmap.md). In the current
roadmap, V1.1 covers prompt repair and the explainable static evaluation
foundation. Error localization and targeted regeneration are reserved for
V1.2 so they cannot be claimed before the evidence layer is calibrated.

## CPU-only bridge staging

This command performs no model inference:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=TriCompose-v1.1/src:TriCompose-v1.0/src:src \
python TriCompose-v1.1/tools/stage_existing_v1_ehrs.py \
  --source-v1-run <existing-v1-staging-run> \
  --output-root artifacts/protected/tricompose_v1_1/staging \
  --run-id <new-opaque-run-id>
```

Every case contains an unchanged canonical EHR copy, V1.1 facts, final prompts,
hash lineage, and a shared clinical-intent hash. The aggregate validation report
compares V1.0 and V1.1 prompt uniqueness before any CXR is regenerated.

## Current validated bridge

The current immutable CPU-only bridge run is:

```text
artifacts/protected/tricompose_v1_1/staging/
  bridge_gpt2_pool80_v11_20260813_003/
```

Aggregate validation over the 80 existing canonical EHR cases:

- canonical EHR hashes preserved: 80/80;
- successful V1.1 staging: 80/80;
- unique shared clinical intents: 49;
- unique prompts per CXR model: 7 in V1.0, 49 in V1.1;
- conditioning tiers: 14 direct-plus-context, 1 direct-only, 45
  context-only, and 20 honest neutral fallbacks;
- all evidence-lineage, unknown-semantics, deterministic-rendering, and
  no-random-nonce checks passed.

The 20 neutral fallbacks are not artificially diversified: the available EHR
does not support a radiology-relevant condition for them. This is reported as
underconditioning rather than hidden with case IDs or random text.

RoentGen-v2 uses an explicit four-context surface budget while retaining the
complete context list in the clinical-intent lineage. This keeps all 80 final
prompts within its 77-token limit (observed maximum: 71). Sana and PixArt final
prompts are also within their 300-token limits (observed maxima: 87 and 126).

No CXR was regenerated during this repair. Image-side diversity and clinical
response to the new prompts must be checked next with a user-approved Slurm
smoke test.

## Current implementation matrix

The repository contains more V1.1 implementation than the initial bridge
smoke described above. Protected execution status must still be verified on
CARC because generated artifacts are intentionally excluded from Git.

| Component | Repository state | Scientific state |
|---|---|---|
| EHR fact and prompt bridge | Implemented and documented on the protected 80-case bridge run | Prompt uniqueness improved; downstream clinical control still requires evaluation |
| CXR request/candidate contract | Implemented with hash and parent-lineage validation | Frozen-model pool completion must be checked in protected manifests |
| Report request/candidate contract | Implemented for MAIRA-2, CXRMate-single, LLaVA-Rad, and CheXagent-2 | All four paths remain CXR-conditioned |
| Report structural quality | Implemented under `TriCompose-v1.0/eval/report_v1_1/` | Reference-free structural quality only, not clinical factuality by itself |
| Three finding edges | Implemented as support, explicit contradiction, and coverage | Default XRV thresholds remain diagnostic until real matched calibration |
| Candidate registry | Implemented for the intended 80 × 3 × 4 = 960 lineage grid | Actual protected counts and evidence coverage must be audited |
| Static baselines | Fixed paths, deterministic random, and exhaustive static reranking implemented | Same-cohort best fixed path is descriptive, not a held-out paper baseline |
| Targeted repair | Not part of V1.1 | Planned for V1.2 after calibration and corruption tests |

The detailed evaluator contract is in
[`../TriCompose-v1.0/eval/report_v1_1/README.md`](../TriCompose-v1.0/eval/report_v1_1/README.md).

### Remaining V1.1 priorities

1. Audit which protected pool80/pool160 CXR and report jobs completed.
2. Record exact XRV, CheXbert, BioViL-T, and Qwen checkpoint revisions and
   evidence coverage.
3. Calibrate CXR per-finding thresholds on a disjoint matched real validation
   set; default 0.5 thresholds are diagnostic only.
4. Build random-swap, same-disease hard-negative, and minimal fact-perturbation
   calibration sets.
5. Verify that prompt repair changes CXR clinical content and does not merely
   increase text hashes.
6. Complete fixed, random, all-fixed-path, and exhaustive-static comparisons
   before implementing V1.2 actions.

Do not call `CXRMate-single` an EHR+CXR model. A strong V1.2 claim that report
agreement can localize an erroneous CXR requires a genuine independent path
such as CXRMate-ED or explicit validation as a heuristic using controlled
error injection.

## Frozen CXR smoke interface

The V1.1 request contract binds the unchanged synthetic EHR hash, V1.1 fact
hash, clinical-intent hash, exact final-prompt hash, model revision, and seed.
The execution adapter reads that prompt file directly and records
`adapter_added_prefix: false` in every candidate.

The prepared two-case smoke request run is:

```text
artifacts/protected/tricompose_v1_1/cxr_requests/
  prompt_repair_smoke2_allmodels_s2_20260813_001/
```

It contains 12 requests: two conditioned synthetic EHR cases, three frozen CXR
models, and two seeds. The complete, not-yet-submitted Slurm array script is:

```text
TriCompose-v1.1/slurm/10_prompt_repair_smoke2_all_cxr_a40.sbatch
```
