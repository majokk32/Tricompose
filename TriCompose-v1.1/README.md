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
