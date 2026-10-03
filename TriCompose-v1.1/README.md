# TriCompose V1.1

TriCompose V1.1 fixes the conditioning collapse observed in the immutable V1.0
Pool-100 baseline. It does not modify or regenerate the existing synthetic EHR
records. Instead, it rebuilds the deterministic bridge:

```text
existing canonical synthetic EHR
  -> conservative direct radiographic facts (latest visit)
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

The current CPU-only bridge repair is:

```text
artifacts/protected/tricompose_v1_1/staging/
  bridge_pool80_semantics_20260922_001/
```

It preserves all 80 EHRs byte-for-byte. CHF is context, not an automatic
cardiomegaly/edema/congestion finding. Word-boundary and conservative polarity
checks prevent organism-name substrings and uncertain/historical diagnoses
from becoming asserted current pathology. These remain transparent lexical
rules, not a clinically validated diagnosis extractor.

Validation: 80/80 staged; 8 with direct positive radiographic conditions,
50 context-only, 22 neutral; 46 distinct clinical intents. Sana/PixArt have
46 distinct prompt texts and RoentGen-v2 has 45 (its context surface budget can
merge different intents). Hash diversity is descriptive, not a correctness
gate. No nonce or invented disease was added to increase uniqueness.

The hash-only delta to the August bridge is at:

```text
artifacts/protected/tricompose_v1_1/bridge_deltas/
  delta_pool80_semantics_20260922_001/
```

34 cases have changed prompts: 100/240 model prompts changed, 140 unchanged.
Unchanged prompt bytes do **not** authorize silently relabeling old CXR/report
candidates with new fact/intent hashes. Their old lineage remains immutable;
reuse needs an explicit lineage certificate. The 12-image smoke completed in
Slurm job 12256783 (first two changed opaque case IDs, three models, two seeds).
This establishes execution, not clinical conditioning: both cases have clinical
context but no direct positive radiographic facts. See the input audit below.

### Historical August bridge and completed pool

The historical immutable bridge run is:

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

The initial bridge construction used no GPU. A subsequent protected audit on
2026-09-22 confirmed 240 CXR candidates and 960 report candidates completed
against this **August** bridge. They are not outputs of the September repair.
XRV covers 240/240 CXRs and CheXbert 960/960 reports; BioViL-T and Qwen evidence
are still missing. Threshold calibration and independent clinical validation
remain incomplete.

## Current implementation matrix

The repository contains more V1.1 implementation than the initial bridge
smoke described above. Protected execution status must still be verified on
CARC because generated artifacts are intentionally excluded from Git.

| Component | Repository state | Scientific state |
|---|---|---|
| EHR fact and prompt bridge | Implemented and documented on the protected 80-case bridge run | Prompt uniqueness improved; downstream clinical control still requires evaluation |
| CXR request/candidate contract | Implemented with hash and parent-lineage validation | August pool: 240/240; September smoke: 12/12 completed, clinical control unvalidated |
| Report request/candidate contract | Implemented for MAIRA-2, CXRMate-single, LLaVA-Rad, and CheXagent-2 | All four paths remain CXR-conditioned |
| Report structural quality | Implemented under `TriCompose-v1.0/eval/report_v1_1/` | Reference-free structural quality only, not clinical factuality by itself |
| Three finding edges | Implemented as support, explicit contradiction, and coverage | Default XRV thresholds remain diagnostic until real matched calibration |
| Candidate registry | Verified 80 × 3 × 4 = 960 historical lineage grid | XRV/CheXbert coverage complete; independent secondary evidence absent |
| Static baselines | Fixed paths, deterministic random, and exhaustive static reranking implemented | Same-cohort best fixed path is descriptive, not a held-out paper baseline |
| Targeted repair | Not part of V1.1 | Planned for V1.2 after calibration and corruption tests |

The detailed evaluator contract is in
[`../TriCompose-v1.0/eval/report_v1_1/README.md`](../TriCompose-v1.0/eval/report_v1_1/README.md).

### Remaining V1.1 priorities

1. Review the September CPU audit and semantic repair delta; August pool
   completion and XRV/CheXbert evidence coverage are now verified.
2. Review the completed 12-image/48-report smoke; score the current candidates
   and validate runtime tokenizer tracing before expanding CXR regeneration.
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
hash, clinical-intent hash, exact supplied-prompt hash, model revision, and seed.
The outer execution adapter reads that prompt file directly and records
`adapter_added_prefix: false`. This does NOT imply that an official model
pipeline leaves tokenizer text unchanged.

The new prepared two-case smoke request run is:

```text
artifacts/protected/tricompose_v1_1/cxr_requests/
  smoke2_semantics_all3_s2_20260922_001/
```

It contains 12 requests: two changed synthetic EHR cases, three frozen CXR
models, and two seeds. Cases are not selected for favorable output. The script
was explicitly approved and completed as Slurm job 12256783:

```text
TriCompose-v1.1/slurm/60_semantics_smoke2_all_cxr_v100.sbatch
```

Resource request: three array tasks, each one V100, four CPUs, 32 GiB RAM,
20-minute wall-time limit. This is a limit, not a runtime/queue prediction.
RoentGen uses float16; all weights remain frozen. Logs stay protected. Show the
entire script and obtain explicit approval before any new submission.

### Tokenizer boundary audit and submitted reports (2026-09-22)

All 12 synthetic images have verified request/image hashes and valid PNG
dimensions. Each CXR run below contains four images:

```text
artifacts/protected/tricompose_v1_1/cxr_candidates/
  semantics_smoke2_roentgen_v2_s2_12256783/
  semantics_smoke2_chexgenbench_sana_s2_12256783/
  semantics_smoke2_chexgenbench_pixart_s2_12256783/
```

Current installed source shows that Sana adds its official text-encoder
instruction prefix and lowercases/strips the supplied prompt; PixArt also
lowercases/strips it. RoentGen's supplied text is unchanged in this reconstruction.
Keep these official behaviors; do not remove the prefix merely to make a hash
match. The immutable companion audit is under
`artifacts/protected/tricompose_v1_1/tokenizer_input_audits/smoke12_source_reconstruction_12256783_001/`.
It explicitly says `current_source_reconstruction_only`, `runtime_observed: false`.
No old candidate or image was rewritten. Current source does not prove the
historical environment or actual token IDs.

Future V1.1 CXR execution now records `tokenizer_input.txt`, `tokenizer_trace.json`,
and input/text/token-ID hashes inside each protected candidate. This observes
tokenizer calls, not a text-encoder hook; it preserves the official arguments.
It has synthetic unit coverage but has NOT yet been GPU-validated.

The prepared report request run is
`artifacts/protected/tricompose_v1_1/report_requests/semantics_smoke12_four_experts_12256783_001/`:
12 existing images × 4 frozen CXR-only experts = 48 requests, each retaining
hash-bound references to the reconstruction audit. Script
`slurm/61_semantics_smoke12_reports_v100.sbatch` requests four array tasks, each
1 V100, 4 CPUs, 32 GiB RAM, and a 20-minute limit. After review of the complete
script and explicit user approval, it was submitted as array job 12256939.
All four tasks completed (exit 0), with 12 reports each. Scheduler elapsed
times were MAIRA-2 167s, CXRMate-single 31s, LLaVA-Rad 104s, and CheXagent-2 98s.
All 48 reports are nonempty with verified hashes and protected permissions.
Clinical quality has not yet been verified. Output run names are `semantics_smoke12_<model>_12256939`
under `artifacts/protected/tricompose_v1_1/report_candidates/`.

Reference-free structure checks and an explicit 2-case/2-seed/48-row registry
are now under `artifacts/protected/tricompose_v1_1/evaluation/smoke48/`.
The one-GPU evaluation script
`TriCompose-v1.0/eval/report_v1_1/slurm/30_smoke48_score_select_v100.sbatch`
was explicitly approved and completed as job 12257177 (67 seconds, exit 0).
XRV covers 12/12 images, CheXbert 48/48 reports, and BioViL-T 48/48 pairs.
All 48 scored rows and two selected triples passed output integrity checks.
Outputs are under `evaluation/smoke48/scoring_12257177/` within the protected
V1.1 root. Both EHR direct-comparison edges remain NA; selection is diagnostic
only and has not been independently clinically validated.

The fixed-versus-selected protected review is at
`artifacts/protected/tricompose_v1_1/evaluation/smoke48/review_12257177_001/`.
Its decomposition shows fixed-path positive/negative support of 2/0 versus
selected-path 1/9, with zero explicit contradictions in both. The apparent
coverage gain therefore comes mainly from negative label agreement, not a
demonstrated clinical-quality gain.

An offline positive-focused policy ablation is saved under
`artifacts/protected/tricompose_v1_1/evaluation/smoke48/positive_focused_12257177_001/`.
It does not reward negative agreement. It selects PixArt seed 0 + CheXagent-2
for case_000 and PixArt seed 1 + CXRMate-single for case_003. Positive support
is 4/24 versus 2/24 for the fixed path, with zero contradictions in both.
This is a two-case sensitivity analysis, not a clinical claim.
See the evaluation README for the frozen metrics, fixed-seed baseline, and
diagnostic-only interpretation.
