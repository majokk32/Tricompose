# TriCompose v1.0

> **Frozen historical baseline.** V1.0 remains immutable for regression and
> comparison. The active status and future plan are in
> [`../docs/version_roadmap.md`](../docs/version_roadmap.md).

TriCompose composes existing frozen models at inference time to generate
candidate fully synthetic patient triples intended to be cross-modally
coherent:

```text
(Synthetic structured EHR, Synthetic CXR, Synthetic radiology report)
```

The project does not train a new multimodal model, router, adapter, LoRA, or
consistency scorer. Version 1.0 generates candidates with frozen models,
constructs a cross-modal candidate graph, evaluates the candidates with frozen
verifiers and deterministic rules, and selects one final triple.

The supported scientific description is:

> Inference-time composition of frozen generators for coherent synthetic
> EHR-CXR-report generation.

## User-facing EHR option

The user does **not** choose an EHR model. The only first-stage choice is:

```text
use_ehr_prompt = false | true
```

### `use_ehr_prompt = false` — strict cold start

The internal candidate pool contains both deployed SynEHRgy-v2 models:

- SynEHRgy Qwen2-40bins;
- SynEHRgy GPT2-10bins.

Both generate an EHR from the BOS token only. The user does not select between
Qwen2 and GPT-2. TriCompose carries both EHR candidates through the downstream
CXR/report graph and selects among the complete triples.

The two models use the authors' MIMIC-IV-format schema. Discrete numerical
values remain official 10-bin or 40-bin tokens. TriCompose does not invent bin
boundaries or de-quantized laboratory values.

### `use_ehr_prompt = true` — exploratory prompted EHR

TriCompose internally routes the EHR stage to PromptEHR. The released
PromptEHR checkpoint does not consume a free-form natural-language request.
Its “prompt” is a partial structured synthetic-EHR seed plus demographic
conditioning.

This route remains synthetic because it uses the official public synthetic
demo cohort, but it is:

- MIMIC-III rather than MIMIC-IV;
- seed-conditioned rather than BOS-only;
- not strict cold-start generation.

The execution plan records these facts as `strict_cold_start=false` and lists
PromptEHR as an exploratory incompatible route. Prompted and strict cold-start
results must be reported separately.

If future TriCompose versions accept natural-language user requirements, that
will require a separate audited structured-prompt adapter; the current code
does not pretend that PromptEHR already provides that interface.

## End-to-end pipeline

```text
use_ehr_prompt
       |
       +-- false -> SynEHRgy Qwen2 + SynEHRgy GPT-2 candidates
       |
       +-- true  -> PromptEHR candidate (exploratory)
                              |
                              v
                  Canonical synthetic EHR
                              |
                              v
             Deterministic radiology-fact bridge
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
     RoentGen-v2             Sana              PixArt
          |                   |                   |
          +-------- Synthetic CXR candidates ----+
                              |
         +----------+---------+---------+----------+
         |          |                   |          |
         v          v                   v          v
      MAIRA-2   CXRMate-single      LLaVA-Rad  CheXagent-2
                              |
                              v
                  Synthetic report candidates
                              |
                              v
          Consistency, quality, uncertainty, cost
                              |
                              v
             One selected synthetic patient triple
```

The deterministic EHR bridge is an internal conditioning adapter, not another
output modality. It exposes only an audited set of positive, radiographically
observable facts and supported devices. Missing EHR facts remain `unknown`;
they are never converted into negative findings or a fabricated normal CXR.

## Frozen model registry

The complete registry is `configs/model_registry_v1.json`. Every active learned
component is required to remain frozen.

### EHR generation

| Model | Route | Input | v1 status |
|---|---|---|---|
| SynEHRgy Qwen2-40bins | no prompt | BOS only | active |
| SynEHRgy GPT2-10bins | no prompt | BOS only | active |
| PromptEHR | prompted | partial structured synthetic seed | exploratory |
| HALO | unavailable | no compatible deployed checkpoint | disabled |

### CXR generation

| Model | Actual input signature | v1 status |
|---|---|---|
| RoentGen-v2 | radiology-style text -> CXR | active |
| CheXGenBench Sana | radiology-style text -> CXR | active |
| CheXGenBench PixArt-Sigma | radiology-style text -> CXR | active |
| RadEdit | source CXR + edit instruction -> edited CXR | disabled; future repair edge |
| MeDiM text-to-image | report-style text -> CXR | disabled; official-checkpoint positive control failed |
| EHRXDiff | previous CXR + interval EHR -> future CXR | disabled in cold start |
| DDL-CXR | previous CXR + longitudinal condition -> future CXR | disabled |
| UniDisc text-to-image | interleaved text/image generation | smoke incomplete |
| Liquid | general text-to-image | disabled as unreliable CXR modality |

None of the active CXR models natively accepts structured EHR. The accurate
signature is:

```text
Synthetic structured EHR
  -> deterministic radiology-style text
  -> frozen text-conditioned CXR generator
  -> Synthetic CXR
```

TriCompose therefore does not claim direct structured EHR-to-CXR generation.
It also never supplies a random, blank, retrieved, or real previous CXR to make
a longitudinal model appear cold-start capable.

### Report generation

| Model | Actual input signature | Output | v1 status |
|---|---|---|---|
| MAIRA-2 | one current synthetic CXR | findings | active |
| CXRMate-single | one current synthetic CXR | findings + impression | active |
| LLaVA-Rad | one current synthetic CXR | findings | active |
| CheXagent-2 | one current synthetic CXR | structured findings | active |
| UniDisc | one current synthetic CXR | report text | legacy; disabled in aligned V1 pool |
| CXRMate-ED | EHR + CXR | report | not locally deployed |

The locally deployed CXRMate model is `CXRMate-single`, not CXRMate-ED; it must
not be described as an EHR+CXR report generator.

## Candidate expansion

The default static exhaustive plan uses one seed per registered active model.

| EHR option | EHR candidates | CXR candidates | Report candidates | Generator calls |
|---|---:|---:|---:|---:|
| `use_ehr_prompt=false` | 2 | 6 | 24 | 32 |
| `use_ehr_prompt=true` | 1 | 3 | 12 | 16 |

Each CXR candidate retains its source EHR ID. Each report candidate retains
both its source EHR ID and source CXR ID. The graph rejects reports or images
whose lineage does not form one exact EHR->CXR->Report path.

This exhaustive plan is the version-1 static baseline. A later cost-aware
dynamic policy can skip unnecessary calls while using the same model registry,
artifact contracts, graph, and selector.

## Consistency and selection

Version 1.0 represents every pairwise judgment as one of:

```text
support | contradiction | unknown
```

`unknown` is not converted to zero and is not treated as a negative clinical
finding. Required but unavailable evidence triggers `verify_more`.

The current selector consumes:

- EHR structural validity and quality;
- CXR artifact/quality checks;
- report quality and hallucination penalties;
- EHR-CXR support from a frozen XRV-based verifier;
- CXR-report matching from frozen Qwen2.5-VL;
- deterministic EHR-report fact consistency;
- model-call cost.

The uncalibrated engineering score in `configs/policy_v1.json` is:

```text
S = 0.05 * EHR quality
  + 0.15 * CXR quality
  + 0.15 * Report quality
  + 0.20 * EHR-CXR consistency
  + 0.30 * CXR-Report consistency
  + 0.15 * EHR-Report consistency
  - 0.001 * relative cost units
```

These weights are an auditable version-1 default, not clinically calibrated
values. Final experiments must calibrate thresholds using matched real triples
and hard negatives without exposing patient-level artifacts.

The deterministic policy can return:

- `stop_and_select`;
- `verify_more`;
- `generate_alternative`;
- `regenerate_cxr`;
- `regenerate_report`;
- `stop_without_selection`.

A strong EHR-CXR contradiction regenerates the CXR branch. A strong
CXR-report or EHR-report contradiction regenerates the report branch. A triple
is selected only when its global score, all required edge scores, and the
top-candidate margin pass their configured thresholds.

Report-to-CXR cycle consistency is not active in version 1.0. If added later,
it must receive low weight and cannot be used as self-confirming evidence.

### Implemented V1 score stack

The V1 scorer is split into three frozen GPU jobs and one deterministic CPU
aggregation step because the local checkpoints require different environments:

```text
12 synthetic CXR candidates
  |-- frozen XRV DenseNet
  |     -> finding probabilities
  |     -> EHR-prompt-intent/CXR support
  |     -> deterministic image non-degeneracy gate
  |
  |-- frozen BioViL-T
  |     -> EHR-derived final-prompt/CXR raw cosine
  |     -> CXR/generated-report raw cosine
  |
48 CXR-report candidate pairs
  |-- frozen local Qwen2.5-VL
  |     -> compact 14-finding image states
  |     -> compact 14-finding report states
  |     -> CXR-report match and strong contradictions
  |
  `-- deterministic aggregation
        -> report quality flags
        -> EHR-report support/contradiction/unknown
        -> complete per-triple score vectors
        -> one independent ranking for each fixed synthetic EHR
```

The shared finding order is the 14 CheXpert categories: atelectasis,
cardiomegaly, consolidation, edema, enlarged cardiomediastinum, fracture, lung
lesion, lung opacity, pleural effusion, pleural other, pneumonia,
pneumothorax, support devices, and no finding. Every finding state is one of:

```text
positive | negative | uncertain | unknown
```

Missing or unmentioned information remains `unknown`; it is never converted
to a contradiction. The XRV checkpoint has no support-device output, so that
dimension remains unavailable rather than receiving a fabricated score.

The three learned score families remain separate in the saved score vector.
BioViL-T raw cosine is not normalized into a clinical probability or silently
mixed into the selector. Qwen and XRV outputs are also explicitly marked
uncalibrated until protected matched-pair and hard-negative calibration exists.
The deterministic CXR quality check detects only blank/degenerate images; it is
not called a realism metric. A local frozen CheXbert weight has not been found,
so the current EHR-report fallback is conservatively lexical and is labeled as
such.

Selection is performed separately for every fixed synthetic EHR. TriCompose
must not compare two patients and discard the more difficult EHR. For one EHR,
the current two-seed smoke produces:

```text
3 CXR models x 2 seeds = 6 CXR candidates
6 CXR candidates x 4 report models = 24 complete triple candidates
```

The scorer entry points are:

```text
tools/score_tricompose_v1_xrv.py
tools/score_tricompose_v1_biovil.py
tools/score_tricompose_v1_qwenvl.py
tools/aggregate_tricompose_v1_scores.py
```

They have been implemented, unit-tested, and executed on the current bank.
XRV and BioViL-T produced complete coverage. The initial compact Qwen contract
did not reliably return all 14 finding positions. A protected two-pair debug
showed that its scalar CXR-report scores were valid but the positional finding
vectors were ambiguous. The repaired contract therefore retains only valid
scalar scores and marks invalid-width finding evidence as unavailable; it does
not pad or invent labels. The final Qwen scalar run parsed 48/48 pairs, after
which the static aggregator selected one triple independently for each fixed
synthetic EHR. Finding-level Qwen localization remains unavailable in V1.

## Schemas and code layout

```text
workspace/
├── TriCompose-v1.0/
│   ├── README.md
│   ├── configs/
│   │   ├── model_registry_v1.json
│   │   ├── policy_v1.json
│   │   └── tricompose_v1_cases.txt
│   ├── schemas/
│   │   ├── canonical_ehr_v1.schema.json
│   │   ├── ehr_facts_v1.schema.json
│   │   ├── candidate_graph_v1.schema.json
│   │   └── execution_plan_v1.schema.json
│   ├── src/tricompose_v1/
│   │   ├── candidate_bank.py
│   │   ├── contracts.py
│   │   ├── execution.py
│   │   ├── facts.py
│   │   ├── graph.py
│   │   ├── prompts.py
│   │   └── scoring.py
│   ├── slurm/
│   └── tests/
└── tools/
    ├── score_tricompose_v1_xrv.py
    ├── score_tricompose_v1_biovil.py
    ├── score_tricompose_v1_qwenvl.py
    └── aggregate_tricompose_v1_scores.py
```

The canonical EHR adapter preserves source tokens and visit lineage. A separate
`ehr_facts.json` contains the audited radiology bridge: every non-unknown fact
has evidence and canonical source-field pointers. PromptEHR can be normalized
to the same container, but its source schema and non-cold-start status remain
explicit.

## Protected staging contract

The CPU-only staging step materializes the immutable contract used by all later
model adapters:

```text
SynEHRgy case JSON
  -> canonical synthetic_ehr.json
  -> evidence-grounded ehr_facts.json
  -> exact final model prompts
  -> aggregate validation report
```

All CXR prompt renderers consume the same `ehr_facts.json`. Version 1 strictly
reproduces the previously run RoentGen-v2/Sana/PixArt bridge: all three receive
the same demographic + `PA chest radiograph` + findings text, and only the
legacy-supported positive diagnoses and devices enter the prompt. Negative,
uncertain, and unknown states remain available in `ehr_facts.json` for later
verification but are not passed as generation conditions. The fixed PA phrase
comes from the prior model-aligned renderer rather than an inferred EHR view.
No laterality, location, severity, or additional device is invented. The saved
prompt text is the complete final tokenizer input; adapters add no prefix.

```bash
python tools/stage_tricompose_v1.py \
  --source-root artifacts/protected/synehrgy_v2/runs/<source_run_id> \
  --case-ids-file TriCompose-v1.0/configs/tricompose_v1_cases.txt \
  --output-root artifacts/protected/tricompose_v1 \
  --run-id <new_opaque_run_id> \
  --prompt-models roentgen_v2,chexgenbench_sana,chexgenbench_pixart \
  --validate
```

The case list is explicit and ordered. Staging never selects cases by expected
generation difficulty, never modifies its SynEHRgy source, never calls an API
or GPU, rejects existing run IDs, and commits case/run directories atomically.
Rejected cases are recorded with sanitized reason codes.

## Three-modality I/O contract

Version 1 does not pass anonymous text or image paths between model-specific
scripts. Every transition is bound by candidate IDs, parent IDs, artifact
SHA256 values, model revision, seed, and frozen-model status:

```text
EHR candidate
  -> CXR request (exact saved final prompt)
  -> CXR candidate (EHR/facts/prompt lineage retained)
  -> Report request (one current synthetic CXR)
  -> Report candidate (EHR and CXR parents retained)
  -> consistency graph + stop_and_select
  -> one atomic EHR-CXR-report export
```

The executable contract is implemented in `src/tricompose_v1/contracts.py`.
Its key invariants are:

- all three active CXR generators consume the same canonical EHR and fact hash;
- their model-specific prompt files are already final tokenizer inputs;
- adapters cannot add another prefix or rebuild facts from the source EHR;
- every active V1 report model receives exactly one current synthetic CXR;
- CXR-only report models retain EHR lineage but do not receive EHR content;
- CXR and report candidates use model-independent output fields;
- a partial or cross-lineage result cannot be exported;
- final export is atomic, protected, immutable, and contains all three modalities.

The final exporter is intentionally separate from candidate generation:

```bash
python tools/export_tricompose_v1.py \
  --staging-run artifacts/protected/tricompose_v1/<staging_run_id> \
  --cxr-candidate-json <protected-cxr-candidate.json> \
  --report-candidate-json <protected-report-candidate.json> \
  --selection-json <protected-stop-and-select.json> \
  --output-root artifacts/protected/tricompose_v1/triples \
  --export-id <new_opaque_export_id>
```

It refuses to write anything unless `stop_and_select` names one EHR, CXR, and
report with matching parentage and hashes. It never overwrites an export.

### Unified execution adapters

The CPU-only request preparation command rejects underconditioned cases by
default and expands only an explicit case list, model list, and seed list:

```bash
python tools/prepare_tricompose_v1_cxr_requests.py \
  --staging-run artifacts/protected/tricompose_v1/<staging_run_id> \
  --case-ids-file <explicit-two-case-list> \
  --output-root artifacts/protected/tricompose_v1/cxr_requests \
  --run-id <new_request_run_id> \
  --models roentgen_v2,chexgenbench_sana,chexgenbench_pixart \
  --seeds 0,1
```

GPU jobs then use one shared CXR entry point. The selected runtime receives the
saved prompt text byte-for-byte and returns the common CXR candidate contract:

```bash
python tools/run_tricompose_v1_cxr.py \
  --request-run <protected-request-run> \
  --output-root artifacts/protected/tricompose_v1/cxr_candidates \
  --output-run-id <new_cxr_run_id> \
  --model-id <roentgen_v2|chexgenbench_sana|chexgenbench_pixart> \
  --model-dir <read-only-model-snapshot> \
  --batch-size 1
```

The four V1 report models are connected to one common report entry point while
retaining their already-tested model-specific preprocessing and decoding:

```bash
python tools/run_tricompose_v1_report.py \
  --cxr-run <protected-common-cxr-run> \
  --output-root artifacts/protected/tricompose_v1/report_candidates \
  --output-run-id <new_report_run_id> \
  --model-id <maira2|cxrmate_single|llavarad|chexagent2> \
  --model-dir <read-only-model-snapshot>
```

LLaVA-Rad additionally requires `--model-base`, `--biomedbert-dir`,
`--external-root`, and a fresh `--runtime-dir`. CheXagent-2 additionally
requires `--vision-dir`. These are the same read-only assets used by their
previous successful report experiments.

These two runner commands contain model inference and therefore must only run
inside an explicitly approved Slurm allocation. The request preparation and
contract tests are CPU-only.

## Planning interface

The commands below are lightweight dry runs. They do not launch inference.

No EHR prompt:

```bash
PYTHONPATH=TriCompose-v1.0/src python -m tricompose_v1.cli plan \
  --registry TriCompose-v1.0/configs/model_registry_v1.json
```

Use the PromptEHR route:

```bash
PYTHONPATH=TriCompose-v1.0/src python -m tricompose_v1.cli plan \
  --registry TriCompose-v1.0/configs/model_registry_v1.json \
  --use-ehr-prompt
```

Inspect the validated internal registry:

```bash
PYTHONPATH=TriCompose-v1.0/src python -m tricompose_v1.cli registry-summary \
  --registry TriCompose-v1.0/configs/model_registry_v1.json
```

## Artifact and privacy contract

Fully synthetic protected runs will use new opaque run IDs under:

```text
artifacts/protected/tricompose_v1/<run_id>/
```

The frozen V1.0 implementation originally creates owner-only directories with
mode `0700` and files with mode `0600`. The current collaborative workspace
policy uses project-group `2770`/`0660`; authorized migration is performed by
`tools/set_collaboration_permissions.sh`, while V1.1 creates group-shared
artifacts natively. Synthetic EHR
content, generated prompts, images, and reports must never be copied into this
README, Git history, or public Slurm logs.

No raw patient-level MIMIC EHR row, identifier, CXR, or radiology report may be
sent to an external API. If a future LLM policy is used, it may receive only
anonymous candidate IDs, scores, action history, uncertainty, and model cost.

All model inference and GPU verification must run through Slurm. Every batch
script and resource request must be shown and explicitly approved before
submission.

## Current implementation status

Implemented and tested:

- validated frozen-model registry;
- boolean `use_ehr_prompt` routing interface;
- SynEHRgy and PromptEHR canonicalization contracts;
- evidence-grounded EHR-to-radiology-fact contract;
- exact legacy-aligned shared RoentGen-v2, Sana, and PixArt prompt renderer;
- protected non-overwriting atomic staging and aggregate validation reports;
- common CXR request/candidate and Report request/candidate contracts;
- unified frozen CXR runners for RoentGen-v2, Sana, and PixArt;
- unified frozen report runners for MAIRA-2, CXRMate-single, LLaVA-Rad, and
  CheXagent-2;
- atomic final three-modality exporter;
- exhaustive dependency-aware plan generation;
- typed candidate graph with lineage validation;
- deterministic cost-aware triple selector;
- protected candidate-bank finalizer with complete model/lineage coverage;
- candidate-bank adapters for frozen XRV, BioViL-T, and Qwen2.5-VL scoring;
- deterministic report quality and EHR-report state comparison;
- independent per-fixed-EHR static ranking and selection aggregation;
- JSON schemas and lightweight unit tests.

### Completed current two-case generation run

The aligned V1 two-case smoke has completed generation. Its immutable unified
entry point is:

```text
artifacts/protected/tricompose_v1/candidate_banks/
  v1_smoke2_all_models_20260806_001/
```

Its canonical EHR/fact/prompt staging source is:

```text
artifacts/protected/tricompose_v1/
  staging_qwen2_pool10_v1aligned_20260806_001/
```

The finalized manifest records:

| Modality | Count | Expansion |
|---|---:|---|
| Fixed synthetic EHR | 2 | two explicitly fixed eligible cases |
| Synthetic CXR | 12 | 3 models x 2 seeds x 2 EHRs |
| Synthetic report | 48 | 4 models for every synthetic CXR |
| Complete candidate triples | 48 | 24 alternatives per fixed EHR |

The component generation runs are:

```text
artifacts/protected/tricompose_v1/cxr_candidates/
  v1_smoke2_roentgen_v2_s2_20260806_001/
  v1_smoke2_sana_s2_20260806_001/
  v1_smoke2_pixart_s2_20260806_001/

artifacts/protected/tricompose_v1/report_candidates/
  v1_smoke2_maira2_allcxr_20260806_001/
  v1_smoke2_cxrmate_allcxr_20260806_001/
  v1_smoke2_llavarad_allcxr_20260806_001/
  v1_smoke2_chexagent2_allcxr_20260806_001/
```

All listed protected run directories were originally created with V1.0's
owner-only `0700` policy. They may be migrated to the current CARC
project-group privacy contract by the authorized collaboration-permissions
tool. The
candidate-bank manifest has `complete_generation_candidate_bank=true` and
`selection_performed=false`.

### What ordinary generation returns without scoring

Generation and selection are intentionally separate. If scoring is disabled,
the pipeline returns the complete immutable candidate bank: for each EHR, six
CXR alternatives and four reports per CXR. It does **not** call any candidate
"best" and it does not export an arbitrary triple.

A no-scoring experimental baseline may be defined as a pre-registered fixed
path such as one fixed CXR model, one fixed seed, and one fixed report model.
That baseline is useful for comparison, but its result is a deterministic
baseline output rather than an automatically selected optimum. Choosing a
candidate after viewing its image/report would be manual cherry-picking and is
not the V1 method.

The current two-case no-scoring baseline has been exported with this fixed
greedy path:

```text
RoentGen-v2 -> seed 0 -> MAIRA-2
```

Its protected immutable output is:

```text
artifacts/protected/tricompose_v1/baselines/
  v1_smoke2_greedy_fixed_20260806_001/
```

It contains two complete EHR-CXR-report triples under `triples/`. The run
manifest records `scoring_used=false`,
`post_hoc_output_inspection_used=false`, and identifies its scientific role as
an ordinary fixed-pipeline baseline rather than an optimum. Every triple keeps
the candidate lineage and hashes. The immutable export was originally created
with V1.0 owner-only `0700`/`0600` permissions; current CARC collaboration may
migrate it to project-group `2770`/`0660` without changing its contents.

The current V1 static scoring run is complete:

- frozen XRV: 12/12 CXR candidates scored;
- frozen BioViL-T: 12/12 prompts and 48/48 reports scored;
- frozen Qwen2.5-VL: scalar score parsed for 48/48 report pairs, with 0/48
  complete 14-finding vectors;
- aggregation: 48 score vectors materialized; both fixed EHR cases returned
  `stop_and_select` under the uncalibrated V1 engineering policy;
- both selected `CheXGenBench Sana, seed 1 -> CXRMate-single`, with a global
  score of `0.90055133`.

These are static, uncalibrated best-of-N outputs rather than clinically
validated optima. A hash audit found that the two selected CXR files and the
two selected reports are respectively identical. The two canonical EHRs are
different, but the current bridge mapped both to the same CHF-only prompt, so
the duplicate downstream outputs are a prompt-collapse/information-bottleneck
result rather than two independent successes. V1.1 fixes the bridge and builds
the explainable static evaluation foundation. Targeted regeneration is reserved
for V1.2 after V1.1 calibration and controlled corruption tests.

The final protected teacher handoff containing the bilingual report, fixed
baseline, scored-best exports, and final scoring artifacts is:

```text
artifacts/protected/tricompose_v1/handoffs/
  v1_teacher_scored_best_20260806_001.tar.gz
```

The older `v1_teacher_progress_20260806_001.tar.gz` is a preliminary engineering
snapshot and does not contain the final scored selection.

## Tests

```bash
PYTHONPATH=TriCompose-v1.0/src:src PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s TriCompose-v1.0/tests -v
```
