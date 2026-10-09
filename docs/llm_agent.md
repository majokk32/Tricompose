# LLM-driven TriCompose: entry points and limits

## Current V1.2 interface (2026-10-08)

New code is grouped under `TriCompose-v1.2/agent/`. No old experiments,
outputs, checkpoints, scorers or consumed worker scripts were moved/changed.
The old single-score interface below remains available for compatibility.

Current planner: frozen local **Qwen2.5-VL-7B-Instruct**, with no training,
fine-tuning or external API. The user requested a later planner-strength study:
can a stronger API decision model achieve the same final quality with fewer
generation calls and lower **total** cost? It is a deferred hypothesis, not an
assumed improvement or authorization to connect an API. The controlled design,
privacy boundary and full-cost accounting are recorded in
[further development](further_development.md#deferred-experiment-planner-strength-versus-total-generation-cost-2026-10-08).
Keep the current local-Qwen execution path and frozen acceptance rules for now.

Next prepared interface: `TriCompose-v1.2/agent/run_current_image_policy.py`
and `tricompose_llm/current_image_qwen.py`. The previous five-observation
interface and its consumed outputs remain immutable. This new version supplies
only the two reports actually observed on each current seed-one image, their
numeric edge/risk readouts, the cached image-disagreement guard, and count-only
feedback from the rejected expert switch. Other-image expert histories and
BioViL endpoint scores are excluded. Only the two remaining untried report
experts (LLaVA-Rad and CheXagent-2), or stop/abstain, are permitted. No clinical
acceptance, CXR fault attribution by voting, or new scorer/threshold is introduced.

A deterministic fixed-priority rule uses the exact same packet and prospective
two-worker allowance. Its reserved planning allowance is **not** an incurred
model call. This is a same-observation/same-action-budget **decision control**,
not measured GPU savings or evidence that Qwen beats a rule. Matching deferred
requests must not be double-executed and counted as different outcomes. Any
live generator comparison and endpoint evaluation needs separate approval.

CPU-prepared plan (zero new model calls):
`artifacts/protected/tricompose_v1_2/current_image_policy_plans/current_policy2_12827441_001/`.
Manifest SHA256:
`6bdfadedbd14f18548dba9fd622d0e297a822e17692350dbc5f4a3eb6d180786`.
The frozen local Qwen loader retains its minimum 24-GiB memory guard; the
approved script is `agent/slurm/15_current_image_policy2_debug_a40.sbatch`
(one A40, two CPUs, 32G host RAM, five-minute cap, at most two Qwen calls).
It was submitted unchanged after complete script/resource review and explicit
approval, as job `12851312`. Every generation action remains a separately
approved deferred request. `agent/audit_current_image_policy.py` performs metadata-only plan/run
replay, including real versus reserved costs and exact dispatch/journal checks.
Preparation and independent CPU plan audit completed in existing CPU Slurm
`12827441`: all four observed report rows are represented, 319 source pins and
138 artifact pins were checked, and protected permissions passed. The audit is
at `artifacts/protected/tricompose_v1_2/current_image_policy_audits/current_policy2_plan_12827441_001/`
(manifest SHA256 `5f4fe919c939b87059613de15d7b857bbca04a556bf930dc3784df83f879fe4d`).
All 14 new authored boundary/replay tests and all **3,875 V1.2 tests** passed
(full suite 43.608 seconds); shell syntax and whitespace checks passed.
No new model call occurred in that CPU preparation step.

The approved decision job completed on debug A40 `b11-09` in **1 minute
5 seconds**, exit `0:0`. Both fresh Qwen decisions were valid and requested
LLaVA-Rad, exactly the same action/target as the fixed rule (**2/2 matches**).
No worker/generator/scorer was invoked; no original winner changed. Worker
runtime was 30.724 seconds and peak Torch-allocated VRAM was 15.887 GiB
(not total GPU memory). These observations do **not** show LLM superiority,
repair effectiveness or measured generation savings.

Completed output:
`artifacts/protected/tricompose_v1_2/current_image_policy_runs/current_policy2_12851312/`
(manifest SHA256 `470e4a483ae43c62ef1b2620cbeedc2698b1508ac51412893d8bb13a25f22bed`).
CPU postflight replayed actual calls, actions, the durable journal, rule control,
pending requests and costs without body/pixel inspection:
`artifacts/protected/tricompose_v1_2/current_image_policy_audits/current_policy2_12851312_12851223_001/`
(manifest SHA256 `adfde92fec930f1e9467922e97dce07cd46d554977dcadfb07a409b954afdfd5`).
Protected output, task and public log modes retain the project access boundary.

Completed shared execution: `agent/run_current_image_requested_reports.py`
authenticates both matching strategies and deduplicates four provenance rows
into **two physical reports plus two CheXbert calls** on the unchanged images.
The same output can be referenced by both policies, but is not two independent
outcomes; Qwen's already-incurred two calls are retained, while the rule's
model-call cost is zero. The existing report adapters, frozen gate, old exhausted
ledgers and original selected triples remain unchanged. The generator is not
executed merely because a planner requests it.
Plan: `artifacts/protected/tricompose_v1_2/current_image_report_plans/current_report2_12851223_001/`
(manifest SHA256 `fec15af05c44acf054de6ac8d60dca06a37a27af5b41db93749d1d9fbf734b25`).
Eight new fixture tests and all **3,883 V1.2 tests** passed (full suite
28.948 seconds). The reviewed complete script is
`agent/slurm/16_current_image_shared_reports2_debug_a40.sbatch`: one A40,
two CPUs, 32G RAM, eight-minute cap; no new Qwen/CXR/XRV/image observer,
training, retry, original-winner promotion or external API. LLaVA-Rad retains
its existing 24-GiB planning memory class; P100 is below it. After complete
script/resource display and explicit approval, the unchanged script was
submitted as job `12851790`; it completed on debug A40 `b11-09` in **2 minutes
28 seconds**, exit `0:0`. Two reports and two CheXbert operations completed;
four attempts were charged, zero failed, and **zero branch replacements**
passed the unchanged gate. This shared result is not an LLM-versus-rule win.

| Opaque pair ordinal | Baseline expert | EHR–Report support / known, baseline → LLaVA | CXR–Report support / known, baseline → LLaVA | EHR–Report proxy opposition, baseline → LLaVA |
|---|---|---|---|---|
| 0 | CXRMate-single | 0/1 → 0/1 | 4/8 → 3/8 | 0 → 1 |
| 1 | MAIRA-2 | 1/1 → 0/1 | 2/8 → 2/8 | 0 → 1 |

These are unweighted **proxy** fact counts, not clinical accuracy. All
CXR–Report supported facts here are negative labels. EHR–CXR stayed at 0/1
support and one proxy opposition for each case. Pair 0 lost image-comparable
facts and added an EHR–Report opposition; pair 1 lost EHR support, added an
opposition, and introduced unsupported temporal comparison language. Official
section contracts passed for both new reports, so this is not an empty-output
or missing-section failure. Existing branch reports and original seed-zero
winners were retained. One image guard has missing comparison evidence and
the other has scorer disagreement; neither establishes an actual image fault.
Do not blindly regenerate more reports or retune the scorer to force acceptance.

Output:
`artifacts/protected/tricompose_v1_2/current_image_report_runs/current_report2_12851790/`
(manifest SHA256 `ff6cbed07e712dbbfc89dc02ae76115414325348f1936cd24eb755f912b6e1d8`).
`score_table.csv` contains baseline/new report readouts; the paired gate details
are in `paired_report_comparison.json`. CPU-only postflight replayed all 12
new-phase journal events, unchanged old ledgers, actual receipt/artifact hashes,
gate choices and byte-exact CSV. The actual two CheXbert label-file hashes were
also verified without parsing report bodies or pixels. Audit:
`artifacts/protected/tricompose_v1_2/current_image_report_audits/current_report2_12851790_12851223_001/`
(manifest SHA256 `7038d7f0070cbb1b92e93b4a94b91d5711ff218eef342479746b52f6d3886d37`).
Six new authored postflight tests and all **3,889 V1.2 tests** passed (full suite
30.637 seconds). No new GPU task or automatic retry was submitted after this
run. The next useful diagnostic is independent image/evidence assessment,
not claiming this expert switch repaired the triple.

Completed fixed-image report-only reachability diagnostic in CPU allocation
12851223 (no new model calls). `agent/diagnose_current_image_evidence.py` and
`tricompose_llm/fixed_image_reachability.py` analyze the same two development
cases, retaining all three actually observed reports per current image.
Both have one explicit fixed EHR–XRV opposition. A report can agree with
either side but cannot agree with both; unknown/uncertain only silence report
comparisons. Thus the frozen-label all-three support ceiling is **0/1 for each**,
irrespective of how many report models are called. This is a label-space bound,
not clinical fault localization, a decoder-attainability result, or a ban on
improving report structure/other findings. The diagnostic is NOT installed
as a routing/stopping policy and does not alter the consumed gate.

The cached blind image guard remains `unresolved_missing_image_evidence` for
pair 0 and `unresolved_scorer_disagreement` for pair 1. These statuses do not
prove the generated CXR is wrong. The observer shares the planner checkpoint;
report consensus and global BioViL cosine cannot become independent image
truth. XRV's legacy `finding_probabilities` are operating-point-normalized
scores, not calibrated probabilities; boundary distances are diagnostic only.
No threshold fitting, head-mask change or post-hoc best scorer/template choice.

Output: `artifacts/protected/tricompose_v1_2/current_image_evidence_diagnostics/evidence2_12851223_001/`.
Read `RESULTS_CN_EN.md`; numeric tables are `image_evidence_table.csv` and
`report_observations.csv`. The per-finding score distances and four symbolic
states are in `image_fact_diagnostics.json`; they are not newly generated
reports or clinical image annotations. Manifest SHA256:
`5a890ac1f3e044d73500bd335da3d5c2d252d501cf8065c54b4690f426f61776`.
Independent arithmetic checked all 112 symbolic entries; six artifact hashes,
327 source hashes, 44 direct reader hashes, byte-exact table/report replay and
2770/0660 project permissions passed. Eighteen new invented-fixture tests
passed; all **3,907 V1.2 tests** passed in 30.071 seconds. No raw source data,
generated report prose or pixels were parsed.
The next useful evidence is separately validated image presence/absence,
not blind report retries or score relaxation. Further inference still requires
a complete new Slurm script/resource review and explicit approval.

Completed follow-up, **job 12853380**: evaluated the same frozen Qwen
image-only observer on the existing RSUA50 pneumonia/normal-cohort proxy
diagnostic, preserving the XRV/BioViL/SigLIP cohort and original source order.
`real_validation/rsua_qwen_observer.py` uses the unchanged eight-finding prompt,
decoder settings and lazy mechanical guard. Only pneumonia has a reference;
unknown/uncertain/failure remains separate, all 50 denominators are retained,
and references enter evaluation only after prediction fsync. No majority-vote
truth, fitted threshold, synthetic-generation saving or clinical repair claim.

Plan: `artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_plans/rsua_qwen50_12851223_001/`.
Manifest SHA256 `e8a38e289adc778bdc4fc89a91d31cdf4fc832414226db5261932ce3d5a9d6b3`.
Program SHA256 `d0b97f73924c031ca272e31bbf123abd79a847b80d25f12869b350a6b11ef16d`.
317 source hashes / ten artifact hashes / 50 image stats / seven asset stats
passed CPU preparation checks; no source pixels or cohort labels were parsed.
Nineteen new fixtures and all **3,926 V1.2 tests** passed in 30.371 seconds.
Approved and submitted unchanged: `agent/slurm/17_rsua_qwen50_debug_a40.sbatch`, SHA256
`626a1ed86ba21191f0cced0c17d81b0e90e9df980c51593f7ffe2baf00b6f321`.
Resources: debug A40 × 1, two CPUs, 32 GiB host RAM, ten-minute cap; worker
timeout 540 seconds, at most 50 call attempts and one model load, zero retries.
After full-script/resource display and explicit approval, job `12853380` ran
on `b11-09` and completed in **2m41s**, exit `0:0`. It charged 50 observer
calls and one model load; all 50 responses were complete. Worker runtime
146.098s, BF16, peak Torch-allocated VRAM 15.594 GiB (not total device usage).
No new generation, numeric-planner calls, retries, training or external API.

Only **2/25 pneumonia-proxy slots** and **13/25 normal-proxy slots** were
explicitly supported. Sixteen answers were uncertain, none unknown/unavailable.
Explicit coverage: 34/50 (0.68); correct explicit support over ALL slots:
15/50 (0.30); conditional support over explicit answers: 15/34 (0.4412).
Unchanged XRV readouts are 31/50 (default 0.5) and 29/50 (transported 0.55457607),
both with full explicit coverage. This is a concerning pneumonia readout for
this unchanged Qwen observer on this development proxy cohort, not a universal
model ranking or clinically adjudicated accuracy. Do not convert uncertainty
into negative, fit a new threshold/prompt after inspection, or promote Qwen
to clinical fault judge. Numeric scheduling and image verification are separate
roles: this result neither validates nor disproves the numeric planner.

Output: `artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_runs/rsua_qwen50_12853380/`.
Manifest `5e4bd836c83d6210237a9cd581f88618de2ab7967b76a43317b9ba7990a3ea48`.
CPU metadata-only audit:
`artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_audits/rsua_qwen50_12853380_12851223_001/`
(manifest `fadfba8badd84e770d75313fa67968a76cdf3d5100e192bcd669a9bb55e90dd0`).
Four output artifacts, 317 source hashes, ten input hashes, 50 PNG byte hashes,
seven asset stats, all 151 journal events, group permissions and summary
arithmetic passed. No raw reference rows/pixels were opened by that audit;
per-image reference matching and model inference were not independently rerun.
This tests a published class proxy, NOT independently adjudicated clinical
image labels; keep the existing image attribution guard closed. Protocol:
`docs/rsua_qwen_observer_protocol.md`; results and next-step limits:
`docs/rsua_qwen_observer_result.md`. Further GPU runs require a separately
reviewed complete script and fresh explicit approval.

Completed medical report-observer comparison, **job 12861745**:
`real_validation/rsua_medical_report_observer.py` follows the existing official
CheXagent-2 SRRG findings interface, not a new JSON/classification prompt.
Same report-model revision, seed 42, greedy decoding, 512-token cap and deployed
float32 precision. A separate existing CheXbert process reads only these new
protected generated reports; overlength inputs are withheld, not truncated.
The models do not coexist on the GPU. All 50 source slots remain, including
blocked/empty/capped/runtime-failed/unattempted stages; no retry or replacement.
Actual report generate and label forward counts are separate from reservations
and model loads. Reference evaluation occurs only after fsynced predictions.

This is an exploratory follow-up selected after inspecting Qwen's proxy result,
not untouched validation. CheXbert is also the candidate scorer; CheXagent's
visual encoder is XraySigLIP. Neither the chain nor inter-model agreement grants
independent clinical truth, repair authority or a measured planner advantage.
The numeric Qwen planner and original clinical attribution guard are unchanged.

CPU-only plan:
`artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_plans/rsua_medical50_12851223_001/`
(manifest `8edb55e03155b546acdce1d627407924e1bf54d50e1bcc716452d0e387f2331f`).
Preflight authenticated 320 code hashes, 14 input artifact hashes, 50 image
stats and 26 model-asset stats; zero new model calls or source-image/label reads.
Twenty-four new authored-fixture tests and all 3,950 V1.2 tests pass (30.763s).
Approved exact script: `agent/slurm/18_rsua_medical50_debug_a40.sbatch`, SHA256
`6762012ecbce971e6a3c2dbb636b672b19bc4ea66246b5ffb199d455891c9861`.
One debug A40, two CPUs, 32 GiB host RAM, ten-minute cap; parent timeout 540s,
report subprocess 420s, label subprocess 60s, maximum 50 requests per stage
and one load per model. After the full script/resource display and explicit
approval, it completed unchanged on `b11-09`, exit `0:0`, elapsed **2m47s**.
Fifty report generate attempts and fifty label forwards completed, zero retry;
worker interval 150.494s, peak Torch allocations 13.362/0.422 GiB respectively.

The pneumonia head returned **50 unknown, zero positive/negative/uncertain**.
Explicit coverage and correct explicit support are 0/50; conditional support
is **NA**, not zero. This is missing clinical comparison evidence, not 50
clinical errors, 50 negatives, or model-runtime failure. Other finding heads
have positive and negative states. Outputs were uncapped and label inputs
untruncated. Metadata alone does not establish why the reports lack a usable
pneumonia assertion or whether their other assertions are correct.

Do not replace unknown by negative, map opacity/consolidation to pneumonia,
relax a guard, revise original winners or promote this chain to an independent
image adjudicator. The comparison cohort remains development/proxy-only.
This diagnostic is separate from numeric Qwen scheduling and does not show
LLM superiority, accepted repair or measured saved calls.

Completed source: `artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_runs/rsua_medical50_12861745/`
(manifest `4df6cf3378aa0a6dd7ab38ce93261fb42909606be4798930b63e671328e33a2f`).
Metadata audit: `artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_audits/rsua_medical50_12861745_12851223_001/`
(manifest `f65051c63a6a9bb1d57aa1f194bd40550c3a779d3744318b1f44d6eae19b766b`).
The postflight byte-hashed 57 result artifacts, 320 source files, 14 input
artifacts and 50 PNGs, checked 26 asset stats, and replayed both 151-event
journals plus numeric state/aggregate contracts. No report text/reference rows
or pixels were opened, no model instantiated, and per-image reference
assignment was not independently recomputed.
Protected `observer_comparison.csv`, `finding_state_counts.csv` and bilingual
readout: `artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_reviews/rsua_medical50_12861745_12851223_001/`
(manifest `405629c3d37c79388bea033410bc58c9406390b505109ae6f36b295ebe6d10a9`).
Result: `docs/rsua_medical_report_observer_result.md`.
Frozen protocol: `docs/rsua_medical_report_observer_protocol.md`; original
worker, tests, script and plan were not rewritten after observing outcomes.
Any additional model/real-image execution needs a newly reviewed complete
script and fresh explicit approval.

Completed next contrast, **job 12862704** (2026-10-08):
`agent/run_cxr_action_diversification.py` uses the same two fully synthetic
EHR anchors, facts and original model-specific prompts. Predeclared actions
are RoentGen-v2 seed 2 and Sana seed 1; each image receives its own XRV,
CXRMate-single report and CheXbert calls. Four image/report slots, maximum
16 charged requests, zero retry; no planner/API, training, scorer update or
original-winner replacement. Cross-generator changes include the original
renderer, so this is a development candidate expansion, not a pure model
ablation or clinically qualified repair. All incomplete/failed slots remain.
Plan: `artifacts/protected/tricompose_v1_2/cxr_action_diversification_plans/cxr_actions2_12851223_001/`
(manifest `da1e819bbd9f0cf901388e5e0a5cf1fcf02cb25701a34a3fbf96db1147a51a16`).
Fourteen new tests and all 3,964 V1.2 tests passed. Protocol:
`docs/cxr_action_diversification_protocol.md`. Proposed script19 requests
one debug A40, two CPUs, 32 GiB RAM and ten minutes; worker timeout 120s,
outer timeout 570s. This is a budget ceiling, not a runtime/queue promise.
The complete script/resources were shown, explicitly approved and submitted
unchanged. It completed on b11-09 in **4m02s**, exit 0:0, four completed
triples/16 validated charged requests, zero failure/retry. EHR–CXR support
remains 0/1 in all four slots; the unchanged proxy gate passes 0/4, without
changing original winners. Metadata-only postflight and all 3,974 tests pass.
Run: `artifacts/protected/tricompose_v1_2/cxr_action_diversification_runs/cxr_actions2_12862704/`.
Detailed counts, positive/negative support distinction and limitations:
`docs/cxr_action_diversification_result.md`. No clinical repair authorization,
LLM advantage or saved-call claim follows; no additional GPU run is approved.

```text
TriCompose-v1.2/agent/
├── run.py                       # demo / cached-candidate pilot entry point
├── run_local_qwen_bounded.py     # current local-only bounded Qwen pilot
├── compare_cached_pilot.py       # paired CPU controls for a verified pilot
└── tricompose_llm/
    ├── contracts.py             # closed numeric state / JSON decisions
    ├── planner.py               # OpenAI-compatible API and explicit rule mock
    ├── local_qwen.py            # optional frozen Qwen planner; GPU Slurm only
    ├── local_qwen_v2.py         # retained typed-content correction
    ├── local_qwen_bounded.py    # pre-call empty-tool-menu guard
    ├── local_pilot.py           # versioned private run receipts and counters
    ├── paired_cache_comparison.py # numeric-only rule/random control policies
    ├── controller.py            # propose → reserve → observe → accept/rollback
    └── demo.py                  # wholly authored, patient-free fixtures
```

Tests: `TriCompose-v1.2/tests/test_llm_repair_agent.py`.

Completed engineering checks on 2026-10-08: 43 new fixture tests, all 3,571
V1.2 tests, and all four legacy API-interface tests passed. The authored
demo and first-ten-case numeric-cache smoke completed in existing CPU Slurm
12799642 with **rule mock only**: no API, local LLM or generator calls.
The cache mock took no acquisition actions, so that ten-case run checks
loading/serialization/privacy, not repair effectiveness. Both protected runs
passed artifact hash and 2770/0660 project-group permission checks:

```text
artifacts/protected/tricompose_v1_2/llm_agent_runs/
├── llm_contract_smoke_12799642_001/
└── llm_cached_contract_smoke_12799642_001/
```

Original reviewed script:
`TriCompose-v1.2/agent/slurm/01_local_qwen_cached_smoke2_debug_a40.sbatch`.
Request: debug, one A40, two CPUs, 32 GiB RAM, 15-minute wall-time cap.
It uses the first two fixed source-order EHR cases, at most four planning
steps each, with current frozen numeric readouts. This is real local LLM
routing over existing candidates, still not fresh generation or a clinical
repair comparison. Explicit approval after full script review is required.

### First local run and typed-content correction (2026-10-08)

After explicit approval, job **12812363** started on A40 `b11-09` and failed
after 23 seconds. The audited Qwen loaded frozen; peak GPU allocation was
16,636,220,928 bytes. The first planning request failed with zero generated
tokens and zero candidate acquisitions. The second case was not attempted.
No API calls, generator calls, model training or original winner changes
occurred. Retained protected evidence:
`artifacts/protected/tricompose_v1_2/llm_agent_runs/llm_local_smoke2_12812363/`.

Source inspection found a reproducible boundary bug: the installed processor
with `tokenize=True` traverses `message['content']` and reads each block's
`type`, while the initial local adapter supplied strings. The known-working
Qwen workers and [official Qwen2.5-VL examples](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)
use typed text blocks. The pure-Python regression reproduces v1's failure
before generation; it does not require a model/GPU. This establishes an
adapter defect consistent with the run, not a clinical model-quality verdict.

New, versioned implementation, without changing consumed v1 sources:

- `agent/tricompose_llm/local_qwen_v2.py`: same audited frozen loader, prompt,
  budget and strict JSON validator; system/user content are text-block lists.
  Adds fixed-code failure-stage counts and response hashes, never response text.
- `agent/run_local_qwen_v2.py`: explicit local-only entry point; reuses the old
  parser, protected IO, numeric cache loader and controller unchanged.
- `tests/test_local_qwen_typed_content.py`: 11 authored-fixture regression tests.
- `agent/slurm/02_local_qwen_typed_smoke2_debug_a40.sbatch`: subsequently
  approved and submitted as job **12812738**; findings below.

All 11 new regression tests and all **3,582 V1.2 tests** passed in existing
CPU Slurm 12799642. Script pins, consumed v1 source hashes, failed-run artifact
hashes and protected project-group modes were independently rechecked.

### Typed GPU result and affordable-menu boundary (2026-10-08)

Approved job **12812738** ran on A40 `b11-09` and exited with failure after
33 seconds. Frozen Qwen completed four generation attempts: the first three
JSON action proposals passed validation and acquired three different cached
report candidates. None passed the unchanged proxy replacement contract;
the initial candidate was retained. The fourth response parsed as JSON but
failed decision validation. Its body was not saved; the exact offending
field is not established. The second case was not attempted.

The numeric action journal independently establishes that the fourth request
had **zero affordable, untried tools**: the three alternative report models
were exhausted and the remaining two simulated units could not fund an
image probe. This is a program boundary that should terminate before asking
a model, not a reason to relax decision or clinical acceptance validation.
Peak GPU allocation was 17,007,434,752 bytes. There were no API calls, fresh
generator calls, accepted transitions or original winner changes. Evidence:
`artifacts/protected/tricompose_v1_2/llm_agent_runs/llm_typed_smoke2_12812738/`.
Its artifacts, consumed code pins and protected modes were rechecked.
The retained v2 receipt's `cases_completed=1` means one processed failed case,
not a successfully completed case; new receipts distinguish these counters.

New sources preserve all consumed v1/v2 implementations and runs:

- `agent/tricompose_llm/local_qwen_bounded.py`: validates numeric state first.
  If no tool is available, issues an explicit **programmatic abstention**
  before Qwen inference. Otherwise it delegates unchanged to the real frozen
  typed-content planner. Invalid model outputs still fail closed; they are
  not rewritten into valid stops or repair successes.
- `agent/tricompose_llm/local_pilot.py` and `agent/run_local_qwen_bounded.py`:
  reusable local-only pilot runner; the original controller, strict acceptance
  gates, scores, bank and selected outputs are unchanged. Records separate
  `cases_processed`, `cases_completed`, `policy_requests`, actual Qwen
  attempts and programmatic terminal stops.
- `tests/test_local_qwen_affordable_guard.py`: eight wholly authored tests,
  including three rejected probes followed by a no-model terminal guard.
- `agent/slurm/03_local_qwen_bounded_smoke2_debug_a40.sbatch`: subsequently
  approved after full script review and submitted as job **12813565**; first
  two source-order cases, maximum four policy requests, budget 16 units each,
  one debug A40 / two CPUs / 32 GiB / 15-minute cap. Completed findings below.

All eight guard tests and all **3,590 V1.2 tests** passed in approved CPU
Slurm 12799642. The bounded entry point's imports/help and the proposed batch
script syntax/source pins were checked before submission.
The unchanged controller conservatively charges one policy unit even for a
programmatic terminal guard, but that invocation is **not** counted as an
actual Qwen inference. Source-attributed receipts make the distinction
explicit. Abstention is not evidence of a clinically correct triple.

### Completed bounded local-Qwen pilot (2026-10-08)

Approved job **12813565** completed on A40 `b11-09` in 41 seconds (runner:
38.769 seconds). Both fixed source-order cases completed. There were seven
actual frozen Qwen inference attempts, eight conservative policy requests,
one explicitly programmatic empty-menu abstention, seven cache acquisitions,
and one accepted proxy transition. All tokenizer/generation/JSON/decision
failure-stage counts were zero. GPU peak allocation: 17,017,336,320 bytes.
No API, CXR generator or report generator was called. The original bank,
source EHR, original selected outputs and frozen scoring code were not changed.

| Pilot index | Actual Qwen calls | Cache attempts | Accepted proxy transitions | Terminal | Simulated units |
| --- | ---: | ---: | ---: | --- | ---: |
| 0 | 3 | 3 | 0 | programmatic abstention | 14 |
| 1 | 4 | 4 | 1 | four-request step limit | 16 |

For pilot index 1, the image stayed fixed and the selected report expert
changed from MAIRA-2 to CheXagent-2. CXR–Report comparable findings and positive
support both increased from **1 to 3**, out of 12 explicit classifier-side
findings; explicit opposition stayed zero. Exact fact-set rechecking found
no new opposition, lost comparable facts, silenced conflicts or structure
quality regression. One subsequent request re-acquired an already observed
cache candidate and was rejected as a charged worker failure, not accepted
as new evidence. Valid LLM decisions do not imply every acquisition is useful.

For pilot index 0, no candidate passed the unchanged strict proxy gate, so
the original candidate was retained. Both pilot cases have **zero known EHR
findings in the existing frozen projection**. This is not evidence that their
canonical EHRs contain no diagnoses; it means this readout does not provide
an explicit EHR–CXR or EHR–Report comparison here. Therefore the accepted
transition is only a classifier/labeler CXR–Report proxy improvement, not
verified three-modal correctness, clinical repair, or LLM superiority.

Protected receipts:

```text
artifacts/protected/tricompose_v1_2/llm_agent_runs/
└── llm_bounded_smoke2_12813565/
    ├── start_manifest.json
    ├── events.jsonl
    ├── case_000.json
    ├── case_001.json
    ├── summary.json
    └── manifest.json
```

Post-run review independently loaded only the hash-pinned numeric cache,
recomputed all completed action credits and final selections, checked cost
reservations, and verified artifact/code/source hashes and project-group
2770/0660 permissions. No raw EHR, report body or image was opened. The
controller/LLM still never received unrequested candidates or the secondary
endpoint. This is a completed routing-interface smoke, not a benchmark claim.
Next steps remain same-case/cost-controlled comparisons and then a separately
reviewed fresh-worker bridge. Any further GPU batch needs its own complete
script/resource review and explicit approval.

### Paired engineering controls for the same two cases (2026-10-08)

`agent/compare_cached_pilot.py` reuses the verified completed GPU receipt and
runs CPU-only controls under actual Slurm. It checks the original code/source
and artifact hashes, then reproduces **every recorded Qwen state, decision,
action credit, charged failure and selected snapshot** without calling Qwen.
All policies use the same numeric visibility boundary, initial candidate,
registered tools, frozen scores, strict local acceptance/rollback, fixed EHR,
budget 16 and maximum four policy requests. No new scorer or model is trained.

Controls are fixed-path; two count-only rules (report-first, and feedback that
switches to an eligible image branch after an unsuccessful report attempt);
and uniform affordable-tool selection with seeds 0–4. Rule coverage gaps
motivate exploration, not clinical contradiction. Missing comparisons never
become negative labels or successful consistency. These controls were added
after the two-case pilot and are **exploratory engineering controls**, not a
preregistered final test or a claim about every possible deterministic policy.
This comparison does not yet include budgeted static reranking.

Both rule variants and random use the original controller's conservative
one-unit dispatch convention. Equal simulated budgets **do not mean equal
measured CPU/GPU costs**: the rules require no LLM inference; Qwen previously
required seven. Fixed-path makes no dispatch calls and uses four units.
Failed/repeated acquisitions remain charged for every policy. Random replicas
are averaged within each EHR before equal-weight averaging across EHRs.

| Method | Mean CXR–Report positive support count | Mean CXR–Report coverage | Mean cache attempts | Mean simulated units |
| --- | ---: | ---: | ---: | ---: |
| Fixed path | 0.5 | 4.17% | 0 | 4 |
| Count rule with feedback | 1.5 | 12.50% | 3.5 | 15 |
| Count rule, report first | 1.5 | 12.50% | 3.5 | 15 |
| Random affordable tools, five seeds | 0.9 | 7.50% | 2.9 | 15.6 |
| Recorded local Qwen | 1.5 | 12.50% | 3.5 | 15 |

All numbers describe the **same two synthetic cases**, not clinical accuracy.
All methods' EHR-edge coverage ratios are NA with zero available cases. The
two rule variants select the same result as Qwen here: **there is no observed
LLM advantage**, and Qwen has additional inference overhead. Because neither
case has known EHR findings, this cohort does not exercise the rule's image
feedback branch. Low report/classifier comparison coverage and the absence
of independent evaluation prevent a clinical three-modal repair claim.

Completed in existing approved CPU Slurm **12799642**, with no additional
model/API/generator calls and no original run changes. New protected run:

```text
artifacts/protected/tricompose_v1_2/llm_agent_comparisons/
└── llm_rules_paired2_12799642_001/
    ├── start_manifest.json
    ├── trials.json              # 18 trial rows; random has five seeds per case
    ├── control_receipts.json    # numeric action journals, not text or images
    ├── per_case.json            # random seeds averaged within each EHR
    ├── score_table.json         # five methods, paired-case denominators and NA
    ├── summary.json
    └── manifest.json
```

Manifest SHA256:
`da7670b6876ca4bfa4ba9532239571d5bf057d27bf2595dda2ee8c65b3d5b38f`.
The 22 wholly authored control tests and all **3,612 V1.2 tests** passed.
Artifact/code hashes, score-table aggregation and protected group modes were
independently rechecked. Original GPU receipts and the candidate bank remain
unchanged.

The fully shown `agent/slurm/04_local_qwen_matched10_debug_a40.sbatch` was
explicitly approved and submitted as **12814592**. One debug A40, two CPUs,
32 GiB, maximum 15 minutes; first ten fixed source-order cases, budget 16,
maximum four policy requests, followed by the same CPU comparison. All
policies/readouts were frozen before this request. No EHR was replaced and
no eligibility/easy-case filter was used. Results follow.

### Completed fixed ten-case extension (2026-10-08)

Job **12814592** completed on A40 `b11-09` in **1 minute 50 seconds**. The
local runner took 105.819 seconds and the CPU comparison 0.393 seconds.
All ten cases completed: 32 frozen Qwen inference attempts, 40 conservative
policy requests, eight programmatic empty-menu abstentions, 32 report-cache
acquisition attempts and two accepted proxy transitions in two distinct
cases. All tokenizer/generation/JSON/decision failure counts were zero.
Two re-acquisitions of already observed candidates were rejected and charged
as worker failures. No image acquisition was requested by Qwen or either
rule. Peak GPU allocation was 17,017,460,736 bytes. No API, CXR generator or
report generator was called, and no original run/output was overwritten.

The five-method comparison has 90 trial rows: ten cases times one fixed,
two rule, five random and one recorded-Qwen trial. Replicas are not patients.

| Method | Mean CXR–Report positive support count | Mean CXR–Report coverage | Mean cache attempts | Mean simulated units |
| --- | ---: | ---: | ---: | ---: |
| Fixed path | 0.70 | 8.33% | 0 | 4 |
| Count rule with feedback | 1.10 | 12.50% | 3.2 | 14.4 |
| Count rule, report first | 1.10 | 12.50% | 3.2 | 14.4 |
| Random affordable tools, five seeds | 0.86 | 10.00% | 2.84 | 15.48 |
| Recorded local Qwen | 1.10 | 12.50% | 3.2 | 14.4 |

**Qwen and both rules selected exactly the same final candidate for every
case**, not just the same aggregate score. Each changed two reports while
keeping the associated image and EHR fixed. There is no observed LLM benefit
in this extension; Qwen adds 32 inference calls. This is not a reason to
change frozen scores, relax acceptance, or select an easier cohort.

Only one of the ten EHRs has a known finding in the frozen projection, and
it agrees with its fixed classifier label. Nine EHR-edge ratios are NA.
The remaining EHR–Report coverage is zero because its report labels do not
provide a comparable explicit state; this is missing evidence, not an
observed contradiction. CXR–Report coverage remains low. All images pass
only basic artifact validity and all reports score 1.0 on the simple structure
proxy; neither certifies clinical/anatomical correctness. This extension
includes the prior two cases and is a routing-interface check, not a held-out
clinical or error-localization benchmark.

Protected outputs:

```text
artifacts/protected/tricompose_v1_2/llm_agent_runs/
└── llm_matched10_12814592/

artifacts/protected/tricompose_v1_2/llm_agent_comparisons/
└── llm_rules_matched10_12814592/
    ├── score_table.json
    ├── per_case.json
    ├── trials.json
    ├── control_receipts.json
    ├── summary.json
    └── manifest.json
```

Pilot manifest SHA256:
`0a86036cff89ece105302bdd6152475de9b65d04ddb53d4fae7e15a2efada576`.
Comparison manifest SHA256:
`be0971006392f95f7d344a5bf73909a7e93010bd68d4b7366417e0e96a44de53`.
Post-run checks verified source/code/artifact hashes, group permissions,
aggregation, exact per-case selection equality, and charged failures. The
comparison reproduced every original policy state/action/credit without
model inference. The 3,612-test regression passed before this submission;
no consumed Python source was changed afterward.

The next implementation boundary is the real frozen-worker bridge, not
another score revision or an immediate eighty-case numeric-routing run.
Source inspection confirms existing generator/scorer workers can be reused,
but the current controller deliberately rejects live execution. The bridge
must distinguish cache acquisition from fresh generation, use verified
single-call receipts and the unchanged fixed-EHR/fact-preservation gates,
and execute models sequentially. In particular, an approximately 17 GB
resident Qwen planner must not overlap a report expert planned for 40 GiB on
one 48 GiB A40. A new live implementation, tests, complete batch/resource
review and explicit approval are still required; it has not been implemented
or run here. Fresh generation alone would not establish an LLM advantage.

Use `run_local_qwen_bounded.py` for proposed new local pilots. The old
`run.py` remains the API/mock entry point; v1/v2 local sources remain retained
for audit and reproducibility.

The LLM can propose `regenerate_report`, `regenerate_cxr`, `stop`, or `abstain`.
A registered tool binds the model/seed locally; the LLM cannot submit a shell
command, change EHR, invent a disease/device, or write its own clinical prompt.
It sees ONLY rebased ordinal candidate/evidence/tool/model IDs, edge-count
scores, uncertainty counts, quality proxies, action history and cost units.
Even original synthetic case IDs, artifact hashes/paths, finding names and
EHR provenance remain local. Extra keys, stale IDs, unknown tools, duplicate
JSON keys, refusals and truncated outputs fail closed. Remote HTTP and HTTP
redirects are disabled; loopback HTTP is allowed for an already-approved
model service. No secret/authentication files are read.

The planner is not constrained to the old deterministic argmax. Its proposed
replacement is checked by the **unchanged** `probe_repair_v1.action_credit`:
fixed EHR hashes/states/provenance, fixed CXR on report-only actions, fixed
report expert on image actions, exact supported/comparable fact preservation,
no new proxy opposition, no unknown/uncertain silencing, no duplicate artifact
credit and no quality regression. All of these are **proxy invariants**, not
independent clinical validation. Stop/abstain never certifies a correct triple.

### Readiness: routing prototype, not fresh generation

This first interface supports authored demos and replay of the existing
hash-pinned, fully synthetic numeric cache. During cache replay only the
requested completed candidate becomes visible; the full bank and secondary
evaluation endpoint are not sent to the planner. The first N cases follow
source order, never disease eligibility or generation difficulty.

**Cache replay calls no CXR/report generators.** Actions named `regenerate_*`
acquire an existing candidate in this mode; manifests explicitly record
`actual_generator_calls=0`, `actual_regeneration_executed=false`, and clinical
repair success as null. Live execution is deliberately rejected until a
separately reviewed bridge to `LocalFrozenBackend` and its verified receipts
is implemented. Qwen here is a numeric text planner, **not** a qualified
multimodal clinical critic. No training or learned router is introduced.

Initial acquisition is four simulated units, one planner call one unit,
report acquisition/reverification two units, and image acquisition plus its
fixed-expert report/reverification four units. These are planning units,
**not GPU seconds, dollars or measured fresh-generation call counts**. Actual
API attempts/tokens and local planner inference attempts are reported
separately. Failed calls remain charged; no automatic retries. Reservations
are fsynced to a protected journal before a call, including planner calls.
Interrupted runs stay in place for audit and cannot be overwritten/resumed.

### Offline authored smoke (no API, inference or patient data)

From the workspace root, with a NEW run ID:

```bash
PYTHONDONTWRITEBYTECODE=1 python TriCompose-v1.2/agent/run.py \
  --mode demo --planner mock --run-id llm_demo_001
```

The rule mock only tests contracts and rollback; it is not an LLM experiment.
Outputs, journals and code/source pins stay under
`artifacts/protected/tricompose_v1_2/llm_agent_runs/<run_id>/` with project
group modes 2770/0660. Nothing is added to public README or Git from outputs.

### Existing-candidate LLM pilot (requires an approved Slurm allocation)

External API, reusing the original endpoint/model/key environment names:

```bash
PYTHONDONTWRITEBYTECODE=1 python TriCompose-v1.2/agent/run.py \
  --mode cache-replay --planner api --allow-network \
  --case-count 10 --budget-units 16 --max-steps 8 \
  --run-id llm_cached_pilot_001
```

Set the endpoint/key as described below; never paste the key into chat, code,
command-line arguments or Git. `--response-format json_object` is an explicit
compatibility option for a provider without JSON-schema support, with the
same strict local decision validation. No silent format/model fallback.
API output formatting follows the official [OpenAI Structured Outputs
guide](https://developers.openai.com/api/docs/guides/structured-outputs);
valid JSON is not evidence that a clinical decision is correct.

Alternatively, **inside separately approved GPU Slurm only**, the existing
audited frozen Qwen checkpoint can plan without any external network call:

```bash
PYTHONDONTWRITEBYTECODE=1 python TriCompose-v1.2/agent/run_local_qwen_bounded.py \
  --mode cache-replay --planner local-qwen \
  --model-path /project2/ruishanl_1185/reyanshg/models/Qwen2.5-VL-7B-Instruct \
  --case-count 10 --budget-units 16 --max-steps 8 \
  --run-id llm_local_pilot_001
```

Use the existing compatible Qwen environment, set all cache/temp/bytecode
locations inside the workspace, and allocate a GPU with at least 24 GiB.
The loader enforces the already-audited weight hash and freezes every parameter.
The earlier contract smoke used only a rule mock. Real local Qwen inference
was subsequently attempted in the four explicitly approved jobs above. The
third completed the two-case interface pilot; the fourth completed the fixed
ten-case extension. None establishes clinical repair effectiveness or LLM
superiority over deterministic rules.
Show a complete batch script/resources and obtain explicit approval before
submission. The CLI does not submit Slurm or download a model.

### Fresh report-worker bridge (2026-10-08, completed engineering run)

New versioned entry points, separate from the consumed cache controller:

- `TriCompose-v1.2/agent/run_fresh_report_agent_v2.py prepare|run` (current default)
- `TriCompose-v1.2/agent/run_fresh_report_agent.py` (retained first preflight)
- `TriCompose-v1.2/agent/run_local_qwen_decision.py`
- `TriCompose-v1.2/agent/tricompose_llm/live_report_bridge.py`
- `TriCompose-v1.2/agent/tricompose_llm/fresh_report_worker.py`

This first live interface is deliberately **report-only**: fixed synthetic EHR,
unchanged original RoentGen-v2 seed-0 prompt, one fresh image and XRV receipt,
then an initial CXRMate-single report/CheXbert receipt. A frozen local Qwen may
request an untried MAIRA-2, LLaVA-Rad or CheXagent-2 expert on that same image.
It receives only the closed numeric state, not EHR/report bodies or pixels.
Report generation and verification are real calls, not cached acquisitions.
There is **no CXR regeneration tool** in this bridge. Report switches cannot
repair an EHR/image mismatch; any such mismatch remains unresolved.

The planning subprocess exits before a report worker starts, releasing Qwen's
GPU memory. Only then does the unchanged official frozen report adapter run;
unchanged frozen CheXbert labels the new output. EHR hashes, image hashes,
scorer checkpoints, threshold bundle and disabled-head mask must remain fixed.
Initial and proposed outputs use the same existing fresh eight-enabled-XRV-head
profile, not the historical numeric-cache profile. Consequently, new results
cannot be presented as a like-for-like extension of the ten-case cache result.

The existing fresh-output veto is reused without threshold tuning. A proposal
must also preserve the *currently retained* report's comparable/supported facts
and common structure/risk checks. Duplicate, cosmetic, conflict-silencing and
regressive outputs are not accepted. These are proxy-preserving transitions,
not demonstrated clinical repairs. An invalid/section-ineligible baseline or
unavailable evidence is not silently called successful.

The fixed two already-examined development anchors come from the authenticated
earlier bounded-regeneration CPU plan; no easy-case search or EHR replacement.
Limits per case: at most ten generator/verifier reservations (four baseline
phases plus three two-phase report switches), three policy requests, zero
automatic retries. Qwen requests are reserved/fsynced before spawning;
generation/scoring reservations are durably recorded in the existing ledger.
Failed calls stay charged, and partial runs stay protected for reconciliation.
Counts/planning units are not measured GPU time or a demonstrated cost saving.
No training, downloads, external API, real target reads or historical overwrite.

Twenty authored-receipt/ledger tests, eight supplemental-asset tests and 28
postflight tests passed without loading models. The full V1.2 suite passed
**3,668 tests** in the existing CPU allocation `12817261`. This
establishes contract behavior, not stable
live inference or quality gains. CPU preflight only checks source/request/asset
hashes and local environments; a new GPU run still requires a complete batch
script/resource display and fresh explicit approval.

Expected new protected output: `llm_fresh_report_runs/<opaque_run_id>/` below
`artifacts/protected/tricompose_v1_2/`, containing `selection.json`,
`score_rows.json`, `score_table.csv`, `completed_triplets.json`,
`execution_summary.json`, `policy_audits.json` and hash-bound `manifest.json`.
Per-case selections are sealed; originals and the cached comparison are unchanged.

The supplemental-asset entry point keeps the first sealed CPU preflight/source
byte-identical and adds a new protected plan, rather than rewriting it. Its
inventory includes Vicuna base weights, BiomedBERT, CheXagent vision weights,
CheXbert BERT initialization files and the declared LLaVA Python source tree.
Only explicit model directories/source files are traversed; datasets and
credential/cache roots are not. The new plan locks these supplemental bytes
alongside the existing main-model/scorer pins. The run emits an additional
`execution_verification_v2.json` and `manifest_v2.json`, distinguishing charged
dispatch attempts from confirmed completed generator/scorer phases. A failed
worker reservation is not proof that an artifact was successfully generated.

CPU preflight completed in the existing approved allocation `12799642`, with
zero new model calls. Current plan:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_plans/fresh_report2_fullassets_12799642_001/`.
Manifest SHA256:
`01516d6965c3b0e287acff715da1735086580ddc8aa8d9ef39525ea8a9533e1f`.
302 source pins and 75 supplemental file pins were present; source/manifest
hashes and private project-group modes passed the post-preflight metadata audit.
This does not establish GPU inference success.

Approved and completed script (job `12817356`, A40 `b11-09`, Slurm elapsed
14 minutes 13 seconds, exit `0:0`):
`TriCompose-v1.2/agent/slurm/05_fresh_report_agent_smoke2_debug_a40.sbatch`.
Request: debug, one A40, four CPUs, 64G host RAM, 45-minute maximum allocation.
The cap includes model reloads and strong checkpoint/source checks, not a
prediction that one image needs 45 minutes. The script pins the reviewed plan
and entry points, redirects caches/native logs below protected output, and
refuses existing output directories. Output will be
`artifacts/protected/tricompose_v1_2/llm_fresh_report_runs/llm_fresh2_<Slurm_job_id>/`.
At most two fresh baseline CXR artifacts, eight report artifacts, twenty charged
generator/verifier reservations and six policy reservations. Early stopping may
produce fewer. The user explicitly approved the complete displayed script and
resource request before this submission. A different script/resource request
or subsequent run still requires its own complete display and approval.

New independent postflight entry point:
`TriCompose-v1.2/agent/audit_fresh_report_agent.py`. It runs only in an existing
CPU Slurm cgroup, after both completed manifests are present. It authenticates
the reviewed plan, stored source pins, final artifact hashes and protected modes;
replays the existing ledger and `FreshReportSession` against the durable policy
journal; and requires saved numeric states to contain only then-completed
evidence. Saved decisions, feedback, selected candidate, worker/policy charges,
score CSV and completed-phase counts must reproduce exactly. Synthetic EHR,
image and report artifacts are byte-hashed only, never body-parsed or decoded.
Its new audit run does not change the consumed generation/scorer code or source
artifacts. An incomplete run is refused, not silently resumed or called valid.
These checks establish bookkeeping/interface integrity, not clinical correctness.

Completed output:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_runs/llm_fresh2_12817356/`.
Supplemental root manifest SHA256:
`39f74f4abd00e15c5fece64c7545ddb97aa9632d3058af9c6d8b80539b354096`.
There were two completed CXR phases, two XRV phases, eight completed report
phases and eight CheXbert phases; zero worker failures. Qwen completed six
decisions, acquiring all three alternatives for each image. One report change
passed the unchanged preservation gates; the other case kept its initial
report. Both ended at their reservation limits, not a demonstrated efficient
quality-based stop. Twenty worker calls plus six policy calls were charged;
the fixed two-baseline path would require eight worker calls and no planner.
That arithmetic is not an empirical cost-matched superiority result, and
allocation/driver wall time is not measured per-model GPU time.

CPU postflight completed in allocation `12817261`:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_audits/fresh_report2_12817356_12817261_001/`.
Audit manifest SHA256:
`0c08af97545c869f57b56a38140d3a8ff7181e1badb925fba532540c16c33151`.
Durable ledger/policy replay, no-future-evidence state checks, artifact hashes,
source pins, protected permissions, CSV readouts and all completed-phase/cost
counts passed. The audit contains `score_comparison.csv`, with unweighted raw
baseline/retained edge readouts and explicit NA; it does not add another scorer.
The audit performed zero model calls and did not parse EHR/report bodies or
decode image pixels. Original outputs and consumed source files remain intact.

Important remaining limitation: both fixed EHR/image pairs retain an explicit
classifier-proxy opposition after selection. The accepted change improves one
image/report positive-support count, not the unresolved EHR/image relation.
Clinical truth is absent from this new run; the independent endpoint below is
a separate measurement, not another selection step. The two already-examined
development cases cannot establish efficacy, localization, cost savings, or
LLM superiority over static/rule selection.

### Independent fresh-report endpoint (completed, 2026-10-08)

`TriCompose-v1.2/agent/run_fresh_report_secondary.py prepare|run` reuses the
unchanged existing frozen BioViL-T endpoint and vendor bootstrap. CPU preflight
authenticates job `12817356` and its completed postflight, pins all eight report
pairs/two images, and seals fixed CXRMate, the existing expert-order first-strict-
preserving static replay, and the original LLM choices before endpoint scores
exist. The static comparison uses the shared already-generated candidate pool;
it is not an executed cost-matched online policy. Both static and LLM choices
match on these two cases. That does not support an LLM advantage.

The endpoint never changes a choice, generator, primary scorer, threshold or
EHR. It scores all eight pairs rather than only the accepted change. Whole
reports exceeding the deployed text context stay explicitly NA; there is no
silent truncation or denominator dropping. Raw cosine is an independent
secondary readout, not calibrated clinical truth. With only two previously
examined development cases, no significance/efficacy claim is warranted.

Sealed CPU plan:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_endpoint_plans/fresh_report2_biovil_12817261_001/`.
Manifest SHA256:
`c5f9435f49a15f5f43a0c58fe0a7fdf71869b265d24aa04f76bef218d6497745`.
334 source pins and eight model/tokenizer asset pins were checked with zero
model calls, source-body parsing or model construction. Fifteen new authored
endpoint/control tests passed; the full suite passed **3,683 tests** in CPU
allocation `12817261`.

Approved script, submitted unchanged after complete-script/resource display
and the user's explicit approval:
`TriCompose-v1.2/agent/slurm/06_fresh_report_secondary_v100.sbatch`.
Request: gpu partition, one V100, two CPUs, 8G host RAM, ten-minute allocation
cap; worker timeout eight minutes. Job `12827754` completed on `d14-08` in
**25 seconds**, exit `0:0`. All eight pairs were available: two image encodings
and eight text encodings, no generation or primary scoring. Driver wall time
including load/I/O was 18.478 seconds; peak Torch allocated VRAM was 0.579 GiB
(not total device memory use). Encoding costs are separate from
the original twenty worker/six-policy calls. Failed endpoint work is retained
and charged as a worker attempt; unknown completed encoding counts are not
invented. Caches and native logs stay protected, external network/API is off.

Completed output:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_endpoints/fresh_report2_biovil_12827754/`,
with `scores.json`, `score_table.csv` (all eight candidates),
`case_comparison.csv` (presealed fixed/static/LLM choices), `summary.json` and
hash-bound `manifest.json`. Endpoint manifest SHA256:
`0b19369ef949ccac32988cd4e07d4f33b4d7744fc88b69ed85c05d181f5d2cc2`.
The CPU preflight keeps its original historical prepared-state manifest;
it is not rewritten after submission.

The one previously accepted report change increased independent BioViL-T raw
cosine by **0.312147** relative to the fixed CXRMate report. The unchanged case
had zero delta. Static and Qwen selected the same candidates in **2/2 cases**;
this result does not demonstrate an LLM advantage. Raw cosine is not accuracy,
a probability, calibrated clinical correctness, or proof of clinical repair.
The EHR/image classifier-proxy oppositions remain unresolved. No threshold,
gate, candidate or original selection was changed using the endpoint scores.

Metadata-only CPU postflight:
`TriCompose-v1.2/agent/audit_fresh_report_secondary.py`.
Completed audit:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_endpoint_audits/fresh_report2_biovil_12827754_12817261_001/`.
Audit manifest SHA256:
`efc46a67157f81123baa4a4da7c6b062e5ec2b0dcbb39abe854ea6314c0be0e8`.
Artifact/source/model pins, protected permissions, sealed choices, exact CSVs,
summary arithmetic and encoding counts passed. This audit made zero model
calls and did not parse EHR/report bodies or decode image pixels; it does not
independently rerun the embedding computation or establish clinical truth.
Eight new postflight tests passed; the full suite passed **3,691 tests** in
CPU allocation `12817261`.

The audit's `veto_diagnostics.csv` records four gate-blocked alternatives with
higher independent cosine than the retained report. Higher cosine does not
make a blocked alternative clinically preferable: some were rejected for new
explicit classifier-proxy opposition. One was rejected for no strict discrete-
evidence improvement, showing a disagreement between sparse label evidence
and embedding evidence. Another higher-cosine alternative was static-eligible
but not chosen, so it must not be counted as gate-blocked. These are diagnostic
limitations to investigate under frozen thresholds, not grounds to tune the
gate or retrospectively replace winners on these two development cases.
Any further GPU run still needs its own complete-script/resource review and
explicit approval.

### Prospective one-image action (completed, metadata audited)

`TriCompose-v1.2/agent/run_fresh_cxr_agent.py prepare|run` installs the first
fresh CXR action without editing the consumed report-only controller. Its
pure session is `agent/tricompose_llm/live_cxr_bridge.py`. The two previously
examined development EHRs, original final prompt hashes, generators/scorers,
thresholds and accepted report choices are authenticated from the completed
report-only run and CPU audit above, not rebuilt from endpoint scores.
The source pool's four report observations per image are all historical,
already available evidence. Qwen receives only closed rebased numeric states;
BioViL scores, report/EHR bodies, pixels, hashes, paths and clinical names are
not exposed to the planner.

Only a direct EHR/XRV explicit opposition makes the image action available.
This is a heuristic trigger, not proof of image fault; agreeing CXR-derived
reports are not independent votes. Qwen can request one RoentGen-v2 seed-one
action, stop or abstain. Every request preserves all original inputs/settings
except the seed/request ID. The new image gets XRV, a freshly generated report
from the previously retained expert, and CheXbert: four separately charged
worker phases. In particular, an old report cannot be reused for a new image.
The existing cross-image gate protects both the original fixed baseline and
retained report evidence, with no changes to uncertainty, coverage, quality
or opposition rules. A veto/failed worker retains the old selected triple.
Stop remains unverified, not an automatic claim of clinical consistency.

Each case has a separate four-worker-attempt ledger and at most one durable
policy reservation. Failure remains charged; no retries, format repair,
silent resume or historical ledger-budget enlargement is allowed. Historical
twenty worker and six policy calls stay separate as shared sunk work, not zero
cost. The earlier fixed/static comparisons use observed cached outputs, not
newly executed cost-matched online controls. A frozen rule action is recorded
for comparison; one available tool cannot demonstrate a general agent's
decision advantage, cheaper repair, or learned dynamic stopping.

Completed CPU preflight, zero model calls and no artifact-body parsing:
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_plans/fresh_cxr2_12817261_001/`.
Manifest SHA256:
`807f136342fd0c837962b3b860462df8e08df6a197a0899b6de3b7e5682f1163`.
308 source pins and the original worker/Qwen asset pins passed. The maximum
is two new CXR artifacts, two new reports, eight new worker attempts and two
policy calls. Twenty-seven fixture tests cover numeric privacy, requests,
seed-only changes, own-image report lineage, strict evidence/quality gates,
rollback, every worker-phase failure, policy charging, exact selected exports
and guard-before-read ordering. Full suite: **3,718 tests**, CPU allocation
`12817261`. Fixtures do not certify actual model quality or GPU execution.

Approved script, submitted unchanged after complete-script/resource display
and the user's explicit approval:
`TriCompose-v1.2/agent/slurm/07_fresh_cxr_probe2_debug_a40.sbatch`.
Request: debug partition, one A40, four CPUs, 64G host RAM, thirty-minute
allocation cap. Worker/policy timeouts: 150 seconds each. A40 accommodates
the retained MAIRA-2 expert's existing 40-GiB planning requirement; policy
and generation subprocesses exit sequentially rather than sharing resident
weights. The availability snapshot showed two free debug A40s, not guaranteed
instant scheduling. Job `12834413` initially waited, then completed on
`b11-09`, exit `0:0`, in **4m59s** (driver wall time including load/I/O:
206.509s). Both new own-image/report chains completed: two RoentGen, two XRV,
two retained report experts and two CheXbert calls, plus two local policy
requests. **Zero probes passed replacement**, so both previous selections
remain. This is not a successful clinical repair or evidence of an agent
advantage. No API, downloads, training or scorer revision is included.
Completed output:
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_runs/llm_cxr2_12834413/`.
It includes `selection.json`, `score_rows.json`, raw `score_table.csv`,
`selected_triplets.json` (exact pointers, not clinically proven best outputs),
new-chain triplets, policy audits and separate ledger/cost summaries. A later
independent endpoint must measure the newly sealed choices, without tuning
the gate from their evaluation results. This is still a development interface
check, not clinical localization or a validated CVPR result.
Script SHA256:
`2d9fdb2662a63108abb6390f1eb815edd6e4b2a51c434e95a0708bc046327ab2`.
No resource swap or duplicate job was submitted. Any further GPU script or
different resource request must be shown and explicitly approved separately.

CPU postflight completed in allocation `12817261`:
`TriCompose-v1.2/agent/audit_fresh_cxr_agent.py`.
It authenticates the reviewed plan and completed output manifest, replays
only the historical numeric inventory before each decision, and checks the
new image/XRV/report/CheXbert chain, frozen worker identities, gate decisions,
rollback and durable failure costs. Stored selections and exact exported
artifact pointers must agree; incomplete runs or missing costs are refused.
It byte-hashes synthetic EHR, image and report artifacts without parsing
their clinical bodies or pixels. It adds no model calls or scorer changes.
The protected audit saves `audit.json` and an unweighted `score_comparison.csv`
with reference/selected edge counts, coverage and explicit NA. It does not
independently recompute model outputs or establish clinical correctness.
Sixteen new authored tests passed, and the full suite passed **3,734 tests**
in CPU allocation `12817261` before this endpoint extension. Actual audit:
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_audits/fresh_cxr2_12834413_12817261_001/`.
Manifest: `c77ef7951a3eb2979f89df27200a77816b947a6e08d90ee53ab0c579c7d12fe4`.
Metadata, policy/ledger replay, artifact lineage, raw CSV arithmetic and costs
passed; no body/pixel inspection or model calls. The submitted script and all
308 sealed source pins remain byte-identical.

### Separate V100 image-probe endpoint (completed as `12840340`)

`TriCompose-v1.2/agent/run_fresh_cxr_secondary.py prepare|run` uses the existing
BioViL-T scorer without changing its checkpoint, full-report policy or any
primary extractor/gate. It authenticates the completed image-probe audit and
original plan, then seals all **ten report pairs on four images**. The
historical fixed/static choices, before-probe reference, selected result and
new probe are distinct comparison arms, not choices derived from new cosine
scores. A rejected probe remains measured, not silently dropped or promoted.
Missing/overlength text remains NA in the original denominator. No new EHR,
CXR/report generation, primary scoring, training or external API calls.

CPU plan:
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_endpoint_plans/fresh_cxr2_biovil_12817261_001/`.
Manifest: `914e06ae76e3788523c4910d52b696e1d0bb3a62a63c1b8a8a9f65dbfa11e917`.
Fourteen new authored tests pass; prepare called no model and parsed no
clinical bodies/pixels. Full suite: **3,748 tests**, CPU allocation `12817261`
and workspace-local temporary directory. An initial invocation without that
temporary-directory override failed one unrelated RICORD workspace-path
fixture; rerunning with the established override passed without code changes.
Approved exact script:
`TriCompose-v1.2/agent/slurm/08_fresh_cxr_secondary_v100.sbatch`.
Resources: gpu partition, one V100, two CPUs, 8G RAM, five-minute allocation
cap; scorer process timeout 240s. Model residency is only BioViL-T, not the
generator/MAIRA-2/Qwen chain. The complete exact script/resources were displayed,
and the user's subsequent `go` approved submission. Submitted unchanged as
`12840340`; completed on V100 node `d13-10`, exit `0:0`, in **25 seconds**.
The scorer driver used 19.447s including load/I/O; peak Torch allocated VRAM
was 0.583 GiB (not total GPU use). All **10/10** pairs available, four image
and ten report encodings, no generation or primary scoring calls. Actual output:
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_endpoints/fresh_cxr2_biovil_12840340/`,
including `score_table.csv` and `case_comparison.csv`.
Manifest: `035657ccb53c6a2937d32ab05ce33b07104b74ce5c1db94e0e20570a73c2f9e8`.
Script SHA256: `50d4f08e5a339b589588c60aa08a2ee3cc4f0449879c65bcf275d246ffb60aef`.
CPU metadata verification in allocation `12817261` checked reviewed plan
identity, 340 source pins/eight model-asset pins, all output hashes, durable
one-attempt reservation, encoding inventory, source selections, exact CSV and
summary reconstruction. It did not independently rerun embedding computation.

| Synthetic case | Before reference | New image/report probe | Probe minus reference | Final selected |
| --- | ---: | ---: | ---: | ---: |
| case_009 | -0.1001293585 | 0.7779716253 | +0.8781009838 | -0.1001293585 |
| case_018 | 0.2352395505 | 0.6336349845 | +0.3983954340 | 0.2352395505 |

These are raw BioViL-T cosine scores, not probabilities or clinical accuracy.
Both probes score higher, but **neither was accepted**. The sealed gate for
case 009 reports `no_strict_fixed_ehr_image_evidence_gain` and
`official_report_structure_failed_or_common_risk_worse`. Case 018 additionally
reports `lost_supported_or_comparable_fact_ids` and
`new_explicit_proxy_opposition` (one new opposition, zero resolved).
Endpoint scores were not available to the planner and do not retrospectively
relax the gate or change final selections. Higher cosine therefore motivates
examining cross-edge/gate validity, not claiming successful EHR-grounded repair.
Independent cosine measurement is still secondary development evidence,
not clinically adjudicated truth, held-out repair efficacy or LLM superiority.

Read-only Slurm `--test-only` checks of the same resource shape on gpu A100
and L40S predicted starts on 2026-10-11, compared with the existing debug A40
job's 2026-10-08 prediction. No alternative job was created; test-only IDs are
not submitted experiment IDs. These predictions can change and do not certify
GPU memory compatibility. Visible idle/planned cards alone do not establish
that an eligible thirty-minute scheduling window exists. The approved A40
job was left unchanged; any real alternative submission still needs approval.

### Next experiment, not another scorer revision

Freeze the current evidence extractors. Compare fixed path, static reranking,
rule-based probes, LLM candidate selection, and LLM-directed probes on the
same predeclared cases/cost caps. First establish that the model follows the
schema and reacts to completed action feedback in cache replay. The first
fresh report-worker interface and separate independent embedding endpoint are
now attached and audited above. The separate one-CXR action and its independent
embedding endpoint have completed; zero probes passed the frozen gate. Diagnose
evidence/gate disagreement with thresholds frozen, and compare against the
same fixed/static/rule controls under prospectively matched budgets. Only that later
experiment can measure clinical regeneration benefit. Cached proxy gains
and a self-assessing LLM cannot establish clinical repair or a CVPR contribution.

### Decomposed veto and completed blind image observer (2026-10-08)

Completed diagnostic tool `TriCompose-v1.2/agent/diagnose_fresh_cxr_probe.py`
replays the actual cross-image comparison, separates every quality subcheck
and authenticates the original classifier metadata. Protected result:
`artifacts/protected/tricompose_v1_2/llm_probe_diagnostics/fresh_cxr2_12840340_12817261_001/`.
Manifest `5ad0624520486ff48dafa6987b03c1f2cf33c7653c0e511b91fc6f7f2cc9d7ea`.
Zero model calls and no clinical bodies or image pixels opened. Initial
metadata parsing exposed the legacy collection key `records`; it was corrected
and fixture-tested before any diagnostic output was committed.

Case 009's format/sections and common-risk flags do not worsen. The precise
quality blocker is four-gram repetition 0 -> 0.00833333, not a failed format.
Case 018 passes quality but loses support identity and gains one opposition.
Both have **one known EHR finding**, XRV negative before/after and no strict
EHR/image gain. Operating-point-normalized scores are 0.18525863 -> 0.11036517
and 0.06479656 -> 0.19761756; threshold 0.55457607. These are not probabilities.
The positive/negative cutoffs coincide, so the head has no nonzero uncertainty
interval; a negative bin is not clinically confident absence. Case 018's
EHR/report support changes 0 -> 1 while XRV stays negative. This does not
localize a clinical fault. Silence by lost comparability is counted separately
from resolved facts. No scorer, threshold, prompt, EHR, gate or winner changed.

Submitted unchanged as **job `12843415`** after complete-script display and
explicit approval; completed on V100 node `d14-06`, exit `0:0`, Slurm elapsed
**2m38s**:
`TriCompose-v1.2/agent/verify_fresh_probe_images.py prepare|run`.
All four existing CXRs, both seeds on both cases, no outcome filtering.
Original frozen image-only Qwen prompt, eight findings, seed zero, 384-token
cap and pixel limits. Only pixels and the generic prompt reach the model.
Responses and call costs are durable; predictions are fsynced before cached
label comparison. Invalid/truncated responses are unavailable, never negative.
Zero retries/resume; model-load and call failures remain charged. Comparison
has 32 unique image/finding slots, not repeated independent report votes.
This observer is not yet an installed action in the consumed numeric planner.

CPU plan `artifacts/protected/tricompose_v1_2/llm_probe_image_plans/probe_images4_12817261_001/`.
Manifest `4913be6d356954b46acbaabd6415e0f7f4aa390d727fb9fcebf036f1451a114f`.
Approved `TriCompose-v1.2/agent/slurm/09_probe_images4_qwen_v100.sbatch`:
one V100, two CPUs, 32G host RAM, five-minute cap. The existing loader uses
FP16 on non-BF16 GPUs and requires at least 24 GiB device memory. CARC's
Discovery specifications list 32GB V100s:
https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery
The four-image GPU smoke completed with **4/4 complete responses**. Script,
observer program and prepared plan remain pinned.
Thirteen new diagnostic and ten observer fixture tests pass; full suite
**3,771 tests**, CPU allocation `12817261`. Consumed original files stay fixed.
Qwen shares its checkpoint with numeric planning; it is not a radiologist,
qualified primary scorer or clinically independent gold standard.

Protected result:
`artifacts/protected/tricompose_v1_2/llm_probe_image_observers/probe_images4_12843415/`.
Manifest `a55e210f3c489b1dd9d7d691b885a6d979c56eac68a586a9e659c0aa041f6f8b`.
Observation worker runtime 43.173s, peak Torch-allocated VRAM 15.789 GiB
(not total device memory); charged four observer calls and one model-load
attempt. No generation, retries, training, gate or selection changes.
An inline numeric postflight in CPU allocation `12827441` authenticated all
sealed plan/program/artifact/model pins and protected permissions, replayed
all nine journal events and reproduced the exact 32-slot comparison CSV.
No clinical bodies or image pixels were opened by that CPU postflight.

The four known-EHR image/finding slots have **three Qwen supports and one
uncertain observation**, versus four XRV oppositions. Across the full 32-slot
image/finding grid, Qwen/XRV relations are 23 explicit agreements, eight
oppositions and one uncertainty-not-comparable. These are two previously
examined development anchors with two seeds, not four independent patients,
clinical accuracy, validated error localization or evidence of LLM superiority.
The result supports treating scorer disagreement as unresolved; it does not
authorize overriding the frozen primary gate or declaring Qwen correct.

### Optional image-evidence veto and counterfactual preview (2026-10-08)

Pure interface `TriCompose-v1.2/agent/tricompose_llm/image_evidence_guard.py`
adds an optional prospective policy layer, without changing the consumed
controller/session/public-state version:

- `build_guard`: one named state comparison per image/finding, not per report.
- `guarded_state`: remove only image-regeneration tools when attribution is
  unresolved; retain report tools, original scores, budget and charged history.
- `guarded_decision`: conservatively withhold a valid image request or unresolved
  stop as abstain; preserve the raw proposal. Invalid decisions still fail closed.

The guard counts only direct known EHR constraints for its veto. Unknown and
uncertain never imply absence; observer failure is separately unavailable.
Unmentioned findings do not become EHR constraints, report agreement is not an
independent vote, and no similarity value/threshold enters this policy. Scorer
agreement remains unvalidated and does not authorize clinical acceptance or
bypass the original gate. This development policy was informed by the existing
two-case diagnostic, not evaluated as a held-out efficacy method.
Private upstream provenance must bind the observer to the same image and fixed
EHR. The closed numeric projection contains no clinical text, finding names,
artifact paths or hashes. The adapter also checks rebased current-candidate and
evidence IDs and unchanged primary EHR/image count algebra; these checks alone
are not cryptographic source authentication. No API call was made.

CPU builder `TriCompose-v1.2/agent/build_image_disagreement_preview.py` completed
using allocation `12827441`:

```text
artifacts/protected/tricompose_v1_2/llm_image_evidence_guards/image_guard4_12827441_001/
  image_guards.json
  image_guard_table.csv
  decision_preview.json
  summary.json
  manifest.json
```

Manifest `e899bf7127fc1123a9015a1ebbe64eaee61aa52d9a13ddf78fd61a7204276df9`.
All four images remain represented: three known-EHR scorer disagreements and
one missing/equivocal comparison. Replay uses the **two actual historical
numeric planning states and proposals**, not fabricated seed-two requests for
the rejected probes. Both raw image-regeneration requests become abstain in
this counterfactual preview. The saved original evidence/counts, charged budget,
history, EHRs, prompts, artifacts and winners remain unchanged.

Zero new model calls/submissions/training. Existing source cost remains eight
generation/verifier attempts, two numeric policy requests and four blind-image
observer calls plus one load attempt. This reuse does not erase historical cost;
measured saved calls are NA. The result demonstrates mechanical veto behavior,
not prospective cost savings, validated error localization, clinical repair or
LLM superiority. The optional guard is **not yet installed in the consumed live
runner**, and Qwen still shares its checkpoint with numeric planning.

Twenty new authored-fixture tests pass, including exhaustive four-state
combinations, unavailable-vs-unknown, immutable evidence/costs, closed numeric
projection, unchanged report actions and invalid-decision rejection. Full suite
**3,791 tests in 35.261s**. Inline postflight authenticated the new manifest and
all source/artifact pins, replayed four guard projections and two historical
decisions, reproduced the exact CSV columns/bytes and verified modes 2770/0660
and project group access. No model factory or clinical body/pixel reader ran.

### Guard before backend construction: completed and audited (2026-10-08)

`TriCompose-v1.2/agent/tricompose_llm/guarded_action_dispatch.py` installs the
optional veto at an actual lazy-backend execution boundary. It fsyncs the guard
decision and reserves a dispatch before constructing an authorized backend.
Unresolved image evidence abstains without calling the factory; no installed
backend yields an explicit pending request. A failed reservation/backend cannot
be replayed. One dispatch is not one worker model call: the worker ledger and
original acceptance gate remain mandatory. This does not mutate archived
controllers or make scorer agreement a clinical verdict.

`TriCompose-v1.2/agent/run_guarded_image_probe.py` is the new bounded entry.
Its completed GPU run performed **two fresh frozen image-only observations** on
the original fixed seed-zero reference images, seals predictions, then applies
the dispatcher to the two authenticated cached numeric CXR-regeneration
intentions. It does not select inputs using previously observed image results.
The new observations receive no EHR, report, scores or IDs. The numeric intents
are explicitly cached: new numeric planning calls are zero. Generation and
primary scoring backends are not installed, and unblocked requests require a
separate complete batch script and explicit approval. No outputs or historical
selections can be silently replaced.

Sixteen added fixture tests cover lazy-factory veto, durable reservation failure,
no retry, unavailable versus unknown observations, fixed reference/intent
binding and GPU/CPU guards before artifact reads. Full suite: **3,807 tests in
33.853s**, passing. Preparation is CPU metadata/hash work under allocation
`12827441`; GPU execution was subsequently submitted as **job `12848110`** after
the full script was displayed and the user explicitly approved it. It completed
on d14-16 in **2m22s**, exit 0, with **2/2 complete responses** and two actual
guarded abstentions for unresolved scorer disagreement. No generation backend
was constructed: new backend attempts, fresh numeric-planner calls, generation,
primary scoring and pending generation requests are all zero. The original
choices and charged history remain intact. This limited development smoke
tests forward execution ordering, not held-out localization, successful clinical
repair, measured savings, complete fresh-pipeline execution or LLM superiority.

Prepared plan:
`artifacts/protected/tricompose_v1_2/guarded_image_execution_plans/guarded_probe2_12827441_001/`
with manifest SHA256
`71340e7a74c0407e81550e0c73e23429530498b694111088f1c8d23e3966e18c`.
Executed script:
`TriCompose-v1.2/agent/slurm/10_guarded_image_probe2_v100.sbatch`.
Resources: gpu partition, one V100, two CPUs, 32G host RAM, five-minute wall cap;
worker timeout 240s with 15s kill grace. The observer requires at least 24 GiB,
so the available 16-GiB P100 cards are intentionally excluded. Scheduler
availability is transient and does not guarantee an immediate start. The script
was submitted unchanged (SHA256
`7e85cb1d160b9153fd401a2ec4adf73acbedf5a36768585a0b0563a271a5eee4`);
execution outputs use a new job-ID-bound run rather than overwrite any prior
run. Slurm initially reported pending/Priority, with no predicted start. The
V100 node group was reserved until 18:00 on 2026-10-08 (America/Los_Angeles).
Expiry did not guarantee allocation, but the original job started when resources
became available. A subsequently approved pending-only cancellation request did
not affect the now-running job. No resource swap or duplicate was submitted.

New CPU postflight `TriCompose-v1.2/agent/audit_guarded_image_probe.py` authenticates
the completed run and replays its two charged observer calls and one
load reservation, prediction-sealing event, actual guard dispatch and explicitly
pending requests. A failed/unavailable response remains charged, not negative.
The audit rejects altered image/order bindings, invented savings, hidden new
generator/planner calls, automatic-submission authority, clinical-acceptance
claims and changed historical selections. This metadata replay does not rerun
image inference or validate clinical accuracy. It does not modify the consumed
script, worker, plan or original model outputs.
Nine added postflight fixture tests pass. Full regression: **3,816 tests in
41.046s**, passing; no GPU/model inference was performed by that test run.

Completed protected run:
`artifacts/protected/tricompose_v1_2/guarded_image_execution_runs/guarded_probe2_12848110/`
(manifest `ea05ee34f65e03e0fd823bc482767aa73acdc1677eba691c5429c6d8cec2d70c`).
Internal worker interval through summary creation: 34.394s; peak Torch-allocated
VRAM 15.789 GiB, not total device memory. Both actual intents became guarded
abstentions. Completed CPU audit under allocation 12827441:
`artifacts/protected/tricompose_v1_2/guarded_image_execution_audits/guarded_probe2_12848110_12827441_001/`
(manifest `49c235caa6576a9f95f8b10062b1bb142c9df550a70e8bbdc38ddeb2e5ad695f`).
Exact observer/dispatch journal replay, original image/EHR/state bindings and
protected project-group permissions passed. The audit adds zero model calls.
This is not a clinically adjudicated error localization, successful repair,
held-out efficacy result, fresh LLM-planning demonstration or measured cost
saving. Scorer disagreement remains unresolved; saved calls and accuracy are NA.

Resource-swap proposal, conditionally approved but not submitted:
`TriCompose-v1.2/agent/slurm/11_guarded_image_probe2_debug_a40.sbatch`.
Complete request: debug, one A40, two CPUs, 32G host RAM, five minutes. The wrapper
verifies the exact previously displayed body SHA256 before executing it with
Bash, without any nested sbatch or extra allocation. Its own SHA256 is
`c103126e45f959c4d2c943deae3ef17b2b9420966f96e584633e5b0f8933a6ae`.
A non-submitting scheduling dry-run predicted 17:36:32 on b11-09 on 2026-10-08;
availability could change. After the complete wrapper was displayed, the user
approved replacing the original **only if still pending**. The state-filtered
cancellation did not affect the original running job; it was retained to
completion and no A40 job was submitted. The loader's existing automatic
BF16/FP16 choice depends on hardware, so an A40 run would not be a strict
precision-equivalence comparison with archived V100 observations.

### Fresh guarded decisions on existing probe branches: completed (2026-10-08)

The original reference images had no legal image action after the guard, and
their four report experts were already observed. Calling an LLM on that empty
menu would not demonstrate expert selection. The new entry
`TriCompose-v1.2/agent/run_guarded_fresh_policy.py` instead uses both previously
completed seed-one probe branches, each with one observed report and **three
untried report experts in the sealed inventory**. Original EHRs and selected
seed-zero triples stay fixed. The probes remain diagnostic candidates, not
accepted replacements; this is a two-case development planning smoke.

`tricompose_llm/guard_aware_qwen.py` makes genuine new frozen `model.generate`
calls using typed text blocks and a closed numeric-only packet. Four previous
expert readouts belong to another image of the same EHR; explicit image groups
prevent presenting them as current-image reports or unseen report scores.
They are weak scheduling context, not independent votes. Cached blind image
observations supply the disagreement guard; no image observer runs again.
The model receives no clinical text, images, artifact paths/hashes or external
service credentials. It can request an untried expert, stop or abstain. The
execution-boundary guard runs again on the validated decision.

The new-phase budget reserves one policy unit before each actual call and
allows two units for a potential report-plus-CheXbert request. Preparation has
not incurred those calls or charges. Historical observer/generator/planner
costs remain recorded as shared sunk cost. Maximum new policy calls: two;
new generation, primary scoring and backend dispatches: zero. No backend is
installed, so any report request is written as pending and requires another
complete reviewed script plus explicit approval. Failed attempts remain charged,
with no fallback, automatic retry, submission or winner change. A validated
decision is not clinical acceptance or evidence of successful repair.

CPU-prepared plan under allocation 12827441:
`artifacts/protected/tricompose_v1_2/guarded_fresh_policy_plans/fresh_policy2_12827441_001/`
with manifest `f575d3be72e16242b50e711d187dc9be9db59f929d380bb65e2dd771731a46ae`.
Metadata postflight verified 314 source pins, 101 artifact pins, two bound
packets, exactly three report requests per case and protected modes 2770/0660.
Fifteen new fixture tests pass; full suite **3,831 tests in 42.276s**.
No model inference occurred during preparation or tests.

Reviewed batch script:
`TriCompose-v1.2/agent/slurm/12_guarded_fresh_policy2_debug_a40.sbatch`.
Resources: debug, one A40, two CPUs, 32G host RAM, five minutes; worker timeout
240s plus 15s termination grace. The existing loader requires at least 24 GiB
and uses hardware-dependent BF16/FP16, so this does not establish numerical
equivalence with archived V100 observations. After complete-script display and
fresh explicit user approval, the unchanged script was submitted as **job
12849550**. It completed on debug A40 node b11-09 in **1m01s**, exit 0. Two
genuine new planner calls returned valid decisions with zero failures; the two
branches respectively requested **MAIRA-2** and **CXRMate-single**. Both requests
were deferred for separate approval. Actual image/report generation, primary
scoring, observer calls, backend dispatches and original-winner changes are zero.
No clinical efficacy, localization accuracy, saving or held-out advantage is
claimed.
The complete script's SHA256 is
`a6b9aab5c83dda2f1cacbedb46d005962bdf08653a14cd09cd28a818d759cdf4`.
A read-only `sbatch --test-only` predicted a start at 18:45:40 on b11-09 on
2026-10-08 (America/Los_Angeles); it created no submitted job. This changeable
estimate does not guarantee allocation. Shell syntax and diff-whitespace checks
passed. Only the subsequent explicitly approved submission is an actual job;
the test-only scheduler identifier is not a submitted job. Report generation
still requires a separate reviewed script and explicit approval.

Completed protected output:
`artifacts/protected/tricompose_v1_2/guarded_fresh_policy_runs/fresh_policy2_12849550/`
(manifest `189c9cf86afdbb16e86287b2b51390a2ad94f52eacc57192a6b20da196de45bb`).
Worker interval through summary creation: 31.257s; peak Torch-allocated VRAM:
15.978 GiB. These are not total batch duration or device-memory usage. Token
usage: 5,021 input and 156 output tokens. Both policy calls and one model-load
attempt are charged. No failure, retry, fallback, training or external API.

CPU metadata postflight under allocation 12827441 authenticated plan and output
pins, replayed all nine journal events and both guarded deferred requests,
verified current image and fixed EHR bindings, exact pending-request records,
token/call accounting and private project-group modes. Audit:
`artifacts/protected/tricompose_v1_2/guarded_fresh_policy_audits/fresh_policy2_12849550_12827441_001/`
(manifest `6337ab9a18629ce9725d699ef98136fb8dafd060cb87bb07bff1f4a7ceb04e70`).
The replay invokes no models and does not certify clinical truth. **Both LLM
decisions cite historical other-image evidence, not the current-image receipt.**
Those citations can be weak expert-scheduling context but cannot substantiate
current-report fault attribution. Successful parsing/dispatch is not evidence
that either requested expert improves the current report. The separately
reviewed and approved execution and unchanged same-image comparisons below
rejected both replacements. Do not claim clinical repair, promote probe
triples or erase the prior cost.

### Actual report requests: completed, both replacements rejected (2026-10-08)

`TriCompose-v1.2/agent/run_guarded_requested_reports.py` binds the two valid
pending expert requests to the original fixed synthetic EHRs, existing seed-one
CXR bytes and cached frozen XRV receipts. MAIRA-2 and CXRMate-single use the
unchanged deployed adapters; each newly generated report received one
unchanged CheXbert call. No Qwen planner, image observer, new XRV call, image
generation, external service, threshold update or training is installed.

The old ledgers each exhausted a four-attempt budget; changing that budget
would change their hash-chain contract. Instead `RequestedReportCalls` has a
separate **two-attempt new-phase budget** per case and authenticates cached
parent dependencies separately. It fsyncs a reservation before a backend can
be constructed, validates the generated report before reserving CheXbert, and
retains failed or in-flight charges without retry. Old ledgers/costs are not
rewritten or reported as free. New subprocesses and logs are protected and
local-only; their environments/checkpoints stay read-only.

Paired comparisons use the existing same-image expert gate: preserve supported
and comparable facts, prohibit new proxy opposition and worsening common report
risks, and require a strict gain. Unknown is not negative/agreement; removing
a conflict by losing its comparison is not repair. A permitted diagnostic
branch change is not clinical acceptance and cannot promote the previously
rejected seed-one CXR or replace the original retained triple. Both LLM reasons
remain historical-image citations, not evidence of current-report fault.

Prepared plan:
`artifacts/protected/tricompose_v1_2/guarded_report_execution_plans/guarded_report2_12827441_001/`
with manifest `9eee28509cd594cc5449d3e643b2194ed23da734a0109317a741682b20274f17`.
Metadata postflight confirmed both requests, fixed image bindings, unchanged
exhausted old ledgers, 316 source pins, 117 artifact pins and private modes.
Preparation invoked zero models. Fifteen new fixture tests cover charged
failures, durable pre-factory reservation, no replay, exact model/image/report
bindings, strict proxy gain, unknown/conflict-silencing rejection and CPU/GPU
guards. Full suite: **3,846 tests in 42.255s**, passing, using the existing
report-context Python with pinned DICOM dependency paths preloaded and
workspace-local temporary directories. Initial incomplete test environments
were corrected by selecting existing dependencies, not installing/editing them.

Reviewed and executed complete script:
`TriCompose-v1.2/agent/slurm/13_guarded_requested_reports2_debug_a40.sbatch`;
SHA256 `5cfd6804f8720b14b6b41e0891a65eeea89159fa188f326c49a78a9366f26e85`.
Resources: debug, one A40, two CPUs, 32G host RAM, eight-minute cap; coordinator
timeout 420s plus 15s termination grace. Per-report timeout 120s, per-CheXbert
45s. Planning minimum 40 GiB for MAIRA-2; inference does not imply every model
fits a 16-GiB card. Prior analogous frozen A40 worker calls took approximately
63s/9s for MAIRA-2/CXRMate-single and 8s per CheXbert, not total batch time.
Shell syntax and diff checks passed. A read-only scheduling dry-run predicted
19:10:26 on b11-09 on 2026-10-08, without submitting a job or guaranteeing a
start. After the complete script was shown and the user explicitly approved it,
the unchanged script was submitted as **job 12850244** and completed on debug
A40 node b11-09 in **2m02s**, exit 0. Both reports and both CheXbert calls
completed: **four charged new worker attempts, zero failures and zero accepted
branch changes**. No new planner/image observer/XRV/CXR calls, training,
automatic retry, threshold changes or original-winner replacement occurred.
Worker interval through summary creation was 96.929s, not total Slurm duration.
The MAIRA-2 and CXRMate-single subprocesses took 54.554s and 7.742s; their
CheXbert subprocesses took 7.741s and 6.695s, including startup/I/O.

Completed protected output:
`artifacts/protected/tricompose_v1_2/guarded_report_execution_runs/guarded_report2_12850244/`
(manifest `677482fd514cd4d788b5e6904ead00cc309c9388b98c1d39e2dd8bef079ab6e8`).
`score_table.csv` contains the two baseline and two new same-image rows;
`paired_report_comparison.json` records each veto. New `completed_triplets.json`
holds diagnostic alternatives only, not new winners. Raw readouts:

| Requested change | EHR–Report support/known | CXR–Report support/known | CXR–Report coverage | CXR–Report proxy opposition | Branch change |
| --- | --- | --- | --- | --- | --- |
| CXRMate-single → MAIRA-2 | 0/1 → 0/1 | 4/8 → 2/8 | 4/8 → 2/8 | 0 → 0 | Rejected |
| MAIRA-2 → CXRMate-single | 1/1 → 0/1 | 2/8 → 4/8 | 3/8 → 4/8 | 1 → 0 | Rejected |

Support is positive plus negative support; **every supported image/report label
in this comparison is negative**. These are proxy-label counts, not clinical
accuracy. EHR–CXR remains 0/1 supported with one proxy opposition per branch.
The first expert switch improves the temporal-language flag but loses image
comparability. The second gains image-label support but loses EHR support and
comparability, silences a prior comparison, and introduces unsupported
temporal-comparison language. Both meet their model-specific section contracts;
that does not certify factuality. Unknown remains unknown. The unchanged gate
retains both prior branch reports. This demonstrates actual LLM-requested
generation followed by a veto, **not clinically successful self-correction**.

CPU postflight under allocation 12827441 exactly replayed all 12 new-phase
journal events, model requests/results, report/receipt hashes, fixed EHR/image/
seed bindings, unchanged source ledgers, four charges and shared sunk costs. It
authenticated 316 source pins, 117 artifact pins and the byte-exact score CSV.
A preliminary text-mode CSV assertion differed only by CRLF normalization; no
artifact contents were changed. Four CUDA cache directories initially used
2700 and were changed to project-group 2770; all protected run modes then passed
2770/0660 checks without content/group changes. Audit:
`artifacts/protected/tricompose_v1_2/guarded_report_execution_audits/guarded_report2_12850244_12827441_001/`
(manifest `ddf8f0f69092f38a4e0eb6b42a27fff11199f07ca1ce664844532bd7185be8bd`),
including `paired_scores.csv`, with zero model calls. No further GPU job was
submitted. Clinical truth, error-localization accuracy, independent held-out
advantage and compute savings remain unestablished. Both planner decisions
cited other-image history; future evaluation needs current-image-grounded
planning and cost-matched controls under a separately frozen endpoint, not
relaxed gates or promotion of previously rejected probe triples.

### Continue current outputs: secondary BioViL-T evaluation completed (2026-10-08)

`TriCompose-v1.2/agent/run_guarded_report_secondary.py` reuses the unchanged
frozen BioViL-T scorer for four actual reports (two old/new pairs) on the two
fixed seed-one images. This is endpoint evaluation, not another LLM call or
scorer redesign. Both expert-switch decisions are sealed first; BioViL cannot
revise either veto or replace an original retained triple. Full report inputs
remain local/protected; overlength/empty inputs emit NA with reasons and stay
in the requested denominator. No new generator, primary scorer, training,
threshold update, API call or automatic retry is permitted by this job.

Prepared protected plan:
`artifacts/protected/tricompose_v1_2/guarded_report_endpoint_plans/guarded_report2_biovil_12827441_001/`
(manifest `f7d221aaab9d02054c083930cf6bb7e94e3774089ba645d9b7fac08241f4f431`).
CPU preparation and postflight verify source/audit manifests, fixed EHR/image
bindings, gate decisions, source/artifact/model hashes, prior costs and
2770/0660 project-group permissions. No clinical body, pixels or model were
loaded. Fifteen new tests and **all 3,861 tests in 42.290s** pass; fixtures
include a rejected candidate with a higher secondary score, an accepted proxy
candidate with a lower score, explicit missingness and guarded CPU/GPU entry.

Reviewed and executed complete script:
`TriCompose-v1.2/agent/slurm/14_guarded_report_secondary_debug_p100.sbatch`
(SHA256 `8208914b964bfda4632020d47e921146b7415f822ebbed36cfe921900b77f504`).
Resources: debug, one P100, two CPUs, 8G host RAM, five-minute cap, 240s worker
timeout plus 15s grace. The deployed metrics environment reports PyTorch
2.6.0+cu126 compiled with `sm_60`; this was read under the CPU allocation,
without CUDA inference. Preparation-time `noderes -f -g` showed a debug P100
available; that is not an allocation/start-time guarantee. No A40/A100, new
model download or API connection was requested. After complete-script/resource
display and explicit approval, the unchanged script was submitted as **job
12850816**. It started one second after submission on debug P100 node e23-02
and completed in **28s**, exit 0. There was no retry or duplicate job.
Completed outputs include secondary raw cosine scores, a same-image
old/requested/retained comparison, charged attempt and manifest.

Protected endpoint:
`artifacts/protected/tricompose_v1_2/guarded_report_endpoints/guarded_report2_biovil_12850816/`
(manifest `8f239d53dd12abccaa9f9495653b5a5934826a856d25d7409666015a2d9ca95a`).
All **four image/report pairs** were available; the worker made two image
encodings and four text encodings. One endpoint attempt is charged separately
from source generation/planner costs. Worker runtime including load/I/O:
21.798s; peak Torch-allocated VRAM: 0.579 GiB. These are not total batch time
or total device-memory use. Raw full-report BioViL-T cosine:

| Proposed switch | Existing | Requested | Requested minus existing | Retained minus existing |
| --- | --- | --- | --- | --- |
| CXRMate-single → MAIRA-2 | 0.777972 | 0.583549 | -0.194422 | 0.000000 |
| MAIRA-2 → CXRMate-single | 0.633635 | 0.715938 | +0.082304 | 0.000000 |

These are uncalibrated image/text cosines, not clinical accuracy or EHR
consistency. The second requested alternative improves this secondary score,
but loses EHR support/comparability and adds temporal-language risk under the
unchanged primary gate. It is therefore still vetoed. The other alternative
worsens the cosine. **Neither accepted result changed**. No successful clinical
repair, LLM advantage or measured compute saving is established; this is a
two-case development evaluation, not a held-out clinical benchmark.

CPU metadata postflight replays the exact frozen choices, all four score rows
and both CSV tables; it authenticates source/artifact/model hashes, encoding
counts, one charged endpoint attempt and protected output/cache/log modes.
No clinical body or pixel reads and zero new model calls. Protected audit:
`artifacts/protected/tricompose_v1_2/guarded_report_endpoint_audits/guarded_report2_biovil_12850816_12827441_001/`
(manifest `59851fa3c7418a5b4462a1debfc5fabde951a877632fbf3bbf9d4f64b15d145e`),
with `audit.json` and `case_comparison.csv`. No further GPU
task was submitted. The future stronger-API-planner experiment remains a
documented hypothesis; the current local Qwen interface and guards are intact.

## Legacy single-report policy connection

TriCompose can call any OpenAI-compatible `chat/completions` endpoint as a
routing policy. This is a lightweight API request; it does not run a model on
the CARC login node.

The client sends only an allowlisted state:

- opaque candidate IDs;
- Qwen candidate scores;
- peer-agreement metrics;
- selection thresholds;
- allowed action names.

It never sends EHR rows, images, report text, patient identifiers, artifact
paths, or hashes. Start with the built-in patient-free synthetic demo.

Configure the endpoint without placing a secret in shell history:

```bash
export TRICOMPOSE_LLM_BASE_URL='https://YOUR_ENDPOINT/v1'
export TRICOMPOSE_LLM_MODEL='YOUR_MODEL_NAME'
read -rsp 'API key: ' TRICOMPOSE_LLM_API_KEY
export TRICOMPOSE_LLM_API_KEY
```

For a local or otherwise unauthenticated OpenAI-compatible endpoint, leave
`TRICOMPOSE_LLM_API_KEY` unset. The model server itself must already be running
on an approved compute service; do not launch local inference on a login node.

Run a synthetic connection test with a new opaque output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  -m tricompose.agent.openai_compatible_policy \
  --demo-state \
  --output-dir artifacts/protected/agent_demos/demo_001
```

After the synthetic test succeeds, an explicitly approved run may use the
protected numeric state from the deterministic selector:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  -m tricompose.agent.openai_compatible_policy \
  --agent-decision artifacts/protected/real_smoke_003/agent/report_selection_v2/agent_decision.json \
  --output-dir artifacts/protected/real_smoke_003/agent/llm_policy_v1
```

The LLM proposes an action, but a local hard guard validates it. In particular,
`select` is overridden to `verify_more` unless the deterministic minimum-score
and margin gates pass and the selected candidate is the eligible top
candidate. This keeps the LLM useful for orchestration without making it the
sole clinical-consistency authority. The protected result follows
`schemas/llm_agent_decision_v1.schema.json`.
