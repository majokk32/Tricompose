# TriCompose — current research README

**Status: 2026-11-05.** This is the current project overview for collaborators and
advisers. It distinguishes working software, completed *development* experiments,
and claims that still need independent validation. The older [root README](README.md)
and version READMEs preserve historical context; do not infer present status from
their original timestamps.

> **Research question.** Can a training-free controller compose frozen EHR,
> text-to-CXR, and CXR-to-report generators, use cross-modal evidence to select or
> repair candidates, and produce more coherent *fully synthetic* triples at lower
> **measured** cost than strong fixed, random, and static-search controls?

The output is `(synthetic structured EHR, synthetic CXR, synthetic radiology
report)`. No new generator, scorer, or router is trained. The current CXR
generators consume **EHR-derived radiology text**, not native structured EHR.
Consequently the demonstrated cold-start route is **EHR → deterministic text
bridge → CXR**, not a direct structured-EHR-conditioned CXR model. EHRXDiff
requires a previous image and is a separate longitudinal route. The current
`CXRMate-single` report worker receives a single CXR; it is **not** CXRMate-ED
and does not establish an independent EHR+CXR report path.

中文一句话：**三模态生成和候选池已经跑通；目前最需要突破的是有独立依据的择优与
错误定位，而不是再堆模型或把代理分数叫作临床正确率。**

## Results to look at first — with the denominator attached

| Experiment / endpoint | Completed aggregate result | What it does **not** show |
| --- | --- | --- |
| Frozen BioViL-T, real matched vs random cross-patient image–report retrieval | Validation/test AUROC **0.934 / 0.940** | Retrieval separation, not clinical correctness or same-image report selection. |
| BioViL-T, same-view and shared-positive-finding *report-label-defined* hard negatives | Validation/test AUROC **0.931 / 0.886**; matched-score wins **96.2% / 90.5%** | Donor conflicts came from report-derived labels, not independent image truth. |
| Three fixed synthetic EHRs, 16 complete candidates per EHR | Proxy three-modal positive support: operational fixed path **2/3**, existing static selector **3/3**; all 16 individual fixed paths at most **2/3** | Tiny, development, selector-related proxy endpoint; not clinical superiority or generalization. |
| Historical 80-EHR full-cap cached bank, same acquired inventory | Scored-final BioViL mean **0.6122** vs genuinely score-free random-final **0.5082**; 63/80 EHR means higher, 17/80 lower | Secondary similarity on cached candidates; **simulated** calls, not online savings. At caps 8 and 12, score-free final was better on this endpoint. |
| RICORD-1C same 50 *real* images, opacity-only report mention | CheXagent-2 explicitly supported **23/25** positive references; MAIRA-2 **17/25**. Neither explicitly supported any of 25 negative references. Different report states on **12/50** images. | Development cohort; one finding only. Unknown is **not** a correct negative. Hindsight either-report coverage is not an online selector. |
| RadEval same-image expert-error selection: 44 complete 3-report anchors / 34 patient groups | Mean significant errors: random **4.8182**, BioViL choice **4.5682**, Qwen2.5-VL choice **5.0644**, hindsight expert oracle **3.2045**. BioViL–random = **−0.2500**, patient-cluster 95% CI **[−0.7647, 0.1062]**. | Neither frozen score has demonstrated a significant reference-free report-selection gain; expert counts are relative to a reference report, not independently adjudicated image truth. |
| Strict finite-cache repair headroom, 80 fixed EHRs | **13/80** have a same-image report alternative passing the proxy-preservation gate; **0/80** have a passing cross-image alternative in the existing bank. | Not proof that image repair is impossible with new seeds/models, and not clinical repair success. |

Read the underlying aggregate-only receipts: [retrieval gate](docs/fresh500_report_cxr_scorer_gate_20261015.md),
[three-case static comparison](docs/new_case_static_comparison_result.md),
[score-free control](docs/score_free_random_control_result.md),
[50-image report comparison](docs/ricord_two_report_opacity50_result_13744455.md),
[same-image selector gate](docs/report_selector_evidence_gate_20261019.md), and
[repair headroom](docs/cross_modal_repair_headroom_result.md). These are
different cohorts and endpoints; their numbers must not be pooled.

**Bottom line:** There is evidence that the candidate bank contains variation
and that *some* proxy-based selection helps on *some* development endpoints.
There is **no validated clinical best-triple selector**, no demonstrated
reference-free error-localization accuracy, no accepted clinical repair result,
and no demonstrated online GPU-cost advantage over a strong fixed route.

## System and exact model roles

```text
Frozen SynEHRgy synthetic EHR
  → canonical EHR + evidence-grounded facts (direct vs context vs unknown)
  → one shared clinical intent, model-specific final text prompts
  → 3 frozen text-to-CXR generators × registered seeds
  → 4 frozen single-CXR-to-report generators per acquired image
  → frozen observers + structural checks + explicit missingness/cost
  → fixed / random / static / guarded-controller decision
  → selected candidate, abstention, or a bounded new model/seed call
```

| Role | Current registered route | Input → output | Important boundary |
| --- | --- | --- | --- |
| EHR source | SynEHRgy-v2 Qwen2-40bins | Generator output → synthetic structured EHR | Frozen. PromptEHR has separate prototype work; HALO is not part of the current 80-case candidate grid. |
| EHR bridge | V1.0/V1.1 canonicalizer, fact extractor, prompt renderers | EHR → four-state facts → final model-specific text | Source evidence and hash lineage retained. Unknown does not become negative; context is not a hard image label. |
| CXR candidate | RoentGen-v2 | Radiology-style text → CXR | Report/text-conditioned, **not** direct structured EHR → CXR. |
| CXR candidate | CheXGenBench Sana | EHR-derived text → CXR | Frozen text-to-image route. |
| CXR candidate | CheXGenBench PixArt-Σ | EHR-derived text → CXR | Frozen text-to-image route. |
| Report candidate | MAIRA-2 | One current CXR → report | Same-image report alternative. |
| Report candidate | CXRMate-single | One current CXR → report | **Not** EHR+CXR CXRMate-ED. |
| Report candidate | LLaVA-Rad | One current CXR → report | Same-image report alternative. |
| Report candidate | CheXagent-2 | One current CXR → report | Same-image report alternative. |
| Historical/other | EHRXDiff, UniDisc, MeDiM, RadEdit | Longitudinal CXR; CXR→report; text→CXR; image edit, respectively | Not interchangeable cold-start structured-EHR→CXR routes. |

All model weights remain frozen. Registration records a model path, worker,
input signature, planning memory, and Slurm-only execution gate; registration
alone does **not** certify a checkpoint's runtime availability or clinical
quality. See [worker registry](TriCompose-v1.2/src/tricompose_v12/live_workers.py)
and the [V1.1 prompt/worker contracts](TriCompose-v1.1/README.md).

## Connectors and immutable data contract

The connector is **not** an arbitrary free-text prompt. It has explicit,
versioned stages and lineage:

| Stage | Contract / code | Required checks |
| --- | --- | --- |
| Source → canonical EHR | [EHR bridge](TriCompose-v1.0/src/tricompose_v1/ehr_bridge.py), [staging](TriCompose-v1.0/src/tricompose_v1/staging.py) | Stable schema; source generator/path/hash in protected manifest; no replacement of difficult EHR after downstream outcomes. |
| EHR → radiology facts | [V1.1 facts](TriCompose-v1.1/src/tricompose_v11/facts.py) | `positive / negative / uncertain / unknown`; evidence and source fields; distinguish direct image-observable facts from compatible context. |
| Facts → model input | [V1.1 prompts](TriCompose-v1.1/src/tricompose_v11/prompts.py), [tokenizer audit](TriCompose-v1.1/src/tricompose_v11/tokenizer_audit.py) | Same clinical intent across models; final tokenizer text saved and hashed; no duplicate prefix or invented laterality/view/severity. |
| Prompt → CXR | [CXR contracts](TriCompose-v1.1/src/tricompose_v11/cxr_contracts.py), [execution](TriCompose-v1.1/src/tricompose_v11/cxr_execution.py) | Model revision, seed, prompt hash, EHR/fact hashes, image hash, runtime/cost, failure status. |
| CXR → report | [report contracts](TriCompose-v1.1/src/tricompose_v11/report_contracts.py), [execution](TriCompose-v1.1/src/tricompose_v11/report_execution.py) | Exact parent image hash, expert ID, report hash; do not treat two reports of one image as independent clinical votes. |
| Candidate graph / selection | [V1.0 graph](TriCompose-v1.0/src/tricompose_v1/graph.py), [score-table implementation](TriCompose-v1.0/eval/report_v1_1/README.md) | One immutable row per exact EHR–CXR–report lineage; missing scores remain `NA`, not zero. |
| Bounded action binding | [V1.2 action contract](docs/multi_action_repair_plan.md), [native dispatcher](TriCompose-v1.2/agent/tricompose_llm/native_action_resolver.py) | Only observed candidate IDs and registered tools; local validation binds private lineage; no hidden candidate/endpoint access. |

The historical bank has **80 fixed synthetic EHRs × 3 CXR paths × 4 report
paths = 960 triple slots**, with 240 CXR slots. Slots are not unique samples:
there are 80 distinct EHR, 147 CXR and 428 report hashes. Only **8/80** cached
EHRs have directly comparable radiographic facts under the strict projection;
the other 72 EHR-edge scores are unavailable, **not zero**. The historical
index and two authorized private handoff packages are described in the
[first-version delivery](docs/first_version_delivery_result_12714150_001.md).

Prompt de-collapse is real but limited: the historical 80-EHR bridge increased
unique shared clinical intents from **7 to 49**, with 20 honest neutral
fallbacks. Prompt diversity is **not** clinical conditioning accuracy. A
separate, prospectively fixed 500-position SynEHRgy run has **393 canonical-
valid / 107 structurally invalid** positions; **39/393** valid EHRs have any
explicit positive image condition and **33/393** have one of the prespecified
six findings. All 500 positions remain in the denominator. Its 2-EHR /
12-CXR / 48-report downstream pilot is a diagnostic, not a held-out clinical
test; see [prospective scorer gate](docs/fresh500_report_cxr_scorer_gate_20261015.md).

## Metric architecture: evidence, decision, final evaluation

**Do not collapse these three levels into one uncalibrated “consistency score.”**

1. **Policy-visible evidence** is what a selector may inspect *after* paying
   for the relevant model/scorer call: explicit four-state fact support,
   opposition, coverage and uncertainty; report structural checks; limited
   image–text similarity. It must not see an unacquired candidate's scores.
2. **Decision rule** is a frozen, auditable ordering or guarded action policy.
   The historical static selector is lexicographic: validity → explicit
   opposition → direct-EHR support → CXR–report support → report structure →
   declared cost → deterministic ID. It is a proxy-ranked baseline, not a
   clinical quality formula. Guarded repair requires fact preservation and
   genuine gain, not a contradiction disappearing into `unknown`.
3. **Independent final endpoints** assess the selected output on disjoint
   cases, ideally with independent image-grounded and clinician-reviewed
   evidence. The endpoint must not be reused to pick the candidate, set a
   threshold, rewrite the prompt, or decide when to stop.

For an edge and finding, `supported` requires *both* comparable explicit
states to agree; `opposed` requires explicit conflict; `unknown` or unavailable
is neither. A compact reporting contract is:

```text
coverage   = comparable_finding_count / eligible_finding_count
support    = supported_count / comparable_finding_count
opposition = opposed_count / comparable_finding_count
```

Always print counts and denominators alongside rates; a support rate with
near-zero coverage is not impressive. The shared 14-finding order is
atelectasis, cardiomegaly, consolidation, edema,
enlarged-cardiomediastinum, fracture, lung lesion, lung opacity, pleural
effusion, pleural other, pneumonia, pneumothorax, support devices, no finding.
For EHR comparisons, use only explicit *radiographically comparable* direct
facts; a medication, raised troponin, or omission in a report is not
automatically an image contradiction. No finding is not a blanket negative
for every disease.

| Metric family | Current implementation / evidence | Correct interpretation |
| --- | --- | --- |
| EHR unimodal | Schema/validity, demographics and clinical distributions, temporal diagnostics in EHR evaluation and protected staging | Fidelity requires a declared real reference cohort and matched schema; downstream TSTR/TRTS and privacy attacks are **not** established by structural validity. |
| CXR unimodal | Basic image validity/diversity; XRV frozen finding scores; historical FID/KID-style cohort diagnostics | FID/KID and diversity are **cohort** properties, not per-image ranking scores. XRV `op_norm` outputs are not calibrated disease probabilities; Brier/ECE are not valid on them as probabilities. |
| Report unimodal | Empty/section/generic/repetition/template/unsupported-prior checks in [report evaluation](TriCompose-v1.0/eval/report_v1_1/README.md) | These are mechanical quality signals. BLEU/ROUGE/METEOR, BERTScore, CheXbert F1, RadGraph F1, RadCliQ require a suitable **reference** and cannot score the fully synthetic bank as clinical truth. |
| EHR–CXR | Direct EHR facts vs frozen XRV states, six-finding diagnostics, support/opposition/coverage | Sparse direct EHR evidence must stay visible. CHF or diuretic context is weak compatibility, not a hard current-edema label. |
| EHR–report | Direct diagnosis/finding comparison plus diagnosis/medication/lab/vital source coverage | A radiology report need not repeat the entire EHR. Silence is usually unknown, not contradiction. |
| CXR–report | Frozen XRV vs CheXbert states; per-finding support/opposition/coverage; BioViL-T and Qwen2.5-VL secondary | Both labelers are fallible; cosine/VLM scalar is neither a probability nor clinical ground truth. Report hallucination/omission rates need image-grounded truth. |
| Graph and cycle | Frozen RadGraph-XL interface now deployable; same-image cross-path differences; optional report→CXR cycle | RadGraph interface smoke is **not** clinical qualification. Cycle is low-weight because a generator can reproduce its own bias. |
| Efficiency | Exact generator/scorer/planner calls, loads, failures, wall/GPU time, memory, API cost, abstentions | Cached call-count replays are **simulations**, never measured GPU savings. Compare policies at the same *total* cost. |

The frozen [RadGraph-XL deployment](docs/radgraph_deployment_result_12714150_001.md)
passed an authored offline interface smoke, but has not been clinically
qualified or merged into the historical selector. The 2026-09 evaluation
README's old “not found” line predates this deployment. `RadCliQ` is not a
validated local selection signal. Qwen/BioViL are diagnostic observers until
a same-image selection gate succeeds. The [same-image gate](docs/report_selector_evidence_gate_20261019.md)
currently fails to show that benefit.

## Controller / Agent: implemented interface, unproven method claim

The proposed contribution is **cross-modal error localization with targeted
regeneration**, not merely “an LLM chooses a number.” Keep the synthetic EHR
fixed. Available registered actions are `switch_report_model` on the same
image, `switch_cxr_model`, `change_cxr_seed` (with downstream report and score
cost), `select_candidate`, and `stop/abstain`. A prompt-surface action exists
as a contract but must remain disabled until final native-tokenizer handoff
is verified. Action outcomes and failed calls are charged. An optional LLM
planner receives only numeric evidence, opaque IDs, action history and costs;
no raw MIMIC data, synthetic patient prose/image, path, or secret crosses an
external API boundary.

The [V1.2 controller code](TriCompose-v1.2/agent/tricompose_llm/) includes
deterministic, local frozen-model and paid-API action prototypes. Those
interfaces are **not** proof of planning value: a five-case API pilot made
five bounded decisions but executed no live regeneration; a separate
five-anchor report-repair comparison changed no original winners. A 96-call
API transport/action diagnostic used **zero** EHR cases and made no generator
calls. The tested guarded two-case live retry accepted **0/2** replacements.
Five-case local-Qwen advantages on raw BioViL could be reproduced by fixed
order controls and did not establish a reasoning benefit. The finite-cache
strict repair audit found report-only opportunity but no qualifying image
replacement. Thus V1.2 is a **working research prototype and development
diagnostic**, not a validated self-correction algorithm.

Controls required before an Agent claim: predeclared fixed expert/seed path,
uniform score-free random acquisition and final choice, static best-of-budget,
deterministic guarded actions, equal-cost LLM actions, and a clearly labeled
hindsight oracle. Ablate stop/model/seed/verification separately. Evaluate
false repair, localization accuracy, quality, coverage/abstention, diversity,
and **measured** total cost on untouched patient-disjoint cases. The
[prospective benchmark gate](docs/prospective_benchmark_gate_20261015.md)
explains why the historical 80-case intervention bank and existing unreviewed
48-report packet cannot serve as that final test.

## Version map and current progress

| Track | What exists | What is not yet established |
| --- | --- | --- |
| V1.0 | Frozen EHR→text→CXR→report graph, 80-case candidate bank, fixed/random/static baselines, protected delivery | Clinically calibrated winner; direct structured-EHR→CXR. |
| V1.1 | Shared evidence-grounded prompts, de-collapse, hash-bound adapters, four-state three-edge score tables and report structure checks | Independent clinical validity of XRV/CheXbert proxies and thresholds. |
| V1.2 | Bounded action/guard/controller/API prototypes; multiple cache and small live diagnostics; prospective 500-EHR source and 2-case/48-report pilot; RadGraph-XL interface | Demonstrated natural error localization, accepted clinical repair, LLM advantage or online savings. |
| V1.3 | Protocols and readiness work for independent benchmark | Untouched, adequately powered, image-grounded final test and clinical reader adjudication. |

Do not infer that an `sbatch` file or source module means its job has run. The
version map is a *research evidence* map, not a count of Python files.

## Where to inspect without exposing patient data

- [V1.0 pipeline](TriCompose-v1.0/README.md), [V1.1 bridge](TriCompose-v1.1/README.md),
  [V1.2 implementation notes](TriCompose-v1.2/README.md),
  [report metric code](TriCompose-v1.0/eval/report_v1_1/README.md).
- [First-version delivery receipt](docs/first_version_delivery_result_12714150_001.md)
  describes the protected 80-case result index and two private packages under
  `artifacts/protected/tricompose_v1_2/deliverables/`. The metadata package is
  for aggregate review; the sample package contains synthetic payload and
  remains project-restricted. Neither belongs in Git or external APIs.
- [Metric evidence gate](docs/fresh500_report_cxr_scorer_gate_20261015.md),
  [report selector failure gate](docs/report_selector_evidence_gate_20261019.md),
  [50-image report comparison](docs/ricord_two_report_opacity50_result_13744455.md),
  [repair feasibility](docs/cross_modal_repair_headroom_result.md).

All patient-derived artifacts and synthetic generated payload live under
`artifacts/protected/` (project group `ruishanl_1185`; directories `2770`,
files `0660`; NFS may display group ownership as `nobody`). Public docs and
logs contain only aggregate/sanitized results. Do not open or print raw MIMIC
EHR rows, source reports, source CXRs or real targets. Run GPU/heavy work only
under Slurm; before any new `sbatch`, show the *complete* script and resource
request and obtain explicit approval. Existing runs are immutable: use a new
opaque run ID rather than overwriting one.

## Next publishable gate

1. Freeze a patient-disjoint, image-grounded endpoint and a report-review
   protocol **before** observing candidate outcomes. Keep the inspected RICORD
   50 and RadEval 44 anchors as development resources, not final tests.
2. Validate one reference-free same-image report choice against fixed
   CheXagent-2, fixed MAIRA-2, random, and hindsight oracle. Report positive
   *and negative* explicit support, unknown/abstention, false assertions,
   serious error types, uncertainty intervals and total cost.
3. Only if a qualified decision signal exists, test bounded targeted repair
   on prospectively fixed synthetic EHRs, with unchanged-EHR and no hidden
   candidate evidence. Preserve the strong static and fixed baselines.
4. If matched-cost targeted repair does not beat static reranking on an
   independent endpoint, report that honestly; the current software still
   serves as a reproducible frozen-model composition benchmark.
