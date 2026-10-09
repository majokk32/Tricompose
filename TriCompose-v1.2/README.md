# TriCompose V1.2: controlled-intervention preparation

Current LLM-routing prototype (2026-10-08): code is grouped in `agent/`, entry
point `agent/run.py`; usage and privacy/execution limits are documented in
`docs/llm_agent.md` at the workspace root. It reuses the old OpenAI-compatible
endpoint settings but supports bounded report/CXR action proposals and local
accept/rollback checks. The original executors are authored fixtures and
immutable numeric-cache replay; the separate fresh report-only bridge below
has now completed real frozen-worker calls. Neither is clinically validated repair.
The optional local Qwen planner needs separately approved GPU Slurm; no model
training, scorer revision or output-folder relocation accompanies this module.

Completed next generation contrast (2026-10-08), **job 12862704**:
`agent/run_cxr_action_diversification.py` fixes the same two fully synthetic
EHRs/facts and original final prompts, then runs RoentGen-v2 seed 2 and Sana
seed 1. Each new image gets its own XRV observation, CXRMate-single report
and CheXbert labels: four planned image/report slots, at most 16 charged
worker requests, zero retry. The two renderers share clinical intent, not
necessarily identical text; this is not direct structured-EHR conditioning
or a pure generator-architecture ablation. No Qwen/API/training, scorer
revision, original-selection replacement or clinically accepted repair.
Fourteen new fixture tests and all **3,964 V1.2 tests** passed; CPU preparation
checked source/artifact hashes and asset metadata without opening clinical
bodies/pixels or instantiating models. Frozen plan:
`artifacts/protected/tricompose_v1_2/cxr_action_diversification_plans/cxr_actions2_12851223_001/`.
Protocol: `docs/cxr_action_diversification_protocol.md`. Complete script
`agent/slurm/19_cxr_action_diversification2_debug_a40.sbatch` was displayed,
explicitly approved and submitted unchanged: one debug A40, two CPUs, 32 GiB
RAM, ten-minute cap. It completed on b11-09 in **4m02s**, exit 0:0, all four
triples and 16 charged requests complete, zero failures/retries. The unchanged
proxy preservation comparison passes **0/4**; EHR–CXR support stays 0/1 in
all slots. Some CXR–Report counts improve, but this does not establish clinical
repair. Metadata-only CPU postflight and all **3,974 tests** passed.
Output: `artifacts/protected/tricompose_v1_2/cxr_action_diversification_runs/cxr_actions2_12862704/`.
Result and audit limitations: `docs/cxr_action_diversification_result.md`.
Original winners and clinical attribution limits remain unchanged; no next
GPU job is approved.

Next CPU-prepared decision interface: `agent/run_current_image_policy.py`
uses only the two actually observed reports on each current synthetic probe
image, measured risk flags, cached image-disagreement evidence, and numeric
feedback from the unchanged gate's rejected expert switch. Historical
other-image report scores and BioViL endpoint scores are not planner inputs.
The only generation requests are the remaining LLaVA-Rad/CheXagent-2 experts
on that exact image; no generator is installed in the decision job.
`agent/audit_current_image_policy.py` adds CPU-only metadata replay. A fixed rule
receives the same packet and prospective worker allowance but makes zero LLM
calls; this is a decision control, not a quality or GPU-saving result.
Plan: `artifacts/protected/tricompose_v1_2/current_image_policy_plans/current_policy2_12827441_001/`.
The complete `agent/slurm/15_current_image_policy2_debug_a40.sbatch` script was
approved and submitted unchanged (one A40, two CPUs, 32G RAM, five-minute cap,
two decisions). Job `12851312` completed on `b11-09` in **1 minute 5 seconds**.
Both Qwen decisions selected LLaVA-Rad, matching the fixed rule 2/2; no generator
ran, and this does not demonstrate LLM superiority or compute savings. Original EHRs,
selected triples, exhausted old budgets, models and acceptance rules are unchanged.
CPU plan replay and protected permissions passed in allocation `12827441`;
all 14 new fixture tests and the complete 3,875-test V1.2 suite passed. No
new model calls occurred. This input-interface preparation does not establish
repair efficacy or an advantage over the deterministic rule control.
Output: `artifacts/protected/tricompose_v1_2/current_image_policy_runs/current_policy2_12851312/`.
CPU actual-call/dispatch replay passed at
`artifacts/protected/tricompose_v1_2/current_image_policy_audits/current_policy2_12851312_12851223_001/`.

Completed shared report execution: `agent/run_current_image_requested_reports.py`
binds the authenticated matching requests to the original images and deduplicates
four policy provenance rows into two physical LLaVA-Rad reports plus two CheXbert
calls. Shared outputs are not distinct policy outcomes; Qwen's sunk calls are
not refunded. Eight new fixture tests and the full 3,883-test suite passed.
Plan: `artifacts/protected/tricompose_v1_2/current_image_report_plans/current_report2_12851223_001/`.
`agent/slurm/16_current_image_shared_reports2_debug_a40.sbatch` was reviewed,
explicitly approved and submitted unchanged (one A40, two CPUs, 32G RAM,
eight-minute cap). Job `12851790` completed on `b11-09` in **2 minutes 28 seconds**.
Both reports and both CheXbert operations completed; zero attempts failed, but
**0/2 new reports passed the unchanged replacement gate**. EHR–Report support
remained 0/1 for pair 0 and fell from 1/1 to 0/1 for pair 1; CXR–Report support
was 4/8 → 3/8 and 2/8 → 2/8, respectively. Each new report introduced an
EHR–Report proxy opposition. These sparse proxy counts are not clinical truth,
and the matching Qwen/rule requests do not demonstrate LLM advantage.
Existing models, scorers, images,
original winners and acceptance rules stay unchanged; no new LLM/CXR/XRV/API call.
Output: `artifacts/protected/tricompose_v1_2/current_image_report_runs/current_report2_12851790/`
(`score_table.csv`, `paired_report_comparison.json`, protected generated reports).
CPU metadata-only postflight passed at
`artifacts/protected/tricompose_v1_2/current_image_report_audits/current_report2_12851790_12851223_001/`.
Six new audit tests and all 3,889 V1.2 tests passed. Do not treat the unresolved
image-scorer disagreement/missing comparison as an established image fault,
or relax the scorer/acceptance rules merely to pass a replacement.

Completed CPU-only fixed-image diagnostic: `agent/diagnose_current_image_evidence.py`
authenticates both report-execution postflights, the unchanged blind image
observer, all three actually observed report experts per current image, and
the fixed XRV operating-point scores/thresholds. For each case, one known EHR
fact opposes its fixed XRV label: the maximum all-three support achievable by
changing only the report is **0/1 under these proxy labels**. Four symbolic
report states per finding are not generated reports. Unknown/uncertain cannot
remove the immutable EHR–image conflict by reducing report coverage.
This does not rule out report structure improvements or establish an image
fault: pair 0 has missing image evidence; pair 1 has scorer disagreement.
No scorer, threshold, clinical gate, policy, original winner or model was changed.
Output: `artifacts/protected/tricompose_v1_2/current_image_evidence_diagnostics/evidence2_12851223_001/`
(`image_evidence_table.csv`, `report_observations.csv`, `image_fact_diagnostics.json`,
`RESULTS_CN_EN.md`; manifest SHA256
`5a890ac1f3e044d73500bd335da3d5c2d252d501cf8065c54b4690f426f61776`).
All six result artifact hashes, 327 source hashes, 44 direct reader hashes,
112 symbolic state checks, byte-exact tables/report and project permissions
passed verification. Eighteen new authored-fixture tests and all **3,907 V1.2
tests** passed (full suite 30.071 seconds); zero new
model calls, GPU jobs or external API calls. See
`docs/current_image_reachability_protocol.md` for the post-hoc scope and limits.

Completed evidence diagnostic, **job 12853380**, metadata audited:
`real_validation/rsua_qwen_observer.py` reuses the existing 50 public RSUA
images (25 pneumonia / 25 paper-described normal-cohort proxies) to test the
unchanged blind Qwen image observer. Both current fixed-label conflicts concern
pneumonia. Freeze checkpoint, eight-finding prompt, decoding and source order;
evaluate only pneumonia, retain uncertain/unknown/failed/blocked slots, and
open reference labels only after model predictions are fsynced. The reference
is a published cohort proxy, not independent clinical image adjudication.
No new data acquisition, scoring threshold, winner or clinical gate changes.
Plan: `artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_plans/rsua_qwen50_12851223_001/`.
Manifest SHA256: `e8a38e289adc778bdc4fc89a91d31cdf4fc832414226db5261932ce3d5a9d6b3`.
The CPU audit checked 317 source hashes, ten artifact hashes, all 50 image file
stats and seven model-asset stats without parsing source pixels/cohort labels.
Nineteen new fixture tests and all **3,926 V1.2 tests** passed (30.371 seconds).
The complete `agent/slurm/17_rsua_qwen50_debug_a40.sbatch` was reviewed,
explicitly approved and submitted unchanged: one debug A40, two CPUs,
32 GiB RAM, ten-minute cap, no retry. Job `12853380` completed on `b11-09`
in **2m41s**, exit `0:0`; all 50 observer responses were complete, with one
model load. Worker runtime was 146.098s and peak Torch-allocated VRAM 15.594 GiB.
Script SHA256: `626a1ed86ba21191f0cced0c17d81b0e90e9df980c51593f7ffe2baf00b6f321`.
Pneumonia proxy readout: **2/25 positive-reference slots** explicitly supported,
**13/25 negative-reference slots** explicitly supported, **16/50 uncertain**,
zero unknown/unavailable. Explicit coverage is 34/50; correct explicit support
over the full cohort is 15/50 (0.30), and conditional support over explicit
answers is 15/34 (0.4412). Same-image XRV default 0.5 gives 31/50 (0.62) with
full explicit coverage; the unchanged transported threshold gives 29/50 (0.58).
Different abstention behavior and proxy references preclude a clinical-accuracy
claim. The current Qwen image observer is not qualified to decide clinical
image faults or override XRV; do not retune it retrospectively to pass this run.
Output: `artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_runs/rsua_qwen50_12853380/`.
Metadata-only audit: `artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_audits/rsua_qwen50_12853380_12851223_001/`.
All four output artifacts, 317 source pins, ten input artifact pins, 50 source
PNG hashes, seven model-asset stats, 151 exact journal events and aggregate
arithmetic passed. That audit opened no real reference rows or image pixels;
it did not independently recompute per-image reference assignment or inference.
See `docs/rsua_qwen_observer_result.md` and the immutable
`docs/rsua_qwen_observer_protocol.md`. Original winners, score thresholds,
clinical attribution guard and frozen models remain unchanged.

Completed medical-observer diagnostic, **job 12861745**, metadata audited:
`real_validation/rsua_medical_report_observer.py` uses the unchanged deployed
CheXagent-2 SRRG findings prompt/loader, seed 42, greedy decoding, 512-token
cap and float32 precision, then the existing frozen CheXbert implementation.
The two native processes run serially and reuse the same 50 RSUA source images;
no new classifier prompt, model download, cohort, threshold or old-winner edit.
This report-to-label chain is not an independent image truth source: CheXbert
also labels candidates and CheXagent shares XraySigLIP's encoder. Only the
published pneumonia cohort proxy is evaluated; no clinical repair authority.
Empty/capped reports and overlength label inputs are unavailable, never negative.
Plan: `artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_plans/rsua_medical50_12851223_001/`
(manifest `8edb55e03155b546acdce1d627407924e1bf54d50e1bcc716452d0e387f2331f`).
CPU preflight checked 320 source hashes, 14 artifact hashes, 50 image stats
and 26 asset stats without opening source pixels/reference rows or models.
Twenty-four new fixture tests and all **3,950 V1.2 tests** passed (30.763s).
The reviewed `agent/slurm/18_rsua_medical50_debug_a40.sbatch`: one debug A40,
two CPUs, 32 GiB RAM, ten-minute cap, at most 50 report + 50 label requests,
one load per model, zero retries. Script SHA256
`6762012ecbce971e6a3c2dbb636b672b19bc4ea66246b5ffb199d455891c9861`.
After complete-script/resource display and explicit approval, the unchanged
script completed on `b11-09`, exit `0:0`, in **2 minutes 47 seconds**. All 50
reports and all 50 label forwards completed; one load per model, zero retry.
Worker interval: 150.494s; peak Torch-allocated VRAM: 13.362 GiB report model,
0.422 GiB labeler (not total GPU memory).

**Pneumonia was unknown in all 50 reports' CheXbert outputs.** Explicit coverage
is 0/50, correct explicit support is 0/50, and conditional support is NA because
there were no explicit answers. This is not 0% clinical accuracy or 50 negative
cases. The other heads contain positive/negative predictions, reports are
nonempty and uncapped, and label inputs are below the token limit; this is not
an all-empty label vector. No report body or source pixels were inspected in
the metadata postflight. Other findings have no reference here and cannot be
declared correct. Do not relabel opacity/consolidation as pneumonia or tune
the prompt/threshold to pass this already observed cohort.

Output: `artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_runs/rsua_medical50_12861745/`
(manifest `4df6cf3378aa0a6dd7ab38ce93261fb42909606be4798930b63e671328e33a2f`).
CPU metadata audit: `artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_audits/rsua_medical50_12861745_12851223_001/`
(manifest `f65051c63a6a9bb1d57aa1f194bd40550c3a779d3744318b1f44d6eae19b766b`).
It checked 57 result artifact hashes, 320 code hashes, 14 input artifact hashes,
50 PNG byte hashes, 26 model-asset stats and 151 journal events per phase.
It did not rerun inference or per-image reference assignment, parse reports
or decode pixels. Original selected triples and clinical gates stay unchanged.
The protected aggregate comparison and per-finding state table are at
`artifacts/protected/tricompose_v1_2/real_validation/rsua_medical_report_reviews/rsua_medical50_12861745_12851223_001/`
(manifest `405629c3d37c79388bea033410bc58c9406390b505109ae6f36b295ebe6d10a9`).
See `docs/rsua_medical_report_observer_result.md`; the prepared plan, consumed
code/tests and original protocol `docs/rsua_medical_report_observer_protocol.md`
remain immutable. No next GPU run or clinical repair was authorized.

Completed local-Qwen engineering check (job `12814592`, 2026-10-08): ten fixed
source-order cases and paired fixed/rule/random controls completed in 1 minute
50 seconds. Qwen and both rules selected the same candidate for all ten cases;
each improved two report-cache selections under the frozen proxy contract.
This is not fresh regeneration, clinical repair or evidence of LLM advantage.
The score table and exact coverage/cost limits are documented in
`docs/llm_agent.md`; protected output is
`artifacts/protected/tricompose_v1_2/llm_agent_comparisons/llm_rules_matched10_12814592/`.

Next interface (2026-10-08): `agent/run_fresh_report_agent_v2.py` adds a separate
**fresh report-only** bridge; approved job `12817356` completed on A40
`b11-09` in 14 minutes 13 seconds. Clinical quality gains are not established. It
reuses frozen RoentGen/XRV for a fixed initial image, CXRMate-single/CheXbert
for the initial report, then permits local-Qwen requests for the three other
frozen report experts. Qwen exits before each generation worker starts.
Initial/proposed outputs share the existing fresh scorer profile; the old
cache controller and numeric bank are not changed or mixed into this run.
Twenty receipt/ledger tests plus eight supplemental-asset tests passed; the
full V1.2 suite, including 28 new independent postflight checks, passed 3,668
tests. Supplemental Vicuna/BiomedBERT/vision/BERT
weights are included in the new preflight without rewriting the first plan.
This is not clinical repair, an installed CXR-regeneration tool or proof of
LLM superiority. Exact scope, costs and approval boundaries are in
`docs/llm_agent.md`.

CPU preflight is complete, with zero model calls, at
`artifacts/protected/tricompose_v1_2/llm_fresh_report_plans/fresh_report2_fullassets_12799642_001/`.
The approved `agent/slurm/05_fresh_report_agent_smoke2_debug_a40.sbatch` was
submitted unchanged after full script/resource display and explicit approval.
It requests one A40, four CPUs, 64G RAM and a 45-minute cap for two fixed
synthetic cases. `agent/audit_fresh_report_agent.py` adds metadata-only CPU
postflight: it replays observed evidence, action feedback, selections and costs
without reading clinical artifact bodies or altering the frozen score profile.
Any different GPU script or subsequent run still needs separate approval.

Completed fresh interface: two fixed synthetic EHRs, two new RoentGen-v2 CXRs,
eight reports (CXRMate-single, MAIRA-2, LLaVA-Rad, CheXagent-2 per image), twenty
charged generator/verifier calls and six local-Qwen decisions. No worker failed.
One report change passed the unchanged proxy-preservation checks; the other
case retained its baseline. Both exhausted their budgets, so this run does not
demonstrate dynamic stopping or compute savings. Each fixed EHR/image pair
still has an unresolved classifier-proxy opposition; report-only switches
cannot repair it. The labels/structure metadata are not clinical ground truth.

Outputs:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_runs/llm_fresh2_12817356/`.
CPU postflight passed without model calls or body/pixel inspection:
`artifacts/protected/tricompose_v1_2/llm_fresh_report_audits/fresh_report2_12817356_12817261_001/`.
That audit includes `score_comparison.csv` (unweighted baseline/retained raw
readouts) and authenticated replay/charge details in `audit.json`.

Independent endpoint completed (2026-10-08):
`agent/run_fresh_report_secondary.py` reuses frozen BioViL-T on all eight new
reports/two images. Fixed CXRMate, existing first-preserving static replay and
the original LLM choices are sealed before scores exist. Static and LLM choose
the same candidate in both cases; no LLM advantage or cost savings are shown.
Full-report overlength remains NA, not silently truncated. Approved job
`12827754` ran the exact reviewed
`agent/slurm/06_fresh_report_secondary_v100.sbatch` on one V100 (`d14-08`),
two CPUs, 8G RAM and a ten-minute cap; it completed in **25 seconds**, exit
`0:0`. All eight pairs were available, using two image and eight text encodings.
The previously accepted report change increased independent raw cosine by
**0.312147**; the other case was unchanged. Cosine is not clinical accuracy,
and the EHR/image proxy oppositions remain unresolved. No selection, gate,
threshold, generator or primary scorer was revised from these endpoint scores.
The immutable CPU preflight is
`artifacts/protected/tricompose_v1_2/llm_fresh_report_endpoint_plans/fresh_report2_biovil_12817261_001/`.
Results are in
`artifacts/protected/tricompose_v1_2/llm_fresh_report_endpoints/fresh_report2_biovil_12827754/`:
`score_table.csv` contains all eight scores; `case_comparison.csv` contains
the fixed/static/LLM comparison.

`agent/audit_fresh_report_secondary.py` verified the sealed tables, pins,
permissions and encoding counts with zero model calls or body/pixel inspection.
Audit output is
`artifacts/protected/tricompose_v1_2/llm_fresh_report_endpoint_audits/fresh_report2_biovil_12827754_12817261_001/`.
Its `veto_diagnostics.csv` records four gate-blocked alternatives with higher
cosine than the retained report. This diagnoses disagreement between evidence
types, not clinical correctness or a reason to change winners retrospectively.
Fifteen endpoint/control tests and eight new postflight tests passed; the full
suite now passes **3,691 tests**. Any further GPU run requires separate approval.

Research protocol: [further development](../docs/further_development.md) and
[localization protocol](../docs/v1_2_localization_protocol.md).

Prospective CXR action completed (2026-10-08), **metadata audited**:
`agent/run_fresh_cxr_agent.py prepare|run` and
`agent/tricompose_llm/live_cxr_bridge.py` add one seed-one RoentGen-v2 action
after the audited report-only run. They retain both original EHRs, exact final
prompts, frozen checkpoints/scorers/thresholds and each retained report expert.
Qwen sees all four already-observed reports' numeric evidence per image, not
their text, pixels, EHR bodies, paths or BioViL endpoint scores. It may request
the installed CXR action, stop or abstain. New images get their own XRV,
report and CheXbert chain; no old report is reused as a new-image observation.
The unchanged cross-image gate must preserve the fixed baseline and retained
report evidence. Veto or charged failure keeps the previous selected triple.
This is a one-action engineering probe, not validated localization, general
adaptive repair, clinical success, dynamic-stop efficiency or LLM superiority.

CPU plan (zero model calls/body parsing):
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_plans/fresh_cxr2_12817261_001/`.
Manifest SHA256:
`807f136342fd0c837962b3b860462df8e08df6a197a0899b6de3b7e5682f1163`.
Maximum: two new images, two new reports, eight worker attempts and two policy
requests. Historical twenty worker/six policy calls are shared sunk work, not
new free evidence or an executed cost-matched control. The new run saves
`selection.json`, raw `score_table.csv`, `selected_triplets.json` with exact
artifact pointers, and separate fresh ledgers/charge summaries. All completed
old runs and consumed source files remain unchanged. Twenty-seven new fixture
tests passed; the full suite passed **3,718 tests** in CPU allocation `12817261`.
Approved script `agent/slurm/07_fresh_cxr_probe2_debug_a40.sbatch` requests one
A40, four CPUs, 64G RAM and a thirty-minute allocation cap. The A40 request
accommodates the retained MAIRA-2 expert's existing 40-GiB planning requirement;
models run serially, not resident together. The complete exact script/resources
were displayed and explicitly approved; submitted unchanged as job `12834413`.
Job `12834413` completed on `b11-09`, exit `0:0`, in **4m59s**, after initially
waiting for resources. It produced two new CXR/report chains: two RoentGen,
two XRV, two retained report-expert and two CheXbert calls, plus two local
Qwen policy calls. Neither probe passed the unchanged replacement gate:
**zero accepted proxy transitions**, both previous selections retained.
Successful execution is not clinical repair success. The script, source pins
and sealed CPU plan remain unchanged; no A100 job or duplicate was submitted.

Metadata-only postflight completed in CPU allocation `12817261` using
`agent/audit_fresh_cxr_agent.py`. It replays the policy's past-only numeric
state, new four-phase chains, unchanged gates, rollback and separate charge
ledgers; it verifies exact selected artifact pointers and before/after raw
edge tables without reading clinical bodies/pixels or calling any model.
Sixteen authored postflight tests passed; the earlier full suite passed
**3,734 tests**. The completed protected audit is
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_audits/fresh_cxr2_12834413_12817261_001/`
(manifest `c77ef7951a3eb2979f89df27200a77816b947a6e08d90ee53ab0c579c7d12fe4`).
It includes `audit.json` and `score_comparison.csv`. This verifies metadata,
lineage, gates and costs, not independent model recomputation or clinical truth.
The approved script and all 308 sealed source pins remain unchanged.

Separate low-memory endpoint **completed as job `12840340`**:
`agent/run_fresh_cxr_secondary.py prepare|run` reuses the existing frozen
BioViL-T scorer on all eight cached and two new CXR/report pairs, at most four
image and ten full-report encodings. It preserves the presealed fixed/static,
before-probe, final-selected and probe identities; no endpoint reranking or
primary scoring/generation is allowed. Scores are raw uncalibrated cosine,
not clinical accuracy; missing/overlength reports remain explicit NA.
CPU plan, zero model calls:
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_endpoint_plans/fresh_cxr2_biovil_12817261_001/`
(manifest `914e06ae76e3788523c4910d52b696e1d0bb3a62a63c1b8a8a9f65dbfa11e917`).
Fourteen new authored tests passed; the full suite passed **3,748 tests**
in CPU allocation `12817261` with the workspace-local temporary directory.
Approved script
`agent/slurm/08_fresh_cxr_secondary_v100.sbatch` requests **one V100, two CPUs,
8G host RAM, five-minute cap**. It loads no generator, MAIRA-2 or Qwen. The
complete exact script/resources were displayed and explicitly approved, then
submitted unchanged as `12840340`. It completed on V100 node `d13-10`, exit
`0:0`, in **25 seconds** (scorer driver including load/I/O: 19.447s). All ten
pairs were available; four image and ten text encodings. Peak Torch-allocated
VRAM was 0.583 GiB, not total device usage. No new generation or training.
Protected result:
`artifacts/protected/tricompose_v1_2/llm_fresh_cxr_endpoints/fresh_cxr2_biovil_12840340/`
(manifest `035657ccb53c6a2937d32ab05ce33b07104b74ce5c1db94e0e20570a73c2f9e8`).
CPU metadata checks authenticated the plan, source/assets, output hashes,
reservation, full inventory and reconstructed CSV/summary arithmetic; no
embedding recomputation or clinical adjudication was performed.

| Synthetic case | Before-probe cosine | New probe cosine | Probe minus before | Final selected cosine |
| --- | ---: | ---: | ---: | ---: |
| case_009 | -0.100129 | 0.777972 | +0.878101 | -0.100129 |
| case_018 | 0.235240 | 0.633635 | +0.398395 | 0.235240 |

Higher image/report similarity did not override the original three-edge gate.
Neither probe gained strict fixed-EHR/image evidence. Case 009 also failed the
report-structure/common-risk check; case 018 lost support and introduced one
explicit proxy opposition. These are frozen-rule diagnostics, not adjudicated
clinical errors. Selections, scorer thresholds and EHR/prompt inputs stay fixed.

Detailed zero-model-call veto diagnostic completed:
`agent/diagnose_fresh_cxr_probe.py` saved
`artifacts/protected/tricompose_v1_2/llm_probe_diagnostics/fresh_cxr2_12840340_12817261_001/`
(manifest `5ad0624520486ff48dafa6987b03c1f2cf33c7653c0e511b91fc6f7f2cc9d7ea`).
Case 009's official section checks actually pass: the quality veto is solely
the repeated-four-gram ratio increasing from 0 to 0.00833333. Existing temporal
risk is unchanged, not a newly introduced error. Both cases have one explicit
EHR constraint; its XRV readout remains negative in both old and new images.
Case 018's EHR/report support increases from zero to one while XRV still
opposes it. These data cannot establish whether the image, report, EHR mapping
or classifier is clinically wrong. No gate or selected triple was changed.

Secondary observer **completed as job `12843415`**, submitted unchanged after
complete-script display and explicit approval; V100 node `d14-06`, exit `0:0`,
Slurm elapsed **2m38s**:
`agent/verify_fresh_probe_images.py prepare|run` uses the unchanged existing
image-only Qwen-VL eight-finding prompt/settings on all four reference/probe
images. The model receives pixels plus that generic prompt, no EHR, reports,
IDs, prior scores or expected answers. Observations are sealed before cached
labels are compared. The same Qwen checkpoint also powers numeric planning;
this is therefore not an independent clinical adjudicator or validated fault
localizer. It adds at most four model calls, no generation/training/retries,
and cannot change primary thresholds or winners.
CPU plan:
`artifacts/protected/tricompose_v1_2/llm_probe_image_plans/probe_images4_12817261_001/`
(manifest `4913be6d356954b46acbaabd6415e0f7f4aa390d727fb9fcebf036f1451a114f`).
Approved script `agent/slurm/09_probe_images4_qwen_v100.sbatch`: one V100,
two CPUs, 32G host RAM, five-minute cap. The existing loader falls back to
FP16 on non-BF16 GPUs and checks at least 24 GiB device memory. This four-image
V100 smoke completed with **4/4 complete responses**. All **3,771 tests** pass,
including thirteen new decomposition and ten observer-contract fixtures.
The pinned script, observer program and prepared plan remain unchanged.

Protected output:
`artifacts/protected/tricompose_v1_2/llm_probe_image_observers/probe_images4_12843415/`
(manifest `a55e210f3c489b1dd9d7d691b885a6d979c56eac68a586a9e659c0aa041f6f8b`).
Worker observation runtime is 43.173s; peak Torch-allocated VRAM is 15.789 GiB,
not total device memory. Four calls and one load attempt are charged; no new
generator calls, retries, training, threshold changes or selected-triple changes.
CPU allocation `12827441` verified all pinned program/artifact/model hashes,
permissions, the nine-event call ledger and exact 32-slot CSV replay; this is
a numeric postflight, not a clinical adjudication.

Across **four image/finding slots for the known EHR constraints**, Qwen supports
three and marks one uncertain; XRV opposes all four. This is two previously
examined development cases with two seeds each, not four independent patients
or clinical accuracy. Across all 32 unique image/finding slots, Qwen and XRV
explicitly agree on 23, oppose on eight, and have one uncertainty-not-comparable
slot. The disagreement demonstrates scorer sensitivity, not which model is
clinically correct. Qwen remains a secondary observer and shares its checkpoint
with numeric planning; the original gate and winners remain unchanged.

Optional image-evidence guard **implemented and CPU-previewed**, not installed
in the consumed live runner:
`agent/tricompose_llm/image_evidence_guard.py` supplies `build_guard`,
`guarded_state` and `guarded_decision`. It withholds image-error attribution
when direct known EHR constraints have image-scorer disagreement or missing
comparison; no known constraints also remain unresolved. Unknown/uncertain
are not negatives, failed observer responses remain unavailable, and shared
image reports are not votes. Unmentioned EHR findings do not become new hard
constraints. Only image-regeneration tools are removed from the optional new
planner view; report tools, original scores, budgets and charged history stay
unchanged. An unresolved stop becomes explicit abstain. An invalid raw decision
still fails closed; a valid withheld proposal is preserved rather than hidden.
Scorer agreement does not confer clinical acceptance or bypass the old gate.

`agent/build_image_disagreement_preview.py` completed in existing CPU allocation
`12827441`, with **zero new model calls or submissions**, at:
`artifacts/protected/tricompose_v1_2/llm_image_evidence_guards/image_guard4_12827441_001/`
(manifest `e899bf7127fc1123a9015a1ebbe64eaee61aa52d9a13ddf78fd61a7204276df9`).
It retains all four image observations: three guards have known-EHR scorer
disagreement and one has missing/equivocal evidence. Both actual historical
CXR-regeneration proposals become abstain in the **counterfactual preview**.
The rejected probe images are diagnostic guard rows, not two invented extra
planning requests. The observed generation/verifier cost (eight attempts),
numeric planner cost (two requests) and observer cost (four calls plus one load
attempt) remain recorded; measured savings are NA. No historical output,
threshold, EHR, prompt or winner changed. This is not prospective cost reduction,
clinical error localization, successful repair or LLM superiority.
Twenty added authored-fixture tests pass; the full **3,791-test** suite passes
in 35.261s. Inline postflight verified all source/artifact hashes, identity and
state bindings, exact CSV replay, two decision replays and protected permissions.

The optional guard must be explicitly installed in a **new prospective runner**
before it can affect real execution. Upstream private artifact authentication
is still mandatory: matching rebased IDs alone is not image/EHR provenance.
Existing consumed scripts, sessions and observers remain immutable.

### New execution-boundary entry (2026-10-08; completed and audited)

`agent/tricompose_llm/guarded_action_dispatch.py` now applies the optional
image-evidence guard **before constructing a generation backend**. It preserves
the original proposal and charged history in a durable journal. Unresolved image
attribution becomes abstention; an unblocked action without an approved backend
becomes a deferred request, not a claimed model call. A reserved failed dispatch
cannot be replayed, and returned backend output still needs the original
receipt/cost/acceptance gate. Consumed controllers are unchanged.

`agent/run_guarded_image_probe.py` prepares a new forward-ordered, two-image
execution smoke: use the same fixed seed-zero development reference images,
run two fresh frozen image-only Qwen observations, seal them, then dispatch the
two authenticated **cached** numeric intentions through the guard. It does not
read old observer predictions to select cases. No fresh numeric planning,
generation backend, primary scorer, report generation, external API or training
is installed. If an action remains unblocked it is written to protected pending
requests for a separately approved job. Historical selections stay unchanged.

This is execution-boundary validation on examined development inputs, **not** a
held-out efficacy experiment, a complete adaptive pipeline, clinical acceptance,
measured model-call savings or evidence of LLM superiority. Sixteen new fixture
tests pass; the full **3,807-test** suite passes in 33.853s. Plan preparation only
uses metadata and hashes in the existing CPU allocation. After complete-script
display and explicit user approval, the unchanged GPU smoke was submitted as
**job `12848110`**. It completed on V100 node d14-16 in **2m22s**, exit 0.
Both fresh observations were complete; both known-EHR comparisons retained
scorer disagreement and both cached image intents were withheld as abstain.
There were **zero backend dispatches, fresh numeric-planner calls, generation
or primary-scorer calls, pending generation requests, retries or winner changes**.
The worker recorded 34.394s from its internal start through summary creation and
15.789 GiB peak Torch-allocated VRAM; these are not total batch runtime or total
device memory. Two observer calls plus one load attempt remain charged.

Immutable prepared plan (submitted unchanged):
`artifacts/protected/tricompose_v1_2/guarded_image_execution_plans/guarded_probe2_12827441_001/`
(manifest `71340e7a74c0407e81550e0c73e23429530498b694111088f1c8d23e3966e18c`).
Executed complete script: `agent/slurm/10_guarded_image_probe2_v100.sbatch`;
request one V100, two CPUs, 32G host RAM, five-minute allocation cap. This matches
the observer's verified 24-GiB minimum; P100 16-GiB cards are not compatible.
Script SHA256:
`7e85cb1d160b9153fd401a2ec4adf73acbedf5a36768585a0b0563a271a5eee4`.
At submission, Slurm reported pending/Priority with no predicted start. Read-only
reservation inspection showed the V100 node group reserved until 18:00 on
2026-10-08 (America/Los_Angeles); free GPU counts alone did not imply eligibility.
Reservation expiry does not guarantee an immediate start. No resources were
changed and no additional job was submitted.

`agent/audit_guarded_image_probe.py` completed CPU-only postflight under allocation
12827441. It authenticates the plan/artifacts and exactly replays charged
observer reservations, prediction sealing, guard dispatch and pending requests.
It rejects fabricated calls, savings, clinical acceptance and changed selections.
It never reexecutes image inference or reads clinical bodies/pixels. Completed
GPU output is the new protected job-bound run under
`artifacts/protected/tricompose_v1_2/guarded_image_execution_runs/guarded_probe2_12848110/`;
manifest SHA256 `ea05ee34f65e03e0fd823bc482767aa73acdc1677eba691c5429c6d8cec2d70c`.
Completed audit:
`artifacts/protected/tricompose_v1_2/guarded_image_execution_audits/guarded_probe2_12848110_12827441_001/`;
manifest SHA256 `49c235caa6576a9f95f8b10062b1bb142c9df550a70e8bbdc38ddeb2e5ad695f`.
Protected project-group modes 2770/0660 and exact journal replay passed. This is
a **forward execution veto on two examined development images and cached
intentions**, not fresh LLM planning, clinical repair/localization, verified image
accuracy, measured savings or a held-out quality-compute improvement.
Nine new postflight fixture tests pass; the full **3,816-test** regression suite
passes in 41.046s. No model was invoked by these tests or the CPU preparation.

Optional resource-swap proposal (conditionally approved; not submitted):
`agent/slurm/11_guarded_image_probe2_debug_a40.sbatch` requests one debug A40,
two CPUs, 32G host RAM and the same five-minute cap. It hash-checks and executes
the previously displayed worker body as Bash, **not** via a nested sbatch;
the old body cannot request a second V100 allocation. A read-only
`sbatch --test-only` on 2026-10-08 predicted a start at 17:36:32 on b11-09.
That prediction was not a submission or guarantee. After complete-script display
the user approved replacement **only if the original job remained pending**.
The pending-only cancellation request did not affect the original job, which had
started when the V100 reservation expired. It was retained and completed;
the A40 wrapper was never submitted and no duplicate was created. The unchanged
loader selects BF16 on supported hardware versus FP16
on V100; the resource swap would test the gate, not establish precision-equivalent
outputs or change the archived V100 observer results.

### Fresh guarded numeric planning (2026-10-08; completed)

`agent/run_guarded_fresh_policy.py` and
`agent/tricompose_llm/guard_aware_qwen.py` prepare two genuine new Qwen decisions
on the **two existing seed-one development probe images**, not replayed old
intentions. Each image has one observed report and three untried report experts
in the authenticated inventory. Four expert readouts on the other, seed-zero
image of the same fixed EHR are explicitly marked as weak scheduling context,
not unseen report scores or independent clinical votes. The two original
selected triples are retained; these diagnostic branches are not new winners.

The closed typed-content packet contains only opaque IDs, numeric evidence,
cached image-guard observations, image-group bindings, allowed requests and
costs. No clinical bodies, image pixels, paths, hashes or external API enter the
LLM request. Scorer disagreement does not imply image error. At most one new
policy call per case may request an untried report expert or stop/abstain.
Actual generation/scoring backends are absent: a requested report is deferred
until a separate complete script is displayed and explicitly approved. Failed
attempts remain charged, with no automatic retry or submission. There is no
training, threshold change, clinical acceptance or measured savings claim.

CPU preparation completed under allocation 12827441. Plan:
`artifacts/protected/tricompose_v1_2/guarded_fresh_policy_plans/fresh_policy2_12827441_001/`
(manifest `f575d3be72e16242b50e711d187dc9be9db59f929d380bb65e2dd771731a46ae`).
Postflight verified 314 source pins, 101 artifact pins, both closed packets,
three allowed report requests per case and project-group modes 2770/0660.
Preparation invoked **zero** models. Fifteen added authored-fixture tests passed;
the full **3,831-test** suite passed in 42.276s, without model inference.

Reviewed script `agent/slurm/12_guarded_fresh_policy2_debug_a40.sbatch` requests
one debug A40, two CPUs, 32G host RAM and a five-minute allocation cap. It runs
only fresh numeric planning, bounded to 384 generated tokens per decision,
and cannot generate images/reports or replace prior selections. The existing
loader's hardware-dependent BF16/FP16 selection is unchanged; this is not a
precision-equivalence comparison with the archived V100 observer. After the
complete script was displayed and the user explicitly approved it, the unchanged
script was submitted as **job 12849550**. It completed on debug A40 node b11-09
in **1m01s**, exit 0. Both genuine new `model.generate` calls returned strict,
valid decisions with no failures: the two branches respectively requested
**MAIRA-2** and **CXRMate-single**. Both are deferred report requests; no report
or CXR generation/scoring backend was invoked and original winners stay fixed.
No additional generation job has been approved or submitted by this entry.

Protected output:
`artifacts/protected/tricompose_v1_2/guarded_fresh_policy_runs/fresh_policy2_12849550/`
(manifest `189c9cf86afdbb16e86287b2b51390a2ad94f52eacc57192a6b20da196de45bb`).
Two policy requests and one model-load attempt remain charged. Worker runtime
through summary creation is 31.257s, not total batch time; peak Torch-allocated
VRAM is 15.978 GiB, not total device memory. Combined usage: 5,021 input tokens
and 156 output tokens. No retries, training, threshold changes or external API.

CPU postflight exactly replayed the nine-event reservation/decision/dispatch
journal, pending requests, fixed image/EHR bindings, source/artifact hashes and
protected modes. Audit:
`artifacts/protected/tricompose_v1_2/guarded_fresh_policy_audits/fresh_policy2_12849550_12827441_001/`
(manifest `6337ab9a18629ce9725d699ef98136fb8dafd060cb87bb07bff1f4a7ceb04e70`).
The audit adds zero model calls. **Both decisions cite historical other-image
evidence, not current-image evidence.** Thus this demonstrates an operational
expert-request interface, not justified current-report error localization,
clinical repair, superior expert choice or measured compute savings. Separately
approved execution of these two requests and their same-image comparisons is
recorded below; both proposed replacements were rejected.

Script SHA256:
`a6b9aab5c83dda2f1cacbedb46d005962bdf08653a14cd09cd28a818d759cdf4`.
Shell syntax and whitespace checks passed. A non-submitting scheduling dry-run
predicted 18:45:40 on b11-09 on 2026-10-08 (America/Los_Angeles); this transient
estimate is not an allocation or guaranteed start time.

### Two requested report experts executed (2026-10-08; completed, no accepted change)

`agent/run_guarded_requested_reports.py` authenticates the two completed Qwen
requests and prepares actual **MAIRA-2 and CXRMate-single report generation** on
their respective unchanged seed-one probe images. It uses the existing frozen
report adapters and CheXbert scorer without a new planner, image observer, XRV
inference or CXR generation. Cached image labels/receipts are byte-authenticated;
the fixed EHR, image and thresholds stay unchanged. No external API or training.

Each old four-call ledger is exhausted and remains immutable. This new phase
has a separate two-attempt budget per case: reserve one report attempt before
constructing a backend, then reserve CheXbert only after a validated report.
Failed/in-flight reservations keep their charge and never retry automatically.
Historical source costs remain recorded as shared sunk cost, not zero. This
does not change an old budget or fabricate a new image/classifier call.

New reports were compared with the **existing report on the same image**
using the unchanged fact-preservation/structure gate. Unknown cannot become
agreement and silencing a conflict cannot count as correction. The per-branch
diagnostic selection cannot replace the original retained triples. Completed
protected outputs include `score_table.csv`, `paired_report_comparison.json`,
`completed_triplets.json`, separate call journals and a sealed manifest. No
clinical repair, fault-localization accuracy, held-out advantage or savings are
claimed from these outputs or from proxy-gate passage alone.

CPU plan:
`artifacts/protected/tricompose_v1_2/guarded_report_execution_plans/guarded_report2_12827441_001/`
(manifest `9eee28509cd594cc5449d3e643b2194ed23da734a0109317a741682b20274f17`).
Two cases, exactly two deferred experts, 316 source pins, 117 artifact pins,
cached-label provenance and modes 2770/0660 passed metadata checks. Preparation
loaded no models. Fifteen new fixture tests pass; full regression **3,846 tests
in 42.255s** passed using the existing report-context environment with pinned
header/pixel dependencies preloaded and workspace-local `TMPDIR`. Initial
system/partial environments lacked required test dependencies; no environment
was installed, edited or substituted for any generation checkpoint.

Reviewed and executed complete script:
`agent/slurm/13_guarded_requested_reports2_debug_a40.sbatch`
(SHA256 `5cfd6804f8720b14b6b41e0891a65eeea89159fa188f326c49a78a9366f26e85`).
Request: debug, one A40, two CPUs, 32G host RAM, eight-minute allocation cap;
worker coordinator timeout 420s, each report subprocess 120s and CheXbert 45s.
MAIRA-2's registered planning memory class is 40 GiB; P100 16-GiB cards are not
appropriate for this job. Prior analogous A40 calls took about 63s for MAIRA-2,
9s for CXRMate-single and 8s per CheXbert call, excluding other I/O/coordination.
These observations are not a completion-time guarantee. Shell syntax and diff
checks passed. A **non-submitting** scheduling dry-run predicted 19:10:26 on
b11-09 on 2026-10-08; estimates may change. After complete-script display and
fresh explicit user approval, the unchanged script was submitted as **job
12850244**. It completed on debug A40 node b11-09 in **2m02s**, exit 0.
Both requested reports and both CheXbert calls completed: **four new worker
attempts, zero failures, zero accepted diagnostic branch changes**. There was
no new planner, observer, XRV or CXR call, retry or training. Original retained
triples and exhausted historical ledgers remain unchanged. Worker runtime
through summary creation was 96.929s, not total batch duration. Individual
report subprocess intervals were 54.554s (MAIRA-2) and 7.742s (CXRMate-single);
CheXbert intervals were 7.741s and 6.695s, including their subprocess startup/I/O.

Protected completed output:
`artifacts/protected/tricompose_v1_2/guarded_report_execution_runs/guarded_report2_12850244/`
(manifest `677482fd514cd4d788b5e6904ead00cc309c9388b98c1d39e2dd8bef079ab6e8`).
`score_table.csv` has four rows: two existing same-image reports and their two
requested alternatives. `completed_triplets.json` contains the two new
diagnostic triples, **not promoted winners**. Raw before/after readouts:

| Expert switch | EHR–Report supported/known | CXR–Report supported/known | CXR–Report comparable/known | CXR–Report proxy opposition | Accepted |
| --- | --- | --- | --- | --- | --- |
| CXRMate-single → MAIRA-2 | 0/1 → 0/1 | 4/8 → 2/8 | 4/8 → 2/8 | 0 → 0 | No |
| MAIRA-2 → CXRMate-single | 1/1 → 0/1 | 2/8 → 4/8 | 3/8 → 4/8 | 1 → 0 | No |

All supported CXR–Report labels here are **negative**; their counts are not
positive-finding accuracy. EHR–CXR stays at 0/1 supported with one proxy
opposition in each branch; no image evidence changed. The first switch removes
the temporal-language flag but loses previously comparable image facts. The
second improves image-label readouts but loses EHR support/comparability,
silences rather than resolves a prior comparison and introduces unsupported
temporal-comparison language. Unknown/missing is not negative or agreement.
Both pass their model-specific section contracts, which alone do not guarantee
clinical factuality. The unchanged fact-preservation gate therefore keeps both
existing branch reports. This is **successful execution, not successful repair**.

CPU metadata postflight replayed all **12 new-phase journal events**, exact
requests/results, fixed EHR/image/seed lineage, cached dependencies, shared sunk
costs and both same-image gate decisions. It authenticated 316 source pins,
117 artifact pins and the byte-exact CSV. The first text-mode CSV check differed
only because universal-newline reading normalizes CRLF; generation artifacts
were not edited. Four CUDA cache directories were normalized from mode 2700
to the required project-group mode 2770; all run directories/files then passed
2770/0660 checks, with no content or group changes. Protected audit:
`artifacts/protected/tricompose_v1_2/guarded_report_execution_audits/guarded_report2_12850244_12827441_001/`
(manifest `ddf8f0f69092f38a4e0eb6b42a27fff11199f07ca1ce664844532bd7185be8bd`),
with `paired_scores.csv`. The audit invokes zero models. No new GPU job has
been submitted. These two development cases do not establish clinical truth,
error-localization accuracy, superior LLM scheduling or compute savings; both
LLM reasons remain historical other-image citations. A useful next experiment
needs current-image-grounded planning and a separately frozen evaluation
endpoint with cost-matched controls, not threshold relaxation to force a pass.

### Secondary evaluation of the executed expert requests (2026-10-08; completed)

`agent/run_guarded_report_secondary.py` prepares the existing frozen BioViL-T
endpoint for **four reports on the same two existing seed-one images**: each
old report and its actual requested replacement. It seals the unchanged gate
decisions before the secondary scores exist. No new Qwen, CXR/report
generation, primary scorer, threshold fitting, API or training is included.
Whole-report text is used; overlength/empty reports remain explicit NA rather
than being silently truncated or dropped from the denominator. BioViL cosine
is secondary uncalibrated evidence, not clinical truth or a routing oracle.
Both prior branch reports and original retained triples remain fixed.

Protected plan:
`artifacts/protected/tricompose_v1_2/guarded_report_endpoint_plans/guarded_report2_biovil_12827441_001/`
(manifest `f7d221aaab9d02054c083930cf6bb7e94e3774089ba645d9b7fac08241f4f431`).
CPU preparation and postflight authenticated 346 source pins, 140 artifact
pins, eight existing BioViL model pins, same-image comparisons, fixed EHR lineage,
separate historical cost and project-group 2770/0660 permissions; no report
body, image pixels or model was loaded. The exact pin counts are recorded by
the prepared plan. Fifteen new fixture tests and **all 3,861 tests in 42.290s**
passed. Tests establish implementation contracts, not improved generation.

Reviewed and executed script: `agent/slurm/14_guarded_report_secondary_debug_p100.sbatch`
(SHA256 `8208914b964bfda4632020d47e921146b7415f822ebbed36cfe921900b77f504`).
Request: debug, one P100, two CPUs, 8G RAM, five-minute cap; scorer timeout
240s plus 15s termination grace. The existing metrics environment reports
PyTorch 2.6.0+cu126 with compiled `sm_60` support, checked without GPU inference.
Prior analogous BioViL work used about 0.579 GiB Torch-allocated VRAM on V100;
that is not a certified P100 peak or runtime guarantee. `noderes -f -g`
reported debug P100 availability at preparation time; resource state can
change. Shell syntax and whitespace checks passed. After the complete script
and resources were displayed and the user explicitly approved, the unchanged
script was submitted as **job 12850816**. It started on debug P100 node e23-02
one second after submission and completed in **28s**, exit 0. No resource
override, retry, training, API call or duplicate job occurred.
Completed outputs include `scores.json`, `score_table.csv`,
`case_comparison.csv`, an attempt reservation and a sealed manifest. Failed
work keeps its charge and never automatically retries. A secondary gain must
not retroactively override the primary veto or become an LLM-superiority claim.

Protected output:
`artifacts/protected/tricompose_v1_2/guarded_report_endpoints/guarded_report2_biovil_12850816/`
(manifest `8f239d53dd12abccaa9f9495653b5a5934826a856d25d7409666015a2d9ca95a`).
All four requested pairs were scored, with **two image encodings, four text
encodings and zero unavailable reports**. One endpoint worker attempt is
charged separately from source generation and planner costs. Scorer runtime
including load/I/O was 21.798s, not total batch duration; peak Torch-allocated
VRAM was 0.579 GiB, not total device memory. Raw full-report cosine scores:

| Requested expert switch | Existing report | Requested report | Requested minus existing | Retained report |
| --- | --- | --- | --- | --- |
| CXRMate-single → MAIRA-2 | 0.777972 | 0.583549 | -0.194422 | CXRMate-single |
| MAIRA-2 → CXRMate-single | 0.633635 | 0.715938 | +0.082304 | MAIRA-2 |

One requested alternative improves the secondary image/text cosine and the
other worsens it. The second still fails the unchanged primary gate because
EHR support/comparability is lost and temporal-language risk increases. BioViL
does not evaluate EHR fidelity and a higher cosine cannot certify full-triple
correctness. **Both retained-versus-existing deltas remain zero**; the two
diagnostic branch reports, original winners and all acceptance rules stay
unchanged. This is not evidence of successful repair, cost savings or LLM
superiority, and the two development cases are not an untouched test cohort.

CPU postflight authenticated plan/source/artifact/model pins, replayed the
presealed choices and every numeric endpoint row, checked encoding/attempt
accounting, byte-exact CSV tables and protected output/cache/log permissions.
It read metadata and byte hashes only, not report bodies or image pixels,
and invoked zero models. Protected audit:
`artifacts/protected/tricompose_v1_2/guarded_report_endpoint_audits/guarded_report2_biovil_12850816_12827441_001/`
(manifest `59851fa3c7418a5b4462a1debfc5fabde951a877632fbf3bbf9d4f64b15d145e`),
with `audit.json` and `case_comparison.csv`. No new GPU task
has been submitted beyond this approved evaluation.

The deferred user idea, **stronger API planner versus total generation cost**,
is recorded in `docs/further_development.md` and linked from `docs/llm_agent.md`.
Current execution remains on the local frozen Qwen interface; no external API
experiment is activated by this preparation.

Read-only scheduler dry-runs for the same thirty-minute request on gpu A100
and L40S predicted starts on 2026-10-11, whereas the existing debug A40 job was
predicted for 2026-10-08. These are changeable estimates, not guaranteed start
times or certified memory compatibility. No alternative GPU job was created
and no resource change/cancellation was made.

`benchmarks/build_intervention_smoke.py` is a CPU-only, metadata-only builder.
It reads the protected synthetic candidate registry, not source EHR, image
pixels, report text, or real targets. It prepares a fixed-path control, a
different-case report swap, and a different-case CXR swap per case. The
output is a new immutable protected run containing:

- `blind_items.jsonl`: opaque item IDs for policy-facing evaluation;
- `resolver.jsonl`: hash-bound candidate references for a later protected
  artifact loader;
- `intervention_key.jsonl`: mechanical intervention labels and pending
  independent-adjudication status;
- `manifest.json`: source hash, fixed path, arm counts, and status.

The resolver is not itself a model input. A future scoring runner must load
the referenced *artifacts*, compute evidence for each rewired pair, and keep
the intervention key hidden from the policy. An injected swap is not
automatically a clinical error, and an untouched generated pair is not
automatically clinically correct. This stage does **not** report localization
accuracy or claim a validated method. Evaluator inference needs a separately
reviewed Slurm script and explicit submission approval.

`benchmarks/diagnose_cached_pair_sensitivity.py` can recombine existing frozen
XRV and CheXbert artifact labels for a CPU-only manipulation check. It measures
whether a swap changes label disagreement relative to its control. It does not
rerun image-text verification, distinguish image from report fault, calibrate
XRV, or supply clinical ground truth. Results remain protected and diagnostic.

## Initial accessible eight-case smoke (2026-09-30)

The protected intervention run is
`artifacts/protected/tricompose_v1_2/benchmarks/directfacts8_intervention_smoke_20260930_001/`.
It contains eight untouched controls, eight report swaps, and eight CXR swaps
from the predeclared Sana seed-0 / MAIRA-2 path. Artifact bytes and source
patient data were not opened; the builder used the hash-bound candidate table.
The original, group-incorrect duplicate was removed after a byte-identical
project-group copy was validated. The retained directories have mode `2770`
and files `0660`; on this NFS mount, the project group appears as `nobody`.

The cached-label manipulation check is in
`artifacts/protected/tricompose_v1_2/diagnostics/directfacts8_cached_pair_sensitivity_20260930_001/`.
Compared with each control, explicit CXR-report contradictions increased in
only 2/8 CXR swaps and 2/8 report swaps. The untouched controls already had
one contradiction across eight items. These are **not** clinical localization
accuracy, false-repair rate, or evidence that the other swaps were medically
compatible: XRV is uncalibrated here, report and image labelers can miss facts,
and no independent artifact adjudication has occurred. The immediate design
implication is that a single 14-label disagreement score cannot safely decide
which modality to regenerate.

## Eighty-case label-targeted intervention bank (2026-10-01)

`benchmarks/build_targeted_interventions.py` prepares a larger CPU-only,
metadata-only benchmark from the immutable 80-case candidate registry. It
chooses a different-case donor only when the frozen same-modality label state
for edema or pleural effusion is explicitly opposite; unknown and uncertain
never count as negative. It never reads image pixels, report text, or EHR
contents. Each EHR stays fixed, every case retains an untouched control, and
the source case is assigned once to development/calibration/final-test
(48/16/16). The blind item list contains only opaque item IDs; the intervention
answer key is a separate protected file and must not be given to a policy.

The first diagnostic path, Sana seed 0 + MAIRA-2, produced 80 controls and 80
CXR swaps but no qualifying report swaps. The frozen CheXbert output for this
fixed MAIRA-2 path had almost no explicit edema/effusion states; that is a
label-coverage observation, not proof that the reports or model are poor.
The immutable diagnostic run is
`artifacts/protected/tricompose_v1_2/benchmarks/pool80_label_targeted_20261001_001/`.

Two intermediate CheXagent-2 construction runs were retained for provenance:
`pool80_label_targeted_chexagent2_20261001_001` allowed donors to cross split
boundaries, and `pool80_label_targeted_splitlocked_20261001_001` restricted
donors but left the final-test split without a positive report donor. Neither
is the benchmark to score.

The retained benchmark is
`artifacts/protected/tricompose_v1_2/benchmarks/pool80_label_targeted_stratified_20261001_001/`.
Its Sana seed-0 + CheXagent-2 fixed path uses case-level 48/16/16 splits,
stratified solely for positive pleural-effusion donor availability. Donors
remain inside the same split. It has **80 untouched controls, 76 report swaps,
and 80 CXR swaps**; four report swaps are explicitly unavailable. The arm
counts by split are:

| Split | Control | Report swap | CXR swap |
|---|---:|---:|---:|
| Development | 48 | 46 | 48 |
| Calibration | 16 | 16 | 16 |
| Final test | 16 | 14 | 16 |

Artifact hashes, split isolation, and the one-modality-only contract were
revalidated; all 20 synthetic-only unit tests pass. This is **not** clinical
localization accuracy: donor selection uses the same frozen labels that would
make a label-only evaluation circular, and a swapped artifact may remain
clinically compatible. Future verification must use independent evidence and
compare against each case's untouched control; tuning must stay out of
final-test. The `final_test` name in this pilot means held out from subsequent
policy tuning only: the fixed report model was chosen after inspecting
aggregate label coverage over this 80-case pool. A genuinely untouched
paper-grade final cohort must be generated or reserved separately.

`benchmarks/score_intervention_biovil.py` and
`slurm/04_synthetic_interventions_biovil_p100.sbatch` scored the 236 displayed
pairs with frozen BioViL-T as approved job `12530046` (32 seconds). The scorer
read the blinded resolver, not the intervention answer key, and embedded each
of 80 unique synthetic CXRs and reports once. The protected score run is
`artifacts/protected/tricompose_v1_2/benchmarks/pool80_targeted_biovil_20261001_001/`.
The separate CPU-only post-hoc analysis, which read the key only after scoring,
is `pool80_targeted_biovil_analysis_20261001_001/` under the same protected
benchmarks directory. All 22 synthetic-only unit tests pass.

| Arm | Pairs | Fraction with lower BioViL cosine than untouched control | Mean control-minus-swap cosine |
|---|---:|---:|---:|
| Report swap | 76 | 100.0% | 0.431 |
| CXR swap | 80 | 93.75% | 0.377 |

These are strong **cross-case swap sensitivity** results, not error-localization
accuracy or clinical correctness. The score is a single pairwise similarity
number and cannot distinguish an erroneous CXR from an erroneous report.
The report-swap donor pool has few explicit positive examples, so repeated
donors and case/style differences could make this benchmark easier than
natural generation errors. No score threshold was fitted, and the pilot's
`final_test` split is not paper-grade independent because the fixed report
model was chosen after inspecting aggregate label coverage over the pool.

## Direct EHR evidence coverage gate (2026-10-01)

`benchmarks/audit_localization_evidence.py` joined the same blinded benchmark
to the existing V1.1 EHR-edge records without new inference. Its protected
output is `artifacts/protected/tricompose_v1_2/benchmarks/pool80_localization_coverage_20261001_001/`.
The audit counts only a finding that is explicit in the *fixed EHR*, the
displayed CXR label, and the displayed report label. An unknown/uncertain
state is never treated as negative.

Across 80 untouched controls, **72 have no direct radiographic EHR fact**, seven
have a direct EHR fact but no three-way comparable finding, and only **one**
has a three-way comparable finding. The same one-case ceiling appears in both
swap arms (80 CXR swaps and 76 report swaps); the diagnostic state pattern
points to the injected modality for that one case in each arm. This is not
an accuracy estimate: the XRV/CheXbert labels used here also informed donor
selection, so they are not independent evaluation evidence. All 25
synthetic-only unit tests pass.

**Go/no-go:** the current 80-case cohort supports testing pair-mismatch
detection and static selection, but it cannot support a broad claim that the
system identifies whether the CXR or report is clinically wrong. A conservative
policy must abstain on most cases. Before presenting targeted regeneration
as V1.2's contribution, obtain a separately fixed cohort with substantially
more explicit, evidence-grounded radiographic EHR facts and an independent
evaluator, or narrow the claim to mismatch-aware candidate selection.

The initial eight-case smoke was prepared before the required **real
matched-data scorer validation**. It does not clear that gate. The corrected order is documented in
[`docs/further_development.md`](../docs/further_development.md): validate frozen
metrics on real matched and independently adjudicated mismatched pairs first,
then interpret or extend the synthetic intervention benchmark.

## Separate explicit-EHR feasibility cohort (2026-10-01)

`benchmarks/generate_unconditional_ehr_pool.py` uses the already-pinned frozen
SynEHRgy runtime and sampling parameters but writes the new pool with project-
group protected modes. It does not change the older generator or its outputs.
`benchmarks/select_explicit_ehr_cohort.py` predeclares a CPU-only screen for a
**new** 100-case SynEHRgy Qwen2-40bins pool. It checks the immutable source
manifest and every case hash, canonicalizes each fully synthetic EHR, and uses
the V1.1 fact extractor. Eligibility requires a positive, evidence-backed
finding in the latest visit's diagnosis field for cardiomegaly, pleural
effusion, pulmonary edema, pneumonia, pneumothorax, or atelectasis. Clinical
context, uncertain/unknown findings, earlier-visit findings, and inferred
devices do not qualify. The first two eligible cases in source-manifest order
are selected; the complete 100-case denominator and all rejected reasons are
retained in a protected, non-overwriting screen run. If fewer than two qualify,
the pipeline stops without silently generating more or substituting cases.

This is **unconditional EHR generation followed by predeclared eligibility
selection**, not disease-prompted EHR generation or an unbiased prevalence
sample. The checkpoint's `mimic4_hf_bins40` provenance is not established as
the MIMIC-IV-ED schema. `slurm/05_new_explicit_ehr_pool100_p100.sbatch` was
shown in full with its resource request and then explicitly approved. Job
`12531476` completed in 7m28s on one P100, with no CXR/report inference.
Its protected, immutable runs are `ehr_pools/v12_explicit_ehr_pool100_20261001_001`,
`ehr_cohorts/v12_explicit_ehr_screen_20261001_001`, and matching V1/V1.1
staging runs under `artifacts/protected/`. Of 100 generated cases, 74 were
structurally valid and five had an explicit eligible direct finding; the first
two eligible cases were staged. Both selected cases have an explicit pneumonia
finding, so this is a pipeline smoke, not a disease-diverse localization test.
V1 and V1.1 staging validation both passed, and each case has a distinct
clinical intent and distinct prompt for each active CXR model.

The older V1 staging helper initially assigned its new run to the owner's
personal group. The 18-file V1 run was copied byte-for-byte into a project-
group-inheriting directory, rechecked with an identical tree SHA256, and
restored at its original path. The redundant owner-group copy was removed
after verification. All new run directories have mode `2770`, files `0660`;
on this NFS mount project-group ownership displays as `nobody`. The existing
80-case bank remains unchanged.

The next protected CPU-only request run,
`artifacts/protected/tricompose_v1_1/cxr_requests/v12_explicit_two_all3_s2_20261001_001`,
is hash-bound to the two staged EHRs. It contains 12 requests: two cases ×
RoentGen-v2/Sana/PixArt × seeds 0/1. The full frozen CXR inference script
`slurm/06_explicit_two_all_cxr_p100.sbatch` and resource request were shown
and explicitly approved before submitting array job `12533465`. All three
P100 tasks completed without error: four candidates per model, 12 total.
RoentGen-v2 produced 512×512 images (peak 3.283 GiB, 1m37s), Sana 1024×1024
(7.767 GiB, 1m32s), and PixArt 512×512 (12.92 GiB, 3m49s). The three runs
share the exact request-manifest hash; all candidates declare frozen weights,
two cases, seeds 0/1, no added adapter prefix, and runtime-observed tokenizer
input. Output directories/files have modes `2770`/`0660` and inherited the
project mount group. This validates execution and lineage, **not image quality
or clinical agreement**; image contents have not yet been inspected. Both
selected EHRs carry explicit pneumonia, so comparison remains a narrow smoke
test. Report quality and cross-modal scoring require separate evaluation.

A protected, hash-bound 48-request CXR-only report run was prepared at
`artifacts/protected/tricompose_v1_1/report_requests/v12_explicit_two_four_experts_12533465_001`:
12 generated CXRs × MAIRA-2/CXRMate-single/LLaVA-Rad/CheXagent-2. These four
experts receive one current synthetic CXR each; `cxrmate_single` is **not**
CXRMate-ED and receives no structured EHR. The A40 array script
`slurm/07_explicit_two_four_reports_a40.sbatch` and its resource request were
shown and explicitly approved before submitting array job `12538545`. All four
tasks completed with exit code 0: 12 reports per model, 48 total, with one
report per model for each of the same 12 synthetic CXRs. The protected output
runs are `report_candidates/v12_explicit_two_{maira2,cxrmate_single,llavarad,chexagent2}_12538545`
under `artifacts/protected/tricompose_v1_1/`. Runtime and peak VRAM were
MAIRA-2 1m57s/27.359 GiB, CXRMate-single 20s/0.660 GiB, LLaVA-Rad
1m20s/14.334 GiB, and CheXagent-2 1m15s/13.362 GiB. All output manifests
bind to the same request-manifest SHA256, declare frozen weights and a single
current synthetic CXR input, and declare no structured EHR or real target
supplied to the models. Every candidate file exists; output directories/files
have modes `2770`/`0660` with inherited project mount group (displayed as
`nobody` on this NFS mount). These checks establish generation completeness
and provenance, **not clinical quality**; report text has not been manually
reviewed.

For this same October cohort, a bounded CPU-only integrity/structure pass is
complete under
`artifacts/protected/tricompose_v1_1/evaluation/v12_explicit_two_12538545_001/`.
All 12 generated PNGs were readable and nonconstant; all 48 reports were
nonempty and passed their model-specific section contract. The conservative
unsupported-prior/no-change language flag fired on 15/48 reports; this is a
review signal, not a clinical adjudication. The explicit 48-lineage registry
was complete but remained `selection_ready: false` until frozen cross-modal
evidence was computed. The full `slurm/08_explicit_two_score_select_a40.sbatch`
script and resource request were shown and explicitly approved. Job `12538856`
completed in 42 seconds with exit code 0 on one A40. Its protected output is
`evaluation/v12_explicit_two_12538545_001/scoring_12538856/`: frozen XRV
labels for 12 images, frozen CheXbert labels and secondary BioViL-T cosine for
48 image-report pairs, all three cross-modal edge summaries, the 48-row scored
table and two static selections. The selected cases had no hard contradiction
signal under the present diagnostic label rules, but the exhaustive selector
only raised mean support/coverage from 0.2857 to 0.3214 versus the
predeclared fixed Sana seed-0 + MAIRA-2 path, at 6 CXR and 24 report calls per
case rather than 1 + 1. This is same-scorer optimization, not independent
clinical validation. More concerning, the selected `case_035` image-report
pair has BioViL-T cosine -0.020, compared with 0.838 for its fixed-path pair;
the secondary metric was deliberately excluded from selection. This discordance
requires review before any selected output is called clinically better. XRV
scores are not calibrated probabilities, and two cases cannot support a
clinical superiority claim.

The synthetic-only `case_035` selected/fixed/same-image-alternative spot-check
is kept in protected `manual_review_12538856_001/` under that evaluation run.
It found that extra report labels can be rewarded despite less certain
image-grounded detail and an unsupported comparative phrase; it is a
non-radiologist review, not clinical adjudication. A separate, CPU-only
descriptive cross-scorer audit at
`selection_audits/v12_explicit_two_12538856_001/` flags one of the two selected
cases for review (selected BioViL-T rank 21/24, with a large same-image report
gap). This post-hoc flag **does not change selection** and must not be used as
an independently validated threshold or a paper-primary metric.

`benchmarks/analyze_candidate_scorers.py` extends the cached audit to all 48
triples while holding the CXR fixed for each report comparison. The protected
review bundle is
`selection_audits/full48_scorer_audit_12538856_002/` under the same October
evaluation run. It validates candidate/image/report hashes, separates positive
and negative label support, and exports model, image, case and pair tables.
Only two EHR cases underlie the 12 images and 72 report-pair comparisons.
The existing selector and secondary BioViL-T pick the same top report on
2/12 images. Among 52 report pairs with different label support and BioViL
scores, 25 have opposite rankings; 8/13 such pairs with equal total hard
contradiction counts are also discordant. These are descriptive disagreement
counts, not clinical adjudications.

XRV assigns positive states to 43/48 known heads across four PixArt images,
40/48 across four Sana images and 15/48 across four RoentGen-v2 images.
Three images have all 12 mapped, available heads positive. The preprocessing
source uses XRV normalization, center crop and 224-pixel resizing, but the
cause of the positive saturation is not established. This flags a need to
validate operating points and image-domain behavior before treating these
labels as strong selection evidence. On this matched image inventory, MAIRA-2
has the fewest XRV/CheXbert explicit conflict signals (5), while LLaVA-Rad
has the highest mean BioViL-T cosine (0.739). Neither establishes the clinically
best report model. The bundle also retains the previous 80-case swap pilot
as separate context; swap sensitivity is not evidence of within-case report
selection quality. Five CPU-only tests cover unknown-safe counts, image-fixed
comparisons, ties, case denominators and lineage/hash rejection. No new model
inference or GPU job was needed for this audit.

`real_validation/calibrate_cached_xrv.py` with
`slurm/09_cached_xrv_thresholds_score_cpu.sbatch` completed as explicitly
approved Slurm job **12544371** (1 CPU, 4 GiB, no GPU; 3 seconds, exit 0).
The immutable protected result is
`evaluation/v12_explicit_two_12538545_001/weak_thresholds_12544371/`
under `artifacts/protected/tricompose_v1_1/`. It reuses the fixed real
128-validation/128-test XRV cache and reads source patient split metadata only
within Slurm, never real image pixels or report text. Thresholds fit validation
only, with at least 20 explicit positives and 20 explicit negatives per finding,
and are evaluated once on the cached test cohort. Missing reference heads or
insufficient support disable a finding rather than defaulting it to negative.
The same enabled-head subset is used for before/after synthetic positive-call
counts, separating threshold changes from reduced label coverage. The job also
recomputed the October 48-candidate scores and diagnostic selection with cached
CheXbert/BioViL evidence.

Only **pleural effusion** met the predeclared validation support requirement
(40 explicit positives / 24 negatives). Its validation-fitted threshold is
0.745114535. On the held-out-from-fitting test subset (47 positives / 19
negatives), balanced accuracy changed from 0.6473 to 0.7088, sensitivity from
0.9787 to 0.6809, and specificity from 0.3158 to 0.7368. The test negative
count still falls below the predeclared 20-per-class support flag; these are
descriptive weak-reference results, not a validated clinical operating point.
Pneumonia had only 18 / 11 explicit validation references and is disabled;
all other insufficient or unavailable heads are also disabled, never negative.

On the **same one-head subset** across the 12 cached synthetic images, positive
calls changed from 7 to 1. Masking the other heads does not establish that the
earlier multi-head saturation or synthetic-image fidelity has been repaired.
Both fixed EHR cases have pneumonia, so their EHR-CXR edge now has **zero
comparable direct facts**. The reused selector exports zero support and zero
coverage for that edge; this means unavailable image evidence, not a negative
pneumonia finding or clinical inconsistency. The exported selector's inherited
`diagnostic_uncalibrated_cxr_labels` status remains conservative; consult the
calibration provenance and protected `STATUS_AND_INTERPRETATION.md` for the
actual one-head weak-reference threshold scope.

The re-ranked diagnostic outputs are RoentGen-v2 seed 1 + LLaVA-Rad for
`case_035`, and PixArt seed 1 + CheXagent-2 for `case_045`. Their displayed
balance score is 83.33, but it uses a different, much narrower available-label
scope from the original 48-row audit. **Do not compare that number with the
old balance score as an improvement, designate these outputs clinically best,
or use this run to authorize EHR-CXR repair.** The original outputs and choices
remain unchanged. The immediate next gate is a separately fixed real validation
cohort with adequate explicit positive/negative evidence for the relevant
findings, followed by independent clinical evaluation; do not lower the support
requirement post hoc to force every head on.

References remain report-derived weak labels, the test pilot was previously
inspected descriptively, and the historical real adapter lacks a recorded
adapter-file hash. Its protocol is reconstructed from recorded XRV library
fingerprints and the inspected implementation, checked against the synthetic
scorer fingerprint, with that limitation retained in provenance. No model
weights are fitted; this is operating-point selection, not probability or
clinical calibration. No new image/report inference was performed. Seven
synthetic-fixture tests cover this cached runner, and seven existing calibration-
contract tests pass. Every subsequent batch submission still requires its full
script/resource request and explicit approval.

### Completed new-patient reference coverage audit

`real_validation/audit_reference_coverage.py` and
`slurm/10_real_reference_coverage_cpu.sbatch` completed as explicitly approved
job **12548196** (1 CPU, 8 GiB, no GPU; 5 seconds, exit 0).
This CPU-only step reads linkage metadata and official report-derived finding
labels inside Slurm, not images, report text, or EHR clinical fields. It verifies
the earlier real-XRV source hashes, excludes every previously scored patient
(including that patient's other studies), verifies the original train/val/test
patient boundaries, and selects one AP/PA study per remaining patient by a
label-independent deterministic hash. It then selects a bounded, deterministic
label-stratified cohort within each existing split, targeting at least 20
explicit positives and 20 explicit negatives per supported finding. Unknown
and uncertain labels remain separate. At most 512 patients per split are
allowed, and the new image-side pilot is gated on pneumonia quotas in **both**
validation and test. Unsupported findings are reported, not silently filled.

The retained output is a new protected
`real_validation/real_reference_coverage_12548196/` run with aggregate
`summary.json/md`, a hash manifest, and protected internal `cohort.json` row
references. No raw patient key or source artifact path is serialized into the
cohort. A successful quota check only authorizes planning another separately
approved scorer job, not automatic GPU submission or targeted regeneration.
The cohort is label-stratified, not a prevalence sample, and references remain
weak; no paper-primary or independently adjudicated final cohort is claimed.
Nine synthetic-fixture tests cover deterministic selection, patient and split
exclusion, binary quotas and budget failure, missing/uncertain references,
label-independent study choice, and the Slurm-before-real-read guard.

The frozen cohort contains **103 validation patients and 105 test patients**,
all disjoint from the prior 256-patient scorer pilot. Every one of the eight
available reference findings has at least 20 explicit positive and 20 negative
labels in both selected splits. Pneumonia has 24 / 22 validation and 22 / 21
test references; consolidation reaches exactly 20 / 20 in both. The cohort
SHA256 is `7c097915fda822876135b6e4a43b8db485bbab9f3f2c1d92fd33fa4ea4cc9ba0`.
This clears the **reference quantity** gate only, not classifier performance or
independent image-ground-truth validation. No CXR pixels were opened by this job.

`real_validation/score_fixed_reference_cohort.py` and
`slurm/11_fixed_reference_xrv_p100.sbatch` were shown in full with the resource
request and explicitly approved, then submitted as job **12548713**.
The job requests one P100, 2 CPUs, 8 GiB RAM and a 15-minute upper limit.
At submission it was pending scheduler priority; completion and test performance
remain to be verified. The reviewed batch-script SHA256 is
`abaae3e3f068103932f0f116363d92f876b6871316992fc17eaa7e0aacff60df`.
It reconstructs and hash-validates the fixed selection before image reads,
rejects prior/duplicate images, applies the same frozen XRV checkpoint to the
208 real CXRs inside Slurm, and saves protected individual scores plus an
aggregate summary. Unlike the earlier pilot, recorded peak VRAM includes
model loading and the adapter/protocol fingerprints are explicit. This is a
label-stratified weak-reference pilot; possible scorer-pretraining overlap is
not ruled out. No real report text or EHR is sent to the scorer or an API.

The cached-threshold runner now accepts an optional, hash-bound `--fixed-cohort`
for this new 103/105 split. The historical default still requires exactly
128/128. Fit uses validation only; test is held out from fitting, although its
label quotas have already been audited. The job fitted per-finding operating
points and relabel the existing 12 synthetic XRV caches without regenerating
anything; it does **not** select new winners or start a repair loop. Review
test sensitivity/specificity and weak-reference discrimination before another
reranking run. Nine synthetic-fixture tests cover the new fixed-cohort runner
and calibrated protocol/count compatibility; previous tests remain passing.

At the next resource check, normal `gpu` P100 slots were no longer free and
job `12548713` remained pending priority, with a scheduler estimate around
19:43 PDT (an estimate, not a guarantee). `debug` still showed two idle P100s.
`slurm/12_fixed_reference_xrv_debug_p100.sbatch` is a prepared alternative
with the **same** GPU/CPU/RAM/time request and computation, changing only the
partition to `debug`; its SHA256 is
`159b1a264df949270ab35e7c98b95840011ca7cb9b9b2c99db0c87503ff612f0`.
The complete replacement script was shown and explicitly approved. The old
job **12548713** was confirmed pending, canceled with `scancel --state=PENDING`,
and confirmed canceled at zero elapsed compute time. Replacement job
**12549079** was submitted and started immediately on `debug / e23-02`,
requesting one P100, 2 CPUs, 8 GiB and the unchanged 15-minute limit.
There was no duplicate running inference. Job **12549079 completed in 1m34s**
with exit code 0. The fixed 208-image XRV pass took 87.286 seconds; cached
threshold fitting completed afterward. The retained protected runs are
`real_validation/real_xrv_reference_12549079/` (individual scores and aggregate
summary) and `real_validation/real_xrv_thresholds_12549079/` (thresholds,
aggregate held-out-from-fitting results and relabeled synthetic caches).
All new output directories/files have mode `2770`/`0660` with inherited
project-group ownership; the previous generation/selection runs are unchanged.

| Finding | Test weak-reference AUROC | Test BA, default 0.5 | Test BA, validation-fitted |
|---|---:|---:|---:|
| Atelectasis | 0.754 | 0.510 | 0.707 |
| Cardiomegaly | 0.788 | 0.688 | 0.766 |
| Consolidation | 0.850 | 0.625 | 0.725 |
| Edema | 0.775 | 0.585 | 0.658 |
| Lung opacity | 0.741 | 0.559 | 0.716 |
| Pleural effusion | 0.899 | 0.682 | 0.810 |
| Pneumonia | 0.643 | 0.624 | 0.585 |
| Pneumothorax | 0.673 | 0.525 | 0.561 |

Eight findings now have sufficient explicit reference quantity to fit an
operating point; that does not mean all eight are reliable clinical verifiers.
Pneumonia sensitivity fell from 0.7727 to 0.4091 while specificity rose from
0.4762 to 0.7619. Pneumothorax discrimination also remains weak. **Do not tune
thresholds again on these test results, automatically flip back to 0.5, or
authorize targeted repair from these heads.** The test macro AUROC is 0.7653
over eight findings, a descriptive weak-reference score, not clinical accuracy.
All thresholds remain `primary_metric_eligible: false` pending independent
image evidence and clinical review. No test-informed reliability mask or new
selection rule has been fitted.

On the identical eight-head subset of the 12 synthetic images, positive calls
changed from 64/96 to 40/96. This is a verifier decision change, not improved
generated image quality or proof that all saturation is fixed. Six unavailable
reference heads remain unknown. The cached labels were subsequently used for
the separately approved CPU-only **diagnostic** 48-candidate re-score below.

`slurm/13_cached_eight_head_rescore_cpu.sbatch`
(1 CPU, 4 GiB, 5-minute upper limit, no GPU) was shown in full and explicitly
approved, then completed as job **12550488** in **2 seconds**, exit code 0.
It verifies
the pinned registry, relabeled XRV cache and aggregate validation hashes,
reuses the immutable CheXbert/BioViL evidence, recomputes the three edges, and
applies the same diagnostic lexicographic policy to the 48-row/two-case bank.
The retained output is `eight_head_rescore_12550488/` under the October V1.1
evaluation root, not a replacement of old runs or the 80-case bank. There are
48 unique candidate triples, 12 unique synthetic CXRs and two selected triples;
each fixed EHR has 24 candidate triples. No image/report model was called again.

`benchmarks/annotate_scoring_scope.py` produces a companion `scoped/` candidate
CSV with unchanged score values/winner flags and explicit
`primary_metric_eligible=false`, `targeted_repair_approved=false`, and pending
independent-clinical-evaluation fields. Its protected verifier profile attaches
the aggregate per-finding test AUROC, operating-point performance and scope.
It does not fit a reliability mask, change thresholds, alter a selection or
declare the selected pair clinically best. Five invented-fixture tests verify
unchanged scores, rejection of patient-record inputs/forged primary claims,
cohort-count consistency and absence of test-informed mask fitting.

| Method, identical eight-head scope | Diagnostic balance | Support / known facts | Hard contradictions / known facts | Coverage |
|---|---:|---:|---:|---:|
| Predeclared Sana seed 0 + MAIRA-2 | 52.5 | 0.20 | 0.15 | 0.35 |
| Exhaustive lexicographic selection | 70.0 | 0.40 | 0.00 | 0.40 |

The displayed balance is `50 * (1 + (supported - contradictions) / known)`;
it is not the selection objective or a correctness probability. Selection
still prioritizes hard gates, contradiction counts, direct EHR support,
CXR-report support, structure, runtime and deterministic ID. Generation costs
remain 6 CXR + 24 report calls per case versus 1 + 1 for the fixed baseline;
the two-second cache replay is not the cost of generating those candidates.

**The ranking conflict remains unresolved.** One selected pair still has
secondary BioViL-T cosine -0.020 versus 0.838 on its fixed path, and its
EHR-report edge has zero comparable facts. Unknown is not a contradiction,
but zero contradictions is not full evidence of correctness either. Across
the 12 images, the label-based selector and BioViL-T select the same top report
on only 3/12 images. Both fixed EHRs contain pneumonia, whose weak-reference
test AUROC is 0.643 and fitted sensitivity 0.409. This result therefore shows
that the cached scoring/selection contract runs, **not** that clinical quality
has improved or that error localization/targeted repair is now safe.

The reusable artifacts are `scoped/candidate_score_table.csv`,
`scoped/verifier_validation_profile.json`, and
`selection/selected_triples.jsonl` inside the new run. All 23 manifest-listed
artifact hashes were checked, and all 47 run entries passed protected
directory/file mode and inherited project-group checks. The inherited V1.1
selection summary still says uncalibrated labels; the actual scope is
validation-fitted operating points against weak report-derived references,
**not** probability or independent clinical calibration. The scoped companion
profile carries that provenance without changing the original summary.
Independent within-case evidence and clinical adjudication are the next gate;
no test-driven threshold/mask changes, agent or repair loop were implemented.

### Completed modality-separated secondary verifier

`benchmarks/verify_candidate_findings_qwen.py` reuses the existing local frozen
Qwen2.5-VL checkpoint and compatibility loader without downloads or changes to
the external model/environment. Its approved-allocation guard precedes artifact
loading and GPU-framework imports. The bounded run performed **12
image-only and 48 report-only calls**: no prompt contains both modalities, the
EHR, candidate/model IDs, old scores or winner flags. Every image is assessed
once, then reused for its four report comparisons. The extraction scope is the
same eight named findings as the current weak-reference pilot, not a complete
clinical ontology; devices, location, severity and temporal facts are outside
this extraction contract.

Responses must contain all eight named keys with
positive/negative/uncertain/unknown states. Missing keys, duplicate keys,
positional vectors, extra fields or token-limit exhaustion make evidence
unavailable, never negative or an invented perfect score. There is no scalar
LLM score or response-based automatic retry. Model-file hashes, input artifact
hashes, adapter/prompt versions, runtime and peak allocated VRAM including
model loading are recorded only in new protected outputs.

`benchmarks/analyze_qwen_evidence.py` joins these secondary states to the
unchanged 48-candidate selection and existing grounded EHR states. It exports
`candidate_evidence.csv` with the original scores/winner flags preserved,
separate support/contradiction/coverage for all three edges, and XRV-image /
CheXbert-report state disagreements. Fixed-versus-selected comparisons remain
descriptive. It does **not** treat Qwen as clinical ground truth, establish
pretraining/data independence, re-rank candidates, fit thresholds/masks or
authorize a repair action.

`slurm/14_separated_qwen_review_a40.sbatch` was shown in full and explicitly
approved, then submitted as job **12553939**: one A40, two CPUs, 32 GiB host RAM,
20-minute upper limit on `gpu`. The submitted script SHA256 is
`b4a9c88f78dc90b2d3db5c536c445e7c74c38e5ee89b7c5f4ff39384446af510`.
The initial scheduler check reported pending priority; the job subsequently
ran on `a03-06` from 21:31:37 to 21:35:13 PDT and **completed in 3m36s**, exit 0.
The main verification program took 210.548 seconds, including checkpoint
fingerprinting; model loading/inference took 182.185 seconds. Peak PyTorch
allocated VRAM including model loading was **15.642 GiB**, not total driver
memory or reserved-memory usage. The queued request was not canceled, changed
or duplicated because completion was established before the proposed GPU switch.
The existing unquantized weight file is approximately 16.6 GB, so this request
leaves substantially more activation-memory margin than a 16-GiB P100.
The latest resource snapshot showed planned/mixed A40 slots, not guaranteed
immediate allocation. Synthetic input lineage/hash checks passed without
inference; 13 new invented-fixture tests and all **80 V1.2 tests** passed after
including the existing bridge/SynEHRgy import paths below. Output is protected
under `verification_runs/qwen_separated_12553939/` and its `_analysis` companion
inside `artifacts/protected/tricompose_v1_2/`. All 12 image responses and 47/48
report responses met the strict named-finding contract. One report-side Qwen
response failed parsing without reaching its token limit; its eight states
remain unknown/unavailable, and the original generated report is retained.
All 48 candidate rows remain represented. The five manifest-listed output
hashes, protected modes/group and preservation of all old score/winner fields
were checked.

| Original method | Qwen CXR-report known / comparable | Supported | Contradiction signals | Coverage |
|---|---:|---:|---:|---:|
| Fixed Sana seed 0 + MAIRA-2 | 14 / 4 | 3 | 1 | 0.286 |
| Original static selections | 13 / 8 | 6 | 2 | 0.615 |

The selected pairs gain Qwen support/coverage but also have more contradiction
signals. On comparable facts the signal rate is 1/4 versus 2/8 (both 0.25);
different coverage prevents declaring either method clinically superior.
Both selected EHR-CXR and EHR-report edges have zero comparable direct facts
under the separate Qwen states; uncertain is not negative, support or success.
Across all twelve images, Qwen and XRV have opposite explicit states in 28/80
comparable image-finding pairs (35%). That measures evaluator disagreement,
not the error rate of either evaluator. The original zero-contradiction static
choices therefore remain **diagnostic, not clinically verified**. No winner,
threshold, mask or candidate artifact was changed, and no repair loop was run.

### Completed cached per-finding evidence contract

`benchmarks/build_finding_review.py` is a bounded **metadata-only** follow-up;
it does not run an evaluator, open image pixels/report text, compute a new
clinical score, change a winner or execute an action. It verifies the preceding
Qwen analysis, source CSV/JSONL, cached finding vectors and staging-fact hashes.
It reuses the existing `DIRECT_EHR_TO_CHEXPERT` mapping rather than interpreting
raw EHR again or introducing new clinical rules. Each non-unknown direct EHR
fact must retain evidence IDs and source-field pointers; known states without
provenance are refused. Weak CHF/medication context remains excluded.

The immutable protected result is
`artifacts/protected/tricompose_v1_2/diagnostics/qwen_finding_review_12553939_001/`:

- `fact_evidence.jsonl/csv`: **384 rows = 48 triples × 8 findings**. Each row
  retains five states (EHR, XRV, CheXbert, image-only Qwen, report-only Qwen),
  support/opposition/unknown/not-comparable relations, missing-evidence reasons,
  fixed-EHR and artifact hashes, staged EHR evidence IDs/source fields, and a
  deterministic finding evidence ID.
- `review_items.jsonl`: 48 candidate-level review records with triggering
  evidence IDs; no confirmed faulty modality or regeneration authorization.
- `adjudication_template.jsonl`: 48 pending review placeholders, **not gold
  labels or an independently adjudicated benchmark**.
- `summary.json/md`: descriptive counts. Image-only edges use 96 unique
  image-finding pairs rather than 384 repeated report-linked rows.

All 48 triples have at least one image-evaluator opposition signal somewhere
in their shared image. This is **not 48 clinically failed samples**: XRV and
Qwen may disagree, and four reports reuse each of twelve images. Report-label
extractors explicitly disagree on only two candidate-finding rows; their much
larger unknown/uncertain region must not be counted as agreement. The two
original winners each retain one Qwen image/report opposition in the protected
summary, and neither is automatically attributed to an image or report error.
Clinical severity and confirmed faulty modality remain unset. A direct EHR
finding is an existing structured-clinical proxy, not independent image truth.

The run took under one second of lightweight cached-metadata work and needed
no Slurm/GPU submission. All six output artifact hashes, bound source hashes,
group ownership and modes passed checks. Ten new invented-fixture tests cover
deterministic evidence IDs, EHR provenance, immutable state/hash bindings,
unknown/uncertain/unavailable semantics, image-side deduplication and refusal
to assign fault from correlated or contradictory evaluators. All **90 V1.2
tests** pass. This completes a traceable fact/review table, not a trained router,
clinical localization validation, blinded review, or targeted repair.

### Completed selected-disagreement engineering audit

`benchmarks/audit_selected_disagreements.py` follows the cached finding table
without inference or interpretation of report text/image pixels. It reproduces
the previous named-state join, checks generation manifests and actual synthetic
artifact byte hashes, replays all frozen XRV finding decisions, and checks
recorded Qwen adapter/prompt fingerprints and response token metadata.
It neither recalibrates a threshold nor changes historical scores/winners.

The protected result is
`artifacts/protected/tricompose_v1_2/diagnostics/selected_disagreement_audit_12553939_001/`.
The two original selected-pair Qwen opposition rows passed these engineering
checks; their responses were complete and not token-limited. `selected_review.json`
retains evidence IDs, artifact pointers/hashes, finding states and exact frozen
operating points. `peer_report_states.csv` contains eight related report rows
on the same two images; these are correlated descriptions, not independent votes.
`summary.json/md` records the completed checks and unresolved clinical boundary.

No clinical fault has been assigned. An XRV score's distance from its cutoff
is not confidence or a calibrated error probability. Qwen raw response text was
not retained, so this follow-up cannot independently reparse it or audit its
semantic reasoning. Missing EHR evidence and CheXbert unknown remain unavailable,
not negatives. The next clinical gate still requires independent artifact
review and controlled artifact-level interventions, not majority voting or
automatic regeneration. Nine new invented-fixture tests pass; all **99 V1.2
tests** now pass. No new Slurm submission was needed for this bounded audit.

### Completed minimal report-polarity diagnostic

`benchmarks/build_report_polarity_benchmark.py` preserves the two selected
synthetic images/EHRs and creates separate protected report copies. It accepts
only one isolated, explicitly affirmative finding sentence, refuses mixed
findings/negation/uncertainty/repeated mentions, and records a reversible edit
with unchanged prefix/suffix. Opacity is not promoted to consolidation to
manufacture an eligible edit. This whitelist is a narrow construction contract,
not a new clinical extractor or an assertion about the image.

The immutable bank is
`artifacts/protected/tricompose_v1_2/benchmarks/report_polarity_smoke2_12553939_001/`:
two byte-identical original controls, two token-preserving whitespace controls,
one explicit-positive-to-negative report edit, and one unavailable edit with
its rejection reason. All five report copies, the resolver, intervention key,
summary and source hashes passed protected permission/hash checks. The original
EHRs, CXR/report files, scores and winners are unchanged.

`benchmarks/score_report_polarity.py` runs frozen CheXbert and
BioViL-T in separate existing environments. Its inference guard precedes model
and data access. Evaluators read only a hash-checked allowlisted resolver, not
the intervention key; the answer key is consumed only by the subsequent cached
analysis. It measures target polarity extraction, non-target label changes,
whitespace invariance and raw cosine differences. An unchanged original is not
clinical gold, and a cosine decrease is not assumed to be the correct direction.
This is a one-edit diagnostic, not clinical localization accuracy or an
independent validation of the original winner/image. No thresholds, ranking,
or repair actions are changed.

The approved `slurm/15_report_polarity_debug_p100.sbatch` completed as job
`12558784` on debug node `e23-02` (P100) in **49 seconds**, exit 0. Its request
was one GPU, two CPUs, 16 GiB host memory and a ten-minute wall-time cap;
the cap was not expected runtime. Outputs are under
`artifacts/protected/tricompose_v1_2/verification_runs/report_polarity_12558784_{chexbert,biovil,analysis}/`.

| Mechanical check | Result | Scope |
|---|---|---|
| Explicit target positive-to-negative report edit | CheXbert positive -> negative; no other head changed | One eligible polarity pair, not clinical detection accuracy |
| Whitespace-only controls | Identical CheXbert vectors in both cases; BioViL cosine deltas 0 | Token-preserving formatting invariance, not clinical agreement |
| BioViL response to the polarity edit | Cosine delta -0.02515189 | Sensitivity only; the decrease is not proof that the edited pair is wrong |
| Second planned target edit | Unavailable; no explicit consolidation phrase | Kept in the denominator; no invented target statement |

CheXbert processed five unique reports in 17.128 seconds and BioViL-T processed
two images/five reports in 21.094 seconds, including loading and output-related
work. Their peak PyTorch allocated VRAM **including loading** was 1.227/0.570
GiB; these are not total driver-reserved GPU memory measurements. All three
stages completed, source/output hashes and protected modes/group passed checks,
and original reports/images, source-score files and winner flags are unchanged.
No Qwen inference, EHR/CXR generation, real-target access or model download was
included. Twelve new invented-fixture tests and all **111 V1.2 tests** pass;
the batch script passes `bash -n`. Clinical artifact review remains pending,
and neither a repair policy nor clinical localization has been validated.

### Completed development-only expanded polarity diagnostic

`benchmarks/prepare_expanded_polarity.py` first froze a metadata-only case plan
before opening report text. It inherits the historical 48/16/16 development,
calibration and final-test roles, fixes Sana seed 0 -> CheXagent-2, and reads
only the **48 development reports** semantically. No scores, winner flags or
construction availability choose cases. The other 32 reports are not opened
as text, and there is no new EHR generation or threshold fitting.

The case plan and staged bank are respectively
`artifacts/protected/tricompose_v1_2/benchmarks/report_polarity_dev48_plan_20261002_001/`
and `report_polarity_dev48_20261002_001/` under the same parent. The constructor
tries all eight named findings in every fixed case: **384 attempts**. It accepts
only an isolated explicit finding assertion, changes its polarity in either
direction, preserves unrelated bytes/header text, and retains all rejections.
Qualified absence such as "no large effusion" is not global absence. Missing,
uncertain, mixed or repeated statements are not rewritten into invented facts.

The bank contains **193 report copies**: 48 byte-identical original controls,
48 token-preserving whitespace controls and 97 eligible edits. Eligible targets
are cardiomegaly (17), consolidation (1), pleural effusion (40) and pneumothorax
(39); the other four heads have no constructible edit. All **287 unavailable
attempts** retain reasons. They describe construction coverage, not model errors
or absence of disease. There are only **30 distinct image hashes, 26 original
report hashes and 100 report-copy hashes**; neither 48 case records nor 97 edits
should be called independent clinical examples.

`benchmarks/score_expanded_polarity.py` reuses the frozen CheXbert mapping and
BioViL-T loader, with bounded classifier batches of eight. Scorers see only a
hash-bound allowlisted resolver, not intervention labels. The subsequent cached
analysis reports case-weighted and deduplicated finding/text-pair fractions,
formatting invariance, non-target state changes and raw cosine differences.
There is no new clinical score, calibrated threshold, winner or repair action.
This historical development pool was already used for static evaluation; it is
not independent final-test evidence or clinically adjudicated gold.

The explicitly approved `slurm/16_expanded_polarity_debug_p100.sbatch` completed
as job **12560507**, exit 0, on debug node `e23-02` (P100) in **54 seconds**.
Its request was one GPU, two CPUs, 16 GiB host memory and a ten-minute cap;
the cap was not expected runtime. Outputs are under
`artifacts/protected/tricompose_v1_2/verification_runs/report_polarity_dev48_12560507_{chexbert,biovil,analysis}/`.

The table measures whether CheXbert extracted **both** the original and edited
explicit text assertions, not whether either report correctly describes its
image. Deduplication uses the original/edited text-hash pair within each finding.

| Finding | Eligible case-linked pairs | Both states extracted | Case-linked fraction | Unique text pairs | Successful unique pairs | Deduplicated fraction |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 17 | 16 | 94.1% | 6 | 5 | 83.3% |
| Consolidation | 1 | 1 | 100.0% | 1 | 1 | 100.0% |
| Pleural effusion | 40 | 15 | 37.5% | 21 | 12 | 57.1% |
| Pneumothorax | 39 | 7 | 17.9% | 20 | 6 | 30.0% |

The other four planned findings remain **unavailable**, not failed evaluations.
Overall, 39/97 case-linked pairs and 24/48 unique finding/text pairs met the
two-state extraction check; the direction mix is imbalanced (17 affirmative-to-
negative versus 80 negative-to-affirmative edits). These totals are exploratory
development diagnostics, not independent clinical accuracy estimates. The
single consolidation pair cannot establish general reliability.

All **48 whitespace controls** had identical CheXbert state vectors and zero
BioViL cosine differences at the saved precision. Original assertion extraction
matched in 96/97 eligible pairs. For pleural effusion, 24/40 edited reports still
received a negative label; for pneumothorax, 32/39 did. Changes in non-target
heads occurred in 41/97 pairs, including support-device state changes in 36
pairs. These are extraction/context-sensitivity findings, not evidence of 41
clinical errors or changes to the original images. Neither score adapters nor
the text construction protocol were changed after inspecting these results.

Raw BioViL cosine changes are recorded in `analysis/per_intervention.csv` and
JSON. Their direction is **not** scored as clinical correctness: the original
image/report pair lacks independent clinical adjudication. This diagnostic does
not establish CheXbert as a standalone trigger for automatic localization or
regeneration; extraction failures must be investigated before relying on that
decision rule. Existing winners, thresholds and repair permissions are unchanged.

CheXbert processed 100 distinct report texts in 13 batches (18.565 seconds,
peak PyTorch allocated VRAM including loading 1.227 GiB). BioViL-T processed
30 distinct images and 100 texts (23.453 seconds, peak allocated VRAM including
loading 0.603 GiB). Peak figures are not driver-level/reserved-memory measurements.
All seven output artifact hashes and run file/directory project permissions
passed validation. Nineteen new invented-fixture tests and all **130 V1.2 tests**
pass. Original source artifacts and the completed two-case diagnostic remain
unchanged. No generation, Qwen inference, training, threshold fitting, real-data
access, selection or repair was part of this job.

### Completed read-only polarity audit and context/batch diagnosis

The follow-up audit reconstructed all **97 edits** from the immutable controls;
unrelated bytes were identical. The loaded head order and four-state indices
agree with [Stanford's constants](https://github.com/stanfordmlgroup/CheXbert/blob/master/src/constants.py)
and [CSV class conversion](https://github.com/stanfordmlgroup/CheXbert/blob/master/src/label.py).
Tokenizer-only checking of the 100 distinct synthetic texts found **zero token
ID differences** between the current wrapper preprocessing and the official
[CSV tokenizer path](https://github.com/stanfordmlgroup/CheXbert/blob/master/src/bert_tokenizer.py).
Lengths were 55--141 tokens, not near the 512-token limit. No model initialization
or inference ran on the login node. These checks exclude the observed mapping,
edit-reconstruction and token/truncation explanations, but do not independently
authenticate the checkpoint origin or prove the remaining root cause.

`benchmarks/diagnose_chexbert_polarity.py` is a **diagnostic**, not an
evaluator replacement. It uses all 100 frozen report-copy texts, all 97 eligible
case-linked edits and their 15 distinct original/edited target sentences. It
does not choose only failures, open the reserved 32 cases, or modify any sources.
Under an approved Slurm GPU allocation it will:

- Reproduce the existing wrapper labels and compare its argmax with the
  instrumented frozen-head logit path.
- Compare full-report batch sizes eight and one, and check exact token-ID parity
  with official CSV preprocessing.
- Contrast full-report and isolated-target-sentence states, retaining every
  successful, failed and unknown extraction and both duplicate-aware denominators.
- Store raw logits/margins as diagnostics, not calibrated clinical confidence.

This context ablation intentionally reads the protected construction key to
obtain target sentences, but never passes expected labels, findings, case IDs or
prior scores into the model. It is not a blinded or independent final-test
benchmark. Sentence-only scores must not replace the current report score, and
neither context sensitivity nor batch stability establishes image truth or
clinical error localization. No selection, thresholds or repair rules change.

The explicitly approved `slurm/17_chexbert_context_debug_p100.sbatch` completed
as job **12561407**, exit 0, on debug node `e23-02` (P100) in **19 seconds**.
The request was one GPU, two CPUs, 16 GiB host memory and a five-minute cap,
not a five-minute runtime estimate. Outputs are under
`artifacts/protected/tricompose_v1_2/verification_runs/chexbert_context_dev48_12561407/`.

All 193 cached item labels reproduced exactly; none of the 100 distinct report
texts changed labels between batch sizes eight and one. The instrumented
logit-path argmax matched the existing wrapper, with zero official CSV token-ID
mismatches. Thus the observed misses are reproducible and are not explained by
these checked batching, interface, cache or preprocessing differences.

| Finding | Case-linked pairs | Full-report both-state extraction | Isolated-sentence both-state extraction | Unique report text pairs | Full-report deduplicated successes | Isolated deduplicated successes |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 17 | 16 | 17 | 6 | 5 | 6 |
| Consolidation | 1 | 1 | 1 | 1 | 1 | 1 |
| Pleural effusion | 40 | 15 | 40 | 21 | 12 | 21 |
| Pneumothorax | 39 | 7 | 39 | 20 | 6 | 20 |

Overall full-report versus isolated-sentence extraction was **39/97 versus
97/97** case-linked pairs and **24/48 versus 48/48** unique finding/report-text
pairs. There are only **15 distinct isolated sentences**; the 97 intervention
pairs are not independent clinical examples. Every pair retained its original
denominator and direction. There were 57 edited assertions that matched only
when isolated; one additional pair failed only the full-report original
assertion check. No edited assertion failed in both contexts.

This controlled ablation supports input-context/representation sensitivity as
an important factor: the model can extract these simple assertions in isolation,
but full-report outputs can differ. It does **not** establish which surrounding
sentence, header, position or semantic conflict causes each difference. It also
does not rule out conflicting or synonymous assertions elsewhere in an edited
report, nor prove that the isolated output is clinically correct. Do not turn
the 100% toy-sentence result into a clinical accuracy claim, replace full-report
scores with isolated ones, tune thresholds, or automatically regenerate a CXR.
The next gate is to inspect report-level semantic conflicts and validate
evidence-scoped extraction independently before changing a selection rule.

The program took 13.594 seconds including loading/output work; peak PyTorch
allocated VRAM including loading was 1.227 GiB. Its 315 encoder examples include
the diagnostic/wrapper double pass, batch-size replay and sentence ablations;
they are not 315 unique reports or clinical cases. Output/source hashes,
protected project modes/group and in-memory cached-analysis replay passed.
Twelve new invented-fixture tests bring the lightweight V1.2 suite to **142
passing tests**. Original artifacts, primary scores, winners and repair
permissions remain unchanged; no additional GPU task was submitted.

### Completed evidence-scoped full-report follow-up (secondary diagnostic)

The follow-up uses the existing frozen local Qwen2.5-VL-7B checkpoint; it does
not download RadGraph or a new model. The local dependency audit does not show
an available RadGraph checkpoint. A coarse read-only text-structure check found
remaining negation cues in all 97 edited full reports, but negation about a
different disease does not constitute a same-finding contradiction. Counting
words or repeated exact disease names cannot establish report-level semantics.

`benchmarks/verify_report_evidence_qwen.py` is a secondary, report-only
development diagnostic with exactly four previously testable findings:
cardiomegaly, consolidation, pleural effusion and pneumothorax. It reads the
immutable blinded resolver, deduplicates the 193 report copies to **100 texts**,
and makes one greedy, frozen call per distinct text. No image, EHR, intervention
key, expected state, previous score or winner is passed to the model. The 32
reserved cases remain unopened. This is not an independent final evaluation.

Each positive/negative/uncertain assertion must have a short, exact, contiguous
source quote and unambiguous Unicode-character offsets. Missing assertions stay
unknown; quoted positive and negative assertions are retained as a conflict
signal rather than majority-voted away. Invalid JSON, duplicate keys, invented
quotes, ambiguous offsets, incomplete output or the token limit make the whole
response unavailable, with all four states unknown and the failure denominator
retained. Protected raw responses are saved for parser replay. Each polarity is
capped at two quotes: neither an empty array nor exact quote alignment proves
semantic correctness or exhaustive absence of another assertion.

Only the separate CPU-only `benchmarks/analyze_report_evidence_qwen.py` reads
the construction key after inference. It verifies raw-response hashes and exact
parser replay, reconstructs every edit, and aligns quotes to the known target
span versus other report positions. It compares those evidence patterns with
the existing full-report CheXbert diagnostic while retaining all 97 pairs and
duplicate-aware counts. Expected-polarity quotes in the target span and
opposite-polarity quotes elsewhere are **model-attributed evidence**, not
independently adjudicated clinical labels or an established explanation for
CheXbert's behavior. Absence of a recorded quote does not establish absence of
a conflicting assertion. No current primary score, threshold, selection,
generator or repair rule is changed.

`slurm/18_report_evidence_qwen_flexible_gpu.sbatch` was shown in full with its
resource request and explicitly approved before submission as **job 12562445**
on 2026-10-02: one GPU from V100/A40/A100/L40S, two CPUs, 32 GiB host memory, twenty-minute
wall-time cap. The runtime enforces a conservative 24 GiB visible-VRAM guard
(not a measured minimum requirement) and records
actual GPU/dtype; V100 uses FP16 whereas compatible newer devices can use
BF16, so bitwise cross-device equivalence is not claimed. The existing PyTorch
build supports V100's architecture. The last resource check showed only
drained A40/A100 nodes and free P100s; a compatible request can still queue.
The cap is not a runtime or queue-time promise. The initial scheduler check
reported **PENDING / Priority**, with a provisional 06:31:34 local start time;
at that check, no diagnostic results existed. Full
script/resource disclosure and explicit user approval remain required before
any additional submission. The submitted script SHA256 is
`d7a666711ed798c3250263d3294b8786f8cf1c9e2bde4e29ccf9ed71ec6a08c6`;
all eleven pinned input/code checks and the shell syntax check passed.

After the user explicitly approved an in-place queue switch, the existing
pending job was updated at approximately **03:08 PDT** using
`scontrol update JobId=12562445 Partition=debug Features=a40`. The live
allocation request was changed to **debug / A40**, still one GPU, two CPUs, 32 GiB
host memory and a twenty-minute cap. No job was canceled or resubmitted, and
the submitted script body, model, prompts, code hashes and output IDs are
unchanged. The last observed pre-run scheduler estimate was **03:46:56 PDT**,
with state **PENDING / Resources**. The earlier test-only estimate of 03:31
did not create a second job. Neither estimate was an actual start time:
accounting now confirms execution from **03:22:11 to 03:30:10 PDT** on
`b11-09`, **7m59s**, **COMPLETED / exit 0**. No second submission was needed.

P100 is not a drop-in replacement for the current Qwen runtime: the installed
PyTorch `2.9.1+cu128` build reports compiled architectures beginning at
`sm_70`, without P100's `sm_60`. This was a lightweight build-metadata check
with CUDA uninitialized, not login-node inference. The earlier A40 run's
15.642 GiB allocated peak also excludes driver/reserved-memory overhead and
does not establish safe operation on a 16-GiB P100. A P100 migration would
need a separately validated compatible environment and memory plan; neither
was implemented or submitted during this queue switch.

The completed, non-overwriting protected outputs are:

```text
artifacts/protected/tricompose_v1_2/verification_runs/
  qwen_report_evidence_12562445/
    evidence.json, raw_responses.json, summary.json, manifest.json
  qwen_report_evidence_12562445_analysis/
    pair_alignment.json, summary.json, summary.md, manifest.json
```

Twenty-three additional invented-fixture evidence/alignment tests bring the
lightweight V1.2 suite to **165 passing tests**. This only validates contracts
and bookkeeping; it does not establish semantic accuracy or a working repair
method. No model inference ran on the login node. Submission alone does not
establish successful execution, semantic accuracy or completed results.

The verifier made **100 frozen, greedy text-only calls** using A40 BF16.
Its recorded program runtime was 473.942 seconds including fingerprinting,
loading and artifact writing, with peak PyTorch allocated VRAM including
loading **15.597 GiB**. This is not total driver/reserved-memory use. The
separate alignment program made zero model calls. Six manifest-listed artifacts
passed SHA256 checks; both completed output trees passed project-group modes
`2770`/`0660`. Raw-response parser replay and the entire cached analysis replay
were identical, and the 165 lightweight tests still passed.

**Contract result:** 69/100 distinct texts had complete, exact-source evidence;
31/100 were unavailable because at least one returned quote was not an exact
substring. All failed records remain present with four unknown states, not
negative states. By construction arm, failures were 3/26 distinct unchanged
texts, **26/26 whitespace-only texts**, and 2/48 distinct minimally edited
texts. Thus all 48 case-linked whitespace controls had an unavailable member;
their zero recorded state changes do **not** establish formatting invariance.

A subsequent bounded, read-only cached audit found 118 nonmatching quotes:
105 matched after whitespace folding. In **23/31 failed texts**, every
nonmatching quote was whitespace-equivalent to a source substring; the other
eight texts had at least one mismatch not explained by this check. These
counts diagnose alignment brittleness, not validated normalization, semantics
or paraphrase acceptance. No failed record was silently repaired or promoted.

| Finding | Case-linked edit pairs | Qwen target-polarity quotes captured in both reports | Earlier full-report CheXbert both text states |
|---|---:|---:|---:|
| Cardiomegaly | 17 | 16 | 16 |
| Consolidation | 1 | 0 | 1 |
| Pleural effusion | 40 | 12 | 15 |
| Pneumothorax | 39 | 12 | 7 |

Qwen captured both target quotes in **40/97** case-linked pairs and **23/48**
unique finding/report-text pairs. Three case-linked pairs had an unavailable
report, and 54 complete pairs did not capture the expected target polarity in
both reports. These are text-assertion diagnostics with repeated templates,
not independently adjudicated clinical accuracy or a like-for-like clinical
comparison with CheXbert. There is no clear overall extraction gain: the
earlier CheXbert counts were 39/97 and 24/48, respectively.

**Traceability is not polarity correctness:** the minimally injected positive
effusion and pneumothorax assertions were captured in 40/40 and 39/39 edited
reports. However, in the corresponding source-negative checks, Qwen placed a
quote inside the mechanically negative target span in the **positive** array
for 27 case-linked effusion checks and 27 pneumothorax checks. Each such quote
still contained a negation marker. These are not 54 independent patients or
confirmed clinical errors: each finding's cases share 11 distinct source
reports, with only three and two distinct negative target sentences,
respectively. The parser faithfully retained the model's assigned polarity;
verbatim substring validation does not validate its semantic interpretation.
Consequently the lack of recorded opposite assertions elsewhere cannot clear
the report-conflict gate or establish the cause of CheXbert's behavior.

**Decision:** retain Qwen as secondary, unvalidated evidence and keep all old
scores, thresholds, selections and repair permissions unchanged. The next
small step is a separately versioned, cache-only quote-alignment diagnostic
with explicit original-character mapping, plus checks of negation scope and
multi-finding sentences. Preserve this strict baseline; do not accept arbitrary
paraphrases, replace primary scores, or regenerate an EHR/CXR/report because
of a verifier's text-extraction failure. No additional GPU job, training,
download or external API call was performed during the cached follow-up.

### Completed cache-only evidence interface repair (secondary, not new scores)

`benchmarks/repair_cached_report_evidence.py` implements the approved small
follow-up using the existing 100 frozen Qwen responses. It imports no model
runtime, makes **zero model calls**, and changes no report, CXR, EHR, primary
score, threshold or selected triple. The repair pass sees only the immutable
synthetic source text and cached response; it completes before the separate
post-hoc analysis reads the construction key. Reserved cases remain unopened.

Alignment collapses Unicode whitespace **only to locate** a unique source
substring. Stored evidence is restored to the original, byte-unchanged report
with Unicode-character offsets and both original/returned quote hashes.
Case, punctuation, numbers, units, qualifiers and negation are not normalized;
paraphrases, repeated locations, duplicate normalized spans, malformed JSON,
partial inventories and token-limit failures are rejected. Whole-record
failures retain four unknown states and all denominators.

The separately versioned four-finding literal scope guard can only **veto**
model-attributed evidence. It never moves a quote into another polarity or
manufactures a new negative. Qualified absence and explicit uncertainty cannot
become global absence. Source punctuation, contrast and explicit independent
clauses bound negation; original line structure distinguishes a new presence
clause from a wrapped negation/list. A remaining veto cannot resolve a conflict
into a determinate opposite state. Uncovered synonyms/grammar are explicitly
unchecked, and retained quotes are still not clinically verified.

Scope and pseudo-trigger considerations have precedents in
[NegEx](https://pubmed.ncbi.nlm.nih.gov/12123149/) and
[ConText](https://pubmed.ncbi.nlm.nih.gov/19435614/).
This narrow hand-written veto is **not** either official implementation and
is not the dependency-based [NegBio](https://arxiv.org/abs/1712.05898).
It does not reuse the benchmark constructor's vocabulary/answer key as a
scorer and does not inherit any published clinical performance claims.

The inspected completed diagnostic is:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  qwen_cached_evidence_repair_12562445_002/
    summary.json, summary.md, repaired_evidence.json,
    pair_alignment.json, manifest.json
```

The earlier `_001` cache diagnostic is retained, not overwritten. Its first
scope version over-abstained when whitespace folding erased separate original
lines; `_002` uses `literal-four-finding-veto-v2-original-line-scope` with new
invented regression fixtures. Neither run was used to select or regenerate
data. Refining a guard on this development diagnostic is **not** held-out
validation; independent false-veto/clinical extraction evaluation is pending.

| Stage | Complete source-quote contracts / 100 | Unavailable / 100 | Non-unknown finding states / 400 | Target quotes in both / 97 | Unique target pairs / 48 |
|---|---:|---:|---:|---:|---:|
| Original strict baseline | 69 | 31 | 273 | 40 | 23 |
| Unique whitespace alignment | 91 | 9 | 360 | 40 | 23 |
| Alignment + conservative scope veto | 91 | 9 | 230 | 40 | 23 |

Twenty-two failed contracts were recovered and no previously complete contract
was lost. Eight remaining responses contain non-whitespace mismatches; one
contains duplicate/conflicting normalized source spans. Accepted aligned
evidence includes 275 exact and 89 whitespace-equivalent quotes. The current
guard vetoes 130 quotes: 52 explicit-positive/negation mismatches and 78
uncertainty/qualified-absence mismatches. Its 130 state transitions are all to
unknown, with **zero unknown promotions or polarity flips**. Coverage loss is
visible rather than counted as success. Quote-contract recovery is **not 91%
clinical accuracy**, and target-quote extraction has not improved.

A post-hoc audit confirms that the earlier 27 effusion and 27 pneumothorax
case-linked negative-target quotes assigned positive were all vetoed. These
54 checks involve only 22 unique finding/source-text pairs and repeated
mechanical assertions, not 54 independent clinical mistakes or a validated
false-repair rate.

Whitespace controls are now comparable in **44/48** case-linked checks, with
four unavailable. **23/44** comparable pairs still have different state
vectors in both the alignment-only and guarded stages. This exposes sensitivity
in the cached model outputs; fixing quote provenance cannot fix those frozen
responses or prove that generated images/reports are clinically wrong.
Qwen remains secondary diagnostic evidence, unsuitable as the sole selector
or regeneration trigger. Input whitespace canonicalization with offset mapping
is a possible next controlled model diagnostic, not an implemented cure; any
new inference requires complete Slurm disclosure and explicit approval.

Thirty-nine added invented-fixture tests bring the lightweight V1.2 suite to
**204 passing tests**. They include original-line scope, cropped quotes,
negated lists, independent clauses, qualifiers, ambiguous/duplicate locations,
key-read ordering and non-overwrite refusal. Artifact/program/source hashes,
exact cached replay, protected modes `2770`/`0660` and project-group boundary
all passed. Historical verifier/parser/script and primary-score/winner hashes
remain unchanged. No real patient data, GPU, training, download or external
API was used in this cached repair.

### Completed line-preserving input/cache interface (engineering, not a new clinical benchmark)

`benchmarks/canonical_report_evidence_interface.py` follows the cached repair
with an explicit request-normalization contract. It collapses horizontal
spaces/tabs only, preserves original line breaks, punctuation, case, numbers,
negation and ordering, and never strips or flattens the report. Every input
retains a canonical-to-original Unicode offset map. Quotes are projected to
unchanged source text and the inverse coordinates are checked before the
unchanged conservative scope guard runs.

A read-only feasibility audit found that all-whitespace folding would produce
74 inputs but **zero exact old-input cache hits**, while also deleting original
line boundaries. That is not the chosen interface. Line-preserving horizontal
normalization instead produces **74 exact old-input cache hits** from the 100
source texts: 26 groups of two formatting variants and 48 singleton edited
texts. Every input was already processed in the completed frozen Qwen run, so
this interface experiment required **zero new model calls, no GPU allocation
and no Slurm submission**.

Cache identity binds the completed run's frozen checkpoint/asset fingerprints,
producer revision, fixed request prompt, sampling settings, execution metadata,
exact canonical text and request hashes. A missing/nonmatching canonical input
refuses execution rather than approximately reusing a response or invoking a
model on the login node. A future cache miss needs a separately disclosed and
approved inference job. No score, expected finding, construction key or winner
is supplied to the normalization/projection pass; the key is read only in the
subsequent diagnostic evaluation. All original-format cached responses remain
available as the measured instability baseline.

The completed, replay-checked output is:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  canonical_report_input_cache_12562445_002/
    canonical_inputs.json, source_mappings.json, evidence.json,
    pair_alignment.json, summary.json, summary.md, manifest.json
```

The earlier `_001` output is preserved. `_002` makes group-size metadata keys
explicit JSON strings so both in-memory and persisted summary replay agree;
this bookkeeping correction changes no evidence or clinical conclusion.

| Interface stage | Complete original-source projections / 100 | Complete distinct canonical inputs / 74 | Target quotes in both / 97 | Comparable formatting controls / 48 | Formatting state differences |
|---|---:|---:|---:|---:|---:|
| Exact canonical-input cache + quote projection | 92 | 69 | 40 | 45 | 0 (shared input/cache) |
| Same interface + unchanged scope veto | 92 | 69 | 40 | 45 | 0 (shared input/cache) |

Five distinct canonical inputs still have unavailable evidence contracts; their
eight original-text projections remain present and unknown. Three case-linked
formatting controls are unavailable. **The zero differences are by construction:**
identical canonical inputs share the same old response. They do not independently
measure repeated-inference stability, model invariance, clinical accuracy or
an improvement over the previously observed 23/44 different state vectors.
The mechanical target-quote checks remain 40/97 and 23/48 unique finding/text
pairs. There is no clinical extraction gain, better selected triple or approved
automatic-regeneration trigger. This step establishes reproducible input/cache
plumbing only; independent clinical/scope validation remains necessary.

Twenty-three additional invented-fixture tests bring the lightweight V1.2
suite to **227 passing tests**, covering line preservation, Unicode inverse
maps, exact cache binding, producer pinning, cache-miss refusal, unknown-safe
failures, key-read ordering, JSON-stable metadata and non-overwrite refusal.
All six output artifacts, program/source fingerprints, complete in-memory
replay and protected project modes/group passed. Original verifier/parser,
cached repair, primary score table and selected-triple hashes remain unchanged.
No reserved cases, real patient inputs, external API, download or training were
used. The interface is not wired into the main selector or generation pipeline.

### Completed authored assertion challenge, conditional veto audit and frozen CheXbert diagnostic

`benchmarks/report_assertion_challenge.py` fixes **56 wholly invented short
reports**, independently of the earlier mechanical polarity-edit constructor.
No patients, real targets, previous generated reports, or reserved cases are
used. The four checked heads are cardiomegaly, consolidation, pleural effusion
and pneumothorax. There are **80 designated finding checks** (20 per head),
with 15 authored positives, 22 negatives, 20 uncertain states and 23 unknowns;
the complete 224-entry vectors are secondary diagnostics. Template families
are correlated and **the same project author assigned the expected states**:
this is a language challenge, not independent radiologist annotation or
held-out clinical validation. In particular, qualified absence is conservatively
marked uncertain at the broader finding level, unmentioned/unassessed findings
stay unknown, and opposed assertions stay uncertain. These are disclosed
annotation choices, not universal CheXbert gold labels.

The immutable protected runs are:

```text
artifacts/protected/tricompose_v1_2/benchmarks/
  authored_assertions_20261002_001/
    reports/, resolver.jsonl, references.jsonl, summary.json, manifest.json
artifacts/protected/tricompose_v1_2/diagnostics/
  authored_assertion_veto_20261002_001/
    details.json, summary.json, manifest.json
```

The CPU-only audit freezes the **pre-challenge** literal scope guard at SHA256
`7c96501e71a4f3b3b1cbfc463ab6d03d31e4f532d9bcd0ec86d8773f6c9fbf18`.
For each designated check it supplies the entire invented report as an oracle
quote and probes positive, negative and uncertain attribution: 240 probes in
total. Full quotes may contain multiple assertions. It tests only conditional
veto behavior, **not quote extraction, actual model error, clinical accuracy
or a calibrated false-repair rate**. The rule code is not changed to fit results.

| Conditional rule probe | Count |
|---|---:|
| Exact authored polarity supplied | 57 |
| Exact-polarity probes retained | 56 |
| Incorrect determinate polarity deliberately supplied | 123 |
| Incorrect determinate probes vetoed | 65 |
| Incorrect determinate probes not vetoed | 58 |
| All probes without a checked literal mention | 42 |

This broader diagnostic exposes incomplete scope coverage: the existing veto
must not be promoted into an automatic regeneration trigger. No score,
threshold, candidate or selected triple is replaced. **中文结论：当前规则在
人为构造的否定/不确定性压力测试中仍有漏拦，尚不能安全地决定自动返工；
以上数字不是模型真实错误率，也不是临床验证。**

`benchmarks/score_report_assertion_challenge.py` prepares a separate **frozen
CheXbert** measurement using the existing checkpoint and unmodified CXRMate
wrapper. It loads only invented text, hashes and opaque IDs, never the authored
reference key. It preserves the audited internal mapping (0 unknown, 1 positive,
2 negative, 3 uncertain), which differs from the CSV export convention documented
by the [official CheXbert repository](https://github.com/stanfordmlgroup/CheXbert).
Batch-8 inference is replayed at batch 1 (112 encoder examples, 63 forward
batches); inputs are checked to prevent silent 512-token truncation. Predictions
are committed before separate CPU analysis reads the reference key. Analysis
reports the four-state confusion matrix, per-state F1, macro F1, per-family
and per-head results, hard positive/negative flips and unsafe commitments on
unknown/uncertain states. Unavailable predictions remain in the denominator
and never earn unknown-state credit.

The complete `slurm/19_authored_assertions_chexbert_debug_p100.sbatch` script
and resource request were shown, then explicitly approved. Job **12580901**
completed with exit code 0 on `e23-02`: one debug P100, two CPUs, 16 GiB host
memory and a five-minute wall-time limit. It briefly queued for priority, then
ran for **13 seconds**; the prediction program recorded **10.914 seconds** and
**1.227 GiB peak allocated VRAM including loading**. The approved script SHA256
is `30c595fb9391c34dfa347627c1469e7abf62f84e4b37a052e77804bbcf4a91e7`.
All 56 prediction records completed, with no truncation (6–33 tokens), and
batch 8 versus batch 1 produced **zero changed four-head state vectors**.

The immutable protected outputs are:

```text
artifacts/protected/tricompose_v1_2/verification_runs/
  authored_assertions_chexbert_12580901/
    predictions.json, manifest.json
  authored_assertions_chexbert_12580901_analysis/
    summary.json, details.json, manifest.json
```

| Authored expected state | Designated checks | Exact matches | State F1 |
|---|---:|---:|---:|
| Positive | 15 | 15 | 0.6522 |
| Negative | 22 | 11 | 0.5946 |
| Uncertain | 20 | 8 | 0.4000 |
| Unknown | 23 | 14 | 0.7568 |

Across the **80 designated checks**, 48 matched the predeclared authored state
(60.0%) and four-state macro F1 was 0.6009. There were seven positive/negative
flips (all authored negative → model positive) and 13 determinate commitments
on authored uncertain/unknown states. Literal positive controls and the
no-evidence family each matched 4/4, while postposed absence matched 0/4
(all four became positive); cannot-be-excluded matched 0/4. These small,
correlated language families are not independent clinical samples. Qualified
absence and conflicting-assertion results also depend on the disclosed author
annotation policy. The secondary full-vector result, 188/224 = 83.93%, includes
**159 authored unknown entries** and must not replace the harder designated
checks or be presented as patient-level clinical accuracy.

**中文结论：冻结 CheXbert 能提取简单阳性和部分常见否定，但在这组虚构语言
挑战中仍存在否定及不确定性错误；batch 大小和输入截断不能解释这些错误。
本轮是在检查报告标签提取器，不是在证明生成图像或报告的临床质量。
现有规则与 CheXbert 标签都不足以单独授权自动定位错误模态和返工。**

The 250 lightweight synthetic-fixture tests, source/artifact fingerprints,
protected project-group permissions, deterministic CPU audit replay and
complete post-hoc analysis replay passed. All predictions were committed before
reference-key analysis; the scorer did not read the key or receive expected
states. Existing primary score table, edge details and selected-triple hashes
remain unchanged. No checkpoint, guard, threshold or annotation was changed to
fit these results; no regeneration trigger was enabled. Training, downloads,
external API calls and patient-data access were not performed.

### Completed source-bound scope gate, official ConText and source comparison

The reusable interface now lives in
`src/tricompose_v12/report_assertions.py`, rather than hiding clinical state
decisions inside another model adapter. Each finding retains the original
proposal, gated state, source report SHA256, exact Unicode offsets/quote hash,
evidence ID, scope-check reasons, coverage, and a review action. **Scope verified
means only the limited literal rules covered the proposal; it is not independent
clinical verification or evidence that the report describes its CXR.**

`benchmarks/gate_report_assertion_predictions.py` applies this gate blindly to
the completed frozen CheXbert predictions before separate authored-key analysis.
The underlying pre-challenge scope guard is unchanged. A proposal can be
retained only when every matched literal scope is covered and agrees with its
existing state. A non-veto is no longer sufficient: uncovered vocabulary or
scope now abstains. The gate never changes positive to negative, promotes an
unknown proposal, creates a finding, or authorizes regeneration. Its outputs
are `scope_commit`, `abstain`, and `no_model_assertion`; the latter two are not
counted as successful decisions, and unknown is never treated as negative.

The completed protected runs are:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  report_scope_gate_12580901_001/
    evidence.json, decision_table.csv, manifest.json
  report_scope_gate_analysis_12580901_001/
    summary.json, details.json, manifest.json
```

The decision table has all **224 report/finding rows**, not just successful
rows. Risk/coverage analysis keeps the full **80 designated-check denominator**:

| Development diagnostic | Count |
|---|---:|
| Raw CheXbert exact authored-state matches | 48/80 |
| Raw non-unknown proposals | 66/80 |
| Scope commits | 30/80 (37.5% coverage) |
| Exact authored-state matches among commits | 30/30 |
| Abstained non-unknown proposals | 36/80 |
| No model assertion | 14/80 |
| Raw errors not committed | 32/32 |
| Raw matching states not committed | 18 |

Among all 224 rows there are 34 scope commits, 40 abstentions and 150 missing
model assertions. **This gate was designed after the authored challenge results
were observed.** Its 30/30 conditional match is a post-hoc development result,
not 100% overall accuracy, independent validation, clinical specificity, or a
false-repair estimate. It explicitly sacrifices coverage and also drops 18
matching states (14 unknowns and four non-unknown assertions). It cannot replace
the original score table or establish an improvement in generated triples.

**中文进展：报告事实接口已经输出完整的原状态、证据、规则覆盖和弃权表；
不是把分数强行提高。当前门控保留 30/80 项，但没有对其余 50 项作出明确
判断。下一步需要独立对照与未用于开发的临床标注，而非立即触发返工。**

No NegBio/ConText package was found in the inspected existing environments.
`benchmarks/score_report_assertions_context.py` and
`slurm/20_report_context_cpu.sbatch` implement an additional **official medspaCy
ConText** comparison, following its [official API](https://github.com/medspacy/medspacy/tree/1.3.1).
The job uses a new isolated environment, blank English tokenization, official
PyRuSH sentence rules and unmodified official English ConText rules; no learned
NLP model or CXR checkpoint is loaded. The existing limited four-finding literal
inventory is shared and disclosed, so this is a separate context algorithm,
**not independent clinical votes or a full clinical synonym extractor**.
ConText cue/scope spans and historical/hypothetical/family/negated/uncertain
flags retain source hashes and character offsets. An unmodified mention's
conventional positive baseline is explicitly marked unverified.

The script installs pinned medspaCy 1.3.1, spaCy 3.7.5, PyRuSH 1.0.12
and NumPy 1.26.4 into a new per-job environment. The official 244,637-byte
medspaCy source distribution is SHA256-pinned; installed transitive versions,
rule/code hashes and installation metadata are retained. No original model
environment is changed. The job compares ConText with the authored references
only after blind predictions are committed, then measures a second-parser
veto on the original CheXbert proposals. Parser disagreement can only reduce
coverage, not flip a state or promote an unsupported finding. There is no
fitted threshold, custom trigger, training, new router or original-score change.

The complete standalone script and request were shown: **main, two CPUs,
8 GiB host memory, 15-minute cap including first installation, no GPU**. Instead
of submitting another job, the exact script was executed with `bash` inside
the current session's existing CPU Slurm allocation **12576792**, on `b05-04`.
Its allocation and cgroup were verified before execution: four CPUs and
32 GiB allocated, worker threads limited to two and command timeout 15 minutes.
`#SBATCH` directives do not allocate resources when a script is run with `bash`.
**No new `sbatch`, GPU request or standalone benchmark job was submitted.**
The command completed with exit code zero; the private execution receipt is
`task_runtime/report_context_12576792/execution_receipt.json` below the protected
V1.2 root. The earlier separate-submission approval request is no longer needed
for this completed run; it must not cause a duplicate submission.

Official ConText processed the 56 invented reports twice in **2.363 seconds**
(parser-program elapsed time, excluding dependency installation); no replay
state/evidence changed. It loaded 102 unchanged official rules and no trained
NLP component. None of those default rules restrict `allowed_types` or
`excluded_types`, so the finding-label entity names were not excluded by such
restrictions. This does not establish that the default rule set covers all
clinical assertions or that another ConText setup would have the same result.

Completed official-parser runs below `artifacts/protected/tricompose_v1_2/`:

```text
verification_runs/
  authored_assertions_context_12576792/
  authored_assertions_context_12576792_analysis/
  authored_assertions_context_12576792_gate/
  authored_assertions_context_12576792_gate_analysis/
diagnostics/
  report_extraction_sources_12576792_001/
    comparison.json, source_table.csv, manifest.json
  report_extraction_sources_analysis_12576792_001/
    summary.json, details.json, manifest.json
```

`benchmarks/compare_report_assertion_sources.py` adds a **separate literal
readout**, using the same unchanged scope checker frozen before this challenge.
It proposes its own source state only when all literal scopes are covered;
otherwise it returns unknown. It does not change the CheXbert/ConText labels,
invent a new trigger, or use authored answers during extraction. The comparison
wrapper and this development analysis were added after observing the challenge,
so these are not held-out results even though the underlying rules are older.

| Source extraction diagnostic, 80 designated checks | Authored-state matches | Four-state macro F1 | Positive/negative flips | Determinate output on unknown/uncertain reference |
|---|---:|---:|---:|---:|
| Frozen CheXbert | 48/80 | 0.6009 | 7 | 13 |
| Official default medspaCy ConText | 42/80 | 0.5224 | 14 | 24 |
| Frozen limited literal readout, separate proposal | 69/80 | 0.8680 | 0 | 0 |

The literal readout covers only **46/80** checks. Its remaining 34 outputs are
unknown: 23 match authored unknown states, and 11 miss non-unknown states.
Thus 69/80 is not broad clinical coverage or independent clinical accuracy.
Among the 46 covered checks, 16 literal proposals differ from CheXbert. There
are 25/80 CheXbert/ConText disagreements; ConText corrects three CheXbert
authored-state mismatches but disagrees with nine correct CheXbert states.
These are **extraction disagreements, not established report or CXR errors**.
Every comparison row explicitly keeps `report_content_error_established`,
`image_error_established` and `regeneration_authorized` false.

As a second-parser veto, ConText reduces the CheXbert scope gate from
**30/80 retained (37.5%)** to **21/80 (26.25%)**. Both have zero incorrect
retained states on this authored development set; the additional veto drops
nine more matching states and shows no added benefit here. It is not promoted
into primary selection. Agreement over the same text remains correlated
evidence, not a majority of independent clinical observers.

**中文结论：这一轮已实际跑完，不是在等待新作业。官方 ConText 在当前
语言挑战上没有改善结果；旧字面规则能解释部分提取分歧，但覆盖有限。
现在有完整证据表与弃权接口，下一步应做独立标注验证，不能把解析器
分歧直接变成生成报告／胸片返工。**

The reusable interface, official serializer contracts, absence/uncertainty
handling, source binding, non-overwrite, blind-key ordering, separate literal
readout and abstention denominators are covered by **298 passing lightweight
tests**. All six new runs' 11 artifacts, producer/source hashes, official parser
replay, analyses and exact CSV bytes were rechecked. Protected modes/groups
pass; original primary selection hashes remain unchanged. No real patient
inputs or generated-cohort reports were opened for this diagnostic.

## Metadata-only blinded review of generated reports

The report assertion validation follow-up is implemented in
`src/tricompose_v12/report_review.py` and
`benchmarks/prepare_blinded_report_review.py`, with its return/evidence audit in
`benchmarks/audit_blinded_report_review.py`. It does not use another model to
invent independent human gold. The unchanged two-case pilot's **48 reports
(12 images × four experts)** are assigned opaque review IDs, with no winner,
score, model, case ID or source path in the reviewer files. All four current
scope findings remain represented: **192 annotations per human reader**.
Identical report texts would share one review item without dropping candidate
lineages; this pilot has 48 distinct text hashes.

Protected output:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_20261002_001/
    reviewer/       # blind items, two blank templates and instructions
    investigator/   # source resolver, frozen caches and method/code hashes
    summary.json, manifest.json
  report_blind48_readiness_20261002_001/
    summary.json, manifest.json
```

The metadata preparation and readiness audit completed with **zero model
calls, zero opened report texts/pixels/EHR records and zero gold annotations**.
Actual human annotation progress is 0/192 jointly reviewed rows. Pending is
null, not a reviewed unknown or negative. Extractor comparison and reader
kappa therefore stay unavailable. **328 lightweight tests** pass; the two runs'
nine artifacts and exact metadata replay were checked with a guard that
refuses report-text/pixel opens. Source and original winner hashes are unchanged.

Once qualified separate humans return annotations, evidence checking requires
verified source text in Slurm. Reviewed assertions need exact original Unicode
quotes/offsets/hashes. Unknown, unassessable, incomplete and conflicting reviews
are distinct; their coverage denominators are preserved. The audit can compare
unchanged CheXbert/Qwen and the frozen limited literal readout against matching
reader states and report the separate gate's risk/coverage. **Those matching
states are provisional, not automatically adjudicated clinical gold.** No
parser resolves reader conflicts or authorizes a regeneration action.

This is a previously inspected two-EHR **development review**, not independent
final-test data or 48 independent patients. Report-only annotation can validate
what the text asserts; it cannot establish whether a CXR is correct. Authorized
members retain access to investigator files, and report style may expose model
identity, so model blinding is procedural, not an access-control guarantee.

The prepared `slurm/21_blinded_report_copies_existing_cpu.sh` was shown in full.
After explicit user approval with `go`, the unchanged script **completed with
exit zero** in the existing CPU allocation `12576792` on `b05-04`. The allocation
has four CPUs and 32 GiB; copying was limited to one worker thread and a
120-second timeout. Command wall time was approximately **0.53 seconds**.
No new `sbatch`, GPU request, inference or generation was performed.

The guarded `benchmarks/materialize_blinded_report_review.py` wrote a separate
immutable reviewer-readable run:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_text_12576792/
    reports/report_0000.txt ... reports/report_0047.txt
    items.json, reviewer_a_template.json, reviewer_b_template.json
    INSTRUCTIONS.md, summary.json, manifest.json
  report_blind48_text_readiness_12576792_001/
    summary.json, manifest.json
```

All **48/48** synthetic report copies match their original hashes, and all
53 manifest-listed artifacts passed checks. Protected modes/groups are correct.
Source EHR/images/real reports, raw patient inputs, external APIs, new checkpoints
and training remain out of scope. No synthetic text was placed in this README,
public logs or Git. The original metadata packet is unchanged, including its
historical unmaterialized status. The copied-template audit retains 192 pending
rows per reader, zero completed human labels and unavailable human-reference
metrics. Original score and winner hashes remain unchanged.
The flag `--allow-synthetic-report-copy` remains only a program guard; user
approval, not the flag or a Slurm job ID alone, authorized this completed copy.

See [report-fact interface handoff](../docs/report_assertion_status_20261002.md)
for the reviewed methods, annotation boundary and completed/pending status.

### Easier human review: protected reading book and CSV returns

The approved synthetic copies now have a deterministic reading book and two
blank CSV forms, under:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_sheets_20261002_001/
    reports_for_review.md
    reader_a.csv, reader_b.csv
    reader_a_identity.json, reader_b_identity.json
    HOW_TO_REVIEW.md, summary.json, manifest.json
  report_blind48_sheet_roundtrip_a_20261002_001/
  report_blind48_sheet_roundtrip_b_20261002_001/
  report_blind48_sheet_readiness_20261002_001/
  report_blind48_sheets_verification_20261002_001/
```

`benchmarks/report_review_sheets.py --mode export|import` shares the unchanged
annotation contract. The importer preserves human status/state/reason and
only computes exact Unicode evidence offsets/hashes. Ambiguous repeated
quotes require a human-selected zero-based occurrence; paraphrases and changed
source hashes are rejected. CSV evidence retains original CRLF characters.
Missing review is not reviewed unknown, and neither means negative.

The book contains **48 reports from two EHR cases**, not 48 patients. Two blank
192-row imports and the readiness audit completed in the existing CPU Slurm
allocation, with **0 completed human labels**. The book replay, source bindings,
12 manifest-listed artifacts, protected permissions and old winner hashes
passed checks; overwriting the existing export was refused. **356 tests pass**,
using invented fixtures for the human-filled cases. A final CRLF-only importer
fix followed the blank imports; exact current-core replay matches those pending
artifacts, and the current CLI newline behavior is tested on invented text.

No new Slurm submission, model/API call, independent clinical gold, selection
update or regeneration occurred. Human-reference accuracy/kappa remain
unavailable. Separate humans must copy the forms to new protected return
directories before filling them independently; do not edit immutable runs.
The [handoff](../docs/report_assertion_status_20261002.md#csv-review-workflow--csv-审核流程)
contains copy/import/audit commands and the scope limits.

## Candidate-level report-scope availability (2026-10-02)

`benchmarks/build_report_scope_table.py` now propagates the **unchanged frozen
four-finding report syntax gate** into a separate diagnostic table for the
existing 48 candidates. It reads only approved copied synthetic reports and
hash-bound cached metadata. No source EHR, image pixels, real targets, human
annotation returns, model/API, new checkpoint, primary score or selector is
used. Raw CheXbert states and all eight cached findings remain intact.

The immutable output is:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  report_candidate_scope_12576792_001/
    fact_scope_table.jsonl
    candidate_scope_table.csv
    candidate_scope_table.json
    report_scope_assertions.json
    cross_path_groups.json
    summary.json, summary.md, manifest.json
  report_candidate_scope_verified_12576792_001/
    summary.json, manifest.json
```

| Edge | Inventory denominator | Raw comparable facts | Scope-usable comparable facts |
|---|---:|---:|---:|
| EHR–CXR | 96 unique image/finding pairs | 12 | 12 (report gate does not affect this edge) |
| EHR–Report | 384 candidate/finding pairs | 11 | 0 |
| CXR–Report | 384 candidate/finding pairs | 146 | 34 |

The 384 rows mean **48 candidates × eight findings, from two fixed EHRs**.
Image-side counts deduplicate by CXR byte hash, not report count or candidate
alias. Four scope-supported findings give 192 report assertions: 34 syntax
commits, 68 abstentions and 90 absent model assertions. The other 192 rows
explicitly remain `outside_scope_inventory`; they are not negative findings.

Both fixed EHRs' comparable direct finding is pneumonia, which the frozen
four-finding syntax gate does not cover. Therefore EHR–Report has **no
scope-usable comparison**, not a score of zero or a proven generation failure.
Unsupported findings must not be silently dropped from the denominator or
assigned an invented checking head. Raw EHR–CXR signals remain unvalidated
clinical evidence despite their unchanged availability.

CXR–Report raw/scoped support signals are 93/21 and opposition signals 53/13.
Withdrawing 112 comparable facts is **lost evidence coverage, not better
clinical consistency or repaired reports**. Conditional support actually moves
from 93/146 to 21/34; those fractions are diagnostic arithmetic, not accuracy.
The 12 raw positive/negative report-disagreement groups become zero under the
limited scope mask, which likewise does not mean cross-path errors disappeared.
Reports share an image, so no independent votes or fault attribution result.

The gate never flips a polarity, fills an unknown, changes the EHR or selects a
new winner. Candidate-level clinical selection scores stay null and automatic
repair eligibility stays false. The run and seven artifacts replay exactly;
source/code hashes, protected permissions and original winner hashes passed.
**379 lightweight tests pass.** Execution used the existing CPU Slurm allocation
`12576792`; no new submission or GPU inference occurred. An existing-run
overwrite was refused.

The next scientific requirement is still independent human evidence and
proper coverage of the intended finding inventory before clinical selection or
targeted regeneration. The existing four-finding human-review packet remains
unchanged and has zero returned labels; it cannot by itself validate the
uncovered pneumonia EHR–Report edge. Do not tune the frozen guard to make this
development table look better.

## Real-data scorer validation pilots

`real_validation/biovil_matched_pairs.py` and
`slurm/01_real_biovil_valtest128_p100.sbatch` prepare the first real-pair
scorer check. The Slurm-only program reads the existing read-only
`three_modalities/v2_labs_vitals` linkage manifest internally, verifies
patient-disjoint splits, chooses one frontal study for each of 128 validation
and 128 test patients, and compares frozen BioViL-T cosine for the matched
report versus a different patient's report. No raw image, report, subject ID,
study ID, source path, or report excerpt enters stdout or Git. Individual
scores, opaque row indices and hashes are written only below
`artifacts/protected/tricompose_v1_2/real_validation/`.

The pilot reports AUROC, average precision and within-case matched-score win
rate separately for validation and untouched test. Cross-patient reports are
*mechanical* mismatches and may remain clinically compatible; this tests basic
pair retrieval, not fine-grained clinical correctness or a selection threshold.
BioViL-T may share MIMIC-CXR training data and is not an independent final
evaluator. Hard negatives with clinical adjudication, XRV/CheXbert calibration,
and image/report review remain required. EHR validation is deferred until the
dataset's admission-level diagnosis/procedure features are proven to predate
the image; current source documentation explicitly warns of temporal leakage.

The first pilot completed as Slurm job `12492909` in 68 seconds. Its protected
aggregate results were val/test AUROC 0.934/0.940 for matched reports versus
random cross-patient reports. These are retrieval results, not clinical
contradiction validation. The complete script and resource request for **each
new** job must be shown to the user for explicit approval before `sbatch`.

`real_validation/hard_negative_biovil.py` reuses the fixed 128+128 real cohort and
previous scores. Within each split, it uses official report-derived CheXpert
labels to choose a different-patient, same-view report that shares at least
one positive finding yet has an explicit opposite finding. It scores those
reports with the same frozen BioViL-T checkpoint, then compares matched-vs-
hard and matched-vs-random on the **same available subset**. Unknown and
uncertain labels are not negatives. Cases without a qualifying donor remain
unavailable and are counted. The source labels are not independent image truth
and the challenge remains unadjudicated. The approved batch script
`slurm/02_real_biovil_hardneg_p100.sbatch` completed as job `12508001` in
60 seconds. The protected result is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_hardneg_valtest128_20260930_001/`.

| Split | Qualifying pairs | Unavailable | Matched vs hard AUROC | Matched vs random AUROC on same subset | Matched > hard |
|---|---:|---:|---:|---:|---:|
| Validation | 78 | 50 | 0.931 | 0.920 | 96.2% |
| Test | 74 | 54 | 0.886 | 0.926 | 90.5% |

The lower test AUROC against hard negatives than random negatives demonstrates
a harder retrieval challenge, but **not** independently verified clinical
contradiction detection. Donors were preselected by labels extracted from
their own reports; matched pairs can contain reporting errors, and the actual
CXR finding has not been independently adjudicated. No threshold or
probability calibration was performed. The next gate is independent image-side
evidence and case-level adjudication before using BioViL-T as a clinical
selection score.

The next check is
`real_validation/xrv_real_findings.py` with
`slurm/03_real_xrv_weak_findings_p100.sbatch`. It applies the frozen XRV
image classifier to the same fixed 128+128 real images and reports per-finding
AUROC only where both report-derived classes have at least ten examples.
Unknown/uncertain reference labels are excluded, and classes with insufficient
support remain unavailable. Its reference is still report-derived, not an
independent radiologist image label; XRV may also have trained on MIMIC-CXR.
This check diagnoses classifier coverage and failure modes but cannot alone
clear the clinical-adjudication gate or justify a selection threshold.

The approved script completed as job `12529380` in 1 minute 48 seconds. Its
protected result is
`artifacts/protected/tricompose_v1_2/real_validation/real_xrv_weak_valtest128_20261001_001/`.
The table shows AUROC only where each class had at least ten explicit labels;
`NA` means insufficient support, not a score of zero.

| Finding | Val AUROC | Test AUROC | Test positive / negative |
|---|---:|---:|---:|
| Cardiomegaly | 0.935 | NA | 31 / 6 |
| Edema | 0.828 | 0.904 | 18 / 18 |
| Pleural effusion | 0.864 | 0.809 | 47 / 19 |
| Pneumonia | 0.576 | 0.790 | 10 / 10 |
| Pneumothorax | 0.684 | 0.690 | 16 / 25 |
| Atelectasis, consolidation, lung opacity | NA | NA | insufficient explicit negatives |

Only five findings on validation and four on test met this minimum support.
The variation for pneumonia and consistently weak pneumothorax separation
argue against treating raw XRV values as authoritative image truth. These are
operating-point-normalized scores, not calibrated probabilities. No threshold
was fitted. The run's recorded `peak_vram_gib` excludes model loading because
CUDA peak statistics were reset after initialization; do not use that field as
total GPU memory cost.

The approved official human-report-label **metadata audit** implemented in
`real_validation/audit_official_report_gold.py` completed through
`slurm/25_official_report_gold_source_schema_existing_cpu.sh`: existing CPU
allocation `12576792`, four CPUs / 32 GiB allocated, one worker, 120-second
timeout, no new `sbatch` or GPU. It ran in 0.745 seconds and replayed exactly.
Output: `artifacts/protected/tricompose_v1_2/real_validation/official_report_gold_coverage_12576792_source1/`.
Patient-disjoint linkage, four-state denominators, deduplication, artifact/source
hashes and protected permissions passed; original score/winner hashes stayed
unchanged. Status: `completed_verified_metadata_only_not_model_validation`.
Test report-path **metadata** is available; existence/content remains unchecked.

The initial exact-column audit rejected **Airspace Opacity** versus the expected
**Lung Opacity**. The final audit preserves the original source label name and
leaves the common Lung Opacity mapping null, without altering label values or
frozen report rules. Historical wrappers `22`–`24` preserve old pins, not
current rerun commands. See [the handoff](../docs/report_assertion_status_20261002.md)
for failure provenance and the protected execution/verification receipt.
This audit consumed only approved label/linkage metadata, not report files,
separate EHR files or image pixels. It produced no model accuracy, fitted
threshold, fault attribution, selection update or regeneration. Official report
labels are not image truth; CheXbert training overlap/annotation alignment
remain unverified. **398 invented-fixture tests pass**, including 19 audit
tests. A real-report/model benchmark needs separate execution approval.

The real-report label diagnostic is implemented in
`real_validation/official_report_benchmark.py` and
`real_validation/run_official_report_benchmark.py`, with
`slurm/26_official_report_chexbert_debug_p100.sbatch`: debug, one P100, two CPUs,
16 GiB RAM, ten minutes maximum. **Approved and completed as job 12594397**,
exit `0:0` on `e23-02`, allocation elapsed 15 seconds. Program runtime was
12.025 seconds and peak allocated VRAM was 1.227 GiB. It uses only a
declared conservative Impression section, retains missingness, compares frozen
categorical states to the original human labels, excludes unaligned opacity/
binary heads from four-state metrics, and diagnoses the unchanged four-finding
scope guard without changing selection. Source text/keys/paths/images/reference
rows are not exported. [Protocol and limitations](../docs/report_assertion_status_20261002.md)
are frozen before real results. **421 invented-fixture tests pass**, including
23 benchmark tests. Separate approval of the complete batch script preceded
real-report/model consumption; prior metadata approval was not a submission approval.
Nonempty predictions, artifact hashes, inventory denominators and protected
permissions passed post-run checks. The output is
`artifacts/protected/tricompose_v1_2/real_validation/official_chexbert_reports_12594397/`;
the manifest SHA256 is
`aeea0df50e73c4b57af35d3d686ee5d3677f1663174cf28a570205f5248deec6`.
Per-finding results remain protected. This does not clear an independent
clinical/image/EHR validation gate, tune any rule, change a score/winner or
authorize targeted regeneration.
The readable bilingual diagnostic is
`artifacts/protected/tricompose_v1_2/real_validation/official_chexbert_review_12594397_001/RESULTS_CN_EN.md`,
with a hash-bound receipt and manifest. It separates full inventory coverage,
four-state F1, non-unknown matches, omissions, explicit polarity flips and
guard commit coverage. Statistics were recomputed from derived confusion
matrices, not by reopening source reports or rerunning a model.
The existing synthetic pool also has a derived diagnostic overlay at
`artifacts/protected/tricompose_v1_2/diagnostics/candidate_report_diagnostic_12594397_001/`:
48 candidate triples, two EHR cases, 384 finding rows. Its CSV connects cached
report states and scope availability to per-head real-report diagnostics,
without new scores, ranks, model calls or raw-text/image reads. This does not
establish checkpoint/section equivalence, distribution transfer or clinical
truth, and must not be used to fit weights on the official test source.

## Next fact-conflict diagnostic: BioViL-T polarity / 单事实正负极性

`real_validation/biovil_fact_polarity.py` and
`slurm/27_real_biovil_fact_polarity_debug_p100.sbatch` were separately approved
and completed as **job 12597637** on `e23-02`, exit `0:0` in 49 seconds.
They reuse the existing fixed 128+128 real CXR cohort, test eight findings with
three authored presence/absence template pairs, and keep images fixed. All
prompts are scored label-blind; report-derived cached references enter only
the summary, with unknown/uncertain denominators retained. Raw source report
text and EHR fields are not read, and reference rows are not exported.

The diagnostic separates positive/negative polarity wins, balanced wins,
margin AUROC where support permits, fixed template-family results and the
predeclared mean. A scorer can have strong retrieval or margin AUROC while
always preferring positive text; balanced polarity wins expose that failure.
No threshold/template/rule fitting, changed winner or repair claim is enabled.
References remain weak report labels, not independently adjudicated image truth;
the reused splits and possible MIMIC training overlap preclude an independent
final-test claim. See [the frozen protocol](../docs/biovil_fact_polarity_protocol.md).

Request: debug P100 × 1, CPU × 2, host RAM 16 GiB, ten-minute resource cap.
The complete script received separate explicit approval before submission.
**440 fixture tests pass**, including 19 new polarity tests; no real image/model
was loaded during preparation.
The submitted output is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_fact_polarity_12597637/`.
Program runtime was 45.827 seconds and allocated peak VRAM including model
loading 0.576 GiB. Derived metrics recomputed exactly; fixed source/reference
bindings, all templates, protected permissions and old winner/table hashes
passed verification. The readable result is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_polarity_review_12597637_001/RESULTS_CN_EN.md`.
Polarity performance depends on finding and authored sentence family; retain
BioViL-T as secondary evidence, not a sole clinical repair judge. No independent
image adjudication, calibrated threshold, score/winner update or regeneration
authorization is claimed by this weak-reference result.

Unit tests use only invented synthetic metadata:

```bash
PYTHONPATH="src:TriCompose-v1.0/src:TriCompose-v1.1/src:experiments/synehrgy_v2/src:TriCompose-v1.2/real_validation:TriCompose-v1.0/eval/report_v1_1" \
  PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s TriCompose-v1.2/tests -p 'test_*.py' -v
```

## Decision-interface preview / 决策接口预览 (2026-10-02)

The pure `src/tricompose_v12/decision_preview.py` connects existing cached
finding states to a deterministic **verification request**, not an executable
clinical repair policy. See [the interface protocol](../docs/decision_preview_protocol.md).
It retains all eight findings, fixed EHR hashes and source evidence IDs; direct
EHR coverage gaps and correlated report/scorer disagreement remain explicit.
Raw/scoped edge arithmetic is independently checked before any decision hint.
Agreement cannot grant clinical acceptance. A provisional report/CXR verification
target is not a confirmed faulty modality, and every model-execution flag is false.

`benchmarks/preview_candidate_actions.py` completed a derived-only run inside the
existing CPU allocation `12576792`; no new `sbatch`, model, GPU, raw source
patient input, synthetic report body or image pixels were accessed. It consumed
the full prior scope inventory: **48 candidate triples, two fixed EHRs, 384 facts**.
Caller-specified caps of four additional model calls and 120 additional GPU
seconds are engineering limits only; no verification-cost estimate was invented.
Missing runtime/estimates do not become zero, and failed/retried calls count in
the tested ledger. Old bank generation cost is separate from any future calls.

Protected output:
`artifacts/protected/tricompose_v1_2/decision_previews/decision_preview_12576792_001/`
contains `decisions.jsonl`, `action_preview.csv`, `summary.json`,
`README_CN_EN.md`, and a hash-bound `manifest.json`. The original source and
winner/table hashes, output inventories and project-group modes were verified.
All proposed next actions request more verification; none executes a repair or
changes acceptance/rejection of any existing case. This demonstrates the guarded
interface, **not** clinical localization accuracy or improved generation.

**473 V1.2 invented-fixture tests pass**, including 33 new tests covering
determinism, unknown/uncertain semantics, unsupported pneumonia scope, unchanged
EHR, scope withdrawal, forged eligibility, correlated reports, budgets and
refusal-before-protected-read outside Slurm. Independent adjudication and the
clinical/calibration gate remain pending; a successful cache preview cannot
enable targeted regeneration or substitute for that evidence.

## No-human automatic policy track / 无人工自动策略路线 (2026-10-02)

The user cannot provide human checks or feedback. That no longer blocks
**exploratory automatic policy experiments**. It does not create independent
clinical truth or authorize radiologist-validated claims. Historical pending
review packets and decision previews are unchanged. The new explicit track is
defined in [the no-human protocol](../docs/automatic_no_human_protocol.md).

Implemented:

- `configs/automatic_replay_v1.json`: frozen model order, six call budgets,
  five random seeds, existing V1.1 candidate objective and invocation accounting.
- `src/tricompose_v12/automatic_replay.py`: fixed, random, prefix static rerank,
  and targeted heuristic, observing only requested candidate slots.
- `benchmarks/run_automatic_proxy_replay.py`: hash-checked protected cache
  comparison, with alternate automatic readouts and per-step action/cost traces.
- `tests/test_automatic_replay.py`: 26 invented-grid tests, including no-peeking,
  fixed EHR, image reuse, budget bounds, no secondary-score routing, missingness,
  raw-versus-guarded scope, legacy normal-statement metrics and source integrity.

Completed protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_replay_12576792_001/`.
The run uses all 48 candidates from two fixed EHRs, not 48 independent patients.
It has 96 replay trials: two EHRs × six budgets × (three deterministic methods
+ five random replicates). The table is `method_comparison.csv`; per-step
records are `replay_outcomes.jsonl`; readable interpretation is `RESULTS_CN_EN.md`.
The program took 0.030806 seconds within the existing CPU allocation, with zero
new model calls, GPU execution, patient source inputs, text/image opens or Slurm
submissions. Exact replay, source/result hashes, protected permissions and old
winner/table hashes passed. The full local V1.2 suite has **499 passing tests**.

This is no longer an all-`verify_more` interface: heuristic actions explore
existing report/model/seed alternatives and stop at a proxy constraint or call
budget. It still does **not** physically regenerate an image/report. New
prospective execution requires its own reviewed Slurm script and approval.
The private table deliberately separates optimization proxies from BioViL-T
and guarded readouts; raw label gains are not evidence of clinical correctness.
No calibration, mask, stopping rule or model priority was retuned on the replay
outcomes. The next no-human step is transfer to a separately frozen larger
cohort/alternative automatic evaluator, followed by bounded prospective calls
with full actual generation/verification/failure cost accounting.

## Fixed 80-EHR automatic replay / 固定 80 例自动对比 (2026-10-02)

Implemented `src/tricompose_v12/legacy_replay_adapter.py`,
`configs/automatic_replay_pool80_v1.json`,
`benchmarks/run_legacy_automatic_replay.py` and a private verification reader.
They reuse cached synthetic states/lineage only, not image pixels/report bodies.
All 80 fixed EHRs and 960 source triples were retained; 3,200 budget/seed trials
completed inside the existing CPU allocation with no new models or submissions.
The source bank is the explicitly named historical uncalibrated fourteen-head
profile, not the new calibrated eight-head profile. Do not pool their scores.

Protected outputs below
`artifacts/protected/tricompose_v1_2/automatic_replays/`:

- `automatic_replay_pool80_12576792_001/method_comparison.csv`: full budget curves.
- `automatic_replay_pool80_12576792_001/subgroup_comparison.csv`: direct-EHR
  and no-direct-EHR groups, with missing edge rates preserved as NA.
- `automatic_replay_pool80_review_12576792_001/RESULTS_CN_EN.md`: readable
  result directions, coverage-definition distinction and verification status.

Exact replay, budget/summary arithmetic, artifact/source/winner hashes and
private permissions passed. Historical prompt-conditioning tiers (15/65)
must not be confused with the directly comparable cached EHR labels (8/72).
No facts were inserted and no cases were discarded. The maximum-budget
outcome remains a proxy cost/quality tradeoff, not independently validated
clinical repair or an established advantage over exhaustive static selection.

The distinct BioViL-T endpoint request is frozen in
`automatic_biovil_request_12576792_001/`. All metadata lineage/preflight checks
passed without opening report/image bodies. `benchmarks/score_automatic_replay_biovil.py`
and `slurm/28_automatic_pool80_biovil_debug_p100.sbatch` use one
P100/two CPUs/8 GiB/ten minutes. The complete script was displayed and approved;
**GPU scoring completed as job 12607645**, exit `0:0`, on `e23-02` in 36 seconds
(program 33.322983 seconds, peak GPU allocation 0.61 GiB). BioViL is excluded
from policy selection; full-report context failures stay NA, not silently
truncated scores. No training, downloads, original-winner changes or GitHub
actions were performed. **537 invented-fixture tests pass**.

## BioViL endpoint comparison / 补充评分对比 (2026-10-02)

Implemented `src/tricompose_v12/automatic_secondary.py`,
`benchmarks/merge_automatic_secondary.py` and thirteen new invented-cache tests.
The immutable secondary scores are overlaid in a NEW protected run without
changing any selected candidate, action trace, original score or fixed EHR.
All five random replicates are averaged within a case; paired comparisons
require both endpoints and retain missing-case counts, rather than treating
repeated seeds as patients or silently filtering incomplete replicates.

Read the current completed comparison under
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_secondary_pool80_12607645_001/`:

- `RESULTS_CN_EN.md`: readable method/budget table and interpretation.
- `method_comparison.csv`: proxy metrics, simulated calls and BioViL availability.
- `subgroup_comparison.csv`: direct-EHR versus no-direct-EHR readouts.
- `paired_case_comparison.csv`: paired BioViL differences and full-cohort call
  differences for every budget against fixed/random/static baselines.

The maximum-budget BioViL paired mean is lower for targeted search than all
three baseline method IDs. Here `random` means random-order acquisition followed
by the same score-based reranking, NOT a random final triple. At the complete-bank
budget it selects exactly the static winner in all five seeds for all 80 cases;
random/static are therefore not two independent endpoint confirmations.
Thus current proxy improvements do NOT demonstrate a general
consistency/repair advantage. Preserve the negative result; do not change the
cohort, weights, priorities or existing winners to improve this endpoint.
BioViL remains an alternate uncalibrated readout, not independent clinical
truth or confirmed fault localization. Exact arithmetic, source/result hashes,
unchanged actions/winners/EHRs and protected permissions passed verification.
No generation/training/repair/API/GitHub action was executed. Actual endpoint
GPU scoring cost is reported separately from simulated policy invocations.

## Frozen discrepancy diagnostic / 冻结分歧诊断 (2026-10-02)

Implemented `src/tricompose_v12/automatic_discrepancy.py`,
`benchmarks/diagnose_automatic_discrepancy.py` and fifteen invented-cache tests.
The diagnostic reuses cached synthetic states, hashes, choices and endpoint
scores only: no report bodies, image pixels, raw patient inputs or models.
It decomposes explicit positive/negative support, retains unknown/uncertain
and missingness, and distinguishes legacy source adjustments from raw states.
Fixed EHRs, source counters, model order, action traces and winners are unchanged.

Completed inside the existing CPU allocation `12605930` in 0.607697 seconds,
with zero new model calls, GPU work or Slurm submissions. Protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/automatic_discrepancy_pool80_12605930_001/`.

- `RESULTS_CN_EN.md`: readable decomposition and interpretation.
- `case_contrasts.jsonl` / `action_slices.csv`: 800 contrasts, grouped by budget,
  EHR evidence coverage and hash-based image/report changes.
- `same_image_report_pairs.jsonl` / `same_image_controls.csv`: 532 observed
  report-pair controls on 210 case/image groups; within-case means preserve
  correlation and the selected-union sampling limitation.
- `model_selection_frequencies.csv`: selection frequencies, not model accuracy.
- `summary.json` / `manifest.json`: counters, limitations and hash provenance.

At maximum budget versus fixed, 30 cases keep the same artifacts, 12 change
only the report, and 38 change both image and report. The report-only group has
a positive mean BioViL difference; the overall decline is concentrated in the
joint-change group. Support gains include both polarities, but the negative
increment is larger. Normal agreement is legitimate: do not add diseases or
discard healthy/underconditioned EHRs to alter this diagnostic. Joint changes
also change the classifier reference, so an improved proxy does not identify
which modality was repaired. These are descriptive associations, not causal
clinical localization or a significance result.

Common-image comparisons also reveal semantically different reports tied on
the earlier coarse ranking criteria and decided by runtime. This motivates a
separately preregistered fixed-image/cost-tie control, not retrospective policy
tuning or a claim that faster models are worse. Existing controlled fault
diagnostics are not relabeled as a newly completed repair benchmark.
Exact regeneration, all source/result hashes, old artifacts and protected modes
passed checks; the full V1.2 suite now has **552 passing invented-fixture tests**.
Next controls and evidence boundaries are recorded in the no-human protocol.

## Fixed-image report control / 固定图像报告选择对照 (2026-10-02)

Implemented `src/tricompose_v12/fixed_image_control.py`,
`benchmarks/run_fixed_image_control.py`, a separate frozen control configuration,
and nineteen invented-cache tests. The original Sana seed 0 image is fixed for
every EHR, without consulting report/endpoint scores. Static report-only search
visits the four registered reports; targeted report-only search retains the
old report-switch/stop rules and explicitly blocks any image-change request.
Source ranking, fixed EHRs, unknown/uncertain semantics and old outcomes remain
unchanged. BioViL scores are attached only AFTER choices; absent scores stay NA,
never a reason to substitute a previously scored candidate.

The new control is exploratory on the already inspected development cohort,
not retrospectively preregistered independent validation. See
[the frozen control protocol](../docs/fixed_image_control_protocol.md).
The completed protected run is
`artifacts/protected/tricompose_v1_2/automatic_replays/fixed_image_control_pool80_12605930_001/`.
It contains `RESULTS_CN_EN.md`, `method_comparison.csv`,
`subgroup_comparison.csv`, `paired_case_comparison.csv`, `control_outcomes.jsonl`,
`case_contrasts.jsonl`, `terminal_reasons.csv`, `frozen_control.json`, and hashes.

All 80 cases were retained, with 800 new report-only replay trials and 1,200
reused deterministic baseline trials. The program took 0.909379 seconds inside
the existing CPU allocation, with zero new model calls/submissions or text/image
opens. All control winners already had cached endpoint scores; zero pairs need
GPU scoring. Exact regeneration, source/result hashes, old selection artifacts
and protected permissions passed; **571 invented-fixture tests pass**.

At maximum cap, report-only selection has a higher mean BioViL readout than the
fixed/joint methods, but versus fixed only seven cases improve, nine decline
and 64 remain identical. A positive mean is not uniform improvement or clinical
accuracy. Report-only static and targeted select exactly the same final outputs
for all 80 cases: current dynamic rules have no additional endpoint-quality
gain there. Targeted saves only one blocked case's remaining report calls; 79
cases exhaust all four reports, none achieves the proxy-stop condition. Do not
describe that exhaustion as successful repair.

Cost is simulated, not measured regeneration GPU savings: one image plus four
reports costs ten generator/scorer invocations, while the joint inventory can
cost thirty. Equal caps do not equate search spaces/expenditure. This supports
separating report selection from image-switch verification in the next control,
not claiming that this old proxy already provides a working repair algorithm.

## Ranking and invariant-switch controls / 排序与固定证据对照 (2026-10-02)

Implemented `src/tricompose_v12/ranking_switch_controls.py`,
`benchmarks/run_ranking_switch_controls.py`, a separate frozen configuration and
twenty invented-history tests. See the
[control protocol](../docs/ranking_switch_control_protocol.md).
The runtime ablation reselects ONLY from each trial's original observed slots;
all earlier quality-key components, action traces, costs and stop reasons stay
fixed. Original proxy-stop choices are retained. It is final-ranking ablation,
not a new routed execution or removal of cost accounting. BioViL/availability
cannot affect a choice. Unscored choices remain NA with a pending hash inventory.

The invariant diagnostic holds explicit cached EHR states/hashes fixed, separates
image/report agreement within versus outside those constraints, and preserves
unknown/uncertain and no-direct-EHR cases. It never promotes weak medication/lab
context or diagnoses into newly asserted image truth. Agreement counts remain
unverified automatic proxies, not an error-localization/repair gate.

Protected run:
`artifacts/protected/tricompose_v1_2/automatic_replays/ranking_switch_controls_pool80_12605930_001/`.
It contains `RESULTS_CN_EN.md`, `ranking_comparison.csv`,
`invariant_switch_summary.csv`, protected per-case histories/contrasts,
`pending_endpoint_pairs.json`, frozen configuration and source/result hashes.
All 80 EHRs, 2,000 ranking trials and 800 invariant contrasts completed in
1.787128 seconds inside the existing CPU allocation, with zero new models,
GPU work, text/image opens or submissions. Exact replay, original histories,
all source/result hashes, old artifacts and private modes passed.
**591 invented-fixture tests pass**.

At maximum cap, removing runtime changes one report-only winner and ten joint
winners (nine change images); the mean BioViL differences are negative. This
does NOT show that runtime always improves clinical quality, but the simple
remove-runtime hypothesis is unsupported here. All maximum-cap endpoint pairs
are available. One new pair outside that scored union remains NA at a lower cap;
its pending metadata is not GPU-job authorization.

Of the original targeted policy's 38 image changes, 33 have no comparable
explicit cached EHR finding and five do. In those five, image support for the
fixed EHR increases in one case, while all-three supported-fact count is unchanged.
Most extra image/report agreement occurs outside explicit EHR constraints.
This is a lack of verification evidence, NOT proof those images are clinically
wrong or that those EHRs have no useful context. Normal agreement remains valid.
Future image-switch verification must distinguish invariant evidence from a
new classifier reference; do not fit a revised policy on this cohort and call
it independent evaluation. Existing winners/rules remain unchanged.

## Invariant verification hook / 固定 EHR 验证接口 (2026-10-02)

Implemented `src/tricompose_v12/invariant_verification.py`,
`benchmarks/bind_invariant_verification.py`, a frozen interface configuration
and twenty-five invented-cache tests. See
[the interface contract](../docs/invariant_verification_interface.md).

The public Python hooks are `anchor_from_cached_candidate`, `verify_candidate`
and `verify_transition`. An immutable EHR anchor binds states, cached source
categories and EHR/facts hashes; its evidence IDs never depend on the selected
image/report. Candidate receipts bind artifact/evidence lineage and all three
raw edge readouts with separate positive/negative support, opposition and
missingness. Transitions track specific fixed-EHR support gained/lost and
opposition added/removed, not a score-derived repair-success flag.

Integration currently binds these hooks to original **cached generation
histories**, not to a newly executed GPU pipeline. It does not edit the V1.0/
V1.1 generators or the existing scoped eight-head decision preview. This
version accepts validated completed legacy triples; partial image-only
verification and prospective inference-adapter hooks remain separate work.
The raw fourteen-head profile is explicitly unverified and uncalibrated.

Completed protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/invariant_verification_pool80_12605930_001/`.
It contains `ehr_anchors.jsonl`, `candidate_receipts.jsonl`,
`verification_events.jsonl`, `trial_bindings.jsonl`, status CSVs,
`RESULTS_CN_EN.md`, frozen configuration and source/result hashes.
All 80 anchors, 960 unique candidate receipts, 3,200 original replay trials,
15,001 requested observations and 11,801 transitions were retained. The run
took 6.452556 seconds in the existing CPU allocation, with zero new model,
GPU, patient input, text/image open or submission. Exact replay, all source/
result hashes, every original winner/action/cost/stop reason, old artifacts and
protected modes passed verification. **616 invented-fixture tests pass**.

No-direct-EHR comparisons stay unavailable; unknown/uncertain are not negative.
Legacy global No-Finding source adjustments cannot edit receipt states.
Unchanged artifacts cannot acquire changed classifier/report vectors, and
changed receipt hashes/arithmetic/scope/acceptance claims are refused.
Scalar BioViL/runtime/proxy scores never alter the reference or receipt.
All agreement/improvement remains proxy evidence, not clinical truth. A
report-only change cannot claim image repair. Receipts grant neither clinical
acceptance nor model-execution approval and do not rerank or drop cases.

Next is prospective hook integration with an independently frozen execution
policy/cost ledger and separately approved bounded inference. This interface
test demonstrates integrity/coverage tracking, not improved generation or a
new clinically validated controller.

## CXR-before-report partial verification / 报告前阶段验证 (2026-10-02)

Implemented `src/tricompose_v12/partial_image_verification.py`,
`benchmarks/bind_partial_image_verification.py`, a separate frozen configuration
and twenty-one invented-state tests. See
[the partial-result contract](../docs/partial_image_verification_contract.md).
The original completed-triple verifier and archived full run are unchanged.

`CachedImageEvidence` and `verify_image_phase` support a CXR with an existing
cached XRV result before a report is available. The image projection works with
all report fields absent; report labels, structure, triple gates, scalar scores,
winner flags and costs do not affect this receipt. Its EHR–CXR readout retains
explicit support, opposition and missingness. Both report edges, report
identity/hash/states and all-three support remain `null / not_generated`.
This is not an all-unknown report vector or a perfect-consistency score.

`bind_completed_report` links a completed receipt without overwriting the image
receipt. It requires the same fixed EHR, image identity/hash, XRV vector and
EHR–CXR counts. Receipts use canonical serialized-content validation, including
refusal of Python-equal but hash-different false/zero or integer/float changes.
No clinical acceptance, repair success, confirmed fault or execution permission
is inferred. The eight-head scoped preview remains separate and unchanged.

Validated protected output:
`artifacts/protected/tricompose_v1_2/automatic_replays/partial_image_verification_pool80_12605930_002/`.
It contains `ehr_anchors.jsonl`, `image_partial_receipts.jsonl`,
`completed_report_bindings.jsonl`, `summary.json`, `RESULTS_CN_EN.md`, frozen
configuration and source/result hashes. All 80 fixed EHRs, 240 image partial
receipts and 960 completed-report bindings completed in 1.658559 seconds in
the existing CPU allocation, with zero new model/GPU/API/submission or raw,
report-body/image-pixel access. Exact reconstruction, source/result hashes,
unchanged full-run/underlying EHR anchors and private modes passed; existing-run
overwrite is refused. **637 invented-fixture tests pass**.

Image-phase statuses: nine explicit proxy agreements, six explicit proxy
oppositions, nine missing image comparisons, and 216 image receipts with no
direct comparable EHR constraint. These are 240 image-level cache readouts, not
240 independent patients or counts of clinical successes/failures. The 216
reflect the original 72 no-direct-EHR cases times three image candidates; they
remain included with NA EHR support/coverage. No disease/device was inserted.

This is explicitly **retrospective phase reconstruction**, NOT proof that the
original job verified images before producing reports, saved report calls,
improved generation or ran a new real-time pipeline. Prospective adapter hooks,
a frozen bounded execution policy and measured failure/retry/verifier costs
remain the next step. Every new GPU job needs full-script/resource approval.

## Bounded execution accounting / 有预算的调用执行层 (2026-10-02)

Implemented `src/tricompose_v12/execution_ledger.py`,
`src/tricompose_v12/runtime_dispatch.py`, `benchmarks/smoke_bounded_execution.py`,
a frozen invented-fixture configuration and thirty-nine new tests. See
[the execution contract](../docs/bounded_execution_ledger_contract.md).
The existing generation/scoring policies, protected full/partial verification
runs and V1.0/V1.1 teammate code are unchanged.

The ledger separates CXR generation, XRV, report generation and CheXbert as
individual single-case invocation **attempts**. It fsyncs a private reservation
before backend invocation, retains failed/in-flight charges, limits retries,
and permits explicit zero-new-cost reuse of committed image/scorer results.
Phase dependencies and fixed EHR/image/report/model-audit/seed hashes cannot
change silently. Restart validates the canonical event chain and frozen budget/
retry/mode contract; an in-flight worker is not automatically rerun or refunded.
Python output is redirected to private operation logs, not public stdout.

Validated protected smoke:
`artifacts/protected/tricompose_v1_2/execution_smokes/bounded_execution_fixture_12605930_001/`.
It contains `RESULTS_CN_EN.md`, `summary.json`, per-case durable journals,
ledger snapshots, invented partial/full verification receipts, frozen config,
private fixture logs and source/result hashes. **676 invented-fixture tests
pass**. Exact journal restoration, event/cost arithmetic, protected modes,
all source/result hashes, original full/partial run hashes and no-overwrite
checks pass. Runtime is 0.047964 seconds in the existing CPU allocation.

This uses TWO explicitly INVENTED IDs, not the historical 80-case cohort.
There are ten reserved fixture attempts, one invented timeout, and zero actual
model calls or generation outputs. One fixture reuses its image/scorer and
reaches the five-attempt budget with a second report unscored; the other charges
the image-scorer retry. Unscored/unresolved output is not a negative finding,
clinical failure or acceptance. Partial/full hooks preserve the fixed anchor.

**At this archived fixture stage, only the invented backend was supplied.** The dispatcher/ledger primitives
are implemented, but authentic generator/scorer-provenance subprocess adapters
are not connected yet. A recorded frozen audit hash or receipt ID is not itself
proof of freezing, clinical truth or approval. A deployed adapter must validate
real artifact hashes and checkpoint/scorer provenance, capture child/native
output, retain failed/in-flight journals and enforce Slurm resources. Do not
relabel fresh inference as the historical fourteen-head cache or silently mix
it with the eight-head scoped interface. No GPU/API/download/training/job was
started, and no script has been submitted or implicitly approved.

Next connect those already deployed frozen workers to this bounded execution
layer with a separate small-cohort policy and complete script/resource review.
This is accounting/dispatch readiness, not a completed prospective pipeline,
measured GPU saving or a quality-improvement experiment.

## Real frozen-worker adapters / 真实后端接入 (2026-10-02)

Implemented `live_workers.py`, `live_plan.py`, `live_execution.py`, separate
fresh `live_receipts.py`, CPU preflight and GPU-only runner CLIs. Existing
V1.0/V1.1/model adapters and the archived verification/ledger runs are unchanged.
See [the live-worker contract](../docs/live_frozen_workers_contract.md).

The protected plan is
`artifacts/protected/tricompose_v1_2/live_worker_plans/live_workers2_12605930_001/`.
The original inventory indices 0 and 1, EHR/facts/prompt hashes, real checkpoint
bytes, model audits, scorer sources and frozen threshold bundle passed CPU
preflight in 24.726 seconds. **706 synthetic-only tests pass**. All checkpoint/
source/run hash checks and plan-private modes passed; **zero model calls during preflight**.

Three CXR generators and four CXR-only report experts are registered. The
prepared first P100 subset is RoentGen-v2 + Sana → CXRMate-single, with fresh
XRV and CheXbert verification between phases: four images/four reports, sixteen
normal attempts, maximum ten attempts per fixed EHR including failures/retries.
MAIRA-2's current float32 adapter is not squeezed onto a 16-GB P100. Registered
excluded experts are not newly preflighted/inference-certified by this plan.

The source EHR/prompts are never rewritten/enriched; underconditioned cases
remain with NA direct-EHR support/coverage. Fresh XRV uses eight enabled frozen
threshold heads, **not probabilities** or the old fourteen-head cache. Receipts
record raw positive/negative support, opposition and missingness per edge;
clinical acceptance/repair/primary eligibility stay false/NA. No new scorer,
agent/router training, adaptive quality policy or best-triple selection is added.

After full-script/resource review and explicit user approval,
`slurm/29_live_workers_smoke2_debug_p100.sbatch` was submitted as **job 12615231**:
one P100, two CPU cores, 24 GB host RAM, twenty minutes. It completed on e23-02
in 00:09:10 with exit 0:0. Its stable private output directory contains model
outputs, receipts, durable charged journals and `score_table.csv`; interrupted
runs are retained, not overwritten.
Native subprocess output/caches are private and timeout cleanup targets only
the owned worker group. CPU allocation cannot launch the real GPU workers.

The retained run is `live_worker_runs/live_workers2_12615231/`: **four CXRs,
four reports, sixteen validated calls, zero failures/retries**. The separate
`live_worker_audits/live_workers2_audit_12615231_001/` audit rechecked checkpoint/
source/input/output hashes, actual artifact/tokenizer lineage, exact durable
ledger restoration and all recomputed receipts/score CSV. Worker maximum
allocated VRAM was 7.169 GiB. Framework temporary/cache permissions and the
two Slurm logs were normalized only within this new run to the protected modes;
original source/model/checkpoint files remain unchanged.

Both fixed EHRs lack direct radiographic facts, so **all four completed receipts
remain unverified with NA EHR-related support/coverage**. The successful live
chain is not three-modal clinical correctness, adaptive repair, improved image/
report quality, best-triple selection or measured compute savings. The audit
did not review report bodies/image pixels. Next separately preflight/approve
additional report experts and independent endpoint scoring, then freeze an
adaptive policy/evaluation without fabricating facts or excluding these cases.

## Same-image report choice / 固定胸片的报告对照 (2026-10-03)

Prepared a separate offline control reusing the exact four fresh CXRs and
four CXRMate-single reports. MAIRA-2 will add four reports using the unchanged
official image-only adapter. The explicit frozen proxy policy chooses one
report per fixed image before BioViL-T is invoked; opposition, missing positive
image evidence, positive support and NA EHR coverage remain separate readouts.
Negative agreement is not a reward. No EHR/image is regenerated or enriched.
See [the complete contract](../docs/fixed_image_report_expansion_contract.md).

`prepare_fixed_image_reports.py` completed CPU preflight in 43.230 seconds,
strongly pinning local checkpoint bytes, synthetic artifacts, report requests
and source code, with zero model/runtime calls. **722 synthetic-only tests pass**.
The plan is `fixed_image_report_plans/fixed_reports4_12605930_001/` under the
protected V1.2 root. The A40 debug script is **prepared, not submitted** and
requires separate full-script/resource approval. No new report/endpoint scores
are claimed until that job actually runs.

固定两份 synthetic EHR 和四张 CXR，补 MAIRA-2 报告得到八个图文候选；先按
明确、不可变的标签规则择优，再独立测 BioViL-T。规则把阳性标签缺失也计入
代价，避免只靠省略获得高分；unknown/uncertain 不当阴性，两份报告不当独立
临床投票。当前两份 EHR 的直接影像条件为零，因此 EHR 相关比率仍为 NA。
这一轮是同图报告选择的工程对照，不能宣称完整三模态正确性、错误定位成功、
自适应修复或论文级质量提升。

Subsequent resource check: debug A40 is drained. Prepared
`slurm/31_fixed_image_reports_gpu_a40.sbatch` for the `gpu` partition with the
same frozen plan, one A40, two CPU cores, 48G host RAM and twenty-minute cap.
It also locks down its own Slurm log files. The original debug script is
unchanged. After full-script/resource review and explicit user approval,
script 31 was submitted as **job 12622258**. Initial status was
`PENDING (Priority)`; submission is not execution or a new scoring result.

Added `audits/audit_fixed_image_reports.py` for the eventual completed run:
recompute fixed image receipts, all eight proxy rows, pre-BioViL selection,
paired secondary deltas and CSV; check frozen checkpoint/source hashes,
ordered pre-spawn reservations, missing endpoint reasons and protected modes.
It opens no report bodies/image pixels and makes zero model calls. **734
synthetic-only tests pass**. The audit is ready, not yet run on a completed
GPU output; clinical correctness remains unverified.

The bounded observer `audits/watch_fixed_image_reports.py` is now running in
existing CPU allocation 12621834 (one CPU/1G step, no new GPU or `sbatch`). It
waits at most three hours, polls approved job 12622258 once per minute and
audits only after completed/publication status. No automatic model retries or
new submissions are allowed. Private observer status lives in
`fixed_image_report_watch/observer_12621834_12622258_001/`; the eventual audit
will be `fixed_image_report_audits/fixed_reports4_audit_12622258_001/`, both
under the protected V1.2 root. **740 synthetic-only tests pass**. Observer
startup is not GPU execution, completed audit or clinical validation.

## Fixed-image report control completed / 同图报告对照完成 (2026-10-03)

Approved job **12622258 completed**, exit `0:0`, allocation elapsed **00:02:13**.
The published run is `fixed_image_report_runs/fixed_reports4_12622258/` under
the protected V1.2 root: two unchanged EHRs, four unchanged images, four original
CXRMate-single reports and four new MAIRA-2 reports. All eight pairs have a
BioViL-T endpoint. The frozen selector retains the original baseline on every
image, so its paired secondary gain is zero. This does not prove either report
expert clinically correct or validate tri-modal agreement.

The bounded observer finished with `metadata_audit_completed`; the protected
`fixed_image_report_audits/fixed_reports4_audit_12622258_001/` passed exact
hash/selection/pairing/cost/private-permission checks with no new model calls.
Neither audit nor this update reviewed report bodies or image pixels.

## Cached score coverage diagnostic / 评分覆盖度诊断 (2026-10-03)

Added `src/tricompose_v12/score_coverage.py`,
`benchmarks/diagnose_score_coverage.py` and twenty invented-metadata tests.
See [the diagnostic protocol](../docs/score_coverage_diagnostic_protocol.md).
The fresh eight-enabled-head engineering control and historical fourteen-field
80-EHR development cache are analyzed **separately**, never pooled or rescored.
Every fixed EHR and candidate is retained; no threshold, mask, priority, winner,
generator or API changes. Primary-prefix ties exclude only later ranking terms
for diagnosis; they do not execute an alternative selector.

Protected result: `score_coverage_diagnostics/score_coverage_12621834_001/` under
the V1.2 protected root. Read `RESULTS_CN_EN.md` and `summary.json`; separate
candidate/model coverage and same-image pair CSVs retain NA reasons and lineage.
The CPU program completed in 1.183257 seconds in existing allocation 12621834,
with zero new model calls or submissions. Exact source/artifact hashes, full
recomputation of every CSV/report/summary profile and private modes passed.
**760 synthetic-only tests pass**; that is software verification, not clinical
validation.

新对照的八项启用图像标签全部阴性，半数报告没有可比较的图像标签；因此
“零矛盾”不能当作正确。旧池每张图都有阳性标签，两套分布不能混为一谈。
旧池全部同图报告对均保留，未评分候选明确为 NA，而不是零分或删除。
平局中的 BioViL 差异只能说明两个自动读数的区分能力不同，不能证明临床
错误或据此修改原赢家。后续对照必须先固定覆盖范围、规则和成本。

## Full-bank endpoint request prepared / 全候选补评分请求就绪 (2026-10-03)

Added `complete_endpoint_inventory.py` and `benchmarks/complete_bank_endpoint.py`:
request the complete cached inventory minus the old scored union, regardless of
endpoint values, labels, model priorities or winners. All 960 candidates remain
fixed. The old 596 endpoint records are retained; 364 previously unscored pairs
are frozen for supplementary measurement (at most 216 image/364 text encoder
samples). Old NA scores are not retried, and new full-text failures stay NA.

CPU preflight passed in existing allocation 12621834 in 15.308925 seconds:
all candidate/parent/source hashes, local frozen BioViL weights and project
permissions checked, no report-body/image-pixel review and zero new model calls.
Protected plan: `score_coverage_diagnostics/full_bank_endpoint_request_12621834_001/`.
The [full coverage protocol](../docs/complete_bank_endpoint_protocol.md) explains
why this is still an inspected development cohort, not untouched validation or
permission to optimize BioViL-based winners.

`slurm/32_complete_bank_biovil_gpu_p100.sbatch` is **prepared, not submitted**.
It requests gpu/P100 x1, two CPU cores, 8G host RAM and a ten-minute cap, and
requires complete-script/resource review plus explicit approval. **773
synthetic-only tests pass**. No new endpoint result, clinical accuracy,
regeneration improvement or GPU savings is claimed before execution/audit.

## Full-bank endpoint completed / 全候选图文评分完成 (2026-10-03)

After complete-script/resource review and explicit approval, script 32 ran as
**job 12624822** on **e21-03/P100**, exit `0:0`, elapsed **00:00:47**. All 364
missing pairs were scored using 216 image/364 text encoder samples; zero NA,
program runtime 43.38084 seconds, peak allocated GPU memory 0.611 GiB.
This is supplementary evaluation, not new generation, training or repair.

The lossless CPU merge/audit completed in 5.761636 seconds in existing allocation
12621834. The protected result is
`complete_bank_endpoints/complete_bank_secondary_12624822_001/` under the V1.2
protected root. Start with `candidate_score_table.csv` (all 960 candidates,
unchanged raw edge readouts and newly complete BioViL coverage) and
`RESULTS_CN_EN.md`. All 1,440 same-image report pairs are measured; all eighty
EHRs and the original 596 endpoint records are retained exactly. EHR-edge
denominators remain unavailable when their cached direct findings are absent.

The source ranking prefix ties in 79 report pairs; all now have endpoints,
versus 25 previously. Their raw embedding differences are descriptive evidence
about proxy discrimination, not clinical error localization, a new winner or
an endpoint-tuned policy. **786 synthetic-only tests pass**. Exact source/result
hashes, full tables/report arithmetic, unchanged old scores and private modes
passed verification. See the [completed protocol](../docs/complete_bank_endpoint_protocol.md).

补评分与合并已完成，不是还在排队。现在完整旧候选池都有 BioViL 分数，但
72 份 EHR 缺少直接可比较的缓存影像 finding，因此 EHR 边的相关比率仍为 NA。
不能将图文评分完整误写为所有三模态评分都有可靠依据；原赢家、阈值、病例
内容与原生成结果全部保持不变，没有宣称已证明临床最优或自动修复成功。

## Same-image frozen report-path comparison completed / 同图报告路径对照完成 (2026-10-03)

Added `report_path_comparison.py`, `benchmarks/compare_frozen_report_paths.py`,
`audits/audit_report_path_comparison.py` and 26 invented-metadata tests.
See [the fixed protocol](../docs/frozen_report_path_comparison_protocol.md).
This CPU-only comparison retains all 80 synthetic EHRs, 240 fixed images and
960 reports; five paths per image produce 1,200 comparison rows. Source-only
choices were sealed before reading complete endpoint values. BioViL is
evaluation-only, not a selection input, clinical score or endpoint oracle.

Protected result under `artifacts/protected/tricompose_v1_2/`:
`report_path_comparisons/report_paths_pool80_12621834_001/`.
Start with `RESULTS_CN_EN.md`, `method_comparison.csv` and
`paired_comparison.csv`; `selected_path_score_table.csv` preserves every
selected-path raw edge count and lineage. Separate tables retain each image
generator, direct/no-direct EHR strata and per-EHR paired changes.
Completed in 5.342129 seconds in existing CPU allocation 12621834, with zero
new model calls or Slurm submissions. Manifest SHA256:
`b3e381f35bcab51e0758874097d33b2335ffa512766cdc66a581f9856771172e`.

| Report path | EHR-balanced raw BioViL mean | Raw proxy opposition / image-known facts | Comparable / image-known facts |
|---|---:|---:|---:|
| Fixed MAIRA-2 | 0.5655 | 137 / 2880 (4.76%) | 314 / 2880 (10.90%) |
| Fixed CXRMate-single | 0.3826 | 303 / 2880 (10.52%) | 760 / 2880 (26.39%) |
| Fixed LLaVA-Rad | 0.5918 | 311 / 2880 (10.80%) | 866 / 2880 (30.07%) |
| Fixed CheXagent-2 | 0.4998 | 421 / 2880 (14.62%) | 1072 / 2880 (37.22%) |
| Same-image original source-key choice | 0.4876 | 36 / 2880 (1.25%) | 479 / 2880 (16.63%) |

BioViL first averages the three images within each EHR, then the eighty EHRs.
2880 is a repeated image-finding denominator, not 2880 patients; positive
support and negative support remain separate in the score tables. Source-key
selection supports 196/1332 positive image facts and 247/1548 negative image
facts. Seventy-two EHRs still lack direct cached radiographic constraints;
their EHR-edge rates remain NA. CXRMate here is **CXRMate-single**, not an
EHR+CXR CXRMate-ED path. Historical 14-field readouts stay separate from the
fresh eight-enabled-head smoke test.

The source selector lowers its label-opposition proxy but does NOT establish
overall superiority: its BioViL mean is 0.1043 below fixed LLaVA-Rad and
0.0779 below fixed MAIRA-2, with different coverage. Against LLaVA-Rad, the
same-image selector has higher/lower/equal EHR-mean BioViL in 8/71/1 cases.
It also costs more simulated generator/scorer calls (10 versus 4 per image;
8 versus 2 incremental report/scorer calls once the image is fixed).
There is no measured GPU saving or clinical-accuracy conclusion. All four
fixed report baselines are retained, not only a favorable comparison.

Full recomputation, independent case-mean arithmetic, source/output SHA256,
project-private modes/group and original Sana equivalence (80/80 identical)
passed. Immutable audit receipt:
`report_path_comparisons/report_paths_audit_12621834_001/audit.json`.
**812 synthetic-only tests pass**; this verifies software, not clinical efficacy.
Original EHRs/prompts/images/reports, source key, thresholds and joint winners
remain unchanged. No raw patient input/target, report body or image pixels
were opened. This already inspected cohort is a development control, not
held-out evidence for clinical error localization or successful repair.

下一步应单独预先固定“阳性信息保留、覆盖度与矛盾分开核验”的对照，再在
独立病例上验证；不能用本轮 BioViL 结果反向调权重，或仅凭低矛盾率宣称
自动修复有效。新的 GPU 运行仍需完整脚本/资源展示及明确批准。

## Label-preserving report headroom completed / 保留事实的报告替代空间 (2026-10-03)

Added `report_repair_headroom.py`, `benchmarks/diagnose_report_repair_headroom.py`,
`audits/audit_report_repair_headroom.py` and 32 invented-evidence tests.
See [the frozen diagnostic protocol](../docs/report_repair_headroom_protocol.md).
This is a DEVELOPMENT opportunity analysis, **not a new selection or repair
policy**. It retains all 80 EHRs, 240 images, 960 reports, 2,880 directed report
pairs (4×3 per image) and 1,200 unchanged baseline path slots.

An alternative must preserve specific supported image-positive facts, direct
EHR supports and comparable-fact scope; introduce no new explicit opposition;
pass existing metadata gates; not lower the available structure score; and
strictly improve at least one raw evidence set. Losing a supported disease
while adding another is not preservation. Changing a denial to unknown/uncertain
is not a correction. Unknown image findings remain unknown, not hard negatives.
This diagnostic uses legacy raw four-state labels, not global No-Finding
expansion or BioViL weights. Clinical correctness is still unverified.

Protected completed run under `artifacts/protected/tricompose_v1_2/`:
`repair_headroom/report_headroom_pool80_12621834_001/`.
Read `RESULTS_CN_EN.md` and `method_opportunities.csv`; all directed evidence
changes/endpoints are in `directed_report_pairs.jsonl`, with unchanged baseline
paths in `baseline_path_opportunities.jsonl`. The label-only plan was sealed
before endpoint attachment. CPU runtime was 5.839954 seconds in existing
allocation 12621834, with no new models, inference, submissions or winner changes.

| Unchanged baseline | Fixed images | Images with a strict label-preserving alternative | EHRs with any such opportunity | Qualifying alternative pairs |
|---|---:|---:|---:|---:|
| MAIRA-2 | 240 | 44 | 37 | 54 |
| CXRMate-single | 240 | 76 | 62 | 81 |
| LLaVA-Rad | 240 | 33 | 33 | 34 |
| CheXagent-2 | 240 | 5 | 5 | 5 |
| Original same-image static source-key winner | 240 | 0 | 0 | 0 |

Across all four fixed-model baselines, 174 directed alternatives qualify;
109 have higher and 65 lower secondary BioViL. Among 1,030 directed pairs with
lower raw opposition counts, 1,011 fail the preservation/quality predicate;
978 silence at least one previously comparable conflict. These are overlapping
pair-level diagnostics, not independent patients, confirmed hallucinations or
actual repaired outputs. The conditional endpoint means average ALL qualifying
alternatives within image, then opportunity-bearing images/EHRs. They are not
best-BioViL choices or full-cohort policy effects; zero opportunities stays NA.

The existing static winners have no alternative under this strict predicate.
This is partly structural: the source key already favors related support and
opposition dimensions, so such winners can naturally be non-dominated under
this constrained diagnostic. **Zero headroom is not independent validation,
clinical optimality or proof that existing alternatives/new generations cannot
improve real quality.** Trade-offs and evaluator disagreement remain unresolved;
do not tune a new policy to the observed cosine gaps or automatically prescribe
image regeneration when report alternatives are absent.

Full source/output/old-choice hashes, exact table/report recomputation,
independent set-predicate arithmetic and project-private modes passed.
Immutable audit: `repair_headroom/report_headroom_audit_12621834_001/audit.json`.
**844 synthetic-only tests pass**. No patient source/target, report bodies or
image pixels were opened. The 72 EHRs without direct comparable radiographic
facts remain; report-label headroom does not establish their EHR fidelity.
The next prospective trial needs a separately frozen retry/abstention policy,
compatible scorer provenance, new candidate calls and actual cost accounting.

## Fixed-image native n-best precursor completed (2026-10-03)

The four local report adapters use deterministic decoding; changing seed alone
does not establish new candidate diversity. A narrowly scoped precursor now
extends CXRMate-single's native four-beam output to three sequences per fixed
image, with unchanged official preprocessing, weights and length settings.
See [the exact protocol](../docs/report_nbest_control_protocol.md). This is
**implemented/tested and verified in a two-case GPU engineering run**, not full
online targeted repair. The original 80-case bank and winners are unchanged.

`benchmarks/prepare_report_nbest.py` authenticates two opaque cohort ordinals
and original Sana images, pins source/checkpoint/scorer/threshold hashes, and
stages a private metadata-only subset without claiming old tokenizer traces.
`benchmarks/run_report_nbest.py` is GPU-guarded and sequentially runs native
generation, fresh XRV/CheXbert, a sealed fact-preserving choice and secondary
BioViL-T. All three raw edges, structure flags, duplicates, encoder counts,
stage/native-call journals, wall time and memory remain private/auditable.

Two native calls return six sequences (NOT six independent calls). Rank zero
is the fresh baseline. A lower-rank nonduplicate is eligible only if support
and comparable fact IDs are retained, opposition is not newly introduced or
silenced, structure does not worsen, and finding evidence strictly improves.
Otherwise the baseline is unresolved. Selection cannot see BioViL outcomes;
negative independent endpoint changes and explicit NA remain reported.

The fresh eight-enabled-head diagnostic profile stays separate from legacy
14-field caches. Unknown EHR/image states are not negatives or clinical truth.
The batch journal is not the adaptive single-case ledger, and this precursor
does not claim targeted repair, clinical success, GPU savings or paper efficacy.
All generated candidates stay under `artifacts/protected/`. Any further GPU
script must be shown in full and explicitly approved before submission.

CPU preflight completed in existing allocation 12621834 (9.514 seconds), with
zero model factories instantiated and zero new inference calls. Immutable plan:
`artifacts/protected/tricompose_v1_2/report_nbest_plans/nbest2_12621834_001/`;
manifest SHA256:
`783d4540b9960b14adc82afcf45a0e422dd8c9a71e248fefc21c4cc83bbebebe`.
Source/asset/metadata equality and protected group/modes were checked again.
**867 synthetic-only tests pass**. The reviewed submission script was:
`slurm/32_report_nbest_debug_p100.sbatch` (one P100, two CPUs, 24GB host RAM,
20-minute wall limit).

After explicit approval, job **12629940** completed on debug P100 in **56
seconds** (controller wall 53.408 seconds). Two native calls returned six
reports, with three exact/normalized distinct texts per image. Generation-only
times were 0.9704 and 0.7915 seconds; generator load was 7.686 seconds. Peak
allocated generator VRAM was 0.7169 GiB. Sequential startup, hashes, scoring
and independent verification explain the rest of the end-to-end wall time;
this is not total-GPU-utilization accounting.

The fresh XRV scored two images, CheXbert six reports, and BioViL encoded two
images/six texts, all six endpoints available. **Zero of four alternatives
passed the strict evidence-improvement gate**, so both top-one baselines were
retained as unresolved. Distinct surface text did not provide a qualifying
finding-evidence improvement. The selected-minus-baseline BioViL delta is zero
because selection did not change, NOT proof of successful repair.

Both predeclared EHRs have zero directly comparable cached radiographic facts;
their EHR edge support/coverage rates remain NA. Each beam rank has 6/16
comparable image-reference facts, all six negative supports and zero explicit
oppositions; no image-positive reference facts are available in these two
classifier readouts. Low opposition alone does not establish completeness or
clinical correctness. This small precursor verifies interfaces and honest
abstention, not full three-modal accuracy, online repair or efficacy.

Immutable completed output:
`artifacts/protected/tricompose_v1_2/report_nbest_runs/nbest2_12629940/`.
Read `score_table.csv`, `selection.json` and `comparison.json`; generated texts
are in `reports/candidates/`, and costs in both controller/native journals.
Recomputed receipts, decisions, endpoint arithmetic, CSV raw counts/NA,
duplicate counts, source/artifact hashes and protected modes passed:
`artifacts/protected/tricompose_v1_2/report_nbest_audits/nbest2_result_audit_12629940/`.
No raw patient inputs/targets, report bodies or image pixels were opened by
the CPU audit. Only the explicitly approved GPU workers consumed synthetic
images/text internally. Original thresholds, inputs and winners remain fixed.

## Same-image four-expert fresh-profile control completed (2026-10-03)

The n-best pilot showed surface-text diversity without a finding-vector change.
The next separately declared control compares the EXISTING MAIRA-2,
CXRMate-single, LLaVA-Rad and CheXagent-2 outputs on those same two original
Sana images, without generating anything again. See
[the exact four-expert protocol](../docs/report_expert_control_protocol.md).
It is **completed and metadata-audited** (job 12630490). No clinical efficacy
or repair success is implied.

The eight frozen reports are staged as metadata-only copies with unchanged
artifact paths/hashes. Historical and newly generated CXRMate top-one hashes
match for both images. Reuse only the exact, authenticated fresh eight-enabled-
head XRV bundle from job 12629940; legacy 14-head XRV and old report labels are
not mixed in. Re-score all eight reports with frozen CheXbert, compute fresh
receipts and seal the choice before independent full-text BioViL-T.

The cross-expert rule requires the same fact-ID preservation/non-silencing
constraint, but uses official MODEL-SPECIFIC section contracts: CXRMate needs
findings/impression, the other experts need findings only. Missing an optional
impression is not a penalty. Common temporal/generic/repetition risks cannot
worsen. Priority is predeclared; BioViL never chooses the expert. The published
n-best predicate and old winners remain unchanged.

Added `report_expert_control.py`, `benchmarks/prepare_report_expert_control.py`,
`benchmarks/run_report_expert_control.py`, and 22 invented-fixture tests.
**889 synthetic-only tests pass**. CPU preflight (existing allocation 12621834)
took 9.684 seconds and instantiated no model. Protected plan:
`artifacts/protected/tricompose_v1_2/report_expert_plans/experts2_12621834_001/`.
Manifest SHA256:
`123ed19ca10cc04d4b90f32f0192515e2e0ae1915a21336810824845e0c6b2ac`.
Image receipt reconstruction, source/report hashes and project-private
permissions passed; preflight audit is under
`artifacts/protected/tricompose_v1_2/report_expert_audits/experts2_preflight_audit_12621834_001/`.

Reviewed exact script: `slurm/33_report_experts_debug_p100.sbatch`, one P100,
two CPUs, 24GB host RAM and ten-minute wall cap. It performs no new generation
or XRV inference; only eight fresh CheXbert report samples and up to two/eight
BioViL image/text encodings. Historical generation costs are not erased or
called GPU savings. It was shown in full and explicitly approved before
submission. Outputs include `score_table.csv`, `model_comparison.csv`,
sealed decisions and actual verification costs. The two EHRs remain sparse;
their EHR-edge rates stay NA, and this is not online targeted repair or an
independent clinical benchmark.

Job 12630490 completed in **33 seconds** (controller 31.007 seconds); all eight
BioViL endpoints were available. Peak allocated VRAM was 0.448 GiB for CheXbert
and 0.579 GiB for BioViL. **Zero of six alternatives passed** the declared
fact-preservation/no-new-opposition gate. Both original CXRMate reports remain
unresolved baselines. More expert calls did not yield an eligible improvement
on these two fixed images; we do not change thresholds or relax the gate to
manufacture a positive result.

| Report expert | Comparable / image reference facts | Supports (+ / -) | Proxy oppositions | Mean raw BioViL-T cosine |
| --- | ---: | ---: | ---: | ---: |
| CXRMate-single | 6 / 16 | 0 / 6 | 0 | 0.5460 |
| MAIRA-2 | 1 / 16 | 0 / 0 | 1 | 0.0747 |
| LLaVA-Rad | 8 / 16 | 0 / 6 | 2 | 0.1709 |
| CheXagent-2 | 6 / 16 | 0 / 3 | 3 | 0.5009 |

Here 16 means eight enabled classifier heads on each of two images, NOT
clinical ground truth. All explicit image-reference states happen to be
negative; both EHRs have zero direct constraints. The EHR edges therefore
remain NA. These two cases establish neither full triple fidelity nor general
expert ranking. All eight reports pass their existing per-model section
contracts, but structure checks and cosine are not clinical factuality.

Protected output:
`artifacts/protected/tricompose_v1_2/report_expert_runs/experts2_12630490/`.
Recomputed metadata audit:
`artifacts/protected/tricompose_v1_2/report_expert_audits/experts2_result_audit_12630490/`.
Eight source-bound receipts, sealed choices, raw CSV counts/NA, model aggregates,
endpoint arithmetic, source/checkpoint hashes and project-private permissions
passed. No original bank output/winner was overwritten. See the protocol for
the audit hash and scientific interpretation; no report bodies/pixels were
opened by the CPU audit.

Added a metadata-only result auditor and nine invented-fixture regression
tests for independent aggregation, exact CSV/NA preservation and Slurm guard
ordering. **898 synthetic-only tests pass** (9.752 seconds in the existing CPU
allocation). Pinned generator/scorer implementations were not modified.

## Full fixed-pool fresh report benchmark completed (2026-10-03)

After both small controls abstained, expand the inventory, not the gate:
retain all **80 fixed synthetic EHRs / 240 original CXRs / 960 existing
reports**. Re-score all images with the same frozen eight-enabled-head XRV
profile and all reports with frozen CheXbert. Do not mix legacy image/report
labels into the fresh receipt profile or regenerate samples. The same-image
four-expert rule remains unchanged; original winners stay immutable.

Re-use the authenticated 960-pair full-report BioViL cache only AFTER sealing
choices. Report 12 fixed generator/expert combinations plus three same-image
first-eligible selectors. Statistical comparisons resample EHR cases after
within-case aggregation, not images or 960 correlated candidate pairs. This
is an exploratory previously inspected pool, NOT an untouched final test or
an online repair/clinical-efficacy result. Missing direct EHR evidence remains
explicit and its rates are NA rather than negative or perfect agreement.

Prepared plan:
`artifacts/protected/tricompose_v1_2/full_pool_report_plans/pool80_fresh_12621834_001/`.
Manifest SHA256:
`ffa4691cb2df67ba6d7c8e8a7bbd235fff1b592d82620837f6b26e1011ae65ca`.
CPU preflight took 17.064 seconds with zero factories/inference; **917 tests
pass**. See [the full-pool protocol](../docs/full_pool_report_control_protocol.md).

The exact script `slurm/34_full_pool_fresh_scores_debug_p100.sbatch` requests
one P100, two CPUs, 24GB host RAM and a 20-minute cap, including a CPU metadata
audit. After full script review and explicit approval, **job 12631194 completed
in 87 seconds**, including automatic CPU audit (controller 64.740 seconds;
audit 14.629 seconds). The previous two-case results remain valid and unchanged.
Any further GPU script still requires separate full review and approval.

Full metadata/source-hash preflight also passed, including canonical EHR,
facts and final-prompt byte hashes. Its protected audit is
`artifacts/protected/tricompose_v1_2/full_pool_report_audits/pool80_preflight_12621834_001/`.
Exactly eight fixed EHRs have direct radiographic states; 72 do not. Both
strata remain in the cohort, and this direct-evidence criterion is not the
earlier prompt/context-conditioning criterion. No secondary values or
clinical bodies/pixels were opened during preparation. The approved GPU workers
subsequently consumed synthetic images/text internally; the CPU audit did not.

The fresh full-pool run has **96/720 gate-passing alternatives**, switching
reports on **70/240 images**, spanning 63/80 EHRs. The other 170 reports remain
unresolved CXRMate baselines, not accepted clinical results. All 960 historical
secondary pairs remain available. Across fixed image paths, CXR/report supports
rise 554→640 (positive 79→97, negative 475→543), proxy opposition falls 114→107,
and mean BioViL rises 0.3826→0.4684 relative to fixed CXRMate. Gains in the gate's
own label metrics are not independent proof of clinical correctness.

| CXR path | CXRMate BioViL | Selected BioViL | Paired change |
| --- | ---: | ---: | ---: |
| Sana | 0.6258 | 0.6158 | -0.0101 |
| PixArt | 0.6088 | 0.6540 | +0.0452 |
| RoentGen-v2 | -0.0867 | 0.1354 | +0.2222 |

Important negative/control results: the aggregate gain is concentrated on the
low historical RoentGen/CXRMate baseline, and fixed LLaVA-Rad has higher pooled
BioViL (0.5918) than the selector (0.4684). Fixed MAIRA-2 and CheXagent-2 also
have higher direct EHR/report supports (nine vs six). EHR/image scores cannot
improve here because images are unchanged. This does not establish superiority
to the best fixed model, complete triple fidelity, fault localization or online
regeneration/cost savings. Case-level bootstrap intervals and all negative
deltas are reported in the protocol and private handoff; this old pool is not
an untouched final test.

Protected scores/choices:
`artifacts/protected/tricompose_v1_2/full_pool_report_runs/pool80_fresh_12631194/`.
Automatic recomputed metadata audit:
`artifacts/protected/tricompose_v1_2/full_pool_report_audits/pool80_fresh_12631194/`.
Separate immutable bilingual report:
`artifacts/protected/tricompose_v1_2/full_pool_report_summaries/pool80_fresh_12631194/RESULTS_CN_EN.md`.
The report includes all fixed-expert comparisons, raw label counts/NA,
secondary tradeoffs and direct-EHR strata. Original run outputs were not
modified to add this summary; no generated report body/image is in public docs.

Aggregate handoff generation and six invented-metadata regressions were added
outside the pinned worker source trees. **923 synthetic-only tests pass**
(12.864 seconds in the existing CPU allocation). The separate report's manifest
SHA256 is `2ab96ae8e6576a972b2dac88b979b7c87e7f2036e2ad4cc6cde0cedc2c7901bd`.

### Bounded actual regeneration: completed primary run, no accepted replacements

The next interface is a new [two-case bounded regeneration control](../docs/bounded_regeneration_smoke_protocol.md),
not another rescore of the old candidates. A fixed EHR/unchanged final prompt
can trigger one actual RoentGen seed-1 CXR followed by the unchanged official
CXRMate-single reporter and fresh XRV/CheXbert verification. Direct EHR/XRV
opposition is explicitly a heuristic trigger, not validated clinical fault
attribution. Missing evidence abstains; report consensus alone cannot trigger
image repair. The EHR/prompt cannot be changed or replaced.

Cases are the first two sorted EHR anchors with facts covered by enabled scorer
heads, chosen without image/report/endpoint outcomes. The original 80-case
denominator remains visible. A retry must strictly improve fixed EHR/image
evidence versus both fixed and historical static references, with no supported
fact/comparability loss, no new opposition on any edge, and no worse report
structure/risk. Otherwise keep the old static candidate, record unresolved,
and stop. New winners are separate; all historical winners stay immutable.

At most four charged generation/verification attempts per triggered EHR,
including failures, no operational retries, 120 seconds per subprocess.
Secondary full-report BioViL runs only after a choice seal; its encodings and
failures are charged separately rather than hidden in the four-call limit.
This small trial does NOT establish equal-budget superiority, clinical
localization or three-modal repair efficacy. The GPU job must still receive
explicit approval after the complete script/resources are shown.

CPU preflight and execution-loader validation passed in existing allocation
12625457 with zero factory/model calls; the sealed plan is
`bounded_regeneration_plans/retry2_12625457_001/` under protected V1.2 outputs.
The cohort has eight direct-fact EHRs and five enabled-head EHRs; the fixed
EHR-only stratification selects opaque source indices 5/11. Both trigger the
declared heuristic retry. **951 synthetic-only tests pass** (14.192 seconds).
`slurm/35_bounded_regeneration_debug_p100.sbatch` was fully shown and explicitly
approved, then submitted unchanged as **job 12632531**: debug/P100 ×1, two CPUs,
24G host RAM, twenty-minute cap. It started on `e23-02` immediately and completed
in **4m48s**, exit `0:0`, with two new images, two reports, eight charged primary
attempts, zero primary failures, and a passed automatic metadata audit.
Protected output: `bounded_regeneration_runs/retry2_12632531/`.

**0/2 retry replacements pass**: fixed EHR/image supports remain 0/2 and proxy
oppositions remain two. New reports add eight negative image/report supports,
but lose the static method's one positive support and do not fix EHR/image
evidence. Repetition/temporal-risk nonregression also fails for some references.
Keep both original static results, do not change gates, and do not claim repair
efficacy or fault-localization accuracy. The immutable bilingual handoff is
`bounded_regeneration_summaries/retry2_12632531/RESULTS_CN_EN.md`.

Original secondary BioViL failed with ModuleNotFoundError because this new
invocation omitted the existing vendor bootstrap. Its 7.651s attempt is charged
and original scores remain NA. After complete script/resource display and
explicit approval, the separate pinned recovery worker/plan and
`slurm/36_bounded_retry_secondary_debug_p100.sbatch` ran unchanged as **job
12633003**, debug/P100 ×1, two CPUs, 8G host RAM, five-minute cap. It completed
in **24 seconds**, exit `0:0`, with no generation, primary rescoring or
source/winner changes. Recovery wall time including load/I/O was 16.356 seconds,
peak VRAM 0.583 GiB; four image and five text encodings scored all five pairs.
The original failed attempt remains charged. **963 synthetic-only tests pass**
(12.820s); the 12 recovery/summary regressions were also rerun successfully.

| Two-case method | Mean full-report BioViL raw cosine | EHR/image support / known | Proxy opposition |
| --- | ---: | ---: | ---: |
| Fixed RoentGen/CXRMate | -0.0909 | 0/2 | 2 |
| Historical static | 0.0673 | 0/2 | 2 |
| New retry candidates | 0.7446 | 0/2 | 2 |
| Retained bounded selection | 0.0673 | 0/2 | 2 |

Higher image/report similarity does **not** establish fixed-EHR fidelity. These
two new candidates still fail the preregistered primary gate; BioViL was read
only after choice sealing and never changes the choice. Proxy opposition is
not independently verified clinical error; no repair/localization efficacy
claim follows from this small previously inspected development cohort.

The recovery sidecar is
`bounded_regeneration_endpoints/retry2_endpoint_12633003/`, with five-row
`score_table.csv` and `case_comparison.csv`. Its independent CPU metadata audit
passed in 5.090 seconds, including source/checkpoint hashes, unchanged selection,
raw-score/CSV recomputation, encoding caps and protected permissions:
`bounded_regeneration_audits/retry2_endpoint_audit_12633003/`.
The new immutable bilingual handoff is
`bounded_regeneration_summaries/retry2_with_secondary_12633003/RESULTS_CN_EN.md`,
manifest SHA256 `a7b6b47013f9bce8137de6bc24b9f9b2cfddfacc62bed2bbd3cbf610009a4caa`.
The original failed run, original NA summary and historical selections remain
unchanged. Every further GPU submission still requires separate approval.

### EHR/image scorer reliability audit: metadata correct, clinical gate unresolved

The [cached reliability audit](../docs/ehr_cxr_reliability_audit.md) recomputed
3,388 saved image-state entries and both original retry decisions without
discrepancy. It used existing CPU allocation 12625457, no inference or new job,
and no real source/body/pixel reads; all historical artifacts stay unchanged.
Twenty-two relevant invented-fixture tests pass.

The existing report-derived weak-reference test is not independent image truth.
For the triggering pneumonia head, fitted test sensitivity is 9/22 (0.4091)
and balanced accuracy 0.5855; the historical default-0.5 BA is 0.6245. Old/new
synthetic scores are below both thresholds, not near-boundary flips. No
threshold/gate is changed to accept the new images. Only five of eighty fixed
EHRs have explicit facts covered by active XRV heads; absent comparisons stay
missing. A high BioViL endpoint cannot resolve these evaluator limitations.

New protected diagnostic report:
`ehr_cxr_reliability_audits/retry2_reliability_12625457_001/RESULTS_CN_EN.md`,
with exact synthetic score margins and eight-head weak-reference aggregate
metrics in separate CSVs. Runtime 3.410s; manifest SHA256
`583fbae662f9ff98c4b9bd43de192127748aa11f29b810ce905018d902f9ca15`.
Next audit image-annotated benchmark availability/provenance before more
regeneration; no clinical localization or repair success is established here.

### Image-reference resource check: acquisition/access still needed

The [reference-resource audit and integration proposal](../docs/image_reference_benchmark_preparation.md)
checked three shared data roots through bounded, read-only metadata inspection
(433 entries). Existing MIMIC automatic/manual report labels were confirmed;
no independent image-annotation resource was confirmed within that scope.
No real data row or pixel, archive content or credential was opened.

The proposed first candidate is official VinDr-CXR image-level annotations,
including pneumonia, subject to local availability and dataset-specific DUA.
Do not substitute the installed XRV VinBrain bounding-box-label helper for the
global-pneumonia annotation source. The current all-source checkpoint declares
RSNA/NIH and CheXpert training, so those datasets cannot simply be called
independent unseen-image tests. VinDr overlap and DICOM preprocessing still need
verification; no new benchmark result or clinical qualification is claimed.

Private inventory/report:
`reference_resource_audits/image_reference_12625457_001/`; manifest SHA256
`7a29156768423f9099e2bcda5ea41bfff80a8fe6c920df680f6de6c8d61cc4e5`.
Nothing was downloaded or submitted. Authorized data access is needed before
preparing an executable, separately reviewed GPU validation request.

### VinDr-CXR readiness scaffold: no real benchmark executed

`real_validation/vindr_readiness.py` now supports a no-data readiness receipt,
an optional unauthenticated HEAD-only check, and a separately guarded authorized
header-only inspection. The [integration documentation](../docs/image_reference_benchmark_preparation.md)
specifies the non-secret access attestation and remaining scientific gates.
Default execution reads no dataset row, pixel or credential and runs no model.
It never downloads a dataset, refits thresholds, changes winners or submits jobs.

The protected run `vindr_readiness_runs/readiness_12625457_001/` returned
`blocked_authorized_local_data_required`; the optional HTTP probe returned 403
without authentication, redirects or a response-body read (0.713s). This does
not assess the user's authenticated access. Manifest SHA256:
`c9d01274d9465a6c20ff43fae2fa6d4f003a61ff9bd45a2eebe66bad9e3b3ca7`.
Its hashes and 2770/0660 project-private permissions were checked independently.
Twenty-five new invented-fixture tests pass; the full V1.2 regression suite
passes all 988 tests in 14.394 seconds. No real VinDr schema or decoder
has been validated, no benchmark scores are available, and all earlier retry
decisions remain unchanged. Genuine dataset-specific DUA/training and group
reader eligibility confirmation, plus an authorized local path, are required
before the next source preflight. Every new Slurm submission still needs the
complete script/resources displayed and explicit user approval.

### Public RSUA alternative: approved diagnostic completed

The [RSUA pneumonia pilot protocol](../docs/rsua_pneumonia_pilot.md) records an
accessible public CC BY 4.0 resource while VinDr-specific access is unavailable.
Only the official Validated ZIP was downloaded: 34,738,327 bytes, with exact
published SHA256 verification. Header-only inspection found 292 chest images
and 292 masks, each in two formats, not 1,168 independent patients.

A fixed seed-0, image-level 25-pneumonia + 25-normal-proxy cohort is sealed at
`real_validation/rsua_pilot_cohorts/cohort50_12625457_001/` below the protected
V1.2 root. COVID-19, masks and alternate NPY copies are excluded. The paper
calls the comparison group normal, while the repository calls it Non-Covid;
its negative state remains a published-cohort proxy, not independently
adjudicated image truth. Segmentation-mask validation does not validate disease
labels. Patient grouping, age domain and training-image overlap are unverified.

After full script/resource display and explicit approval,
`slurm/37_rsua_xrv50_gpu.sbatch` was submitted unchanged as job **12636566**:
one GPU, two CPUs, 8G RAM and five minutes. It ran on a P100 after about six
seconds in queue and completed in **23 seconds**, exit `0:0`. The worker took
11.687416s and 0.045 GiB Torch peak allocated VRAM, scoring all 50 fixed images.
Checkpoint, preprocessing and default/saved weak-reference thresholds stayed
unchanged; no training, calibration, generation or historical winner changes.

Pneumonia AUROC is **0.7248**, AP 0.69888206. Default-0.5 sensitivity,
specificity and balanced accuracy are 0.48/0.76/0.62; unchanged transported
threshold 0.55457607 gives 0.24/0.92/0.58. The false-negative risk means XRV
negative alone cannot establish clinical absence in a generated image. This
also does not establish earlier generation correctness. The cohort-class proxy
limitations remain, and `primary_metric_eligible` stays false.

Protected result: `real_validation/rsua_xrv_pilots/rsua_xrv50_12636566/`.
Bilingual handoff: `real_validation/rsua_pilot_reviews/review_12636566_001/RESULTS_CN_EN.md`.
Result-manifest SHA256:
`92a64874818f37e0d363e8406064d4619cb9edefa01ccab43881cf1a1df7d0e0`.
Hashes and group-private permissions passed an independent metadata check.
Full V1.2 invented-fixture regression passed 1,011 tests (13.019s) before
submission. Future Slurm submissions require a new complete script/resource
display and explicit approval.

### RSUA same-cohort secondary evidence: completed diagnostic

The [BioViL-T follow-up protocol](../docs/rsua_biovil_followup.md) reuses the
exact 50 fixed RSUA images to test three predeclared pneumonia presence/absence
probes and their fixed mean. It measures margin AUROC/AP, class-specific
polarity wins, ties and disagreement with both unchanged XRV profiles; it does
not convert correlated model votes into clinical adjudication or repair gates.
The official BioViL BMP incompatibility is handled only inside approved Slurm
by pixel-verified lossless grayscale PNG conversion, leaving the original
inputs and official model preprocessing unchanged.

The immutable protected plan is
`real_validation/rsua_biovil_plans/plan_12636566_001/`, manifest SHA256
`043e8f8ad708fa84fe2529a48ad0a0c2139e211e9814199fffff673ccb922e5b`.
No model calls or pixel decodes occurred during preparation. All 1,027
invented-fixture tests pass (13.056s), including 16 new tests. Prepared script
`slurm/38_rsua_biovil50_gpu.sbatch` requests one flexible GPU, two CPUs, 8G
RAM and five minutes: 50 image encodings plus two fixed-order replays, one
eight-text batch including duplicate controls. After full script/resource
display and explicit approval, job **12637081** ran unchanged on P100 e21-03:
approximately 23s in queue, **15s** elapsed, exit `0:0`; worker 8.8523s and
0.606 GiB Torch peak allocated VRAM. All 50 pixel-preserving conversions and
artifact/permission checks passed; image/text control differences were zero.

The predeclared three-template mean-margin AUROC is **0.5520**, AP 0.58437032,
positive polarity wins 21/25, negative wins 8/25 and balanced wins 0.58, with
no ties. All three templates are reported, not post-hoc selected. This weak
cohort discrimination and presence preference does not qualify BioViL as a
direct clinical pneumonia judge or invalidate all its retrieval uses. There
are 22/50 disagreements with default XRV and 32/50 with transported-threshold
XRV. Both agreement and disagreement contain opposite proxy-reference classes;
no majority-vote truth or regeneration decision follows.

Protected result: `real_validation/rsua_biovil_pilots/rsua_biovil50_12637081/`.
Bilingual report and aggregate CSV:
`real_validation/rsua_pilot_reviews/review_biovil_12637081_001/`.
Result-manifest SHA256:
`a8eed9db9e20146cb1bc7ca597aed78f8bfb5cd1683b29e82dde99497b08992a`.
Old XRV results, thresholds, gates and winners remain unchanged. Every future
GPU submission still needs complete script/resources and separate approval.

### Full-bank reliability sidecar: completed, no rescoring or winner change

The [lossless reliability sidecar](../docs/candidate_scorer_reliability_sidecar.md)
annotates all **960 candidate rows / 80 fixed EHRs / 240 CXR slots** using
existing cached synthetic finding states and real-benchmark aggregates only.
No bodies, pixels, real source inputs or model weights are opened, and no
model calls, new Slurm submissions, ranking or threshold fitting occur.

All original 50 columns, 48,000 cell values and row order remain unchanged;
18 `reliability_` columns are appended. The output includes 13,440 four-state
fact rows, 3,360 image/finding dependency groups and a scoped scorer profile.
There are 428 distinct report-artifact hashes. Under the historical cached
direct-fact rules, 8/80 EHRs have explicit comparison facts and 72/80 do not;
that does not mean those EHRs lack clinical content. No case was dropped or
altered. Unknown/uncertain stay unchanged and missing evidence is not a zero
or perfect score.

Cached CheXbert report proposals have explicit positive–negative disagreement
in **206/3,360 same-image/finding groups**, affecting 544/960 candidate rows.
These are correlated proxy disagreements, not independently verified clinical
errors or majority-vote truth. Candidate-level same-finding image-scorer
disagreement remains unmeasured/NA. RSUA pneumonia metrics are not transferred
as synthetic labels, thresholds or weights; full-report BioViL cosine remains
secondary retrieval evidence, not a finding-negation or fixed-EHR fidelity
score. The old fourteen-state and newer eight-enabled-head profiles are not
pooled. Clinical fault localization and regeneration remain unqualified.

Protected output:
`candidate_reliability_overlays/reliability_pool960_12632006_001/`, including
`candidate_score_table.csv` and `RESULTS_CN_EN.md`. Result-manifest SHA256:
`d057efecd06a4db11c6fec27e79f2ecf085b1c7afe0fa7174fce554c8e85b673`.
Execution used the existing CPU allocation 12632006. Twenty new invented-fixture
tests and all **1,047 V1.2 regressions** pass (10.527s). Independent read-only
checks verified original cells/order, all fact/count/group relations, six
output artifact hashes, 13 source hashes and project-private permissions.
Historical scores and selected triples remain sealed and unchanged.

### Conservative full-bank evidence preview: completed, zero model calls

The [reliability evidence-request preview](../docs/reliability_evidence_preview_protocol.md)
now consumes the sealed full-bank sidecar, keeping **80 EHRs / 240 CXR slots /
960 candidate triples** and the old fourteen-finding profile. It does not
extend the older eight-finding scoped interface, alter EHRs/rankings/winners,
execute verification or authorize clinical acceptance, rejection or repair.

Only **15/96 explicit-EHR candidate/finding occurrences** have explicit states
for all three modalities: nine agree, two have an image-proxy opposition, one
has a report-proxy opposition and three have both downstream proxies opposing
EHR. These are repeated candidate/finding occurrences, not independent
patients or clinical truth. Other missing comparisons remain unknown/uncertain.

959 candidates request verification in preview and one retains unresolved
agreement. This does **not** mean 959 generation failures. The 3,526
candidate/request links deduplicate to **2,072 logical finding-level requests**:
72 EHR-observability checks, two conditioning/fact-scope checks, 220 image
finding checks, 606 report-assertion checks and 1,172 image/report-relation
checks. Shared image/report dependencies are reused; no majority voting or
BioViL-based request ranking is used. Logical requests are not model calls;
additional budget, runtime and compute savings remain unavailable.

Protected output:
`reliability_evidence_previews/evidence_pool960_12632006_001/`, with
`decision_preview.csv`, full `decisions.jsonl`, deduplicated
`evidence_requests.jsonl` and `RESULTS_CN_EN.md`. Manifest SHA256:
`b38227d1e42bd36e90718558f837a5f442d4cf84bde2795f43b13050db629a9f`.
It used existing CPU allocation 12632006 without new Slurm/model/API calls or
body/pixel reads. All **1,075 tests** pass (10.581s), including 28 new invented
fixtures. Independent checks passed for source/output hashes, candidate order,
fact/trace arithmetic, request dependency/consumer inventories and protected
permissions. Next bind a small frozen request subset to existing automatic
verification; every GPU submission still requires separate script review and
approval, and all historical outputs remain unchanged.

### Frozen report-text scope smoke: completed without GPU/model calls

The [historical report-scope smoke](../docs/legacy_report_scope_smoke.md) fixes
the first two sorted opaque EHRs and all **24 reports / six image slots** before
text inspection, without scores/winner/guard-coverage case selection. It reads
only these synthetic texts internally in existing CPU allocation 12632006;
no real target, EHR body, image pixel, model, new job or external API is used.

The unchanged four-finding literal gate retains 16 original proposals,
abstains on 26 and leaves 54 original missing assertions unknown (96 supported
checks). Another 240 rows keep unsupported-head status; all 336 fourteen-head
raw states remain intact. **All 16 retained proposals are negative**, on
effusion/pneumothorax; no cardiomegaly/consolidation proposal is retained.
This demonstrates limited rule coverage in this subset, not that the 26
withdrawn model proposals are wrong. Do not use this checker as the sole
ranking/regeneration gate or fit new rules to these observed cases.

Within the same four supported heads, CXR/report comparability falls 42→16,
support 25→7 and opposition 17→9. Lower opposition reflects abstention/lost
coverage, not improved generated artifacts. Report-disagreement groups fall
4→0 by withdrawing evidence, not confirmed repairs. EHR-edge rates remain NA
because these two cases have no explicit cached reference findings; both EHRs
are retained. No original score, threshold, report, EHR or winner changes.

Protected plan: `legacy_report_scope_plans/scope2_plan_12632006_001/`.
Protected receipts/table/report:
`legacy_report_scope_runs/scope2_checked_12632006_001/`, with
`candidate_scope_table.csv`, `scope_fact_table.jsonl` and `RESULTS_CN_EN.md`.
Result-manifest SHA256:
`64b46bd52292462274e4a9ac2f16aade397189f945081060a906875032733e3b`.
All **1,096 tests** pass (10.453s), including 21 new invented fixtures;
independent state/count/group/offset/hash checks and protected permissions
passed. Twenty-four report file slots/22 unique text hashes were verified.
Source offsets/hashes are stored without copying report quotes to new outputs.
Next prepare a frozen semantic assertion check on the same cohort rather than
blindly expanding the narrow gate; every GPU job still requires separately
displayed complete script/resources and explicit approval.

### Same-cohort frozen semantic report check: completed, limited availability

The [semantic-check protocol](../docs/legacy_report_semantic_check.md) adds a
separate frozen Qwen assertion extractor for the same **24 report slots / 22
distinct exact texts** as the literal-scope smoke. It reuses the existing
four-head prompt, checkpoint, model loader and strict quote decoder without
changing EHRs, reports, old states, scores or winners. At most 22 greedy calls,
512 output tokens, no retries, no new training/API/downloads. Failed or
unmatched evidence stays unavailable, not negative. Quotes establish source
traceability, not independently validated semantic truth.

The metadata-only plan is sealed at
`legacy_report_semantic_plans/semantic_scope2_12632006_001/`, manifest SHA256
`2c76baa73aea4b0af0bd4745ff164387727107a627a519524122c1ce84a99d2c`.
Worker: `benchmarks/verify_legacy_report_semantics.py`; submission proposal:
`slurm/39_legacy_report_semantics_flexible_gpu.sbatch`. After complete
script/resource display and explicit approval, **job 12639326 completed on
2026-10-04 using L40S in 1m56s**, exit code 0:0, with exactly 22 model calls
and no retries. Every further submission requires complete script/resource display
and explicit approval. This is secondary extractor
comparison, not a new clinical ranking, accepted repair or fault localization.
All **1,114 V1.2 tests** pass (10.647 seconds), including 18 new invented-fixture
tests. Sealed parent outputs, 157 source fingerprints, protected permissions,
shell syntax and refusal to overwrite the metadata plan were verified.

Strict quoted-evidence availability is **10/22 distinct responses**. Twelve
responses are unavailable: nine quote/source mismatches, two polarity schema
failures and one ambiguous source location, not twelve confirmed bad reports.
Across 24 slots × four heads: 11 positive, 35 negative, 50 unknown assertions;
CheXbert/Qwen share 22 explicit states, oppose on two and cannot compare 72.
All 336 fourteen-head rows remain present. No EHR edge, score or winner changed;
source-grounded assertions are not clinical truth. The two fixed EHRs still
lack explicit cached reference facts for EHR-edge comparisons.

Receipts: `verification_runs/qwen_scope2_12639326/` (manifest
`d543ac0287b40bc1596944f348a1780b76c742b60a5cdc7decaddc8d2352dca4`).
Comparison: `verification_runs/qwen_scope2_12639326_analysis/` (manifest
`86052b941f55749a16eed660c7bf33e2d086ce43741e4208ef82d67d7741bc6e`).
Independent hashes/counts/lineage/permissions passed; peak allocated GPU
memory was 15.604 GiB, not authorization to reduce the memory guard.

CPU-only frozen whitespace diagnosis recovered one response, **11/22
diagnostic complete**; eight source mismatches, two schema failures and one
ambiguous location remain. Strict receipts are unchanged. Audit:
`legacy_semantic_receipt_audits/receipt_scope2_12632006_001/` (manifest
`9295833b40cb5d65e1029e3d5ad43fb7b82a780b7ba227458202075dbe5b4cce`).
Nine new invented tests, independent offset/hash/no-overwrite checks pass;
full suite **1,123 tests, 10.511 seconds**. No new model or Slurm calls.

Next: [source-span selection interface](../docs/report_span_selection_plan.md)
to avoid rewriting source quotes; the verifier is not yet qualified for
primary ranking or targeted repair.

### Source-span selection V1: completed, output-hierarchy failure diagnosed

`interfaces/report_span_selection.py` and `tools/verify_report_span_selection.py`
add a separate versioned interface selecting existing source-location IDs,
preserving full original context and offsets. Invalid/missing IDs or keys,
duplicate same-finding assignments and truncated responses fail closed.
Unknown remains unknown. This does not verify that a selected span supports
the medical interpretation; valid references alone are not clinical truth.

The same 24 slots / 22 text hashes stage successfully: **22 ready requests,
zero refusals, 3–14 spans per text**, using existing CPU allocation 12632006.
Protected plan: `report_span_plans/span_scope2_12632006_001/`; manifest SHA256
`bc650307d1713f3f572cf3593a8965d7ccb67c2f702a3bba273e4e7112dc93ea`.
All **1,155 tests pass** (10.658 seconds), including 32 new invented/mock tests;
187 source entries, request reconstruction, offsets, permissions and overwrite
refusal were checked. No old report, EHR, score or winner was changed.

After complete script/resource display and explicit approval, **job 12639717
completed on A100 in 1m20s**, exit 0:0, 22 calls, no retries. Approved script:
`slurm/40_report_span_selection_flexible_gpu.sbatch`, 1 compatible GPU, 2 CPUs,
32G RAM, 20-minute cap; same frozen Qwen weights/loader, at most 22 greedy calls,
512 output tokens, no retries. Completed receipts/comparison:
`verification_runs/span_scope2_12639717/` and
`verification_runs/span_scope2_12639717_analysis/`. Every further submission
requires complete script/resources and explicit approval. Results remain secondary development diagnostics, not independent
votes, new clinical scores, improved factuality or authorized repairs.

Strict V1 availability is **2/22**, worse than the exact-quote baseline's 10/22.
Twenty responses use finding-to-array mappings, losing the required polarity
object; all fail `polarity_inventory_mismatch`, none hit the token cap. The
V1 format instruction's ambiguous use of “each value” is an interface defect.
Do not infer polarity from other scorers, fill missing fields or reinterpret
these flat responses as successes. Across 96 candidate/head checks there are
two positive/six negative/88 unknown states. Five explicit comparisons agree,
zero oppose, 91 are not comparable; lower opposition is lost coverage, not
clinical improvement. Old scores and winners remain unchanged.

V1 receipt manifest:
`6a0a56a76c04b2f60987cba98b8ef1b82bb999767dd3e98cb9f2ad345319b4d9`;
comparison manifest:
`fe9e44045d3bbd9889bfc39fcc7432c0a234f524cfa16dde899d4b64ed9a78f8`.
Independent 22-response reparse, 336-row counts, offsets/hashes/permissions passed.

### Span V2: completed; format availability restored, semantics not qualified

V2 adds an explicit finding-object/polarity-array instruction and complete
four-object/twelve-array JSON shape. **Only output-format guidance changes**;
the same 22 texts, inventories, decoder, clinical instructions, frozen weights
and inference settings are reused. No V1 file/result is changed. Empty template
outputs will be counted separately, not presented as clinical success.

Code: `interfaces/report_span_selection_v2.py`,
`tools/verify_report_span_selection_v2.py`; approved/submitted script:
`slurm/41_report_span_v2_flexible_gpu.sbatch`. Protected plan:
`report_span_v2_plans/span_v2_scope2_12645021_001/`, manifest SHA256
`576483c1b4f544b7fd99f5b6e7aca94d8c22c83694294fd41cd3ab9d1aa74afc`.
The same 24 slots stage as 22 ready requests; 189 sources and unchanged V1
artifacts were verified. All **1,166 tests** pass (8.670s), 11 new invented/mock
fixtures. Preparation made no new GPU/model/API call. After complete
script/resource display and explicit approval, **job 12645404 completed on
A100 in 1m56s**, exit 0:0, exactly 22 calls and no retries/token-cap failures.
Peak allocated VRAM: **15.744 GiB**, BF16.

Strict interface availability is **22/22**, versus V1's 2/22 and the exact-quote
baseline's 10/22. **19 responses select evidence; three remain all-unknown**.
Across 96 candidate/head checks there are 18 positive/40 negative/38 unknown
states; CheXbert/V2 has 41 same explicit states, one opposition and 54 not
comparable. These counts are not clinical accuracy or independent evaluator
votes. Both fixed EHRs still lack explicit cached EHR-edge reference findings.
No old score, artifact or winner changed; no generation/repair was performed.

Protected receipts/comparison:
`verification_runs/span_v2_scope2_12645404/` and
`verification_runs/span_v2_scope2_12645404_analysis/`.
Manifest hashes are respectively
`0521476dddf9e86a8e61732bb3872df08d68e11274388d4b4fb0b0b8cc59c332` and
`4defd00b806b3187bc9657521bfae22b6b63e55e7d0b86812e4846344e67121b`.
Independent 22-response reparse, exact reproduction of all three comparison
artifacts, 336-row counts, hashes and protected permissions passed.
Next assess evidence semantics on independent held-out evidence before
primary ranking/regeneration. Every further GPU submission still requires
complete script/resource display and explicit approval. This observed-cohort
format improvement is not held-out clinical qualification.

### Span V2 versus manual report labels — completed, limited agreement

The [manual-label protocol](../docs/official_report_span_protocol.md) reuses
the existing official-label audit: all 687 annotation entries are retained,
all 27 linked test reports are eligible, and the unchanged strict Impression
parser previously admitted 15. No case is selected by labels or scores.
The frozen V2 four-head extractor was compared with manual report states
and the existing CheXbert predictions on exact same input hashes. Failed
requests, omissions, polarity flips and all-unknown outputs remain explicit.
This is a small report-label diagnostic, not independent image truth, gold
span adjudication, an untouched final test or permission to repair/rerank.

Worker: `tools/verify_official_report_spans.py`; approved/submitted script:
`slurm/42_official_report_spans_flexible_gpu.sbatch`. Protected metadata-only
plan: `official_span_plans/official_span_v2_12645021_001/`, manifest SHA256
`c27454cf8b0eaf36222fef7617862ccd3774bdb4f92b6fe3f2dbf660bb3adc13`.
Preparation opened no original annotation/linkage rows or report texts and
made no new model/API/Slurm calls. Full suite: **1,182 passing tests, 8.692s**,
including 16 new invented/mock fixtures. Real text can be consumed only inside
the separately approved GPU job; no raw text/response, patient key, source
report path or reference row is exported. Old scores and winners are unchanged.

After complete script/resource display and explicit approval, job **12645961**
completed on A100 in **1m09s**, exit 0:0, at **2026-10-04 12:14:35–12:15:44
server time**, exactly 15 calls and no retries/token-cap failures. All 15 ready
reports pass the format/reference contract; one is all-unknown. Peak allocated
VRAM: **15.724 GiB**, BF16.

Of **26 annotated finding checks**, V2 matches **20 (76.92%)**: cardiomegaly
4/5, consolidation 2/2, effusion 8/11 and pneumothorax 6/8. The six nonmatches
include two hard polarity flips and four binary references predicted uncertain;
no annotated assertion is omitted. Five unmentioned-reference checks receive
determinate predictions; unknown is not negative and these are not independently
adjudicated clinical hallucinations. Paired CheXbert matches 26/26, but this small,
previously used sample with no uncertain references does not qualify either
model as a universal clinical judge. This tests report extraction, not generated
image/report factuality, independent evidence-span correctness or fault repair.

Protected output: `real_validation/official_span_v2_12645961/`; manifest
`b9274424e49301b8e41e147716a0e751cf3743fc5f1a032f91a4dc1b7baf4b5b`.
Independent 128-source/hash/permission checks, 687-record opaque inventory,
15-call accounting, evidence field allowlist/state derivation, aggregate metric
algebra and exact paired baseline statistics passed without opening original
reference rows or real text. Old V2/full-bank endpoint manifests remain intact.
V2 stays secondary; no primary-score, threshold, winner or regeneration changes.
Every further submission needs its own complete script/resources and approval.

### Report-check progress sidecar — completed without new model calls

The [metadata-only progress protocol](../docs/report_verifier_progress_protocol.md)
connects the completed four-head synthetic report checks to the existing evidence
requests, while keeping manual-reference extraction diagnostics separate from
synthetic model disagreements. It does not replace scores or add another Agent.
Worker: `tools/diagnose_report_verifier_progress.py`; full suite: **1,201 tests
passed**, including 19 new invented fixtures.

Protected run: `report_verifier_progress/progress_scope2_12645021_001/` beneath
`artifacts/protected/tricompose_v1_2/`. All 336 fact rows (24 slots / 22 text hashes)
and 2,072 request rows retain every original cell. Within 96 four-head checks:
41 explicit agreements, one polarity opposition, 16 single-extractor assertions,
38 both-unmentioned; 240 other finding rows remain outside scope. Unknown is not
negative, and agreement is not independently established clinical truth.

Eight existing report requests now have technical-review sidecars; two matching
requests are outside scope and 2,062 are uncovered by this check. **Zero requests
are clinically resolved.** Exact candidate/finding/report hashes are required;
report-only evidence does not fulfill image/report checks. Original execution
history, primary scores, winners and EHRs are unchanged. No new model/GPU/API or
Slurm call, raw real-data read, calibration or regeneration occurred.

Independent source/artifact, preserved-cell, matrix, join and permission audits
passed. Manifest SHA256:
`b8999d053b6f6adb82127915e8a9194cc4fa3d65ea503ca239cce21bbd861837`.
This is an observed-data development diagnostic, not preregistered validation;
the manual-reference 76.92% is never assigned as a candidate confidence.

### Frozen V2 authored-language diagnostic — completed, limited semantics

The [authored-language protocol](../docs/authored_report_span_protocol.md) reuses
all 56 existing invented texts / 80 designated four-state checks, including
uncertainty and missing-assessment contexts absent from the small manual-report
reference. This is a same-team language stress test, not independent clinician
gold, image factuality or untouched final testing. No cases/texts/expected states
are edited, and the V2 prompt, decoder and frozen Qwen checkpoint remain unchanged.

Worker: `tools/verify_authored_report_spans.py`. Preparation reads blinded inputs
only; inference writes and fsyncs predictions before reference/baseline parsing.
Only reference IDs/offsets/hashes and derived states are exported, not quotes or
model responses. Existing CheXbert predictions provide the same-input baseline.
Failures remain unavailable in all denominators; no score/winner/repair change.

Prepared protected plan: `authored_span_plans/authored_span_v2_12645021_001/`
under `artifacts/protected/tricompose_v1_2/`, manifest SHA256
`053f5378689e17e6dd2c4d03670cdd616ca9ecb0682bce9f90b4b5aaa96476a7`.
The exact plan rebuild, 180 source bindings, 56 input hashes and protected modes
passed; no reference key, baseline prediction, real patient input or model was
read during preparation. Full suite: **1,216 passing tests**, 15 new invented/mock
fixtures. Source syntax and batch-shell checks pass.

Proposed `slurm/43_authored_report_spans_flexible_gpu.sbatch`: gpu partition,
one A40/A100/L40S-compatible GPU (worker requires ≥24 GiB), two CPUs, 32 GiB host
RAM, 10-minute cap, at most 56 greedy model calls with no retries/downloads.
After the complete script/resources were shown and explicitly approved, the
unchanged script was submitted once as **job 12646689**, at
**2026-10-04 13:02:09 server time**. Initial pending priority/resources cleared;
it completed on **A40 `b05-09` in 4m20s**, exit **0:0**, at
**13:04:39–13:08:59 server time**, 56 calls, zero retries/token-cap failures,
peak **15.628 GiB**, BF16. All 180 sealed preparation bindings were rechecked
before submission. No alternate GPU/model job or parameter change occurred.

See the [completed language result](../docs/authored_report_span_result.md).
All 56 schema/source-ID contracts pass, but designated state matches are
**50/80 (62.50%)**, macro F1 **0.5930**, versus CheXbert **48/80**, **0.6009**.
Qwen has eight positive/negative flips and 18 determinate outputs on uncertain/
unknown references. Only **2/20 uncertain** references are recovered (CheXbert
8/20). These are authored language labels, not clinical image/report accuracy.
Do not use the 194/224 secondary vector matches (mostly unmentioned labels)
as a substitute for the 80 designated checks or activate automatic repair.

Protected result: `verification_runs/authored_span_v2_12646689/`; manifest
`2d104e65a350bdca9970d8e4b080e0b5a9c39b93a3572a4f3a30897e94fb82eb`.
Independent 184-source/four-artifact hashes, all prompt/span bindings, state
derivation, denominators, paired baseline/metric/Markdown replay and permissions
passed. Historical full-bank and progress manifests remain intact; no original
case, score, winner or frozen protocol was edited. Qwen remains secondary,
not a sole clinical judge or regeneration trigger.

### Cached report scope gate — completed, substantial abstention

The [scope-gate protocol](../docs/cached_report_scope_gate_protocol.md) and
[completed result](../docs/cached_report_scope_gate_result.md) apply the existing
unchanged literal-context guard to cached Qwen V2 and CheXbert assertions. No
rules, prompts, weights, labels or thresholds are tuned; no new model/GPU/API
call or Slurm submission occurs. This CPU-only follow-up used existing allocation
12645021. The gate had been developed on the authored challenge previously, so
these are post-hoc development results, not held-out clinical qualification.

All 80 designated authored checks remain in each extractor's denominator.
Qwen retains **24/80 (30.0%)**, with **24/24 conditional matches**; CheXbert
retains **30/80 (37.5%)**, with **30/30 conditional matches**. All 30 Qwen and
32 CheXbert raw errors are noncommitted on this fixed challenge. This is not
100% overall accuracy: Qwen also does not retain seven matching non-unknown
assertions, alongside 19 matching unknown checks that have no assertion to
release. CheXbert similarly loses four matching positive assertions; its other
14 matching noncommitted checks are unknown. Abstention is null, not a corrected
negative/unknown prediction, and repeated same-report agreement is not an
independent clinical vote.

For the fixed **24 synthetic slots / 22 report hashes**, all **336 original fact
rows** remain unchanged. The 96 supported four-head checks contain **17 scope
commits, 41 abstentions and 38 no-model-assertion decisions**; the other 240
finding rows remain outside scope. The 17 retained assertions occur in **9/24
slots** (16 negative, one positive), not 17 clinically verified findings. There
is no independent candidate truth or EHR-edge qualification here. No old score,
winner, EHR, generation or request execution history changes; automatic repair
remains disabled.

Worker: `tools/gate_cached_report_spans.py`. Full suite: **1,233 passing tests**
(17 new invented/mock fixtures). Protected output:
`verification_gates/scope_gate_cached_12645021_001/` beneath
`artifacts/protected/tricompose_v1_2/`; manifest SHA256
`86a3e37a212d06fb665e7797397dd7736a449f92e30bb19c322923cc3f76e66a`.
Independent checks passed for 209 source bindings, six artifact hashes, all
original 448 authored states / 336 candidate rows, unchanged-gate replay,
denominators, evidence field restrictions, report bytes and protected modes.
This gates report assertion availability only: it does not establish that a
report describes its CXR correctly or authorize modality-error localization.

### Full-bank report availability handoff — completed, not rescoring

The [availability protocol](../docs/report_gate_availability_protocol.md) and
[completed handoff](../docs/report_gate_availability_result.md) attach the
cached scope-gate results to the historical **80 EHR / 240 CXR / 960 triple**
bank. `tools/attach_report_gate_availability.py` preserves all **65,280 original
candidate-table cells**, **13,440 original fact rows** and **2,072 original
evidence requests**, appending only `reportgate_` fields. Existing scores,
profiles, labels, row ordering, EHRs, winners and request execution history
remain untouched. There is no new scalar clinical score or selection policy.

Exactly **24 candidate slots** have this report check; **936 remain not_checked**.
Their decision counts/retained states are null, not zero errors or new negatives.
Among all fact rows: 17 scope commits, 41 abstentions, 38 no-model assertions,
240 outside-scope rows and **13,104 unchecked rows**. The 17 retained assertions
occur in nine slots and still have no independent clinical validation.

Only report-assertion requests can receive a source-scope sidecar. Of the eight
previously technically reviewed in-scope report requests, **one has a retained
assertion and seven abstain**; two other requests remain outside scope and
2,062 are not checked by this sidecar. **Zero requests are clinically resolved**.
Exact candidate/finding/report identity and hashes are required; identical
text cannot expand the fixed candidate subset. Report-only checks never fulfill
image-finding, EHR-observability or image/report relation requests.

Completed protected run:
`report_verification_availability/reportgate_pool960_12645021_001/` under
`artifacts/protected/tricompose_v1_2/`. Start with `candidate_score_table.csv`
for original scores plus availability; detailed rows are in
`fact_verification_availability.jsonl` and `evidence_request_availability.jsonl`.
Manifest SHA256:
`efb507376ce8cd8b23b28a6b55d719c7725a66fb9395f79bc70b32e76121299a`.

The CPU-only handoff used existing allocation 12645021; no new model/GPU/API,
submission, report/image/EHR-body or reference-key access occurred. Full suite:
**1,254 passing tests**, including 21 new invented metadata tests. Independent
17-source/five-artifact hash, exact preserved-cell, gate-join, request-scope,
denominator, original-manifest and permission checks passed. This operational
handoff does not authorize error localization or automatic regeneration.

### Same-subset image-only verifier — approved and completed

The [image-only protocol](../docs/cached_image_findings_protocol.md) prepares
the SAME two fixed EHR cases / six existing CXR slots / 24 report slots.
Every historical RoentGen-v2/Sana/PixArt seed-zero image is retained, not chosen
by scores or report-gate coverage. `tools/verify_cached_image_findings.py`
reuses the unchanged modality-separated eight-finding image prompt, decoder,
inference helper and frozen Qwen checkpoint. Requests contain the current
image only: no EHR, report, old labels/scores or model/candidate names.

Protected prepared plan:
`image_only_plans/image_only_scope2_12645021_001/` under
`artifacts/protected/tricompose_v1_2/`; manifest SHA256
`5fbcd984de88e7f776f3ef52081a98d978d620848f15feb63cfc19b4bedbcdfe`.
Independent 147-source/hash checks, exact plan rebuild, all six input/model
file stats, unchanged parent manifests and private modes passed. Preparation
opened no image bytes/pixels, model weights, EHR/report bodies or reference
keys and made no model/GPU/API/submission call. Full suite: **1,279 tests pass**
(25 new invented/mock tests). Historical staging-header location and canonical
JSON-versus-file-byte request hashes required compatibility handling; original
data, prompts, extraction rules and runs were not changed.

Approved complete script: `slurm/44_cached_image_findings_flexible_gpu.sbatch`.
Request: gpu partition, one A40/A100/L40S-compatible GPU (worker requires
at least 24 GiB), two CPUs, 32 GiB RAM, ten-minute execution cap; **six calls,
zero retries**, 384 output tokens. Cache/temp/logs are protected and offline.
After the full script/resources were displayed and explicitly approved, the
unchanged script was submitted once as **job 12649136**, at
**2026-10-04 14:27:33 server time**. It started nine seconds later on
**A40 `b04-10`**, ran **14:27:42–14:28:46 (1m04s)** and completed exit **0:0**.
Queue wait is separate from the execution cap. Every further submission needs
its own complete-script/resource display and explicit approval.

On approved execution, image predictions are fsynced before report-gate
comparison metadata is parsed. The 336 original gate rows stay unchanged;
new sidecars distinguish unsupported heads, unavailable model responses,
uncertainty, withdrawn report assertions and explicit model agreement/opposition.
Unique image/finding counts prevent four correlated reports from multiplying
the image denominator. This developmental check will not establish image
truth, validate an EHR edge, change scores/winners or authorize regeneration.

See the [completed image-only result](../docs/cached_image_findings_result.md).
All **six** image responses pass the eight-state contract, with zero retries or
token-cap failures; peak allocated VRAM **15.642 GiB**. The 48 unique supported
image/finding checks have **25 explicit XRV/Qwen agreements, 21 explicit polarity
oppositions and two uncertain comparisons**. These are scorer disagreements,
not natural image errors or independent clinical accuracy. In particular, XRV
labels all six pneumonia checks negative while Qwen gives four positive and
two uncertain states. No post-result threshold or prompt adjustment is made.

The same 336 original report-gate rows are preserved. All **17 retained report
assertion occurrences** match Qwen image states; 175 supported-image-head rows
have no retained report assertion and 144 are outside the eight image heads.
The 17 occurrences represent only **10 distinct image/finding pairs in five
image slots**, not 17 independent cases. Nine of those report occurrences still
oppose XRV. Same-Qwen image/text agreement plus a literal rule is not independent
clinical truth and cannot resolve which modality or scorer is wrong.

Protected output: `verification_runs/image_only_scope2_12649136/`; manifest
`07aef0cf22a59ecf6441a07601afca3989c95dd5a5b608be5c4ee05a2bce17e1`.
Independent 156-source/four-artifact hashes, six source-image/result bindings,
336 unchanged gate rows, all relation/state denominators, token/call accounting,
old manifests and private permissions passed. No clinical request is resolved;
scores, winners and EHRs remain unchanged, automatic regeneration disabled.

### Image evidence attached to full-bank score and request tables

The [CPU-only availability handoff](../docs/image_verification_availability_protocol.md)
is [completed](../docs/image_verification_availability_result.md). Original
scores, all reportgate fields, findings, EHR anchors and request execution
histories remain intact; new `imageverify_` fields expose exact image-side
coverage, states, raw XRV/Qwen disagreement and retained report relations.
Image evidence is reused only by exact CXR candidate ID/hash. Report relations
add exact triple/finding/report identity/hash requirements, not text-hash scope
expansion. No calibrated score, winner change or regeneration is introduced.

The full pool still has **80 EHRs / 240 images / 960 triples**. **24** triple
slots have the six-image check; **936** remain unchecked. Unique supported
image/finding counts remain **25 agreement / 21 opposition / two uncertain**,
not independent report votes. **Four image-finding requests** and **20
image/report requests** receive sidecars: respectively 1 agreement / 2
oppositions / 1 outside scope, and 10 agreements / 7 report assertions not
retained / 3 outside scope. Those **24** logical requests are not 24 calls or
clinically resolved requests. EHR scope and report text-extraction requests are
not fulfilled by an image-only check; all clinical resolution flags stay false.

Protected output: `image_verification_availability/imagecheck_pool960_12645021_001/`
under `artifacts/protected/tricompose_v1_2/`. Start with `candidate_score_table.csv`;
per-finding/request sidecars and a deduplicated 84-row image/finding table are
alongside it. Manifest SHA256:
`d933d9ba37181e2eed31f3f92fd8831d514a313fc225c6a6bb6f4260ad08f1bf`.
Existing CPU allocation **12645021** only; zero new inference/API/GPU/submission
or source-body reads. **1,306 tests pass** (27 new metadata tests). Independent
23-source/six-artifact checks verify original 76,800 candidate cells, 13,440 fact
rows, 2,072 request rows, exact dependencies/relations, source manifests and
project-only permissions. This is diagnostic availability, not independent
clinical accuracy, error localization or validated repair.

### No-information image safety controls — approved and completed

The [separately frozen control protocol](../docs/image_abstention_control_protocol.md)
and [preparation receipt](../docs/image_abstention_control_preparation.md) test a
remaining basic failure mode: whether the existing image-only verifier abstains
on uniform black/white images instead of asserting disease presence or absence.
These are isolated verifier controls, never previous CXRs, generator inputs,
replacement patient images or accepted candidates. Original images are not
assumed clinical gold. Existing cross-case swap diagnostics are not repeated.

The unchanged six-image subset has **18 logical slots** (original, black, white
per source image). Exact normalized RGB pixels/dimensions deduplicate calls;
**at most 18 calls, zero retries**, using the original eight-head prompt,
384-token cap, frozen checkpoint and greedy seed zero. No EHR, report, scores,
IDs, intervention names or expected answer enters the model request. Original
predictions are fsynced before cached reference states are parsed for readout
stability. Failed decoding is unavailable, not a successful unknown response.

Protected plan: `image_control_plans/noinfo_scope2_12645021_001/` under
`artifacts/protected/tricompose_v1_2/`; manifest SHA256
`60dda9d2f15ace54c3e121fab3adba1285ed47a4b49ba3b3ec92fac945c7c217`.
CPU allocation **12645021**, no image bytes/pixels, full weights, source clinical
bodies, model calls or new submissions in preparation. **1,331 tests pass**
(25 new invented/mock tests); independent 154-source/one-artifact hashes,
exact plan rebuild, scope/budgets and private modes passed.

Approved complete script: `slurm/45_image_abstention_controls_flexible_gpu.sbatch`;
gpu partition, one A40/A100/L40S-compatible GPU (>=24 GiB), two CPUs, 32 GiB RAM,
ten-minute execution cap, offline protected runtime. After the full script was
displayed and explicitly approved, it was submitted once as **12650073** at
**2026-10-04 15:08:57 server time**, started at 15:09:11 on **A40 b04-10** and
completed at 15:10:05, **54 seconds**, exit **0:0**. Queue wait was 14 seconds.
See [the completed result](../docs/image_abstention_control_result.md).

The 18 logical slots deduplicate to **10 actual calls**: six originals, two
black and two white frames (512×512 / 1024×1024). All ten named-state contracts
complete, with zero retries/token-cap failures and **15.642 GiB** peak allocated
VRAM. Original readout states match the cached run **48/48**; that is stability,
not clinical correctness. Of the **32 unique uninformative-frame/finding slots**,
**28 unknown** and **four unsupported negatives** occur. All four negatives are
cardiomegaly, one per unique blank frame. There are no positive assertions, but
absence is also unsupported without anatomy; **12.5% explicit assertion rate**
on these controls prevents treating the verifier as perfectly abstention-safe.
Repeated same-size controls are not independent cases or extra model calls.

Protected run: `image_control_runs/noinfo_scope2_12650073/`; manifest SHA256
`87d1e211e21fe497bda8647ae356a5c5a61246801bd5a288a828263f4f908ecd`.
Independent **162-source/nine-artifact** hashes, unchanged prompts and original
states, all dedup/counts/rates, PNG hashes/dimensions, previous manifests and
private modes passed. No original score, winner, EHR, clinical eligibility or
repair permission changes. A separately frozen mechanical image-validity guard
is the next safety step, not installed by this diagnostic; non-uniform pixels
would still not establish valid anatomy or clinical correctness. Every further
GPU submission needs its own complete-script/resource display and approval.

### Mechanical image-validity guard — implemented, cached check completed

The [separately frozen guard](../docs/image_validity_guard_protocol.md) and
[completed result](../docs/image_validity_guard_result.md) add fail-closed PNG
checks before clinical scoring. `tools/image_validity_guard.py` exposes bounded
inspection, hash-bound receipts, cached guarded views and `guarded_invoke`.
The pre-call hook supplies the exact checked in-memory RGB image to a callback
only on basic pass; it does not grant inference/API authorization. Historical
sealed GPU workers have **not** been replaced, and no model callback was run.

In existing CPU allocation **12645021**, all **four** saved uniform black/white
controls are blocked; all **six** original synthetic images pass only the basic
check. The guarded view withholds the controls' **32** finding readouts,
including four unsupported negative assertions, while preserving every raw
verifier record and all **48** original-image states. Null means blocked or
unavailable, not an invented negative/all-unknown response. Exact uniformity,
bounded native L/RGB decoding and hash checks are mechanical rules: noise,
wrong anatomy or single-pixel variation can still pass. No clinical acceptance
or independent image truth is established.

The append-only 960-row table marks **24** exact image-bound triple slots
`basic_pass_not_clinical` and **936** `not_checked` with null permission. Those
24 slots reuse six images with four reports each, not 24 independent images;
the four controls never enter the patient candidate bank. All **94,080 original
CSV cells** and order remain unchanged, as do scores, winners, EHRs and request
histories. No regeneration is authorized, no clinical request resolved and no
prospective compute savings claimed.

Protected output: `image_validity_guards/pngguard_scope2_12645021_001/` under
`artifacts/protected/tricompose_v1_2/`; start with `candidate_score_table.csv`.
Manifest SHA256:
`0c9654d2147b7268890527c5db5f85f0c455217f501f17484ddf23365983ccba`.
**1,362 tests pass** (31 new invented/mock tests). Independent **177 source /
four decoder / six artifact** hash checks, ten native-image/pixel bindings,
unchanged verifier records and CSV cells, exact joins, parent manifests and
project-only permissions passed. Zero new model/GPU/API/download/submission
calls; no real target or EHR/report body was opened. A future newly approved
scoring worker can use this hook; the current result is an engineering safety
handoff, not a clinical metric improvement or pipeline-wide replacement.

### Prospective guarded image worker — approved execution completed

`tools/verify_guarded_image_findings.py` now connects the sealed mechanical
guard to the unchanged frozen image-only verifier. See the [new protocol](../docs/guarded_image_findings_protocol.md)
and [preparation receipt](../docs/guarded_image_findings_preparation.md).
Unlike the completed cached guard view, this new worker checks each input
before a lazy model callback; blocked inputs have null states and no call.
Requests contain only the exact checked pixels and unchanged image prompt,
never EHRs, reports, IDs, scores, arm names or expected answers. No historical
worker, prompt, checkpoint, score, winner or EHR was modified.

Fixed protected plan: `guarded_image_plans/guardhook_scope2_12645021_001/`,
manifest `2864941d29f60e890cdd51e88a7126986027b495345d4cf8a0ff23bdc2b77959`.
Ten unique existing inputs / 18 logical slots, six originals and four controls;
at most ten calls, no retries. Native guard decisions are recomputed on execution,
not copied from old results. Predictions are fsynced before cached baseline
states are parsed. Mechanical controls never enter the patient candidate bank.

CPU preparation opened metadata only: no pixels, weights, EHR/report bodies or
real targets. **1,385 tests pass** (23 new invented/mock tests); independent
**159 source / four decoder / one artifact** checks, exact inventory/budgets,
source metadata, unchanged parents and project-only permissions passed.
CPU preparation made zero new model/API/submission calls. The subsequent
approved execution is now [completed](../docs/guarded_image_findings_result.md).

Approved complete script: `slurm/46_guarded_image_findings_flexible_gpu.sbatch`;
gpu partition, one A40/A100/L40S-compatible GPU (>=24 GiB), two CPUs, 32 GiB RAM,
ten-minute execution cap, no hard node binding, protected offline runtime.
After the complete script/resources were displayed and explicitly approved,
the unchanged script was submitted once as **12651080**, at **2026-10-04
16:12:21 server time**. It started at **16:13:54**, completed **16:15:00**, exit
**0:0**, on **A100 40GB b01-20**: queue wait **1m33s**, allocation **1m06s**.
No replacement submission or resource change. Every further submission needs
its own full-script/resource display and explicit approval.

All four uniform frames are blocked before any model call, retaining null
states and zero calls. Six original images each receive one complete response:
**six actual calls**, zero retries/token-cap failures, peak allocated VRAM
**15.642 GiB**. Recorded worker time is **29.0984s excluding preflight**, not
full job time. Original states match the unguarded run **46/48**; one
consolidation and one pneumonia slot change **positive -> uncertain**. These
withdraw certainty, not reverse polarity; do not turn them into hard negatives,
confirmed errors or regeneration permission. Their cause is not established,
and repeatability is not clinical accuracy.

Protected output: `guarded_image_runs/guardhook_scope2_12651080/`; manifest
`850f90f5b4624bc15d26aedcdc50585238805a1cd3c28c3b276055ba3d046149`.
Independent **172 source / four decoder / five artifact** checks, exact scope,
guard/call/null contracts, token/budget accounting, state-transition arithmetic,
unchanged sources and private modes passed. Four fewer calls versus the old
ten-call run apply only to these mechanical controls. Different GPUs and
54s/66s allocations do not establish a runtime speedup. No independent clinical
correctness, error localization, repair success or cohort-level compute savings
is established. No candidate scores/rankings are rewritten or fresh states
attached to the full bank by this run; historical workers remain sealed.

### Guarded readouts attached to full-bank score table — CPU handoff completed

The [append-only handoff](../docs/guarded_image_availability_protocol.md) is
[completed](../docs/guarded_image_availability_result.md) in CPU allocation
**12645021**. `tools/attach_guarded_image_availability.py` adds `liveimage_`
fields to the existing guard-annotated table: **960 rows / 139 columns**,
105 unchanged columns plus 34 new fields. Old scores/states, EHRs, winners and
request histories are untouched. The fresh guarded GPU run is reused, not
rerun; no new model/API/GPU/download/training/submission or source-body access.

Only **six images / 24 triple slots** have fresh readouts; **234 images / 936
triples** remain unchecked with null states/counts, not zero errors. Exact CXR
candidate ID/hash and case/EHR/facts lineage are required. Four uniform controls
never join patient candidates. The 13,440-row finding inventory is preserved;
the deduplicated 84-row image/finding sidecar has **48** supported head slots
and **36** outside scope. Repeated report uses are not independent image calls.

The **two** positive-to-uncertain readout changes belong to **one image**,
affecting **four triples / eight finding occurrences**. Unique matches are
**46/48**, not clinical accuracy. Fresh XRV/Qwen relations are **25 explicit
agreements / 19 explicit oppositions / four uncertain**, compared with old
25/21/two. The reduction in explicit opposition is abstention, not a repaired
image or proven consistency gain. All 17 retained report assertion occurrences
still agree with the fresh states, but same-Qwen image/text evidence is not
independent truth. Clinical scores/eligibility and repair authorization stay
unqualified/false; no request is clinically resolved or candidate reranked.

Protected output: `guarded_image_availability/liveimage_pool960_12645021_001/`
under `artifacts/protected/tricompose_v1_2/`; start with `candidate_score_table.csv`.
Manifest `0120146adc043b482f51ef515d70ad3dd1b78132c3b989b22e7bd918b3c68dcb`.
**1,412 tests pass** (27 new invented-metadata tests). Independent **27 source /
five artifact** hashes, exact joins, state/null/uncertainty semantics, all new
relations, **100,800 original CSV cells / 551,040 original fact cells**, row
order and project-only permissions passed. This is evidence availability,
not independent clinical evaluation, error localization or validated repair.

### Deterministic verification frontier — planning-only handoff completed

The [frontier protocol](../docs/verification_frontier_protocol.md) and
[validated result](../docs/verification_frontier_result.md) connect the fresh
image readouts to existing next-evidence requests without running a model.
`tools/build_verification_frontier.py` preserves all **960 candidate rows /
139 original columns** and **2,072 original requests**, then appends action
availability. Missing initial checks are surfaced even where the old heuristic
had no corresponding request. Priorities are engineering investigation tiers,
not clinical scores, error probability, optimal compute allocation, or winners.

There are **21 supplemental independent-evidence requests**: two changed
readouts and 19 explicit XRV/Qwen oppositions, deduplicated across **six images**
and **84 candidate/finding links**. The two changed findings share one image;
four reports do not supply four independent image votes. Candidate-level next
requirements are **four unstable / 20 scorer-disagreement / 936 initial-check**
slots, not four/20/936 confirmed errors. For 159 slots the next requirement has
no old request ID; an empty ID list is not a pass or execution permission.
Unknown/uncertain are not negatives, and stable agreement is not clinical truth.

Start with `candidate_action_table.csv` in the validated protected run:
`artifacts/protected/tricompose_v1_2/verification_frontiers/frontier_pool960_12645021_validated/`.
Manifest `204b499d2eeab6cd7f849ca854a8beefbc433a53876bb03b507a3aa56e0493bc`.
**1,445 tests pass**, including 33 frontier tests. Independent **22 source /
five artifact** hashes, **133,440 original candidate cells / 68,376 original
request fields**, dependency joins, unique requests and private modes passed.
The preliminary write and its byte-exact pre-correction test fixture remain
documented separately in the result note; use the validated run above.

No training, model/API/GPU/download/new submission, source-body read, selection
change, EHR replacement, clinical resolution or regeneration. Clinical scores,
cost estimates and an execution budget remain null; all clinical/execution
eligibility flags are false. Next obtain suitable independent evidence, not
same-scorer retries or automatic repair to satisfy an unqualified proxy.

### Alternate XraySigLIP scorer — completed, diagnostic only

The [fixed probe protocol](../docs/xraysiglip_probe_protocol.md) and
[preparation handoff](../docs/xraysiglip_probe_preparation.md) add a real existing-
model scoring backend for the six unchanged images behind the 21 supplemental
requests. `tools/probe_xraysiglip_findings.py` uses the native frozen SiglipModel
and processor, eight heads × three fixed presence/absence pairs, max-length
padding and raw cosine/logit readouts. The checkpoint's config specifies
**512×512 / float32 / 64 text tokens**, not the resolution in its repo name.
No templates, thresholds, disease facts or selected patients are fitted.

Plan: `artifacts/protected/tricompose_v1_2/xraysiglip_plans/siglip_scope2_12645021_001/`.
Manifest `fbf4d8bd4e5bf93078533ef90f4e65a2d861a53cc2296269fd31570898be3f89`.
**1,474 tests pass**, with 29 new probe tests; 45 source/config/runtime hashes,
exact references, metadata-only preparation and project modes pass. No image/
weight bytes or EHR/report bodies were opened by preparation, and zero models
or new jobs ran during preparation. The subsequently approved GPU run is
documented below; preparation and execution are separate stages.

The complete batch `slurm/47_xraysiglip_findings_flexible_gpu.sbatch` requests
one P100/V100/A40/A100/L40S-compatible GPU, two CPUs, 16 GiB RAM and a ten-minute
cap on debug/gpu, no hard node binding. This broadens resource eligibility
without changing precision/prompts or submitting unseen work. The complete
script/resources were displayed and explicitly approved before job **12654030**.
It completed with exit code 0 on debug P100 `e23-02` in **36 seconds**;
worker time including preflight was 20.532 seconds and peak allocated VRAM
was **2.603 GiB**. All six forwards succeeded, one model load, zero retries;
all four isolated uniform controls were blocked before any callback.

Protected run:
`artifacts/protected/tricompose_v1_2/xraysiglip_runs/siglip_scope2_12654030/`.
Manifest `d846602fabba7abca85745e18fd5daba00832ac251ce458bd9f53ff3fa3c54eb`.
See the [aggregate result](../docs/xraysiglip_probe_result.md). Of the 21 disputed
finding requests, **14 have template-sensitive polarity**; seven consistently
prefer absence text. Among those seven the preference has the same direction
as current Qwen in four and raw XRV in three. These are unqualified comparisons,
not correct predictions. Across all 48 image/finding combinations, 25 are
template-sensitive, six prefer presence and 17 prefer absence. No template was
selected or threshold fitted after these results. Output hashes, 48 bounded
source/config/runtime hashes, exact request bindings, attempt accounting and
project modes passed the post-run metadata audit; image and weight bytes were
not reopened by that audit.

XraySigLIP is distinct from Qwen/XRV but is CheXagent-2's shared visual encoder.
Do not call it independent clinical truth or final independent evaluation of
that generator. Template preferences are not clinical finding labels or
calibrated probabilities. All 21 clinical requests remain unresolved, and no
EHR, historical score, winner, generation or regeneration policy changes.

### Full-bank XraySigLIP availability — completed, append-only

The [handoff protocol](../docs/xraysiglip_availability_protocol.md) and
[aggregate result](../docs/xraysiglip_availability_result.md) attach the frozen
alternate readouts to all **960 candidate rows**, with **162 old columns
preserved and 14 `siglip_` columns appended**. Worker:
`tools/attach_xraysiglip_availability.py`. The current protected table is:

```text
artifacts/protected/tricompose_v1_2/xraysiglip_availability/siglip_pool960_12645021_001/
  candidate_score_table.csv
  fact_verification_availability.jsonl
  unique_image_finding_table.jsonl
  supplemental_image_evidence_requests.jsonl
  evidence_request_frontier.jsonl
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest `764a114166a817c06c9a36d4b50265db35764da7d324b2f68407e5bf04112a0e`.
The six checked images' **24 exact original report consumers** receive readout
availability; **936 candidates remain not_checked**, with null numeric counts.
All 13,440 original finding rows remain, including unsupported and unchecked
heads. Raw three-template margins/preferences live in the finding sidecar;
they are not clinical labels/probabilities or new cross-modal edge scores.
The 21 supplemental requests retain their old execution histories and remain
clinically unresolved. Original 2,072-request history is copied byte-for-byte.

**1,494 tests pass** (20 new attachment tests). All 33 consumed source hashes,
seven artifacts, 155,520 candidate cells, 752,640 finding fields, 672 supplemental
request fields, project modes and deterministic byte-equal metadata replay
passed. Existing CPU allocation 12645021 only; zero new inference, GPU/Slurm
submission, weight/image/EHR/report body reads or training. No EHR replacement,
priority/action update, new winner, clinical acceptance or regeneration.

### Public same-cohort XraySigLIP diagnostic — completed, exploratory only

The [fixed protocol](../docs/rsua_xraysiglip_protocol.md) and
[preparation note](../docs/rsua_xraysiglip_preparation.md) define the next
automatic-readout diagnostic on the **same existing RSUA 50 images**, with
50 primary forwards plus two fixed technical repeats, zero retries. It reuses
all eight heads / three template families and evaluates only pneumonia against
the published cohort-class proxy. No case selection, template optimization,
threshold fitting, new labels, downloads or candidate/winner changes.

Worker `real_validation/rsua_xraysiglip.py`; **1,518 tests pass**, including 24
new invented-vector and safety tests. The sealed metadata-only plan is
`artifacts/protected/tricompose_v1_2/real_validation/rsua_siglip_plans/plan50_12645021_001/`,
manifest `ca2fb6e62237b0c66cff988425eeaf3d51e3d09b029e5ddf002f34af0a90037f`.
All 47 source hashes, opaque input ordering, 50 image-file stats/permissions
and deterministic plan replay pass; no pixels/weight bytes were read.

Prepared complete batch `slurm/48_rsua_xraysiglip50_flexible_gpu.sbatch` requests
one compatible flexible GPU, two CPUs, 8 GiB RAM and five minutes on debug/gpu.
The full script/resources were displayed and explicitly approved before job
**12654662**. It completed in **44 seconds** on debug P100 `e23-02`, exit 0;
worker runtime 33.173 seconds and peak allocated VRAM 2.603 GiB. All 50 primary
readouts and two fixed technical repeats completed, 52 actual forwards, one
load, zero retries; both repeat endpoint differences were zero.
RSUA's control group is a paper-described normal proxy, not independently
adjudicated per-finding clinical truth. Any result is developmental, with
training overlap/age/patient grouping unverified and clinical repair disabled.

Protected result:
`artifacts/protected/tricompose_v1_2/real_validation/rsua_siglip_pilots/rsua_siglip50_12654662/`,
manifest `3c7cea5b9da2f2e05963af44f40b522c7faccbda55f06b16ba9783c0dbae6d6a`.
See the [aggregate result](../docs/rsua_xraysiglip_result.md). Fixed mean-margin
AUROC/AP are **0.7552 / 0.71671861**, positive-reference wins **8/25**, negative
wins **21/25**, balanced wins **0.58**. All three template outcomes are reported;
**48/50 images change polarity direction across templates**, only 2/50 have a
stable direction. This supports ranking signal on the published cohort proxy,
not a reliable clinical presence/absence arbiter or an automatic repair gate.
No new operating point, template or winner was chosen to improve these results.

Independent metadata audit checked 54 bounded source/reference hashes, five
artifacts, 104 journal events, exact case/image/guard bindings, protected modes
and metric/repeat/comparison replay; 51 image/weight hashes remain approved-GPU
attestations without reopening pixels/weights. Original candidate tables,
scores, fixed EHRs, thresholds and winners remain unchanged.

### Frozen evaluator scorecard — cached comparison completed

The [post-hoc analysis protocol](../docs/frozen_evaluator_scorecard_protocol.md)
and [aggregate result](../docs/frozen_evaluator_scorecard_result.md) consolidate
XRV, BioViL-T and XraySigLIP on the same fixed 50 RSUA reference images.
Existing CPU allocation **12645021**, zero new model/GPU/API/submission or
image/weight/EHR/report-body reads; construction before commit 0.297866 seconds.
All original template/operating-point statistics replay exactly. This remains
already inspected DEVELOPMENT with cohort-class proxies, not clinical truth.

The fixed three-template SigLIP AUROC is 0.7552 versus XRV 0.7248, but their
paired difference **+0.0304** has a descriptive image-level 95% interval
**[−0.1105, +0.1648]**. It does not justify a superiority claim or a new repair
judge. Exactly 2,000 class-stratified paired samples use seed 0. All fixed
comparisons/AP intervals and ten template/profile rows are retained, without
best-template choice, threshold fitting or natural-prevalence claims.
Stable template coverage is BioViL-T 36/50 and SigLIP 2/50; conditional wins
are not full-cohort clinical accuracy. Patient independence/training overlap
remain unverified. Global retrieval and finding negation are different tasks.

Protected output:
`artifacts/protected/tricompose_v1_2/real_validation/evaluator_scorecards/scorecard50_12645021_001/`,
manifest `91a2899919a14cb03e6f43ac0bc58f37bb0290e553628831b80da689ec2833f1`.
`scorecard.csv`, `bootstrap_comparison.json` and `score_usage_contract.json`
retain ranking, polarity, coverage, dependency and unavailable qualifications
separately. The usage contract is **documentation, not a deployed controller**;
existing edge scores, actions, thresholds and selected triples stay unchanged.

**1,544 tests pass** (26 new invented-fixture tests). The post-run audit verifies
16 bounded sources, five artifacts, ten exact CSV rows, aggregate replay and
private modes. A separate Mann–Whitney/tied-AP implementation reproduces all
12 model/paired-contrast interval checks. This is numeric verification, not
clinical adjudication. No new inference is authorized by this result.

### Score-free final-choice baseline — cached comparison completed

The previously missing **true score-free final-choice** baseline now has a
separate [development protocol](../docs/score_free_random_control_protocol.md)
and [aggregate result](../docs/score_free_random_control_result.md). It copies
the old `random` acquisition sets and simulated costs, then chooses uniformly
using only observed IDs and the same artifact gate. Old `random` remains
random acquisition **plus scored final reranking**; it is not renamed.

All 80 fixed EHRs/960 triples, five caps and all five acquisition/final seed
settings remain. **10,000 control trials are seed replicates, not new patients
or model calls**; average within each EHR before reporting. Only the old-random
versus new score-free contrast shares exactly equal acquisition/expenditure.
This selector-only ablation intentionally retains scorer charges and is not
a cost-optimal random pipeline, actual regeneration or measured GPU savings.

At full cap 30, scored final BioViL mean is **0.6122**, score-free final
**0.5082**, paired difference **+0.1041** (63 EHRs higher, 17 lower). Raw
CXR–Report support/known is 0.2365 versus 0.1551, proxy opposition/known
0 versus 0.1035, but coverage is only 0.2365 versus 0.2586. Zero opposition
is not complete consistency or clinical correctness. At caps 8/12 the
alternate mean instead favors score-free final; every budget is retained.
Static and old-random retain identical full-inventory winners. Targeted's
full-cap mean **0.6075** still does not exceed static, while the fixed path
uses four simulated calls with mean **0.6186**. This supports selection utility,
not a demonstrated clinical/localization/targeted-repair advantage.

Protected result:
`artifacts/protected/tricompose_v1_2/automatic_replays/score_free_random_12654973_001/`,
manifest `771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124`.
Existing CPU allocation **12654973**, 1.078405 seconds before serialization;
zero model/GPU/API/new submission, EHR/report/image-body reads or training.
No EHR enrichment/drop, threshold/policy update or old winner change.

**1,563 tests pass** (19 new fixtures). Audit verifies ten source hashes, six
artifacts, all 10,000 score-blind choices/equal-cost receipts, 2,000 case means,
75 method/subgroup rows, 60 paired contrasts, old endpoints and private modes.
Cached direct-EHR availability remains 8/80; the 72 missing direct-fact EHRs
stay with NA edges. Freeze any next policy separately; further GPU execution
still needs full-script/resources and explicit approval.

### Scope-guarded stopping — cached action-veto comparison completed

The [separate frozen protocol](../docs/scope_guarded_stopping_protocol.md)
and [aggregate result](../docs/scope_guarded_stopping_result.md) test a single
change to the original targeted heuristic: without cached basic-invalid-image
metadata or an explicit directly evidenced EHR–XRV opposition, try remaining
same-image reports and then stop **unresolved**, rather than explore a new
image. Unknown/uncertain evidence and correlated report votes never qualify.
Ranking, initial model/order, EHRs and old scores/winners are unchanged; no
new numerical threshold or BioViL-driven policy tuning. Allowed image branches
are proxy cache exploration, not confirmed faults or clinical repair.

All 80 fixed EHRs/960 triples and five caps remain. At cap 30, guarded mean
simulated calls/BioViL are **10.050 / 0.6397**, versus all-image static
**30.000 / 0.6122** and original targeted **28.825 / 0.6075**. However, raw
CXR–Report support/coverage are lower than all-image static, and at cap 20
static's alternate BioViL mean is higher. This is not across-metric dominance,
clinical correctness or measured GPU savings.

The essential **fixed-image report-only static** control is **10.000 / 0.6388**:
**79/80 final triples are identical**, with a guarded BioViL delta of only
**+0.0009179** and 0.05 additional simulated calls. No material adaptive
advantage over that simple control has been demonstrated. At full cap, 72
outcomes lack a direct EHR–image reference, three lack comparable image
evidence and five lack direct opposition; **all 80 remain unresolved**.
79 cases explore one image and one explores two. Direct EHR-edge availability
remains 8/80; missing edges stay NA, without replacing or enriching EHRs.

Protected output:
`artifacts/protected/tricompose_v1_2/automatic_replays/scope_guarded_stop_12654973_001/`,
manifest `3d4553641af5408d639e857b3f20d830761458c1c08ceff9064dc6cb3d549360`.
Existing CPU allocation **12654973**, 2.194871 seconds before serialization;
400 new cache trials, zero new model/GPU/API/submission, body/pixel/weight
reads or training. **1,587 tests pass** (24 new fixtures). Audit validates 25
source hashes, seven artifacts, exact replay, lineage/action bases, private
modes and all 2,800 case means / 105 comparison rows / 90 paired contrasts,
with independent three-edge and within-EHR arithmetic checks. Sealed code,
protocols, tests and protected output remain immutable.

### Cross-modal repair feasibility — finite-cache diagnostic completed

The [frozen protocol](../docs/cross_modal_repair_headroom_protocol.md) and
[aggregate result](../docs/cross_modal_repair_headroom_result.md) check whether
an alternative adds explicit proxy evidence without losing old supported or
comparable finding IDs, introducing new opposition or lowering cached report
structure. Unknown/uncertain cannot silence a conflict and earn repair credit.
Same-image report opportunities reuse the existing immutable headroom predicate;
cross-image alternatives must strictly improve the fixed-EHR image edge and
pass preservation against **both** fixed and report-only static references.
Baseline labels are conservative proxy constraints, not clinical truth.

All 80 fixed EHRs/960 triples remain: **880 comparisons = 240 report + 640
cross-image alternatives**. **13/80 EHRs** have **21** qualifying report
alternatives; **0/640** cross-image alternatives pass both references. This
is a finite-cache/predicate result, not proof that new seeds cannot help.
The previous guarded selections remain unchanged: 63 unchanged, 16 report
changes and one image change. **13 of the 17 changed outputs pass**, four do
not. Three non-passing changes nevertheless have higher BioViL; of the 13
passing changes, five have higher BioViL and eight lower. Global similarity
alone cannot identify a fact-preserving repair or establish clinical truth.

The label plan is hashed before endpoint attachment. Average all 21 qualifying
report gaps within EHR, then across 13 opportunity-bearing EHRs: conditional
BioViL delta **+0.0507731**, not a selected-output or full-80-case effect.
Cross-image conditional benefit remains NA. No alternative is selected by
BioViL, no new winner is exported, and no existing score/threshold/policy or
EHR is modified. Direct-EHR coverage remains 8/80; missing edges stay NA.

Protected output:
`artifacts/protected/tricompose_v1_2/repair_headroom/cross_modal_headroom_12654973_001/`,
manifest `9f179fa5597a88acd9aa66f5a2d77078ed68bce781468f9ef812cd98837382a3`.
Worker `tools/audit_cross_modal_repair_headroom.py`; existing CPU allocation
12654973, 2.280905 seconds before final serialization, zero new model/GPU/API,
submission, training or body/pixel/weight reads. **1,614 tests pass** (27 new
fixtures); audit checks 38 source hashes, eight artifacts, exact replay,
complete denominators, private modes and **1,520 independently recomputed
raw-state set gates**. The next output acceptance layer is proposed, not
installed here; clinical fault localization/repair remain unvalidated.

### Observed-only output acceptance — cached final-selection integration completed

The [separate protocol](../docs/output_acceptance_protocol.md) and
[aggregate result](../docs/output_acceptance_result.md) implement the reusable
`tools/apply_output_acceptance.py::assess_output(baseline, proposed, observed_candidates)`
gate. It strips secondary/winner/rank fields, preserves fixed EHR and shared
artifact evidence, and uses only actually observed candidates. Image changes
must preserve the initial reference and best eligible **observed** initial-image
report; an unseen full-bank static winner is not a free reference. Unknown/
uncertain cannot silence a conflict or earn repair credit. Passing is proxy-
preserving change, not clinical acceptance or new inference authorization.

The same wrapper is applied to all five caps for scope-guarded and report-only
static parents: **800 decisions**, with parent acquisition, stop histories
and already incurred charges unchanged, including vetoes. At full cap, gated
scope retains 13 report changes, leaves 63 unchanged and vetoes four proposals
back to the fixed reference. Gated report-only retains the same 13 changes,
leaves 64 unchanged and vetoes three. **Both choose identical outputs on
80/80 EHRs at every cap**; there is no demonstrated adaptive advantage.

Full-cap gated mean calls/BioViL are **10.050 / 0.6280** for scope and
**10.000 / 0.6280** for report-only. Gating lowers the original scope mean
from 0.6397, with support/coverage decreasing to 0.1302/0.1344. Report the
constraint trade-off, not an across-metric improvement. All 80 results remain
clinically unqualified, eight EHR-edge denominators remain available and 72
remain NA. No EHR is enriched/replaced, no score threshold fitted, no new
image/report generated and no old winner overwritten.

Protected output:
`artifacts/protected/tricompose_v1_2/output_acceptance/output_gate_12654973_001/`,
manifest `211912fa0cf4e38f5eaf146f57050985b7546497639933be7e6d66ec2ae9d545`.
Selection decisions are hashed before full-bank endpoint/readout attachment;
selected-reference metadata points to old generated artifacts without copying
their bodies. Existing CPU allocation 12654973, 3.864852 seconds before final
serialization; zero new model/GPU/API/submission, training or body/pixel/weight
reads. **1,646 tests pass** (32 new fixtures). Audit validates 42 source hashes,
nine artifacts, all 800 decisions/cost receipts/reference bindings, independent
raw-state gate arithmetic, private modes and all 3,600 case means / 135 method
rows / 75 paired contrasts. The gate is integrated into **cached replay only**;
live fresh-eight-head receipts still need a separate provenance-correct adapter
and approved prospective test. Clinical localization/repair stay unvalidated.

### Fresh output-receipt adapter — metadata interface completed, live controller unchanged

The [fresh protocol](../docs/fresh_output_acceptance_protocol.md) and
[aggregate result](../docs/fresh_output_acceptance_result.md) add
`tools/fresh_output_acceptance.py::assess_fresh_output`. It consumes fixed-EHR
provenance, the separate eight-enabled-head scorer context, explicit observed
metadata and a complete durable cost ledger. It does not relabel fresh receipts
as the legacy fourteen-head profile. Disabled heads remain unknown and missing
EHR edges remain NA. Same-artifact states, model/seed/dependency hashes and
completed receipt lineage are checked; alternate endpoints/winners are stripped.

Report/image proposals use their existing frozen, distinct fact-preservation
gates. Cross-image comparisons include the initial reference and an eligible
already observed report-only reference, never an unseen bank winner. Failed
proposals fall back to the initial section-eligible reference, not a new search;
this does not modify historical static fallbacks. Failure/pending/veto costs
remain charged. A self-digest or cached-origin marker is not source trust:
upstream artifact/model authentication remains mandatory.

The archived two-case retry compatibility smoke makes **four independent
checks**: one existing report change passes, one remains unchanged, and both
existing image proposals are vetoed. Original **eight attempts** are retained
and counted once, not multiplied by the number of decision checks. This is
already-inspected DEVELOPMENT metadata, not new generation, sequential repair,
clinical efficacy, another independent sample or prospective compute saving.

Protected output:
`artifacts/protected/tricompose_v1_2/fresh_output_acceptance/fresh_receipts_12654973_001/`,
manifest `ecd4f197ccc2df8bee750203e6a49b787508b24513f1d14a1ec74734e98bb263`.
CPU allocation 12654973, 0.155397 seconds before serialization, zero model/GPU/
API/submission, training, body/pixel/weight reads. **1,683 tests pass** (37 new
invented-fixture tests). Audit checks 262 source hashes, three artifacts, all
four exact decisions, independent fact-set gates, eight retained reservations,
ledger chains, private modes, overwrite refusal and unchanged preceding runs.
The reusable fresh interface is complete; it is **not installed into the live
controller**, not model-execution authorization and not clinical acceptance.
Further prospective execution needs a separately frozen policy/control and a
complete actual batch script/resources followed by explicit approval.

### Prospective online report escalation — prepared, GPU not submitted

The [new frozen protocol](../docs/online_report_smoke_protocol.md) and
[preparation note](../docs/online_report_smoke_preparation.md) define the first
new chronological report-only smoke. New tool `tools/online_report_smoke.py`
reuses frozen historical workers and the separate fresh-eight-head acceptance
interface without changing them. **1,707 tests pass**, including 24 new
mock/invented-worker tests; these are software tests, not clinical efficacy.

Keep the same two already-inspected EHR-only-stratified DEVELOPMENT anchors
and all clinical prompts unchanged. Each gets one new RoentGen-v2 image at
preregistered seed 2, XRV and a CXRMate-single/CheXbert initial report. A pure
rule either stops unresolved/unverified or requests CheXagent-2/CheXbert and
applies the frozen fresh-output veto. No image-repair branch, new training,
EHR substitution or direct structured-EHR-to-image claim.

Controls are the initial fixed path and always-second static under the same
six-attempt maximum. A stopped online choice is sealed before paid shadow
static acquisition and cannot observe it. Normally fixed uses four attempts,
online four or six, static six; collection pays all actual attempts and cannot
claim measured GPU savings. Choices precede full-report BioViL endpoint
measurement, which cannot route or regenerate. Missing evidence/failure stays
NA/charged; both EHRs remain in the denominator.

Existing CPU allocation 12654973 sealed and preflighted the plan at
`artifacts/protected/tricompose_v1_2/online_report_plans/online2_12654973_001/`,
manifest `fca888e160ce692eaf100aa5cddf3aabd504c283a1b17e35ce6bf64c628dafac`.
Factory construction/model calls: zero. Prepared script
`slurm/49_online_reports2_v100.sbatch` requests one V100, four CPUs, 48G host
RAM and a thirty-minute allocation cap. At most two new images, four reports
and twelve primary attempts, with secondary encodings separately charged.
The metadata audit is included in that same proposed allocation. **No new
GPU task is submitted, and no prospective result exists yet.** Submission
requires the full concrete script/resource display and subsequent approval.

Subsequent approved execution: the user replied `go` after full script review,
and job **12657042** was submitted with the exact prepared resource request.
It started on V100 node d14-15; completion/quality are not yet asserted in this
update. See [submission record](../docs/online_report_smoke_submission_12657042.md).
New outputs are job-keyed under
`artifacts/protected/tricompose_v1_2/online_report_runs/online2_12657042/`;
historical outputs and sealed protocols remain unchanged.

### Prospective online report smoke — completed, no improvement demonstrated

Approved job **12657042** completed on V100 node d14-15, exit 0:0, Slurm
elapsed **9 minutes 16 seconds**. The [aggregate result](../docs/online_report_smoke_result_12657042.md)
records two new RoentGen images and four reports (two CXRMate-single, two
CheXagent-2), twelve successful primary attempts/zero failures, plus two BioViL
image and four text encoder calls. Full-report endpoint scoring occurred only
after choices were sealed; all four endpoint pairs were available.

Both online cases stopped **unresolved image proxy opposition** after the
initial four-attempt prefix. Second reports were paid, static-only shadow
controls, excluded from online observations. Both static proposals were vetoed
to the initial reference. **Fixed, online and static selected identical outputs
on 2/2 EHRs**, mean uncalibrated BioViL cosine **0.4671736** for each. Primary
prefix counts were fixed eight, online eight, static twelve; actual collection
paid all twelve. No improvement, clinical repair, fixed-baseline advantage or
measured actual GPU saving is demonstrated. These are two already-inspected
development cases, not generalization evidence or a scientific benchmark.

Selected-output proxy totals: EHR–CXR 0 supported/2 known, two oppositions;
EHR–Report 0/2 known, one comparable/one opposition; CXR–Report 8/16 known,
eight comparable/zero opposition. These discrepancies are scorer-dependent
diagnostics, not proof of a clinically faulty modality. Investigate bridge,
conditioning and scorer reliability rather than blindly appending reports.

The same-job metadata audit reconstructed all four completed receipts,
choices, actual costs and online-before-shadow chronology, without reopening
bodies/pixels. Subsequent CPU checks verified root hashes, audit lineage and
private project modes across the new run, audit and runtime. Clinical
acceptance remains false. No historical run/protocol/model/EHR/threshold/winner
was altered. Final score table:
`artifacts/protected/tricompose_v1_2/online_report_runs/online2_12657042/method_comparison.csv`.
Run manifest `aaf6da59ce052de0722e2e1c47e4813f8ad773a95153442803816bff79f8c2fb`;
audit manifest `0fce53d6d3faff1d74f8b7f14d2c2d2d93ef0c93e5c06b586a4b08b0387270af`.

### Online conflict provenance diagnosis — transport passes, clinical cause unresolved

The [new diagnostic protocol](../docs/online_conflict_diagnostic_protocol.md)
and [aggregate result](../docs/online_conflict_diagnostic_result.md) authenticate
the same two EHRs, new run/audit and prior reliability evidence, without
changing any consumed code, inputs, scores or choices. Tool
`tools/diagnose_online_conflicts.py` reads bounded metadata only; canonical
EHR/facts, prompts, tokenizer text/traces and image bytes are not clinically
parsed. No model/GPU/API/download/refit/training/submission or weight read.

Both observed tokenizer inputs byte-match their unchanged supplied prompts.
Recorded unpadded counts are **60/66**, below the frozen **77** rejection limit;
no observed prefix or text replacement. Request/prompt metadata retain the
diagnosis intent and explicitly marked derived opacity prior. This proves
transport integrity, **not** clinical extraction correctness, effective encoder
conditioning or absence of global semantic prompt collapse.

Six matched pneumonia score records across the **same two EHRs**, seeds 0/1/2,
remain below both unchanged threshold 0.55457607 and default 0.5, range
**0.06545602–0.43705505**. The separately predeclared opacity head is also
negative on both new images. Saved report states remain negative/unknown, not
independent image adjudication. No favorable head/seed/threshold was selected.

Same-checkpoint weak-reference sensitivity **9/22 = 0.4091** and specificity
**16/21 = 0.7619** do not qualify clinical fault localization; public RSUA
cohort proxies/template-sensitive SigLIP cannot supply current image truth.
No transport bug was found, but the cause remains unresolved among semantic
bridging, generation fidelity and scorer error. Keep generation intent and
verified image evidence separate, validate the judge before clinical repair.

Protected output:
`artifacts/protected/tricompose_v1_2/online_conflict_diagnostics/conflicts2_12654973_001/`,
manifest `3a36f7df08f94402b8bd011cf89b79bd134bedac3d9ce4d046df51cf9a9cc959`.
Existing CPU allocation 12654973, **0.169611 seconds** before serialization;
**1,720 tests pass** (13 new invented-metadata tests). Independent checks
validate 287 source hashes, three artifacts, matched scores/margins, tokenizer
byte bindings, reference confusion arithmetic, private modes and overwrite
refusal. Original results and all clinical-unresolved statuses are retained.
