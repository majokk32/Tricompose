# Invented-text conditioning probe / 手写文本条件接口诊断

This is a prospective engineering diagnostic prepared AFTER the failed
two-EHR action contrasts. It is not a held-out clinical benchmark, automatic
repair, an EHR enrichment, or a retrospectively chosen successful model.
No existing EHR, clinical intent, prompt, score, threshold or winner changes.

## Question and fixed design

Does the unchanged native generator receive different token IDs and produce
different images / XRV readouts when an invented radiology-style main prompt
asserts versus denies focal consolidation/pneumonia? The absent arm is a
NORMAL main prompt with negation, NOT Diffusers' `negative_prompt` parameter.
The two authored strings contain no patient/source/target text, demographics,
device, view, laterality or severity. Both strings are specified in source
before any new generation; no best-prompt optimization follows this readout.

Two existing generators × seeds0,1 × present/absent arms = EIGHT image slots.
Within each model/seed pair, hold the seed, original settings, checkpoint,
precision and runtime fixed; change only the authored main prompt. The two
models have different native settings/outputs; cross-model comparisons are
not a pure architecture ablation. A shared seed controls the native random
generator but does not guarantee identical internal noise processing across
different models. No synthetic EHR is fabricated to fit an execution contract.

Reuse `run_cxr.py`'s native factories, RoentGen75steps/CFG3/512px and
Sana20steps/CFG4.5/1024px, float16 as already deployed. Do not rewrite these
consumed runtimes. The new wrapper explicitly sets module eval/grad flags and
checks they remain frozen; there is no optimizer, training/fitting or LoRA.
Observe actual tokenizer calls using the existing observer, retaining Sana's
official prefix/normalization rather than adding another prefix.

An unchanged frozen XRV runtime then scores each COMPLETED image with its own
image hash. Reuse native grayscale/crop/224px preprocessing; record raw
operating-point-normalized pneumonia and consolidation scores. No threshold
fitting, probability claim, accuracy against prompt labels, Brier/ECE, new
weighted score or image/report consensus truth. No report generation or LLM.

## Readouts and interpretation

Keep all eight generation slots and four paired rows, including blocked,
failed and missing classifier outputs. For each pair retain tokenizer-ID hash
difference, image-byte difference, both raw scores and present-minus-absent
delta. Unavailable values stay null/NA, never negative, zero gain or success.
No best seed/model, clinical-acceptance gate, p-value or confidence interval
is chosen on two seeds. Token-ID changes prove delivery at the tokenizer
boundary, not attention usage or clinical image correctness; byte differences
prove output changes, not anatomy. An XRV response may reflect classifier bias.

If this interface probe responds but the EHR chain does not, extraction/text
specificity becomes a hypothesis for a NEW experiment, not a proven cause.
If it does not respond, model conditioning versus observer inadequacy remains
unresolved. Neither outcome justifies modifying the existing EHR/scorer or
counting these hand-authored prompts as repaired EHR-conditioned outputs.

## Execution / permissions

CPU preparation reads only sealed asset receipts, source hashes and weight
file stats; it does not load/hash large model bytes or instantiate factories.
Reuse checkpoint receipts from the completed image-action plan. GPU workers
verify their asset bytes before loading; every new model-load/forward attempt
is durably reserved before invocation. No failed attempt is refunded. At most
two generator loads plus one classifier load, eight generation attempts and
eight classifier attempts; at most three separate worker processes. Models
are not co-resident. First load/forward failure blocks that worker's remaining
slots, with no retry; the next predeclared worker is a different branch.
Native caches/logs/journals, tokenizer traces and images remain protected.

Output uses an exclusive opaque run; interrupted prefixes and charged journals
remain for review, never blind resume. Source/body/PAT data is not read or
printed. No network/API/download is authorized. New V100 execution is proposed
because scheduler metadata shows free V100 capacity; this does NOT verify
actual GPU memory or guarantee runtime/kernel compatibility. Workers require
at least16GiB and fail closed otherwise. Proposed one V100, two CPUs,32GiB
host RAM and a15-minute cap, not a runtime/queue promise. CPU-only fixtures
verify scope, missingness, lineage and frozen flags before any GPU job.

Entry: `TriCompose-v1.2/agent/run_text_conditioning_probe.py`.
Protected plans/runs: `tricompose_v1_2/text_conditioning_probe_plans/` and
`text_conditioning_probe_runs/` under `artifacts/protected/`,2770/0660.
The COMPLETE batch script and resource request must be displayed and explicitly
approved before sbatch. A "keep working" message before that display does not
authorize submission. No original consumed program/protocol/run is altered.
