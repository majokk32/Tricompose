# XraySigLIP alternate automatic readout (not clinical arbitration)

Status: prospective development probe. No new training, scorer fitting, model
download, generator call, EHR change, or winner update. Model execution requires
its complete Slurm script/resources to be displayed and explicitly approved.
This does not satisfy the frontier's independent **clinical** evidence request.

## Existing assets and scope

Use the already deployed `StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli`
snapshot `f0edbf5d90dba44edb7f4f96d8663537cb0749bf`, its complete `SiglipModel`
text and vision encoders, and the existing CheXagent-2 environment. The actual
config/processor specify **512 x 512**, float32, and a 64-token tokenizer
length; do not infer preprocessing from the repository name. Use the native
`SiglipModel`/`SiglipProcessor`, no randomly initialized classification head,
remote code, new adapter, quantization, FlashAttention, or TF32.
The installed implementation is Transformers 4.41.2; the official
[SigLIP interface](https://huggingface.co/docs/transformers/v4.41.2/en/model_doc/siglip)
specifies max-length padding. Medical polarity templates below are our fixed
diagnostic, not an official disease-classification metric or threshold.

It is distinct from the current Qwen/XRV checkpoints, but is also the visual
dependency of CheXagent-2. Shared training data/representation and correlation
remain unverified; it is **not** an independent final evaluator of that report
generator, a calibrated clinical classifier, or an adjudicator.

Authenticate the validated frontier manifest
`204b499d2eeab6cd7f849ca854a8beefbc433a53876bb03b507a3aa56e0493bc`
and original guarded input plan manifest
`2864941d29f60e890cdd51e88a7126986027b495345d4cf8a0ff23bdc2b77959`.
Use the same six original synthetic images plus four existing isolated uniform
controls (ten unique inputs / 18 logical slots). The 21 frontier image/finding
requests are metadata references, not 21 new patients or model calls.
Do not expand/select patients, replace images or create a new EHR prompt.
Preparation reads only bounded quote-free metadata, small model config/tokenizer
files and installed source files; it does not open images, reports, EHR bodies,
weights, responses, source patient keys, or recursively follow manifest sources.
Record image/weight size and mtime; full weight and image hashing occurs only
in the approved GPU allocation. Never transmit patient artifacts to an API.

## Blind fixed scoring and output semantics

Every valid original image receives all eight heads in the same fixed order:
atelectasis, cardiomegaly, consolidation, edema, lung opacity, pleural effusion,
pneumonia, pneumothorax. For each head use all three authored polarity pairs:
`The chest X-ray shows X.` / `The chest X-ray shows no X.`,
`There is evidence of X.` / `There is no evidence of X.`, and
`X is present.` / `X is absent.` Do not choose templates after seeing scores.
No view, location, device, severity, historical comparison or EHR fact is added.

Use one native joint image/text forward per image: 48 text strings, max-length
padding, no truncation, float32, batch-one image tensor (1,3,512,512). No hidden
text-encoder passes or uncharged retries. Fail if loading reports missing,
unexpected, mismatched or error keys; force eval and disable every gradient.
Validate finite bounded cosines and raw logits; save both, positive-minus-negative
cosine margins, mean/range across all three templates, and template sensitivity.
Preference names are `present_prompt_higher`, `absent_prompt_higher`,
`tied_templates`, or `template_sensitive`. These are **not** positive/negative
clinical labels, confidence, calibrated probabilities, or clinical truth.
No sigmoid/softmax, fitted operating point, score pooling or primary metric.

The existing mechanical guard is invoked before the callback. Uniform controls
must have null scores and zero callbacks/forwards; they never join candidates.
Callback receives only checked pixels; all generic texts are fixed, not derived
from old Qwen/XRV labels, reports, IDs, scores, requests, or control arm names.
Preparation parses the bounded request metadata but retains only identity
references; it never uses proxy states to form texts or select among images.
Freeze/fsync predictions before computing any post-hoc proxy-state comparison.
Missing/uncertain/unknown states remain not comparable;
agreement or opposition is qualified as an automatic readout, never correctness.
Keep all 21 requests clinically unresolved, no image/report fault attribution.

## Cost, failures and preservation

Maximum six scoring attempts, zero retries. Reserve and fsync a protected
attempt-journal event before invoking a callback; record whether a model forward
actually started, completion/failure type and wall time. Report attempts, actual
forward attempts, model-load attempts and guard blocks separately. A failed
callback still consumes its attempt budget; unavailable scores remain null.
Do not repeatedly load a failed model. Interrupted/failed run journals stay in
their new protected temporary directory for reconciliation, not blind resume.
No inference or heavy hashing on the login node; preparation uses existing CPU
Slurm allocation. Runtime checks actual Slurm cgroup, explicit allow flag, CUDA,
minimum 12 GiB usable VRAM, exact program/runtime/config hashes, asset stats,
complete weight loading and original normalized pixel hashes. Public output is
sanitized status/runtime/memory/hashes only; all per-input scores stay protected.

Worker `TriCompose-v1.2/tools/probe_xraysiglip_findings.py`; invented-metadata
tests `TriCompose-v1.2/tests/test_xraysiglip_probe.py`. Fresh atomic protected
plans/runs under `xraysiglip_plans/` and `xraysiglip_runs/`; never overwrite.
Project-only directories 2770/files 0660. Seal consumed program/metadata and
artifact hashes; preserve old rows, scores, states, EHRs, winners and histories.
All clinical-acceptance/resolution/primary-metric/regeneration flags stay false.
This is a selected development diagnostic, not real matched calibration,
held-out localization accuracy, successful repair, or quality improvement.
