# Fixed-EHR CXR action diversification / 两条图像动作对照

Development candidate expansion after the Qwen and medical-report image-observer
diagnostics. Not independent clinical adjudication, qualified error localization,
an automatic repair policy, a confirmatory test or an LLM advantage experiment.
The failure of an observer does not establish that a generated CXR is wrong.

## Fixed inputs and predeclared actions

Keep the same two synthetic EHR anchors, facts and model-specific final prompts
from the previously consumed two-case generation plan. Do not choose a new EHR,
add a finding/device, edit prompt wording or filter cases by outcomes. Use:

1. RoentGen-v2, seed 2 (same model as the prior seed 0/1 path).
2. CheXGenBench Sana, seed 1 (model plus its original approved renderer).

Both requests retain their own original prompt hashes and the identical shared
clinical-intent hash/direct-fact inventory. They need not have identical text,
token counts or diffusion settings. The cross-generator action includes model-
specific surface rendering; this does not isolate a pure architecture effect.
Neither generator natively consumes structured EHR.

Each new image gets its own XRV observation, CXRMate-single report and CheXbert
labels. CXRMate-single remains CXR-only, not CXRMate-ED. Use the same expert for
both arms and both cases. No cached report is attached to a new image. Compare
the new rows with the existing same-EHR RoentGen seed-zero CXRMate baseline
using the frozen exploratory preservation gate, without selecting/replacing an
original triple. Report every failed/incomplete branch and all four planned slots.
Do not promote new rows based solely on this already inspected proxy evidence.

## Assets, execution and budgets

Reuse authenticated local checkpoint receipts from the full-assets report plan
and original operational Sana plan; no download, new model, training/fitting,
LoRA, new scorer, threshold, prompt or external API. CPU preparation reads only
request/receipt metadata and file hashes/stats, not EHR/fact/prompt/report bodies
or image pixels. Weight hashes are reused from sealed receipts and complete
model bytes checked by the separately approved GPU preflight/worker.

Two cases × two image actions × (CXR + XRV + report + CheXbert) = at most
**16 reserved worker requests**, at most four new images and four new reports.
Eight requests per case, zero retry. Every request is reserved/fsynced by the
existing bounded ledger before invocation; a failed load/request stays charged.
Failure stops that branch; the next predeclared branch is not a retry. Global
timeout retains partial outputs and charged journals, never resumes automatically.

Models run in separate subprocesses, not simultaneously. Reuse the unchanged
official generation parameters and frozen worker adapters. Proposed allocation:
one debug A40, two CPUs, 32 GiB RAM, ten-minute ceiling; parent timeout 570s,
owned-child cleanup through SIGINT, 15s kill grace, per-worker timeout 120s.
This is a ceiling, not a completion/queue promise. The registered active
generators/report worker have a 16-GiB planning memory class. The A40 is an
availability choice, not a training requirement; a different GPU/script needs
its own review and approval.

Stable exclusive protected run paths are necessary because existing adapters
embed absolute paths. No overwrite, old-run move, source mutation or blind
resume. Public output is sanitized status/hash only; generated content, source
requests, caches, logs and results remain protected with project 2770/0660 modes.

## Readouts and limits

Save four-slot outcomes, complete triples, raw score table, per-case ledgers,
baseline/new proxy fact comparisons and exact lineages. Report edge support,
opposition and comparable coverage separately, image/report duplicates, official
report structure and unsupported temporal language, charged/validated/failed
request counts and measured worker wall time. Equal request count is not equal
GPU time; the two image models have different native inference settings. Raw
XRV normalized scores are not clinical probabilities. Unknown stays unknown.

The fixed EHR/image proxy may remain unresolved; a proxy preservation pass is
not a clinically accepted repair. Reference evaluation, primary clinical
accuracy, measured saved calls and LLM-versus-rule advantage remain unavailable.
No Qwen/image observer/API runs are included. A secondary BioViL endpoint, if
needed later, requires separate approval and cannot become a primary clinical
judge. Do not tune this experiment after observing its outcome.

The complete new Slurm script and resource request must be shown and explicitly
approved before submission. Earlier approvals do not authorize this job.
