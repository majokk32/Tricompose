# Fixed-bank BioViL-T opacity evidence overlay

PREPARED ONLY. New GPU inference/submission requires the complete batch script,
resource display and explicit subsequent user approval. No implicit permission
to regenerate, rerank or change the old winners.

## Scope and immutable source

Retain the complete **80 EHR / 240 CXR / 960 report-candidate slot** bank.
Existing exact-opacity sidecar manifest:
`778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94`.
Its original preparation manifest:
`650f129ce6b20e6e278bd937accfd348cef4e68328fcf793ab722caa55d7323a`.
All **66 existing columns / 63,360 CSV cells**, row order, EHR/fact/image/report
lineage and every historical selection stay unchanged. Refuse already overlaid
tables and existing output runs. Inputs are wholly synthetic; no source EHR,
report/prompt bodies, real image or benchmark reference row is opened.

Metadata prepare verifies source closure, reconstructs the exact-opacity CSV
from its original plan/cache, binds 240 source PNG paths/hashes and freezes the
new plan. It does not produce a fake scored table full of pending values.
Only the separately approved GPU worker reads those synthetic image pixels.

## Frozen probe contract

Reuse the just-completed RICORD BioViL-T opacity diagnostic, manifest
`d50e5594e43f7979f385f0e14745d66103714b16693e7656ddcf7812e85480ef`,
its checkpoint/vendor/package/loader bindings and the same six authored generic
opacity assertions. Three families: shows/no, evidence/no, present/absent.
Primary diagnostic is the unchanged arithmetic mean of all three positive-minus-
negative cosines. No favorable-template selection, threshold/deadband fitting,
class/severity/view substitutions, retraining or EHR enrichment.

L2-normalized 128-dimension embeddings; official loader internally min/max maps
PNG to uint8, resize512/crop448. This is not identical to XRV preprocessing and
does not make architecture votes independent. Frozen/eval both encoders, with
inference_mode and offline existing assets. No report body is encoded: the new
margin is an IMAGE versus GENERIC FINDING-text preference, not a newly measured
full generated-report/image cosine.

Mean > 0 / < 0 gives positive/negative **preference**, exact zero remains unknown.
Do not label it calibrated disease probability or confidence. Keep all three
raw cosine pairs/margins, mean, min/max and template-direction pattern; mixed
templates do not become a learned uncertainty probability or an action gate.

One encoding per image SLOT shared by its four report candidates. Exactly 240
primary image encodings, plus one replay for original image indices 0 and 1:
full completion expects **242 image calls**, one eight-input text batch
(six probes plus two duplicate inputs). No retry or replacement. Failed main
image outcomes retain all four report slots, explicit failure status and null
new margins/unknown preference. Replay failures do not erase main predictions.

## Evidence table, not a selector

Append **20** `opacity_biovil_` fields to the existing 66-column table. Keep
CheXbert's cached positive/negative/uncertain/unknown proposal unchanged.
Use the existing exact-XRV state at fixed 0.5; no new XRV inference, legacy-max
substitution or No-Finding expansion. Only explicit states are comparable.
Append separate image-source and report-proposal proxy relations plus:

- image_evidence_unavailable;
- image_sources_disagree;
- report_evidence_unavailable;
- three_proxy_sources_agree;
- report_proposal_opposes_two_image_sources.

These are descriptive dependency patterns, NOT truth or instructions to repair
a report/image. Two agreeing image scorers can share bias and source content;
an opposing report label may be an extraction error. No faulty modality is
assigned, clinical accuracy remains null, selector_used false.

For this opacity scope all 80 cached EHR states remain unknown; both EHR-related
clinical edges stay unavailable, not zero, perfect or filled from other modalities.
Unknown/uncertain absence is not a contradiction or successful support.

Aggregate counts at correct units: 240 image slots; 960 correlated candidate
slots; 80 fixed EHRs. Show support/opposition WITH coverage and full denominators,
per-expert availability, joint patterns and failure counts. Zero comparison has
NA agreement. No model winner, weighted total score, clinical error rate,
patient-independent p-value or repaired-triple benefit is inferred.

## Qualification and execution

The 50-case benchmark supports a bounded opacity diagnostic, not synthetic
domain transport or CheXbert assertion qualification. Its labels were selected
by derived unanimity and outcomes already seen. This overlay is post-hoc
DEVELOPMENT, not an independent final endpoint or a complete clinical triple
score. Do not use reference performance as a candidate score or scorer weight.

Worker: `TriCompose-v1.2/tools/score_cached_opacity_biovil.py`.
Invented-fixture tests: `TriCompose-v1.2/tests/test_cached_opacity_biovil.py`.
Metadata prepare in an existing CPU Slurm allocation. GPU evaluate requires
explicit approval flag + actual Slurm cgroup + CUDA + offline guard before image
or model access. New atomic protected run; directories 2770, files 0660, project
group only. Redirect cache/temp/logs to new workspace job runtime. External
models/environments/checkpoints and all original outputs remain read-only.

Public logs contain sanitized status/runtime/memory/hashes only. New image
scores, candidate table and reports remain protected, Git-ignored. After run,
independently recompute all 960 row joins, original cell preservation, fixed
means/preferences/relations, failures/calls/replays, hashes and private modes.
Consumed helpers/tests/protocol/plan are immutable. Further scoring, selection
or repair requires a separately specified/approved task.
