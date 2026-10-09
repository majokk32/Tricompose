# Fixed-bank opacity conflict registry

CPU-only metadata diagnostic following the completed exact-XRV/BioViL-T
overlays. No model execution, text/pixel inspection, clinical truth assignment,
ranking, threshold fitting, filtering, regeneration or source modification.

## Immutable source and denominator

Bind `candidate_opacity_biovil_runs/biovil_opacity240_12670345/manifest.json`
at SHA256 `6b3027a012ec146b46a29a911465bb8fe1bdded087b6ee074d5994919c89c828`.
Retain the complete 80 EHR / 240 image-slot / 960 candidate-slot inventory.
Use image-model IDs from the hashed image outcomes, never parse an ID or infer
one from a filename. Verify all source pins, output artifacts and the frozen
plan. Reconstruct every added field from the old 66 fields and saved cosine
pairs. Preserve named-cell values, row order, EHR hashes and original winners.

Every row has one unchanged mutually exclusive dependency pattern. Add an
explicit metadata-only verification lane:

| Dependency pattern | Diagnostic lane |
| --- | --- |
| image_evidence_unavailable | missing_image_evidence |
| image_sources_disagree | verify_image_evidence |
| report_evidence_unavailable | missing_report_assertion |
| three_proxy_sources_agree | proxy_agreement_control |
| report_proposal_opposes_two_image_sources | verify_report_assertion |

Lanes are task descriptions, **not actions invoked**, clinical labels, stopping
rules, image/report repair authorization, or confidence levels. Template-sign
variation is an orthogonal metadata flag, not a fitted uncertainty threshold.
Unknown/uncertain remain unknown/uncertain; evidence unavailability is not
negative or agreement. EHR opacity remains unknown in the complete source.

## Dependency-aware outputs

Write a fresh protected atomic run with:

- `candidate_registry.csv`: all 960 projected metadata rows in source order;
  original row digest, fixed EHR/fact/image/report hashes, cached states,
  unchanged pattern, diagnostic lane and all false repair/selection flags.
- `image_registry.json`: all 240 slots and their four report members, original
  model/lineage, state/preferences, template pattern and counts at slot level.
- `report_registry.json`: all unique exact report-content hashes, opaque
  ordinal group IDs and every candidate/image/case/model membership. A shared
  report must retain a single cached assertion state; same text in different
  image contexts can have different pairwise relations. Do not collapse those.
- `case_registry.json`: all 80 fixed EHR anchors and every image/report slot.
- `verification_requests.json`: deduplicated report-assertion group requests
  for the report-opposition lane and image-slot requests for image disagreement.
  These contain opaque IDs/hashes, not text or images, predictions as input to
  a new verifier, executable paths, or claim of ready model inference. No
  resolved/passed request exists in this stage. All other lanes remain in the
  complete registries; these requests are developmental investigation targets,
  not a new benchmark or representative accuracy sample.
- `summary.json`, `RESULTS_CN_EN.md`, and hash-bound `manifest.json`.

Deduplicate a future **text-only assertion check** by exact report bytes, not
case/model; it has one task per unique text and reattaches to every source
context. An image-text check is context-specific and cannot be deduplicated by
report text alone. Shared reports or scorer votes are not independent clinical
evidence. Report group counts across patterns overlap and must not be summed
as total unique reports. Likewise case sets can overlap across lanes.

Show full denominators and within-pattern unique images, EHRs and report hashes,
per-image-model (80 slots/model) and per-report-model (240 slots/expert) counts,
exact-text multiplicity, cross-image reuse and cross-pattern reuse. Agreement
controls are unadjudicated controls, not clean clinical gold. Finding-limited
conflict absence is not complete triple correctness. No lexical rule for
opacity, new medical label, scorer weight or best model is invented here.

## Execution and acceptance

Worker: `TriCompose-v1.2/tools/build_opacity_conflict_registry.py`.
Tests: `TriCompose-v1.2/tests/test_opacity_conflict_registry.py`.
Run only in the actual existing CPU Slurm cgroup; no new submission/GPU job.
All outputs are protected, group-only 2770/0660, atomic and non-overwriting.
No credentials, external API, model download or external directory write.

Acceptance checks: all 960/240/80 slots retained; every hash/row/context join;
unique ordinal report IDs; shared-report states stable; counts/denominators
agree; template flags do not authorize actions; requests are one/text versus
one/image as appropriate; same-text opposite contexts remain separate; unknown
does not mean negative; deterministic results and unchanged source bytes.
Consumed code/tests/protocol and protected output remain immutable. Human
feedback is not required for exploratory automatic checks, but missing clinical
validation is not filled by this registry. A new learned-model verification job
still needs its complete Slurm script/resource presentation and approval.
