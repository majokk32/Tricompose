# TriCompose server progress — 2026-09-22

## Latest update: generation and diagnostic selection complete

The sections below this update retain the earlier pre-inference audit snapshot.
Since then, explicitly approved Slurm array 12256783 completed all three tasks:
RoentGen-v2, Sana, and PixArt each produced four CXRs (12 total). Scheduler
elapsed times were 55, 86, and 160 seconds respectively. Hash lineage, PNG
dimensions and protected access modes were verified. This is an interface smoke,
not proof of clinical conditioning; both selected cases are context-only.

An immutable source-reconstruction input audit was created at
`artifacts/protected/tricompose_v1_1/tokenizer_input_audits/smoke12_source_reconstruction_12256783_001/`.
Sana's official instruction prefix and Sana/PixArt lowercasing remain intact.
The audit is NOT a historical runtime observation and does not claim actual
token IDs. The 12 original images/manifests are untouched. New CXR runtime
instrumentation observes tokenizer text/ID hashes, with unit tests but no
GPU validation yet.

Submitted after explicit approval: 48 report requests (12 images × MAIRA-2,
CXRMate-single, LLaVA-Rad, CheXagent-2), at
`artifacts/protected/tricompose_v1_1/report_requests/semantics_smoke12_four_experts_12256783_001/`.
All four routes use CXR only; EHR is retained for lineage, not passed to these
report models. Script `TriCompose-v1.1/slurm/61_semantics_smoke12_reports_v100.sbatch`
requests four tasks, each 1 V100 / 4 CPUs / 32 GiB / 20 minutes. Array job 12256939
completed all four tasks (exit 0), 12 reports each. Scheduler times were
MAIRA-2 167s, CXRMate-single 31s, LLaVA-Rad 104s, CheXagent-2 98s. All 48
report hashes, lineages, nonempty contents and protected modes were verified.
Clinical report quality has not yet been verified.
The original 12 CXR outputs are reused without regeneration. Checkpoint/environment
paths are readable; no weights were loaded on the login node. The new protected
audit/request entries have zero permission/group-mode mismatches. V1.1 now has
27 passing lightweight tests, including immutable audit binding, tamper rejection,
prefix capture, tokenizer restoration, and legacy-output handling.

The current evaluation work adds an explicit case/model/seed grid, preserving
the old 80-case single-seed default. Structure checks, bounded image integrity
checks and a 48-row pending registry have completed under
`artifacts/protected/tricompose_v1_1/evaluation/smoke48/`. No clinical scores or
winners have been fabricated. Model-specific section contracts pass 48/48;
temporal-language regex flags occur in 2/12 MAIRA-2, 8/12 CXRMate-single,
5/12 LLaVA-Rad and 2/12 CheXagent-2 outputs. These are review flags, not
adjudicated errors. Three MAIRA-2 reports have fewer than 10 lexical tokens.

Explicitly approved and completed as job 12257177 on V100 (67 seconds, exit 0):
`TriCompose-v1.0/eval/report_v1_1/slurm/30_smoke48_score_select_v100.sbatch`
(1 V100 / 4 CPUs / 24 GiB / 15 minutes), for XRV, CheXbert, secondary BioViL-T,
edge aggregation and diagnostic static selection. The fixed baseline declares
Sana seed 0 + MAIRA-2 before scoring. Exhaustive cost is 6 CXR + 24 report
calls per EHR, excluding evaluators and EHR generation. The selector rejects
all-invalid cases and preserves NA direct EHR edges. The new source changes
have 37 passing evaluation tests including a synthetic in-memory evidence
merge/selection integration test. XRV 12/12, CheXbert 48/48 and BioViL-T 48/48
completed; 48 score rows and two selections passed hash/lineage/mode checks.
Both EHR direct edges remain NA. Selected report-CXR support/coverage is 10/24
versus 2/24 for the fixed path, with zero explicit contradiction signals for
both. These are same-evaluator diagnostic results, not clinical accuracy or
independent evidence of improvement. Protected job outputs are under
`artifacts/protected/tricompose_v1_1/evaluation/smoke48/scoring_12257177/`.

The protected fixed-versus-selected review is under
`artifacts/protected/tricompose_v1_1/evaluation/smoke48/review_12257177_001/`.
It preserves the original selection and separates positive/negative support,
contradictions, and classifier-positive omissions. Fixed support is 2 positive
/ 0 negative; selected support is 1 positive / 9 negative; both have zero
explicit contradictions. The earlier coverage gain is therefore not evidence
of clinical improvement.

## Earlier audit snapshot (before Slurm job 12256783)

## Version status

| Version | Work completed now | Still required |
|---|---|---|
| V1.0 | Historical baseline retained; lightweight tests pass | No new clinical claim attached to the historical baseline |
| V1.1 | Read-only pool audit, CHF/polarity prompt repair, immutable 80-case staging, delta/request preparation, scoring arithmetic/provenance corrections | Approved tokenizer/image smoke; matched validation calibration and independent static evaluation |
| V1.2 | Localization benchmark and action/cost protocol documented | V1.1 review, corruption benchmark, then policy implementation and experiments |
| V1.3 | Independent evaluation protocol and paired case-bootstrap utility | V1.2 gate, independent evaluators/reviewers, held-out results and scale |

This work used no training, model inference, external API, checkpoint download,
or Slurm submission. Model environments/checkpoints and teammate-owned EHR/CXR
evaluation directories were not changed. The report evaluator lives in the
legacy `TriCompose-v1.0/eval/report_v1_1/` location but is V1.1 code.

## Verified historical bank

The August V1.1 cohort has 80 synthetic EHRs, 240 CXR candidates (80 each from
Sana, PixArt and RoentGen-v2) and 960 reports (240 each from MAIRA-2,
CXRMate-single, LLaVA-Rad and CheXagent-2). Candidate and artifact hash lineage
was checked. Four reports share each image; all four paths are CXR-only.

XRV labels cover 240/240 images; CheXbert covers 960/960 reports. BioViL-T and
Qwen evidence cover 0/960 rows despite local model assets being available.
No reviewed calibration bundle was found. Existing best-of-N results remain
diagnostic; there are no paper-primary eligible triples yet.

Only eight EHR cases support direct radiographic comparisons in the existing
edge evaluation. That limitation cannot be repaired by inventing diagnoses or
interpreting unknown as negative. Clinical output quality is not inferred from
file counts, structural report checks or same-evaluator reranking gains.

## New protected outputs

All paths below are relative to
`artifacts/protected/tricompose_v1_1/`:

| Output | Path |
|---|---|
| Revised fixed-80 staging | `staging/bridge_pool80_semantics_20260922_001/` |
| Hash-only old/new delta | `bridge_deltas/delta_pool80_semantics_20260922_001/` |
| Prepared 12-image requests | `cxr_requests/smoke2_semantics_all3_s2_20260922_001/` |
| Corrected scalar metric companion | `evaluation/metric_corrections/xrv_score_semantics_20260922_001/` |

The revised bridge preserves 80/80 EHR hashes. CHF stays context; it no longer
automatically writes cardiomegaly/edema/congestion into prompts. Conservative
word boundaries/polarity prevent substring and uncertain/historical assertions.
There are 46 shared intents, 46 Sana/PixArt prompts and 45 RoentGen prompts;
hash uniqueness is not treated as clinical correctness. The cohort has eight
direct-conditioned cases, 50 context-only and 22 neutral fallbacks.

Compared with the August bridge, 34 cases change at least one prompt:
100 model prompts change and 140 do not. All old artifacts are preserved.
The first two changed opaque case IDs are fixed for interface smoke, with no
outcome-based selection. No new CXR/report has yet been generated from these
requests. Future reuse of unchanged prompts needs explicit old/new lineage
certification, not overwriting old candidate metadata.

## Evaluation corrections

The installed TorchXRayVision source confirms that `DenseNet.forward()` applies
sigmoid followed by operating-point normalization. Historical fields named
`finding_probabilities` therefore hold normalized scores, not calibrated disease
probabilities. New extraction metadata makes this explicit; Brier/ECE are
suppressed for this output space. The scalar companion recomputes the existing
240-record diagnostic summary without inference or changing selections.

Average precision now handles tied scores at the same threshold, removing an
optimistic positive-first tie artifact. Threshold selection utilities require
explicit binary reference support, reserved patient-disjoint groups, and
checkpoint/preprocessing/mapping fingerprints. Their output remains pending
independent review, not automatically paper-eligible. No actual matched-real
threshold fit has been run.

## Verification completed

- Root tests: 18 passed; V1.0: 39; V1.1: 20; report evaluation: 28.
  Total: 105 lightweight tests passed.
- Deterministic facts/prompt rebuild checked again for all 80 staged cases.
- 754 new protected filesystem entries checked: zero permission/group-mode
  mismatches (project-group NFS mapping supported).
- New Slurm script passes `bash -n`; repository diff passes whitespace checks.
- Protected run outputs are ignored by Git. No commit or push was performed;
  the pre-existing untracked teammate handoff file was left untouched.

## Next executable step (approval required)

Prepared script:
`TriCompose-v1.1/slurm/60_semantics_smoke2_all_cxr_v100.sbatch`.

Three array tasks, each 1 V100 / 4 CPUs / 32 GiB / 20-minute limit; four images
per task. The live `noderes -f -g` snapshot showed multiple free V100 resources.
Availability is transient and is not a start-time guarantee. Review the entire
script and explicitly approve before submitting. It uses frozen existing
checkpoints, no network downloads, unique run IDs and protected logs.

After smoke: verify exact saved/tokenized prompt lineage and image conditioning;
then approve any larger changed-prompt regeneration. Separately prepare a
patient-disjoint matched validation contract and frozen-label inference job.
Do not advance to targeted repair before reviewing V1.1 calibration evidence.

Related protocols: `v1_2_localization_protocol.md`,
`v1_3_evaluation_protocol.md`, and `version_roadmap.md`.
