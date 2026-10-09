# Exact-opacity sidecar: prepared, not submitted

Prepared 2026-10-05 inside existing **CPU** allocation 12666569. No new Slurm
submission, inference, generation, API or download. This task does not change
historical winners or teammate V1.0/V1.1 work.

## What the read-only metadata audit found

| Inventory | Count |
| --- | ---: |
| Unchanged synthetic EHR anchors | 80 |
| Existing CXR slots, three models | 240 |
| Existing report/triple slots, four experts | 960 |
| Unique report artifact hashes | 428 |
| Explicit cached EHR opacity facts | **0/80** |
| EHR opacity unknown | **80/80** |
| Report opacity positive / negative / unknown | 243 / 25 / 692 slots |
| Report opacity positive / negative / unknown, artifact-deduplicated | 141 / 18 / 269 |
| Old max-head image state positive / negative | 187 / 53 image slots |
| Exact opacity scores present in old cache | **No** |

The cached EHR finding inventory is not a medical re-reading of EHR contents.
Pneumonia/CHF/device facts must not be relabeled as explicit opacity evidence.
The old checkpoint hash matches the RICORD run, but old cached preprocessing
source fingerprint is unavailable. New direct and max-head scores from the
same call will therefore be reported separately from old-vs-new cache changes.

Old max-head CXR/report cached relation: **244 proxy supports, 24 proxy
oppositions, 692 not-comparable**. That is 244/268 = 91.04% agreement among
comparable slots, but only 268/960 = **27.92% comparable coverage**. It is not
clinical accuracy, ground truth or triple consistency, and same-image reports
are correlated. These are existing cached counts, not newly measured exact-
head results. Old metrics/choices remain unchanged.

## Completed implementation and checks

New isolated worker `TriCompose-v1.2/tools/score_cached_opacity_candidates.py`
implements metadata-only `prepare` and separately approved GPU `evaluate`.
It preserves all original 50 CSV columns/cells and row order, appending 16
`opacity_` fields in a new output. Fixed exact-head threshold is 0.5; the
same-call max-head comparison also uses 0.5. No selector or repair action.

All **1,817 V1.2 tests passed in 14.255 seconds**, including **34 new tests**
with invented scores/hashes/vectors. Tested boundaries include raw-vs-max
definition, unknown/uncertain, failed image denominators, no comparable NA,
CSV preservation, fixed lineage, deduplication, explicit EHR provenance,
GPU-before-access guard, explicit plan hash and sanitized errors. These are
engineering contracts, not clinical scorer validation.

Frozen protected preparation:

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_plans/opacity_pool240_12666569_001/
  plan.json
  preparation.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`650f129ce6b20e6e278bd937accfd348cef4e68328fcf793ab722caa55d7323a`.
All **298 source pins**, artifact hashes and private group modes pass the
independent audit. Preparation took 0.823652 seconds before serialization.
No image pixels, clinical EHR/report/prompt bodies or benchmark case records
were opened. Consumed worker/tests/protocol/plan are now immutable.

## Prepared next GPU job — approval still required

Script: `TriCompose-v1.2/slurm/51_cached_opacity240_v100.sbatch`.
Request: ruishanl_1185 / gpu / **one V100, 2 CPUs, 12G RAM, 10-minute cap**.
`noderes -f -g` showed idle/mixed V100 capacity and drained A40 nodes; no
queue-time promise or pin to a specific node. The ten-minute allocation cap
is not an inference estimate.

Frozen XRV will score each existing image once: **up to 240 attempts**, zero
CXR/report generation, zero training, zero checkpoint download. The 960
appended rows reuse each image score for four report slots. Failure leaves
unknown new evidence while retaining all planned rows/images. Expected output:

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_runs/opacity_pool240_<job_id>/
  image_scores.json
  candidate_score_table.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

This output does **not exist yet**: no job was submitted. The complete actual
script/resources must be displayed and approved before `sbatch`. After
completion, independently recompute relation/call counts, original-cell
preservation, source/output hashes and permissions. Report all 80/240/960
denominators and do not convert this axis into a validated full-triple score.

## Subsequent approved submission

After the above preparation record, the complete script/resources were
displayed and the user explicitly approved with **go**. Job **12668204** was
submitted once, with exactly the displayed V100/2 CPU/12G/10-minute request.
See `docs/cached_opacity240_submission_12668204.md` for the submission audit.
The earlier paragraph records the preparation-time state, not the latest
queue/completion state. Frozen tool/tests/protocol and plan remain unchanged.

Job **12668204** subsequently **COMPLETED** on V100 in **25 seconds**, scoring
**240/240** images and producing the protected **960-row / 66-column** sidecar.
See `docs/cached_opacity240_result_12668204.md` for audited aggregate results.
This completion does not upgrade the opacity proxy to clinical triple truth.
