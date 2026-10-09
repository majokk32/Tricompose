# 240 CXR / 960 candidates: opacity evidence overlay preparation

Prepared 2026-10-05 inside existing CPU Slurm allocation **12666569**.
Status: **new scores pending**, no GPU inference/submission in preparation.

## What this adds / 本轮新增内容

复用冻结 BioViL-T 的六条 opacity 正负通用文本，给原有 **240 张 synthetic
CXR slots** 每张做一次主编码，再共享到其四个报告候选，共 **960 行**。
固定三个模板的平均 margin，不挑最好模板，不拟合阈值，不重新生成。

Future `candidate_evidence_table.csv` appends **20 diagnostic fields** to the
existing 66 named fields: all three cosine pairs and margins, fixed mean,
preference state, scoring status, template pattern/range, report-proposal and
exact-XRV proxy relations, joint evidence pattern, null clinical accuracy and
false selector flag. It is not a new weighted score or winner table.

The image score compares CXR with generic finding text, **not generated report
text**. Cached CheXbert remains a separate report-state proposal. Two image
scorers are dependent on the same CXR; agreement does not establish truth.
Preserve unknown/uncertain and all failures. EHR opacity remains unknown on
all 80 anchors, so the new metric does not fill either EHR-related clinical edge.

Joint patterns distinguish missing image evidence, image-source disagreement,
missing report evidence, three proxy proposals agreeing, and report proposal
opposing both image sources. These are not fault-localization labels or repair
actions. No selector, regeneration or old winner replacement is enabled.

## Completed metadata checks / 已完成检查

- Source plan reconstructs the completed exact-opacity table exactly by field
  name; all **63,360 original named cells**, 960 row order and 66 fields remain.
  Consumers must use field names, not positional label vectors. JSON plan
  serialization sorts dictionary keys; the new CSV's field layout need not be
  byte-identical to the old file, whose bytes are never modified.
- All **80 fixed EHR / 240 image / 960 candidate** lineages, four reports per
  image and twelve candidates per EHR verified; no score-based case filtering.
- Every source PNG path/hash is bound without decoding pixels. Every historical
  source/selection remains unchanged. No EHR/report/prompt body or real reference
  row was opened.
- **557 source pins**, plan/artifact hashes, project group modes 2770/0660 and
  script/worker/plan bindings independently verified. New protected plan is
  Git-ignored, atomic and refuses overwrite.
- **32 new invented-fixture tests pass**; full V1.2 regression **1,897 tests
  passed in 11.784 seconds** with the previously documented complete paths.
- New model/selector/generation/training/API/download/submission counts: **0**.

Protocol: `docs/cached_opacity_biovil_protocol.md`.
Worker: `TriCompose-v1.2/tools/score_cached_opacity_biovil.py`.
Tests: `TriCompose-v1.2/tests/test_cached_opacity_biovil.py`.

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_biovil_plans/biovil_pool240_12666569_001/
  plan.json
  manifest.json
```

Plan manifest SHA256:
`340c0774ee5f7c593397023f4ea48eb5eb16cffaeab68467d4a6a7720184b187`.
Worker SHA256:
`e40eb348604fecbcbdf48b61baae4a8fcc2be76353ac67a38db86fb35a47d1d7`.
Batch script SHA256:
`080a7dd1e35cf4555b580fafa6effce46b6f3e1e1369a770dc237cbad7544c26`.

The worker/tests/protocol and their dependency closure are now consumed by the
sealed plan and must remain immutable. Existing runs are not overwrite targets.

## Pending approved-script review / 待批准

Complete script: `TriCompose-v1.2/slurm/53_cached_opacity_biovil240_v100.sbatch`.
Request: account `ruishanl_1185`, partition `gpu`, **one V100, two CPUs, 8G host
RAM, five-minute upper limit**. Preparation `noderes -f -g` showed available
idle/mixed V100s; no forced node or queue/runtime guarantee. The cap is not
actual inference time.

Full completion expects **242 image encodings**: 240 primary slots plus original
indices 0 and 1 each replayed once. One eight-input text batch: six probes plus
two duplicate inputs. No retry, new XRV call or report generation. Image/report
candidate counts do not become 960 independent patients or 960 image calls.

Future protected output:

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_biovil_runs/biovil_opacity240_<job_id>/
  image_scores.json
  candidate_evidence_table.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

**No `sbatch` submitted.** Display the entire script and resources, then obtain
explicit subsequent approval before:

```bash
sbatch TriCompose-v1.2/slurm/53_cached_opacity_biovil240_v100.sbatch
```

Offline assets only; job-local cache/temp/logs are protected. Independent result
audit must verify row/cell preservation, all means/states/relations, complete
failure denominators, calls/replays, hashes and permissions before conclusions.
No broad clinical accuracy, localization, synthetic-domain validation or dynamic
repair benefit is established by this post-hoc developmental evidence overlay.
