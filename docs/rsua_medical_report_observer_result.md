# RSUA50 official SRRG + CheXbert: result / 结果

The reviewed, unchanged script completed as job **12861745**, debug A40
`b11-09`, exit `0:0`, elapsed **2m47s**. It reused all 50 original source
PNGs, the frozen CheXagent-2 official findings prompt, revision
`9f7225fc382ddd1297ade1aa796da660237940bc`, float32, seed 42, greedy decoding
and 512-token cap, followed by the existing frozen CheXbert in a separate
process. No EHR, source report, reference class or earlier scores were model
inputs. Predictions were fsynced before the worker read reference classes.
No original winner, gate, threshold, prompt or model was changed.

## Readout / 核心结果

These are published pneumonia versus paper-described normal-cohort **proxy**
references, 25 each. The official negative release says Non-Covid, not
independently adjudicated per-image normality. Only pneumonia has a reference.
Patient grouping, age domain, training overlap and transfer to synthetic images
remain unverified. This follow-up was chosen after viewing Qwen's result:
development comparison, not an untouched final test.

| Frozen observer | Positive-proxy explicit support | Negative-proxy explicit support | Explicit coverage | Correct explicit support / all 50 | Conditional support / explicit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official CheXagent-2 findings + CheXbert | 0/25 | 0/25 | 0/50 | 0/50 | NA: zero explicit |
| Previous Qwen image observer | 2/25 | 13/25 | 34/50 | 15/50 | 15/34 |
| Cached XRV, default 0.5 | 12/25 | 19/25 | 50/50 | 31/50 | 31/50 |
| Cached XRV, transported 0.55457607 | 6/25 | 23/25 | 50/50 | 29/50 | 29/50 |

CheXbert's pneumonia head returned **unknown on all 50 reports**. There were
zero positive, negative or uncertain pneumonia answers and zero unavailable
executions. Explicit-only TP/FN/TN/FP are all zero because no slot was explicit,
not because every prediction was correct or incorrect. Conditional support is
undefined, not zero. Missingness bounds are [0,1], not confidence intervals or
an observed success rate. Categorical states do not yield AUROC/AP/Brier/ECE;
no probabilities were invented. XRV was not rerun.

肺炎这项当前没有可比较证据：不是“50 张都无肺炎”，也不是测到了 0% 临床
准确率。这条医学报告→标签链因此不能承担肺炎一致性的主裁判。它不证明原来的
CXR 生成失败，也不能据此判定其他 finding 正确。Qwen 的 numeric planner 是
另一种角色，本实验不检验调度策略优劣或生成节约。

## Other heads and mechanical checks / 其他标签与工程检查

| Finding | Positive | Negative | Uncertain | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Atelectasis | 2 | 0 | 1 | 47 |
| Cardiomegaly | 12 | 32 | 0 | 6 |
| Consolidation | 0 | 39 | 1 | 10 |
| Edema | 8 | 16 | 0 | 26 |
| Enlarged cardiomediastinum | 23 | 15 | 0 | 12 |
| Fracture | 5 | 0 | 0 | 45 |
| Lung lesion | 2 | 0 | 0 | 48 |
| Lung opacity | 4 | 5 | 0 | 41 |
| Pleural effusion | 1 | 42 | 0 | 7 |
| Pleural other | 0 | 0 | 0 | 50 |
| Pneumonia | 0 | 0 | 0 | 50 |
| Pneumothorax | 0 | 45 | 0 | 5 |
| Support devices | 5 | 0 | 0 | 45 |
| No finding | 8 | 0 | 0 | 42 |

These are output-state distributions, **not accuracy for the other 13 heads**.
CheXbert head order and 0/1/2/3 mapping match the unchanged deployed source
(`cxrmate/tools/metrics/chexbert.py` and the existing extraction adapter).
Other heads have explicit states, so this is not an all-empty label vector.
Consolidation/lung opacity cannot be substituted for pneumonia after seeing
the result. No report bodies were opened in this postflight; the evidence does
not isolate report omission from label extraction error or establish clinical
correctness of the generated wording.

All 50 report generations and all 50 label forwards completed. One model load
per stage, no retry. Report output lengths were 54–110 tokens; label input
lengths 46–98. No 512-token cap or input truncation. There were 34 distinct
report byte hashes; repeated wording is not by itself a clinical diagnosis
of prompt collapse or a semantic diversity metric.

Worker interval through summary creation: **150.494s**. Peak Torch-allocated
VRAM: **13.362 GiB** for CheXagent and **0.422 GiB** for CheXbert, not total GPU
memory usage. Models ran sequentially, not co-resident. The observer cost is
50 report generates plus 50 label forwards; it is not a free classifier.
No new CXR, planner, XRV or API call, training or accepted repair occurred.

## Provenance and readout files / 文件位置

Relative to `/project2/ruishanl_1185/inference_3mod`:

```text
artifacts/protected/tricompose_v1_2/real_validation/
  rsua_medical_report_runs/rsua_medical50_12861745/
    reports/                       # protected generated reports only
    report_records.json
    predictions.json
    summary.json
    reports.journal.jsonl
    labels.journal.jsonl
    RESULTS_CN_EN.md
    manifest.json
  rsua_medical_report_audits/rsua_medical50_12861745_12851223_001/
    audit.json
    audit_source.py
    manifest.json
  rsua_medical_report_reviews/rsua_medical50_12861745_12851223_001/
    observer_comparison.csv
    finding_state_counts.csv
    summary.json
    RESULTS_CN_EN.md
    export_source.py
    manifest.json
```

Source manifest SHA256:
`4df6cf3378aa0a6dd7ab38ce93261fb42909606be4798930b63e671328e33a2f`.
Audit manifest SHA256:
`f65051c63a6a9bb1d57aa1f194bd40550c3a779d3744318b1f44d6eae19b766b`.
Readout manifest SHA256:
`405629c3d37c79388bea033410bc58c9406390b505109ae6f36b295ebe6d10a9`.

Existing CPU allocation `12851223` audited 57 result artifact hashes, 320
source hashes, 14 input artifact hashes, all 50 PNG byte hashes, 26 asset
stats, 50 named 14→8 state projections and exact replay of **151 journal
events per phase**. Group/modes passed. This independently rechecks receipt
contracts and aggregate arithmetic, **not inference or per-image reference
assignment**. No pixels, report bodies or real reference rows were opened.
Large model bytes were fully checked in the GPU preflight; CPU postflight
checked their stats rather than claiming another full weight hash pass.
The pre-submission 24 worker fixture tests and all 3,950 V1.2 tests passed.
Postflight/export utilities do not revise consumed code or protocols.
After the run, all **3,950 existing V1.2 tests passed again in 28.483s**;
12 additional invented metadata-arithmetic fixtures passed. Final recheck
verified all 64 artifacts across the source/audit/readout manifests, 17
directory and 74 file modes/groups, and both aggregate CSVs against their
JSON records. These are engineering checks, not clinical validation.

## Next-step consequence / 后续决策

Keep this chain's pneumonia evidence unavailable for primary fault attribution.
Unknown must remain unknown in the score table and planner state. Do not
increase a score by writing negatives, relabeling opacity as pneumonia,
changing the reference classes, fitting a threshold or requiring the generator
to mention the answer. The other heads need their own appropriate reference.
CheXbert also scores report candidates and CheXagent shares XraySigLIP's
visual encoder; agreement is neither independent adjudication nor a gold vote.

Prioritize evidence scope/coverage and suitable independently annotated
observable findings before granting image-regeneration authority. Preserve
stop/abstain when no supported decision exists. Further model/data execution
requires its own reviewed script and approval. This result does not authorize
an automatic retry, API planner or clinical repair. The original plan, script,
worker/tests and `docs/rsua_medical_report_observer_protocol.md` stay immutable.
