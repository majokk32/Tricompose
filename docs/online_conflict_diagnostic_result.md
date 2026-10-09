# Two-case online conflict diagnosis / 本轮冲突检查结果

This is post-hoc DEVELOPMENT metadata analysis of completed job 12657042,
not another generation experiment or clinical re-adjudication. It does not
modify old EHRs/facts/prompts, code, thresholds, scores, choices or output runs.

## 已确定：本次没有发现输入传递错误

Both frozen EHR anchors retain their original diagnosis-category pneumonia
generation intent. Original seed-0 and new seed-2 requests differ only in
seed/request ID. EHR/facts/clinical-intent/request hashes and the prompt
manifest's declared direct-fact inclusion agree with the actual generated
candidate metadata. The renderer's derived prior is explicitly marked.

| Metadata check | Result |
| --- | --- |
| Fixed EHR/facts and seed-only request binding | 2/2 pass |
| Declared pneumonia intent in request/prompt manifest | 2/2 retained |
| Observed tokenizer-text bytes equal supplied prompt bytes | 2/2 exact |
| Recorded unpadded token counts | 60 and 66 |
| Frozen adapter rejection limit | 77 tokens |
| Adapter-added prefix / pipeline-changed text | Neither on these two records |
| Rendered clinical contexts | Four per prompt |
| Omitted non-direct context counts | Zero and one |

The tokenizer text and trace files were **hashed without decoding their clinical
contents**. The saved candidate is a GPU-worker tokenizer-call attestation,
not a direct encoder hook. These checks establish transport binding and exclude
an observed prefix/text replacement/overlength condition for these two records.
They do **not** establish that the generator used conditioning effectively,
that canonicalization/extraction was medically correct, or that the full
80-case bank has no many-to-one semantic prompt collapse. Source clinical
bodies were not re-extracted, inspected or re-adjudicated in this analysis.

## 已确定：不是单次随机 seed 或本次阈值边界翻转

The authenticated earlier diagnostic has two EHRs at seeds 0 and 1. Adding
this run's seed 2 gives **six image/finding records, still only two EHRs**.
All six normalized pneumonia scores are below both the unchanged transported
threshold **0.55457607** and historical default **0.5**. Range across all six:
**0.06545602–0.43705505**. No seed/threshold is chosen or optimized from this.
The scores are operating-point-normalized values, **not probabilities**.

For the two new seed-2 images, both pneumonia states and the separately
predeclared supplemental lung-opacity states are negative. Thus this analysis
does not find a positive opacity proxy that would resolve the disagreement.
The supplemental head is not promoted into a new EHR fact, surrogate disease
truth or changed acceptance score; no more favorable head was chosen.

Saved report-label metadata, not report-body inspection:

| Expert | Pneumonia states over two reports | Lung-opacity states over two reports |
| --- | --- | --- |
| CXRMate-single | One negative, one unknown | One negative, one unknown |
| CheXagent-2 | Two unknown | Two unknown |

Unknown means unavailable assertion, not negative agreement. The four reports
are CXR-dependent, not independent image references. Static veto reasons include
lost supported/comparable finding IDs (two cases), silenced rather than corrected
conflict (one), and new explicit proxy opposition (one); reasons overlap.
Existing online/static choices and all incurred costs remain unchanged.

## 仍未确定：临床上到底谁错了

The actual code separates the diagnosis mention from its derived image text:
`pneumonia_to_airspace_opacity_prior_v1` renders an opacity-compatible generation
prior. That is authored task intent, not an independently observed image fact.
The label endpoint still evaluates the classifier's pneumonia head. This
intent/prior/classifier distinction must not be hidden by naming all three
"ground truth" or saying an inference wrapper is a native structured-EHR model.
RoentGen remains EHR-derived radiology-text-to-image generation.

The prior weak-reference metrics were authenticated against the **same recorded
XRV checkpoint and threshold hashes**; weight bytes were not rescanned here.
For pneumonia at the transported threshold: positive reference denominator 22,
**TP 9/FN 13**, sensitivity **0.4091**; negative denominator 21, **TN 16/FP 5**,
specificity **0.7619**, balanced accuracy **0.5855**. These are against
**report-extracted weak labels**, not adjudicated current-image truth. They
cannot be applied as the clinical false-negative rate of these two synthetic
images or used to excuse/condemn this generator.

The already inspected 50-image RSUA scorecard is a separate published-cohort
proxy, not patient-independent clinical adjudication. Default XRV sensitivity
12/25 and transported sensitivity 6/25 show operating-point limitations on
that different domain. SigLIP's fixed mean margin AUROC 0.7552 does not qualify
its sign as a clinical presence/absence verdict; prior template stability was
only 2/50. BioViL/SigLIP/report votes do not resolve the two current cases.
No best evaluator/template is selected and no clinical metric is replaced.

**结论：没有发现本次 prompt 传递故障，但目前不能在“语义桥接不适当”、
“生成图像未遵守条件”和“评分器误判”之间作出临床归因。重复换 seed 或报告
不能补齐这个证据缺口，降低阈值也不是本轮允许的修复。**

## Reproducibility / 可复现记录

Tool: `TriCompose-v1.2/tools/diagnose_online_conflicts.py`.
Frozen protocol: `docs/online_conflict_diagnostic_protocol.md`.
Tests: `TriCompose-v1.2/tests/test_online_conflict_diagnostic.py`.
**1,720 tests passed in 10.336 seconds**, including 13 new invented-metadata
tests. These are software-contract tests, not clinical accuracy tests.

Existing CPU allocation **12654973**, analysis time before serialization
**0.169611 seconds**. Zero model calls, GPU tasks, new Slurm submissions,
training, refitting, downloads, API calls or body/pixel/weight parsing.
The independent CPU metadata check verified **287 source hashes**, three
artifact hashes, all six matched score rows/margins, exact tokenizer byte
bindings, weak-reference confusion arithmetic, supplemental state counts,
private project modes and refusal to overwrite this run.

```text
artifacts/protected/tricompose_v1_2/online_conflict_diagnostics/conflicts2_12654973_001/
  diagnostic.json
  transport.json
  matched_seed_scores.csv
  manifest.json
```

Manifest SHA256:
`3a36f7df08f94402b8bd011cf89b79bd134bedac3d9ce4d046df51cf9a9cc959`.
New private directories/files retain 2770/0660 and the project/NFS group.
No generated report text, patient record or image is copied into this note.
Consumed diagnostic source/tests/protocol and outputs are now immutable.

## 下一步：分开两层证据，先验证裁判

1. Keep **generation intent** (what the fixed EHR/prompt requests) separate from
   **verified image evidence** (what a qualified image-reference evaluation can
   support). Preserve the legacy diagnostic counts/choices; do not retroactively
   improve them by changing fact roles, thresholds or missingness.
2. Establish finding-specific image-reference reliability with existing
   independently annotated data before converting a negative XRV proxy into a
   clinically localized CXR fault. Use current classifier/thresholds unchanged;
   report discrimination, operating-point performance, coverage and training-
   overlap/reference limitations, not only AUROC or model consensus.
3. Only then compare controlled-error localization and targeted modality repair
   to fixed/static controls at equal budgets. A paired, fact-preserving prompt
   surface experiment can test conditioning behavior, but cannot on its own
   provide image truth or clinical repair credit.

No next policy or GPU experiment is installed/authorized by this diagnostic.
Further GPU execution requires a complete new script/resources and approval.
