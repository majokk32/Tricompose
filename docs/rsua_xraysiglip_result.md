# RSUA frozen XraySigLIP result / 公共分组评分诊断

The [fixed protocol](rsua_xraysiglip_protocol.md) and sealed plan were unchanged.
The complete script/resources were displayed, explicitly approved and submitted
as **12654662**. It completed with exit code 0 on debug P100 `e23-02` in
**44 seconds** (approximately two seconds from submission to start).
Worker time including preflight was **33.172562 seconds**, peak allocated GPU
memory **2.603 GiB**, maximum resident host memory 3,581,304 KiB.
This used one GPU, two CPUs, 8 GiB RAM and a five-minute allocation cap.

All **50 fixed primary images** and **two fixed technical repeats** completed:
52 callbacks/forwards, one model load, zero retries, unavailable scores or
generation calls. Both technical repeats have maximum endpoint difference 0.
The exact pretrained full-weight hash matches the earlier successful native
SigLIP run. No patient selection, checkpoint update, threshold/template fitting,
training, download, API, EHR/report generation or candidate/winner change.

## Predeclared pneumonia results

The frozen RSUA developmental cohort has 25 published pneumonia-class and
25 paper-described normal-class proxy images. The source page calls the latter
Non-Covid. [The data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10570962/)
primarily validates segmentation masks, not independent per-finding clinical
adjudication. These are image-level cohort proxies, not patient-independent
clinical fault labels. Age domain, grouping and training overlap are unverified.

| Frozen text reduction | Margin AUROC | AP | Positive-reference wins | Negative-reference wins | Balanced polarity wins |
|---|---:|---:|---:|---:|---:|
| shows / shows no | 0.6656 | 0.69807810 | 0/25 | 25/25 | 0.50 |
| evidence / no evidence | 0.4944 | 0.51961568 | 1/25 | 24/25 | 0.50 |
| present / absent | 0.7216 | 0.65289897 | 25/25 | 2/25 | 0.54 |
| Predeclared mean of all three | 0.7552 | 0.71671861 | 8/25 | 21/25 | 0.58 |

There are no exact ties. All three template results and their predeclared mean
are retained; the best template is not selected after seeing results. Zero
margin is an authored-text ordering boundary, **not** a calibrated clinical
threshold. AUROC measures rank ordering on the published cohort classes, not
understanding of negation or the reliability of a presence/absence decision.

**48/50 images are template-sensitive**: the polarity changes across the three
fixed templates. Only **2/50 (4% coverage)** consistently prefer absence text.
Those two agree with the normal-cohort proxy, but 2/2 is not 100% accuracy over
50 images, qualified clinical truth or a usable general repair decision rule.
The seven other probed heads have no reference evaluation or invented labels.

**中文结论：这轮模型与完整权重确实跑通，平均差值对公开分组有一定排序信号
（AUROC 0.7552）。但是“存在／不存在”的判断高度依赖模板：同一个模型用
shows/no 几乎总选否定，用 present/absent 几乎总选肯定。平均分对肺炎组的
阳性文本胜率只有 32%，对正常代理组的阴性文本胜率为 84%；48/50 随模板
改变方向。因此不能把这个分数的正负直接当作临床真值或自动返工裁判。**

## Same images, different frozen readouts

| Frozen readout | AUROC | AP | Positive-reference sensitivity / wins | Negative-reference specificity / wins | Balanced result |
|---|---:|---:|---:|---:|---:|
| XRV default 0.5 | 0.7248 | 0.69888206 | 12/25 | 19/25 | 0.62 |
| XRV unchanged transported 0.55457607 | 0.7248 | 0.69888206 | 6/25 | 23/25 | 0.58 |
| BioViL-T fixed mean margin | 0.5520 | 0.58437032 | 21/25 | 8/25 | 0.58 |
| XraySigLIP fixed mean margin | 0.7552 | 0.71671861 | 8/25 | 21/25 | 0.58 |

XRV uses its frozen operating-point-normalized score thresholds; the two
vision-language readouts use the sign of a raw text-margin. These are not
identical decision semantics or calibrated probabilities. The observed AUROC
differences are descriptive; no significance test or superiority claim is
made. No primary scoring metric is replaced.

Mean-margin directions differ from BioViL-T on **28/50**, default XRV on
**14/50**, transported XRV on **10/50**. Agreement is not clinical truth,
and different architectures are not necessarily independent evidence.
SigLIP shares CheXagent-2's visual encoder. No majority-vote arbitration or
regeneration is authorized. Historical scores, thresholds and choices stay fixed.

## Outputs and audit

```text
artifacts/protected/tricompose_v1_2/real_validation/rsua_siglip_pilots/rsua_siglip50_12654662/
  predictions.json
  attempt_journal.jsonl
  summary.json
  prompts.json
  RESULTS_CN_EN.md
  manifest.json
```

Result manifest SHA256:
`3c7cea5b9da2f2e05963af44f40b522c7faccbda55f06b16ba9783c0dbae6d6a`.
Approved script SHA256:
`aff3993322183744ab9af2c29cd9f824963385079a1eda20fc768eb3c91965d5`.
Plan manifest SHA256:
`ca2fb6e62237b0c66cff988425eeaf3d51e3d09b029e5ddf002f34af0a90037f`.

The independent post-run metadata audit rechecked **54 bounded source/config/
reference hashes**, **five artifact hashes**, all 50 primary and two repeat
identities, exact guard receipts, 104 attempt journal events, null clinical
states/probabilities, project permissions and byte-bound dependencies.
It independently replayed the template metrics, stability, repeats and all
cross-scorer comparisons. Fifty unique normalized-pixel hashes are recorded.
The **51 recorded image/weight hashes are GPU-worker attestations**; the CPU
audit did not reopen pixels or weights. Reference classes are from public
opaque cohort metadata, never MIMIC patient rows, reports or real targets.
Predictions were fsynced before cohort/XRV-reference analysis in the approved job.

**1,518 invented-fixture tests passed** before execution, including 24 new
diagnostic tests. Protected directories/files retain project modes 2770/0660
with NFS project GID 65534. The sealed code, protocol, tests, plan and result
have not been edited. This aggregate note does not contain individual images,
patient identifiers or source report text.
