# RadGraph-XL deployment and real CPU interface test

## Outcome / 结果

The official frozen XL implementation is deployed and passed a real, offline
CPU inference smoke. This is a working extractor/reference-metric interface,
not a clinically qualified scorer or a new TriCompose winner selection.

官方冻结 RadGraph-XL 已部署，并完成实际离线 CPU inference。实体/关系提取
与官方报告对比接口可运行；这不是临床准确率验证，没有改变旧分数和择优结果。

User approval: the continuation after the explicit 416 MB / 2 GB-capped
deployment proposal. Execution used the already active CPU Slurm allocation
12714150 on b05-06, confirmed by the actual process cgroup. No `sbatch`, new
GPU reservation, training, fine-tuning, external clinical API, real patient
report/EHR/image input or new generation was used.

## Local installation

```text
RadGraph/                                      # clean official source checkout
runtime/venvs/radgraph-xl-v12-12714150-001/      # isolated CPU environment
runtime/models/radgraph-xl-v12-12714150-001/    # pinned archive + extracted XL
.cache/radgraph_v12/                           # project-local package/HF cache
```

Official source: https://github.com/Stanford-AIMI/radgraph
at `87f11a1ff4d2046a838be5f0243857d93780ddec`, package 0.1.18.
The installed package was built from that exact official archive, without
editing the checkout. All **405** installed official Python files match it.

Official weights: https://huggingface.co/StanfordAIMI/RRG_scorers
at `6646433b3ad83a10f6e141db76d0ece44312b236`.
The 416,179,571-byte XL archive was checked against SHA256
`fadb5a3454e8996714b609e3105a07e447fa61aa88fffe966b650959475117b6`
and safely extracted with path/link/size checks.

The selected upstream implementation uses `AutoModel.from_config` and then
loads the complete XL state dict. No separate encoder-weight download is
necessary. The config/tokenizer identity is
`microsoft/BiomedVLP-CXR-BERT-general`, revision
`6172dbfa7c061d635a4b86761b80e324e1995496`. Three public small files were
verified by Git blob hashes, additionally SHA256-recorded, and placed into an
explicit pinned offline cache. Combined model/tokenizer download:
**416,415,674 bytes**, below the approved 2 GB cap. Environment and expanded
cache storage are additional to download size.

Core versions: Torch 2.6.0+cpu, Transformers 4.44.2, NumPy 1.26.4,
huggingface-hub 0.36.2. `pip check` passes. All installed package versions are
recorded in the protected audit environment manifest. Source/checkpoints/
caches/results are ignored by project Git; existing environments are unchanged.

## Actual model test

Six invented report-pair probes were specified before inference. Five had
nonempty inputs; one explicitly tested missing-input handling. They are not
MIMIC source reports or radiologist-annotated validation cases.

| Authored probe | Entity F1 | Relation-presence F1 | Full-relation F1 |
|---|---:|---:|---:|
| Identical text | 1.0000 | 1.0000 | 1.0000 |
| Negation changed | 0.4000 | 0.4000 | 0.4000 |
| Uncertainty changed | 0.3333 | 0.3333 | 0.3333 |
| Laterality changed | 0.7500 | 0.7500 | 0.7500 |
| Measurement changed | 0.6667 | 0.6667 | 0.6667 |
| Empty hypothesis | NA | NA | NA |

These are the three **official reward components**, not a new locally invented
clinical F1. Equality of the three components in these particular probes does
not make their formulas identical. The interface keeps them separate.

- 10 actual report forward passes, CPU only.
- Initialization 3.473 seconds; inference 2.052 seconds.
- Total including integrity checks/imports: **13.157 seconds**.
- Process peak RSS: **1.600 GiB**.
- `eval()`, all parameter gradients disabled, `torch.inference_mode()`.
- HF/Transformers offline mode, implicit-token use disabled and socket
  connections blocked throughout model loading and inference.
- No entity absence converted to negative; empty pairs are ineligible/null.
- V2 preserves all **11** exact native XL labels. Measurement and anatomy
  entities are not silently merged into disease-positive/negative counts.

The negation/uncertainty probes produced the corresponding native observation
states. This is a narrow interface observation, not measured clinical accuracy.
**A laterality conflict still receives 0.75 F1:** an aggregate graph similarity
must not by itself declare a report clinically consistent or safely repaired.
Native graph labels also do not establish current-patient/prior-study scope.

## Protected outputs and independent integrity audit

```text
artifacts/protected/tricompose_v1_2/radgraph_interface_runs/
  native_xl_12714150_001/
    score_table.json
    native_annotations.json
    summary.json
    run_manifest.json
    worker.log

artifacts/protected/tricompose_v1_2/radgraph_interface_audits/
  native_xl_12714150_001/
    audit.json
    environment_manifest.json
```

Run manifest SHA256:
`ffab7c7305bf02e77afd93873085931d154fb686cdbd86fd76d5d497b882e389`.
Audit SHA256:
`f77a3c587815fb47e58ebb6093cec7fac01a366868b608a4f3910ccaab5fe9c5`.

Independent receipt replay verified source/output pins, native graph hashes/
counts, all **15** reward components against the unchanged official reward
code, missing-input handling and protected permissions. This validates
artifact/integration integrity, not clinical truth. Directories are 2770;
files are 0660 at the CARC project mount boundary (NFS-exposed group 65534).

Entry point: `TriCompose-v1.2/slurm/57_radgraph_interface_existing_cpu.sh`.
It is **not an sbatch script** and binds the historical allocation and run ID;
do not rerun it into the existing output. New executions require a new opaque
run ID and a valid approved allocation. Worker/contract sources consumed by
this receipt remain unchanged; subsequent changes require a new version.

## What this enables, and what remains

Available: local frozen native graph extraction and official reference-based
report comparison. Not yet done: running the extractor on the historical
candidate bank, independent clinical qualification, qualified hard-contradiction
mapping, or modifying the selector/regenerator.

The fully synthetic cohort has no real reference report. Comparing two model
reports is **cross-path agreement**, not independent image factuality. A real
expert-annotated benchmark with known scope/annotation encoding is still needed
before promoting this to a clinical score or automatic regeneration trigger.
The prior official MIMIC pilot must not be relabelled an untouched new test,
and unresolved RadEvalX blank annotations must not silently become zero errors.
