# XraySigLIP alternate scorer — preparation record

Current status: the complete script/resources were subsequently displayed and
explicitly approved; job **12654030** completed on a debug P100 in 36 seconds.
See [the result](xraysiglip_probe_result.md). The preparation-stage record below
does not itself describe GPU execution. Do not resubmit the completed probe
without a fresh explicit submission approval.

The [fixed protocol](xraysiglip_probe_protocol.md) now has a concrete existing-
model worker, `TriCompose-v1.2/tools/probe_xraysiglip_findings.py`, and a sealed
metadata-only plan. It is not another claim of validated clinical scoring.
No inference, full weight hashing, image decoding, report/EHR body access,
model download, training, or new Slurm submission occurred during preparation.

Prepared using existing CPU allocation `12645021`:

```text
artifacts/protected/tricompose_v1_2/xraysiglip_plans/siglip_scope2_12645021_001/
  plan.json
  manifest.json
```

Manifest SHA256:
`fbf4d8bd4e5bf93078533ef90f4e65a2d861a53cc2296269fd31570898be3f89`.
All 45 consumed source/config/runtime/metadata hashes, the plan hash, exact
request/image references, asset statistics and 2770/0660 project modes pass.
**1,474 V1.2 tests pass**, including 29 invented-metadata probe tests.
Batch syntax check passed; GPU execution and numerical loading were pending
at the end of this preparation stage and have since completed as linked above.

## Exact scope and scientific limits

- Six unchanged original synthetic images and four isolated uniform controls,
  ten unique inputs / 18 historical logical slots.
- All eight findings and all three polarity template families per valid image:
  24 pairs / 48 generic text strings. No template/case selection by score.
- The existing checkpoint's actual processor/config is 512×512, float32;
  max-length text padding is 64, no truncation or randomly initialized head.
- At most six callback attempts / native joint image-text forwards, no retries;
  uniform controls are blocked before callback. Attempts, forwards and failed
  model loads are accounted separately in a fsynced protected journal.
- 21 existing disputed image/finding requests map to these six images, not 21
  calls or patients. Proxy comparisons occur after predictions are frozen.
- Raw cosine/logit pairs, three-template margin range and text preference are
  supplementary automatic readouts, not clinical labels/probabilities or truth.
- XraySigLIP is distinct from Qwen/XRV but shared with CheXagent-2; training
  overlap and evidence independence remain unverified. It cannot independently
  validate that report generator or clear a clinical repair/localization gate.
- All EHRs, old candidates, scores, winners and histories stay unchanged.

## Reviewed submission request (subsequently approved and completed)

Complete script: `TriCompose-v1.2/slurm/47_xraysiglip_findings_flexible_gpu.sbatch`.
SHA256: `b95a2976d5fdbe618351d7a4794d05dd90d0e9c879a8699bdb9bec41ecc6c246`.
Request one GPU from `p100|v100|a40|a100|l40s` on `debug,gpu`, two CPUs,
16 GiB host RAM, ten-minute allocation cap, no node binding. Runtime requires
at least 12 GiB usable VRAM and native float32/eager inference; no A40-only or
BF16/FlashAttention restriction. Queue wait is separate from this time cap.
The latest `noderes -f -g` snapshot showed an idle debug P100 node and a mixed
V100 node; availability can change and immediate start is not promised.

Only after complete-script/resource display and subsequent explicit approval:

```bash
sbatch TriCompose-v1.2/slurm/47_xraysiglip_findings_flexible_gpu.sbatch
```

The completed output is
`artifacts/protected/tricompose_v1_2/xraysiglip_runs/siglip_scope2_12654030/`.
Completed readouts do not imply solved clinical requests, clinical accuracy,
repairs, GPU savings or a new best triple.
