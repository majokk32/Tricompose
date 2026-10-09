# Frozen XraySigLIP probe result / 补充自动评分结果

The [fixed protocol](xraysiglip_probe_protocol.md), consumed worker, tests,
sealed plan, prompts, images and approved batch script remain unchanged.
The full script/resource request was displayed, then explicitly approved.
Job **12654030** completed with exit code 0 on debug P100 `e23-02` in
**36 seconds**. One GPU, two CPUs, 16 GiB RAM, ten-minute allocation cap;
worker runtime including preflight 20.531916 seconds, peak allocated VRAM
2.603 GiB. No A40-only allocation, generation, training or download.

## Scope and execution

- Six existing synthetic images, all eight heads and all three fixed
  presence/absence template pairs per head: **48 image/finding combinations**.
- Six reserved attempts, six successful joint forwards, one model load,
  zero failures and zero retries. Twelve journal events (reserve/completion).
- Four isolated uniform controls blocked before callback, with null scores
  and zero forwards. Ten unique inputs including those controls.
- **21 supplemental finding requests** across the six images, not 21 patients
  or 21 model calls. Predictions were fsynced before proxy-state comparisons.
- All EHRs, candidate images/reports, historical scores and winners unchanged.

## Text preference, not clinical labels

The sign of positive-minus-negative cosine is inspected for all three fixed
templates. A consistent sign is a text preference only. Mixed signs remain
`template_sensitive`; no best template is chosen and no threshold is fitted.
Clinical state and calibrated probability are null in every finding readout.

| Preference | All 48 image/finding combinations | 21 disputed finding requests |
|---|---:|---:|
| Presence text consistently higher | 6 | 0 |
| Absence text consistently higher | 17 | 7 |
| Template-sensitive direction | 25 | 14 |
| All templates tied | 0 | 0 |

Across the 21 disputed requests:

| Unqualified direction comparison | Current Qwen | Raw XRV |
|---|---:|---:|
| Same direction | 4 | 3 |
| Opposite direction | 3 | 4 |
| Not comparable (template-sensitive) | 14 | 14 |

These aggregate counts are correlated within six images and selected disputed
findings. They are not accuracy, clinical contradiction rates, confidence,
independent votes or population estimates. Across all 48 combinations the
three-template mean cosine margin ranges from -0.013891543 to 0.009547509;
its magnitude is not a calibrated disease probability.

**中文结论：接口与完整权重已成功运行，但 21 个争议 finding 中有 14 个会随
固定文本模板改变偏好方向。其余 7 个只稳定偏好否定文本，并不能证明否定
在临床上正确。这轮没有提供足以自动裁决 CXR／Report 谁错了的证据，不应
用多数投票或挑选模板把这些争议直接变成自动返工。**

XraySigLIP differs from Qwen/XRV but shares CheXagent-2's vision encoder;
training overlap and independence remain unverified. It cannot independently
validate that report generator. All **21 clinical requests stay unresolved**,
clinical accuracy stays null, and primary-metric/regeneration eligibility
remains false. The appropriate current use is an uncertainty sidecar, not a
clinical arbiter or a replacement primary scoring metric.

## Protected outputs and audit

```text
artifacts/protected/tricompose_v1_2/xraysiglip_runs/siglip_scope2_12654030/
  predictions.json
  attempt_journal.jsonl
  request_readout_comparisons.jsonl
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Run manifest SHA256:
`d846602fabba7abca85745e18fd5daba00832ac251ce458bd9f53ff3fa3c54eb`.
Approved script SHA256:
`b95a2976d5fdbe618351d7a4794d05dd90d0e9c879a8699bdb9bec41ecc6c246`.
Plan manifest SHA256:
`fbf4d8bd4e5bf93078533ef90f4e65a2d861a53cc2296269fd31570898be3f89`.

The approved GPU worker hashes image bytes and full weights, validates frozen
runtime/loading, and seals unchanged consumed sources before atomic commit.
Post-run CPU metadata audit rechecked **48 bounded source/config/runtime
hashes**, **five artifact hashes**, 21 exact request identities/dependencies,
all numerical scores, six attempts/forwards, four blocked controls, null
clinical scores and project permissions. Eleven recorded image/weight source
hashes remain GPU-worker attestations: the CPU audit did **not** reopen pixels
or weights or independently revalidate their contents. Protected directories
use 2770 and files 0660, project GID exposed as 65534 on this NFS mount.

The existing **1,474 synthetic-fixture tests**, including 29 probe tests,
passed before execution. No consumed source, test or protocol was modified
after sealing. Raw patient EHR, report or target image bodies were not opened,
and no external API was used. This result note contains only aggregate metadata.
