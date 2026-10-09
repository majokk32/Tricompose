# Fresh-receipt output adapter: completed interface, not a new generation result

## 本轮完成了什么

Implemented the separately frozen `fresh_output_acceptance.py` callable for
fresh eight-enabled-head XRV/CheXbert completed receipts. The previous cached
fourteen-head output veto and all historical workers, thresholds, policies,
scores and selections remain unchanged. No new model inference, GPU task,
training, download, external API or clinical body/pixel/weight read occurred.

接口位置：`TriCompose-v1.2/tools/fresh_output_acceptance.py`。
协议：`docs/fresh_output_acceptance_protocol.md`。

The callable takes an initial baseline ID, proposed ID, an explicitly ordered
observed inventory, fixed-EHR/scorer context and a complete case cost ledger.
It verifies provenance, six disabled/unknown XRV heads, fourteen named states,
all three raw-edge denominators, same-artifact label identity and completed
generator/scorer dependencies. Cached references and actual new acquisitions
have distinct origin types; a self-digest is not source authentication.

Same-image and changed-image proposals use their existing, distinct frozen
predicates. Cross-image proposals must preserve both the initial reference and
the first passing *observed* same-image report reference, when different. No
unseen winner, alternate endpoint, learned scorer or new weighted score enters
the decision. Failed proposals retain the initial section-eligible reference;
an ineligible baseline returns no output. No cost refund, including failures
and pending attempts. It never declares a modality clinically faulty.

## Archived compatibility check / 已有回执兼容性检查

Authenticated the already completed and previously audited two-case bounded
retry DEVELOPMENT run. This is NOT new prospective inference, a new two-case
generation experiment, a sequential controller run or untouched holdout.
For each EHR, checked its observed same-image static proposal and its existing
completed image-retry proposal independently: four checks total, two EHRs.

| Check | Count | Outcome |
| --- | ---: | --- |
| Existing same-image proposals | 2 | One proxy-preserving report change; one unchanged |
| Existing completed image proposals | 2 | Both vetoed; zero passes |
| Original actual charged attempts | 8 | All retained, counted once per EHR ledger |
| New inference / GPU tasks | 0 / 0 | None |

这只证明接口能正确接收并检查已有新回执，不证明新生成质量提高。
两次换图失败是已知发展集结果，不能算本轮新发现或自然错误定位。
Four decision checks do not multiply the original eight attempts into sixteen.
The original retry selector's static fallback is untouched; this new adapter's
initial-reference fallback is a separately declared policy, not a rewrite of
the old result. Output eligibility is not anatomy validation or clinical truth.

## Verified software and immutable output

**1,683 tests passed in 9.895 seconds**, including 37 new invented-fixture
tests. Tests cover observation boundaries, endpoint isolation, fixed-EHR and
artifact/scorer identity, strict improvement versus silence, both-reference
preservation, disabled heads, provenance, structure risks, missingness and
charged failure/pending/no-refund behavior. Mocked journals are not clinical
truth or measured prospective GPU saving.

An independent metadata audit checked **262 source hashes**, three output
artifacts, exact replay of all four decisions, independently reconstructed raw
fact-set gates/edge arithmetic, eight original charged reservations, ledger
hash chains, project-private modes, overwrite refusal and unchanged previous
output-veto/headroom/scope-guard manifests. Source authentication here pins the
old audit and selected metadata, not a repeated scan of model or image bytes.

Protected output:

```text
artifacts/protected/tricompose_v1_2/fresh_output_acceptance/fresh_receipts_12654973_001/
  decisions.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`ecd4f197ccc2df8bee750203e6a49b787508b24513f1d14a1ec74734e98bb263`.
Existing CPU allocation 12654973; metadata smoke CPU wall time before
serialization **0.155397 seconds**, not model/GPU runtime.

## Remaining boundary / 下一步边界

The callable is ready for a separately reviewed new controller to invoke after
artifact authentication and completed scoring. It is **not installed into the
existing live GPU controller** and does not itself schedule or regenerate.
Next declare a small prospective execution policy, including the initial
reference versus incumbent/fallback choice, fixed EHRs/prompts, report ordering,
cost limits, completion/stop/rejection status and an equal-cost fixed/report-only
control. Do not repeat image retries solely because a higher similarity score
is desired. Inspect resources; show the complete actual Slurm script and request
and obtain subsequent explicit approval before submission. No new submission
is authorized or performed by this compatibility check.
