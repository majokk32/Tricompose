# RSUA50 frozen Qwen image observer: result / 结果

The approved, unchanged script completed as Slurm job **12853380** on
2026-10-08, debug A40 `b11-09`, exit `0:0`, elapsed **2m41s**. The same frozen
checkpoint, image-only eight-finding prompt and original 50-image order were
used. No labels, EHR, reports or earlier scores were model inputs. Predictions
were fsynced before reference evaluation. No threshold, model, original winner
or clinical acceptance rule changed.

## Readout / 核心结果

The reference is the existing 25 published pneumonia / 25 paper-described
normal-cohort proxy split. The official release calls the latter Non-Covid;
this distinction is preserved, not converted into independent per-image
clinical adjudication. Only pneumonia has a reference. Results for the other
seven heads are unevaluated, not validated negatives.

| Frozen observer/readout | Positive-reference explicit support | Negative-reference explicit support | Explicit coverage | Correct explicit support / all 50 | Conditional support / explicit answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen image observer, unchanged | 2/25 (0.08) | 13/25 (0.52) | 34/50 (0.68) | 15/50 (0.30) | 15/34 (0.4412) |
| XRV default threshold 0.5, cached | 12/25 (0.48) | 19/25 (0.76) | 50/50 (1.00) | 31/50 (0.62) | 31/50 (0.62) |
| XRV transported threshold 0.55457607, cached | 6/25 (0.24) | 23/25 (0.92) | 50/50 (1.00) | 29/50 (0.58) | 29/50 (0.58) |

All methods use the same 50 source images; XRV was not called again. The
different abstention behavior matters. These support fractions are NOT
clinically adjudicated accuracy, nor evidence of a universally best scorer.

| Proxy reference group | Qwen positive | Qwen negative | Qwen uncertain | Qwen unknown | Failed/blocked/unattempted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published pneumonia, n=25 | 2 | 13 | 10 | 0 | 0 |
| Paper-described normal cohort, n=25 | 6 | 13 | 6 | 0 | 0 |

Explicit-only confusion: TP=2, FN=13, TN=13, FP=6. The other 16 slots remain
uncertain; they are not folded into FN, TN or negative predictions. Positive
group missingness bounds are 0.08–0.48; negative group bounds are 0.52–0.76.
Overall bounds are 0.30–0.62, NOT confidence intervals: the upper bound merely
allows every unresolved answer to be correct and is not an observed result.
Qwen supplied discrete states, not calibrated probabilities; its AUROC, AP,
Brier score and ECE are NA. No probabilities were invented for uncertainty.

Among the 34 explicit Qwen answers, agreement with cached XRV is 26/34 at
threshold 0.5 and 29/34 at the transported threshold. This agreement is not
clinical truth, and the observers must not become a majority-vote gold standard.

## Execution and independent metadata audit

- Actual observer call attempts: 50; actual generate attempts: 50; one model load.
- Complete responses: 50; runtime exceptions, unavailable responses and retries: zero.
- Worker interval through summary creation: 146.098s; Slurm elapsed: 161s.
- BF16; peak Torch-allocated VRAM: 15.594 GiB, not total GPU memory usage.
- New generation/planner/XRV/API calls: zero. All model weights remained frozen.

Output directory, relative to workspace:

```text
artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_runs/rsua_qwen50_12853380/
  predictions.json
  summary.json
  RESULTS_CN_EN.md
  calls.journal.jsonl
  manifest.json
```

Result manifest SHA256:
`5e4bd836c83d6210237a9cd581f88618de2ab7967b76a43317b9ba7990a3ea48`.
The original plan and protocol retain their pre-execution state; they were
not rewritten after outcomes were observed.

CPU allocation `12851223` verified all four result artifact hashes, 317 source
hashes, ten input artifact hashes, all 50 PNG byte hashes, seven model-asset
stats, the unchanged approved script hash, exact replay of 151 journal events,
prediction order, eight-state response contracts, mechanical guard receipts,
aggregate arithmetic and project-group permissions. No reference rows or image
pixels were opened by this postflight. It is metadata verification, not a rerun
of image inference, per-image reference assignment or clinical adjudication.

Audit directory:
`artifacts/protected/tricompose_v1_2/real_validation/rsua_qwen_audits/rsua_qwen50_12853380_12851223_001/`.
Audit manifest SHA256:
`fadfba8badd84e770d75313fa67968a76cdf3d5100e192bcd669a9bb55e90dd0`.
The pre-submission suite passed 3,926 tests, including 19 invented-fixture
tests for this worker. No new code or tests were introduced after the run;
this result updates aggregate documentation only.

## Decision for TriCompose / 对后续流程的影响

工程执行成功，但当前冻结 Qwen 观察器在该肺炎代理参考上的明确支持较弱，
不能用它直接判定 CXR 出错或推翻 XRV。这个结果也不证明 XRV 是临床真值：
参考本身是公开类别代理，且病例分组、年龄域、训练数据重叠尚未确认。

保留原有 clinical attribution guard、阈值和历史 winner；未知、uncertain、
不可用保持独立。不要因为本次结果不好而事后选择最佳 prompt 或阈值。
Qwen 的 numeric scheduling 与 image verification 是两个不同角色，本次只
检查后者，不能据此声称 planner 好坏、临床修复成功或节约了生成开销。

The next methodological work is to preserve scorer reliability/availability
in the scheduling state and test a separately frozen, medically grounded
image observer against an appropriate reference before granting clinical
repair authority. Any development-time change needs a new protocol/version
and independent evaluation; it must not be installed into this consumed run.
No next GPU run, model download or clinical repair is authorized by this report.

Protocol: `docs/rsua_qwen_observer_protocol.md`. The same limitations continue
to apply: development cohort, sparse pneumonia-only proxy labels, no evidence
of transfer to generated images, no LLM advantage, and no measured saved calls.
