# Native + ConText scope88: interface complete, semantic qualification failed

## 本轮结论 / Outcome

已完成通用语境接口及词干字符边界适配。V2 可以处理全部 88 条自编
development probes，但**不能把同源 parser agreement 当作正确临床判断**：
保留的 39 条 signed 判断只有 8 条符合预先保存的任务参考，31 条不符合。
因此不接入临床主评分、选优决策或自动重新生成。

This is an engineering and authored task diagnostic, not a clinical-gold
benchmark, corrected triple cohort, or evidence that the generators failed.
The selected templates stress particular scope limitations; these rates are
not estimates of performance on representative radiology reports.

## 做了什么 / Implementation

1. Official frozen CheXpert/NegBio produces its unchanged native 14-category
   labels and exact annotation inventory. No Finding remains aggregate-only;
   it does not negate unmentioned diseases.
2. Frozen medspaCy ConText processes those same full cleaned source strings and
   native mention ranges. History, family, hypothetical, uncertainty and
   polarity disagreement can veto a signed comparison. A veto does not flip a
   label or manufacture a correct unknown.
3. All 14 categories are represented in the interface. The diagnostic itself
   covers eight findings: lung opacity, pneumonia, edema, cardiomegaly, pleural
   effusion, pneumothorax, atelectasis and consolidation. It does not validate
   the other six categories.
4. Both parsers see the same report. Their agreement is correlated evidence,
   not two independent clinical votes. Anatomy, qualified absence and
   current-patient scope are not established by absence of a context flag.

New immutable implementation files:

```text
TriCompose-v1.2/src/tricompose_v12/native_context_scope.py
TriCompose-v1.2/src/tricompose_v12/native_context_scope_v2.py
TriCompose-v1.2/benchmarks/native_context_scope_controls_v1.py
TriCompose-v1.2/interfaces/native_scope88_legacy_worker_v1.py
TriCompose-v1.2/tools/benchmark_native_context_scope.py
TriCompose-v1.2/tools/benchmark_native_context_scope_v2.py
TriCompose-v1.2/tests/test_native_context_scope.py
TriCompose-v1.2/tests/test_native_context_scope_v2.py
docs/native_context_scope_protocol.md
docs/native_context_scope_v2_protocol.md
```

Existing consumed source files, rules, model parameters and outputs were not
edited. No new trained scorer/router, phrase rules or fitted threshold.

## V1 → V2：修复对齐，不是修复临床语义

V1 had 38/88 unavailable ConText results. The native vocabulary contains stem
matches, including `opaci`, `atelecta` and `consolidat`, whose character ranges
can end inside a spaCy token. Strict whole-token alignment therefore failed.
These failures were unavailable evidence, not unknown findings or successful
anatomy rejection.

V2 splits only an unparsed Doc's token boundaries at existing native character
offsets. Substrings concatenate to the original token; text, SHA256, annotation
IDs and native ranges remain unchanged. It uses the documented
[spaCy retokenizer split API](https://spacy.io/api/doc#retokenizer.split).
Default token boundaries explicitly change, so scope/window behavior can
change even though the original text and 102 official context rules do not.

There were 41 token splits across 38 texts. V2 replayed the exact cached native
predictions: **zero new native parser calls**. V1 results were already known;
this is an engineering replay, not a fresh held-out test.

## 结果 / All-attempted diagnostic

The 88 inputs comprise 80 finding-by-scope probes plus eight nonpulmonary
opacity probes. All are wholly authored, with 88 distinct source hashes and
no exact duplicate of the prior authored112 inputs. Ten scope families cover
current presence/absence, uncertainty, history, family, hypothesis, resolution,
qualified absence, change-only language and opposed current assertions.

Reference semantics are investigator-authored task conventions. In particular,
qualifier/change-only probes use a conservative non-signed target. They are
not clinician-reviewed natural reports or an independent clinical truth set.

| Diagnostic | V1 strict token alignment | V2 character-boundary bridge |
| --- | ---: | ---: |
| Attempted inputs | 88 | 88 |
| Native complete | 88 | 88 (cached) |
| Context complete | 50 | 88 |
| Context unavailable | 38 | 0 |
| Raw native four-state matches | 8/88 | 8/88 |
| Raw signed proposals | 87 | 87 |
| Raw signed proposals not matching reference | 79 | 79 |
| Soft signed judgments retained | 20 | 39 |
| Retained matches | 5 | 8 |
| Retained non-matches | 15 | 31 |
| Incorrect signed proposals withheld | 64 | 48 |
| Correct signed proposals withheld | 3 | 0 |
| Soft coverage of all attempted inputs | 22.73% | 44.32% |
| Match fraction conditional on retention | 25.00% | 20.51% |
| Clinical qualification / hard-action eligibility | No / 0 | No / 0 |

Definitions: retained means the unchanged signed native proposal survived the
veto policy; correct means it matches the authored target, not clinical truth.
Coverage is retained/88. Conditional match is retained matches/retained. A
withheld result is **not** counted as a correct unknown. Overall gated clinical
accuracy remains null.

Restoring availability exposes additional wrong agreements. Do not present
50→88 context availability or 5→8 retained matches as an improvement in clinical
accuracy while hiding the rise from 15→31 retained non-matches.

### V2 by family

| Family | Inputs | Raw matches | Retained | Retained matches | Retained non-matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current presence | 8 | 8 | 8 | 8 | 0 |
| Current absence | 8 | 0 | 0 | 0 | 0 |
| Uncertain current | 8 | 0 | 8 | 0 | 8 |
| Historical only | 8 | 0 | 0 | 0 | 0 |
| Family only | 8 | 0 | 0 | 0 | 0 |
| Hypothetical only | 8 | 0 | 8 | 0 | 8 |
| Resolved only | 8 | 0 | 1 | 0 | 1 |
| Qualified absence | 8 | 0 | 7 | 0 | 7 |
| Change only | 8 | 0 | 0 | 0 | 0 |
| Opposed current | 8 | 0 | 0 | 0 | 0 |
| Nonpulmonary opacity | 8 | 0 | 7 | 0 | 7 |

History/family and some disagreements can be withheld. But vetoing the current
absence rows does not recover correct negative labels; hypothetical/uncertain,
nonpulmonary and qualified descriptions still produce wrong retained judgments.
Neither unmarked positive nor same-source agreement verifies anatomy or scope.

## 实际执行与独立核验

Executed inside existing CPU Slurm allocation **12714150**, node `b05-06`,
partition `main`, allocated 4 CPUs / 32 GB / 9-hour limit. No new submission,
GPU job, training, model download or external API.

- V1 native initialization: 5.767188 s; 176 native passes (88 primary + 88
  replay): 4.244358 s. Context plus gate primary/replay phase: 0.170496 s.
- V2 new native passes: 0. Context plus gate primary/replay: 0.298617 s.
  Historical native timing is not billed as newly executed V2 inference.
- Native, context and gate full-evidence replays: zero changed outputs.
- New interface tests: 32 V1 + 9 V2 = 41. Full V1.2 regression rerun:
  **2,197 tests passed in 14.074 s**; one existing Click deprecation warning.
- Independent standard-library audit, importing neither parser nor interface:
  verified **2,200 unique source pins**, four sealed manifests, 25 artifact
  hashes and 33 protected filesystem entries. Rebuilt all 88 scored rows,
  14-category veto decisions, flag mappings, per-family/per-finding statistics,
  source spans, bridge offsets, prediction receipts and replay equalities.
- V1/V2 input/reference bytes and cached native prediction bytes are identical.
  Official assets/rule dictionary identity is unchanged. The original candidate
  bank manifest hash is unchanged.
- Protected directories use 2770, files 0660, project group boundary only
  (NFS exposes group metadata as `nobody`). Protected artifacts and local audit
  are ignored by Git. No source patient input or candidate report body opened.

Independent read-only audit:

```text
.tmp/audit_native_context_scope_12714150_001.py
SHA256 e86d26038a775296246d494e4567ce0bcd0ee557354b4185d69d16708d7cc022
```

The prediction receipts distinguish reference-byte hashing for provenance from
decoding reference semantics. Predictions were saved/fsynced/hashed before
reference semantics were decoded for evaluation; authored target knowledge by
the developer is explicit. These receipts are not claims of clinical blinding.

## 输出与不可变 provenance

Paths below are relative to `/project2/ruishanl_1185/inference_3mod`.

```text
artifacts/protected/tricompose_v1_2/native_context_scope_plans/scope88_12714150_001/
  manifest SHA256 7896582f63978b0954e863444a11d0bc023a0fd90abdcd2fc639e512a8c95771

artifacts/protected/tricompose_v1_2/native_context_scope_runs/scope88_12714150_001/
  manifest SHA256 ab1e93abb349b28b2b7168bbe0d213f67b5950aa95f17dee7b07c310790267dc

artifacts/protected/tricompose_v1_2/native_context_scope_plans/scope88_bridge_12714150_001/
  manifest SHA256 42901d05dd1744bfd6837448933395b70990030ea30187dd40a2259c0bd73d3d

artifacts/protected/tricompose_v1_2/native_context_scope_runs/scope88_bridge_12714150_001/
  manifest SHA256 7f176ebfb8cd1f123fd1af15a24d51df04d296c249e9680b7da902fb81a831ed
```

Each run contains `summary.json`, `scored_checks.json`, source-bound native and
context evidence, `scope_gates.json`, replay/phase receipts and
`RESULTS_CN_EN.md`. V2 native prediction SHA256:
`2536f114efc82b164748630c26553c624d2fe1795986809659e9e99a56bb80b3`.

Original bank metadata anchor remains
`complete_bank_endpoints/complete_bank_secondary_12624822_001/manifest.json`,
SHA256 `ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`.
No old labels, scores, selected triples, gates or EHR facts were overwritten.

## 下一步 / Decision

1. Keep this interface as an explicit availability/veto sidecar only. Do not
   convert its soft retention into a validated score, hard conflict, winner
   change or regeneration trigger.
2. Inspect an already-frozen alternative report verifier and locate an
   independently labeled report benchmark. Prepare a versioned evaluation
   contract with scope/anatomy/uncertainty tests, calibration/evaluation split,
   coverage and retained-error reporting. Do not obtain apparent improvement
   by patching phrases against these now-known 88 templates.
3. Any patient-level computation stays inside an approved Slurm job with
   aggregate/sanitized output. A new batch job still requires showing its full
   script/resources and explicit approval. Only after independent qualification
   should report judgments affect error localization and targeted regeneration.

Until then: `clinical_qualified=false`, `clinical_score=null`,
`hard_action_eligible=false`, `selection_changed=false`,
`regeneration_authorized=false`.
