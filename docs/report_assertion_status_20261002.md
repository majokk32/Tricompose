# TriCompose: report-fact interface handoff / 报告事实接口进展

Status: development diagnostics, not validated clinical improvement.
No training, no new generation, no primary-score/winner replacement.

## Why this work is needed / 为什么先修评分输入

The eventual controller observes facts extracted from reports, not their true
clinical content. Incorrect negation or uncertainty extraction can therefore
send regeneration toward the wrong modality. A higher composition score does
not establish improvement if the underlying labels are wrong.

目前检查的是报告事实提取与证据范围，不是重新生成患者，也不是判断胸片
是否真实。先分清“报告文字说了什么”，再比较图像、EHR 与报告是否一致。

## Completed / 已完成

1. Fixed 56 wholly invented reports with 80 designated finding checks and four
   states. These are same-author annotations, not independent clinician gold.
2. Ran frozen CheXbert as approved P100 job `12580901`: 13 seconds, 1.227 GiB
   peak allocated VRAM, no truncation, no batch-8/batch-1 state differences.
3. Measured 48/80 authored-state matches, macro F1 0.6009, seven negative-to-
   positive flips, and 13 determinate outputs on uncertain/unknown references.
4. Implemented a reusable source-bound scope/abstention interface and applied
   it to the unchanged cache. It writes every row, including missing evidence.
5. Installed and ran pinned official medspaCy ConText in a new workspace
   environment, with no learned NLP model or modified official trigger.
6. Added a separate source readout using the unchanged pre-challenge literal
   rules; it does not overwrite model proposals or determine artifact errors.
7. Replayed the official parser, protected artifacts/hashes/modes, analyses
   and CSV bytes, and passed 298 lightweight invented-fixture tests. Original
   selection hashes remain unchanged.

The gate's development results are **30/80 scope commits, 36/80 abstentions,
14/80 missing model assertions**. Its 30/30 conditional authored-state match
does not mean 100% accuracy: 50 checks have no committed decision. The 32 raw
errors were not committed, but 18 raw matching states were also not committed.
The gate was designed after these authored results were seen. A separate,
predeclared clinical evaluation is still required.

## Reusable contract / 可复用的输出契约

Implementation: `TriCompose-v1.2/src/tricompose_v12/report_assertions.py`.

| Field | Meaning |
|---|---|
| `proposed_state` | Unchanged frozen extractor's original state |
| `state` | Same proposed state if retained; otherwise unknown |
| `decision` | `scope_commit`, `abstain`, or `no_model_assertion` |
| `evidence` | Exact original Unicode offsets, quote hash and evidence ID |
| `scope_verified` | Limited literal rule coverage, not clinical verification |
| `reason` | Traceable retention/abstention reason |
| `review_action` | Retain the assertion, verify more, or no comparable fact |
| `regeneration_authorized` | Always false in this development interface |

Unknown is never negative, a non-veto without scope coverage is not approval,
and a second parser cannot promote or flip the original proposal. Independent
code paths over the same report are not independent clinical votes. This
interface must not be confused with CXR-report factuality or an image truth label.

## Outputs / 输出位置

Below `artifacts/protected/tricompose_v1_2/`:

- `verification_runs/authored_assertions_chexbert_12580901/`: original frozen
  predictions and producer/checkpoint metadata.
- `verification_runs/authored_assertions_chexbert_12580901_analysis/`: four-state
  confusion matrices and per-family/per-finding authored diagnostics.
- `diagnostics/report_scope_gate_12580901_001/`: protected `evidence.json` and
  **224-row** `decision_table.csv` (56 reports × four findings).
- `diagnostics/report_scope_gate_analysis_12580901_001/`: the 80 designated-check
  risk/coverage results and per-finding breakdowns.
- `verification_runs/authored_assertions_context_12576792/` and its `_analysis`,
  `_gate`, `_gate_analysis` siblings: completed official parser and veto results.
- `diagnostics/report_extraction_sources_12576792_001/source_table.csv`: all
  224 rows of CheXbert, ConText and separate literal-source proposals.
- `diagnostics/report_extraction_sources_analysis_12576792_001/`: 80-check
  aggregate comparison and per-check diagnostic records.
- `task_runtime/report_context_12576792/execution_receipt.json`: existing-Slurm
  execution mode, resources, script hash and installation-report hash.

All output directories/files retain project-group `2770`/`0660`. No real source
EHR/report/image was opened. Actual generated report text is not copied here.

## Official parser execution / 官方解析器已运行

The complete `TriCompose-v1.2/slurm/20_report_context_cpu.sbatch` and standalone
request were shown: main, two CPUs, 8 GiB memory, 15-minute cap including first
dependency installation, no GPU. **No new `sbatch` was submitted.** Instead,
the current Codex session was verified to already be running in CPU Slurm
allocation `12576792`, on `b05-04`, with four CPUs and 32 GiB allocated. The
exact script was run with `bash` there, with two worker threads and an explicit
15-minute command timeout. It completed with exit code zero. Its `#SBATCH`
directives did not create another allocation. The earlier approval question
for a separate submission is obsolete for this completed execution.

It installs pinned official medspaCy into a **new** per-job environment, not an
existing model environment. It uses blank English tokenization, official PyRuSH
sentence rules and unchanged official English ConText rules, with no learned
NLP model or new checkpoint. See the [official medspaCy release](https://github.com/medspacy/medspacy/tree/1.3.1).
The shared literal four-finding vocabulary is disclosed; this is not a complete
clinical synonym recognizer. Unmodified mentions' default positive state is
explicitly unverified. Blind extraction, separate authored-key analysis and
second-parser veto/risk-coverage comparison all completed. The 56 reports were
processed twice in 2.363 seconds excluding installation, with no replay change
and 102 unchanged official ConText rules. No original model environment,
checkpoint or patient dataset was changed or used.

## Diagnostic results / 诊断结果

| Source, 80 designated checks | Authored-state matches | Macro F1 | Positive/negative flips | Unsafe determinate outputs on unknown/uncertain |
|---|---:|---:|---:|---:|
| Frozen CheXbert | 48/80 | 0.6009 | 7 | 13 |
| Official default ConText | 42/80 | 0.5224 | 14 | 24 |
| Frozen limited literal rules, separate readout | 69/80 | 0.8680 | 0 | 0 |

The separate literal readout covers 46/80 checks, with the other 34 unknown
(23 authored unknown matches and 11 uncovered non-unknown states). Its
underlying rules predate this challenge, but the readout/comparison wrapper
was designed after seeing these development cases. **These numbers are not
clinical accuracy, clinician review, held-out validation, or evidence that
generated triples improved.** All parsers use the same report and a shared
limited vocabulary, not independent patient evidence.

ConText adds no benefit as a veto here: retained coverage falls from 30/80
(37.5%) to 21/80 (26.25%), dropping nine additional matching states. It was not
promoted into primary selection. This limited default setup's result must not
be generalized to all ConText implementations or clinical reports.

There are 25 CheXbert/ConText disagreements and 16 covered literal proposals
that differ from CheXbert. These identify **extraction-review candidates**, not
verified errors in report content or images. Comparison rows explicitly keep
`report_content_error_established`, `image_error_established` and
`regeneration_authorized` false.

中文：现在真正补上的是“原标签 → 原文证据 → 范围覆盖 → 弃权／复核”的
接口，而不是把低分样本直接重生成。官方 ConText 在这一组挑战上没有帮忙，
我们保留完整结果、不调规则迎合答案，也没有改变已有最优 triple。

## Next validation gate / 下一步验证门槛

Keep code/rules and annotation policy frozen before an independent annotated
evaluation. Track abstention and class-specific coverage alongside errors.
Then propagate supported/contradicted/unknown evidence into candidate-level
comparisons without replacing the fixed EHR or treating correlated report
generators as independent votes. Only after independent localization and
false-repair checks should those records authorize targeted regeneration.
Equal-budget comparisons must include fixed path, static reranking and targeted
repair, with real generation/verification cost rather than cached replay time.

## Blinded generated-report review packet / 已生成报告的盲审接口

The next step now has an implemented metadata-only packet and readiness audit:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_20261002_001/
    reviewer/items.json
    reviewer/reviewer_a_template.json
    reviewer/reviewer_b_template.json
    reviewer/INSTRUCTIONS.md
    investigator/resolver.json
    investigator/frozen_predictions.json
    investigator/method_freeze.json
    summary.json
    manifest.json
  report_blind48_readiness_20261002_001/
    summary.json
    manifest.json
```

The complete fixed two-EHR pilot is retained: 12 CXRs × four report experts =
**48 reports**, with 48 distinct report hashes. Four findings give **192 rows
per reader**, or 384 slots for two independent human readers. This is not 48
independent patients or an untouched test cohort. No sample was chosen because
of its winner flag, low score, finding prevalence, or rule coverage.

The reviewer-facing files omit model/case/candidate IDs, source paths, model
predictions and winner flags. An opaque item maps to its source only in the
sibling investigator resolver. Blinding is procedural: authorized project
members could access the investigator files, and report writing style can
reveal the generator. Do not claim access-isolated or perfect model blinding.

`src/tricompose_v12/report_review.py` validates human annotations, exact Unicode
evidence spans/hashes, source identity, distinct reader aliases, complete row
inventories and independence attestations. Pending labels remain null; reviewed
unknown is a different status. Unassessable records retain their denominator.
An attestation is not externally verified reviewer independence or credentials.

`benchmarks/audit_blinded_report_review.py` can report reader agreement and
nominal four-state kappa, plus frozen CheXbert/Qwen/literal-rule comparison and
scope-gate risk/coverage against **provisional matching human states**. Conflicts
are not resolved with model labels; adjudication remains a separate human step.
A failed Qwen response cannot receive credit for a reviewed unknown. Agreement
between readers is not automatically clinical gold or image factuality.

The actual readiness audit has **0/192 jointly reviewed rows**. Extractor
accuracy, kappa and gate risk/coverage against human labels are unavailable,
not zero or perfect. No gold labels have been invented or copied from the
earlier authored challenge. All nine new packet/audit artifacts, source/code
bindings, protected modes/groups and exact replay were checked. **328 tests
pass**, including invented-fixture copying; original score/winner hashes remain
unchanged. The preparation/replay checks actively refused report-text and
image-file opens, confirming this step was metadata-only.

After the complete script and explicit synthetic-copy request were shown,
the user approved execution with `go`. The unchanged
`slurm/21_blinded_report_copies_existing_cpu.sh` **completed, exit zero**, inside
the current CPU Slurm allocation `12576792` on `b05-04`. There was no new
`sbatch`, GPU request or model inference. The existing allocation has four CPUs
and 32 GiB; the copy command used one worker thread and a 120-second timeout.
It completed in approximately 0.53 seconds of command wall time.

The reviewer-readable bundle is:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_text_12576792/
    reports/report_0000.txt ... reports/report_0047.txt
    items.json
    reviewer_a_template.json
    reviewer_b_template.json
    INSTRUCTIONS.md
    summary.json
    manifest.json
  report_blind48_text_readiness_12576792_001/
    summary.json
    manifest.json
```

All 48 synthetic reports are **byte-identical** to their source hashes; all
53 manifest-listed artifacts and protected modes/groups passed checks. No real
EHR/report/image or raw patient input was read, and no generated text enters
this document, public logs or Git. The original metadata packet remains
unchanged; its "copy not materialized" fields describe its historical stage.
The new copies are a separate immutable run, not an update to that packet.
Execution resources and hashes are recorded in
`task_runtime/blinded_report_copies_12576792/execution_receipt.json`.

The copied-template readiness audit also completed. Both readers still have
192 pending rows, with **zero human annotations**; all human-reference
extractor/gate scores and reader kappa remain unavailable. No existing score,
winner, rule, threshold or report content was changed.

中文：48 份合成报告的匿名正文已经落盘，可以直接打开 `reports/` 阅读。
两位标注者应各复制一份模板到新的 protected 返回目录再填写，不能改原始
run。下一步是独立人工标注及分歧裁决；这仍是两病例开发审核，不是论文级
临床验证，也不会据此自动返工或改已有最优结果。

## CSV review workflow / CSV 审核流程

The protected reading book and two blank CSV forms are now available:

```text
artifacts/protected/tricompose_v1_2/human_review/
  report_blind48_sheets_20261002_001/
    reports_for_review.md
    reader_a.csv
    reader_b.csv
    reader_a_identity.json
    reader_b_identity.json
    HOW_TO_REVIEW.md
    summary.json
    manifest.json
```

The book retains the original text inside inert code fences, with opaque
report IDs and no model labels/scores/winner flags. Text is not repeated in
this public document. The 48 reports still represent **two EHR cases**.
Each reader has 192 rows: four findings for every report. No labels are
prefilled, and neither a CSV nor matching evidence proves clinical truth.

`benchmarks/report_review_sheets.py --mode import` preserves the human's
status, four-state label and reason. It computes only verbatim quote offsets
and hashes, rejecting source changes, absent/paraphrased quotes, incomplete
row inventories, duplicate evidence and ambiguous occurrences. When a quote
occurs multiple times, the reader must specify a zero-based occurrence index.
The importer does not reinterpret the label, even if it looks implausible.
It uses the unchanged frozen `report_review.py` contract. CRLF inside a quoted
CSV cell is retained rather than silently normalized.

Both blank-form roundtrips and the readiness audit completed:

```text
report_blind48_sheet_roundtrip_a_20261002_001/
report_blind48_sheet_roundtrip_b_20261002_001/
report_blind48_sheet_readiness_20261002_001/
report_blind48_sheets_verification_20261002_001/
```

The exact book replay, 12 manifest-listed artifacts, source hashes, complete
pending imports, protected modes/groups, and unchanged score/winner hashes
passed checks. An overwrite attempt was refused. **356 tests pass**, including
invented human-completed CSV fixtures. Blank imports preceded the final
CRLF-only CLI fix; current-core pending replay is identical, and current CLI
CRLF behavior is separately tested on invented strings. These are interface
checks, not completed human review. There are still **0 human labels**,
192 pending rows per reader, no human-reference accuracy or kappa, no new
`sbatch`, inference, API call, selection change or regeneration.

### Copy and fill / 复制后填写

Do not edit the original book/forms/report copies. Each human should create a
different NEW protected return directory, with no access to model predictions
or winner flags during this review. For reader A, from the workspace root:

```bash
cd /project2/ruishanl_1185/inference_3mod
REVIEW_ROOT="$PWD/artifacts/protected/tricompose_v1_2/human_review"
FORMS="$REVIEW_ROOT/report_blind48_sheets_20261002_001"
RETURN_DIR="$REVIEW_ROOT/returns_reader_a_20261002_001"
umask 007
mkdir -m 2770 "$RETURN_DIR" &&
  cp -n "$FORMS/reader_a.csv" "$RETURN_DIR/reader.csv" &&
  cp -n "$FORMS/reader_a_identity.json" "$RETURN_DIR/identity.json"
```

Reader B uses a separate `returns_reader_b_20261002_001` directory and the B
templates. If a directory already exists, choose a new opaque name; never
overwrite an existing run. Read `reports_for_review.md` and `HOW_TO_REVIEW.md`
inside the protected folder, using an authorized local editor. Do not paste
report text into chat/public logs or an external API.

Fill `identity.json` with an opaque alias and your actual role:
`human_domain_annotator`, `clinician`, `radiologist`, or `non_expert`.
Set the three independence attestation fields truthfully. Model-generated
labels, seen model predictions or winner flags make that return ineligible
for this independent review. `non_expert` is allowed for a language review but
must not be presented as clinician/radiologist gold. Credentials/attestations
are not externally verified by the software.

For each CSV row, choose `reviewed` plus `positive`, `negative`, `uncertain`,
or `unknown`, and a permitted reason. Annotate the **current report assertion**,
not whether it matches an unseen image/EHR. Copy supporting quotes verbatim;
unknown/not-mentioned needs no fabricated quote. Leave unreviewed rows
`pending` with blank state/reason/evidence. Unassessable has null state and its
own reason. Keep every row and source ID/hash unchanged. Full details and the
two-quote-slot limit are in the protected `HOW_TO_REVIEW.md`.

### Import and audit / 导入与核验

These commands are **for actual human returns**, not another way to create
labels. Completed returns need the approved synthetic source verification
inside Slurm. Do not run them on a login node or submit an unapproved job.
Use an existing authorized CPU allocation, or first show a complete CPU batch
script/resource request and obtain approval for a new submission.

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="TriCompose-v1.2/benchmarks:TriCompose-v1.2/src:TriCompose-v1.0/eval/report_v1_1"
python TriCompose-v1.2/benchmarks/report_review_sheets.py --mode import \
  --bundle-run "$REVIEW_ROOT/report_blind48_text_12576792" \
  --sheet-file "$RETURN_DIR/reader.csv" \
  --identity-file "$RETURN_DIR/identity.json" \
  --output-root "$REVIEW_ROOT" \
  --run-id report_blind48_human_return_a_001
```

Repeat independently for B, with its return path and a distinct run ID.
Then audit the two imported files in the authorized CPU allocation:

```bash
python TriCompose-v1.2/benchmarks/audit_blinded_report_review.py \
  --packet-run "$REVIEW_ROOT/report_blind48_20261002_001" \
  --reader-a "$REVIEW_ROOT/report_blind48_human_return_a_001/reader_annotations.json" \
  --reader-b "$REVIEW_ROOT/report_blind48_human_return_b_001/reader_annotations.json" \
  --output-root "$REVIEW_ROOT" \
  --run-id report_blind48_human_audit_001
```

Reader agreement and frozen-extractor comparison use provisional matching
states only. Human disagreement remains unresolved until separate adjudication;
no model breaks ties. This report-only pilot cannot establish image errors or
authorize targeted regeneration. No source EHR, source image, real target or
external API is needed for this workflow.

中文：现在能直接读匿名报告、填 CSV，再导入核验。真正的下一步是两人
独立填写，不是再运行一个模型来替人填答案。当前只是接口已验证，人工
参考指标仍然没有；原来的 score table 和最优 triple 完全没动。

## Candidate scope table / 候选级原文证据可用性

A separate diagnostic table now connects the unchanged report syntax gate to
the fixed 48-candidate bank, without changing the primary scorer/selector:

```text
artifacts/protected/tricompose_v1_2/diagnostics/
  report_candidate_scope_12576792_001/
    candidate_scope_table.csv
    candidate_scope_table.json
    fact_scope_table.jsonl
    report_scope_assertions.json
    cross_path_groups.json
    summary.json, summary.md, manifest.json
  report_candidate_scope_verified_12576792_001/
    summary.json, manifest.json
```

The 384 facts retain all eight cached findings for all 48 candidates. The
source-bound syntax checker only supports four, so 192 facts are explicitly
outside its inventory. Of the other 192 assertions, 34 commit, 68 abstain and
90 have no model assertion. These are extraction/coverage statuses, not
independent human findings or clinical truth.

| Edge | Full denominator | Raw comparable | Scope-usable comparable |
|---|---:|---:|---:|
| EHR–CXR | 96 unique image/finding pairs | 12 | 12 |
| EHR–Report | 384 candidate/finding pairs | 11 | 0 |
| CXR–Report | 384 candidate/finding pairs | 146 | 34 |

Image evidence is deduplicated by byte hash; reports derived from one image
remain correlated. The EHR–CXR edge is unaffected by a report-only gate. The
direct EHR finding in both pilot cases is pneumonia, which this gate lacks:
EHR–Report becomes unavailable rather than being judged wrong. Its clinical
selection score remains null. Current four-finding human forms cannot validate
this uncovered edge; broader independently reviewed coverage needs a separate
protocol, not edits to the frozen method or old labels.

Raw/scoped CXR–Report support is 93/21 and opposition is 53/13. Fewer opposition
signals are **not** a measured reduction in clinical contradictions: 112
comparisons were withdrawn. Coverage falls from 146/384 to 34/384. The 12 raw
report-disagreement groups becoming zero after masking also establishes no
clinical improvement or majority-vote winner. No fault modality is assigned.

All seven artifacts and full replay passed, along with source/method hashes,
protected permissions and unchanged original score/winner hashes. **379 tests
pass** on invented fixtures. This CPU-only diagnostic ran inside allocation
12576792 with no new `sbatch`, inference/API call, source EHR/image/real-target
access, fabricated human label, ranking update or regeneration. The reviewer
book/forms remain unchanged and pending.

中文：现在这张表展示的是“旧标签还有多少能找到当前检查器认可的原文
证据”，不是新最优评分。肺炎尚未被这层检查覆盖，所以 EHR–Report 暂时
没有可用项；不能通过丢掉分母、乱补规则或将 unknown 当阴性解决。

## Official human-label source: metadata audit completed / 官方标签元数据审核完成

Filesystem metadata confirms a local file named
`mimic-cxr-2.1.0-test-set-labeled.csv` (23,759 bytes). The
[official MIMIC-CXR-JPG documentation](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)
describes this release's file as manually annotated report labels for 14
categories, including pneumonia. Filename and size alone do not prove local
release byte authenticity or report availability; release authenticity remains
unverified even after this audit.

After the complete wrapper was shown and the user explicitly approved it,
`TriCompose-v1.2/real_validation/audit_official_report_gold.py` executed inside
the existing CPU allocation `12576792` (four CPUs, 32 GiB allocated), with one
worker, a 120-second limit and 10-second termination grace. No GPU, new
`sbatch`, download, API or model call was used. The old MeDiM job was untouched.

The original attempt and safe-error-code retry rejected the source schema.
The schema-only diagnostic preserved that failure rather than relaxing label
values. It identified **Airspace Opacity**, whereas the original contract
expected **Lung Opacity**. The final adapter accepts either exact 15-column
source schema, keeps its original class names, and **does not map Airspace
Opacity to Lung Opacity**. In this source variant, the Lung Opacity common-head
mapping remains null. Frozen extraction rules and the candidate bank were not
retuned. Historical wrappers `22`–`24` preserve their old code pins and are
not current rerun commands.

The final `slurm/25_official_report_gold_source_schema_existing_cpu.sh` completed
the metadata audit in **0.745 seconds** of program runtime. A read-only replay
matched exactly in 0.721 seconds. It verified patient-disjoint linkage splits,
study/image deduplication, four-state denominators, artifact and source hashes,
project-group `2770`/`0660` modes, and unchanged original score/winner hashes.
Status: **completed_verified_metadata_only_not_model_validation**.

The approved job internally consumed the 14 label cells and study/patient
linkage metadata, testing only whether report-path fields are nonempty. It did
not open those report paths, separate source EHR files or image pixels. No
patient keys, source rows, paths or report excerpts were exported. Protected
outputs retain aggregate schemas, class counts, split/study denominators and
hashes; these patient-derived aggregates are not copied into this document.
Blank means unknown, CSV `0` negative and `-1` uncertain. Public output contains
sanitized status, runtime and hashes only.

Completed output:
`artifacts/protected/tricompose_v1_2/real_validation/official_report_gold_coverage_12576792_source1/`.
Artifacts: `summary.json`, `summary.md`, `manifest.json`.
The protected execution/verification receipt is in
`artifacts/protected/tricompose_v1_2/task_runtime/official_gold_metadata_12576792_source1/execution_receipt.json`.
Manifest SHA256:
`bf0b5064c5a477bc673196080e2c16b2dc7dc15672f6cf589161761ddfe77fc4`.

Sanitized readiness status is `linked_test_gold_metadata_available`: there is
test linkage with nonempty report-path metadata. File existence/content has
**not** been checked. No model accuracy, thresholds, winner, fabricated human
return, clinical fault attribution or repair action resulted.

The [official CheXbert README](https://github.com/stanfordmlgroup/CheXbert/blob/master/README.md)
discloses MIMIC radiologist-labeled data in its public checkpoint training.
Local checkpoint identity and patient overlap with this annotation source are
unverified. Human **report labels** also are not independent image truth or
gold evidence spans. A successful coverage audit therefore cannot establish
held-out clinical accuracy, clear the localization gate, or replace the
pending generated-report human review. Official test labels must not fit
thresholds/policies. Any later model benchmark needs separate approval.

Nineteen invented-CSV tests cover four-state mapping, source-name preservation,
14/eight-head coverage, deduplication, split rejection, missingness, safe error
codes/schema diagnostics, privacy and approval guards. **398 V1.2 tests pass**;
this is software validation, not clinical accuracy. Original report rules,
score table, winner and reviewer forms are unchanged. The next benchmark must
separately approve real-report consumption and frozen model execution, audit
training overlap and report-section/annotation policy alignment, and retain
all unavailable classes rather than force a complete score.

## Real-report CheXbert benchmark: completed / 真实报告评测完成

Implementation:

- `TriCompose-v1.2/real_validation/official_report_benchmark.py`: pure schema,
  section selection, four-state confusion/F1/missingness and scope summaries.
- `TriCompose-v1.2/real_validation/run_official_report_benchmark.py`: explicitly
  approved GPU Slurm consumption, unchanged frozen CheXbert and private output.
- `TriCompose-v1.2/slurm/26_official_report_chexbert_debug_p100.sbatch`: complete
  request shown and explicitly approved; submitted as **12594397**.

The approved request was debug, **one P100**, two CPUs, 16 GiB host RAM,
10 minutes maximum, batch eight, offline existing assets. Preparation-time
`noderes -f -g` reports available P100s including debug; the listed A40/A100
nodes are drain. This is not a reservation or a queue-time guarantee.
No inference, raw real-report access or dependency/model download occurred in
preparation. The earlier metadata permission did not authorize this new
report-text/model job; separate explicit approval was obtained before submission.
Slurm reports `COMPLETED`, exit `0:0`, on `e23-02`, with a 15-second allocation.
Program runtime was **12.025 seconds**; peak allocated VRAM including model
loading was **1.227 GiB**. Nonempty predictions were produced. Artifact hashes,
inventory denominators and protected modes passed the post-run read-only check.

Predeclared protocol before any real text/model result is observed:

1. Recheck the completed metadata-audit binding, source hashes and fixed model,
   tokenizer, four-finding scope-library and guard hashes.
2. Retain every annotation row with an opaque row-order item ID. Select **all**
   linked test studies with a unique nonempty path; deduplicate repeated image
   linkage. Do not select easy findings. Non-test/unlinked/missing/ambiguous
   entries retain reasons and denominators. The inference bound is 1,024 reports.
3. Use a declared **TriCompose conservative section adapter**, not the full
   official MIMIC parser: exactly one explicit nonempty uppercase Impression,
   Conclusion or combined Findings/Impression section. No history/full-report/
   findings fallback, patient-ID exception table or gold-guided sentence choice.
   Section-policy equivalence to the human source remains unverified. The
   [official MIMIC section utilities](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt)
   follow a broader protocol; do not claim an exact official reproduction.
4. Preserve unchanged local CheXbert preprocessing. Reject rather than silently
   truncate overlength sections. Keep all unavailable statuses; do not fabricate
   unknown predictions for unprocessed records.
5. Encode only report section text; human reference labels are never model
   inputs. Freeze/eval every parameter. Save 14 categorical predictions but
   evaluate four-state metrics only on aligned heads: the source's Airspace
   Opacity remains distinct from Lung Opacity, and binary No Finding is not a
   full four-state head. This source leaves 12 potential four-state heads and
   seven of the common eight; actual scored support is measured, not assumed.
6. Report per-state/per-finding confusion, annotated-state matches, F1 over
   reference-supported states, omissions, explicit positive/negative flips and
   determinate promotions on uncertain/unknown references. Unknown-only heads
   cannot inflate the pooled mean; their confusion/promotions remain visible.
   Categorical outputs do not support AUROC, Brier or ECE.
7. Run the unchanged four-finding syntax guard separately; commits can only
   retain the raw model's state. Report conditional match **with coverage**,
   never call lost comparisons corrected reports or independent image facts.
8. Replay the first eight eligible inputs individually, selected only by fixed
   source order, to diagnose batch sensitivity. Do not use gold states to pick
   replay examples. Preserve original primary predictions even if replay differs.

Completed protected output:
`artifacts/protected/tricompose_v1_2/real_validation/official_chexbert_reports_12594397/`
with `predictions.json`, `summary.json`, `summary.md`, `manifest.json`. No report
body/quote/span, original key/path, image, EHR field or per-row reference labels
are exported. Job-local caches/logs also remain protected with project modes.
Public stdout has sanitized status, runtime, VRAM and manifest hash only.
The completed manifest SHA256 is
`aeea0df50e73c4b57af35d3d686ee5d3677f1663174cf28a570205f5248deec6`.
Detailed per-finding metrics remain in the protected `summary.json`/`summary.md`;
do not copy real-report-derived reference/prediction rows into Git or chat.
The subsequent protected reader report and execution receipt are in
`artifacts/protected/tricompose_v1_2/real_validation/official_chexbert_review_12594397_001/`:
`RESULTS_CN_EN.md`, `execution_receipt.json`, `manifest.json`. They add coverage,
per-state precision/recall/F1, omissions, polarity-flip denominators and scope
commit coverage without changing the original run. Confusion-derived metrics
were independently recomputed; scope decisions retain original states, replay
inventory and declared-order Markdown match, and old winner/table hashes are
unchanged. This post-run check reads derived artifacts, not raw source reports.

This is a **report-label diagnostic, not an independent final clinical test**:
the public checkpoint's MIMIC training is disclosed, local annotation overlap
is unverified, manual section-policy alignment is unverified, and the analysis
unit is study/report, not independent patients or a patient-bootstrap result.
No threshold fitting, score-table replacement, fault assignment or regeneration
is enabled. No later primary policy may be fitted to this official test set.
Actual image/EHR correctness and pending synthetic-report human review remain
separate gates.

**421 V1.2 invented-fixture tests pass**, including 23 new benchmark tests.
Syntax and refusal-before-real-read guards pass. Current status:
`completed_real_report_diagnostic_no_primary_clinical_gate_cleared`.

### Connection to existing synthetic candidates / 接入现有候选诊断

The protected derived-only overlay is
`artifacts/protected/tricompose_v1_2/diagnostics/candidate_report_diagnostic_12594397_001/`:
`report_evidence_diagnostic.csv`, `summary.md`, `source_manifest.json`, `manifest.json`.
It retains 48 candidate triples from **two** EHR cases and all 384 candidate-
finding rows, connecting cached report states/scope decisions with the real-
report per-head diagnostic. No report text, image pixels or raw source patient
inputs are reopened; new model calls and GPU jobs are zero.

This is an evidence-availability overlay, **not** a new candidate score or
ranking. Candidate checkpoint/section equivalence and real-to-synthetic transfer
remain unverified. Dataset-level F1 must not become a candidate's score or a
router weight. Scope coverage is separate from raw-head evaluation: pneumonia
has a raw diagnostic but no frozen scope guard; Lung Opacity cannot borrow
Airspace Opacity metrics. Original scope records, score tables and winners stay
immutable. Primary clinical eligibility and regeneration authorization remain
false. Source/artifact hashes, unique row inventory and project permissions pass.

### Image/text fact-conflict probe / 图文事实极性检查

Prepared source: `TriCompose-v1.2/real_validation/biovil_fact_polarity.py`.
Prepared job: `TriCompose-v1.2/slurm/27_real_biovil_fact_polarity_debug_p100.sbatch`.
Frozen [protocol](biovil_fact_polarity_protocol.md): fixed existing 128+128
real CXRs, eight findings, three authored positive/negative sentence pairs;
all image/text scoring is label-blind. Cached report-derived references are
used only for aggregate diagnostics, never exported per row or called image truth.
No new raw source report/EHR text is read, no case is chosen by finding/score,
and no existing output/winner is overwritten.

The request is one debug P100, two CPUs, 16 GiB RAM and ten minutes maximum;
no submission or model inference occurred during preparation. **440 tests
pass**, including 19 new fixtures for negation direction, ties, missing support,
unknown/uncertain exclusion, always-positive bias and AUROC/polarity mismatch.
The complete script was shown and explicitly approved, then submitted as
**job 12597637**. Initial status was `RUNNING` on `e23-02`, no queue wait.
The submitted output is
`artifacts/protected/tricompose_v1_2/real_validation/real_biovil_fact_polarity_12597637/`;
It completed `0:0` in 49 seconds; program runtime 45.827 seconds, peak allocated
VRAM 0.576 GiB including model load. The original manifest SHA256 is
`5aeedee4f0aaaf8e5afcd534cc7cdf6ef44b0b4f9f0d544cedb2d5682c5e935c`.
All fixed cases, template inventories, reference bindings, derived statistics,
Markdown ordering, replay inventories, modes/group and original winner hashes
passed verification. Detailed polarity/coverage metrics and limitations are in
the protected `real_biovil_polarity_review_12597637_001/RESULTS_CN_EN.md` in the
same parent, with receipt/manifest. No original run was overwritten, no test-
guided template/weight/threshold fitting occurred, and no independent clinical
factuality, modality-localization or regeneration gate is cleared.
