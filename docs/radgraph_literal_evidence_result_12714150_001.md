# Native-observation evidence table — 2026-10-06

## Completed / 已完成

The historical **80 fixed EHR / 240 CXR-slot / 960 report-slot** bank now has
an append-only, source-bound native-observation diagnostic. It separates
explicit label opposition, unmentioned observations, uncertain/mixed states,
anatomy-context differences and modifier differences rather than collapsing
them into one scalar. All original scores, EHRs and winners remain unchanged.

已把细粒度对照实际接到全池，而不只是写一个空接口：428 份不同报告的
既有 RadGraph 全部处理成功，1,440 对同图报告均有对照结果，960 个候选
表格新增 15 列。这个表记录“哪些证据不同”，不替医生判断谁错了。

This follows the weak global-score expert diagnostics. See the
[fixed literal-evidence protocol](radgraph_literal_evidence_protocol.md).
It reuses cached **fully synthetic** graphs only: zero model/API/GPU calls,
zero new submissions, and no raw patient source, report file, EHR, image,
prompt or checkpoint reads. Construction took **0.546 CPU seconds** inside
existing Slurm allocation 12714150. This is not GPU inference time saved.

## Representation and scope

Use the native XL observation states with exact casefolded/whitespace-normalized
tokens plus native anatomy context. Observation modifiers and measurement
entities are not independent disease findings. Export hashes, native entity
IDs and word offsets, not report text or entity tokens. No disease dictionary,
synonym guesses, rule/weight fitting, No-Finding expansion or new trained scorer.

Positive–negative opposition is only a **proposal**: current-patient, temporal
and clinical scope remain unverified. Missing mention is unknown comparison,
not an explicit negative, hallucination or verified omission. Different
locations can coexist; different modifier token sets are not automatically a
severity error. Native agreement is not image truth or an independent vote.

## All measured differences / 实际对照数量

The unit below is an atom occurrence in a same-image report-pair comparison,
not an independent patient, image, or clinically adjudicated finding.

| Native comparison relation | Occurrences |
| --- | ---: |
| Same literal atom, native state agrees | 1,043 |
| Same literal atom, explicit positive/negative opposition proposal | 37 |
| Uncertain or mixed native states | 7 |
| Nondefinite native anatomy context | 0 |
| Unmentioned in the left report | 6,650 |
| Unmentioned in the right report | 9,302 |
| Total union atom comparisons | 17,039 |

Additional detail differences: **799** literal-concept anatomy-context
differences and **276** observation-modifier token-set differences. These
overlap the atom comparison accounting; do not add them as independent errors.

Exact lexical/context matching leaves **15,952 / 17,039** atom comparisons
unmentioned on one side. This is a major comparison-coverage limitation and
can reflect synonyms, extraction differences, report scope or content coverage.
It is **not** a 93.6% clinical omission rate, proof that reports are poor, or
evidence that all unmatched terms are incompatible. The left/right asymmetry
also depends on the fixed model ordering, not a favorable-versus-bad direction.

## Opposition lineage and dependence

| Denominator | Opposition proposals touch |
| --- | ---: |
| Same-image report-pair slots | 37 / 1,440 |
| CXR slots | 36 / 240 |
| Distinct CXR byte hashes | 31 / 147 |
| Synthetic EHR case slots | 35 / 80 |
| Report/triple candidate slots | 73 / 960 |

One pair can touch two candidate slots; mirrored counts are not extra evidence.
The 1,440 slots contain **891** distinct exact image/native-graph-pair bindings,
and those still share images, reports and prompts. No independence or clinical
confidence interval is claimed. Zero detected opposition is not acceptance.

| Same-image report-model pair | Pair slots | Slots with opposition proposals |
| --- | ---: | ---: |
| CheXagent-2 / CXRMate-single | 240 | 0 |
| CheXagent-2 / LLaVA-Rad | 240 | 2 |
| CheXagent-2 / MAIRA-2 | 240 | 0 |
| CXRMate-single / LLaVA-Rad | 240 | 34 |
| CXRMate-single / MAIRA-2 | 240 | 0 |
| LLaVA-Rad / MAIRA-2 | 240 | 1 |

These counts cannot identify the wrong report or rank models clinically. Low
shared-entity coverage can yield zero opposition; two agreeing reports may
share errors. CXRMate-single remains a CXR-only expert, not CXRMate-ED.

## Existing eight-head interface: coverage preflight, not inference

A separate **post-hoc, quote-free** preflight reads the eight existing finding
names directly from the frozen image-verifier source using AST, without
importing/loading that model. Compare their literal hashes with all 37
opposition proposals; do not add semantic aliases after seeing the result.

Only **2/37** proposals match an exact existing head: one edema and one
pneumonia, on two image slots. The other **35 remain unmatched** by this
strict linkage. This does **not** mean 35 clinically non-radiographic facts or
35 concepts definitely outside the model's semantic ability. Abbreviations,
partial mentions and paraphrases are deliberately not guessed into labels.
Location/modifier/current-patient scope is not verified by either exact match.

This prevents launching an eight-label verification job and silently claiming
it resolved every native-observation difference. All 37 remain unresolved and
no verification request or regeneration has been submitted. Next freeze and
validate a suitable semantic/scope linkage, or use a separately validated
broader verifier; do not manufacture coverage by adding case-specific aliases.

## Tables, receipts and checks

```text
artifacts/protected/tricompose_v1_2/
  radgraph_literal_evidence_runs/literal_pool960_12714150_001/
    candidate_score_table.csv
    graph_literal_evidence.json
    same_image_literal_evidence.json
    summary.json
    manifest.json
  radgraph_literal_evidence_audits/literal_pool960_12714150_001/audit.json
  radgraph_literal_head_scope_runs/scope37_12714150_001/
    records.json
    summary.json
    manifest.json
```

The candidate CSV retains **60 original columns and all 960 original rows**,
then appends 15 `rg_literal_*` fields. Measured counts are populated; unavailable
graphs would retain null counts rather than zero failures. `rg_literal_clinical_score`
remains null and clinical/scope/regeneration flags remain false. This is not
a silently revised selector or new best-output export.

- Literal-evidence manifest:
  `da1bfb5c82f1ce51d000843ee11471c6edddaafc20a5ab075227a4c675eb57c9`.
- Independent audit:
  `6c2d1b343b94dd710cdd6cf95f4b4e9f53bbd13d1528377b566125b47a24622e`.
  It replays 19 source/code/output hashes, 428 native graphs, 2,966 literal
  atoms, 3,063 source entity occurrences, all 1,440 pairs / 17,039 atom
  relations, all 72,000 original-plus-new CSV cells, and protected permissions.
- Separate exact-head preflight manifest:
  `84e4b5cddb3a5557b828315e581a441a1b56f4fcab6dad5c0f85b7c4f8b26032`.
  It retains all 37 proposals, pins the actual existing head source and passed
  reversed-pair deterministic replay. It is not part of the earlier literal-
  evidence audit or independent clinical annotation.
- 24 new literal-evidence fixtures and seven exact-head fixtures pass. They
  check source spans, missingness, uncertainty, native anatomy/measurement
  distinctions, modifier cycles, peer direction, lineage and refusal of
  clinical promotion. Full V1.2 suite: **2,436 tests pass** in **14.284 seconds**;
  Git whitespace checks pass and protected result tables remain ignored.
  Engineering checks do not establish clinical accuracy.

Keep detailed graphs, candidate paths and source-bound evidence protected;
do not publish synthetic report/graph text. This public note is aggregate-only.
