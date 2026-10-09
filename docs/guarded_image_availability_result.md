# Guarded image readouts attached to the full-bank score table

The [frozen CPU handoff protocol](guarded_image_availability_protocol.md) is
completed using the existing Slurm CPU allocation **12645021**. This consumes
the [completed guarded run](guarded_image_findings_result.md), not a new model
call. Old candidate scores, gate/image states, EHR anchors, winner records and
request histories remain immutable. New `liveimage_` fields are diagnostic
availability, not calibrated clinical selection scores.

## Scope and output / 范围与输出

Protected output, relative to `artifacts/protected/tricompose_v1_2/`:
`guarded_image_availability/liveimage_pool960_12645021_001/`.

Start with `candidate_score_table.csv`. It has **960 rows / 139 columns**:
105 unchanged source columns plus 34 appended fields. Per-finding metadata,
the deduplicated image/finding table, summary, bilingual note and manifest are
alongside it. No report/EHR/image body, real reference, checkpoint or response
text was opened; no new model/API/GPU/download/training/submission occurred.

| Inventory | Total | With fresh guarded readout | Unchecked |
|---|---:|---:|---:|
| Synthetic EHR cases | 80 | 2 represented by checked images | Remaining cases have no new image check |
| CXR candidate slots | 240 | 6 | 234 |
| Triple slots | 960 | 24 | 936 |
| Fact inventory rows | 13,440 | 336 rows in checked image slots | 13,104 |

Each checked image has four report candidates. The 24 slots are correlated
uses of six image responses, not 24 image calls. Among the 336 finding rows,
192 uses fall within the eight image heads and 144 are outside scope. The
deduplicated sidecar has **84 rows = six images × 14 inventory findings**:
**48** supported readout slots and **36** outside-scope slots. Four uniform
controls are excluded from all patient candidate joins.

## Reading the new fields / 新字段

- `liveimage_status`: checked_clinically_unqualified, not_checked or unavailable.
- `liveimage_guard_status` and receipt/run hashes: exact mechanical-check lineage,
  not anatomical or clinical acceptance.
- `liveimage_state_<finding>`: the fresh named four-state readout, only for the
  eight supported heads. Unchecked/unavailable is empty/null, not a negative.
- `liveimage_repeat_readout_slots`, `liveimage_changed_readout_count`: technical
  comparison with the old fixed readout, not clinical error counts.
- `liveimage_xrv_*_count` and `liveimage_report_*_count`: unchanged four-state
  relation definitions applied to fresh image states and existing exact facts.
  These remain unqualified scorer relations, not independent truth.

Old states remain in the original `imageverify_` metadata. Per-finding sidecars
show baseline/fresh states separately; the original EHR and report facts are
not re-extracted. Exact CXR ID AND encoded hash plus case/EHR/facts lineage are
required for joins. A shared image/text hash alone cannot expand checked scope.
For all 936 unchecked candidates, fresh states, counts and source hashes stay
null; absence of checking is not zero errors.

## Readout changes and scorer relations / 变化与评分器关系

Two unique finding readouts changed **positive -> uncertain**: consolidation
and pneumonia. Both belong to **one image**, linked to **four triples**.
They therefore produce **eight repeated candidate/finding occurrences**, not
eight independent clinical errors. Unique readout matching remains **46/48**.
The cause of those changes is not established. They withdraw certainty and
must not become hard negatives, confirmed modality errors or repair triggers.

| Unique XRV/Qwen relation | Old readout | Fresh guarded readout |
|---|---:|---:|
| Explicit agreement, unqualified | 25 | 25 |
| Explicit polarity opposition, unqualified | 21 | 19 |
| Uncertainty, not comparable | 2 | 4 |
| Outside the eight image heads | 36 | 36 |

The two fewer explicit oppositions result from uncertain readouts, not corrected
images or improved clinical accuracy. Do not report this as a consistency gain
without accounting for abstention/comparable coverage. The 17 retained report
assertion occurrences still explicitly match the fresh states; this uses
correlated reports and the same Qwen image/text checkpoint, not independent
clinical confirmation. No clinical request is resolved or ranking changed.

本轮完成的是新证据与原 score table 的对齐。旧分数保留，新旧读数并列，
未检查、不确定和明确阴性分开表示。不会用“少了两个明确分歧”声称修复成功。

## Verification and limits / 验证与边界

Run manifest SHA256:
`0120146adc043b482f51ef515d70ad3dd1b78132c3b989b22e7bd918b3c68dcb`.

**1,412 tests passed**, including 27 new invented-metadata tests. Independent
checks passed **27 source hashes / five artifact hashes**, exact scope and
joins, isolated controls, 48-state readout accounting, all new relations,
candidate/finding repetition denominators, original **100,800 candidate cells**
and **551,040 fact cells** with row order preserved, unchanged parent metadata
and project-only permissions. Directories 2770/files 0660; a fresh run was
committed atomically without overwrite. Consumed code/tests/protocol and runs
are now sealed; documentation does not change their hashes.

The source GPU run made six model calls and blocked four controls; **this CPU
handoff made zero calls and submitted zero jobs**. Clinical accuracy stays
null; primary-metric eligibility, selection changes and regeneration permission
remain false. No EHR-edge gap, clinical calibration, modality localization or
repair success is solved by this availability table. Further checking must
preserve these boundaries rather than manufacture a numeric clinical score.
