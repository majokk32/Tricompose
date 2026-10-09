# Frozen RadGraph on the historical synthetic report bank

## Purpose

Add native entity/relation extraction and **same-image cross-path report
agreement** to all historical candidates, without changing any old score,
EHR, image, report, threshold, selection flag or winner. This is a development
diagnostic, not independent clinical qualification or targeted regeneration.

The source is the sealed protected first-version delivery. Its 80 fixed EHRs
have three text-conditioned CXR slots and four CXR-only report experts:
MAIRA-2, CXRMate-single, LLaVA-Rad and CheXagent-2. No native structured
EHR-to-image or genuine CXRMate-ED path is claimed. Only pinned synthetic report
artifacts are read for model computation; EHRs, prompts, image pixels and raw
MIMIC sources/real targets are not read by this worker.

## Frozen all-candidate inventory

- Source manifest SHA256:
  `a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc`.
- Source candidate index SHA256:
  `f8f7c7536b6db9af75cd3f58a0275ef2f28ec0dbca4885017eee4d6728138980`.
- All 960 slots are retained; no score-guided case selection.
- 428 distinct report byte hashes: one frozen native forward pass each at most.
- Per CXR: all six unordered model pairs. Across 240 image slots: 1,440 pairs.
- Every candidate attempts comparison with the other three same-image reports.
- Identical report bytes are explicitly counted, not independent corroboration.

Before model calls, the worker checks all source report byte hashes and writes
the complete plan with source/code/model/tokenizer hashes. Installed official
Python files must match the pinned clean upstream checkout. Inference is CPU
only inside the existing approved Slurm allocation, offline and with socket
connections disabled, using eval mode, frozen parameters and inference mode.
No new weights, GPU job, API, training or report generation is authorized here.

## New columns, not a replacement score

The official `compute_reward(..., 'all')` is applied to native graphs. The three
components remain separate: entity F1, relation-presence F1 and full-relation
F1. Raw pair values are preserved in `same_image_agreement.csv/json`.

`candidate_score_table.csv` retains every original column/value/row and appends:

- graph ID, extraction status, entity and relation counts;
- attempted peers (3), available peers, identical-text peers;
- three mean peer-agreement components, with explicit available denominators;
- `radgraph_agreement_clinical_qualified=false`.

These peer means are descriptive agreement, not a new clinical composite
score. No weights or thresholds are fitted. Empty reports and failed/missing
graphs receive explicit status and null values, never fabricated zero/unknown
labels. Valid zero rewards are retained. Long inputs are rejected rather than
silently truncated. The 11 native labels are preserved; anatomical/measurement
entities are not disease-positive counts, and entity absence is not negative.

Native graphs do not establish current-patient scope. All four report experts
depend on the same generated image: agreeing models can share a clinical error.
No real reference report exists for this fully synthetic cohort. Thus these
values are not reference-based clinical accuracy, CXR–Report correctness,
EHR–Report consistency, or evidence that the CXR is wrong. A laterality flip
already scored 0.75 in the interface smoke; global graph F1 alone cannot serve
as a hard contradiction or regeneration decision.

## Protected outputs and audit

```text
artifacts/protected/tricompose_v1_2/radgraph_bank_runs/
  native_pool960_12714150_001/
    frozen_plan.json
    candidate_score_table.csv
    report_graph_index.csv
    graph_receipts.json
    native_graphs.json
    same_image_agreement.csv
    same_image_agreement.json
    summary.json
    manifest.json
    worker.log
```

All files 0660 and directories 2770 at the project mount/group boundary.
Do not copy generated graph text to project README/Git/public logs/chat.
Public stdout contains only sanitized status, runtime/memory and hashes.

The separate audit checks source/code/output hashes, exact preservation of
the 960 original score rows, complete within-image pair identities, native
graph counts, recomputation of each reward component with unchanged official
code, peer means, null handling and permissions. This is an integrity audit,
not a substitute for expert clinical annotations.

Entry point: `TriCompose-v1.2/slurm/58_radgraph_bank_existing_cpu.sh`, **not
sbatch**. It binds allocation 12714150 and a new, non-overwriting output.
The wrapper has an 1,800-second timeout and uses two CPU threads. Future runs
must use fresh opaque IDs and valid approved Slurm allocations. Any new sbatch
requires complete script/resource review and explicit approval.
