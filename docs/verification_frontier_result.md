# Planning-only verification frontier / 验证动作表

Completed in existing CPU Slurm allocation `12645021`, with no new model call,
API, GPU task, training, download, or Slurm submission. Source EHR/report/image
bodies and real references were not opened. The [protocol](verification_frontier_protocol.md)
defines engineering investigation priorities, not a clinical quality score.

Use the validated protected run:

```text
artifacts/protected/tricompose_v1_2/verification_frontiers/frontier_pool960_12645021_validated/
  candidate_action_table.csv
  evidence_request_frontier.jsonl
  supplemental_image_evidence_requests.jsonl
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Manifest SHA256:
`204b499d2eeab6cd7f849ca854a8beefbc433a53876bb03b507a3aa56e0493bc`.

## What is now connected

The 80 fixed EHRs / 240 image slots / 960 triple candidates retain all original
139 CSV columns and their missingness. The original 2,072 logical requests
retain their states, reasons, dependency hashes and execution history.
Append-only fields show which evidence should be checked next, whether an
initial exact-image/report check is missing, and whether there is an existing
request ID for that requirement. No winner or EHR was changed.

| Additional independent-evidence requirement | Unique image/finding requests | Candidate/finding links |
|---|---:|---:|
| Changed same-image readout | 2 | 8 |
| Explicit current XRV/Qwen opposition | 19 | 76 |
| Total | 21 | 84 |

These 21 requests cover **six image slots**, not 21 patients or 84 independent
observations. The two changed findings belong to **one** image; each image
has four downstream report consumers. New supplemental requests remain
separate from original history even if an older request touches the same fact.
All are `not_executed`: they are neither calls nor confirmed clinical errors.
Same-checkpoint retries cannot satisfy the independent-evidence requirement.

Candidate-level priorities are 4 slots with changed-readout requirements,
20 with explicit scorer opposition, and 936 needing initial checks. Those
936 triple slots reuse 234 unchecked image slots. They are not assigned zero
quality, negative findings, or inferred clinical compatibility.
For 159 candidate slots the next requirement has no corresponding old request
ID; the action table explicitly surfaces that planning gap rather than
silently treating those slots as passed. No new initial-check executor is
authorized. There are 72/80 fixed EHRs without direct radiographic reference
facts under the existing evidence definition; this does not authorize changing
EHRs, adding disease/devices, or assuming normal images.

## What this does not establish

Clinical scores, cost estimates, and a declared execution budget remain null.
All clinical-selection, primary-metric, clinical-resolution, model-execution,
and regeneration flags remain false. Priority tiers are engineering conventions,
not calibrated severity, optimal compute allocation, or error-localization
accuracy. Stable agreement remains unqualified; unknown/uncertain and missing
report assertions cannot become negative or hard contradiction. Qwen image/text
readouts and same-image report experts are not independent votes.

Before adding a repair executor, obtain suitable independently validated
evidence for the flagged findings, with a separately reviewed script/resource
request if inference is needed. Do not re-generate simply to please the same
unqualified scorer. This frontier is development planning, not held-out
evaluation or proof of a quality improvement.

## Validation and revision provenance

All **1,445 V1.2 tests pass**, including 33 frontier tests using invented
metadata. Independent audit verified 22 consumed source hashes, five artifact
hashes, preservation of 133,440 original candidate cells and 68,376 original
request fields, deduplicated consumer links, and project-only 2770/0660 modes.

The earlier `frontier_pool960_12645021_001` write is retained as preliminary,
not the validated handoff: final regression uncovered two fixture operations
attempting to replace an element of an immutable tuple. Only those operations
were corrected to clear the contained list; production code and protocol did
not change. Its manifest
`edf990780e2330ffdf2022bf7ef8f62cf76c0845a2cafb32ba62d167cf6658b7`
records the old test hash, whose exact bytes are preserved in
`TriCompose-v1.2/source_snapshots/frontier_edf99078_fixture.py` (SHA256
`d1525a250f209793a66d38579a32cdef37538d10c23a113aceb7321ee2cbd40b`).
Do not claim that all of its source paths match the current corrected test.
All five preliminary data artifacts are byte-identical to the validated run;
the validated manifest pins the corrected tests. Neither run overwrote a
previous output or changed any original candidate evidence.
