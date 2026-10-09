# First-version demonstration handoff / 首版可汇报交付

Completed 2026-10-06 in existing CPU Slurm allocation `12714150`, `b05-06`.
No training, model inference, GPU call, new Slurm submission, large download
or external patient-data API call. Original scores/selections/assets unchanged.

## Two ready-to-use private packages

Both are below:

```text
/project2/ruishanl_1185/inference_3mod/artifacts/protected/tricompose_v1_2/deliverables/
```

1. `first_version_12714150_001.tar.gz` — **294,038 bytes**. Metadata-only
   bilingual report, 80-case result index, 960-candidate path/hash/score index,
   full baseline tables and metric definitions. Read `START_HERE_CN_EN.md` first.
2. `first_version_samples_12714150_002.tar.gz` — **2,385,718 bytes**. Two existing
   synthetic cases selected strictly by opaque case integer order, not score
   or visual quality. Each contains the same synthetic EHR plus ordinary fixed
   and historical static-selected CXR/report files. Read `README_CN_EN.md` first.

The packages are for authorized project collaborators, not public Git or
external clinical APIs. They contain no raw MIMIC input, real CXR/report target,
checkpoint, runtime environment or credential. The second package contains
private **synthetic outputs** and must retain restricted handling.

## What was verified

- Exactly 80 fixed EHRs, 240 CXR slots, 960 report slots, 12 model combinations
  per EHR; all original 80 historical static choices preserved.
- 80 distinct EHR hashes, **147 distinct CXR hashes**, **428 distinct report
  hashes**. Candidate slots are not unique images or independent patients.
- Each CXR model has 80 slots and 49 distinct prompt/image hashes. This is the
  historical August one-seed bank; these counts are not evidence of clinical
  conditioning or a new October generation run.
- Direct comparable cached EHR evidence is available for 8/80 cases; 72/80
  remain in all comparisons, with unavailable EHR-edge ratios kept NA.
- All 10,000 score-free final draws reproduced exactly. All 2,000 EHR-level
  means, 75 method/cap/subgroup rows and 60 paired contrasts reproduced.
  Five budgets and every random seed are retained, not cherry-picked.
- Historical static and full-cap cached-static choices match on all 80 cases.
  Historical random acquisition + scored final selection is distinguished
  from true score-free final choice.
- 1,600 indexed synthetic artifact files were authenticated as **bytes only**.
  No EHR/fact/prompt/report payload was parsed or image decoded by the builder.
- Synthetic example export has 10 identical payload copies, all checked against
  the same immutable candidate index. No report text or image was displayed,
  summarized or clinically interpreted during copying/audit.
- Metadata bundle independent audit checked 3,073 source pins, 2,325 aggregate
  calculations, 13 files and the exact archive allowlist. Sample audit checked
  16 source pins, all 10 synthetic copies and 12 archive members.
- The full V1.2 test suite passed **2,276 tests in 14.067 seconds** after the
  exporter fix. Tests verify software/contracts, not clinical effectiveness.
- Output directories are `2770`, files/archives `0660`, project NFS group
  boundary (`nobody` metadata), excluded by Git.

The initial example export rejected the new 1,153,979-byte metadata manifest
against a generic 1-MiB limit before creating output. The new exporter alone
now allows a bounded 4-MiB delivery manifest with the **same exact SHA256**.
The source data, cohort, scores and historical reader were not modified.
Successful examples use fresh run `first_version_samples_12714150_002`.

## Scientific handoff

This is a **frozen-model engineering baseline and DEVELOPMENT comparison**.
The CXR models accept radiology text, not native structured EHR;
CXRMate-single is CXR-only, not CXRMate-ED. Static means proxy-selected,
not a proven clinically best synthetic patient.

Same-bank scores are not independent evaluation. BioViL-T is secondary global
similarity. Sparse EHR comparison and low coverage must remain visible beside
support/opposition. The ordinary fixed path is competitive on this secondary
readout; not every budget favors scoring. Cached call counts are simulated,
not measured GPU savings. The actual two-case regeneration accepted 0/2
replacements; independent report-metric alignment still awaits clarification
of blank annotation counts. Neither test counts nor hashes close those gaps.

## Entry points and immutable pins

```text
Builder:
TriCompose-v1.2/tools/build_first_version_delivery.py
Metadata audit:
TriCompose-v1.2/audits/audit_first_version_delivery.py
Example exporter:
TriCompose-v1.2/tools/export_first_version_samples.py
Example audit:
.tmp/audit_first_version_samples_12714150_002.py

Metadata manifest:
a5e670528b50d05e738b34b0ec213739e3a26d1022f648ac97d51e3c5a1ab7bc
Metadata archive:
e5538b3ff6537396c0e1fd2ed3ef41fc9df04996e0fdfdab95d5b0ebf9307413
Example manifest:
792ba9d648f7866207fc2491de309a609d7f28fb1d87743fcd9f06527dabeaa4
Example archive:
374c827cece49d03dd2c9ddcf6081f233ccbbc3b1fb615096ad8826e8cc97ace
Metadata audit source:
63526131d65438efea524bf2578318bc78fae99bce41ad6f36cbcaf203ddd8f0
Example audit source:
3f7715945a84c9231d9018d73d1a654a16e2cb6c81f0beceab7669cb5eac40dc
```

The builder completed its checks/serialization preparation in 8.025321 seconds;
this is metadata processing time, not model runtime. Neither immutable package
is edited to add later notes. Future evaluation produces a new companion run.
