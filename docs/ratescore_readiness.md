# RaTEScore: proposed semantic report benchmark, not clinical qualification

Status (2026-10-06): **do not deploy the BioLORD/default RaTEScore route**.
The user explicitly reports no UMLS/SNOMED CT license. Source/model metadata
were inspected and preparation scripts exist, but remain unexecuted.
**Not downloaded, installed, submitted or run.**
Do not describe this as a completed benchmark, working checkpoint or improved
TriCompose score. Historical code, scores, selected triples and runs remain intact.

## Why inspect this component?

The literal RadGraph comparison preserves exact native entities but does not
link synonymous descriptions. The existing 37 polarity-opposition proposals are
not clinically verified errors; only two link literally to the current eight
image-verifier head names. Adding observed-bank aliases would not establish
semantic correctness or solve assertion scope.

[RaTEScore (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.836/) is an
existing entity-aware report similarity metric. The
[official implementation](https://github.com/MAGIC-AI4Med/RaTEScore) uses a
released medical NER model and BioLORD-2023-C for entity similarity. We propose
testing that frozen implementation rather than developing or training a new
scorer. Its performance on our benchmark is **unknown**.

This addresses a semantic-matching hypothesis, not the whole localization
problem. It requires two texts, has no image/EHR input, and cannot independently
identify the faulty report or image. Comparing two generated reports is
cross-path agreement, not clinical factuality. The four report generators
remain CXR-dependent, so agreement is not an independent image-truth vote.

## Fixed source and checkpoints

| Component | Fixed revision | Selected weight bytes |
| --- | --- | ---: |
| MAGIC-AI4Med/RaTEScore | `a970406b6f4155f7c7a9cb0d29b4fe94437e3361` | source only |
| Angelakeke/RaTE-NER-Deberta | `66d4b8189374f9b137e1b8633951ca4b4d758673` | 735,384,420 |
| FremyCompany/BioLORD-2023-C | `4bb5abd13930002136e3a5750b84474763c775c5` | 437,971,872 |

Weights total **1,173,356,292 bytes** (1.17 GB decimal / 1.09 GiB); tokenizer,
configuration and source add up to **1,184,749,604 asset bytes**. The plan caps
assets at 1.30 GB. New Python dependencies/cache/storage are separate; reserve
approximately 8–10 GiB as a planning allowance, not measured installation size.
No duplicate `.bin` weights or clinical datasets are requested.

Small source/configuration files were retrieved in memory for metadata/hash
inspection; model weights were not fetched. Weight SHA256 values come from
the public HF LFS metadata; small-file hashes were independently computed from
the pinned bytes. `configs/ratescore_assets_v1.json` preserves both.

### Licensing is a real prerequisite

The NER card marks its checkpoint MIT. The
[BioLORD model author](https://huggingface.co/FremyCompany/BioLORD-2023-C#license)
requires users to ensure appropriate **UMLS and SNOMED CT licensing** before
using this model. Being able to download an ungated model does not prove this
requirement is satisfied. MIMIC authorization is not proof of these separate
licenses. The project/user must confirm this prerequisite; the script refuses
acquisition without that confirmation. It never inspects license credentials,
HF tokens, `.netrc`, Git credentials or home authentication files.

## Official API details that must stay explicit

- Five native entity types: anatomy, abnormality, non-abnormality, disease,
  non-disease. These are **not** a validated current-patient four-state finding
  contract; no native uncertainty/history/device/laterality contract is proved.
- The pinned code uses BioLORD **CLS**, with a 30-token entity limit. Its
  model-card example uses mean pooling. Replacing official CLS with mean
  pooling would be a different method, not faithful official RaTEScore.
- It has released `long`/`short` affinity matrices. Use default **long** as the
  declared primary diagnostic and short only as a separately reported
  sensitivity. No local fitting, matrix selection by test scores or new weights.
- The official NER tokenizer truncates sentences at 512 tokens. Record token
  limits and truncation, not silently treat a truncated input as complete.
- Official empty-entity pairs return **0.5**. Preserve that native result in
  an audit field, but mark it non-comparable/null for clinical interpretation.
  Empty input, failed inference, unmentioned disease and all-normal are distinct.
- The source splits sentences with medspaCy and includes an enable-name typo
  (`medspacy_conte`) in one module. Do not patch it or claim the old ConText
  scope failures are now solved; first test the pinned dependency behavior.
- Original NER labels/strings and vectors, if subsequently produced on clinical
  inputs, must stay protected. Exact quote/span traceability and current scope
  still need separate validation. Do not let a matching cosine authorize repair.

## Prepared deployment only

**This proposal is now ineligible:** the user reports the required licenses
are absent. The script below is retained as preparation history, not an
executable recommendation. Do not set its license-confirmation variable or
pass `--license-confirmed` on the user's behalf. No deployment has occurred.

```text
TriCompose-v1.2/configs/ratescore_assets_v1.json
TriCompose-v1.2/configs/ratescore_runtime_v1.txt
TriCompose-v1.2/tools/prepare_ratescore_assets.py
TriCompose-v1.2/tools/seal_ratescore_setup.py
TriCompose-v1.2/slurm/61_ratescore_setup_cpu.sbatch
TriCompose-v1.2/tests/test_ratescore_setup.py
```

The proposed setup job: **main partition, 1 node, 1 task, 4 CPUs, 16G RAM,
1-hour cap, zero GPUs**. This is a deployment cap, not a queue/start or runtime
prediction. No reason to request A40 for acquisition. It downloads only the
pinned public assets and installs dependencies in a fresh workspace environment;
it does **zero inference**, reads zero clinical reports/images/EHRs, performs
no training and does not submit a child job. Source bytes are unchanged.
Top-level compatibility versions are pinned; transitive dependencies are
snapshotted after install, not falsely described as an upstream exact lock.

After explicit approval of the fully displayed batch script, model downloads,
environment installation and licensing confirmation, the submission form is:

```bash
sbatch --export=ALL,TRICOMPOSE_APPROVE_DOWNLOADS=1,TRICOMPOSE_CONFIRM_BIOLORD_LICENSE=1 \
  TriCompose-v1.2/slurm/61_ratescore_setup_cpu.sbatch
```

Do not set the license variable merely to bypass a failed guard.

Outputs (job ID forms the fresh opaque ID; existing paths are refused):

```text
RaTEScore/                                      # official source-only snapshot
runtime/models/ratescore_v12/assets_<job_id>_001/ # ner/, encoder/, receipts
runtime/venvs/ratescore-v12-<job_id>-001/
artifacts/protected/tricompose_v1_2/ratescore_setup_<job_id>_001/
```

`setup_receipt.json` verifies bytes and dependency metadata only. It explicitly
keeps model import/loading, stable inference and clinical qualification **false**.
Existing 2750/2770 runtime parent modes are preserved; new asset/receipt
directories are 2770 and asset/receipt files 0660 within the project boundary.
Model trees/caches are ignored by project Git. Partial failed acquisitions remain
recoverable in their fresh directory, not overwritten or deleted.

## Subsequent evaluation order (not authorized by setup)

1. Separate versioned CPU smoke on wholly authored text. Verify initialization,
   all-attempted availability, deterministic replay and native empty-entity
   behavior. Keep the official code/checkpoints/matrices frozen; new Slurm
   submissions require their own complete script display and approval.
2. Freeze the **same 624 expert-reference pairs** from the already authorized
   RadEvalExpert contract before model calls; do not select pairs by observed
   score or disagreement. Read source text only internally in an approved
   Slurm worker. Predict and seal before decoding expert outcomes. This cohort
   has already been analyzed by other metrics; do not call it untouched testing.
3. Report RaTEScore versus existing RadGraph at the same 624-pair denominator,
   and versus BioViL-T/RadGraph on the exact **132 image-accessible pairs**.
   Preserve missing/malformed annotations, all 492 unavailable-image rows and
   exact patient/source clusters. Spearman/significant error, category-specific
   relationships, and same-anchor selection/error differences with clustered
   intervals; no weight/threshold fitting, score sign flips or favorable-subset
   promotion. Existing 0.5 empty-entity returns get a separate coverage analysis.
4. Only after independent evidence, decide whether to add a **secondary**
   semantic agreement column to the fully synthetic 960 candidates. This does
   not by itself verify scope, EHR truth, image factuality or repair causality.
   Strong regeneration eligibility remains disabled until independently justified.

No clinical scoring result or performance improvement is claimed in this note.

## Preparation verification

Twenty acquisition/approval/path/streaming fixtures pass without a network or
model call. Full V1.2 regression: **2,456 tests passed in 18.490 seconds**, in
the existing CPU Slurm allocation 12714150. Shell syntax and Git whitespace
checks pass. Read-only preflight confirms exactly 22 planned public files,
1,184,749,604 bytes, no deployed source/model tree and zero inference/download
execution. Source/checkpoint/protected receipt paths are ignored by Git.
The historical bank, literal-evidence and head-scope manifest hashes are
unchanged. These tests verify implementation guards, not RaTEScore accuracy
or installation/inference readiness.

## License-compatible alternative inspected, not authorized or deployed

The [NCBI MedCPT Query Encoder model card](https://huggingface.co/ncbi/MedCPT-Query-Encoder)
marks the model public-domain; its
[published license](https://huggingface.co/ncbi/MedCPT-Query-Encoder/blob/main/LICENSE)
does not list the additional UMLS/SNOMED requirement that blocked BioLORD.
[MedCPT (Bioinformatics 2023)](https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842)
is a biomedical semantic retrieval model, **not a radiology contradiction
classifier or a replacement for native RaTEScore**. Its card explicitly
supports short-text/query-to-query representation with the query encoder alone.

Read-only checkpoint metadata:

- Repository: `ncbi/MedCPT-Query-Encoder`.
- Revision: `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`.
- Selected candidate file: `model.safetensors`, **437,951,328 bytes**.
- Published LFS SHA256:
  `19d78c0d5eaee2f81e6c47c5425bbadcc0c6af016cbb5da4a000d64e59d6e342`.
- Official code revision: `11e129be74102c98d16a11c310b0b5ce74c1db5e`.
- No checkpoint found in the inspected workspace runtime/model/cache roots.
- No weights, dependency installation, new environment, model call or submission.

Minimal proposed role: reuse already-installed frozen RadGraph extraction,
then test whether frozen MedCPT helps retrieve semantically related observation
descriptions. Keep polarity, anatomy and patient/temporal scope separate. A
similarity value cannot turn pneumonia/consolidation, left/right or current/prior
into equivalent facts. No embedding threshold, synonym equivalence, clinical
score or regeneration trigger is qualified yet. Exact identity remains the
baseline; a new controlled/independent evaluation is required before promotion.

Do not silently replace BioLORD in RaTEScore and report official default
RaTEScore results. This proposal would be a separately named experimental
semantic matching component. It needs separate user approval for downloading
the approximately 438 MB weight plus tokenizer/configuration. Reuse the
existing CPU runtime if verified compatible; no new GPU or training is needed
for the proposed small smoke. Any new batch submission still needs a complete
script/resource display and explicit approval. Old scores and winners remain fixed.

## Subsequent MedCPT approval and completion (2026-10-06)

The preceding alternative section records the **initial read-only proposal**.
The user subsequently explicitly approved MedCPT acquisition and a wholly
authored CPU smoke. That work is now complete; see
[the deployment and diagnostic result](medcpt_authored_smoke_result_12714150_001.md).
Eleven public asset files pass size/hash checks; native frozen CPU encoding of
38 distinct texts completes with exact two-pass replay. Forty nonempty probe
pairs are scored; one empty pair stays null. No new submission/environment or
patient/candidate inputs were used. Independent numeric replay and all 2,487
V1.2 tests pass; old manifest hashes/scores/winners are unchanged.

This does **not** qualify a clinical scorer: mean cosine is 0.742 for authored
paraphrases but 0.987 for negated variants, with high similarities also for
history, hypothetical and opposed details. Keep it a retrieval diagnostic,
not a consistency/repair threshold. Expert or candidate-bank scoring needs a
separate frozen input protocol/authorization. BioLORD/default RaTEScore remains
unlicensed and **not deployed**; MedCPT is not an official RaTEScore substitute.
