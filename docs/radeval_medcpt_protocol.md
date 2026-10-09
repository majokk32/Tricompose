# MedCPT report-reference expert benchmark: prepared, not executed

Prepared after the authored CPU diagnostic showed that cosine alone does not
separate negation or patient/temporal scope. This next test measures association
with released expert error burden, not clinical correctness, image truth or
regeneration eligibility. A separate approval of the complete execution entry
and author-report input scope is required. Preparation does zero inference and
does not open the raw author report CSV. No new `sbatch` is authorized.

## Fixed cohort and comparator inventory

Use the already acquired authorized author RadEvalExpert CSV, fixed accepted
inventory 003 and original 624 attempted reference/candidate pairs. Its 208
source rows define 762 exact-byte distinct texts and 180 dependence groups:
77 recognized MIMIC patient folders, 31 CheXpert patient folders and 72 opaque
unrecognized author source keys. The latter are not verified patients. One
released reader cell per exact pair; 623 significant totals and 622 all-error
totals were eligible in the prior analysis. Missing annotations stay null.

The source and released outcomes have already been analyzed. This is neither
blind nor untouched final testing; training overlap is unverified. All rows,
sections and three candidate slots remain. No case choice based on disease,
error count, old winner or new MedCPT score. The input CSV is read only inside
the separately approved Slurm worker; raw identifiers/text never leave it.

Reuse all three cached native RadGraph components without new RadGraph calls.
Reuse cached BioViL-T scores, with 132 image-accessible pairs and 492 unavailable
rows, without reading or re-encoding any real image. Two comparisons:

1. Full common text-score cohort: MedCPT and all three RadGraph components.
2. Common cached-image cohort: those four plus BioViL-T, identical mask.

Keep the all-attempted 624-pair denominator. A new unavailable MedCPT input
remains null and masks that pair for every metric in its paired comparator
cohort. The old complete-cohort RadGraph/BioViL results remain immutable;
recomputed common-cohort analyses do not replace them. Selection diagnostics
require all three scores and all three expert counts for each fixed anchor.
Do not drop one inconvenient candidate while retaining the other two.

## Fixed model recipe; not the official retrieval metric

Use the already deployed NCBI Query Encoder revision
`d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc` and verified 437,951,328-byte
weight SHA256 `19d78c0d5eaee2f81e6c47c5425bbadcc0c6af016cbb5da4a000d64e59d6e342`.
Native BERT CLS vectors, CPU FP32, 768 dimensions, eval/inference mode,
requires_grad false, seed 0, two threads, batch 8, exact text-hash deduplication
and deterministic hash-sorted inputs. No prefix, rewrite, aliases, mean pooling,
new weights, training, adapter or fitted thresholds.

This experimental **full-report query-query cosine** is explicitly different
from the official short-query example's 64-token recipe and query/article
inner-product retrieval metric. Use the unchanged model's **512-position
capacity**, with native special tokens counted. Full input only: do not truncate
any report. Above-capacity, empty and failed inputs are preserved separately as
unavailable, not encoded prefixes or scores of zero. No chunking/long-context
optimization, parameter sweep or post-result choice of token limit.

Hash exact native token IDs and verify actual batched tokenizer input. Persist
hash-linked vectors only below protected storage. Replay the first eight
hash-sorted eligible texts once: this is a **small predeclared replay**, not
two-pass replay of the entire expert cohort. Its availability and exact-match
status are reported even if a batch fails. Do not retry failures selectively.

Only the author reference/candidate strings are model inputs; expert counts,
source/patient identifiers and existing scores are never tokenizer/model inputs.
The pre-existing inventory contains expert counts, so do not claim outcomes
were unknown. Seal predictions/receipts before statistical integration.

## Endpoints and interpretation

Primary: Spearman and Kendall tau-b against **negative clinically significant
expert error count**. Secondary: negative all-error total, all seven significant
and insignificant categories, retained separately. Do not combine them with
new weights or flip a score sign after seeing outcomes.

For both total endpoints: 1,000 dependence-cluster bootstrap resamples, seed 0,
percentile 95% intervals; keep dependent candidate/section rows together.
Report all attempted/available denominators and source groups. Category
correlations are descriptive, not multiple-comparison discoveries.

Within-anchor comparison: expected errors from highest score versus analytical
uniform random selection and minimum-count oracle. Exact top-score ties use
uniform expected choice. Report complete/incomplete anchors, strict pairwise
ranking, ties and clustered intervals for selected-minus-random error burden.
These are reference-based ranking diagnostics, not actual synthetic-bank
selection, image adjudication, modality fault localization or repair success.

No clinical threshold, winner, semantic equivalence or regeneration trigger is
promoted by a positive result. Scope/polarity/anatomy still require independent
evidence; no high cosine converts unknown into negative or support.

## Execution, safety and outputs

Only actual existing allocation 12714150: **4 allocated CPUs / 32GB / 0 GPU**;
worker uses two threads and a **20-minute process cap**, not a runtime promise.
If the allocation expires, refuse the entry point; a new complete batch script
and resource request must be shown and explicitly approved. No login inference,
fake Slurm variables, source/model download, package/env installation, external
clinical API, source EHR/image input or generation is involved.

Read-only existing environment `runtime/venvs/radgraph-xl-v12-12714150-001`:
Torch 2.6.0+cpu, Transformers 4.44.2, NumPy 1.26.4, Safetensors 0.8.0,
Tokenizers 0.19.1, huggingface_hub 0.36.2. Local-only loading, implicit tokens
disabled, no credential files, no remote code, network sockets blocked during
model calls. All caches/temp remain in the workspace. Protected new dirs 2770,
files 0660, CARC project group; refuse existing runs and preserve failed partials.

```text
TriCompose-v1.2/tools/score_radeval_medcpt.py
TriCompose-v1.2/src/tricompose_v12/radeval_medcpt_benchmark.py
TriCompose-v1.2/tests/test_radeval_medcpt_benchmark.py
TriCompose-v1.2/slurm/63_radeval_medcpt_existing_cpu.sh
artifacts/protected/tricompose_v1_2/radeval_medcpt_plans/reportref_12714150_001/
artifacts/protected/tricompose_v1_2/radeval_medcpt_runs/reportref_12714150_001/
```

Metadata-only preparation seals source/code/model/runtime/comparator hashes.
Execution needs separate approval and creates scores, token receipts, private
vectors, prediction receipt, common-cohort score table, evaluation, summary and
manifest. Public stdout only sanitized status, runtime, memory and hashes.
Model warnings/exceptions stay protected, with safe failure-type codes.
Do not display raw reports, raw source IDs or per-patient inputs/results in chat.
Old 960-candidate scores, EHRs/images/reports/winners and authored smoke stay fixed.
BioLORD/default RaTEScore remains undeployed without the absent licenses.

Official sources: [MedCPT model](https://huggingface.co/ncbi/MedCPT-Query-Encoder),
[MedCPT paper](https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842),
[RadEval dataset](https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset),
[RadEval paper](https://aclanthology.org/2025.emnlp-demos.40/).
