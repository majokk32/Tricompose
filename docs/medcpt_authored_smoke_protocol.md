# MedCPT Query Encoder: approved acquisition and authored CPU smoke

Prepared after explicit user approval: public model acquisition and a small
wholly authored CPU test. BioLORD/default RaTEScore remains prohibited because
the user reports no required UMLS/SNOMED CT license. No source patient record,
real report/image/target, existing generated report/image/EHR or candidate bank
body will be consumed. This protocol is frozen before acquisition/model calls.

## Scope and fixed assets

Use [NCBI's released Query Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder),
not the article/cross encoder, another BERT substitute or an external API.
The published model license is public-domain. Code/license/docs snapshot revision
`11e129be74102c98d16a11c310b0b5ce74c1db5e`; root `MedCPT/` contains unchanged
official **README/LICENSE only**, not a full cloned training/inference repository.
Inference follows the native Transformers example from those docs.

- Checkpoint revision: `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`.
- Weight: `model.safetensors`, 437,951,328 bytes.
- LFS SHA256: `19d78c0d5eaee2f81e6c47c5425bbadcc0c6af016cbb5da4a000d64e59d6e342`.
- Explicit source/model/tokenizer inventory: `configs/medcpt_query_assets_v1.json`.
  Eleven pinned public files, total below a 500 MB acquisition cap. No duplicate
  `.bin` weights, PubMed embedding corpus, clinical datasets or implicit tokens.
- Existing environment `runtime/venvs/radgraph-xl-v12-12714150-001`: Python
  3.11.9, Torch 2.6.0+cpu, Transformers 4.44.2, NumPy 1.26.4, Safetensors 0.8.0,
  Tokenizers 0.19.1, huggingface_hub 0.36.2. Read-only reuse; no new packages/env.

The official [MedCPT paper](https://academic.oup.com/bioinformatics/article/39/11/btad651/7335842)
describes biomedical retrieval, not radiology contradiction adjudication.
The query card permits short-text/query-to-query representation; our cosine
diagnostic is not the official query/article inner-product relevance metric,
calibrated probability, NLI entailment or native RaTEScore.

## Execution

Only actual existing CPU Slurm allocation **12714150**, node b05-06, main,
4 allocated CPUs / 32 GB / 9-hour limit. Worker uses two threads, CPU FP32,
zero GPUs and a 30-minute execution cap. Its real cgroup must match; never forge
Slurm variables or use a login node. No new `sbatch`, training, fine-tuning,
scorer/router fitting, generator call, label/prompt tuning or patient input.
If this allocation expires, stop: any new batch needs its complete script and
resource request shown plus fresh approval.

Explicit HTTPS acquisition only, size and SHA256 checked before model loading.
New source/assets/run IDs refuse existing paths. Preserve failed partial assets.
Root/source/asset/run dirs use 2770, files 0660, project group boundary; existing
2750 runtime parent modes remain unchanged. Caches/temp/bytecode stay local.
During model calls: local paths, offline flags, implicit tokens disabled,
`trust_remote_code=False` and socket connects blocked. No credentials accessed.
Model warnings/responses only in protected `worker.log`; stdout is sanitized.

## Frozen authored probes and model recipe

`medcpt_authored_probes.py` fixes **41 comparisons** before inference:

- Six groups (edema, cardiomegaly, effusion, pneumothorax, atelectasis,
  consolidation) × identity, authored paraphrase, negation, prior-study-only,
  hypothetical and a cyclic different finding: 36.
- Four detail contrasts: laterality, location, severity, device position.
- One explicit empty-input control, ineligible/null rather than zero/negative.

These are investigator-authored language contrasts, not clinician-reviewed
clinical gold, untouched held-out cases, bank-derived aliases or representative
performance estimates. Known probe family names do not adjudicate clinical truth.
Keep all groups/contrasts even if similarity behaves poorly; do not revise them
after outputs are seen. No target/template success-conditioned selection/retry.

Deduplicate exact nonempty text hashes, sort them deterministically, batch 8,
seed 0, `eval()`/`requires_grad=False`/`inference_mode`, native last-hidden-state
**CLS**, dimension 768. No prompt prefix, rewriting, mean pooling or new adapter.
Native tokenization limit 64; reject oversized inputs before the official-style
truncating call, so these small probes have no actual truncation. Hash native
token IDs. Two identical passes; preserve and verify exact vector/score replay.
Record complete parameter binding, versions, calls, time and memory.

Cosine similarities remain raw relatedness diagnostics, grouped by probe type.
Report all six paraphrase-vs-different-finding orderings and corresponding
negation/history/hypothetical contrasts. Do not call these clinical accuracy or
fit a threshold. High similarity to negated/historical/opposed-detail text is
a warning against using embeddings alone as a contradiction or repair judge.
Only identity (~1), finite vectors and empty eligibility are mechanical checks.

## Outputs and interpretation boundary

```text
MedCPT/  # official documentation-only snapshot
runtime/models/medcpt-query-v12-12714150-001/
artifacts/protected/tricompose_v1_2/medcpt_authored_runs/query_12714150_001/
  frozen_plan.json
  score_table.json
  embeddings.json
  token_receipts.json
  summary.json
  manifest.json
```

Current native RadGraph is retained as a future entity source; this test makes
**zero RadGraph calls**. It does not score the old 960 candidate triples or
change old scores/EHRs/images/reports/winners. Historical bank/literal manifests
are hash-guarded. All clinical qualification/regeneration flags stay false.
Next independent expert/controlled entity-matching benchmark needs a separate
frozen protocol/input scope before any qualification or promotion.
