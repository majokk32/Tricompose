# SynEHRgy-v2 frozen synthetic-EHR candidates

This experiment wraps two pinned public checkpoints:

- `hojjatkarami/SynEHRgy-gpt2-10bins` at
  `ac6476b0376c26618386e4ca0d5d0c9d30be8c84`;
- `hojjatkarami/SynEHRgy-qwen2-40bins` at
  `41263cd34d287ed7a32a8122fa8a72a29e2e9e2b`.

Both perform cold-start generation from the BOS token only. No real patient
record, prompt, report, image, or external API is used. The adapter requires an
explicit model variant, checks the architecture, vocabulary, exact file sizes,
and pinned SHA-256 hashes, loads weights locally, switches to evaluation mode,
and disables gradients.

The Qwen2-40bins model is the preferred candidate for the next smoke test. It
has a 4-layer Qwen2 causal-LM architecture, hidden size 384, 3158-token EHR
vocabulary, and 40-bin numerical representation. Although its model config
supports 4096 positions, the author's `config_main.yaml` fixes `n_ctx=2048`, so
the adapter deliberately generates at most 2047 new tokens after BOS.

The first run intentionally preserves numerical observations as the official
discrete bin tokens. Exact de-quantization is disabled until the separately
gated MIMIC-IV vocabulary and quantile files are available. The adapter does
not infer or invent bin boundaries.

Protected case files contain the raw generated token sequence plus a grouping
into visits, covariates, problems, labs, and charts. Public Slurm logs contain
only aggregate structural validity, sequence lengths, runtime, memory, and
cryptographic hashes.

The legacy GPT-2 smoke sampler uses `temperature=0.7` and `top_k=50`. The new
Qwen2 smoke sampler uses the current repository's `gen1.yaml` values:
`temperature=1.0`, `top_k=50`, `top_p=1.0`, and `repetition_penalty=1.0`.
Both are the author's later MIMIC-IV-format models with a 2048-token generation
context, rather than the paper's earlier MIMIC-III model.

Public model files are installed into workspace-managed storage under
`runtime/models/synehrgy_v2/`; the official source checkout under
`SynEHRgy-v2/` remains read-only. Synthetic case outputs remain under
`artifacts/protected/synehrgy_v2/runs/<opaque_run_id>/`.

## Evaluation routing

The 10-case smoke run tests only engineering and structural validity:

- BOS/EOS completion and balanced visit/section delimiters;
- absence of padding and unknown tokens;
- visit counts, token lengths, runtime, and peak GPU memory;
- sequence uniqueness, exact duplicates, and adjacent-token repetition;
- per-section case/visit coverage and unique token counts;
- within-visit problem-token unigram, bigram, and trigram diversity;
- manual inspection of explicitly authorized synthetic outputs.

It cannot establish distributional fidelity. The paper-level evaluation needs
a substantially larger cohort and the exact author preprocessing artifacts:

- ICD fidelity: unigram, bigram, trigram, and cross-visit sequential-bigram
  frequencies, compared with Pearson correlation;
- time-series fidelity: first-48-hour min/max/mean/std embeddings with PRDC,
  temporal correlation-matrix error, correlation-level agreement, and
  measurement co-occurrence/missingness;
- utility: TSTR and real-plus-synthetic LightGBM evaluation for phenotype and
  in-hospital mortality AUROC;
- privacy: nearest-neighbour membership inference using Hamming distance for
  ICD sequences and Euclidean distance for time-series embeddings, summarized
  with Wasserstein distance, Jensen-Shannon divergence, and attack AUROC.

All real-data evaluation must run internally through an approved Slurm job and
may emit only aggregate statistics.
