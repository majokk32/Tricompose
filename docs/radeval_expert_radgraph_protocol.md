# RadEvalExpert: frozen RadGraph versus expert error counts

## Scope and authorization

The user approved acquisition of the authors' small released expert dataset
and internal processing within the already approved CPU Slurm job 12714150.
This is a reference-based evaluator benchmark, not a generation smoke test.
No new sbatch submission, training, external clinical API, image loading,
EHR loading, selection changes, or regeneration are authorized by this run.
Raw CSV and native text-bearing graphs remain in protected project storage.
Only opaque IDs, hashes, availability, error counts, and aggregate statistics
are exported into protected evaluation artifacts. No patient text enters Git.

Official sources: [RadEval code](https://github.com/jbdel/RadEval),
[author dataset](https://huggingface.co/datasets/IAMJB/RadEvalExpertDataset),
[EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-demos.40/).
This is **not** the older IU-Xray RadEvalX cached-score dataset.

## Source and format preflight

- Dataset revision: `b4bd9d6ee75fcd155b6de466f1b7b7bd774408be`.
- File: `reader_study_final_with_annotationsv3.csv`, 691,815 bytes.
- Git blob: `c8862ccd726129a6e2a986b93e07f41b30dcee85`.
- 208 released rows, three candidate slots each: 624 attempted pairs.
- 762 exact-byte distinct reference/candidate texts.
- Two released annotator identities do **not** imply two ratings of each pair.
  Every exact pair in this release has one released reader cell.
- Significant and insignificant annotations each have seven named categories:
  false finding, omission, location, severity, unsupported comparison, omitted
  change, and grammar/readability. Parsing requires explicit integer counts.
- Two malformed category counts stay null: 623 significant totals and 622
  all-error totals are eligible. Blank/malformed is never silently zero.
- Source-section categories are preserved as opaque section IDs. Recognized
  Findings and Impression are named; the third enum remains
  `other_author_section`, not an assumed full-report scope.

The annotation adapter was corrected from author **schema descriptors only**,
before neural evaluation. Failed format-only preflights 001/002 are retained;
accepted contract 003 is pinned. This is a fixed local evaluation protocol,
not a claim of externally registered, blinded, independent testing.

## Dependence and cohort

All 624 attempted pairs are retained; no difficult or low-scoring case is
removed. Graph caching is by exact text SHA256, not clinical labels.
Exact source key, section, candidate slot, reference hash and hypothesis hash
identify a pair; only identical pairs may aggregate released reader counts.
An unavailable reader-category invalidates that aggregate category.

The current contract forms 180 dependence clusters: 77 recognizable MIMIC
patient-folder keys, 31 recognizable CheXpert patient-folder keys, and 72
unrecognized author source keys. Raw keys never leave the worker. The latter
72 clusters are **not verified patient identities**, so the entire analysis
must not be advertised as a fully patient-independent test. The 203 distinct
author image/source keys are not 624 independent patients.

## Frozen metric and statistics

Reuse official RadGraph 0.1.18, pinned source revision
`87f11a1ff4d2046a838be5f0243857d93780ddec`, existing XL checkpoint and offline
CXR-BERT-general tokenizer. CPU only, two threads, eval/inference mode, all
parameters frozen, network sockets blocked during scoring. No new weights.

Report **all three** official components, with no fitted weights:

1. Entity F1.
2. Relation-presence F1.
3. Full-relation F1.

Primary endpoint: Spearman correlation of each higher-is-better component
with **negative mean clinically significant error count**. Secondary endpoint:
all significant + insignificant errors. Report Kendall tau-b, all category
correlations, availability and per-section descriptive correlations.
For primary and all-error totals, use 1,000 cluster bootstrap resamples,
seed 0, percentile 95% intervals, preserving all candidate/section rows in a
cluster. Missing source counts or failed model scores remain null and reduce
coverage. Ties are handled explicitly. No significance-driven metric choice.

## Interpretation gate

This evaluates local text-reference RadGraph behavior against released expert
error burden. It does not validate image factuality, EHR consistency,
reference-free consensus, hard contradiction localization, or regeneration.
Checkpoint training overlap is unverified. Sparse single-reader ratings and
partially verified clustering limit generalization. No binary clinical gate,
model threshold, candidate winner, or primary selector is promoted by a
positive correlation alone.

The existing 960-candidate synthetic score table and 80 historical choices
remain byte-identical. Independent audit replays source parsing, official
rewards, count aggregation, correlations, lineage and protected permissions.
