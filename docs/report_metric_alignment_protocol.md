# Independent expert-error alignment: frozen protocol v1

## Scope, established before CSV score/error values are decoded

First target: the open [RadEvalX 1.0.0 release](https://physionet.org/content/rad-eval-x/1.0.0/).
It has 100 IU-Xray reference/generated-report pairs, consensus annotations from
two radiologists, eight error categories and published metric scores. Readers
compared reports without the images. Its CC-BY-NC-SA-4.0 release is about 144 KB.
The cohort was constructed with abnormality and RadCliQ-related filtering; it
is not a representative or untouched TriCompose test cohort.

Use all released pairs, not handpicked examples, and only published numeric
scores/error counts. This first diagnostic makes no local model calls. It asks
whether released metrics rank higher-error reports below lower-error reports.
It cannot qualify our local CheXbert/RadGraph implementation, current-patient
assertion extraction, reference-free synthetic selection or EHR/CXR correctness.
Published scores and prior literature are available to the investigator; a
prediction-first file receipt is provenance, not clinical blinding.

## Source boundary and fixed schemas

Source root:
`artifacts/protected/tricompose_v1_2/report_metric_sources/radevalx_source_12714150_002/`.
Manifest SHA256:
`7073624ae36b71b1581dc5e288902659513761b4051f0cd5b571821aaf6467ee`.

The three source files match their official release SHA256SUMS; this does not
provide an independent authenticity attestation. Only headers were inspected
before this protocol. Two annotations CSVs have exactly:

```text
report_id,ground_truth,M2Tr-Generation,1,2,3,4,5,6,7,8
```

The published metric CSV has exactly:

```text
report_id,id,bleu4,bleu_2,bertscore,CheXbert,radgraph_f1,radcliq
```

No public filenames contain source keys. Internally join by the exact stripped
report_id string, require uniqueness and identical key sets, and assign opaque
pair/group IDs in metric-file row order. Do not inspect or use ground_truth,
M2Tr-Generation or the secondary id field. The CSV parser necessarily reads
source bytes, but those report fields are unused, not displayed or exported.
No manual source ID, source report, MIMIC row, image or EHR body enters output.

Blank/NaN numeric cells stay unavailable/null, not zero. Reject malformed,
nonfinite, negative or fractional consensus error counts. Require all 100
released pairs, retaining unavailable score/category cells. Missing/duplicate/
unmatched records fail closed, never lead to an easier selected cohort.

## Frozen metric direction and statistical contract

| Published column | Derived name | Orientation |
| --- | --- | --- |
| bleu4 | published_bleu4 | higher is better |
| bleu_2 | published_bleu2 | higher is better |
| bertscore | published_bertscore | higher is better |
| CheXbert | published_chexbert | higher is better |
| radgraph_f1 | published_radgraph_f1 | higher is better |
| radcliq | published_radcliq | lower is better |

These are released columns, not new model inference. The exact CheXbert
score definition/checkpoint and RadGraph variant are not established by column
names. Do not relabel them local CheXbert label F1 or a newly deployed RadGraph-XL
score. RadCliQ is oriented as predicted error burden, per its original
[metric study](https://doi.org/10.1016/j.patter.2023.100802).
No direction may be flipped after observing correlations. No weights, composite
metric, operating threshold or model priority is fitted here.

Primary diagnostic: Spearman correlation between oriented quality and negative
clinically significant error total. Report Kendall tau-b, score availability,
paired availability and every attempted denominator alongside it. Higher
positive alignment means higher quality scores tend to accompany fewer errors;
it is not accuracy or a calibrated probability.

Also report insignificant/all-error totals and each of eight significant/
insignificant categories separately. Keep zero/constant categories visible with
null correlation, not perfect agreement. Fewer than three pairs yield null.
Tie-aware average ranks and tau-b are explicit. Missing any category makes its
corresponding total unavailable; unrelated categories remain available.

For the predeclared primary outcome only: 1,000 source-group bootstrap draws,
seed 0, percentile 95% Spearman interval. Resample groups, not reader cells or
dependent candidate rows. Report usable draw count; require at least 90%
nonconstant draws and 20 usable draws, otherwise interval null. There is one
generated candidate per study here, so this cannot measure within-EHR best-of-N
selection. Per-category analyses are descriptive, not multiple unadjusted
significance claims. Do not fit a new scorer to this benchmark.

## Execution, isolation and output

Run only in the existing actual CPU Slurm allocation after an explicit scope
flag. No GPU, new sbatch, dependency download, inference, training, API or
external clinical upload. Freeze source/helper/test/protocol hashes and plan
before numeric analysis. Save/fsync/hash derived published predictions before
decoding error reference semantics; these were already published, so this is
not a blind model evaluation. Then create separate opaque derived reference and
summary files in a fresh protected atomic run, dirs2770/files0660/project group.

Do not modify source CSVs, old score tables, fixed EHRs, candidate artifacts,
selected triples or prior evaluation gates. Keep `clinical_qualified=false`,
`local_implementation_qualified=false`, `selection_changed=false` and
`regeneration_authorized=false` regardless of correlation.

## Local verifier audit and follow-up

Local metadata/source inspection found frozen CheXbert (1.31 GB checkpoint,
SHA256 `6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1`),
the existing Qwen2.5-VL 7B path and completed diagnostics. Neither is a new
independent evaluator. No RadGraph/GREEN package/checkpoint was found in the
audited workspace model caches, cxrmate environment or evaluator roots; this
is not a claim about every user's storage on CARC.

The frozen [official RadGraph package](https://github.com/Stanford-AIMI/radgraph)
is a useful next extraction candidate because it represents observation status
and anatomy relations rather than only fourteen report labels. The
[official weight listing](https://huggingface.co/StanfordAIMI/RRG_scorers/tree/main)
lists radgraph-xl.tar.gz ~416 MB, original radgraph.tar.gz ~432 MB and modern
RadGraph-XL ~579 MB. These are not downloaded by this task. Explicit approval,
workspace-contained environment/cache setup and a new complete reviewed Slurm
request remain required before deployment/inference. Its F1 compares two texts
and is not a standalone image-report consistency score or automatic repair
verdict. New runtime/code/model revision pins are required, not an unpinned
default download on first call.
