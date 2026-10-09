# Official CheXpert/NegBio: source-only setup and prospective output contract

This step installs only a sparse checkout of published source/rules. It does
not download parsing models, install an environment, label a report, evaluate
clinical accuracy, change old scores, select a new winner or regenerate data.
Component checks run within the existing authorized CPU Slurm allocation.

## Fixed official sources

- [CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler/tree/44ddeb363149aa657296237f18b5472a73c1756f),
  root `chexpert-labeler/`, revision `44ddeb363149aa657296237f18b5472a73c1756f`.
- [NegBio](https://github.com/ncbi-nlp/NegBio/tree/073199e2792824740e89844a59c13d3d40ce4d23),
  root `NegBio/`, revision `073199e2792824740e89844a59c13d3d40ce4d23`.

Clinical example CSVs, upstream examples/tests and images are not checked out.
Both nested repositories are excluded from TriCompose Git. Upstream source is
unchanged; selected files/rules, wrapper, tests and protocol are hash-pinned.

## Exact semantics, not assumed semantics

The pinned CLI's default `sections_to_extract=[]` means **full report**;
`extract_strict=False`. The Loader is named an impression loader but its class
name does not override the actual argument defaults. Explicit section
extraction is a different configured experiment, not a hidden default.

Official per-mention aggregation uses negation-key presence, then uncertainty
key presence, otherwise positive. This is not an independently verified
current-finding assertion. For multiple labels, branch order is:

| Present mention labels | Native aggregate |
| --- | --- |
| Negative and uncertain, including all three polarities | Uncertain |
| Positive and negative, without uncertain | Positive |
| Positive and uncertain, without negative | Positive |
| None | NaN / unmentioned |

Native No Finding and CHF/heart-failure propagation remain unchanged. Native
No Finding is never expanded into fourteen explicit negative assertions.
Official Lung Opacity phrases cover more concepts than the previously authored
literal-opacity head. These heads are not silently interchanged.

Save two distinct views: official numeric labels, and a disclosed TriCompose
mention-conflict view (opposing signs or uncertainty -> uncertain). The latter
is an output adapter policy, **not official CheXpert**, a new linguistic parser,
a second independent clinical vote or a validated repair gate. It does not
resolve current/history/anatomy scope. Native outputs are never overwritten.

## Evidence and failure boundary

Official cleaning changes case, punctuation, whitespace and some slashes.
Annotation offsets are in `official_cleaned_full_report` Unicode codepoints.
Save original and cleaned hashes separately. Do not claim original-source
alignment, exact original quotes or clinically current evidence from these
offsets. Only offsets/hashes and structured flags are serialized, not bodies.

Check parse-tree and dependency availability before native Classifier deletes
sentence intermediates. Bind the receipt to the cleaned passage and exact
mention inventory. This confirms available structures, not correct syntax.
The official converter/detector can catch errors without raising. A future
worker must capture sanitized error counts and fail closed; absent trees,
nodes, mention coverage or detector failures are unavailable/null, not unknown.
The current contract does not yet initialize or run that worker.

## This step's verification

Run unmodified official aggregation on all 40 ordered label sequences of
length zero through three, plus six authored annotation-level special cases.
These are component checks on mock BioC-shaped objects, not report inference,
46 independent patients, accuracy or evidence that the parser is reliable.
Use existing read-only ConText environment solely for NumPy/tqdm to execute
the aggregation component. This does not reproduce the legacy parser runtime.
Add invented serialization/failure/offset/unknown tests and full regression.
Write fresh protected atomic audit output; no existing source/results change.

## Next full-parser boundary

Official environment pins Python 3.6.7 and old dependencies, including BLLIP,
BioC, pandas, NetworkX 1.11, StanfordDependencies and JPype. Full parsing also
needs Java, the Stanford dependency JAR, frozen GENIA+PubMed parsing model and
NLTK punkt/universal-tagset/wordnet resources. No automatic resource download
or writes through HOME are allowed. A parser-path configuration override must
be explicit and separately recorded; never modify HOME or upstream rules.

First obtain approval for isolated environment/resource setup. Show an exact
complete CPU batch script/resources and obtain explicit approval before any
new submission. Establish offline initialization/resource identities, then
freeze a plan over the existing authored48 and authored64 inventories, keeping
their references/results separate and acknowledging known development data.
Save predictions/replays before reading references. Measure native ontology
and literal-head limitations openly; do not transfer these language controls
into candidate clinical accuracy or fit new rules to their observed errors.
Only after end-to-end validation consider a separately approved synthetic-only
candidate diagnostic. EHR facts, old score tables and winners stay unchanged.
