# Frozen CheXpert/NegBio end-to-end authored112 protocol

This is a prospective execution over **already known development controls**,
not a new held-out clinical benchmark. Use the exact existing authored48 and
authored64 inputs, without selecting easy cases, editing texts or fitting rules.
References describe a conservative literal lung-opacity/current-finding task;
the official broader Lung Opacity ontology is not silently relabeled to match.

## Frozen execution

Use the dedicated Python 3.6.7 deployment sealed as
`deployment_12682821_001`, manifest
`f8c8d9a02455780de22e913ce7a1c056450c5e75eafffea6279d869583d9ef85`.
Only CPU Slurm; an existing allocation may be used. Before any new submission,
show the exact complete script/resources and obtain approval. No additional
download, training, model-weight/rule change, GPU or external API.

Prepare an immutable protected plan with source/deployment/resource hashes.
Copy the existing authored inputs and separate references; the legacy worker
accepts only this sealed authored-only plan, not arbitrary report input paths.
The worker does not load either reference file. Every text is parsed twice;
failures are retained, never selectively retried or excluded from denominators.
Fsync/hash predictions and replays before the evaluator reads reference states.
The investigator knows earlier outcomes; phase isolation does not make these
controls unseen, clinician-labeled or independent clinical gold.

## Native inference and observational boundaries

Run the original upstream `Classifier.classify` without editing its source.
Wrap only its parse/converter/detector calls to count sanitized ERROR logs and
obtain a syntax receipt before the official method deletes sentence structures.
Verify parser trees, dependency nodes/edges, source alignment and native
`find_nodes` coverage. These are availability checks, not proof of correct syntax,
current anatomy or clinical factuality. A swallowed exception/error log, absent
intermediate or failed alignment is **unavailable/null**, not unknown or positive.

Use original Loader.clean, text2document and the official splitter in memory,
explicitly full-report mode (`sections_to_extract=[]`, `extract_strict=False`).
CSV input/output is not used. Exact original and cleaned passage hashes differ
when official cleaning/transformation changes text. Offsets refer to cleaned
Unicode codepoints only. Serialize hashes/offsets/flags, no text bodies.

The annotation-free TriCompose output contract is loaded in Python 3.6 by
removing only its unused `future annotations` import in memory; no source or
contract behavior changes. Reject broader annotation/transpilation needs.
Workspace-only model/JAR/NLTK/Java temp paths and PLY no-table configuration
match the previously disclosed deployment; no backend fallback/download.

## Outcomes and interpretation

Keep two views separate: all official fourteen-category numeric/state labels,
and the disclosed TriCompose mention-conflict view. The latter can turn opposing
mentions into uncertain but is **not official CheXpert**, another independent
clinical vote or a validated temporal/anatomical parser. Native output is never
overwritten. Unmentioned means unknown. No Finding does not negate other heads.

For old48/new64 separately report:

- Complete/unavailable counts, failure phases, full-evidence replay differences.
- Four-state confusion/F1 and exact alignment to the authored task for native
  Lung Opacity and the conflict view, always using all attempted rows.
- Signed positive/negative flips; determinate predictions on authored
  uncertain/unknown cases; per-family breakdown.
- A predefined literal opacity/opacities-present subset as a diagnostic slice,
  not an anatomy-qualified or clinical test subset. Nonpulmonary mentions remain.
- Official ontology mismatches in nonliteral/other-disease families; do not
  describe these counts as standalone CheXpert clinical accuracy.

Failure injection tests must demonstrate logged-but-swallowed detector and
converter errors become unavailable, not a complete unknown/positive; missing
trees/nodes/coverage are gated and observational hooks are restored. Replay
agreement is engineering determinism, not clinical validity. No tuning to these
outcomes and no automatic promotion regardless of authored test performance.

Protected plan/run dirs use 2770, files 0660, project-group boundary only.
Existing EHR/CXR/reports, old labels/score tables/selected triples and failed
qualification gates remain unchanged. No 960-slot candidate scoring, selector
promotion or regeneration is authorized by this authored diagnostic.

Official fixed sources:
[CheXpert](https://github.com/stanfordmlgroup/chexpert-labeler/tree/44ddeb363149aa657296237f18b5472a73c1756f),
[NegBio](https://github.com/ncbi-nlp/NegBio/tree/073199e2792824740e89844a59c13d3d40ce4d23).
