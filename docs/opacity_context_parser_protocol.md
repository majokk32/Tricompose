# Official ConText opacity diagnostic, prospective frozen execution

CPU rule parsing in the existing authorized Slurm allocation only. No new
submission, trained model, inference GPU, installation, download, API or patient
input. Reuse read-only `runtime/venvs/report-context-v12-12576792`:
medspaCy 1.3.1, spaCy 3.7.5, PyRuSH 1.0.12, numpy 1.26.4.

## Mechanism and limitations

Use blank English tokenization, official PyRuSH sentence rules and unchanged
official English ConText triggers. No learned NLP component, custom negation
rule, cue vocabulary extension, or threshold. Official implementation:
[medspaCy 1.3.1](https://github.com/medspacy/medspacy/tree/1.3.1).
The old four-finding default parser was not reliable on its authored controls;
this experiment is not justified by an assumption of high accuracy.

The **only new concept inventory** is `\bopacit(?:y|ies)\b`, case-insensitive.
This matches two literal noun forms, not a trained or official lung-opacity
ontology. It deliberately does not equate pneumonia, edema, generic normal
summaries or other diseases with opacity. It cannot itself distinguish an
opacity outside the lung. Disclose errors on nonpulmonary control texts rather
than adding anatomy rules after evaluation.

Reuse the previously frozen five-flag mapping: historical, hypothetical or
family -> unknown; uncertainty -> uncertain; negated -> negative; otherwise
default positive. The unmodified-positive default is NOT verified clinical
presence. Across mentions, any uncertainty or opposing explicit polarities ->
uncertain; otherwise retain the sole explicit state or unknown when none.

Source text is unchanged. Evidence stores exact Unicode offsets and hashes for
mentions and official modifier cues/scopes, without quote bodies. One parser
execution per whole text plus one deterministic replay; failures remain null
and explicitly unavailable. Do not fill failures with unknown predictions.

## Fixed inventories and no answer-assisted processing

Prepare an immutable plan before any new parser execution:

- Existing 48 authored V3 texts, input/reference bytes unchanged; plan manifest
  `238a8e44133c2edc9c69d460a49fcbfbb5e62ef38ed64672699a808eaa088551`.
- New 64 wholly invented texts in 16 families, four per family, from
  `benchmarks/opacity_context_controls_v1.py`. No protected text was consulted.
  Include morphology, explicit negation, uncertainty, historical/family/
  hypothetical/resolved mentions, past-vs-current scope, opposing assertions,
  qualified absence, change-only language, other-entity negation, generic
  summaries, other diseases, sentence boundaries and nonpulmonary opacity.
- Existing V3 predictions are read only after new parser outputs are saved,
  fsynced and hash-frozen; they are used for comparison/veto only, not input.
- Parse authored reference states/families only after outputs are frozen.

The reference conventions are investigator-authored linguistic tasks, not
clinical gold. New texts are frozen before this run, but are **not independent
or held out from the investigator**. Existing 48 and their results were already
known. Never merge both sets into a flattering single headline or tune the
rules/targets after seeing new results. Do not choose a favorable subset.

Pin pipeline configuration, all loaded medspaCy/PyRuSH source and English
resource hashes, versions, tokenizer serialization, target/mapping code,
fixtures, tests and protocol. Require the official rule dictionary and assets
unchanged before/after both passes. Source/hash checks and atomic protected
fresh-run writes fail closed. Refuse existing IDs; modes 2770/0660, project
group ruishanl_1185 (NFS may expose nobody).

## Readouts and veto-only comparison

For old48 and new64 **separately**, report all-attempted four-state match,
class confusion/F1, unsafe certainty on uncertain/unknown controls,
positive/negative flips, availability, replay changes and every family.
Also report target-mention coverage; absence of a matched noun is not a negative.

For old48 only, use a new sidecar over unchanged V3 positive/negative proposals:
retain a soft proposal only if the new parser is complete, has a literal target
and agrees on its state. Unknown/uncertain/failure/disagreement are separate.
This is a different algorithm, **not independent clinical evidence**: both read
the same report. No head is qualified for hard actions. Record incorrect
retained, incorrect withheld, correct lost, coverage and conditional match.
Never call abstention correct unknown or overwrite raw predictions/scores.

Zero new frozen-generative-model calls. CPU parsing time is recorded separately
from previous Qwen inference cost. No clinical triple rejection, scoring-table
promotion, winner change, EHR enrichment, fault assignment or regeneration.
The original V3 language gate stays failed regardless of this diagnostic.

The practical decision is whether unmodified default ConText adds useful
scope evidence under these declared conventions. If not, keep its limitations
and do not deploy it as a reliability fix. Any further generator/verifier GPU
run requires complete script/resources and fresh explicit approval.
