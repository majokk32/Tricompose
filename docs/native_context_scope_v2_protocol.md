# Scope88 V2: repair token alignment, not linguistic outcomes

V1 `scope88_12714150_001` is sealed and unchanged. Its 38/88 context failures
are strict native-character/spaCy-token alignment failures: official vocabularies
include `opaci`, `atelecta`, `consolidat` stems. These are not unknown findings,
not successful anatomy rejection and not improved scope interpretation.

V2 preserves the same 88 texts/references and exactly the same cached native
predictions. The V1 outcomes are known. This is an engineering repair/replay,
not a fresh held-out benchmark or prospective unknown-results claim.

Before context execution, split only blank Doc tokens at existing native
annotation start/end offsets. Subtoken strings concatenate exactly to the
original token; `doc.text`/cleaned SHA256 and all native annotation ranges/hashes
must remain unchanged. No span expansion/contraction, dropped evidence, word
replacement, new cue, language rule, synonym or threshold. Existing parsed Docs
are refused. Default token boundaries **are explicitly changed**; unchanged
text/rules do not imply an unchanged tokenizer configuration. Record original
token offset/hash and split offsets without bodies. This uses the documented
[spaCy retokenizer split API](https://spacy.io/api/doc#retokenizer.split).

Frozen ConText triggers and V1 veto-only decision semantics remain unchanged.
Split tokens can affect matching scope/window behavior; do not interpret any
remaining linguistic result as proven clinical correctness. Never expand the
official stem to a word and claim it was the original evidence span.

Freeze new source/tests/protocol and protected plan before this replay. Reuse
the V1 sealed native outputs and its runtime receipt as historical provenance;
**new native parser calls = 0**. Parse each context/gate twice, save/fsync/hash
predictions before decoding reference JSON. Developer reference knowledge is
explicit; source-byte hashing is not clinical blinding.

Report V1/V2 availability, remaining unavailable rows, all-attempted raw native
metrics, soft retained wrong/correct, incorrect withheld, correct lost, coverage
and every family. Raw native outputs/metrics must be identical. No abstention
credited as a correct unknown. Engineering coverage restoration can expose more
wrong agreements; report them, do not hide them or tune further to this test.

No GPU/new sbatch/download/training/API, candidate report body, source patient
input, old score/EHR/winner change or regeneration. Existing CPU allocation;
protected dirs 2770/files 0660/project group only. `clinical_qualified=false`.
