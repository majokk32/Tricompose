# Opacity stages V3: prospective format-only correction

Prepared, not new inference approval. After V2 job 12677127, structural diagnosis
found 49/50 locator outputs were bare arrays while the decoder required an
object. Its prompt inconsistently requested an empty bare array and a named
object. All original V2 outputs and 47 primary failures remain immutable.

V3 changes **only two locator-prompt instructions**: use the named object for
empty selections; explicitly forbid a bare array. Keep the strict decoder,
mechanical segmenter, source-ID inventory, state reducer and polarity prompt
exactly the existing V2 functions. No array rescue, model modification, dtype
switch, training, new clinical synonyms/rules, new state policy, source edits,
reference relabeling, case replacement or old-score overwrite.

Repeat the paired authored-only diagnostic on the same frozen 48 inputs/12
families/six known controls. The correction follows observed format failures;
this is not an untouched or blinded improvement experiment. Baseline remains
the unchanged V1 quote prompt. Two predeclared staged replays, maximum 148 calls,
seed 0, locator 128/polarity-baseline 512 output tokens, offline existing frozen
Qwen/loader. Actual phase attempts and failures are retained. Same-author keys
are not sent to the model and are parsed only after predictions are fsynced
and hash-closed. Investigator code/data access is not perfect blinding.

Use the unchanged V2 metrics, full unavailable denominators and narrow gate:
48/48 staged states, 48/48 authored selected-ID sets and two stable complete
replays. Even a pass does not authorize primary scoring, automatic repair,
clinical localization or a new selected triple. Report all families, costs,
semantic misclassifications and unavailable outputs, not favorable subsets.

Interface `TriCompose-v1.2/interfaces/opacity_assertion_stages_v3.py` delegates
unchanged parsing/polarity/reduction to the hash-bound V2 interface.
Worker `TriCompose-v1.2/tools/benchmark_opacity_assertion_stages_v3.py` is a
versioned copy of V2 with only version/interface/source-binding identifiers
changed, plus disclosure of the post-failure format correction. Pin both
interfaces, original worker, unchanged authored fixture, all runtime assets
and new tests/protocol. No consumed V2 file may be edited.

New atomic protected plan and result IDs; 2770 directories/0660 files. No
patient/EHR/CXR/synthetic-candidate input is read. Cache/tmp/logs workspace-local;
raw responses stay private. Independent post-run parsing/hash/metric/cost audit
remains required. New sbatch requires the entire exact script/resources shown
and explicit subsequent approval. This protocol is not submission permission.
