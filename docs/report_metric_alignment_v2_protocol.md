# RadEvalX source inventory clarification v2 — before numeric evaluation

The frozen V1 [statistical protocol](report_metric_alignment_protocol.md),
model-free metric directions, category definitions, missingness rules, tie-aware
correlations, 1,000 group bootstrap draws/seed0 and diagnostic-only boundaries
are unchanged. V1 worker/adapter/plan are retained; its attempted evaluation
failed `released_hundred_pair_bound_exceeded`, before references or a scored run
were produced. It wrongly assumed all released score rows were annotated.

A subsequent header/key-only audit found **590 unique score keys**, **100 unique
keys in each annotation file**, identical significant/insignificant key sets,
all100 keys represented among the scores, no duplicate/unmatched annotated keys,
and490 score rows without expert annotation. No numeric score, error count or
report field was accessed in that audit. This clarification is source-inventory
repair, not tuning to metric correlations or changing the clinical reference.

V2 evaluates **all100 expert-annotated pairs** by exact stripped key membership.
Assign opaque pair/group IDs in original metric-file order restricted to those
keys. Retain the full590/100/100/490 inventory summary. Do not turn the490
unannotated pairs into zero-error examples or count them as100 more scored cases.
No easy-case, score-, disease- or error-guided sampling. Fail rather than drop a
duplicate or missing annotated key. The exact590/100 inventories are pinned.

Annotation **key metadata** is decoded before publishing opaque predictions;
annotation **error-count semantics** are decoded only after prediction save,
fsync and hash. The receipt must disclose both, plus investigator access to the
published dataset. No clinical-blinding claim. CSV report fields are unused;
original keys never exported. Source bytes and raw reports remain protected.

Freeze V2 worker/adapter/tests, this clarification, all reused V1 sources and
the source manifest before execution. Fresh atomic plan/run; do not overwrite
V1. Existing CPU Slurm only, no local models/GPU/API/training/new sbatch. All
results remain reference-based diagnostics, never a local scorer qualification,
reference-free score-table replacement, image/EHR truth or repair trigger.
