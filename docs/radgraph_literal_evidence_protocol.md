# Literal native-observation evidence on the synthetic report bank

Status: fixed DEVELOPMENT diagnostic before its first numerical execution.
The expert benchmark results are already known. This is not retrospective
preregistration, new clinical validation or an installed selection/repair policy.

## Scope

Reuse only sealed fully synthetic RadGraph-XL outputs from
`artifacts/protected/tricompose_v1_2/radgraph_bank_runs/native_pool960_12714150_001/`.
Keep all 80 fixed EHRs, 240 CXR slots, 960 report/triple slots, 428 unique report
graphs and 1,440 same-image expert pairs. Do not read source EHR, source/target
MIMIC reports/images, prompts, new generated report bodies, or checkpoints.
Synthetic graph text is consumed internally only; no entity token or report
fragment is exported in the new evidence table, public logs, Git or chat.
Run inside existing CPU Slurm allocation; no model/API/GPU/submission needed.

## Literal representation, not clinical interpretation

Use the native eleven-label ontology unchanged. Nonmeasurement observations
keep positive/negative/uncertain labels; anatomy and measurements do not become
disease labels. Observation entities modifying another observation are native
attributes, not independent findings. Incoming `modify` closure is finite.

An atom matches only casefolded, whitespace-normalized **exact observation
tokens** plus the exact set of native `located_at` anatomy/modifier tokens.
Do not map synonyms, abbreviation, CHF/medications/labs, devices, diagnoses or
CheXpert categories. This has lexical/extraction false-negative limits.
Atom keys and token sets are SHA256 hashes; retain native graph hashes, entity
IDs and word offsets so a protected reviewer can trace proposals later.

For each same-image report pair distinguish:

- Native label agreement on the exact same atom.
- Explicit positive-versus-negative **opposition proposal**, not a verified
  clinical contradiction. Native graph current/patient/temporal scope is unknown.
- Uncertain or mixed states; conflicting mentions are not resolved by voting.
- Uncertain/absent/measurement anatomy context, not a definite comparison.
- Unmentioned atom on either side, never implicitly negative or hallucinated.
- Different anatomy-context sets for the same literal observation; locations
  can coexist, so this is a detail difference, not exclusivity/contradiction.
- Different observation-modifier token sets, not a proven severity error.

No Finding is not expanded into negative labels. Empty native-observation
inventories have zero measured atoms but no clinical compatibility score.
Graph failure/empty-input status stays unavailable, not unknown or zero error.

## Output, dependency and boundary

Append prefixed diagnostic counts to the old candidate table, preserving every
old cell and row order. Each report has three dependent peers; mirrored counts
in candidate rows are not additional independent evidence. Report image-slot
and unique-image/report/pair-hash counts separately from the 960 slot count.
All reports on one CXR and repeated graph hashes remain correlated.

Write a fresh protected run: literal graph receipts, same-image pair evidence,
candidate score table, aggregate summary, sealed plan/manifest. Refuse overwrite;
directories 2770 and files 0660 in the project group. Pin worker/module/tests,
this protocol, consumed source artifacts and source audit. Recheck all pins.

This diagnostic adds measured interpretable differences, not empty score cells,
but **clinical score, current-patient scope, independent truth, fault attribution,
primary-metric qualification and automatic regeneration remain unavailable**.
It does not replace the current 14-label table, qualify EHR edges, change old
winners, fit rules/weights on cases, or authorize another model/job. Clinical
localization still needs appropriate independent validation; many agreeing
generated reports cannot supply that validation.
