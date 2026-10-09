# Frozen first-version demo handoff

This is a metadata-only packaging and exact replay task, not a new selector,
scorer, generation experiment or clinical validation. Preserve the existing
80-case DEVELOPMENT bank, its 240 image slots, 960 report slots and historical
selections. Do not rename these August outputs as the September bridge repair.

Authenticate the fixed full-bank manifest
`ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`,
score-free control manifest
`771e6ef7551f8c336df0dfe21dd3ae2b2ddfe804add9454260797ffbb1934124`
and historical selection manifest
`bd9e4a6f6a261f06305800289c3bd5f81c85f9a48fa9ae7e8c23b4c723b68bbe`.
Use only explicit sealed source caches, candidate metadata and requests.

Reuse the original score-free control implementation unchanged. Verify every
old acquisition trace and endpoint through its existing loader/prepare guard,
replay all 10,000 score-free choices, recompute all 2,000 case means, 75
method/cap/subgroup rows and 60 paired contrasts, and require exact agreement
with the sealed CSVs. Preserve every cap and random seed; never pick a best
seed, budget or favorable subset. Historical random acquisition plus scored
final choice remains distinct from genuine score-free final choice.

Build an allowlisted 960-row candidate index and 80-row case index using
hash-bound metadata only. Check the exact 3-model/1-seed/4-expert grid, immutable
EHR/fact hashes, request-prompt hashes, parent-image/report hashes and one
historical static choice per case. Do not recompute or change selections.
Authenticate synthetic EHR/fact/prompt/image/report **bytes only** without
parsing their payloads, reading pixels, displaying report text or inspecting
source MIMIC inputs/real targets. No training, inference, GPU/API call, weights
or credentials. Run inside an existing actual CPU Slurm allocation; any new
submission still requires complete-script/resource review and explicit approval.

Indices distinguish candidate slots and unique hashes; duplicates stay visible.
All unknown/uncertain and missingness semantics come from the unchanged cached
readout. EHR edges without direct comparable facts remain unavailable. Raw
explicit-state EHR-report columns do not inherit the legacy global No-Finding
adjustment; the historical selector itself is unchanged. Per-edge long-form
tables preserve positive/negative support, opposition, coverage and available
EHR denominators. Simulated call accounting is not measured GPU time/savings.

Write a fresh atomic, non-overwriting run below
`artifacts/protected/tricompose_v1_2/deliverables/`, with project-group boundary,
dirs2770/files0660, source and artifact hashes, and bilingual entry/report.
The optional sibling archive contains exactly the generated metadata-only
allowlist, no synthetic payload, raw source, checkpoint, environment or token.
No source body becomes public documentation or Git content. Recheck immutable
sources before commit and archive inventory/hashes before packaging.

Clinical qualification, best-triple claim, automatic-regeneration authorization
and old-selection-change flags remain false. The two-case actual retry accepted
0/2 replacements; published-score expert alignment has unresolved count-cell
encoding. These limitations remain explicit in the handoff. Software tests,
cache replay and byte hashes do not establish clinical method effectiveness.
