# Full-bank report verification availability: metadata-only handoff

Attach the completed unchanged report-scope gate to the fixed historical pool.
This is an operational availability sidecar, not a new scorer, selector,
clinical adjudication, held-out test or permission to regenerate. The old
80-case / 960-triple scores, facts, EHRs and winners remain immutable.

## Exact bounded inputs

Only five quote-free artifacts and their three pinned manifests are consumed:

1. `candidate_reliability_overlays/reliability_pool960_12632006_001/`:
   `candidate_score_table.csv`, `fact_reliability.jsonl`,
   `report_dependency_groups.jsonl` (960 / 13,440 / 3,360 rows).
2. `verification_gates/scope_gate_cached_12645021_001/`:
   `candidate_gate_fact_table.jsonl` (336 rows / 24 slots).
3. `report_verifier_progress/progress_scope2_12645021_001/`:
   `evidence_request_progress.jsonl` (2,072 original requests).

All paths are relative to protected `tricompose_v1_2/`. Consumed artifact hashes
are checked against sealed manifests. Do not traverse ancestor source paths or
open EHR/report/image bodies, authored reference keys, real annotations, model
weights or response text. The unchanged reliability validator checks lineage,
cached states, proxy arithmetic and correlated report dependencies.

## Join and null semantics

Match by exact triple candidate ID, finding, report candidate ID and report
SHA256. Check the cached CheXbert state agrees with its original fact row.
Require all 14 rows for each of the same 24 checked slots. Never propagate a
check to another slot merely because its text hash matches.

Append `reportgate_` fields; preserve all source cells and row order. Within
checked slots preserve scope_commit, abstain, no_model_assertion,
verifier_unavailable and outside_verifier_scope. Other slots are not_checked,
with null retained states and null decision counts, not zero disagreement,
negative predictions or successful unknowns. Scope commits retain exactly the
original non-unknown proposal, not a corrected or independently verified fact.

Requests retain their execution history. Only verify_report_assertion requests
can receive a matching source-scope sidecar; report-only evidence cannot fulfill
image-finding, EHR-observability or image/report relation requests. Preserve
unmatched consumer counts, partial/mixed availability and clinically unresolved
status. Logical finding requests are not model calls or cost savings.

The authored conditional-match result is never converted to a synthetic
candidate confidence, clinical score, new ranking weight or repair trigger.
No primary score, old winner, threshold, EHR or rule changes.

## Execution and acceptance

Worker: `TriCompose-v1.2/tools/attach_report_gate_availability.py`.
Tests: `TriCompose-v1.2/tests/test_report_gate_availability.py`, invented
metadata only. Standard-library aggregation and existing validators run in an
actual existing CPU Slurm allocation; no new submission or model/API call.

Use a fresh opaque atomic run under
`artifacts/protected/tricompose_v1_2/report_verification_availability/`.
Directories 2770, files 0660, project group only. Recheck all consumed source
hashes before commit and reject existing runs before loading inputs.

Outputs: `candidate_score_table.csv`, `fact_verification_availability.jsonl`,
`evidence_request_availability.jsonl`, `summary.json`, `RESULTS_CN_EN.md`,
`manifest.json`. Validate full fixed inventory, exact join, untouched original
cells/order, no invented scope or truth, and unchanged source hashes. Report
both checked and unchecked denominators and zero clinically resolved requests;
this handoff does not close the independent evidence gap.
