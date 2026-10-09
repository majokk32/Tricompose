# Full-bank report availability handoff: completed

This is the result of the unchanged
[availability protocol](report_gate_availability_protocol.md). No new scorer,
clinical truth, selection policy, ranking or targeted regeneration is introduced.

## What is now available

The full historical candidate CSV keeps every existing score column and adds
report-check availability. Candidate IDs and report hashes join the fixed
24-slot gate subset to the full 960-slot bank; identical text in another slot
does not inherit verification. Null decision counts mean not checked, not
zero contradictions or successful unknowns.

| Fixed inventory | Count |
|---|---:|
| Synthetic EHR cases | 80 |
| CXR slots | 240 |
| Candidate triples | 960 |
| Original candidate cells preserved | 65,280 |
| Original finding rows preserved | 13,440 |
| Checked triple slots | 24 |
| Unchecked triple slots | 936 |
| Slots with retained report assertions | 9 |

| Finding-level availability | Rows |
|---|---:|
| Scope commit | 17 |
| Abstain | 41 |
| No model assertion | 38 |
| Outside the verifier scope within checked slots | 240 |
| Candidate slot not checked | 13,104 |

The 17 retained assertions are the same source-scope proposals, not newly
established clinical facts. This handoff does not validate image truth, EHR
fidelity or which modality is erroneous. Neither missing EHR evidence nor
unchecked report evidence is manufactured into negative findings.

## Original requests remain unresolved

| Report-gate sidecar status | Existing logical requests |
|---|---:|
| Scope commit | 1 |
| Abstain | 7 |
| Outside verifier scope | 2 |
| Not checked by this sidecar | 2,062 |
| Total | 2,072 |

The eight previously technically completed, in-scope report reviews therefore
do not mean eight usable assertions: this unchanged guard retains one and
withholds seven. None is clinically resolved. The old execution_status and
reportcheck progress fields remain unchanged. Report-only checks cannot fulfill
image-finding, image/report relation or EHR-observability requests; source-hash
matches do not establish clinical truth. Logical requests are not model calls.

## Files and audit

Protected run, relative to `artifacts/protected/tricompose_v1_2/`:
`report_verification_availability/reportgate_pool960_12645021_001/`.

- `candidate_score_table.csv`: original scores and appended availability.
- `fact_verification_availability.jsonl`: original facts and appended availability.
- `evidence_request_availability.jsonl`: original requests/history and appended availability.
- `summary.json`, `RESULTS_CN_EN.md`, `manifest.json`: counts and provenance.

Manifest SHA256:
`efb507376ce8cd8b23b28a6b55d719c7725a66fb9395f79bc70b32e76121299a`.

Independent audit verified all 17 source bindings and five artifact hashes,
every original candidate/fact/request cell and ordering, exact scope joins,
all unchecked and outside-scope denominators, unchanged historical full-bank/
gate/progress manifest hashes, and project-only directory/file permissions
(2770/0660). The full suite passed **1,254 tests**, including 21 new invented
metadata tests. Processing used existing CPU allocation 12645021; no new
Slurm submission, model/GPU/API call, raw patient source, report/image/EHR-body,
reference key, model weight, training or tuning was involved.

Primary score changes, winner changes, fixed-EHR changes, clinically resolved
requests and authorized regeneration all remain zero/false. This completed
engineering handoff keeps coverage visible; it does not close the scientific
evidence gap or justify clinical improvement claims.
