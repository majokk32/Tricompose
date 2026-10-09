# Score-free final-choice control on the frozen bank

Exploratory post-hoc DEVELOPMENT control, not a new clinical test. The user
cannot provide human feedback. Keep all 80 fixed synthetic EHRs, 240 CXR slots,
960 report/triple slots, old acquisition histories, old winners and scores.
Use only bounded sealed cache metadata in an existing CPU Slurm allocation.
No source EHR/report text, image/weight reads, model/API calls, training,
threshold fitting, new generation or Slurm submission.

## The comparison being isolated

The historical `random` method randomizes candidate acquisition then applies
the same score-based reranking. It is **not** score-free final selection.
At full inventory its equality with static selection is expected.

The new `random_acquisition_score_free_final` copies each of that method's
2,000 acquisition trials unchanged: 80 cases × five caps (4/8/12/20/30) × five
acquisition seeds (0..4). Hold each observed candidate set and charged calls
fixed. Filter using the **same cached artifact gate only**, not any clinical
score, classifier label, model name, runtime tie-break or BioViL endpoint.
Draw uniformly among eligible observed candidate IDs. Do not see unseen IDs.

Use five independent final-choice seed settings (0..4), retain all outcomes,
and domain-separate final-choice randomness from acquisition randomness:
SHA256 of `score-free-final-choice-v1|case_id|cap|acquisition_seed|final_seed`
seeds the Python standard-library PRNG. Choose from sorted unique eligible
IDs, not score order. Never choose the best final seed. This produces exactly
10,000 cached-control trials, not 10,000 new patients or model calls.

Costs intentionally include the same historically simulated generator/XRV/
CheXbert calls even though final choice does not use those clinical scores.
This **selector-only ablation** isolates selection, not scorer-call avoidance,
a cost-optimal random pipeline or a prospective speedup. Copy charges and
bind the source trial/hash; preserve image reuse, cap, terminal reason and
observed candidates. Invalid/no-eligible outcomes remain unavailable.

## Evaluation after each choice

Attach the complete immutable full-bank BioViL-T endpoint and raw edge count
readouts only after final-choice IDs are fixed. No endpoint or availability
chooses a candidate. Validate fixed-EHR hashes, exact triple/image/report
lineage, full 80×3×4 inventory, source prefixes and same-case traces. Check
every reserved simulated cost and original trial inventory. No fourteen-head
legacy/fresh eight-head mixing, label changes or EHR enrichment/drop.

Compare new score-free choices against unchanged fixed, historical random
acquisition + scored final, static rerank and targeted-heuristic endpoints.
Evaluate every method through the same full-bank endpoint/raw-count cache;
this does not revise their historical selection scores. Keep each of the
three edges separate. Preserve raw known/comparable/support/opposition counts,
positive/negative support and missingness; no no-comparison perfect score.

Average all seed settings within each fixed EHR first, then report cohort
means and explicitly available-case denominators. Paired contrasts use the
same EHR at the same cap; only the random-acquisition/scored-final contrast
also has identical acquisition and expenditure. Other methods can spend
different calls under the same cap; report that difference. Report all five
budgets and direct-EHR/no-direct-EHR groups (cached 8/72, not old 15/65 prompt
tiers), rather than a best budget/subgroup. Seed replicates are not independent
patients. No significance test, clinical truth, confirmed fault/repair accuracy
or measured GPU savings is claimed.

## Artifacts and reproducibility

Worker: `TriCompose-v1.2/tools/score_free_random_control.py`.
Tests: `TriCompose-v1.2/tests/test_score_free_random_control.py`.
Fresh protected `automatic_replays/score_free_random_<opaque_run_id>/` contains
control outcomes, case-level means, all-budget method/subgroup comparisons,
paired contrasts, summary, readable bilingual report and sealed manifest.
Outputs commit atomically; refuse overwrite; project modes 2770/0660.
Bind only explicitly consumed metadata/helpers/tests/protocol, never recurse
into image/weight/source-body trees. Old scores, actions and winners stay
immutable. Fixture tests and exact replay are engineering checks only.
