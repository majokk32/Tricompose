# Observed action effects: a joined diagnostic, not another score

This CPU-only, post-hoc DEVELOPMENT readout joins already inspected synthetic
results from report job12851790 and image job12862704. It is not prospective
preregistration, a new independent benchmark, or a randomized causal estimate.
No clinical bodies/pixels, model inference, API or GPU job are needed.

Retain the same two fixed EHRs and all six observed comparisons:

- Two same-image switches to LLaVA-Rad, each against its own retained seed1
  report. Baseline experts differ between cases.
- Two RoentGen-v2 seed0→seed2 contrasts, with CXRMate-single fixed.
- Two RoentGen-v2 seed0→Sana seed1 contrasts, with CXRMate-single fixed.
  These change generator, original renderer, native settings AND seed.

Different action families have different images/baselines and cost scopes.
They cannot be ranked as a controlled report-versus-image action trial. Six
pairs are repeated observations of TWO EHRs, not six independent patients.
Old acquisition/planning cost remains sunk, not free or refunded. The joined
table counts only downstream attempts in the two named source jobs.

Use authenticated run manifests and their existing CPU postflights. Check
exact row identity, unchanged EHR/facts, numeric receipt arithmetic, original
same-image or cross-image gate comparison, four completed image slots and
actual charged attempts. Preserve positive/negative support, opposition,
coverage and missingness separately. No new weighted score, threshold, head
mask, gate, clinical error attribution, router or winner selection is added.

Unknown remains unknown. Raw XRV scores are not clinical probabilities, and
CheXbert/report agreement is not independent image truth. Improvement on an
uncalibrated proxy is not clinically accepted repair. Byte hashes do not prove
semantic conditioning or image anatomy. No confidence interval is claimed
from two development anchors.

Entry: `TriCompose-v1.2/agent/summarize_fresh_action_effects.py`.
Output: a NEW exclusive run under
`artifacts/protected/tricompose_v1_2/fresh_action_effects/`, project2770/0660,
containing `action_effects.json`, `action_effects.csv`, `summary.json` and
`manifest.json`. Never overwrite either original run or selected triple.

Tests use invented fixtures only: action scope, unchanged EHR/scorers, original
gate, positive/negative decomposition, null handling, duplicate/dropped rows,
charged costs and CPU guard before reads. The existing postflights already
replayed inference journals; this join does NOT independently rerun tokenizers,
image observers, report structure or model inference.

Next experiment, if separately approved, should diagnose conditioning versus
observer limitations. Do not retrospectively relax a failed gate or keep
calling reports to remove an immutable EHR–image proxy conflict. A controlled
invented-text interface test must remain separate from the unchanged EHR
cohort; changing a prompt in such a test is not an EHR-conditioned repair gain.
Any new GPU run still requires complete script/resource display and fresh
explicit approval.
