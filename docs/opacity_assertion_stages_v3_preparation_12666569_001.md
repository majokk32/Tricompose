# Opacity stages V3 format correction: prepared, not submitted

V2 job **12677127** completed on A40 in **2 minutes**. Independent audit passed;
the method did not: 47/48 primary staged outputs were unavailable because the
locator returned bare arrays instead of the required object. The V2 prompt had
contradictory empty/nonempty output instructions. See the immutable result and
`docs/opacity_assertion_stages_v2_result_12677127.md`; no failures are rescued.

Created new V3 files only. The locator now consistently requests the named
JSON object and explicitly forbids a bare array. Original strict decoding,
segment boundaries, polarity prompt, four-state reduction, authored references,
model/checkpoint/loader and cost policy stay unchanged. V2 interface/worker/
tests/protocol/plan/result hashes remain unchanged. No GPU inference, new
sbatch, candidate-bank read, training, score change, selector or repair.

## Tests and preparation audit

**12** new format-only contract tests; full suite **2,015 tests passed in
11.831 seconds**. Tests mechanically verify that the copied worker changes
only declared version/interface/source bindings and the correction-disclosure
flag. `git diff --check` and new batch syntax check pass.

Existing CPU Slurm **12666569** prepared a fresh protected plan. Independent
audit checked **26** source pins, three artifacts, source hashes/mechanical
offsets/four-state/family/cost inventories, 2770/0660 group modes and preserved
six known controls. **inputs.json and references.json are byte-identical to
the V2 plan**. Polarity and baseline prompt hashes/runtime/policy are unchanged;
only locator prompt/interface version and disclosure change. This remains
investigator-known authored development, not independent or held-out accuracy.

## Sealed plan and proposed job

```text
artifacts/protected/tricompose_v1_2/opacity_assertion_stages_plans/authored48_stages_v3_12666569_001/
  inputs.json
  references.json
  plan.json
  manifest.json
```

Plan manifest SHA256:
`238a8e44133c2edc9c69d460a49fcbfbb5e62ef38ed64672699a808eaa088551`.
Worker SHA256:
`3e5aa29dbbac72a61a31f4a9f0298552c431fc80139fd269b6106f8fe0208b60`.
Complete script SHA256:
`ea8f2154574311719de2deba67959c93e3e0212f40a58e23f9ecc0cd6b1f9dd5`.

Script `TriCompose-v1.2/slurm/56_authored_opacity_format48_a40_debug.sbatch`:
**debug / one A40 / two CPUs / 32G host RAM / 15-minute upper limit**, no forced
node. Latest `noderes -f -g` showed two free debug A40 slots; not a reservation.
At most 148 attempted calls, all failed/skipped phase accounting retained.
Show the complete script/resources and get explicit subsequent approval before
submission. The prior V2 approval does not authorize submitting V3.

Expected new result:
`artifacts/protected/tricompose_v1_2/opacity_assertion_stages_runs/authored48_stages_v3_<job_id>/`.

All newly consumed V3 code/tests/protocol/fixtures/plan are now immutable.
Even improved authored metrics or a passed narrow gate would not authorize
clinical localization, new scores, selected triples or automatic regeneration.
