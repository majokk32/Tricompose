# Fixed-EHR image actions: completed, not accepted clinical repair

On 2026-10-08, reviewed script19 (SHA256
`feaf2b6dc8fedb7c67a268d6bde185f272dd2b37be07cd9619336debdceeb76f`)
was submitted unchanged after complete script/resource display and explicit
approval. Job **12862704** completed on debug A40 `b11-09`, exit `0:0`, elapsed
**4m02s**. One GPU, two CPUs, 32 GiB host RAM, ten-minute cap; no training,
planner, new checkpoint, API, prompt/scorer changes or original-winner replacement.

Two unchanged synthetic EHR anchors × RoentGen-v2 seed 2 / Sana seed 1,
each with its own XRV + CXRMate-single + CheXbert chain, yielded **four completed
triples, 16 validated charged requests, zero failures/retries**. Worker interval
212.143s includes startup/I/O; GPU time and saved calls are not measured.

All four rows retain EHR–CXR support **0/1**, one explicit proxy opposition;
the unchanged diagnostic preservation comparison passes **0/4**. CXR–Report
support is 3/8 and 5/8 for RoentGen, 3/8 and 3/8 for Sana. RoentGen support is
negative-only; Sana has two positive + one negative supports per row, accompanied
by one/two oppositions. Three EHR–Report comparisons are missing, not negative;
one is explicit opposition. The two anchors are not four independent cases.
These results do not establish clinical improvement, faulty-image attribution,
repair success, LLM superiority or cost savings. Do not retune thresholds/gates
on this result. Different original renderers preclude pure architecture claims.

`agent/audit_cxr_action_diversification.py` independently replayed metadata
and numeric contracts inside the existing CPU allocation: 81 reader hashes,
305 source hashes, 1,887 input pins, 49 asset stats, durable ledgers, four
own-image report chains, labels-to-receipt arithmetic, frozen comparisons and
CSV byte hash. No source narrative, report body or pixels were parsed;
inference, tokenization and report structure were not independently recomputed.
Ten new invented-fixture checks and all **3,974 V1.2 tests** passed (31.829s).
The consumed generation program/tests/protocol/script and sealed plans were
not rewritten after results.

Run:
`artifacts/protected/tricompose_v1_2/cxr_action_diversification_runs/cxr_actions2_12862704/`
(manifest `cc1448b3b64e691f00260e8139baddbae730cf3a2b4091f2718f1fa6878a040e`).

Audit:
`artifacts/protected/tricompose_v1_2/cxr_action_diversification_audits/cxr_actions2_12862704_12851223_001/`
(manifest `6f008f2436e3e83ddb42760aa07ec6d06576231e9092c5937a34759b73ec041f`).

Bilingual aggregate readout:
`artifacts/protected/tricompose_v1_2/cxr_action_diversification_reviews/cxr_actions2_12862704_12851223_001/RESULTS_CN_EN.md`.

Scientific next step: distinguish conditional generation and evidence limitations
before claiming repair; keep original EHRs/prompts, scorers and selections fixed.
No additional GPU job is authorized by this result. Protocol remains immutable
at `docs/cxr_action_diversification_protocol.md`.
