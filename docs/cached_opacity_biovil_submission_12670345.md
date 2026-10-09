# Fixed-bank opacity BioViL-T submission

Submitted 2026-10-05 after the full batch script/resource request was displayed
and the user explicitly approved with `gogogo`.

- Job: **12670345**.
- Exact script: `TriCompose-v1.2/slurm/53_cached_opacity_biovil240_v100.sbatch`.
- Script SHA256: `080a7dd1e35cf4555b580fafa6effce46b6f3e1e1369a770dc237cbad7544c26`.
- Worker SHA256: `e40eb348604fecbcbdf48b61baae4a8fcc2be76353ac67a38db86fb35a47d1d7`.
- Plan manifest SHA256: `340c0774ee5f7c593397023f4ea48eb5eb16cffaeab68467d4a6a7720184b187`.
- Resources: account `ruishanl_1185`, partition `gpu`, one V100,
  two CPUs, 8G host RAM, five-minute upper limit.
- First running observation: `d13-05`, elapsed 1 second.
- No repeat submission, generation, training, reranking or old-winner change.

Protected output target:

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_biovil_runs/biovil_opacity240_12670345/
```

The worker will append 20 finding-evidence fields to 66 existing named fields.
No generated report text is encoded. A finding-text preference is not a
calibrated clinical probability or verified fault-localization label.

## Completed execution and independent audit

Job **COMPLETED**, exit `0:0`, on `d13-05`; Slurm elapsed **41 seconds**.
Worker elapsed before serialization **23.180682 seconds**; peak allocated CUDA
tensor memory **0.608 GiB** (not the full CUDA context/device footprint).
Host MaxRSS: **1,608,332 K**.

All **240/240** existing synthetic image slots scored, zero failures or
replacement. Exactly **242** image encodings including the two fixed replays;
one eight-input text batch. Duplicate text and both replay score deltas zero.
No generation, training, selector call, old score or winner change.

Independent standard-library metadata audit verified all **559 source pins**,
four result artifacts, **63,360 original named cells**, every added score,
lineage join, state/relation, aggregate denominator, calls/replays and project
permissions. It did not open clinical text or decode any image.

Result manifest SHA256:
`6b3027a012ec146b46a29a911465bb8fe1bdded087b6ee074d5994919c89c828`.

See `docs/cached_opacity_biovil_result_12670345.md` and the protected result
`RESULTS_CN_EN.md`. Completion is an evidence-overlay result, not verified
clinical localization or authorization for automated repair/reranking.
