# Opacity report text check: approved submission 12675347

The user explicitly approved the previously displayed complete batch script and
resource request with `go`. Submitted exactly once at **2026-10-05 22:59:51 UTC**:

```bash
sbatch TriCompose-v1.2/slurm/54_opacity_report_text16_v100.sbatch
```

Slurm job **12675347**, account `ruishanl_1185`, partition `gpu`: **one V100,
two CPUs, 32G host memory, ten-minute upper cap**, no forced node. It was
initially pending for priority and subsequently ran on `d14-10`. No resource
override, duplicate submission or modification to the approved script.

## Frozen bindings

- Batch SHA256:
  `8f57c4442efe6ec2465e0649ecbfba68a79567099b1ba5d8758cb1ccfc1e563b`.
- Worker SHA256:
  `3f5493bbd5b4f4d5c661f308ec87eaae7dd1e5592f77b2f9a1ad9c1f8979606f`.
- Plan manifest SHA256:
  `a8d0530a6c646b52427c30caf35be0f076027543d96084d3dc43913deff095cc`.
- Model weight SHA256:
  `26fa644c8e61aa8185da5055b61aa28d6897fee102c2f43173acab657c4257a1`.

Sixteen unique synthetic report texts, 36 retained candidate contexts, two
predeclared replays and six wholly authored controls. At most 24 attempted
text-only frozen Qwen calls. No image/EHR/old label/score/ID is supplied to the
model; no generation, training, new selection, automatic repair or old-score
replacement. Source texts, raw responses and quoted evidence stay protected.
The task checks report assertions, not clinical image correctness.

Expected new output:

```text
artifacts/protected/tricompose_v1_2/report_opacity_check_runs/opacity_text16_12675347/
```

Independent post-run hash/quote/state/denominator/lineage/permission auditing
is required before reporting results. Preparation and consumed sources remain
immutable. Completion is recorded separately, not retroactively in the sealed
preparation/protocol.
