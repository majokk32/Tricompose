# Exact-opacity candidate sidecar / 240 张现有候选评分完成

Completed 2026-10-05, approved job **12668204**, V100 node **d13-04**.
Slurm **COMPLETED**, exit **0:0**, elapsed **00:00:25**. Worker runtime
**20.750776 seconds**, including verification/loading/inference/table building
before final serialization. The ten-minute resource request was an upper cap,
not actual runtime. This was scoring existing PNGs, not generating 240 images.

## Completion, provenance and unchanged outputs

- **240/240 existing synthetic image slots scored**, 240 XRV calls, zero
  failures/replacements; one frozen model load, zero training/generation calls.
- **80 fixed EHR anchors / 960 report-triple slots / 428 distinct report
  artifact hashes** retained. These are not 960 independent patients.
- New CSV has **66 columns**: all **50 original columns and 48,000 cells**,
  including row order, are unchanged; **16 `opacity_` diagnostic columns** added.
- Existing EHRs/facts/prompts, images, reports, thresholds and historical
  winners are unchanged. No selector, router, repair or clinical verdict ran.
- All **300 source pins**, five output-artifact hashes, full raw-score/label/
  relation arithmetic and per-model denominators independently verified.
- Output and per-job runtime directories/files retain 2770/0660, project group
  exposed as 65534 by NFS. The protected output remains Git-ignored.
- No EHR/report/prompt bodies, source patient inputs, real MIMIC targets or
  benchmark patient records were opened into chat. No manual image inspection.
- Peak PyTorch **allocated tensor** memory 0.043 GiB is not total CUDA/context
  memory. The small sampled batch-wrapper MaxRSS is not used as Python-worker
  host-memory consumption.

```text
artifacts/protected/tricompose_v1_2/candidate_opacity_runs/opacity_pool240_12668204/
  image_scores.json
  candidate_score_table.csv
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Result-manifest SHA256:
`778d8426d87810a3408281baf5486893f3d56053265c22f619dc57061ee1bc94`.
Plan-manifest SHA256:
`650f129ce6b20e6e278bd937accfd348cef4e68328fcf793ab722caa55d7323a`.
Exact approved script SHA256:
`993f08b0825272a107715f775ae3ea478c92c3bce5fdadac4b76ba60318bee32`.

## Score-definition diagnostic / 评分定义的影响

Both new readouts use the unchanged checkpoint, same model call and fixed
threshold **0.5**. Scores are XRV operating-point-normalized outputs, **not
calibrated disease probabilities**.

| Definition | Positive image slots | Negative image slots |
| --- | ---: | ---: |
| Exact `Lung Opacity` head | **112** | **128** |
| Same-call `max(Lung Opacity, Infiltration)` | **187** | **53** |
| Historical max-head cache | 187 | 53 |

**75/240 image-slot states differ** between exact and same-call max heads;
these are necessarily max-positive / exact-negative at the same threshold.
Historical max-head and new same-call max-head states match on **240/240**
slots. The latter checks state-level reproduction, not byte-identical raw
scores or proof of an old preprocessing fingerprint, which was not recorded.

The sealed RICORD-1C result supports exact-head opacity discrimination on
its selected 50-case reference. It does **not** prove that these 75 synthetic
cases are false positives, validate their pneumonia status, establish image
fidelity to EHR, or assign candidate clinical accuracy. No threshold was fitted
or best-scoring definition selected on this synthetic pool.

## CXR–Report opacity agreement / 图文阴影代理一致性

Reports retain the original **cached CheXbert** opacity proposals. They are
not independently adjudicated current-image truth. Positive/negative pairs
only are comparable; unknown/uncertain are not negative or successful support.

| Diagnostic definition | Proxy supports | Proxy oppositions | Comparable slots | Not comparable | Agreement within comparable slots | Coverage over all 960 slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Exact opacity head** | **173** | **95** | **268** | **692** | **64.55%** | **27.92%** |
| Same-call max head, secondary | 244 | 24 | 268 | 692 | 91.04% | 27.92% |
| Original max-head cache, secondary | 244 | 24 | 268 | 692 | 91.04% | 27.92% |

The lower agreement is a **measurement-definition change**, not a degradation
of the unchanged generators, proven repair failure, clinical factual error rate
or evidence that exact-head scores are more accurate on these synthetic cases.
Max-head/report agreement alone was not validated as clinical truth. Reference
labels, domain transport and report assertion qualification remain limitations.

### Aggregate by existing CXR model

Each model has 80 image slots and **320 correlated report/triple slots**.
The percentages below are descriptions, not model rankings or clinical accuracy.

| CXR model | Exact opacity positive / negative images | Supports / comparable report slots | Proxy agreement | Comparable coverage |
| --- | ---: | ---: | ---: | ---: |
| CheXGenBench PixArt | 27 / 53 | 37 / 60 | 61.67% | 18.75% |
| CheXGenBench Sana | 37 / 43 | 46 / 64 | 71.88% | 20.00% |
| RoentGen-v2 | 48 / 32 | 90 / 144 | 62.50% | 45.00% |

### All existing model/expert combinations

Each row has **80** fixed-EHR report slots; same-image reports share image
evidence. Do not infer a winner from agreement without its coverage denominator.

| CXR | Report expert | Supports / comparable | Proxy oppositions | Not comparable | Proxy agreement | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PixArt | CheXagent-2 | 7 / 9 | 2 | 71 | 77.78% | 11.25% |
| PixArt | CXRMate-single | 14 / 26 | 12 | 54 | 53.85% | 32.50% |
| PixArt | LLaVA-Rad | 13 / 22 | 9 | 58 | 59.09% | 27.50% |
| PixArt | MAIRA-2 | 3 / 3 | 0 | 77 | 100.00% | **3.75%** |
| Sana | CheXagent-2 | 17 / 26 | 9 | 54 | 65.38% | 32.50% |
| Sana | CXRMate-single | 6 / 7 | 1 | 73 | 85.71% | 8.75% |
| Sana | LLaVA-Rad | 14 / 22 | 8 | 58 | 63.64% | 27.50% |
| Sana | MAIRA-2 | 9 / 9 | 0 | 71 | 100.00% | **11.25%** |
| RoentGen-v2 | CheXagent-2 | 48 / 77 | 29 | 3 | 62.34% | 96.25% |
| RoentGen-v2 | CXRMate-single | 20 / 21 | 1 | 59 | 95.24% | 26.25% |
| RoentGen-v2 | LLaVA-Rad | 12 / 28 | 16 | 52 | 42.86% | 35.00% |
| RoentGen-v2 | MAIRA-2 | 10 / 18 | 8 | 62 | 55.56% | 22.50% |

100% in 3 or 9 comparable slots is not comprehensive performance. Differences
in report assertion availability/extraction scope, image content and report
artifact duplication confound comparisons. No p-value/superiority or quality
claim is made from these correlated, post-hoc development counts.

## EHR-related edges / 不制造完整 triple 总分

All **80 EHR opacity states are unknown under the existing cached extraction
rules**. This does not mean the original EHRs have no diagnoses or conditions;
it means this specific observable finding lacks direct cached evidence.

For this opacity-only sidecar, **960/960 EHR–CXR and EHR–Report comparisons are
not comparable**, not zero consistency or clinical negatives. Pneumonia, CHF,
diuretics and absent EHR mentions are not promoted to opacity truth. Other
historical finding/quality/edge fields remain untouched, not requalified by this
benchmark. No complete, clinically validated triple score is available here.

## Practical next boundary

The new CSV is ready to inspect as a **separate diagnostic score table**.
Use exact/max columns and coverage together; retain clinical accuracy and faulty
modality as NA. The readout establishes sensitivity to score definition, not
that automatic selection/regeneration works.

Before granting clinical fault-localization or repair credit, qualify the
specific remaining image findings and report assertion scope, and define an
evidence-preserving endpoint on facts available to each modality. Candidate
selection and held-out endpoint must not be conflated. Do not tune thresholds,
replace fixed EHRs, choose high-agreement low-coverage outputs or launch more
generators from this result alone. Any next GPU job requires a new complete
script/resources and explicit approval.
