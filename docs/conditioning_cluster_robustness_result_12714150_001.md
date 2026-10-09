# Frozen-bank conditioning robustness: result

Date: 2026-10-06. Completed cache-only analysis and independent numerical audit.
All80 EHRs and all960 candidates remain unchanged. This is exploratory
development-bank evidence, **not independently validated clinical superiority**.

## What the duplicates actually are

Groups were formed before analyzing method means, using only the ordered tuple
of three model-specific prompt SHA256 values. No EHR, prompt or report text,
image pixels, clinical label or score was used to choose groups. Different
EHRs were not discarded or augmented.

| Metadata quantity | Result |
| --- | ---: |
| Fixed EHR cases / distinct EHR hashes | 80 / 80 |
| Candidate slots | 960 |
| Exact three-model conditioning signatures | 49 |
| Case slots beyond one per signature | 31 |
| Group sizes | 41×1,4×2,2×3,1×5,1×20 |
| Groups sharing an identical three-model image-hash vector | 49/49 |
| Unique prompt hashes per CXR model | 49 |
| Unique image hashes per CXR model | 49 |

“Identical vector” means each model's image hash is identical **across cases in
the same group**, not that Sana, PixArt and RoentGen produced the same image.
This is byte-identity metadata, not anatomical or clinical inspection. Fixed
seed0 and repeated prompts reuse outputs; this does not independently diagnose
a model fault. Exact prompt groups are not proven clinical-EHR equivalents,
and49 groups are not asserted to be49 independently sampled patients.

## Preserve both estimands

The case-weighted estimand retains the original EHR multiplicities. The
equal-conditioning estimand averages within a prompt group, then gives each
available group equal weight. Both are useful sensitivity views; equal weighting
changes the target distribution and is not automatically a corrected truth.
Bootstrap draws resample whole groups, preserving within-group case dependence.
Historical acquisition/final-choice seeds were already averaged inside EHRs.

At full cap30, raw secondary BioViL cosine means are:

| Existing method | Case-weighted | Equal-conditioning | Simulated calls, equal-conditioning |
| --- | ---: | ---: | ---: |
| Fixed Sana+MAIRA-2 | 0.6186 | 0.5677 | 4.0000 |
| Score-free final random choice | 0.5082 | 0.5092 | 30.0000 |
| Random acquisition + scored final | 0.6122 | 0.6266 | 30.0000 |
| Static reranking | 0.6122 | 0.6266 | 30.0000 |
| Targeted heuristic replay | 0.6075 | 0.6188 | 28.2041 |

These are cached proxy readouts and simulated charges, not new model execution,
clinical accuracy, prospective runtime or measured GPU savings. Static and
historical random-scored equivalence at full inventory is expected.

## Paired differences: what is supported, and what is not

All budgets4/8/12/20/30, all11 metrics and all five contrasts are retained.
Below are full-cap examples, **not a selected best budget**. These percentile
intervals are descriptive, not adjusted for multiple comparisons or scorer bias.

| Signed difference | Equal-conditioning mean | 95% group-bootstrap interval | Available EHRs / groups |
| --- | ---: | --- | --- |
| Static − score-free, BioViL | +0.1174 | [0.0552,0.1751] | 80/49 |
| Static − fixed, BioViL | +0.0588 | [-0.0329,0.1466] | 80/49 |
| Targeted − static, BioViL | -0.0078 | [-0.0223,0.0024] | 80/49 |
| Static − score-free, CXR-report support/known | +0.0994 | [0.0645,0.1348] | 80/49 |
| Static − score-free, CXR-report proxy opposition/known | -0.0995 | [-0.1108,-0.0885] | 80/49 |
| Static − score-free, CXR-report coverage/known | -0.0001 | [-0.0372,0.0368] | 80/49 |
| Static − score-free, EHR-CXR support/known | +0.2629 | [0.1200,0.4001] | 8/7 |
| Static − score-free, EHR-report support/known | +0.1486 | [-0.0059,0.3260] | 8/7 |

The raw XRV/CheXbert support/opposition measures reuse selection-related
uncalibrated proxies. Improvements there are not independent clinical proof.
BioViL was not used for those original routing decisions but remains an
unqualified secondary metric. The72 unavailable EHR-edge cases remain null,
not zero, successful, clinically normal or excluded from the fixed cohort.

Important uncertainty: full-cap static-minus-score-free BioViL has a
**case-weighted** group-bootstrap interval[-0.0100,0.1774], unlike the positive
equal-conditioning interval above. Small-budget equal-conditioning comparisons
can also include zero. Neither result may be suppressed to strengthen a claim.
Static-minus-fixed intervals include zero at every tested nontrivial cap.
Targeted-minus-static likewise does not establish a stable advantage. Greater
call budget does not necessarily monotonically improve this secondary metric.

Thus the defensible interpretation is narrow: final selection can improve some
cached proxy comparisons over random final choice, with dependence on budget
and estimand. **The adaptive-repair contribution has not been established.**
Do not pick a best budget after seeing these development results and call it
an independently verified method setting.

## Execution and outputs

Existing actual CPU Slurm12714150; numeric phase2.235919seconds. No training,
model inference, GPU/API call, new download or Slurm submission. Eleven new
invented-fixture tests passed; full V1.2 suite2,295tests passed in14.663seconds.
Independent stdlib audit, without importing worker/statistical implementations,
verified275method means,275paired contrasts,550intervals,12source pins and
protected modes/Git exclusion. Old first-version archive hashes remain unchanged.

Protected output:

```text
artifacts/protected/tricompose_v1_2/conditioning_robustness_runs/conditioning_groups_12714150_001/
  case_groups.csv
  method_means.csv
  paired_contrasts.csv
  paired_contrasts.json
  summary.json
  RESULTS_CN_EN.md
  manifest.json
```

Run manifest SHA256:
`0d6e3ecf79c5a99cd500ee74cb3fee23695cd447ff768bd1c848593d04ce985c`.
Plan manifest SHA256:
`d0a23513c5e42ab879634002e0813b5d56feea108c5e9a2284e2acbf1b0544db`.
Worker SHA256:
`33956880b70f49c8f22e913c61793a5de5f0e7b78a25dcea03e2250cf08a19f8`.
Independent audit SHA256:
`35b8b890addf8765c13408a927625f1ae80a50e2761b2cc1a78c5089b70c126e`.

## Scorer-qualification follow-up

The [official ReXVal documentation](https://physionet.org/content/rexval-dataset/1.0.0/)
defines explicit counts per study/candidate/reader/category/significance and
requires PhysioNet credentialing and its own DUA. Its two expected CSV files
were not found in the inspected workspace artifact/runtime/experiment roots.
No credential was read, no restricted report was opened, and no data was
downloaded from an unofficial mirror. This is not a global CARC search result.
The [RadEvalX release](https://physionet.org/content/rad-eval-x/1.0.0/) still lacks
an explicitly verified blank-cell decoding rule in our bounded source audit.
Previous null and hypothetical-zero analyses remain separate and unchanged.

Use this robustness result in the engineering demo, while keeping independent
local-scorer qualification and clinically validated repair as explicit remaining
gates. No selection, qualification or regeneration authorization flag was enabled.
