# Frozen scorer validation

This directory contains training-free diagnostics, not a validated clinical
repair controller. All generators/scorers stay frozen. CXR generation in
TriCompose uses EHR-derived radiology text, not a native structured-EHR-to-CXR
checkpoint.

## Included benchmark entry points

| Entry point | Scope | Execution status |
|---|---|---|
| `biovil_matched_pairs.py` | Matched versus mechanically cross-patient CXR/report retrieval | Historical completed pilot; mismatches are not adjudicated clinical errors |
| `audit_official_report_gold.py` | Official report-label schema, state and linkage coverage only | Completed metadata audit; no text/image model validation |
| `run_official_report_benchmark.py` | Frozen CheXbert versus manual report labels, with missingness and scope | Job 12594397 completed; 12.025 seconds, 1.227 GiB allocated peak |
| `biovil_fact_polarity.py` | Same image, authored positive/negative finding statements, three fixed templates | Job 12597637 completed; 45.827 seconds, 0.576 GiB allocated peak |

Complete Slurm scripts are under `../slurm/`. Scripts 26 and 27 received
separate explicit approval before submission. Script 25 records the historical
metadata audit in an existing CPU allocation; it is not a login-node command.
Every **new** submission needs the complete script/resource request shown and
explicit approval again. Do not resubmit completed scripts to existing runs.

## Source dependencies versus local assets

The committed source includes the current report-scope helper and its import
dependencies in `../benchmarks/` and `../src/tricompose_v12/`. Their presence
does not mean that the other prototype commands are clinically validated or
authorized for execution. The four-finding guard only retains/vetoes assertions;
it cannot promote labels, adjudicate an image or authorize regeneration.

Existing shared utilities remain in `TriCompose-v1.0/eval/report_v1_1/`.
The BioViL adapter's Slurm guard is included; unrelated V1.0/V1.1 work-in-
progress changes are deliberately not part of this publication.

Local model repositories/environments, checkpoints, datasets and protected
outputs are **not in Git**. The scripts document the CARC paths and bind assets
by hashes; a code clone alone is not an inference environment. In particular:

- BioViL-T uses the existing CheXGenBench metrics environment, local frozen
  image/text checkpoints and read-only vendored `health_multimodal` runtime.
- CheXbert uses the existing CXRMate environment/adapter and local CheXbert/
  BERT assets. It is not an independent final evaluator merely because it is
  frozen; checkpoint training overlap remains unresolved.
- Source manifests, cached references and all results stay under the project
  privacy boundary. No raw MIMIC source report/EHR text, image, patient key or
  per-row real reference labels are included in this repository publication.

The runtime paths are intentionally CARC-specific. The fixture tests require
only the Python standard library, not these model/dataset installations.

## Fixture tests

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=TriCompose-v1.2/real_validation:TriCompose-v1.2/src:TriCompose-v1.2/benchmarks:TriCompose-v1.0/eval/report_v1_1 \
python -m unittest discover -s TriCompose-v1.2/tests -p 'test_*.py' -v
```

The scoped publication includes tests for the official schema audit, official
report diagnostic, fact-polarity probe, real-pair helpers, assertion gate and
cached evidence scope: **151 fixture tests pass** for these six groups. The
full CARC working-tree V1.2 suite also has other
unpublished development tests; do not equate its test count with the smaller
published subset. Tests use invented fixtures, never real source inputs.

## Protected results and next gate

Readable results on CARC:

```text
artifacts/protected/tricompose_v1_2/real_validation/
  official_chexbert_review_12594397_001/RESULTS_CN_EN.md
  real_biovil_polarity_review_12597637_001/RESULTS_CN_EN.md
```

The candidate diagnostic overlay is under
`artifacts/protected/tricompose_v1_2/diagnostics/candidate_report_diagnostic_12594397_001/`.
These links are workspace references, not files distributed through GitHub.
No old winner/score table was overwritten, no threshold or template was fitted
on test results, and no targeted-regeneration gate was cleared.

The real-image polarity reference is still report-derived. The local metadata-
only search found no independent image-level reference in the inspected dataset
directories, and existing human-review readiness remains pending. This is a
scoped audit, not a claim that no such resource exists anywhere on CARC.
Independent image/report adjudication or an approved image-annotated benchmark
is the next clinical-validation requirement. Do not replace it with correlated
model votes or publish mechanical intervention labels as clinical truth.

See [the frozen polarity protocol](../../docs/biovil_fact_polarity_protocol.md).
