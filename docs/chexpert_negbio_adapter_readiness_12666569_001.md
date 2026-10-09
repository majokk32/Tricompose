# CheXpert/NegBio readiness: native output contract verified, full parser pending

## 本轮完成 / Completed

Official source/rules now live at workspace root `chexpert-labeler/` and
`NegBio/`, at the fixed revisions in `chexpert_negbio_adapter_protocol.md`.
These are source-only sparse checkouts, not deployed/qualified full parsers.
Clinical example reports, upstream examples/tests and images are not checked
out. Both repositories are ignored by the private TriCompose Git wrapper.
No upstream code, dependency environment or existing result was changed.

The new output contract saves native fourteen-category labels separately from
a TriCompose mention-conflict view. It records original/cleaned source hashes
and exact cleaned-text Unicode offsets without text bodies. Missing native
mentions remain unknown; dependency/detector failures remain unavailable/null.
It does not infer current anatomy or temporal scope, expand No Finding into
negatives, or authorize scoring, rejection, selection or regeneration.

Implementation:

```text
TriCompose-v1.2/src/tricompose_v12/chexpert_negbio_contract.py
TriCompose-v1.2/tools/audit_chexpert_negbio_readiness.py
TriCompose-v1.2/tests/test_chexpert_negbio_contract.py
```

## 实际检查 / Actual component checks

Within the existing CPU Slurm allocation 12666569, run the unchanged official
Aggregator using authored mock annotation objects. No report parser, GPU,
patient record, candidate body, model download, environment install or API was
used. NumPy/tqdm came from the existing read-only ConText environment; this is
**not reproduction of the official legacy parser runtime**.

- All **40** ordered label sequences, length zero through three, passed the
  native aggregation branch checks.
- All **6** special component cases passed: device-only No Finding behavior,
  positive/uncertain CHF propagation, no propagation for negated CHF,
  negation-key precedence and explicit No Finding annotation handling.
- **16/40** label sequences differ between native aggregation and the disclosed
  conflict view. This is a combinatorial contract difference, **not clinical
  accuracy, 40 patients, an improved metric or independent clinical evidence**.
- Actual pinned CLI defaults are full-report mode: `sections_to_extract=[]`,
  `extract_strict=False`. The source class name does not imply impression-only
  processing. All-three polarities yield uncertain by native branch order;
  positive does not always take precedence.
- **35** invented contract tests passed; full V1.2 regression passed **2,121
  tests in 11.564 seconds** using the available read-only CPU environment.
- Independent standard-library audit verified 117 source pins, three result
  artifacts, the exhaustive matrix, special cases and protected permissions.
  Artifact directories are mode 2770 and files 0660; the NFS group is exposed
  as nobody/65534 at the project-group access boundary.

The previous command referencing `yikeyang_medim_ehr_joint/work/.venv/bin/python`
could not run because that path is unavailable now. The regression result above
comes from the actual current `runtime/venvs/report-context-v12-12576792` Python,
not an assumed historical environment. Initial source readiness rejected an
upstream tracked PLY grammar diagnostic file; the allowlist was corrected before
the successful run was frozen. No old result directory was overwritten.

Output:

```text
artifacts/protected/tricompose_v1_2/chexpert_negbio_readiness_runs/native_contract_12666569_001/
  component_checks.json
  dependency_readiness.json
  summary.json
  manifest.json
```

Manifest SHA256:
`0d73939c1fdba899546c81de4fd38634e4c5600e3a6b67a9177ee8c829e7a5b0`.
Audit: `.tmp/audit_chexpert_native_contract_12666569_001.py`.

The complete-bank endpoint manifest remains unchanged:
`ed44ddf1082a24408e7e1285c901ba240dbe43e1c33b20a60aef21e50f94a40c`.
Old labels, scores, selectors, winners and failed qualification gates remain
unchanged. The new adapter has not processed the existing 960 report slots.

## 尚未完成 / Pending dependency parser

The inspected current environment is Python 3.11.9, whereas the official
environment file pins Python 3.6.7. BioC, pandas, BLLIP, StanfordDependencies and
JPype are unavailable here. Java is not on the current PATH. The explicit
workspace GENIA+PubMed model directory and workspace NLTK punkt/tagset/wordnet
resources do not exist. This is scoped runtime readiness, not a claim that no
other installation exists anywhere on CARC. The Stanford JAR and offline
end-to-end inference have not been validated.

Full parsing therefore remains **not ready**, and report-parser calls this run
are **zero**. Source checkout plus aggregation tests are not successful NegBio
inference. The serialization contract also does not yet implement the complete
error-capturing inference worker. Official detection can swallow exceptions;
that boundary must be tested before native positives become usable evidence.

Next: obtain approval for an isolated legacy-compatible environment and frozen
parser/resources; show a complete CPU Slurm script and resource request before
submission. Then perform offline initialization, freeze a separate prospective
authored48/authored64 protocol, and run predictions/replays before reading
references. Keep ontology mismatch and known-development limitations explicit.
No human clinical gold is fabricated, no rules are tuned to these outcomes, and
no qualification is transferred automatically to clinical candidate scoring.

Official sources:
[CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler),
[NegBio](https://github.com/ncbi-nlp/NegBio).
