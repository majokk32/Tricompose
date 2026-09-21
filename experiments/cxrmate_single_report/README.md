# Frozen CXRMate single-image CXR-to-report candidate

This adapter uses the complete current `aehrc/cxrmate-single-tf` revision
`84dfcba8125c9b9296bfc1ee317749788b6b59b4`. It is the paper's teacher-forced,
non-longitudinal, single-image checkpoint, not CXRMate-ED and not the
longitudinal reinforcement-learning model.

The deployed pathway is:

```text
one protected current synthetic CXR -> findings + impression
```

The model receives no structured EHR, prior image, prior report, or real target
report. Decoding uses deterministic four-beam search with a maximum sequence
length of 256, following the existing local official-model wrapper. Every
parameter is set to evaluation mode with gradients disabled. Generated text
stays below `artifacts/protected/cxrmate_single_report/`.

