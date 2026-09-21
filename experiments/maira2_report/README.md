# Frozen MAIRA-2 CXR-to-report candidate

This adapter uses the complete, current local `microsoft/maira-2` revision
`795a2b1cd4a310624b4e3d14b5a23e41fd273deb` without changing its parameters.

The deployed pathway is deliberately restricted to:

```text
one protected current synthetic frontal CXR -> non-grounded findings
```

It does not provide an indication, lateral view, prior CXR, prior report,
structured EHR, or real target report. This makes it a genuine CXR-to-report
candidate and avoids leakage between TriCompose paths. Decoding follows the
official non-grounded recipe: greedy generation, one beam, and at most 300 new
tokens. Every learned parameter is placed in evaluation mode with gradients
disabled. Generated text stays below `artifacts/protected/maira2_report/`.

