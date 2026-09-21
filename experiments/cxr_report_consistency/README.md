# CXR-report consistency protocol

This protocol keeps two questions separate.

## 1. Reference-report fidelity

When the input is a real CXR and its real source report is available inside an
approved protected Slurm job, report overlap can be measured with BLEU-1/2/3,
ROUGE-L, and METEOR. These metrics measure similarity to one reference report;
they are not image-report consistency metrics. They must not be used as the
primary score when a report was generated from a synthetic CXR that is allowed
to differ from the real target image.

## 2. Reference-free CXR-report consistency

For every synthetic CXR and candidate report, retain independent evidence:

1. Frozen BioViL-T image-text similarity.
2. Frozen image disease probabilities compared with report disease states.
   Relations are support, contradiction, or unknown. Unknown and unmentioned
   findings are never converted to negatives.
3. Frozen Qwen2.5-VL structured judgments for findings, negation, laterality,
   severity, anatomy, and support devices.
4. Deterministic report-quality gates for empty, truncated, repetitive, or
   malformed text.
5. Cross-report disagreement between UniDisc, LLaVA-Rad, MAIRA-2, and
   CXRMate-single as uncertainty, not as ground truth. CheXagent-2 SRRG
   findings are evaluated through the same frozen Qwen2.5-VL contract and can
   be added to the candidate pool after their scores are calibrated.

Strong contradictions include an explicit report finding that conflicts with
the image, left/right conflicts, incompatible device placement, and opposite
negation of a clearly visible major finding. A single strong contradiction is
a deterministic reject gate. Missing EHR facts or findings that cannot be
resolved from the CXR remain unknown.

## Calibration

Qwen scores and embedding similarities remain `uncalibrated` until evaluated
on protected real matched pairs and hard negatives. Hard negatives should
include report swaps with similar disease burden and controlled synthetic edits
of negation, laterality, severity, and devices. Report AUROC/AUPRC, calibration
error, and subgroup performance. Do not choose score weights from the same
cases used for final evaluation.

Until calibration is complete, the selector may use strong-contradiction and
text-quality gates, but must return `verify_more` rather than treating a raw
Qwen score as a clinically meaningful probability.

## Initial deterministic selection

For one CXR, generate all four report candidates. Reject candidates with a
strong contradiction or a failed text-quality gate. Among the remaining
candidates, retain the full score vector rather than collapsing it immediately.
After calibration, static reranking can compare calibrated CXR-report evidence,
EHR-report contradiction evidence, cross-report uncertainty, and model cost.
Dynamic routing is only justified if this static best-of-N baseline is
materially better than every fixed report generator and can be approximated
with fewer model calls.
