# Fixed-image report control / 固定图像报告选择对照

This is a new exploratory control on the already inspected 80-EHR development
cohort, NOT an independent test or retrospective preregistration. The earlier
12 report-only changes were selected by the joint policy; they do not establish
report-only selection's performance on all 80 cases.

Freeze `TriCompose-v1.2/configs/fixed_image_pool80_v1.json` before running this
control. All 80 EHRs stay fixed. Every case uses its existing Sana seed 0 image,
the original fixed-path image, chosen without report/endpoint scores. No new
generation, fact extraction, rule fitting, training or source mutation occurs.

## Methods and cost

Compare the original fixed MAIRA-2 path, original joint targeted/static paths,
and two new SAME-IMAGE report policies at the original caps 4/8/12/20/30:

- `report_only_static`: visit the four registered report models in their frozen
  order and choose the best observed eligible report by the unchanged source key.
- `report_only_targeted`: retain the original heuristic report-switch and proxy
  stop rules. If the heuristic requests another CXR, stop with an explicit
  `fixed_image_cxr_change_blocked` status; do not pretend to repair the image.
  If report inventory is exhausted, return the best eligible observed report
  with unresolved status, not clinical acceptance.

Both policies charge image generation + XRV once (two simulated calls), then
report generation + CheXbert per requested report (two calls). Four reports on
one image cost ten calls. Equal CAPS do not imply equal actual expenditure or
equal search spaces. Preserve all caps, actual simulated calls, eligibility,
early stops, exhaustion and unavailable candidates. Existing bank generation
cost is sunk; no actual prospective GPU saving is established.

## Readouts and availability

Keep the historical fourteen-head proxy profile explicit. Report all three
edges, explicit positive/negative support separately, direct-EHR/no-direct-EHR
subgroups, structure and coverage. Unknown/uncertain are not negative. A fixed
image holds classifier reference labels invariant across reports, but they
remain unverified operational proxies, not clinical truth.

BioViL-T stays excluded from actions, candidate ranking and image choice.
Reuse only hash-bound existing selected-union endpoint scores. A new winner
outside that scored union remains NA, with its lineage recorded for optional
later scoring; do not choose a worse but scored candidate, impute zero, drop
the case or reuse another image/report score. Paired means use the SAME
available EHR cases and display missing counts. Proxy readouts on those pairs
are distinguished from full-cohort readouts. No significance/causal/clinical
repair claim follows from this development comparison.

## Execution and output

The CPU cache replay requires an existing Slurm allocation before reading
protected caches. It opens no patient inputs, report bodies or image pixels.
Write a new protected atomic run with the frozen control, all trial/action
records, per-method/subgroup tables, paired contrasts, pending endpoint pairs,
a bilingual report and source/result hashes. Never overwrite old selections.
No new Slurm submission or GPU execution is part of this control. Any missing
endpoint scoring needs a separately shown complete script and approval.
