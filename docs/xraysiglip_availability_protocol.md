# XraySigLIP readout availability in the full candidate bank

CPU-only append-only metadata handoff, not new scoring or clinical selection.
Use the existing CPU Slurm allocation. No model, GPU, API, download, new Slurm
submission, EHR/report/image body access or weight access.

Authenticate fixed manifests for the validated 960-row verification frontier,
the earlier guarded image-availability facts, the sealed SigLIP input plan and
completed job 12654030. Read only named bounded metadata artifacts; do not
recursively follow manifest sources. Preserve all candidate cells/order, all
13,440 finding records, all 21 supplemental requests and their execution and
clinical histories. Copy the 2,072 original request records byte-for-byte.

Append `siglip_` columns only. Join original image candidate ID AND image hash
AND fixed case/EHR/facts hashes through exact existing consumer identities and
guard receipts. Identical image hashes do not enlarge checked scope. Controls
never become candidates. Authenticate all ten input receipts, six attempts,
six completed forwards and four blocks against the immutable attempt journal.
Failed or blocked readouts remain null/unavailable and still cost attempts.

Reuse all three fixed template pairs for all eight heads, without choosing a
template or fitting a threshold. Recompute raw cosine margins from sealed
predictions and verify the 21 recorded post-hoc comparisons. Preserve all
14 inventory findings: unsupported heads stay outside scope; unchecked image
slots stay not_checked. Missing/unknown/uncertain never become negative or zero.
Raw means/ranges and text preferences are not clinical labels or probabilities.

Only the six checked image IDs' original report consumers receive image-only
readout availability. No image/report, EHR/report or EHR/image clinical edge
score is derived. Report factuality is not validated by reusing an image-only
readout. Count unique image/finding readouts separately from report consumer
links and logical evidence requests. Shares of the same encoder/image do not
constitute independent votes. SigLIP shares CheXagent-2's visual encoder.

All clinical scores remain null; clinical resolution, primary-metric eligibility,
fault attribution, selection and regeneration flags stay false. Preserve the
frontier priority and action: alternate automatic evidence does not satisfy
its request for independent clinical evidence. No budget, acceptance threshold,
new winner or request execution authorization is invented.

Worker: `TriCompose-v1.2/tools/attach_xraysiglip_availability.py`. Tests use only
invented metadata. Fresh atomic protected output under `xraysiglip_availability/`
with project directories 2770/files 0660; refuse overwrite, clean only the new
temporary output on failure. Emit candidate CSV, finding JSONL, unique checked
image/finding JSONL, supplemented request JSONL, original-history JSONL, summary,
bilingual note and source/artifact hash manifest. Recheck every consumed hash,
all original cells/order and permissions before atomic commit. Tests cover
exact lineage, no hash-only spread, numerical/template contracts, unknown-safe
comparisons, failed/blocked/null states, accounting, no authorization, lossless
history, deterministic output and atomic failure handling. Never modify any
already sealed worker, test, prompt, source run or protocol.
