# Text-conditioning interface diagnostic: prepared, NOT submitted

2026-10-09, actual existing CPU Slurm12851223. This is the next small generation
diagnostic after the joined report/image action readout. It does not update a
scorer, change existing EHRs/prompts, generate new reports, select a winner or
establish clinical repair. Protocol and authored template scope:
`docs/text_conditioning_probe_protocol.md`.

Prepared entry: `TriCompose-v1.2/agent/run_text_conditioning_probe.py`.
Program SHA256:
`d30c56d8dca8840573b20b4f70f7e7b66521a133fe0b6955c24fff97133db20a`.

Prepared protected plan:
`artifacts/protected/tricompose_v1_2/text_conditioning_probe_plans/text_probe_12851223_001/`.
Manifest SHA256:
`7a6b6182a32b07295cce9ec117cd6d18c38f26904ab2f3bf6d6eb63b799ea934`.

No input EHR, patient/source/target narrative or source image is consumed.
Previous sealed plan metadata supplies only frozen model/checkpoint receipts.
Two authored present/absent MAIN prompts × two seeds × RoentGen-v2/Sana:
eight image slots and at most eight unchanged XRV observations. The absent
arm is NOT a Diffusers negative-prompt argument. Native settings stay at
RoentGen75steps/CFG3/512px and Sana20steps/CFG4.5/1024px, float16.
Models load separately; explicit eval/gradient freeze flags are checked.
All actual model-load/forward attempts are reserved durably; failure blocks
the remaining worker slots, with zero retry and no hidden refunded cost.

CPU checks performed no factory/model load, large weight-byte hash, API,
download or sbatch. Sixteen authored-fixture tests and all **4,004 V1.2 tests**
passed (full suite29.300s). The plan's exact slots/prompts, prior checkpoint
receipt, source pins, executable environments and model-asset size/mtime
checks passed during preparation. This is not GPU runtime/kernel validation.

`noderes -f -g` and `sinfo` showed idle V100 nodes at inspection. CARC's
[official resource overview](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery)
lists Discovery V100 memory as32GB. Actual memory is rechecked in the approved
allocation; free capacity is a snapshot, not a queue promise. No A40/A100 is
requested. Proposed resources: one V100 in `gpu`, two CPUs,32GiB host RAM,
15-minute ceiling; this is not a predicted runtime. Script:
`TriCompose-v1.2/agent/slurm/20_text_conditioning_probe8_v100.sbatch`.
Reviewed batch-script SHA256:
`a437916942d3da68118447a63f5e106b9b8825a68f41abc3e495070319d12fc2`.
The script/program/plan-manifest hashes were rechecked after preparation;
plan, output-parent, job-runtime and Slurm-log directories are2770 with
CARC NFS project-boundary GID65534; the plan manifest is0660. Batch shell
syntax passed. No GPU job was submitted during these checks.

After complete script/resource display, obtain NEW explicit approval before:

```bash
sbatch TriCompose-v1.2/agent/slurm/20_text_conditioning_probe8_v100.sbatch
```

Expected NEW protected output: `text_conditioning_probe_runs/text_probe_<job>/`.
Files include eight generation slots (including failures), four paired rows,
`paired_readout.csv`, XRV numeric predictions, tokenizer traces, images and
charged journals. Patient-derived data and credentials are never copied into
Git/public logs. Paired score deltas, token-ID differences and image byte
differences are engineering readouts only, not clinical accuracy against a
prompt or automatic repair success. Original selected triples remain intact.
