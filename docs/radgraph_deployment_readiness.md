# Official RadGraph integration readiness

Initial status when proposed: source/API inspection and pure contract tests
only. The subsequent user-approved deployment and CPU interface smoke are
recorded in `radgraph_deployment_result_12714150_001.md`. Clinical qualification
is still pending; no new Slurm submission was made.

## Purpose and scope

Native XL graph extraction preserves entity spans, positive/negative/uncertain
observation labels, anatomy and relations. This adds information beyond a
14-label report vector. It does **not** establish current-patient versus prior
study scope, image truth, EHR truth, or a qualified clinical repair decision.
All generators and extractors remain frozen; no training or fitted router.

The official reference metric takes hypothesis **and reference reports**. Its
three `reward_level="all"` components are kept separate:

| Local column | Official meaning |
|---|---|
| `radgraph_entity_f1` | Exact entity token/label match |
| `radgraph_relation_presence_f1` | Entity match including whether a relation exists |
| `radgraph_full_relation_f1` | Entity match including relation type and destination token |

The contract does not reimplement or optimize the upstream reward. It normalizes
upstream results and checks inventory/means. Empty input pairs are marked
ineligible/null, not treated as a clinically incorrect zero-score report.
Valid model-produced zeros are retained. Missing graphs fail closed. No absent
entity is converted into a negative finding. Returned rows contain opaque IDs,
hashes/counts and scores, never report/entity text.

Initial contract: `TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract.py`.
The active V2 contract additionally preserves the full 11-label vocabulary
found in the pinned archive:
`TriCompose-v1.2/src/tricompose_v12/radgraph_reference_contract_v2.py`.
Invented fixtures: `TriCompose-v1.2/tests/test_radgraph_reference_contract*.py`.
These are interface tests, not model performance or expert clinical validation.

## Audited official assets

Metadata was checked on 2026-10-06 without downloading model weights.

- Official source: https://github.com/Stanford-AIMI/radgraph
- Code revision: `87f11a1ff4d2046a838be5f0243857d93780ddec`
- Package version at that revision: `0.1.18`
- Official weight repository: https://huggingface.co/StanfordAIMI/RRG_scorers
- Weight revision: `6646433b3ad83a10f6e141db76d0ece44312b236`
- Selected published variant: `radgraph-xl`, not the newer modern variant.
- File: `radgraph-xl.tar.gz`, 416,179,571 bytes.
- SHA256: `fadb5a3454e8996714b609e3105a07e447fa61aa88fffe966b650959475117b6`
- Alternate modern archive: 579,388,064 bytes; not selected or downloaded.

The selected archive size is **not** total deployment storage. Source inspection
confirmed that the selected code constructs the encoder from config and loads
its parameters from the XL state dict. No additional encoder weights are
needed. Its configured tokenizer/config still must be separately prepared,
recorded and pinned before inference; archive availability alone is insufficient.
The public model download approval request caps combined model downloads at
2 GB; exceeding it requires a revised request. Environment/expanded-cache
storage is separate and must be measured before installation.

After approval, use a root-level `RadGraph/` checkout, a new isolated runtime
environment, runtime model assets and workspace-local caches. Do not mutate
existing environments/checkpoints. Keep official checkout/checkpoints out of
project Git. Acquire pinned assets explicitly; do not let inference download
unversioned files. Inspect and safely extract the tar archive only after hash
verification. Set HF/Transformers offline mode during all model calls.

Run the first small interface test only inside an existing approved CPU Slurm
allocation, if still active and sufficient. No new GPU job is currently
proposed. Any new `sbatch` needs its complete script/resources shown and fresh
explicit approval. Worker stdout must be sanitized; model-returned text/graphs
and any restricted benchmark data belong only under `artifacts/protected/`.

## Independent clinical benchmark gate remains separate

The previous official MIMIC report-label pilot already ran; repeating it is not
a new independent test. ReXVal requires the user's own authorized PhysioNet
access, and its expected files were not found in the inspected workspace roots.
Do not obtain unofficial mirrors or read authentication files. RadCliQ's
ReXVal fitting overlap must not be ignored.

The existing public RadEvalX cached-score benchmark has unresolved blank error
annotations. Its hypothetical blank-to-zero sensitivity is not verified gold
and cannot qualify a newly deployed local metric. Expert graph annotations or
an independent error-count benchmark still require a documented accessible
source, annotation convention and frozen cohort. No clinical thresholds,
score weights, winners or regeneration decisions change at deployment.

For the fully synthetic cohort there is no real reference report. Native graph
comparison between generated reports is cross-path agreement, **not**
RadGraph clinical factuality or independent CXR–Report correctness.
