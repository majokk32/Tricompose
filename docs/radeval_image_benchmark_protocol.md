# RadEvalExpert image-grounded scoring: fixed small MIMIC subset

Status: **prepared, not executed**. Pixel/model execution requires explicit
approval of the complete existing-Slurm entry script and input scope below.
This protocol does not authorize a new sbatch or a source-image download.

## Why this follows the report-only test

The previous expert-reference RadGraph benchmark had weak pooled correlation
and modest post-hoc within-anchor choice gains. Its inputs include a reference
report, unavailable during fully synthetic inference. This next test uses the
already installed **frozen BioViL-T** to score an actual CXR and a candidate
report, without feeding the reference report or expert counts into the model.
It is a fixed verifier diagnostic, not a new trained scorer or agent policy.

## Metadata findings and source boundary

All 203 image/source keys and 624 report pairs remain in the inventory.
Direct mapping into seven standard roots found no exact files. An additional
bounded join against the existing `three_modalities/v2_labs_vitals/manifest.csv`
found **43 exact source images**, preserving patient/study/DICOM suffixes.
All 43 paths passed metadata stat inside the CARC project. No basename-only
retrieval, different view, changed raster extension or clinical case choice was
used. Checks distinguish filesystem permissions from missing files.

These images map to **44 source/section/reference anchors**, **132 candidate
reports**, **130 exact distinct candidate texts**, and **34 recognizable MIMIC
patient groups**. The cohort is determined by licensed local availability
before image scores, not by positive diagnoses, expert error counts, old
winners, radiology difficulty or visual quality. Do not advertise it as an
untouched final test or a random sample of all MIMIC.

57 other recognized MIMIC image keys are not linked; 31 CheXpert keys lack a
separately confirmed local access path; 72 source layouts are unresolved.
Unavailable pairs remain explicit nulls. No foreign-dataset or substitute CXR
is loaded. Raw paths stay in a separate protected operational resolver, not
public manifests/logs. No images were opened, decoded, hashed or copied during
the metadata preparation.

## Frozen inputs and execution

- Only the 43 linked original local MIMIC JPEGs and the author-released
  candidate text are model inputs. No source EHR, reference report, previous
  CXR or real target for a generation smoke is supplied to the verifier.
- Expert significant/insignificant error counts enter offline evaluation only
  **after** scores are obtained; they are not prompt material or model inputs.
- Existing model weights and hi-ml-multimodal 0.2.2 vendor code are read-only,
  pinned by SHA256. Official global embeddings, resize 512, center crop 448,
  L2 normalization, 128 dimensions; no preprocessing/windowing optimization.
- CPU only in actual allocation 12714150: existing allocation 4 CPUs/32 GiB,
  worker uses two threads, no GPU. Ten-minute process bound. Do not run this
  script on a login node or fabricate a Slurm environment/cgroup.
- Models eval/inference mode, requires_grad false, offline-only assets and
  blocked network sockets during scoring. No training, threshold/weight fit,
  external API, dataset/model download or new sbatch.
- Keep full report input. Official text engine rejects inputs beyond model
  position capacity; failed/empty/oversize reports remain unavailable, not
  silently truncated. No disease facts or radiology labels are added.
- Each image and distinct text is encoded once. Preserve every failure with
  safe type codes and bounded status logs; no patient identifiers or clinical
  text in public output. Native embeddings are protected internal artifacts.
- Check source path stat before computation and image byte hashes inside the
  approved run; record hashes only. No real image rendering/manual inspection.

## Predeclared comparisons and interpretation

Primary: Spearman of BioViL-T raw cosine against negative expert significant
error count. Report Kendall tau-b, all attempted/available denominators and
1,000 patient-cluster bootstrap resamples, seed 0. Cosine is not a probability.
Secondary: all-error correlation and three-candidate, fixed-anchor selection
versus analytical uniform random choice and expert minimum-count oracle.
Uniformly average maximum-score ties; do not pick an arbitrary favorable slot.

Compare all three existing RadGraph reference-based scores on the **identical
available image/text cohort**, without re-running those models or replacing
their full-data result. Missing BioViL inputs invalidate that paired comparator
cell; incomplete anchors do not drop only one bad candidate. No combined score
or metric-specific weight fitting. All four results are reported.

The expert counts compare candidate reports against a reference text. They are
not newly obtained image-radiologist adjudication. This test measures whether
image-grounded scores track that released error reference; it does not make
all reference claims visible on a single CXR. Scope/temporal claims and model
training overlap remain limitations. Even a positive result does not prove
synthetic-domain transfer, EHR consistency, fault localization or targeted
repair. Existing scores, policies, frozen EHRs and winners remain unchanged.

## Execution entry point

Prepare is metadata-only; it seals the cohort, model/code/source hashes, exact
script and all unavailable denominators. It consumes zero image bytes and
executes zero models. The separately approved run command is:

```bash
bash TriCompose-v1.2/slurm/60_radeval_biovil_existing_cpu.sh
```

The complete script must be shown to the user before execution. If allocation
12714150 is no longer active, refuse this entry point and request approval for
a new complete CPU batch script. No process is authorized by changing an
environment variable, this protocol file, or a metadata readiness receipt.
