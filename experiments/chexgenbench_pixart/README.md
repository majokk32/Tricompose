# CheXGenBench PixArt-Sigma frozen baseline

This adapter uses the unmodified official `CheXGenBench/` checkout at workspace
root and the public checkpoint `raman07/CheXGenBench-Models-Pixart-Sigma`, pinned
to revision `ab837e1d3d7b5aeccb4dcbc4d3abcf1c2ed24727`.

It preserves the official FP16, 20-step, guidance-scale-4.5 inference path. The
checkpoint's transformer has sample size 64 and produces 512x512 images through
the official pipeline defaults.

Scientific signature:

`structured EHR -> deterministic radiology-style text -> frozen PixArt -> CXR`

This is not direct structured-EHR-conditioned CXR generation. It never loads a
matched real CXR or source report, and all prompts and outputs stay below
`artifacts/protected/chexgenbench_pixart/runs/`.

The model card does not currently declare an explicit checkpoint license, so
the weights should be treated as research-only pending clarification.

## Fixed 50-case pilot

The non-overwriting run `cgb_pixart50_fixed_20260804_001` uses the same first
50 opaque cases as the Sana pilot. PixArt, its text encoder, and VAE remain in
evaluation mode with gradients disabled. The protected `frozen.json` records
the exact public model revision, all weight hashes, prompt versions, per-case
seeds, generation-record hashes, and image hashes. The run must not be
regenerated in place or selectively filtered after review.
